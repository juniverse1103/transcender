"""
Inspect a single-prompt debug trace from transcender_gpu_reproduction.py.

This helper answers the narrow diagnostic questions needed before trusting a
GPU frontier run:

- Do raw candidate tokens from the tested exit layers ever differ from full depth?
- When raw candidates disagree, does composed top1_agree just fall back to the
  full-depth token?
- If oracle summaries are present, which final-aware oracle modes accept or
  reject at each step?
- Is the trace structurally sane, inconclusive, or suspicious?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_trace(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if "trace" not in data or not isinstance(data["trace"], list) or not data["trace"]:
        raise ValueError(f"{path} does not contain a non-empty `trace` list.")
    return data


def infer_layer_labels(data: Dict[str, Any]) -> List[str]:
    first_step = data["trace"][0]
    labels = list(first_step.get("layers", {}).keys())
    return sorted(labels, key=lambda label: int(label[1:]))


def layer_report(trace_steps: List[Dict[str, Any]], layer_label: str) -> Dict[str, Any]:
    raw_different_steps: List[int] = []
    composed_recovers_raw_disagreement = 0

    for step in trace_steps:
        layer = step["layers"][layer_label]
        raw_matches = bool(layer["raw_exit"]["matches_full_depth"])
        composed_matches = bool(layer["composed_top1_agree"]["matches_full_depth"])
        step_index = int(step["step_index"])

        if not raw_matches:
            raw_different_steps.append(step_index)
            if composed_matches:
                composed_recovers_raw_disagreement += 1

    first_divergence = (
        raw_different_steps[0] + 1 if raw_different_steps else len(trace_steps) + 1
    )
    raw_diff_count = len(raw_different_steps)
    fallback_recovers_all = (
        raw_diff_count > 0 and composed_recovers_raw_disagreement == raw_diff_count
    )

    return {
        "layer": layer_label,
        "first_raw_divergence_position": first_divergence,
        "raw_diff_step_count": raw_diff_count,
        "raw_diff_step_indices": raw_different_steps,
        "composed_matches_full_on_raw_disagreement_count": composed_recovers_raw_disagreement,
        "composed_matches_full_on_all_raw_disagreements": fallback_recovers_all,
    }


def diagnose(reports: List[Dict[str, Any]]) -> tuple[str, str]:
    total_raw_diffs = sum(r["raw_diff_step_count"] for r in reports)
    all_recover = all(
        (
            r["raw_diff_step_count"] == 0
            or r["composed_matches_full_on_all_raw_disagreements"]
        )
        for r in reports
    )

    if not all_recover:
        return (
            "suspicious",
            "Composed top1_agree does not consistently return to full depth after raw disagreement.",
        )
    if total_raw_diffs == 0:
        return (
            "inconclusive",
            "Trace structure is valid, but neither raw layer diverged from full depth on this prompt/length.",
        )
    return (
        "sane",
        "Raw intermediate candidates diverge from full depth and composed top1_agree behaves like a fallback-on-disagreement path.",
    )


def oracle_step_indices(
    trace_steps: List[Dict[str, Any]],
    mode: str,
    layer_label: str | None = None,
) -> Dict[str, List[int]]:
    accepted_steps: List[int] = []
    rejected_steps: List[int] = []

    for step in trace_steps:
        oracle_bundle = step.get("oracles", {})
        if mode not in oracle_bundle:
            continue
        row = oracle_bundle[mode]
        if layer_label is not None:
            row = row.get("per_layer", {}).get(layer_label)
        if not row or row.get("available", True) is False:
            continue
        step_index = int(step["step_index"])
        if bool(row["accepted"]):
            accepted_steps.append(step_index)
        else:
            rejected_steps.append(step_index)

    return {
        "accepted_step_indices": accepted_steps,
        "rejected_step_indices": rejected_steps,
    }


def oracle_reports(data: Dict[str, Any], trace_steps: List[Dict[str, Any]], layer_labels: List[str]) -> Dict[str, Any]:
    summaries = data.get("oracle_summaries")
    if not summaries:
        return {}

    reports: Dict[str, Any] = {}
    for mode, mode_summary in summaries.items():
        if mode_summary.get("kind") == "single_layer":
            per_layer: Dict[str, Any] = {}
            for layer_label in layer_labels:
                if layer_label not in mode_summary.get("per_layer", {}):
                    continue
                per_layer[layer_label] = {
                    **mode_summary["per_layer"][layer_label],
                    **oracle_step_indices(trace_steps, mode, layer_label),
                }
            report: Dict[str, Any] = {
                "kind": "single_layer",
                "per_layer": per_layer,
            }
            if "margin_threshold" in mode_summary:
                report["margin_threshold"] = mode_summary["margin_threshold"]
            if "entropy_threshold" in mode_summary:
                report["entropy_threshold"] = mode_summary["entropy_threshold"]
            reports[mode] = report
            continue

        if mode_summary.get("available", True) is False:
            reports[mode] = mode_summary
            continue

        reports[mode] = {
            **mode_summary,
            **oracle_step_indices(trace_steps, mode),
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a GPU debug trace JSON")
    parser.add_argument("trace_json", help="Path to the debug trace JSON")
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Layer labels to inspect. Defaults to all layer labels present in the trace.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the summary as JSON instead of plain text",
    )
    args = parser.parse_args()

    path = Path(args.trace_json)
    data = load_trace(path)
    trace_steps = data["trace"]
    layer_labels = args.layers or infer_layer_labels(data)

    reports = [layer_report(trace_steps, layer_label) for layer_label in layer_labels]
    verdict, reason = diagnose(reports)
    oracles = oracle_reports(data, trace_steps, layer_labels)
    summary = {
        "trace_json": str(path),
        "prompt_id": data.get("prompt_id"),
        "model": data.get("model"),
        "model_family": data.get("model_family"),
        "steps": len(trace_steps),
        "layers": {report["layer"]: report for report in reports},
        "oracle_reports": oracles,
        "verdict": verdict,
        "reason": reason,
    }

    if args.as_json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Trace: {path}")
    print(f"Prompt: {data.get('prompt_id')}  Steps: {len(trace_steps)}")
    for report in reports:
        print(
            f"{report['layer']}: "
            f"first_raw_div={report['first_raw_divergence_position']}  "
            f"raw_diff_steps={report['raw_diff_step_count']}  "
            f"composed_matches_full_after_raw_diff="
            f"{report['composed_matches_full_on_raw_disagreement_count']}/"
            f"{report['raw_diff_step_count']}"
        )
    if oracles:
        print("Oracle summaries:")
        for mode, report in oracles.items():
            if report.get("kind") == "single_layer":
                threshold_bits: List[str] = []
                if "margin_threshold" in report:
                    threshold_bits.append(f"margin_threshold={report['margin_threshold']}")
                if "entropy_threshold" in report:
                    threshold_bits.append(f"entropy_threshold={report['entropy_threshold']}")
                header_suffix = f" ({', '.join(threshold_bits)})" if threshold_bits else ""
                print(f"  {mode}{header_suffix}")
                for layer_label, layer_report_row in report["per_layer"].items():
                    print(
                        f"    {layer_label}: accept={layer_report_row['accepted_steps']}/{layer_report_row['total_steps']} "
                        f"({layer_report_row['acceptance_rate']:.3f})  "
                        f"fallback={layer_report_row['fallback_steps']}/{layer_report_row['total_steps']}  "
                        f"oracle_EM={layer_report_row['oracle_composed_exact_match_rate']:.3f}  "
                        f"rejected_steps={layer_report_row['rejected_step_indices']}"
                    )
                continue

            if report.get("available", True) is False:
                print(f"  {mode}: unavailable ({report.get('reason')})")
                continue

            extra = ""
            if "pair" in report:
                extra = f" pair={report['pair']}"
            if "selected_layer_counts" in report:
                extra = f" selected_layer_counts={report['selected_layer_counts']}"
            print(
                f"  {mode}:{extra} "
                f"accept={report['accepted_steps']}/{report['total_steps']} "
                f"({report['acceptance_rate']:.3f})  "
                f"fallback={report['fallback_steps']}/{report['total_steps']}  "
                f"oracle_EM={report['oracle_composed_exact_match_rate']:.3f}  "
                f"rejected_steps={report['rejected_step_indices']}"
            )
    print(f"Verdict: {verdict}")
    print(f"Reason: {reason}")


if __name__ == "__main__":
    main()
