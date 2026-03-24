"""
Inspect a single-prompt debug trace from transcender_gpu_reproduction.py.

This helper answers the narrow diagnostic questions needed before trusting a
GPU frontier run:

- Do raw L45/L46 candidate tokens ever differ from full depth?
- When raw candidates disagree, does composed top1_agree just fall back to the
  full-depth token?
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a GPU debug trace JSON")
    parser.add_argument("trace_json", help="Path to the debug trace JSON")
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["L45", "L46"],
        help="Layer labels to inspect (default: L45 L46)",
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

    reports = [layer_report(trace_steps, layer_label) for layer_label in args.layers]
    verdict, reason = diagnose(reports)
    summary = {
        "trace_json": str(path),
        "prompt_id": data.get("prompt_id"),
        "steps": len(trace_steps),
        "layers": {report["layer"]: report for report in reports},
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
    print(f"Verdict: {verdict}")
    print(f"Reason: {reason}")


if __name__ == "__main__":
    main()
