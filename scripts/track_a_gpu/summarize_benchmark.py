"""
Summarize a benchmark JSON produced by transcender_gpu_reproduction.py.

This helper reports:

- raw_exit_avg_exact_match by layer
- composed_avg_exact_match by layer
- avg_top1_agreement_rate by layer
- oracle acceptance / fallback summaries when oracle aggregates are present
- whether the penultimate layer is consistently stronger than the previous
  layer on raw-exit exact match
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_benchmark(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if "aggregates" not in data or "prompt_results" not in data:
        raise ValueError(
            f"{path} does not look like a benchmark JSON from transcender_gpu_reproduction.py"
        )
    return data


def layer_labels(data: Dict[str, Any]) -> List[str]:
    labels = [key for key in data["aggregates"].keys() if key.startswith("L")]
    return sorted(labels, key=lambda label: int(label[1:]))


def penultimate_pair(labels: List[str]) -> tuple[str, str] | None:
    if len(labels) < 2:
        return None
    return labels[-2], labels[-1]


def per_prompt_penultimate_vs_previous(data: Dict[str, Any], previous_label: str, penultimate_label: str) -> Dict[str, Any]:
    scored = [row for row in data["prompt_results"] if not row.get("is_warmup")]
    if not scored:
        scored = data["prompt_results"]

    if not scored:
        return {
            "available": False,
            "total_prompts": 0,
            "previous_label": previous_label,
            "penultimate_label": penultimate_label,
            "penultimate_ge_previous_count": 0,
            "penultimate_gt_previous_count": 0,
            "consistent": False,
        }

    if (
        previous_label not in scored[0]["layer_results"]
        or penultimate_label not in scored[0]["layer_results"]
    ):
        return {
            "available": False,
            "total_prompts": len(scored),
            "previous_label": previous_label,
            "penultimate_label": penultimate_label,
            "penultimate_ge_previous_count": 0,
            "penultimate_gt_previous_count": 0,
            "consistent": False,
        }

    ge_count = 0
    gt_count = 0
    for row in scored:
        previous = row["layer_results"][previous_label]["raw_exit_exact_match_rate"]
        penultimate = row["layer_results"][penultimate_label]["raw_exit_exact_match_rate"]
        if penultimate >= previous:
            ge_count += 1
        if penultimate > previous:
            gt_count += 1

    return {
        "available": True,
        "total_prompts": len(scored),
        "previous_label": previous_label,
        "penultimate_label": penultimate_label,
        "penultimate_ge_previous_count": ge_count,
        "penultimate_gt_previous_count": gt_count,
        "consistent": ge_count == len(scored),
    }


def build_summary(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    labels = layer_labels(data)
    layers = {}
    for label in labels:
        agg = data["aggregates"][label]
        layers[label] = {
            "raw_exit_avg_exact_match": agg["raw_exit_avg_exact_match"],
            "composed_avg_exact_match": agg["composed_avg_exact_match"],
            "avg_top1_agreement_rate": agg["avg_top1_agreement_rate"],
        }

    pair = penultimate_pair(labels)
    comparison = (
        per_prompt_penultimate_vs_previous(data, pair[0], pair[1])
        if pair is not None
        else {
            "available": False,
            "total_prompts": 0,
            "penultimate_ge_previous_count": 0,
            "penultimate_gt_previous_count": 0,
            "consistent": False,
        }
    )
    oracle_summary: Dict[str, Any] = {}
    for mode, mode_summary in data.get("oracle_aggregates", {}).items():
        if mode_summary.get("kind") == "single_layer":
            oracle_summary[mode] = {
                "kind": "single_layer",
                "per_layer": mode_summary.get("per_layer", {}),
            }
            if "margin_threshold" in mode_summary:
                oracle_summary[mode]["margin_threshold"] = mode_summary["margin_threshold"]
            if "entropy_threshold" in mode_summary:
                oracle_summary[mode]["entropy_threshold"] = mode_summary["entropy_threshold"]
            continue
        oracle_summary[mode] = mode_summary
    return {
        "benchmark_json": str(path),
        "model": data.get("model"),
        "model_family": data.get("model_family"),
        "aggregates_scope": data.get("aggregates_scope"),
        "layers": layers,
        "oracle_aggregates": oracle_summary,
        "first_divergence_aggregates": data.get("first_divergence_aggregates", {}),
        "oracle_headroom_summary": data.get("oracle_headroom_summary", {}),
        "penultimate_vs_previous_raw_exit": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a GPU benchmark JSON")
    parser.add_argument("benchmark_json", help="Path to the benchmark JSON")
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the summary as JSON instead of plain text",
    )
    args = parser.parse_args()

    path = Path(args.benchmark_json)
    data = load_benchmark(path)
    summary = build_summary(data, path)

    if args.as_json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Benchmark: {path}")
    if summary.get("model"):
        print(f"Model: {summary['model']}")
    if summary.get("model_family"):
        print(f"Family: {summary['model_family']}")
    print(f"Scope: {summary['aggregates_scope']}")
    for label, layer in summary["layers"].items():
        print(
            f"{label}: "
            f"raw_exit_avg_exact_match={layer['raw_exit_avg_exact_match']:.3f}  "
            f"composed_avg_exact_match={layer['composed_avg_exact_match']:.3f}  "
            f"avg_top1_agreement_rate={layer['avg_top1_agreement_rate']:.3f}"
        )
    if summary["oracle_aggregates"]:
        print("Oracle aggregates:")
        for mode, mode_summary in summary["oracle_aggregates"].items():
            if mode_summary.get("kind") == "single_layer":
                threshold_bits: List[str] = []
                if "margin_threshold" in mode_summary:
                    threshold_bits.append(f"margin_threshold={mode_summary['margin_threshold']}")
                if "entropy_threshold" in mode_summary:
                    threshold_bits.append(f"entropy_threshold={mode_summary['entropy_threshold']}")
                header_suffix = f" ({', '.join(threshold_bits)})" if threshold_bits else ""
                print(f"  {mode}{header_suffix}")
                for layer_label, layer_summary in mode_summary["per_layer"].items():
                    print(
                        f"    {layer_label}: "
                        f"avg_acceptance_rate={layer_summary['avg_acceptance_rate']:.3f}  "
                        f"micro_acceptance_rate={layer_summary['micro_acceptance_rate']:.3f}  "
                        f"avg_fallback_rate={layer_summary['avg_fallback_rate']:.3f}  "
                        f"avg_oracle_EM={layer_summary['avg_oracle_composed_exact_match']:.3f}"
                    )
                continue
            if mode_summary.get("available", True) is False:
                print(f"  {mode}: unavailable ({mode_summary.get('reason')})")
                continue
            extra = ""
            if "pair" in mode_summary:
                extra = f" pair={mode_summary['pair']}"
            if "selected_layer_counts" in mode_summary:
                extra = f" selected_layer_counts={mode_summary['selected_layer_counts']}"
            print(
                f"  {mode}:{extra} "
                f"avg_acceptance_rate={mode_summary['avg_acceptance_rate']:.3f}  "
                f"micro_acceptance_rate={mode_summary['micro_acceptance_rate']:.3f}  "
                f"avg_fallback_rate={mode_summary['avg_fallback_rate']:.3f}  "
                f"avg_oracle_EM={mode_summary['avg_oracle_composed_exact_match']:.3f}"
            )
    first_divergence = summary.get("first_divergence_aggregates", {})
    if first_divergence.get("per_layer"):
        print("First divergence aggregates:")
        for layer_label, row in first_divergence["per_layer"].items():
            print(
                f"  {layer_label}: "
                f"raw_first_div_mean={row['raw_first_divergence_mean']:.3f}  "
                f"raw_first_div_median={row['raw_first_divergence_median']:.3f}"
            )
        pair = first_divergence.get("penultimate_vs_previous", {})
        if pair.get("available"):
            print(
                "  penultimate_vs_previous_first_divergence: "
                f"{pair['penultimate_label']} diverged later than {pair['previous_label']} on "
                f"{pair['penultimate_diverges_later_count']}/{pair['total_prompts']} prompts; "
                f"same_first_divergence={pair['same_first_divergence_count']}/{pair['total_prompts']}"
            )
    headroom = summary.get("oracle_headroom_summary", {})
    if headroom.get("available"):
        print(
            "Oracle headroom: "
            f"earliest_correct_vs_{headroom['penultimate_label']}_top1_agree "
            f"avg_gap={headroom['avg_acceptance_gap']:.3f}  "
            f"micro_gap={headroom['micro_acceptance_gap']:.3f}"
        )

    comparison = summary["penultimate_vs_previous_raw_exit"]
    if comparison["available"]:
        print(
            f"{comparison['penultimate_label']}_vs_{comparison['previous_label']}_raw_exit: "
            f"{comparison['penultimate_ge_previous_count']}/{comparison['total_prompts']} prompts "
            f"had {comparison['penultimate_label']} >= {comparison['previous_label']}; "
            f"{comparison['penultimate_gt_previous_count']}/{comparison['total_prompts']} had "
            f"{comparison['penultimate_label']} > {comparison['previous_label']}; "
            f"consistent={comparison['consistent']}"
        )
    else:
        print("penultimate_vs_previous_raw_exit: unavailable")


if __name__ == "__main__":
    main()
