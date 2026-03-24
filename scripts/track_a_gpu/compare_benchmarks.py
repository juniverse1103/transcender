"""
Compare multiple benchmark JSON files produced by transcender_gpu_reproduction.py.

This helper is intentionally small and conservative. It reuses the existing
single-file summary logic and surfaces the minimum cross-model facts needed for
external-validity work:

- model id / family
- raw_exit_avg_exact_match by tested layer
- composed_avg_exact_match by tested layer
- avg_top1_agreement_rate by tested layer
- oracle aggregate acceptance summaries when present
- first-divergence aggregate summaries when present
- earliest-correct headroom over penultimate top1_agree when present
- whether penultimate raw exit is consistently at least as strong as the
  previous layer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from summarize_benchmark import build_summary, load_benchmark


def compact_record(path: Path) -> Dict[str, Any]:
    data = load_benchmark(path)
    summary = build_summary(data, path)
    comparison = summary["penultimate_vs_previous_raw_exit"]
    return {
        "benchmark_json": str(path),
        "model": summary.get("model"),
        "model_family": summary.get("model_family"),
        "layers": summary["layers"],
        "oracle_aggregates": summary.get("oracle_aggregates", {}),
        "first_divergence_aggregates": summary.get("first_divergence_aggregates", {}),
        "oracle_headroom_summary": summary.get("oracle_headroom_summary", {}),
        "penultimate_vs_previous_raw_exit": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple GPU benchmark JSON files")
    parser.add_argument("benchmark_jsons", nargs="+", help="Benchmark JSON files to compare")
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the comparison as JSON instead of plain text",
    )
    args = parser.parse_args()

    records: List[Dict[str, Any]] = [compact_record(Path(path)) for path in args.benchmark_jsons]

    if args.as_json:
        print(json.dumps(records, indent=2))
        return

    for record in records:
        print(f"Benchmark: {record['benchmark_json']}")
        print(f"Model: {record.get('model')}")
        print(f"Family: {record.get('model_family')}")
        for label, layer in record["layers"].items():
            print(
                f"  {label}: raw_exit_avg_exact_match={layer['raw_exit_avg_exact_match']:.3f}  "
                f"composed_avg_exact_match={layer['composed_avg_exact_match']:.3f}  "
                f"avg_top1_agreement_rate={layer['avg_top1_agreement_rate']:.3f}"
            )
        if record["oracle_aggregates"]:
            print("  Oracle aggregates:")
            for mode, mode_summary in record["oracle_aggregates"].items():
                if mode_summary.get("kind") == "single_layer":
                    for layer_label, layer_summary in mode_summary["per_layer"].items():
                        print(
                            f"    {mode} {layer_label}: "
                            f"avg_acceptance_rate={layer_summary['avg_acceptance_rate']:.3f}  "
                            f"avg_oracle_EM={layer_summary['avg_oracle_composed_exact_match']:.3f}"
                        )
                    continue
                if mode_summary.get("available", True) is False:
                    print(f"    {mode}: unavailable")
                    continue
                print(
                    f"    {mode}: "
                    f"avg_acceptance_rate={mode_summary['avg_acceptance_rate']:.3f}  "
                    f"avg_oracle_EM={mode_summary['avg_oracle_composed_exact_match']:.3f}"
                )

        first_divergence = record.get("first_divergence_aggregates", {})
        if first_divergence.get("per_layer"):
            print("  First divergence aggregates:")
            for layer_label, row in first_divergence["per_layer"].items():
                print(
                    f"    {layer_label}: "
                    f"mean={row['raw_first_divergence_mean']:.3f}  "
                    f"median={row['raw_first_divergence_median']:.3f}"
                )
            pair_div = first_divergence.get("penultimate_vs_previous", {})
            if pair_div.get("available"):
                print(
                    "    penultimate_vs_previous: "
                    f"{pair_div['penultimate_label']} later on "
                    f"{pair_div['penultimate_diverges_later_count']}/{pair_div['total_prompts']} prompts"
                )

        headroom = record.get("oracle_headroom_summary", {})
        if headroom.get("available"):
            print(
                "  Oracle headroom: "
                f"earliest_correct_vs_{headroom['penultimate_label']}_top1_agree "
                f"avg_gap={headroom['avg_acceptance_gap']:.3f}  "
                f"micro_gap={headroom['micro_acceptance_gap']:.3f}"
            )

        comparison = record["penultimate_vs_previous_raw_exit"]
        if comparison["available"]:
            print(
                "  penultimate_vs_previous_raw_exit: "
                f"{comparison['penultimate_label']} >= {comparison['previous_label']} on "
                f"{comparison['penultimate_ge_previous_count']}/{comparison['total_prompts']} prompts; "
                f"strictly better on {comparison['penultimate_gt_previous_count']}/"
                f"{comparison['total_prompts']}; consistent={comparison['consistent']}"
            )
        else:
            print("  penultimate_vs_previous_raw_exit: unavailable")
        print()


if __name__ == "__main__":
    main()
