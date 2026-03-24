"""
Compare multiple benchmark JSON files produced by transcender_gpu_reproduction.py.

This helper is intentionally small and conservative. It reuses the existing
single-file summary logic and surfaces the minimum cross-model facts needed for
external-validity work:

- model id / family
- raw_exit_avg_exact_match by tested layer
- composed_avg_exact_match by tested layer
- avg_top1_agreement_rate by tested layer
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
