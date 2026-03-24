"""
Summarize a benchmark JSON produced by transcender_gpu_reproduction.py.

This helper reports:

- raw_exit_avg_exact_match by layer
- composed_avg_exact_match by layer
- avg_top1_agreement_rate by layer
- whether L46 is consistently stronger than L45 on raw-exit exact match
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


def per_prompt_l46_vs_l45(data: Dict[str, Any]) -> Dict[str, Any]:
    scored = [row for row in data["prompt_results"] if not row.get("is_warmup")]
    if not scored:
        scored = data["prompt_results"]

    if not scored:
        return {
            "available": False,
            "total_prompts": 0,
            "l46_ge_l45_count": 0,
            "l46_gt_l45_count": 0,
            "consistent": False,
        }

    if "L45" not in scored[0]["layer_results"] or "L46" not in scored[0]["layer_results"]:
        return {
            "available": False,
            "total_prompts": len(scored),
            "l46_ge_l45_count": 0,
            "l46_gt_l45_count": 0,
            "consistent": False,
        }

    ge_count = 0
    gt_count = 0
    for row in scored:
        l45 = row["layer_results"]["L45"]["raw_exit_exact_match_rate"]
        l46 = row["layer_results"]["L46"]["raw_exit_exact_match_rate"]
        if l46 >= l45:
            ge_count += 1
        if l46 > l45:
            gt_count += 1

    return {
        "available": True,
        "total_prompts": len(scored),
        "l46_ge_l45_count": ge_count,
        "l46_gt_l45_count": gt_count,
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

    comparison = per_prompt_l46_vs_l45(data)
    return {
        "benchmark_json": str(path),
        "aggregates_scope": data.get("aggregates_scope"),
        "layers": layers,
        "l46_vs_l45_raw_exit": comparison,
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
    print(f"Scope: {summary['aggregates_scope']}")
    for label, layer in summary["layers"].items():
        print(
            f"{label}: "
            f"raw_exit_avg_exact_match={layer['raw_exit_avg_exact_match']:.3f}  "
            f"composed_avg_exact_match={layer['composed_avg_exact_match']:.3f}  "
            f"avg_top1_agreement_rate={layer['avg_top1_agreement_rate']:.3f}"
        )

    comparison = summary["l46_vs_l45_raw_exit"]
    if comparison["available"]:
        print(
            "L46_vs_L45_raw_exit: "
            f"{comparison['l46_ge_l45_count']}/{comparison['total_prompts']} prompts "
            "had L46 >= L45; "
            f"{comparison['l46_gt_l45_count']}/{comparison['total_prompts']} had L46 > L45; "
            f"consistent={comparison['consistent']}"
        )
    else:
        print("L46_vs_L45_raw_exit: unavailable")


if __name__ == "__main__":
    main()
