"""
Summarize a benchmark JSON produced by transcender_gpu_reproduction.py.

This helper reports:

- raw_exit_avg_exact_match by layer
- composed_avg_exact_match by layer
- avg_top1_agreement_rate by layer
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
    return {
        "benchmark_json": str(path),
        "model": data.get("model"),
        "model_family": data.get("model_family"),
        "aggregates_scope": data.get("aggregates_scope"),
        "layers": layers,
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
