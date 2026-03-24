"""
Summarize offline Stage B karma seed sweeps.

Inputs are JSON files produced by evaluate_relation_proxies.py --fit-karma-logistic --as-json
with different --karma-seed values. The output is intended to show split
stability, not to claim deployment robustness.
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

from karma_reporting_utils import (
    extract_karma_record,
    format_feature_list,
    format_metric,
    summarize_numeric_values,
)


def load_records(paths: List[str]) -> List[Dict[str, Any]]:
    records = [extract_karma_record(Path(path)) for path in paths]
    return [record for record in records if record["available"]]


def grouped_by_model(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["model"], []).append(record)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row.get("seed", 0)), row["path"]))
    return grouped


def render_text(grouped: Dict[str, List[Dict[str, Any]]]) -> str:
    lines: List[str] = []
    metric_fields = [
        "accepted_precision",
        "positive_recall",
        "accepted_error_rate",
        "acceptance_rate",
    ]
    for model in sorted(grouped):
        rows = grouped[model]
        thresholds = sorted({row.get("threshold") for row in rows})
        feature_signatures = sorted({row.get("feature_signature", "") for row in rows})
        seeds = [row.get("seed") for row in rows]
        lines.append(f"Model: {model}")
        lines.append(f"  seeds_found={len(rows)}  seeds={seeds}")
        if len(thresholds) == 1:
            lines.append(f"  threshold={thresholds[0]}")
        else:
            lines.append(f"  thresholds={thresholds}")
        if len(feature_signatures) == 1:
            lines.append(f"  features={feature_signatures[0]}")
        else:
            lines.append(f"  feature_sets={feature_signatures}")
        for field in metric_fields:
            stats = summarize_numeric_values(
                [row["karma_evaluation"].get(field) for row in rows]
            )
            lines.append(
                f"  {field}:"
                f" mean={format_metric(stats.get('mean'))}"
                f" std={format_metric(stats.get('std'))}"
                f" min={format_metric(stats.get('min'))}"
                f" max={format_metric(stats.get('max'))}"
            )
        lines.append("")
    if lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def render_json(grouped: Dict[str, List[Dict[str, Any]]]) -> str:
    payload: Dict[str, Any] = {}
    for model, rows in grouped.items():
        payload[model] = {
            "seed_count": len(rows),
            "seeds": [row.get("seed") for row in rows],
            "thresholds": sorted({row.get("threshold") for row in rows}),
            "feature_sets": sorted({row.get("feature_signature", "") for row in rows}),
            "metrics": {
                field: summarize_numeric_values(
                    [row["karma_evaluation"].get(field) for row in rows]
                )
                for field in [
                    "accepted_precision",
                    "positive_recall",
                    "accepted_error_rate",
                    "acceptance_rate",
                ]
            },
        }
    return json.dumps(payload, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize offline Stage B karma seed sweeps")
    parser.add_argument("result_jsons", nargs="+", help="JSON files from evaluate_relation_proxies.py --as-json")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format. Default: text.",
    )
    args = parser.parse_args()

    grouped = grouped_by_model(load_records(args.result_jsons))

    if args.format == "json":
        print(render_json(grouped))
        return
    print(render_text(grouped))


if __name__ == "__main__":
    main()
