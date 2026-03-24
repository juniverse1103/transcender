"""
Summarize offline Stage B karma evaluation JSON outputs.

This helper reads one or more files produced by:

    python scripts/track_a_gpu/evaluate_relation_proxies.py ... --fit-karma-logistic --as-json

It stays offline and descriptive. The output is intended for research notes,
paper drafting, and quick comparison across models. It can emit a compact
paper-facing markdown table or CSV without adding any non-stdlib dependency.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from karma_reporting_utils import (
    BASELINE_RULE_ORDER,
    COMPARISON_BASELINE_RULES,
    extract_karma_record,
    format_feature_list,
    format_metric,
)


EXPORT_COLUMNS = [
    ("model", "model"),
    ("model_full", "model_full"),
    ("features", "features"),
    ("threshold", "threshold"),
    ("stage_b_scope_rows", "stage_b_scope_rows"),
    ("eval_rows", "eval_rows"),
    ("karma_accepted_precision", "karma_accepted_precision"),
    ("karma_positive_recall", "karma_positive_recall"),
    ("karma_accepted_error_rate", "karma_accepted_error_rate"),
    ("entropy_accepted_precision", "entropy_accepted_precision"),
    ("entropy_positive_recall", "entropy_positive_recall"),
    ("entropy_accepted_error_rate", "entropy_accepted_error_rate"),
    ("delta_accepted_precision", "delta_accepted_precision"),
    ("delta_positive_recall", "delta_positive_recall"),
    ("delta_accepted_error_rate", "delta_accepted_error_rate"),
    ("result_json", "result_json"),
    ("input_jsonl", "input_jsonl"),
]


def load_records(paths: List[str]) -> List[Dict[str, Any]]:
    return [extract_karma_record(Path(path)) for path in paths]


def export_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        karma = record.get("karma_evaluation", {})
        entropy = record.get("baseline_comparison", {}).get("penultimate_entropy_le", {})
        deltas = record.get("deltas_vs_penultimate_entropy", {})
        rows.append(
            {
                "model": record.get("model_short"),
                "model_full": record.get("model"),
                "features": record.get("feature_label"),
                "threshold": record.get("threshold"),
                "stage_b_scope_rows": record.get("stage_b_scope_row_count"),
                "eval_rows": record.get("eval_row_count"),
                "karma_accepted_precision": karma.get("accepted_precision"),
                "karma_positive_recall": karma.get("positive_recall"),
                "karma_accepted_error_rate": karma.get("accepted_error_rate"),
                "entropy_accepted_precision": entropy.get("accepted_precision"),
                "entropy_positive_recall": entropy.get("positive_recall"),
                "entropy_accepted_error_rate": entropy.get("accepted_error_rate"),
                "delta_accepted_precision": deltas.get("delta_accepted_precision"),
                "delta_positive_recall": deltas.get("delta_positive_recall"),
                "delta_accepted_error_rate": deltas.get("delta_accepted_error_rate"),
                "result_json": record.get("path"),
                "input_jsonl": record.get("input_jsonl"),
            }
        )
    return rows


def render_text(records: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for record in records:
        lines.append(f"Result: {record['path']}")
        lines.append(f"  model={record['model']}")
        lines.append(f"  input_jsonl={record['input_jsonl']}")
        lines.append(f"  stage_b_scope_rows={record['stage_b_scope_row_count']}")
        if not record["available"]:
            lines.append(f"  karma=unavailable ({record.get('reason', 'unknown reason')})")
            lines.append("")
            continue
        lines.append(f"  karma_threshold={record['threshold']}")
        lines.append(f"  karma_features={format_feature_list(record['chosen_features'])}")
        karma = record["karma_evaluation"]
        lines.append(
            "  karma:"
            f" accept={karma['accepted_count']}/{record['eval_row_count']}"
            f" fallback={karma['fallback_count']}/{record['eval_row_count']}"
            f" acceptance_rate={format_metric(karma['acceptance_rate'])}"
            f" accepted_precision={format_metric(karma['accepted_precision'])}"
            f" positive_recall={format_metric(karma['positive_recall'])}"
            f" accepted_error_rate={format_metric(karma['accepted_error_rate'])}"
        )
        lines.append("  baselines:")
        for rule_name in BASELINE_RULE_ORDER:
            baseline = record["baseline_comparison"].get(rule_name, {})
            if rule_name not in COMPARISON_BASELINE_RULES and baseline.get("acceptance_rate") is None:
                continue
            lines.append(
                f"    {rule_name}:"
                f" acceptance_rate={format_metric(baseline.get('acceptance_rate'))}"
                f" accepted_precision={format_metric(baseline.get('accepted_precision'))}"
                f" positive_recall={format_metric(baseline.get('positive_recall'))}"
                f" accepted_error_rate={format_metric(baseline.get('accepted_error_rate'))}"
            )
        deltas = record["deltas_vs_penultimate_entropy"]
        lines.append(
            "  deltas_vs_penultimate_entropy:"
            f" delta_acceptance_rate={format_metric(deltas.get('delta_acceptance_rate'))}"
            f" delta_accepted_precision={format_metric(deltas.get('delta_accepted_precision'))}"
            f" delta_positive_recall={format_metric(deltas.get('delta_positive_recall'))}"
            f" delta_accepted_error_rate={format_metric(deltas.get('delta_accepted_error_rate'))}"
        )
        lines.append("")
    if lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def render_markdown(records: List[Dict[str, Any]]) -> str:
    rows = export_rows(records)

    def format_count(value: Any) -> str:
        if value is None:
            return "n/a"
        return str(value)

    lines = [
        "| model | features | threshold | stage_b rows | eval rows | karma precision | karma recall | karma error | entropy precision | entropy recall | entropy error | delta precision | delta recall | delta error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model") or "n/a"),
                    str(row.get("features") or "n/a"),
                    format_metric(row.get("threshold")),
                    format_count(row.get("stage_b_scope_rows")),
                    format_count(row.get("eval_rows")),
                    format_metric(row.get("karma_accepted_precision")),
                    format_metric(row.get("karma_positive_recall")),
                    format_metric(row.get("karma_accepted_error_rate")),
                    format_metric(row.get("entropy_accepted_precision")),
                    format_metric(row.get("entropy_positive_recall")),
                    format_metric(row.get("entropy_accepted_error_rate")),
                    format_metric(row.get("delta_accepted_precision")),
                    format_metric(row.get("delta_positive_recall")),
                    format_metric(row.get("delta_accepted_error_rate")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_csv(records: List[Dict[str, Any]]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=[field for field, _ in EXPORT_COLUMNS])
    writer.writeheader()
    for row in export_rows(records):
        normalized = {}
        for field_name, _label in EXPORT_COLUMNS:
            value = row.get(field_name)
            if isinstance(value, float):
                normalized[field_name] = f"{value:.6f}"
            elif value is None:
                normalized[field_name] = ""
            else:
                normalized[field_name] = value
        writer.writerow(normalized)
    return handle.getvalue().rstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize offline Stage B karma result JSON files")
    parser.add_argument("result_jsons", nargs="+", help="JSON files from evaluate_relation_proxies.py --as-json")
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown", "csv"],
        default="text",
        help="Output format. Default: text.",
    )
    args = parser.parse_args()

    records = load_records(args.result_jsons)

    if args.format == "json":
        print(json.dumps(records, indent=2))
        return
    if args.format == "csv":
        print(render_csv(records))
        return
    if args.format == "markdown":
        print(render_markdown(records))
        return
    print(render_text(records))


if __name__ == "__main__":
    main()
