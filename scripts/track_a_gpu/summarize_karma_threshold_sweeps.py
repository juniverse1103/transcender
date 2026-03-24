"""
Summarize offline Stage B karma threshold sweeps.

Inputs are JSON files produced by evaluate_relation_proxies.py --fit-karma-logistic --as-json
with different --karma-threshold values. The script stays descriptive and does
not claim a universal threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from karma_reporting_utils import extract_karma_record, format_metric, harmonic_mean


def load_records(paths: List[str]) -> List[Dict[str, Any]]:
    records = [extract_karma_record(Path(path)) for path in paths]
    return [record for record in records if record["available"]]


def grouped_by_model(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["model"], []).append(record)
    for rows in grouped.values():
        rows.sort(key=lambda row: (float(row["threshold"]), row["path"]))
    return grouped


def selection_rank(
    record: Dict[str, Any],
    mode: str,
    recall_floor: Optional[float],
) -> Optional[tuple]:
    metrics = record["karma_evaluation"]
    precision = metrics.get("accepted_precision")
    recall = metrics.get("positive_recall")
    error = metrics.get("accepted_error_rate")
    acceptance = metrics.get("acceptance_rate")

    precision_value = -1.0 if precision is None else float(precision)
    recall_value = -1.0 if recall is None else float(recall)
    error_value = float("inf") if error is None else float(error)
    acceptance_value = -1.0 if acceptance is None else float(acceptance)

    if mode == "min_error":
        return (error_value, -recall_value, -acceptance_value)

    if mode == "max_precision_under_recall_floor":
        if recall_floor is None:
            raise ValueError("--recall-floor is required for --select-by max_precision_under_recall_floor")
        if recall is None or float(recall) < recall_floor:
            return None
        return (-precision_value, error_value, -acceptance_value)

    if mode == "max_f1_like":
        f1_like = harmonic_mean(precision, recall)
        return (-f1_like, error_value)

    raise ValueError(f"Unsupported selection mode: {mode}")


def pick_best(
    records: List[Dict[str, Any]],
    mode: str,
    recall_floor: Optional[float],
) -> Dict[str, Any]:
    scored: List[tuple[tuple, Dict[str, Any]]] = []
    for record in records:
        rank = selection_rank(record, mode=mode, recall_floor=recall_floor)
        if rank is None:
            continue
        scored.append((rank, record))

    if not scored:
        return {
            "available": False,
            "reason": "no thresholds satisfied the requested policy",
        }

    scored.sort(key=lambda item: (item[0], float(item[1]["threshold"])))
    best_rank = scored[0][0]
    tied = [record for rank, record in scored if rank == best_rank]
    return {
        "available": True,
        "record": scored[0][1],
        "tied_thresholds": [row["threshold"] for row in tied],
        "ambiguous": len(tied) > 1,
    }


def render_text(grouped: Dict[str, List[Dict[str, Any]]], selection_modes: List[str], recall_floor: Optional[float]) -> str:
    lines: List[str] = []
    for model in sorted(grouped):
        rows = grouped[model]
        lines.append(f"Model: {model}")
        lines.append("  threshold  acceptance  precision  recall  error")
        for record in rows:
            metrics = record["karma_evaluation"]
            lines.append(
                "  "
                f"{float(record['threshold']):>8.3f}  "
                f"{format_metric(metrics.get('acceptance_rate')):>10}  "
                f"{format_metric(metrics.get('accepted_precision')):>9}  "
                f"{format_metric(metrics.get('positive_recall')):>6}  "
                f"{format_metric(metrics.get('accepted_error_rate')):>5}"
            )
        if selection_modes:
            lines.append("  selections:")
        for mode in selection_modes:
            best = pick_best(rows, mode=mode, recall_floor=recall_floor)
            if not best["available"]:
                lines.append(f"    {mode}: unavailable ({best['reason']})")
                continue
            record = best["record"]
            metrics = record["karma_evaluation"]
            tie_suffix = ""
            if best["ambiguous"]:
                tied = ", ".join(f"{float(threshold):.3f}" for threshold in best["tied_thresholds"])
                tie_suffix = f"  ties={tied}"
            lines.append(
                f"    {mode}: threshold={float(record['threshold']):.3f}"
                f" acceptance_rate={format_metric(metrics.get('acceptance_rate'))}"
                f" accepted_precision={format_metric(metrics.get('accepted_precision'))}"
                f" positive_recall={format_metric(metrics.get('positive_recall'))}"
                f" accepted_error_rate={format_metric(metrics.get('accepted_error_rate'))}"
                f"{tie_suffix}"
            )
        lines.append("")
    if lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def render_json(grouped: Dict[str, List[Dict[str, Any]]], selection_modes: List[str], recall_floor: Optional[float]) -> str:
    payload: Dict[str, Any] = {}
    for model, rows in grouped.items():
        payload[model] = {
            "rows": rows,
            "selections": {
                mode: pick_best(rows, mode=mode, recall_floor=recall_floor)
                for mode in selection_modes
            },
        }
    return json.dumps(payload, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize offline Stage B karma threshold sweeps")
    parser.add_argument("result_jsons", nargs="+", help="JSON files from evaluate_relation_proxies.py --as-json")
    parser.add_argument(
        "--select-by",
        action="append",
        choices=[
            "min_error",
            "max_precision_under_recall_floor",
            "max_f1_like",
        ],
        default=[],
        help="Optional threshold-selection policy. May be passed multiple times.",
    )
    parser.add_argument(
        "--recall-floor",
        type=float,
        default=None,
        help="Recall floor for --select-by max_precision_under_recall_floor.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format. Default: text.",
    )
    args = parser.parse_args()

    grouped = grouped_by_model(load_records(args.result_jsons))

    if args.format == "json":
        print(render_json(grouped, selection_modes=args.select_by, recall_floor=args.recall_floor))
        return
    print(render_text(grouped, selection_modes=args.select_by, recall_floor=args.recall_floor))


if __name__ == "__main__":
    main()
