"""
Evaluate simple offline proxy rules against token-level Track A relation rows.

This script is intentionally narrow. It does not build an online policy.
It evaluates simple acceptance rules against oracle-derived triage labels:

- earlier_correct
- need_penultimate
- need_full_depth

Stage A and Stage B are reported separately:

- Stage A evaluates earlier-only acceptance on all rows.
- Stage B evaluates penultimate acceptance only on rows where the earlier
  oracle label is not already correct.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


REQUIRED_FIELDS = {
    "model",
    "prompt_id",
    "step_index",
    "earlier_layer",
    "penultimate_layer",
    "earlier_margin",
    "earlier_entropy",
    "penultimate_entropy",
    "adjacent_top1_agree",
    "topk_overlap_at_k",
    "logit_delta_l2",
    "earlier_matches_full",
    "penultimate_matches_full",
    "oracle_triage_label",
}


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(
                    f"{path}:{line_number} is missing required fields: {sorted(missing)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} does not contain any token rows.")
    return rows


def label_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        label = row["oracle_triage_label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def evaluate_accept_rule(
    *,
    stage: str,
    scope: str,
    rows: List[Dict[str, Any]],
    positive_label: str,
    rule_name: str,
    accept_fn: Callable[[Dict[str, Any]], bool],
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    total_rows = len(rows)
    positive_count = sum(1 for row in rows if row["oracle_triage_label"] == positive_label)
    accepted_rows = [row for row in rows if accept_fn(row)]
    accepted_count = len(accepted_rows)
    accepted_label_counts = label_counts(accepted_rows)
    accepted_correct_count = accepted_label_counts.get(positive_label, 0)
    accepted_error_count = accepted_count - accepted_correct_count
    fallback_count = total_rows - accepted_count

    return {
        "stage": stage,
        "scope": scope,
        "rule_name": rule_name,
        "parameters": parameters or {},
        "positive_label": positive_label,
        "total_rows": total_rows,
        "positive_count": positive_count,
        "label_counts": label_counts(rows),
        "accepted_count": accepted_count,
        "fallback_count": fallback_count,
        "acceptance_rate": round(accepted_count / total_rows, 6) if total_rows else 0.0,
        "fallback_rate": round(fallback_count / total_rows, 6) if total_rows else 0.0,
        "accepted_correct_count": accepted_correct_count,
        "accepted_error_count": accepted_error_count,
        "accepted_precision": round(accepted_correct_count / accepted_count, 6)
        if accepted_count
        else None,
        "positive_recall": round(accepted_correct_count / positive_count, 6)
        if positive_count
        else None,
        "accepted_error_rate": round(accepted_error_count / total_rows, 6)
        if total_rows
        else 0.0,
        "accepted_label_counts": accepted_label_counts,
    }


def median_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def stage_b_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row["oracle_triage_label"] != "earlier_correct"]


def default_stage_b_overlap_threshold(rows: List[Dict[str, Any]]) -> int:
    max_overlap = max(int(row["topk_overlap_at_k"]) for row in rows)
    return max(1, min(4, max_overlap))


def default_stage_b_logit_delta_threshold(rows: List[Dict[str, Any]]) -> Optional[float]:
    values = [float(row["logit_delta_l2"]) for row in rows if row["logit_delta_l2"] is not None]
    return median_or_none(values)


def build_stage_a_results(
    rows: List[Dict[str, Any]],
    earlier_entropy_threshold: float,
    earlier_margin_threshold: float,
) -> List[Dict[str, Any]]:
    return [
        evaluate_accept_rule(
            stage="stage_a",
            scope="all rows",
            rows=rows,
            positive_label="earlier_correct",
            rule_name="earlier_entropy_le",
            accept_fn=lambda row: float(row["earlier_entropy"]) <= earlier_entropy_threshold,
            parameters={"threshold": earlier_entropy_threshold},
        ),
        evaluate_accept_rule(
            stage="stage_a",
            scope="all rows",
            rows=rows,
            positive_label="earlier_correct",
            rule_name="earlier_margin_ge",
            accept_fn=lambda row: float(row["earlier_margin"]) >= earlier_margin_threshold,
            parameters={"threshold": earlier_margin_threshold},
        ),
    ]


def build_stage_b_results(
    rows: List[Dict[str, Any]],
    topk_overlap_threshold: int,
    logit_delta_l2_threshold: Optional[float],
    penultimate_entropy_threshold: float,
) -> List[Dict[str, Any]]:
    scoped_rows = stage_b_rows(rows)
    results = [
        evaluate_accept_rule(
            stage="stage_b",
            scope="rows where earlier oracle miss requires deeper decision",
            rows=scoped_rows,
            positive_label="need_penultimate",
            rule_name="adjacent_top1_agree",
            accept_fn=lambda row: bool(row["adjacent_top1_agree"]),
        ),
        evaluate_accept_rule(
            stage="stage_b",
            scope="rows where earlier oracle miss requires deeper decision",
            rows=scoped_rows,
            positive_label="need_penultimate",
            rule_name="adjacent_top1_agree_and_topk_overlap_ge",
            accept_fn=lambda row: bool(row["adjacent_top1_agree"])
            and int(row["topk_overlap_at_k"]) >= topk_overlap_threshold,
            parameters={"threshold": topk_overlap_threshold},
        ),
        evaluate_accept_rule(
            stage="stage_b",
            scope="rows where earlier oracle miss requires deeper decision",
            rows=scoped_rows,
            positive_label="need_penultimate",
            rule_name="penultimate_entropy_le",
            accept_fn=lambda row: float(row["penultimate_entropy"]) <= penultimate_entropy_threshold,
            parameters={"threshold": penultimate_entropy_threshold},
        ),
    ]
    if logit_delta_l2_threshold is not None:
        results.append(
            evaluate_accept_rule(
                stage="stage_b",
                scope="rows where earlier oracle miss requires deeper decision",
                rows=scoped_rows,
                positive_label="need_penultimate",
                rule_name="adjacent_top1_agree_and_logit_delta_l2_le",
                accept_fn=lambda row: bool(row["adjacent_top1_agree"])
                and float(row["logit_delta_l2"]) <= logit_delta_l2_threshold,
                parameters={"threshold": round(logit_delta_l2_threshold, 6)},
            )
        )
    return results


def build_summary(
    rows: List[Dict[str, Any]],
    earlier_entropy_threshold: float,
    earlier_margin_threshold: float,
    topk_overlap_threshold: int,
    logit_delta_l2_threshold: Optional[float],
    penultimate_entropy_threshold: float,
    input_path: Path,
) -> Dict[str, Any]:
    return {
        "input_jsonl": str(input_path),
        "models": sorted({row["model"] for row in rows}),
        "earlier_layers": sorted({row["earlier_layer"] for row in rows}),
        "penultimate_layers": sorted({row["penultimate_layer"] for row in rows}),
        "row_count": len(rows),
        "label_counts": label_counts(rows),
        "stage_a_rules": build_stage_a_results(
            rows=rows,
            earlier_entropy_threshold=earlier_entropy_threshold,
            earlier_margin_threshold=earlier_margin_threshold,
        ),
        "stage_b_scope": {
            "row_count": len(stage_b_rows(rows)),
            "label_counts": label_counts(stage_b_rows(rows)),
        },
        "stage_b_rules": build_stage_b_results(
            rows=rows,
            topk_overlap_threshold=topk_overlap_threshold,
            logit_delta_l2_threshold=logit_delta_l2_threshold,
            penultimate_entropy_threshold=penultimate_entropy_threshold,
        ),
        "notes": (
            "Offline proxy evaluation only. These rules are measured against "
            "oracle-derived labels and are not deployable policies."
        ),
    }


def print_rule(rule: Dict[str, Any]) -> None:
    params = rule.get("parameters") or {}
    param_text = ""
    if params:
        rendered = ", ".join(f"{key}={value}" for key, value in params.items())
        param_text = f" ({rendered})"
    print(
        f"  {rule['rule_name']}{param_text}: "
        f"accept={rule['accepted_count']}/{rule['total_rows']} "
        f"({rule['acceptance_rate']:.3f})  "
        f"fallback={rule['fallback_count']}/{rule['total_rows']}  "
        f"accepted_precision={rule['accepted_precision'] if rule['accepted_precision'] is not None else 'n/a'}  "
        f"positive_recall={rule['positive_recall'] if rule['positive_recall'] is not None else 'n/a'}  "
        f"accepted_error_rate={rule['accepted_error_rate']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline relation-based proxy rules")
    parser.add_argument("token_rows_jsonl", help="Path to token-level relation rows JSONL")
    parser.add_argument(
        "--stage-a-entropy-threshold",
        type=float,
        default=2.5,
        help="Earlier entropy acceptance threshold for the Stage A baseline.",
    )
    parser.add_argument(
        "--stage-a-margin-threshold",
        type=float,
        default=1.0,
        help="Earlier margin acceptance threshold for the Stage A baseline.",
    )
    parser.add_argument(
        "--stage-b-topk-overlap-threshold",
        type=int,
        default=None,
        help="Stage B top-k overlap threshold. Defaults to a conservative overlap derived from the data.",
    )
    parser.add_argument(
        "--stage-b-logit-delta-l2-threshold",
        type=float,
        default=None,
        help="Stage B logit-delta L2 threshold. Defaults to the median observed value.",
    )
    parser.add_argument(
        "--stage-b-penultimate-entropy-threshold",
        type=float,
        default=2.5,
        help="Penultimate entropy acceptance threshold for the Stage B baseline.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the evaluation summary as JSON instead of plain text.",
    )
    args = parser.parse_args()

    path = Path(args.token_rows_jsonl)
    rows = load_rows(path)
    overlap_threshold = (
        args.stage_b_topk_overlap_threshold
        if args.stage_b_topk_overlap_threshold is not None
        else default_stage_b_overlap_threshold(rows)
    )
    logit_delta_threshold = (
        args.stage_b_logit_delta_l2_threshold
        if args.stage_b_logit_delta_l2_threshold is not None
        else default_stage_b_logit_delta_threshold(stage_b_rows(rows))
    )

    summary = build_summary(
        rows=rows,
        earlier_entropy_threshold=args.stage_a_entropy_threshold,
        earlier_margin_threshold=args.stage_a_margin_threshold,
        topk_overlap_threshold=overlap_threshold,
        logit_delta_l2_threshold=logit_delta_threshold,
        penultimate_entropy_threshold=args.stage_b_penultimate_entropy_threshold,
        input_path=path,
    )

    if args.as_json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Token rows: {path}")
    print(f"Models: {', '.join(summary['models'])}")
    print(f"Rows: {summary['row_count']}")
    print(f"Label counts: {summary['label_counts']}")
    print("Stage A:")
    for rule in summary["stage_a_rules"]:
        print_rule(rule)
    print("Stage B:")
    print(
        f"  scope_rows={summary['stage_b_scope']['row_count']}  "
        f"label_counts={summary['stage_b_scope']['label_counts']}"
    )
    for rule in summary["stage_b_rules"]:
        print_rule(rule)


if __name__ == "__main__":
    main()
