"""
Shared helpers for offline Stage B karma reporting.

These utilities are intentionally small and stdlib-only. They operate on the
JSON produced by evaluate_relation_proxies.py --as-json and avoid any serving
or policy claims. The focus is descriptive reporting for offline evaluation.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PRIMARY_BASELINE_RULE_NAME = "penultimate_entropy_le"
BASELINE_RULE_ORDER = [
    "penultimate_entropy_le",
    "penultimate_margin_ge",
    "adjacent_top1_agree",
    "adjacent_top1_agree_and_topk_overlap_ge",
    "adjacent_top1_agree_and_logit_delta_l2_le",
]
COMPARISON_BASELINE_RULES = [
    "penultimate_entropy_le",
    "adjacent_top1_agree",
    "adjacent_top1_agree_and_topk_overlap_ge",
    "adjacent_top1_agree_and_logit_delta_l2_le",
]
METRIC_FIELDS = [
    "acceptance_rate",
    "accepted_precision",
    "positive_recall",
    "accepted_error_rate",
]
FEATURE_SIGNATURE_LABELS = {
    "penultimate_entropy,entropy_delta,margin_delta,logit_delta_l2,rank_of_penultimate_top1_in_earlier,rank_of_earlier_top1_in_penultimate,topk_jaccard_at_k,adjacent_top1_agree": "full",
    "penultimate_entropy,topk_jaccard_at_k,adjacent_top1_agree": "relation_lite",
    "penultimate_entropy": "entropy_only",
}


def load_proxy_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    required = {"input_jsonl", "stage_b_scope"}
    missing = required - data.keys()
    if missing:
        raise ValueError(
            f"{path} does not look like evaluate_relation_proxies.py --as-json output; "
            f"missing {sorted(missing)}"
        )
    return data


def model_name(summary: Dict[str, Any]) -> str:
    models = summary.get("models") or []
    if not models:
        return "unknown"
    return ", ".join(str(model) for model in models)


def short_model_name(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def rule_map(rules: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        rule["rule_name"]: rule
        for rule in rules
        if isinstance(rule, dict) and "rule_name" in rule
    }


def metric_subset(rule: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not rule:
        return {
            "accepted_count": None,
            "fallback_count": None,
            "acceptance_rate": None,
            "accepted_precision": None,
            "positive_recall": None,
            "accepted_error_rate": None,
        }
    return {
        "accepted_count": rule.get("accepted_count"),
        "fallback_count": rule.get("fallback_count"),
        "acceptance_rate": rule.get("acceptance_rate"),
        "accepted_precision": rule.get("accepted_precision"),
        "positive_recall": rule.get("positive_recall"),
        "accepted_error_rate": rule.get("accepted_error_rate"),
    }


def safe_delta(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def delta_metrics(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    return {
        "delta_acceptance_rate": safe_delta(
            left.get("acceptance_rate"),
            right.get("acceptance_rate"),
        ),
        "delta_accepted_precision": safe_delta(
            left.get("accepted_precision"),
            right.get("accepted_precision"),
        ),
        "delta_positive_recall": safe_delta(
            left.get("positive_recall"),
            right.get("positive_recall"),
        ),
        "delta_accepted_error_rate": safe_delta(
            left.get("accepted_error_rate"),
            right.get("accepted_error_rate"),
        ),
    }


def format_metric(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def format_feature_list(features: Sequence[str]) -> str:
    return ",".join(str(feature) for feature in features)


def feature_set_label(features: Sequence[str]) -> str:
    signature = format_feature_list(features)
    return FEATURE_SIGNATURE_LABELS.get(signature, signature)


def summarize_numeric_values(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(present),
        "mean": round(statistics.fmean(present), 6),
        "std": round(statistics.stdev(present), 6) if len(present) > 1 else 0.0,
        "min": round(min(present), 6),
        "max": round(max(present), 6),
    }


def harmonic_mean(
    left: Optional[float],
    right: Optional[float],
) -> float:
    if left is None or right is None:
        return 0.0
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return 2.0 * left * right / (left + right)


def extract_karma_record(path: Path) -> Dict[str, Any]:
    summary = load_proxy_summary(path)
    karma = summary.get("stage_b_karma")
    if karma is None:
        raise ValueError(
            f"{path} does not contain stage_b_karma; rerun with "
            "--fit-karma-logistic --as-json"
        )

    record: Dict[str, Any] = {
        "path": str(path),
        "file_label": path.stem,
        "input_jsonl": summary.get("input_jsonl"),
        "model": model_name(summary),
        "model_short": short_model_name(model_name(summary)),
        "stage_b_scope_row_count": summary.get("stage_b_scope", {}).get("row_count"),
        "stage_b_scope_label_counts": summary.get("stage_b_scope", {}).get("label_counts", {}),
        "available": bool(karma.get("available")),
        "reason": karma.get("reason"),
        "threshold": karma.get("threshold"),
        "seed": karma.get("seed"),
        "train_fraction": karma.get("train_fraction"),
        "train_row_count": karma.get("train_row_count"),
        "eval_row_count": karma.get("eval_row_count"),
        "chosen_features": list(karma.get("chosen_features", [])),
        "feature_signature": format_feature_list(karma.get("chosen_features", [])),
        "feature_label": feature_set_label(karma.get("chosen_features", [])),
        "definition": karma.get("definition"),
        "lower_is_safer": karma.get("lower_is_safer"),
        "train_loss": karma.get("train_loss"),
        "eval_loss": karma.get("eval_loss"),
    }

    if not record["available"]:
        record["karma_evaluation"] = metric_subset(None)
        record["baseline_comparison"] = {}
        record["deltas_vs_penultimate_entropy"] = delta_metrics(
            record["karma_evaluation"],
            metric_subset(None),
        )
        return record

    evaluation = karma.get("evaluation")
    if evaluation is None:
        raise ValueError(f"{path} has stage_b_karma but no evaluation block.")

    baselines = rule_map(karma.get("baseline_comparison", []))
    record["karma_evaluation"] = metric_subset(evaluation)
    record["baseline_comparison"] = {
        rule_name: metric_subset(baselines.get(rule_name))
        for rule_name in BASELINE_RULE_ORDER
    }
    record["deltas_vs_penultimate_entropy"] = delta_metrics(
        record["karma_evaluation"],
        record["baseline_comparison"][PRIMARY_BASELINE_RULE_NAME],
    )
    return record
