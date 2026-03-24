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
import math
import random
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

DEFAULT_KARMA_FEATURES = [
    "penultimate_entropy",
    "entropy_delta",
    "margin_delta",
    "logit_delta_l2",
    "rank_of_penultimate_top1_in_earlier",
    "rank_of_earlier_top1_in_penultimate",
    "topk_jaccard_at_k",
    "adjacent_top1_agree",
]

KARMA_POSITIVE_LABEL = "need_penultimate"
KARMA_NEGATIVE_LABEL = "need_full_depth"


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


def parse_feature_list(raw_features: str) -> List[str]:
    features: List[str] = []
    seen = set()
    for item in raw_features.split(","):
        feature = item.strip()
        if not feature or feature in seen:
            continue
        seen.add(feature)
        features.append(feature)
    if not features:
        raise ValueError("At least one karma feature must be provided.")
    return features


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


def validate_karma_rows(rows: List[Dict[str, Any]], features: List[str]) -> None:
    if not rows:
        raise ValueError("Karma fitting requires at least one Stage B row.")

    allowed_labels = {KARMA_POSITIVE_LABEL, KARMA_NEGATIVE_LABEL}
    unexpected_labels = sorted({row["oracle_triage_label"] for row in rows} - allowed_labels)
    if unexpected_labels:
        raise ValueError(
            "Karma fitting only supports Stage B rows with labels "
            f"{sorted(allowed_labels)}; saw {unexpected_labels}."
        )

    missing_features = sorted(
        feature for feature in features if any(feature not in row for row in rows)
    )
    if missing_features:
        raise ValueError(
            "Karma fitting requested features that are missing from the token rows: "
            f"{missing_features}"
        )


def coerce_feature_value(row: Dict[str, Any], feature: str) -> Optional[float]:
    value = row.get(feature)
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    raise ValueError(f"Feature '{feature}' must be numeric or boolean; saw {value!r}")


def karma_train_eval_split(
    rows: List[Dict[str, Any]],
    train_fraction: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--karma-train-fraction must be strictly between 0 and 1.")

    rng = random.Random(seed)
    grouped = {
        KARMA_POSITIVE_LABEL: [row for row in rows if row["oracle_triage_label"] == KARMA_POSITIVE_LABEL],
        KARMA_NEGATIVE_LABEL: [row for row in rows if row["oracle_triage_label"] == KARMA_NEGATIVE_LABEL],
    }

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for label in [KARMA_POSITIVE_LABEL, KARMA_NEGATIVE_LABEL]:
        label_rows = list(grouped[label])
        rng.shuffle(label_rows)
        if not label_rows:
            continue
        if len(label_rows) == 1:
            train_count = 1
        else:
            train_count = int(round(len(label_rows) * train_fraction))
            train_count = max(1, min(len(label_rows) - 1, train_count))
        train_rows.extend(label_rows[:train_count])
        eval_rows.extend(label_rows[train_count:])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    return train_rows, eval_rows


def default_stage_b_overlap_threshold(rows: List[Dict[str, Any]]) -> int:
    max_overlap = max(int(row["topk_overlap_at_k"]) for row in rows)
    return max(1, min(4, max_overlap))


def default_stage_b_logit_delta_threshold(rows: List[Dict[str, Any]]) -> Optional[float]:
    values = [float(row["logit_delta_l2"]) for row in rows if row["logit_delta_l2"] is not None]
    return median_or_none(values)


def fit_stage_b_karma_logistic(
    *,
    rows: List[Dict[str, Any]],
    features: List[str],
    karma_threshold: float,
    train_fraction: float,
    seed: int,
    topk_overlap_threshold: int,
    logit_delta_l2_threshold: Optional[float],
    penultimate_entropy_threshold: float,
) -> Dict[str, Any]:
    validate_karma_rows(rows, features)

    if not 0.0 <= karma_threshold <= 1.0:
        raise ValueError("--karma-threshold must be between 0 and 1.")

    train_rows, eval_rows = karma_train_eval_split(rows, train_fraction=train_fraction, seed=seed)
    train_counts = label_counts(train_rows)
    eval_counts = label_counts(eval_rows)

    result: Dict[str, Any] = {
        "available": False,
        "definition": "karma = probability_of_need_full_depth",
        "lower_is_safer": True,
        "threshold": karma_threshold,
        "train_fraction": train_fraction,
        "seed": seed,
        "chosen_features": features,
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "train_label_counts": train_counts,
        "eval_label_counts": eval_counts,
        "positive_class": KARMA_POSITIVE_LABEL,
        "negative_class": KARMA_NEGATIVE_LABEL,
    }

    if train_counts.get(KARMA_POSITIVE_LABEL, 0) == 0 or train_counts.get(KARMA_NEGATIVE_LABEL, 0) == 0:
        result["reason"] = (
            "karma training split must contain both Stage B labels "
            f"({KARMA_POSITIVE_LABEL}, {KARMA_NEGATIVE_LABEL})."
        )
        return result

    if not eval_rows:
        result["reason"] = "karma evaluation split is empty; reduce --karma-train-fraction."
        return result

    train_impute_medians: Dict[str, float] = {}
    train_missing_counts: Dict[str, int] = {}
    eval_missing_counts: Dict[str, int] = {}
    train_matrix: List[List[float]] = []
    eval_matrix: List[List[float]] = []

    for feature in features:
        observed = [coerce_feature_value(row, feature) for row in train_rows]
        present = [value for value in observed if value is not None]
        if not present:
            raise ValueError(
                f"Karma feature '{feature}' has no finite values in the training split."
            )
        train_impute_medians[feature] = float(statistics.median(present))
        train_missing_counts[feature] = sum(value is None for value in observed)
        eval_missing_counts[feature] = sum(
            coerce_feature_value(row, feature) is None for row in eval_rows
        )

    for row in train_rows:
        row_values: List[float] = []
        for feature in features:
            value = coerce_feature_value(row, feature)
            row_values.append(
                value if value is not None else train_impute_medians[feature]
            )
        train_matrix.append(row_values)
    for row in eval_rows:
        row_values = []
        for feature in features:
            value = coerce_feature_value(row, feature)
            row_values.append(
                value if value is not None else train_impute_medians[feature]
            )
        eval_matrix.append(row_values)

    y_train = [
        1.0 if row["oracle_triage_label"] == KARMA_POSITIVE_LABEL else 0.0
        for row in train_rows
    ]
    y_eval = [
        1.0 if row["oracle_triage_label"] == KARMA_POSITIVE_LABEL else 0.0
        for row in eval_rows
    ]

    means: List[float] = []
    stds: List[float] = []
    non_constant_mask: List[bool] = []
    for column_index in range(len(features)):
        column = [row[column_index] for row in train_matrix]
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std)
        non_constant_mask.append(std > 1e-12)

    x_train: List[List[float]] = []
    x_eval: List[List[float]] = []
    for raw_row in train_matrix:
        standardized_row = []
        for index, value in enumerate(raw_row):
            if not non_constant_mask[index]:
                standardized_row.append(0.0)
                continue
            standardized_row.append((value - means[index]) / stds[index])
        x_train.append(standardized_row)
    for raw_row in eval_matrix:
        standardized_row = []
        for index, value in enumerate(raw_row):
            if not non_constant_mask[index]:
                standardized_row.append(0.0)
                continue
            standardized_row.append((value - means[index]) / stds[index])
        x_eval.append(standardized_row)

    l2_penalty = 1e-4

    def dot(left: List[float], right: List[float]) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    def sigmoid(logit: float) -> float:
        if logit >= 0.0:
            exp_term = math.exp(-logit)
            return 1.0 / (1.0 + exp_term)
        exp_term = math.exp(logit)
        return exp_term / (1.0 + exp_term)

    def softplus(logit: float) -> float:
        if logit > 0.0:
            return logit + math.log1p(math.exp(-logit))
        return math.log1p(math.exp(logit))

    def logistic_loss(
        feature_matrix: List[List[float]],
        targets: List[float],
        model_weights: List[float],
        model_bias: float,
    ) -> float:
        total = 0.0
        for row, target in zip(feature_matrix, targets):
            logit = dot(model_weights, row) + model_bias
            total += softplus(logit) - target * logit
        total /= len(feature_matrix)
        total += 0.5 * l2_penalty * sum(weight * weight for weight in model_weights)
        return total

    def solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
        augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
        size = len(vector)
        for pivot_index in range(size):
            pivot_row = max(
                range(pivot_index, size),
                key=lambda row_index: abs(augmented[row_index][pivot_index]),
            )
            pivot_value = augmented[pivot_row][pivot_index]
            if abs(pivot_value) <= 1e-12:
                raise ValueError("Karma logistic Hessian is singular.")
            augmented[pivot_index], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_index],
            )
            pivot_value = augmented[pivot_index][pivot_index]
            for column_index in range(pivot_index, size + 1):
                augmented[pivot_index][column_index] /= pivot_value
            for row_index in range(size):
                if row_index == pivot_index:
                    continue
                factor = augmented[row_index][pivot_index]
                if factor == 0.0:
                    continue
                for column_index in range(pivot_index, size + 1):
                    augmented[row_index][column_index] -= (
                        factor * augmented[pivot_index][column_index]
                    )
        return [augmented[index][size] for index in range(size)]

    # The logistic model predicts need_penultimate. Karma is the opposite
    # probability: probability_of_need_full_depth = 1 - p(need_penultimate).
    weights = [0.0 for _ in features]
    bias = 0.0
    parameter_count = len(features) + 1
    for _ in range(50):
        gradient = [0.0 for _ in range(parameter_count)]
        hessian = [[0.0 for _ in range(parameter_count)] for _ in range(parameter_count)]

        for row, target in zip(x_train, y_train):
            logit = dot(weights, row) + bias
            probability = sigmoid(logit)
            error = probability - target
            curvature = probability * (1.0 - probability)

            gradient[0] += error
            hessian[0][0] += curvature
            for row_index, value_i in enumerate(row):
                gradient[row_index + 1] += error * value_i
                hessian[0][row_index + 1] += curvature * value_i
                hessian[row_index + 1][0] += curvature * value_i
                for column_index in range(row_index, len(row)):
                    contribution = curvature * value_i * row[column_index]
                    hessian[row_index + 1][column_index + 1] += contribution
                    if column_index != row_index:
                        hessian[column_index + 1][row_index + 1] += contribution

        row_count = float(len(x_train))
        gradient[0] /= row_count
        hessian[0][0] /= row_count
        hessian[0][0] += 1e-8
        for feature_index in range(len(features)):
            gradient[feature_index + 1] = (
                gradient[feature_index + 1] / row_count + l2_penalty * weights[feature_index]
            )
            hessian[0][feature_index + 1] /= row_count
            hessian[feature_index + 1][0] /= row_count
            for column_index in range(len(features)):
                hessian[feature_index + 1][column_index + 1] /= row_count
            hessian[feature_index + 1][feature_index + 1] += l2_penalty + 1e-8

        step = solve_linear_system(hessian, gradient)
        bias -= step[0]
        for feature_index in range(len(features)):
            weights[feature_index] -= step[feature_index + 1]
        if max(abs(value) for value in step) <= 1e-8:
            break

    train_loss = float(logistic_loss(x_train, y_train, weights, bias))
    eval_loss = float(logistic_loss(x_eval, y_eval, weights, bias))
    probability_need_penultimate = [sigmoid(dot(weights, row) + bias) for row in x_eval]
    probability_need_full_depth = [1.0 - probability for probability in probability_need_penultimate]

    scored_eval_rows: List[Dict[str, Any]] = []
    for row, p_need_penultimate, p_need_full_depth in zip(
        eval_rows,
        probability_need_penultimate,
        probability_need_full_depth,
    ):
        scored_row = dict(row)
        scored_row["probability_of_need_penultimate"] = round(float(p_need_penultimate), 6)
        scored_row["probability_of_need_full_depth"] = round(float(p_need_full_depth), 6)
        scored_row["karma"] = round(float(p_need_full_depth), 6)
        scored_eval_rows.append(scored_row)

    eval_rule = evaluate_accept_rule(
        stage="stage_b",
        scope="held-out Stage B rows",
        rows=scored_eval_rows,
        positive_label=KARMA_POSITIVE_LABEL,
        rule_name="karma_le",
        accept_fn=lambda row: float(row["karma"]) <= karma_threshold,
        parameters={"threshold": round(karma_threshold, 6)},
    )
    baseline_rules = build_stage_b_results(
        rows=eval_rows,
        topk_overlap_threshold=topk_overlap_threshold,
        logit_delta_l2_threshold=logit_delta_l2_threshold,
        penultimate_entropy_threshold=penultimate_entropy_threshold,
    )

    raw_bias = float(bias)
    raw_weights: Dict[str, float] = {}
    standardized_weights: Dict[str, float] = {}
    preprocessing: Dict[str, Dict[str, Any]] = {}
    for index, feature in enumerate(features):
        standardized_weight = float(weights[index])
        standardized_weights[feature] = round(standardized_weight, 6)
        preprocessing[feature] = {
            "train_impute_median": round(train_impute_medians[feature], 6),
            "train_mean": round(float(means[index]), 6),
            "train_std": round(float(stds[index]), 6),
            "train_missing_count": train_missing_counts[feature],
            "eval_missing_count": eval_missing_counts[feature],
        }
        if float(stds[index]) <= 1e-12:
            raw_weights[feature] = 0.0
            continue
        raw_weight = standardized_weight / float(stds[index])
        raw_weights[feature] = round(raw_weight, 6)
        raw_bias -= raw_weight * float(means[index])

    result.update(
        {
            "available": True,
            "train_loss": round(train_loss, 6),
            "eval_loss": round(eval_loss, 6),
            "weights": raw_weights,
            "bias": round(raw_bias, 6),
            "standardized_weights": standardized_weights,
            "standardized_bias": round(float(bias), 6),
            "feature_preprocessing": preprocessing,
            "evaluation": eval_rule,
            "baseline_comparison": baseline_rules,
        }
    )
    return result


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
    karma_logistic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = {
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
    if karma_logistic is not None:
        summary["stage_b_karma"] = karma_logistic
    return summary


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


def print_karma_summary(karma: Dict[str, Any]) -> None:
    if not karma.get("available"):
        print(
            "Karma Stage B logistic: "
            f"unavailable ({karma.get('reason', 'unknown reason')})"
        )
        return

    print("Karma Stage B logistic:")
    print(
        f"  train_rows={karma['train_row_count']}  "
        f"eval_rows={karma['eval_row_count']}  "
        f"train_labels={karma['train_label_counts']}  "
        f"eval_labels={karma['eval_label_counts']}"
    )
    print(f"  chosen_features={', '.join(karma['chosen_features'])}")
    print(
        "  definition=karma=probability_of_need_full_depth  "
        f"lower_is_safer=True  threshold={karma['threshold']}"
    )
    print(f"  bias={karma['bias']}  train_loss={karma['train_loss']}  eval_loss={karma['eval_loss']}")
    print("  weights:")
    for feature in karma["chosen_features"]:
        print(f"    {feature}: {karma['weights'][feature]}")
    print("  held-out evaluation:")
    print_rule(karma["evaluation"])
    print("  held-out Stage B baselines:")
    for rule in karma["baseline_comparison"]:
        print_rule(rule)


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
        "--fit-karma-logistic",
        action="store_true",
        help=(
            "Fit a small offline Stage B logistic model on held-out relation rows. "
            "Karma is reported as probability_of_need_full_depth, so lower is safer."
        ),
    )
    parser.add_argument(
        "--karma-threshold",
        type=float,
        default=0.5,
        help=(
            "Accept held-out Stage B rows when karma <= threshold. "
            "Karma is probability_of_need_full_depth."
        ),
    )
    parser.add_argument(
        "--karma-features",
        default=",".join(DEFAULT_KARMA_FEATURES),
        help=(
            "Comma-separated token-row fields for the karma logistic fit. "
            "Defaults to a narrow interpretable Stage B feature set."
        ),
    )
    parser.add_argument(
        "--karma-train-fraction",
        type=float,
        default=0.8,
        help="Fraction of Stage B rows used for karma training; the rest are held out for evaluation.",
    )
    parser.add_argument(
        "--karma-seed",
        type=int,
        default=0,
        help="Random seed for the deterministic karma train/eval split.",
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
    karma_result = None
    if args.fit_karma_logistic:
        karma_result = fit_stage_b_karma_logistic(
            rows=stage_b_rows(rows),
            features=parse_feature_list(args.karma_features),
            karma_threshold=args.karma_threshold,
            train_fraction=args.karma_train_fraction,
            seed=args.karma_seed,
            topk_overlap_threshold=overlap_threshold,
            logit_delta_l2_threshold=logit_delta_threshold,
            penultimate_entropy_threshold=args.stage_b_penultimate_entropy_threshold,
        )

    summary = build_summary(
        rows=rows,
        earlier_entropy_threshold=args.stage_a_entropy_threshold,
        earlier_margin_threshold=args.stage_a_margin_threshold,
        topk_overlap_threshold=overlap_threshold,
        logit_delta_l2_threshold=logit_delta_threshold,
        penultimate_entropy_threshold=args.stage_b_penultimate_entropy_threshold,
        input_path=path,
        karma_logistic=karma_result,
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
    if "stage_b_karma" in summary:
        print_karma_summary(summary["stage_b_karma"])


if __name__ == "__main__":
    main()
