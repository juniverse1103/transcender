"""
Compare per-step raw-exit hidden/logit diagnostics from a debug trace JSON.

This helper is intentionally narrow. It exists to answer the specific question:
for each tested exit layer, are the raw tensors actually identical to full
depth, or are the logits merely choosing the same argmax token?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_trace(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("non_finite_failure") is not None:
        return data
    if "trace" not in data or not isinstance(data["trace"], list) or not data["trace"]:
        raise ValueError(f"{path} does not contain a non-empty `trace` list.")
    return data


def infer_layer_labels(data: Dict[str, Any]) -> List[str]:
    for step in data.get("trace", []):
        labels = list(step.get("layers", {}).keys())
        if labels:
            return sorted(labels, key=lambda label: int(label[1:]))
    return []


def summarize_layer(trace_steps: List[Dict[str, Any]], layer_label: str) -> Dict[str, Any]:
    diagnostics = []
    for step in trace_steps:
        layer = step["layers"][layer_label]
        if "diagnostics" not in layer:
            raise ValueError(
                "Trace JSON does not contain `diagnostics` entries. "
                "Regenerate the trace with the current transcender_gpu_reproduction.py."
            )
        diagnostics.append((int(step["step_index"]), layer["diagnostics"]))

    steps = len(diagnostics)
    hidden_exact = sum(1 for _, row in diagnostics if row["hidden_exact_equal_to_full"])
    logits_exact = sum(1 for _, row in diagnostics if row["logits_exact_equal_to_full"])
    same_argmax_different_logits = sum(
        1 for _, row in diagnostics if row["same_argmax_but_logits_differ"]
    )
    non_finite_hidden_steps = [
        step_index
        for step_index, row in diagnostics
        if (
            not row["full_hidden_summary"]["all_finite"]
            or not row["raw_exit_hidden_summary"]["all_finite"]
        )
    ]
    non_finite_logits_steps = [
        step_index
        for step_index, row in diagnostics
        if (
            not row["full_logits_summary"]["all_finite"]
            or not row["raw_exit_logits_summary"]["all_finite"]
        )
    ]
    all_nan_logits_steps = [
        step_index
        for step_index, row in diagnostics
        if (
            row["full_logits_summary"]["has_nan"]
            and row["full_logits_summary"]["finite_count"] == 0
            and row["raw_exit_logits_summary"]["has_nan"]
            and row["raw_exit_logits_summary"]["finite_count"] == 0
        )
    ]

    hidden_diffs = [
        row["hidden_max_abs_diff_vs_full"]
        for _, row in diagnostics
        if row["hidden_max_abs_diff_vs_full"] is not None
    ]
    logits_diffs = [
        row["logits_max_abs_diff_vs_full"]
        for _, row in diagnostics
        if row["logits_max_abs_diff_vs_full"] is not None
    ]

    summary: Dict[str, Any] = {
        "steps": steps,
        "hidden_exact_equal_count": hidden_exact,
        "logits_exact_equal_count": logits_exact,
        "same_argmax_but_logits_differ_count": same_argmax_different_logits,
        "non_finite_hidden_step_count": len(non_finite_hidden_steps),
        "non_finite_hidden_step_indices": non_finite_hidden_steps,
        "non_finite_logits_step_count": len(non_finite_logits_steps),
        "non_finite_logits_step_indices": non_finite_logits_steps,
        "all_nan_logits_step_count": len(all_nan_logits_steps),
        "all_nan_logits_step_indices": all_nan_logits_steps,
        "hidden_max_abs_diff_min": min(hidden_diffs) if hidden_diffs else None,
        "hidden_max_abs_diff_max": max(hidden_diffs) if hidden_diffs else None,
        "logits_max_abs_diff_min": min(logits_diffs) if logits_diffs else None,
        "logits_max_abs_diff_max": max(logits_diffs) if logits_diffs else None,
    }

    softcap_values = [row.get("final_logit_softcap") for _, row in diagnostics if "final_logit_softcap" in row]
    if softcap_values:
        summary["final_logit_softcap"] = softcap_values[0]
        summary["full_max_abs_logit_over_softcap_max"] = max(
            row["full_max_abs_logit_over_softcap"] for _, row in diagnostics if "full_max_abs_logit_over_softcap" in row
        )
        summary["raw_exit_max_abs_logit_over_softcap_max"] = max(
            row["raw_exit_max_abs_logit_over_softcap"] for _, row in diagnostics if "raw_exit_max_abs_logit_over_softcap" in row
        )

    if len(all_nan_logits_steps) == steps and steps > 0:
        verdict = "all_nan_logits"
    elif non_finite_logits_steps or non_finite_hidden_steps:
        verdict = "non_finite_present"
    elif logits_exact == steps:
        verdict = "identical_logits"
    elif same_argmax_different_logits == steps:
        verdict = "different_logits_same_argmax"
    elif same_argmax_different_logits > 0:
        verdict = "mixed_same_argmax_and_divergent_argmax"
    else:
        verdict = "divergent_argmax_present"
    summary["verdict"] = verdict

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw-exit vs full-depth logits in a debug trace")
    parser.add_argument("trace_json", help="Path to the debug trace JSON")
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the summary as JSON instead of plain text",
    )
    args = parser.parse_args()

    path = Path(args.trace_json)
    data = load_trace(path)
    if data.get("non_finite_failure") is not None and not data.get("trace"):
        summary = {
            "trace_json": str(path),
            "prompt_id": data.get("prompt_id"),
            "model": data.get("model"),
            "model_family": data.get("model_family"),
            "non_finite_failure": data["non_finite_failure"],
        }
        if args.as_json:
            print(json.dumps(summary, indent=2))
            return
        print(f"Trace: {path}")
        print(f"Prompt: {data.get('prompt_id')}")
        print(
            "Non-finite failure: "
            f"step={data['non_finite_failure']['step_index']} "
            f"stage={data['non_finite_failure']['stage']}"
        )
        return

    layer_labels = infer_layer_labels(data)
    summary = {
        "trace_json": str(path),
        "prompt_id": data.get("prompt_id"),
        "model": data.get("model"),
        "model_family": data.get("model_family"),
        "non_finite_failure": data.get("non_finite_failure"),
        "layers": {
            label: summarize_layer(data["trace"], label) for label in layer_labels
        },
    }

    if args.as_json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Trace: {path}")
    print(f"Prompt: {data.get('prompt_id')}")
    if data.get("model"):
        print(f"Model: {data.get('model')}")
    if data.get("non_finite_failure") is not None:
        print(
            "Top-level non-finite failure: "
            f"step={data['non_finite_failure']['step_index']} "
            f"stage={data['non_finite_failure']['stage']}"
        )
    for label in layer_labels:
        row = summary["layers"][label]
        print(
            f"{label}: verdict={row['verdict']}  "
            f"hidden_exact={row['hidden_exact_equal_count']}/{row['steps']}  "
            f"logits_exact={row['logits_exact_equal_count']}/{row['steps']}  "
            f"same_argmax_but_logits_differ={row['same_argmax_but_logits_differ_count']}/{row['steps']}  "
            f"non_finite_hidden={row['non_finite_hidden_step_count']}/{row['steps']}  "
            f"non_finite_logits={row['non_finite_logits_step_count']}/{row['steps']}  "
            f"all_nan_logits={row['all_nan_logits_step_count']}/{row['steps']}"
        )
        if row["hidden_max_abs_diff_max"] is not None and row["logits_max_abs_diff_max"] is not None:
            print(
                f"  hidden_diff_max={row['hidden_max_abs_diff_max']:.8f}  "
                f"logits_diff_max={row['logits_max_abs_diff_max']:.8f}"
            )
        if "final_logit_softcap" in row:
            print(
                f"  softcap={row['final_logit_softcap']}  "
                f"full|max|/softcap_max={row['full_max_abs_logit_over_softcap_max']:.6f}  "
                f"exit|max|/softcap_max={row['raw_exit_max_abs_logit_over_softcap_max']:.6f}"
            )


if __name__ == "__main__":
    main()
