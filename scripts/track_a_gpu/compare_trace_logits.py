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
    if "trace" not in data or not isinstance(data["trace"], list) or not data["trace"]:
        raise ValueError(f"{path} does not contain a non-empty `trace` list.")
    return data


def infer_layer_labels(data: Dict[str, Any]) -> List[str]:
    labels = list(data["trace"][0].get("layers", {}).keys())
    return sorted(labels, key=lambda label: int(label[1:]))


def summarize_layer(trace_steps: List[Dict[str, Any]], layer_label: str) -> Dict[str, Any]:
    diagnostics = []
    for step in trace_steps:
        layer = step["layers"][layer_label]
        if "diagnostics" not in layer:
            raise ValueError(
                "Trace JSON does not contain `diagnostics` entries. "
                "Regenerate the trace with the current transcender_gpu_reproduction.py."
            )
        diagnostics.append(layer["diagnostics"])

    steps = len(diagnostics)
    hidden_exact = sum(1 for row in diagnostics if row["hidden_exact_equal_to_full"])
    logits_exact = sum(1 for row in diagnostics if row["logits_exact_equal_to_full"])
    same_argmax_different_logits = sum(
        1 for row in diagnostics if row["same_argmax_but_logits_differ"]
    )

    hidden_diffs = [row["hidden_max_abs_diff_vs_full"] for row in diagnostics]
    logits_diffs = [row["logits_max_abs_diff_vs_full"] for row in diagnostics]

    summary: Dict[str, Any] = {
        "steps": steps,
        "hidden_exact_equal_count": hidden_exact,
        "logits_exact_equal_count": logits_exact,
        "same_argmax_but_logits_differ_count": same_argmax_different_logits,
        "hidden_max_abs_diff_min": min(hidden_diffs),
        "hidden_max_abs_diff_max": max(hidden_diffs),
        "logits_max_abs_diff_min": min(logits_diffs),
        "logits_max_abs_diff_max": max(logits_diffs),
    }

    softcap_values = [row.get("final_logit_softcap") for row in diagnostics if "final_logit_softcap" in row]
    if softcap_values:
        summary["final_logit_softcap"] = softcap_values[0]
        summary["full_max_abs_logit_over_softcap_max"] = max(
            row["full_max_abs_logit_over_softcap"] for row in diagnostics
        )
        summary["raw_exit_max_abs_logit_over_softcap_max"] = max(
            row["raw_exit_max_abs_logit_over_softcap"] for row in diagnostics
        )

    if logits_exact == steps:
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
    layer_labels = infer_layer_labels(data)
    summary = {
        "trace_json": str(path),
        "prompt_id": data.get("prompt_id"),
        "model": data.get("model"),
        "model_family": data.get("model_family"),
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
    for label in layer_labels:
        row = summary["layers"][label]
        print(
            f"{label}: verdict={row['verdict']}  "
            f"hidden_exact={row['hidden_exact_equal_count']}/{row['steps']}  "
            f"logits_exact={row['logits_exact_equal_count']}/{row['steps']}  "
            f"same_argmax_but_logits_differ={row['same_argmax_but_logits_differ_count']}/{row['steps']}  "
            f"hidden_diff_max={row['hidden_max_abs_diff_max']:.8f}  "
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
