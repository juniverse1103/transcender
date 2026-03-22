"""
Dense Exit-Layer Agreement Sweep — Frontier Discovery Tool

For a given dense model, measures compute-both agreement rate and quality
at multiple exit layers. This answers the key question:

  "At which exit layer does shallow top1 agree with deep top1 often enough
   to support a speed-quality frontier?"

The output is a JSON artifact + CSV summary that directly informs whether
self-speculative decoding or selective-depth can work at each exit.

Usage:
  python scripts/track_c/transcender_dense_exit_sweep.py \
    --model /path/to/llama-3.1-8b-instruct-4bit \
    --family llama \
    --exit-layers 10,12,14,16,18,20,22,24,26,28 \
    --output artifacts/dense_followup/llama_exit_sweep.json \
    --max-new-tokens 48

  python scripts/track_c/transcender_dense_exit_sweep.py \
    --model /path/to/gemma-3-4b-it \
    --family gemma \
    --exit-layers 10,14,18,20,22,24,26,28,30 \
    --output artifacts/dense_followup/gemma_exit_sweep.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np

# Add sibling script directories for shared imports
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_script_dir.parent / "track_a"))
sys.path.insert(0, str(_script_dir.parent / "track_b"))

from mlx_lm import load as mlx_load
from mlx_lm.models.base import create_attention_mask

# Reuse infrastructure from existing scripts
from transcender_engine import build_harmony_messages
from transcender_track_b_cascade import (
    PROMPTS,
    WARMUP_PROMPT_INDEX,
    apply_generic_chat_template,
    mean,
)

SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class PerTokenTrace:
    position: int
    shallow_top1: int
    deep_top1: int
    agree: bool
    shallow_entropy: float
    shallow_margin: float
    shallow_top1_prob: float

    def to_dict(self) -> dict:
        return {
            "pos": self.position,
            "shallow_top1": self.shallow_top1,
            "deep_top1": self.deep_top1,
            "agree": self.agree,
            "entropy": round(self.shallow_entropy, 5),
            "margin": round(self.shallow_margin, 5),
            "shallow_top1_prob": round(self.shallow_top1_prob, 5),
        }


class DenseExitSweepEngine:
    """Lightweight engine for compute-both agreement profiling at multiple exits."""

    def __init__(self, model, tokenizer, family: str):
        self.model = model
        self.tokenizer = tokenizer
        self.family = family

        inner = model.model
        self.layers = inner.layers
        self.num_layers = len(inner.layers)
        self.embed_tokens = inner.embed_tokens
        self.final_norm = inner.norm
        self.tie_word_embeddings = bool(
            getattr(model.args, "tie_word_embeddings", True)
        )

        # Sliding-window support
        self.fa_idx = int(getattr(inner, "fa_idx", 0))
        self.swa_idx = getattr(inner, "swa_idx", None)
        self.sliding_window = getattr(inner, "sliding_window", None)

        # EOS detection
        self.eos_ids = set()
        eos_attr = getattr(tokenizer, "eos_token_ids", None)
        if eos_attr:
            self.eos_ids = set(int(x) for x in eos_attr)
        elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            self.eos_ids = {int(tokenizer.eos_token_id)}

    def _build_masks(self, hidden, cache):
        fa_cache = cache[self.fa_idx] if cache is not None else None
        fa_mask = create_attention_mask(hidden, fa_cache)
        swa_mask = None
        if self.swa_idx is not None and self.sliding_window is not None:
            swa_cache = cache[self.swa_idx] if cache is not None else None
            swa_mask = create_attention_mask(
                hidden, swa_cache, window_size=self.sliding_window
            )
        return fa_mask, swa_mask

    def _layer_mask(self, layer_idx, fa_mask, swa_mask):
        if getattr(self.layers[layer_idx], "use_sliding", False) and swa_mask is not None:
            return swa_mask
        return fa_mask

    def _run_layers(self, hidden, cache, start, end):
        fa_mask, swa_mask = self._build_masks(hidden, cache)
        for i in range(start, end):
            mask = self._layer_mask(i, fa_mask, swa_mask)
            hidden = self.layers[i](hidden, mask, cache[i] if cache is not None else None)
        mx.eval(hidden)
        return hidden

    def _compute_logits(self, hidden):
        normed = self.final_norm(hidden)
        if self.tie_word_embeddings:
            return self.embed_tokens.as_linear(normed)
        return self.model.lm_head(normed)

    def _confidence_metrics(self, logits):
        last = logits[0, -1].astype(mx.float32)
        probs = mx.softmax(last, axis=-1)
        entropy = -mx.sum(probs * mx.log(probs + 1e-9), axis=-1)
        norm_entropy = entropy / math.log(int(last.shape[-1]))
        top1_id = mx.argmax(last, axis=-1)
        top2 = mx.topk(probs, 2, axis=-1)
        top1_prob = top2[..., 0]
        top2_prob = top2[..., 1]
        margin = top1_prob - top2_prob
        mx.eval(norm_entropy, top1_id, top1_prob, margin)
        return {
            "top1_id": int(top1_id.item()),
            "top1_prob": float(top1_prob.item()),
            "entropy": float(norm_entropy.item()),
            "margin": float(margin.item()),
        }

    def sweep_exit_layer(
        self,
        prompt_ids: List[int],
        exit_layer: int,
        max_new_tokens: int = 48,
    ) -> Dict[str, Any]:
        """
        Run compute-both at a given exit layer.
        Returns per-token traces and aggregate metrics.
        """
        mx.clear_cache()
        mx.reset_peak_memory()

        cache = self.model.make_cache()
        input_ids = mx.array(prompt_ids, dtype=mx.int32).reshape(1, -1)

        t0 = time.perf_counter()

        # Full-depth prefill
        hidden = self.embed_tokens(input_ids)
        hidden = self._run_layers(hidden, cache, 0, self.num_layers)
        logits = self._compute_logits(hidden)
        mx.eval(logits)
        token_id = int(mx.argmax(logits[0, -1], axis=-1).item())

        generated_ids = [token_id]
        deep_generated_ids = [token_id]  # what full-depth would produce
        traces: List[PerTokenTrace] = []
        agree_count = 0
        total_decision = 0

        ttft = time.perf_counter() - t0

        if token_id not in self.eos_ids:
            for step in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                hidden = self.embed_tokens(token_input)

                # Run to exit layer
                hidden = self._run_layers(hidden, cache, 0, exit_layer + 1)
                early_logits = self._compute_logits(hidden)
                mx.eval(early_logits)
                early_metrics = self._confidence_metrics(early_logits)
                early_top1 = early_metrics["top1_id"]

                # Continue to full depth
                hidden = self._run_layers(hidden, cache, exit_layer + 1, self.num_layers)
                full_logits = self._compute_logits(hidden)
                mx.eval(full_logits)
                deep_top1 = int(mx.argmax(full_logits[0, -1], axis=-1).item())

                total_decision += 1
                agree = early_top1 == deep_top1
                if agree:
                    agree_count += 1

                traces.append(PerTokenTrace(
                    position=step + 1,
                    shallow_top1=early_top1,
                    deep_top1=deep_top1,
                    agree=agree,
                    shallow_entropy=early_metrics["entropy"],
                    shallow_margin=early_metrics["margin"],
                    shallow_top1_prob=early_metrics["top1_prob"],
                ))

                # Always follow deep path (compute-both)
                token_id = deep_top1
                generated_ids.append(token_id)
                deep_generated_ids.append(deep_top1)

                if token_id in self.eos_ids:
                    break

        elapsed = time.perf_counter() - t0
        agreement_rate = agree_count / max(total_decision, 1)
        skip_budget = (self.num_layers - exit_layer - 1) / self.num_layers
        # Upper bound only. Assumes zero replay overhead, perfect parallelism.
        # Not a prediction — real speedup will be strictly lower.
        agree_x_skip = agreement_rate * skip_budget
        upper_bound_speedup = 1.0 / (1.0 - agree_x_skip) if agree_x_skip < 1.0 else float("inf")

        # Per-entropy-threshold acceptance simulation (minimal set for go/no-go)
        entropy_thresholds = [0.10, 0.20, 0.50]
        threshold_analysis = []
        for thr in entropy_thresholds:
            accepted = sum(1 for t in traces if t.shallow_entropy <= thr)
            accepted_and_agree = sum(1 for t in traces if t.shallow_entropy <= thr and t.agree)
            accepted_and_disagree = sum(1 for t in traces if t.shallow_entropy <= thr and not t.agree)
            threshold_analysis.append({
                "entropy_threshold": thr,
                "would_accept": accepted,
                "of_which_agree": accepted_and_agree,
                "of_which_disagree": accepted_and_disagree,
                "accept_rate": accepted / max(total_decision, 1),
                "precision": accepted_and_agree / max(accepted, 1),
            })

        return {
            "exit_layer": exit_layer,
            "deep_layers_skipped": self.num_layers - exit_layer - 1,
            "skip_budget": round(skip_budget, 4),
            "generated_ids": generated_ids,
            "completion_tokens": len(generated_ids),
            "ttft_s": ttft,
            "elapsed_s": elapsed,
            "generation_tps": max(len(generated_ids) - 1, 0) / max(elapsed - ttft, 1e-6),
            "peak_memory_gb": mx.get_peak_memory() / (1024**3),
            "agreement_rate": round(agreement_rate, 4),
            "agree_count": agree_count,
            "total_decision_tokens": total_decision,
            "agree_x_skip": round(agree_x_skip, 4),
            "upper_bound_speedup": round(upper_bound_speedup, 3),
            "threshold_analysis": threshold_analysis,
            "per_token_trace": [t.to_dict() for t in traces],
        }


    def run_shallow_path(
        self,
        prompt_ids: List[int],
        exit_layer: int,
        max_new_tokens: int = 48,
    ) -> List[int]:
        """
        Generate tokens following the shallow exit only (no deep continuation).
        Used to measure what output you'd actually get if you trusted the shallow exit.
        Separate cache from sweep_exit_layer — intentionally independent run.
        """
        mx.clear_cache()
        cache = self.model.make_cache()
        input_ids = mx.array(prompt_ids, dtype=mx.int32).reshape(1, -1)

        # Full-depth prefill (same as sweep — prefill is always full depth)
        hidden = self.embed_tokens(input_ids)
        hidden = self._run_layers(hidden, cache, 0, self.num_layers)
        logits = self._compute_logits(hidden)
        mx.eval(logits)
        token_id = int(mx.argmax(logits[0, -1], axis=-1).item())

        ids = [token_id]
        if token_id not in self.eos_ids:
            for _ in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                hidden = self.embed_tokens(token_input)
                # Run to exit layer only — then use norm + lm_head
                hidden = self._run_layers(hidden, cache, 0, exit_layer + 1)
                logits = self._compute_logits(hidden)
                mx.eval(logits)
                token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
                ids.append(token_id)
                if token_id in self.eos_ids:
                    break
        return ids


def build_prompt_pack(tokenizer, family: str):
    pack = []
    for i, user_prompt in enumerate(PROMPTS, start=1):
        messages = build_harmony_messages(user_prompt, SYSTEM_PROMPT)
        try:
            prompt_text, _ = apply_generic_chat_template(tokenizer, messages)
        except Exception as exc:
            if "alternate" not in str(exc).lower():
                raise
            merged = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}]
            prompt_text, _ = apply_generic_chat_template(tokenizer, merged)
        prompt_ids = tokenizer.encode(prompt_text)
        pack.append({"prompt_id": f"P{i}", "user": user_prompt, "prompt_ids": prompt_ids})
    return pack


def run_full_depth_baseline(engine: DenseExitSweepEngine, prompt_pack, max_new_tokens: int):
    """Run full-depth baseline to get reference token sequences."""
    baseline = {}
    for pp in prompt_pack:
        mx.clear_cache()
        mx.reset_peak_memory()
        cache = engine.model.make_cache()
        input_ids = mx.array(pp["prompt_ids"], dtype=mx.int32).reshape(1, -1)
        hidden = engine.embed_tokens(input_ids)
        hidden = engine._run_layers(hidden, cache, 0, engine.num_layers)
        logits = engine._compute_logits(hidden)
        mx.eval(logits)
        token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
        ids = [token_id]
        for _ in range(max_new_tokens - 1):
            token_input = mx.array([[token_id]], dtype=mx.int32)
            hidden = engine.embed_tokens(token_input)
            hidden = engine._run_layers(hidden, cache, 0, engine.num_layers)
            logits = engine._compute_logits(hidden)
            mx.eval(logits)
            token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
            ids.append(token_id)
            if token_id in engine.eos_ids:
                break
        baseline[pp["prompt_id"]] = ids
    return baseline


def main():
    parser = argparse.ArgumentParser(description="Dense exit-layer agreement sweep")
    parser.add_argument("--model", required=True, help="Path to MLX model")
    parser.add_argument("--family", required=True, choices=["llama", "mistral", "gemma"])
    parser.add_argument(
        "--exit-layers",
        required=True,
        help="Comma-separated exit layer indices (e.g., 10,12,14,16,18,20)",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--csv-output", default=None, help="Optional CSV summary path")
    args = parser.parse_args()

    exit_layers = [int(x.strip()) for x in args.exit_layers.split(",")]

    print(f"Loading model: {args.model}")
    model, tokenizer = mlx_load(args.model)
    engine = DenseExitSweepEngine(model, tokenizer, args.family)
    print(f"Model loaded: {engine.num_layers} layers")

    # Validate exit layers
    for el in exit_layers:
        if el < 1 or el >= engine.num_layers - 1:
            print(f"WARNING: exit layer {el} out of valid range [1, {engine.num_layers - 2}], skipping")
            exit_layers = [x for x in exit_layers if x != el]

    prompt_pack = build_prompt_pack(tokenizer, args.family)

    # Full-depth baseline
    print("Running full-depth baseline...")
    baseline_ids = run_full_depth_baseline(engine, prompt_pack, args.max_new_tokens)

    results = {
        "experiment": "dense_exit_sweep",
        "model": args.model,
        "family": args.family,
        "num_layers": engine.num_layers,
        "max_new_tokens": args.max_new_tokens,
        "exit_layers_tested": exit_layers,
        "warmup_prompt_index": WARMUP_PROMPT_INDEX,
        "sweep_results": [],
    }

    summary_rows = []

    for exit_layer in exit_layers:
        print(f"\n--- Exit Layer {exit_layer} (skips {engine.num_layers - exit_layer - 1} layers) ---")
        layer_results = {
            "exit_layer": exit_layer,
            "skip_budget": round((engine.num_layers - exit_layer - 1) / engine.num_layers, 4),
            "prompt_results": [],
        }

        all_agreement_rates = []
        all_shallow_exact_matches = []
        all_tps = []

        for pp in prompt_pack:
            is_warmup = pp["prompt_id"] == f"P{WARMUP_PROMPT_INDEX + 1}"
            print(f"  {pp['prompt_id']}: {pp['user'][:50]}...", end=" ", flush=True)

            result = engine.sweep_exit_layer(
                pp["prompt_ids"], exit_layer, args.max_new_tokens
            )

            # Shallow-path generation: what you'd actually get if you trusted
            # the shallow exit at every step. Separate run, own cache.
            shallow_ids = engine.run_shallow_path(
                pp["prompt_ids"], exit_layer, args.max_new_tokens
            )
            ref_ids = baseline_ids[pp["prompt_id"]]
            shallow_path_exact_match = 1.0 if shallow_ids == ref_ids else 0.0
            result["shallow_path_exact_match"] = shallow_path_exact_match
            result["is_warmup"] = is_warmup
            result["prompt_id"] = pp["prompt_id"]

            print(
                f"agree={result['agreement_rate']:.3f} "
                f"shallow_exact={shallow_path_exact_match:.3f} "
                f"tps={result['generation_tps']:.1f}"
            )

            layer_results["prompt_results"].append(result)

            if not is_warmup:
                all_agreement_rates.append(result["agreement_rate"])
                all_shallow_exact_matches.append(shallow_path_exact_match)
                all_tps.append(result["generation_tps"])

        # Aggregate excluding warmup
        avg_agree = mean(all_agreement_rates) if all_agreement_rates else 0.0
        min_agree = min(all_agreement_rates) if all_agreement_rates else 0.0
        axs = avg_agree * layer_results["skip_budget"] if all_agreement_rates else 0.0
        agg = {
            "avg_agreement_rate": round(avg_agree, 4),
            "min_agreement_rate": round(min_agree, 4),
            "avg_shallow_path_exact_match": round(mean(all_shallow_exact_matches), 4) if all_shallow_exact_matches else 0.0,
            "avg_tps": round(mean(all_tps), 2) if all_tps else 0.0,
            "skip_budget": layer_results["skip_budget"],
            "agree_x_skip": round(axs, 4),
            "upper_bound_speedup": round(
                1.0 / (1.0 - axs) if axs < 1.0 else float("inf"),
                3,
            ),
        }

        # Threshold analysis aggregate
        if all_agreement_rates:
            non_warmup = [r for r in layer_results["prompt_results"] if not r["is_warmup"]]
            if non_warmup and non_warmup[0]["threshold_analysis"]:
                agg_thresholds = []
                for t_idx in range(len(non_warmup[0]["threshold_analysis"])):
                    thr = non_warmup[0]["threshold_analysis"][t_idx]["entropy_threshold"]
                    avg_accept = mean([r["threshold_analysis"][t_idx]["accept_rate"] for r in non_warmup])
                    avg_precision = mean([r["threshold_analysis"][t_idx]["precision"] for r in non_warmup])
                    agg_thresholds.append({
                        "entropy_threshold": thr,
                        "avg_accept_rate": round(avg_accept, 4),
                        "avg_precision": round(avg_precision, 4),
                    })
                agg["threshold_analysis"] = agg_thresholds

        layer_results["aggregate_excluding_warmup"] = agg
        results["sweep_results"].append(layer_results)

        summary_rows.append({
            "exit_layer": exit_layer,
            "layers_skipped": engine.num_layers - exit_layer - 1,
            "skip_budget": layer_results["skip_budget"],
            "avg_agreement": agg["avg_agreement_rate"],
            "min_agreement": agg["min_agreement_rate"],
            "shallow_exact": agg["avg_shallow_path_exact_match"],
            "agree_x_skip": agg["agree_x_skip"],
            "upper_bound": agg["upper_bound_speedup"],
        })

        print(f"  AGGREGATE: agree={agg['avg_agreement_rate']:.3f} "
              f"min_agree={agg['min_agreement_rate']:.3f} "
              f"shallow_exact={agg['avg_shallow_path_exact_match']:.3f} "
              f"a×s={agg['agree_x_skip']:.3f} "
              f"ub={agg['upper_bound_speedup']:.2f}x")

    # Write JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON written to {output_path}")

    # Write CSV summary
    csv_path = args.csv_output or str(output_path).replace(".json", "_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"CSV summary written to {csv_path}")

    # Print final summary table
    print("\n" + "=" * 80)
    print("EXIT LAYER AGREEMENT SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'Exit':>6} {'Skip':>6} {'Agree':>8} {'MinAg':>8} {'ShExact':>8} {'A×S':>8} {'UB':>8}")
    print("-" * 62)
    for row in summary_rows:
        print(
            f"L{row['exit_layer']:>4} "
            f"{row['skip_budget']:>5.1%} "
            f"{row['avg_agreement']:>7.1%} "
            f"{row['min_agreement']:>7.1%} "
            f"{row['shallow_exact']:>7.1%} "
            f"{row['agree_x_skip']:>7.3f} "
            f"{row['upper_bound']:>6.2f}x"
        )
    print("=" * 80)

    # Decision guidance
    best = max(summary_rows, key=lambda r: r["agree_x_skip"])
    axs = best["agree_x_skip"]
    print(f"\nBest agree×skip: L{best['exit_layer']} "
          f"(avg={best['avg_agreement']:.1%} min={best['min_agreement']:.1%} "
          f"× {best['skip_budget']:.1%} skip = {axs:.3f})")

    if axs >= 0.25 and best["min_agreement"] >= 0.70 and best["shallow_exact"] >= 0.60:
        print("GREEN: Dense frontier likely recoverable. Proceed to self-speculative prototype.")
    elif axs >= 0.15 and best["min_agreement"] >= 0.55:
        print("YELLOW: Marginal. Report agreement profile in paper but do not invest in implementation.")
    else:
        print("RED: No viable dense frontier. Publish without dense speed result.")


if __name__ == "__main__":
    main()
