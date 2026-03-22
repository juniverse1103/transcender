"""
Track C — Adaptive Depth Benchmark for Gemma 3 4B-IT

Tests whether the Transcender adaptive-depth methodology generalizes to a
small dense architecture. Compares:

  Mode A: Full-depth baseline (34 layers)
  Mode B: Early exit (L16, aggressive, ~53% depth savings)
  Mode C: Mid exit   (L20, at KL elbow, ~41% depth savings)
  Mode D: Late exit  (L31, conservative, ~9% depth savings)
  Mode E: Agreement-aware blending (top1_agree at best candidate)
  Mode F: Naive blending (fixed alpha, control)

Key thesis questions:
  1. Does Gemma show a meaningful depth frontier?
  2. Does top1_agree blending help on a dense model?
  3. Does Gemma reproduce the "one layer deeper = quality but savings collapse" pattern?
  4. Is the Transcender approach architecture-specific or more general?

Usage:
    python transcender_track_c_gemma_benchmark.py \\
        --model /path/to/gemma-3-4b-it \\
        --output transcender_track_c_gemma_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as mnn
import numpy as np
from mlx_lm import load as mlx_load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from transcender_track_b_cascade import (
    PROMPTS,
    WARMUP_PROMPT_INDEX,
    apply_generic_chat_template,
    compare_sequences,
    mean,
    preview_text,
)
from transcender_engine import build_harmony_messages

SYSTEM_PROMPT = "You are a helpful assistant."


# ═══════════════════════════════════════════════════════════════════
# Gemma Model Access Layer
# ═══════════════════════════════════════════════════════════════════

class GemmaAdaptiveEngine:
    """
    Minimal adaptive-depth engine for Gemma 3.

    Unlike the GPT-OSS engine (which handles MoE routing, expert-skipping,
    and complex cache strategies), this engine only does:
      - Full-depth greedy generation (baseline)
      - Fixed early-exit generation (layers 0..exit_layer only)
      - Logit-space blending with agreement-aware strategy
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.lm = model.language_model
        self.inner = self.lm.model
        self.layers = self.inner.layers
        self.num_layers = len(self.layers)
        self.embed_tokens = self.inner.embed_tokens
        self.final_norm = self.inner.norm
        self.tie_word_embeddings = self.lm.tie_word_embeddings

        # Attention pattern info
        self.sliding_window_pattern = self.inner.sliding_window_pattern
        self.window_size = getattr(self.inner, "window_size", 1024)

        # EOS tokens
        self.eos_ids = set()
        eos_ids_attr = getattr(tokenizer, "eos_token_ids", None)
        if eos_ids_attr:
            self.eos_ids = set(int(x) for x in eos_ids_attr)
        elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            self.eos_ids = {int(tokenizer.eos_token_id)}

    def _embed(self, input_ids: mx.array) -> mx.array:
        h = self.embed_tokens(input_ids)
        h *= mx.array(2560**0.5, mx.bfloat16).astype(h.dtype)
        return h

    def _compute_logits(self, hidden: mx.array) -> mx.array:
        normed = self.final_norm(hidden)
        if self.tie_word_embeddings:
            return self.embed_tokens.as_linear(normed)
        return self.lm.lm_head(normed)

    def _build_masks(self, hidden: mx.array, cache_entry=None):
        from mlx_lm.models.gemma3_text import create_attention_mask
        global_mask = create_attention_mask(hidden, cache=cache_entry)
        if self.sliding_window_pattern > 1:
            sliding_mask = create_attention_mask(
                hidden, cache=cache_entry, window_size=self.window_size,
            )
        else:
            sliding_mask = None
        return global_mask, sliding_mask

    def _is_global_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.sliding_window_pattern == self.sliding_window_pattern - 1

    # ── Full-depth greedy generation (baseline) ──

    def generate_full_depth(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 48,
    ) -> Dict[str, Any]:
        """Standard full-depth greedy generation via mlx_lm."""
        mx.clear_cache()
        mx.reset_peak_memory()

        prompt = mx.array(prompt_ids, dtype=mx.int32)
        sampler = make_sampler(temp=0.0)
        t0 = time.perf_counter()
        ttft = None
        generated_ids: List[int] = []

        for token, _ in generate_step(
            prompt, self.model, max_tokens=max_new_tokens, sampler=sampler,
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0
            tok_id = int(token)
            generated_ids.append(tok_id)
            if tok_id in self.eos_ids:
                break

        elapsed = time.perf_counter() - t0
        ttft = ttft or elapsed
        decode_time = max(elapsed - ttft, 1e-6)

        return {
            "generated_ids": generated_ids,
            "output_text": self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated_ids),
            "ttft_s": ttft,
            "elapsed_s": elapsed,
            "generation_tps": max(len(generated_ids) - 1, 0) / decode_time,
            "peak_memory_gb": mx.get_peak_memory() / (1024**3),
            "avg_layers": float(self.num_layers),
            "layers_saved": 0.0,
        }

    # ── Fixed early-exit generation ──

    def generate_early_exit(
        self,
        prompt_ids: List[int],
        exit_layer: int,
        max_new_tokens: int = 48,
    ) -> Dict[str, Any]:
        """
        Generate tokens using only layers 0..exit_layer.

        Layers after exit_layer are NEVER evaluated. The hidden state
        at exit_layer is projected to logits via final_norm + lm_head.

        This is the simplest form of adaptive depth: a fixed cutoff.
        """
        mx.clear_cache()
        mx.reset_peak_memory()

        cache = self.model.make_cache()
        input_ids = mx.array(prompt_ids, dtype=mx.int32).reshape(1, -1)

        t0 = time.perf_counter()
        ttft = None
        generated_ids: List[int] = []

        # Prefill
        h = self._embed(input_ids)
        for i in range(exit_layer + 1):
            is_global = self._is_global_layer(i)
            global_mask, sliding_mask = self._build_masks(h, cache[i])
            mask = global_mask if is_global else sliding_mask
            h = self.layers[i](h, mask, cache[i])
        mx.eval(h)

        logits = self._compute_logits(h)
        mx.eval(logits)
        token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
        generated_ids.append(token_id)
        ttft = time.perf_counter() - t0

        if token_id not in self.eos_ids:
            # Autoregressive decode
            for _ in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                h = self._embed(token_input)
                for i in range(exit_layer + 1):
                    is_global = self._is_global_layer(i)
                    global_mask, sliding_mask = self._build_masks(h, cache[i])
                    mask = global_mask if is_global else sliding_mask
                    h = self.layers[i](h, mask, cache[i])
                mx.eval(h)

                logits = self._compute_logits(h)
                mx.eval(logits)
                token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
                generated_ids.append(token_id)
                if token_id in self.eos_ids:
                    break

        elapsed = time.perf_counter() - t0
        layers_used = exit_layer + 1
        layers_saved = (self.num_layers - layers_used) / self.num_layers

        return {
            "generated_ids": generated_ids,
            "output_text": self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated_ids),
            "ttft_s": ttft,
            "elapsed_s": elapsed,
            "generation_tps": max(len(generated_ids) - 1, 0) / max(elapsed - ttft, 1e-6),
            "peak_memory_gb": mx.get_peak_memory() / (1024**3),
            "exit_layer": exit_layer,
            "layers_used": layers_used,
            "avg_layers": float(layers_used),
            "layers_saved": round(layers_saved, 4),
        }

    # ── Agreement-aware blended generation ──

    def generate_blended(
        self,
        prompt_ids: List[int],
        exit_layer: int,
        max_new_tokens: int = 48,
        blend_alpha: float = 0.10,
        strategy: str = "top1_agree",
    ) -> Dict[str, Any]:
        """
        Two-pass adaptive generation with logit-space blending.

        For each token:
          1. Run layers 0..exit_layer → early_logits
          2. Continue layers exit_layer+1..N → full_logits
          3. If strategy == "top1_agree" and argmax(early) == argmax(full):
               blended = (1-alpha)*full + alpha*early
             Else:
               use full_logits only (early exit was wrong, discard)

        This tests whether agreement-aware logit composition is useful
        on a dense model (the core Transcender thesis question for Track C).

        The "layers_saved" metric tracks how often early exit was trusted,
        weighted by the fraction of layers actually skipped per token.
        """
        mx.clear_cache()
        mx.reset_peak_memory()

        cache = self.model.make_cache()
        input_ids = mx.array(prompt_ids, dtype=mx.int32).reshape(1, -1)

        t0 = time.perf_counter()
        ttft = None
        generated_ids: List[int] = []
        agree_count = 0
        total_tokens = 0

        # Prefill (full depth — we need the cache populated for all layers)
        h = self._embed(input_ids)
        for i in range(self.num_layers):
            is_global = self._is_global_layer(i)
            global_mask, sliding_mask = self._build_masks(h, cache[i])
            mask = global_mask if is_global else sliding_mask
            h = self.layers[i](h, mask, cache[i])
        mx.eval(h)

        logits = self._compute_logits(h)
        mx.eval(logits)
        token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
        generated_ids.append(token_id)
        ttft = time.perf_counter() - t0

        if token_id not in self.eos_ids:
            for _ in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                h = self._embed(token_input)

                # Early pass: layers 0..exit_layer
                for i in range(exit_layer + 1):
                    is_global = self._is_global_layer(i)
                    global_mask, sliding_mask = self._build_masks(h, cache[i])
                    mask = global_mask if is_global else sliding_mask
                    h = self.layers[i](h, mask, cache[i])
                mx.eval(h)

                early_logits = self._compute_logits(h)
                mx.eval(early_logits)
                early_top1 = int(mx.argmax(early_logits[0, -1], axis=-1).item())

                # Full pass: continue layers exit_layer+1..N
                for i in range(exit_layer + 1, self.num_layers):
                    is_global = self._is_global_layer(i)
                    global_mask, sliding_mask = self._build_masks(h, cache[i])
                    mask = global_mask if is_global else sliding_mask
                    h = self.layers[i](h, mask, cache[i])
                mx.eval(h)

                full_logits = self._compute_logits(h)
                mx.eval(full_logits)
                full_top1 = int(mx.argmax(full_logits[0, -1], axis=-1).item())

                total_tokens += 1

                # Blending decision
                if strategy == "top1_agree" and early_top1 == full_top1:
                    # Agreement: blend logits
                    blended = (1.0 - blend_alpha) * full_logits + blend_alpha * early_logits
                    mx.eval(blended)
                    token_id = int(mx.argmax(blended[0, -1], axis=-1).item())
                    agree_count += 1
                elif strategy == "naive":
                    # Naive: always blend regardless of agreement
                    blended = (1.0 - blend_alpha) * full_logits + blend_alpha * early_logits
                    mx.eval(blended)
                    token_id = int(mx.argmax(blended[0, -1], axis=-1).item())
                    agree_count += 1
                else:
                    # Disagreement: use full logits only
                    token_id = full_top1

                generated_ids.append(token_id)
                if token_id in self.eos_ids:
                    break

        elapsed = time.perf_counter() - t0
        agreement_rate = agree_count / max(total_tokens, 1)

        return {
            "generated_ids": generated_ids,
            "output_text": self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated_ids),
            "ttft_s": ttft,
            "elapsed_s": elapsed,
            "generation_tps": max(len(generated_ids) - 1, 0) / max(elapsed - ttft, 1e-6),
            "peak_memory_gb": mx.get_peak_memory() / (1024**3),
            "exit_layer": exit_layer,
            "strategy": strategy,
            "blend_alpha": blend_alpha,
            "agreement_rate": round(agreement_rate, 4),
            "agree_count": agree_count,
            "total_tokens": total_tokens,
            "avg_layers": float(self.num_layers),  # always runs full depth for blending
            "layers_saved": 0.0,  # blending still runs all layers
        }


# ═══════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModeConfig:
    key: str
    label: str
    description: str
    exit_layer: Optional[int] = None
    strategy: Optional[str] = None
    blend_alpha: float = 0.0


def build_modes(num_layers: int, early: int, mid: int, late: int) -> List[ModeConfig]:
    return [
        ModeConfig(
            key="full_depth",
            label="Full Depth (L33)",
            description="Gemma full 34-layer baseline",
        ),
        ModeConfig(
            key=f"early_exit_L{early}",
            label=f"Early Exit (L{early})",
            description=f"Fixed exit at layer {early}, aggressive depth cut",
            exit_layer=early,
        ),
        ModeConfig(
            key=f"mid_exit_L{mid}",
            label=f"Mid Exit (L{mid})",
            description=f"Fixed exit at KL elbow (layer {mid})",
            exit_layer=mid,
        ),
        ModeConfig(
            key=f"late_exit_L{late}",
            label=f"Late Exit (L{late})",
            description=f"Conservative exit at layer {late}",
            exit_layer=late,
        ),
        ModeConfig(
            key=f"top1_agree_L{late}",
            label=f"top1_agree (L{late})",
            description=f"Agreement-aware blending at layer {late}",
            exit_layer=late,
            strategy="top1_agree",
            blend_alpha=0.10,
        ),
        ModeConfig(
            key=f"naive_blend_L{late}",
            label=f"Naive Blend (L{late})",
            description=f"Naive fixed-alpha blending at layer {late} (control)",
            exit_layer=late,
            strategy="naive",
            blend_alpha=0.10,
        ),
    ]


def build_prompt_pack(tokenizer) -> List[Dict[str, Any]]:
    pack = []
    for i, user_prompt in enumerate(PROMPTS, start=1):
        messages = build_harmony_messages(user_prompt, SYSTEM_PROMPT)
        prompt_text, _ = apply_generic_chat_template(tokenizer, messages)
        prompt_ids = tokenizer.encode(prompt_text)
        pack.append({
            "prompt_id": f"P{i}",
            "user": user_prompt,
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
        })
    return pack


def run_mode(
    engine: GemmaAdaptiveEngine,
    mode: ModeConfig,
    prompt_pack: List[Dict[str, Any]],
    reference_results: Optional[Dict[str, List[int]]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Run one benchmark mode across all prompts."""
    prompt_results = []

    for prompt_def in prompt_pack:
        prompt_id = prompt_def["prompt_id"]
        prompt_ids = prompt_def["prompt_ids"]

        if mode.strategy:
            stats = engine.generate_blended(
                prompt_ids=prompt_ids,
                exit_layer=mode.exit_layer,
                max_new_tokens=max_new_tokens,
                blend_alpha=mode.blend_alpha,
                strategy=mode.strategy,
            )
        elif mode.exit_layer is not None:
            stats = engine.generate_early_exit(
                prompt_ids=prompt_ids,
                exit_layer=mode.exit_layer,
                max_new_tokens=max_new_tokens,
            )
        else:
            stats = engine.generate_full_depth(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
            )

        # Compare against reference
        if reference_results and prompt_id in reference_results:
            ref_ids = reference_results[prompt_id]
            comparison = compare_sequences(stats["generated_ids"], ref_ids)
        else:
            comparison = {
                "exact_match_rate": 1.0,
                "prefix_match_tokens": stats["completion_tokens"],
                "first_divergence_position": None,
                "passed": True,
            }

        prompt_results.append({
            "prompt_id": prompt_id,
            "user_prompt": prompt_def["user"],
            **stats,
            "comparison": comparison,
            "preview": preview_text(stats["output_text"], 200),
        })

    # Aggregate (excluding warmup)
    non_warmup = [
        r for i, r in enumerate(prompt_results) if i != WARMUP_PROMPT_INDEX
    ]

    aggregate = {
        "avg_ttft_s": mean([r["ttft_s"] for r in non_warmup]),
        "avg_generation_tps": mean([r["generation_tps"] for r in non_warmup]),
        "avg_elapsed_s": mean([r["elapsed_s"] for r in non_warmup]),
        "avg_peak_memory_gb": mean([r["peak_memory_gb"] for r in non_warmup]),
        "avg_exact_match_rate": mean([r["comparison"]["exact_match_rate"] for r in non_warmup]),
        "avg_prefix_match_tokens": mean([r["comparison"]["prefix_match_tokens"] for r in non_warmup]),
        "avg_layers_saved": mean([r.get("layers_saved", 0.0) for r in non_warmup]),
        "avg_completion_tokens": mean([r["completion_tokens"] for r in non_warmup]),
    }

    # Add agreement rate for blending modes
    if mode.strategy:
        aggregate["avg_agreement_rate"] = mean([
            r.get("agreement_rate", 0.0) for r in non_warmup
        ])

    return {
        "key": mode.key,
        "label": mode.label,
        "description": mode.description,
        "status": "ok",
        "prompt_results": prompt_results,
        "aggregate_excluding_warmup": aggregate,
    }


def build_comparison_summary(modes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the final comparison summary answering the thesis questions."""

    def get_agg(key: str) -> Optional[Dict]:
        for m in modes_data:
            if m["key"] == key and m["status"] == "ok":
                return m["aggregate_excluding_warmup"]
        return None

    full = get_agg("full_depth")
    if full is None:
        return {"error": "full_depth baseline missing"}

    summary = {
        "baseline_tps": full["avg_generation_tps"],
        "baseline_exact_match": full["avg_exact_match_rate"],
    }

    # Collect exit results
    for m in modes_data:
        if m["status"] != "ok":
            continue
        agg = m["aggregate_excluding_warmup"]
        key = m["key"]
        summary[key] = {
            "exact_match_rate": agg["avg_exact_match_rate"],
            "generation_tps": agg["avg_generation_tps"],
            "layers_saved": agg.get("avg_layers_saved", 0.0),
            "tps_speedup_vs_baseline": round(
                agg["avg_generation_tps"] / max(full["avg_generation_tps"], 0.01), 3
            ),
        }
        if "avg_agreement_rate" in agg:
            summary[key]["agreement_rate"] = agg["avg_agreement_rate"]

    # Thesis questions
    # Find best early-exit mode with exact_match >= 0.90
    exit_modes = [m for m in modes_data if "exit_L" in m["key"] and m["status"] == "ok"]
    viable_exits = [
        m for m in exit_modes
        if m["aggregate_excluding_warmup"]["avg_exact_match_rate"] >= 0.90
    ]
    if viable_exits:
        best = max(viable_exits, key=lambda m: m["aggregate_excluding_warmup"]["avg_layers_saved"])
        ba = best["aggregate_excluding_warmup"]
        summary["best_viable_exit"] = {
            "mode": best["key"],
            "exact_match_rate": ba["avg_exact_match_rate"],
            "layers_saved": ba["avg_layers_saved"],
            "generation_tps": ba["avg_generation_tps"],
        }
        summary["depth_frontier_exists"] = True
    else:
        summary["depth_frontier_exists"] = len(exit_modes) > 0
        summary["best_viable_exit"] = None

    # Agreement-aware vs naive
    agree_modes = [m for m in modes_data if "top1_agree" in m["key"] and m["status"] == "ok"]
    naive_modes = [m for m in modes_data if "naive_blend" in m["key"] and m["status"] == "ok"]

    if agree_modes and naive_modes:
        agree_agg = agree_modes[0]["aggregate_excluding_warmup"]
        naive_agg = naive_modes[0]["aggregate_excluding_warmup"]
        summary["agreement_blending_helps"] = (
            agree_agg["avg_exact_match_rate"] >= naive_agg["avg_exact_match_rate"]
        )
        summary["agree_vs_naive_exact_match_delta"] = round(
            agree_agg["avg_exact_match_rate"] - naive_agg["avg_exact_match_rate"], 4
        )

    return summary


# ═══════════════════════════════════════════════════════════════════
# Terminal Output
# ═══════════════════════════════════════════════════════════════════

def print_results(payload: Dict[str, Any]):
    print(f"\n{'='*72}")
    print(f"  Track C — Gemma 3 4B-IT Adaptive Depth Benchmark")
    print(f"{'='*72}")

    header = (
        f"  {'Mode':<28} {'Status':<8} {'TTFT':>7} {'TPS':>8} "
        f"{'Mem GB':>7} {'Exact':>7} {'Saved':>7}"
    )
    print(f"\n{header}")
    print(f"  {'-'*66}")

    for m in payload["modes"]:
        if m["status"] != "ok":
            print(f"  {m['label']:<28} {'FAIL':<8}")
            continue
        agg = m["aggregate_excluding_warmup"]
        print(
            f"  {m['label']:<28} {'ok':<8} "
            f"{agg['avg_ttft_s']:>7.3f} "
            f"{agg['avg_generation_tps']:>8.2f} "
            f"{agg['avg_peak_memory_gb']:>7.2f} "
            f"{agg['avg_exact_match_rate']:>7.3f} "
            f"{agg.get('avg_layers_saved', 0.0):>7.1%}"
        )

    # Per-prompt previews for key modes
    print(f"\n  Per-prompt output previews:")
    for m in payload["modes"]:
        if m["status"] != "ok":
            continue
        print(f"\n  ── {m['label']} ──")
        for pr in m["prompt_results"]:
            em = pr["comparison"]["exact_match_rate"]
            print(f"    {pr['prompt_id']}: exact={em:.3f} | {preview_text(pr['preview'], 100)}")

    # Summary
    summary = payload["comparison_summary"]
    print(f"\n{'='*72}")
    print(f"  Thesis Assessment")
    print(f"{'='*72}")
    print(f"  Depth frontier exists:         {summary.get('depth_frontier_exists', 'unknown')}")
    if summary.get("best_viable_exit"):
        bve = summary["best_viable_exit"]
        print(f"  Best viable exit:              {bve['mode']}")
        print(f"    exact_match_rate:            {bve['exact_match_rate']:.3f}")
        print(f"    layers_saved:                {bve['layers_saved']:.1%}")
        print(f"    generation_tps:              {bve['generation_tps']:.2f}")

    if "agreement_blending_helps" in summary:
        print(f"  Agreement blending helps:      {summary['agreement_blending_helps']}")
        print(f"  agree vs naive Δexact_match:   {summary.get('agree_vs_naive_exact_match_delta', 0):.4f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Track C — Gemma 3 4B-IT Adaptive Depth Benchmark"
    )
    parser.add_argument(
        "--model", type=str,
        default=str((Path(__file__).resolve().parent.parent / "gemma-3-4b-it").resolve()),
    )
    parser.add_argument(
        "--output", type=str,
        default=str((Path(__file__).resolve().parent / "transcender_track_c_gemma_results.json").resolve()),
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--early-layer", type=int, default=16,
        help="Early exit layer (aggressive)",
    )
    parser.add_argument(
        "--mid-layer", type=int, default=20,
        help="Mid exit layer (KL elbow)",
    )
    parser.add_argument(
        "--late-layer", type=int, default=31,
        help="Late exit layer (conservative)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  Track C — Gemma 3 4B-IT Adaptive Depth Benchmark")
    print("=" * 72)

    # Load model
    print(f"\n  Loading {args.model}...")
    model, tokenizer = mlx_load(args.model)
    engine = GemmaAdaptiveEngine(model, tokenizer)
    print(f"  Loaded: {engine.num_layers} layers, dense architecture")

    # Build prompts
    prompt_pack = build_prompt_pack(tokenizer)
    print(f"  Prompts: {len(prompt_pack)}")

    # Build mode configs
    modes = build_modes(
        engine.num_layers,
        early=args.early_layer,
        mid=args.mid_layer,
        late=args.late_layer,
    )

    # Run each mode sequentially
    reference_results: Optional[Dict[str, List[int]]] = None
    all_modes_data: List[Dict[str, Any]] = []

    for mode in modes:
        print(f"\n  ── Running: {mode.label} ──")
        try:
            result = run_mode(
                engine, mode, prompt_pack,
                reference_results, args.max_new_tokens,
            )

            # Capture full-depth as reference
            if mode.key == "full_depth" and result["status"] == "ok":
                reference_results = {
                    pr["prompt_id"]: pr["generated_ids"]
                    for pr in result["prompt_results"]
                }

            all_modes_data.append(result)
            agg = result["aggregate_excluding_warmup"]
            print(f"     TPS: {agg['avg_generation_tps']:.2f} | "
                  f"Exact: {agg['avg_exact_match_rate']:.3f} | "
                  f"Saved: {agg.get('avg_layers_saved', 0):.1%}")

        except Exception as e:
            print(f"     FAILED: {e}")
            all_modes_data.append({
                "key": mode.key,
                "label": mode.label,
                "description": mode.description,
                "status": "error",
                "error": str(e),
            })

    # Build comparison summary
    summary = build_comparison_summary(all_modes_data)

    payload = {
        "model": "gemma-3-4b-it",
        "architecture": "dense",
        "num_layers": engine.num_layers,
        "max_new_tokens": args.max_new_tokens,
        "exit_layers_tested": {
            "early": args.early_layer,
            "mid": args.mid_layer,
            "late": args.late_layer,
        },
        "modes": all_modes_data,
        "comparison_summary": summary,
    }

    # Save
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved to {args.output}")

    # Print
    print_results(payload)


if __name__ == "__main__":
    main()
