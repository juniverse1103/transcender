"""
Track C — KL Divergence Profiling for Gemma 3 4B-IT

Methodology-validation track: tests whether the Transcender depth-frontier
observations (KL elbow, subspace paradox, agreement-aware blending) are
reproducible on a small dense architecture.

Architecture (Gemma 3 4B-IT):
  - 34 dense transformer layers (no MoE)
  - hidden_size 2560, intermediate 10240
  - Alternating sliding-window / global attention
  - Tied word embeddings (lm_head = embed_tokens.as_linear)

Output:
  - gemma_kl_profile.png  — per-layer KL divergence curve
  - gemma_kl_profile.json — machine-readable profiling artifact

Usage:
    python transcender_track_c_gemma_profile.py \\
        --model /path/to/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as mnn
import numpy as np
from mlx_lm import load as mlx_load

# Reuse the canonical prompt suite from Track B for comparability.
SYSTEM_PROMPT = "You are a helpful assistant."
PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "Summarize why the French Revolution was historically important.",
    "Write a short explanation of recursion for a beginner programmer.",
    "Explain the difference between TCP and UDP in plain English.",
    "Describe what photosynthesis does.",
]


# ═══════════════════════════════════════════════════════════════════
# Numerical helpers (numpy, backend-agnostic)
# ═══════════════════════════════════════════════════════════════════

def softmax_np(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax_np(x, axis=-1):
    shift = np.max(x, axis=axis, keepdims=True)
    return x - shift - np.log(np.sum(np.exp(x - shift), axis=axis, keepdims=True))


def kl_divergence_np(deep_logits, early_logits):
    """KL(Deep || Early) per token. Returns shape (B, S)."""
    deep_probs = softmax_np(deep_logits)
    deep_log = log_softmax_np(deep_logits)
    early_log = log_softmax_np(early_logits)
    return np.sum(deep_probs * (deep_log - early_log), axis=-1)


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, mx.array):
        return np.array(x.astype(mx.float32))
    return np.array(x)


# ═══════════════════════════════════════════════════════════════════
# Gemma Model Accessor
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GemmaModelParts:
    """Provides uniform access to Gemma 3 model internals."""
    model: Any  # top-level mlx_lm Model (gemma3.Model)
    tokenizer: Any
    num_layers: int
    layers: Any  # list of TransformerBlock
    embed_tokens: Any
    final_norm: Any
    tie_word_embeddings: bool

    @classmethod
    def from_loaded(cls, model, tokenizer) -> "GemmaModelParts":
        lm = model.language_model
        inner = lm.model
        return cls(
            model=model,
            tokenizer=tokenizer,
            num_layers=len(inner.layers),
            layers=inner.layers,
            embed_tokens=inner.embed_tokens,
            final_norm=inner.norm,
            tie_word_embeddings=lm.tie_word_embeddings,
        )

    def embed(self, input_ids: mx.array) -> mx.array:
        """Embed + scale (Gemma-specific sqrt(hidden_size) scaling)."""
        h = self.embed_tokens(input_ids)
        h *= mx.array(2560**0.5, mx.bfloat16).astype(h.dtype)
        return h

    def compute_logits(self, hidden: mx.array) -> mx.array:
        """Project hidden states -> logits through norm + lm_head."""
        normed = self.final_norm(hidden)
        if self.tie_word_embeddings:
            return self.embed_tokens.as_linear(normed)
        # Fallback for non-tied models (unlikely for Gemma 3 4B)
        return self.model.language_model.lm_head(normed)


# ═══════════════════════════════════════════════════════════════════
# Gemma-specific mask construction
# ═══════════════════════════════════════════════════════════════════

def build_gemma_masks(parts: GemmaModelParts, seq_len: int, hidden: mx.array):
    """
    Build the global and sliding-window masks that Gemma3Model.__call__
    normally constructs internally. We need these to run layers individually.
    """
    from mlx_lm.models.gemma3_text import create_attention_mask

    inner = parts.model.language_model.model
    sliding_window_pattern = inner.sliding_window_pattern

    # Global mask (full causal)
    global_mask = create_attention_mask(hidden, cache=None)

    # Sliding-window mask
    if sliding_window_pattern > 1:
        sliding_window_mask = create_attention_mask(
            hidden, cache=None, window_size=inner.window_size,
        )
    else:
        sliding_window_mask = None

    return global_mask, sliding_window_mask, sliding_window_pattern


# ═══════════════════════════════════════════════════════════════════
# KL Profiler
# ═══════════════════════════════════════════════════════════════════

def profile_gemma(
    parts: GemmaModelParts,
    prompt_text: str,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Per-layer KL divergence profiling for Gemma.

    For each layer i in [0, num_layers-1], computes:
      KL(full_depth_logits || layer_i_logits)

    This measures information lost by exiting at layer i.
    """
    # Tokenize
    token_ids = parts.tokenizer.encode(prompt_text)[:max_tokens]
    input_ids = mx.array(token_ids).reshape(1, -1)
    S = len(token_ids)

    print(f"\n  Profiling {S} tokens across {parts.num_layers} layers...")

    # Embed
    hidden = parts.embed(input_ids)

    # Build masks
    global_mask, sliding_mask, sw_pattern = build_gemma_masks(parts, S, hidden)

    # Forward pass: collect hidden states at every layer exit
    t0 = time.perf_counter()
    layer_logits_np = []

    h = hidden
    for i in range(parts.num_layers):
        is_global = (i % sw_pattern == sw_pattern - 1)
        mask = global_mask if is_global else sliding_mask
        h = parts.layers[i](h, mask, cache=None)
        mx.eval(h)

        # Project to logit space at this layer
        logits_i = parts.compute_logits(h)
        mx.eval(logits_i)
        layer_logits_np.append(to_numpy(logits_i))

    t_forward = time.perf_counter() - t0
    print(f"  Forward pass: {t_forward:.1f}s ({t_forward/parts.num_layers:.2f}s/layer)")

    # Deep (full-depth) logits = last layer
    deep_logits = layer_logits_np[-1]

    # Compute KL divergence per layer
    per_layer = []
    for i in range(parts.num_layers):
        kl = kl_divergence_np(deep_logits, layer_logits_np[i])
        avg_kl = float(np.mean(kl))
        median_kl = float(np.median(kl))

        is_global = (i % sw_pattern == sw_pattern - 1)
        attn_type = "global" if is_global else "sliding"

        per_layer.append({
            "layer_idx": i,
            "avg_kl": avg_kl,
            "median_kl": median_kl,
            "attn_type": attn_type,
        })

    # Find knowledge elbow (95% KL reduction)
    kl_layer0 = per_layer[0]["avg_kl"]
    optimal_exit = parts.num_layers - 1
    for entry in per_layer:
        reduction_pct = (kl_layer0 - entry["avg_kl"]) / kl_layer0 if kl_layer0 > 0 else 0
        entry["kl_reduction_pct"] = round(reduction_pct * 100, 1)
        if reduction_pct >= 0.95 and optimal_exit == parts.num_layers - 1:
            optimal_exit = entry["layer_idx"]

    # Subspace paradox check: geometric separation between consecutive layers
    geometric_separations = []
    for i in range(1, len(per_layer)):
        if per_layer[i]["avg_kl"] > 0:
            ratio = per_layer[i - 1]["avg_kl"] / per_layer[i]["avg_kl"]
            geometric_separations.append(ratio)

    max_geometric_sep = max(geometric_separations) if geometric_separations else 0
    avg_geometric_sep = np.mean(geometric_separations) if geometric_separations else 0

    return {
        "model": "gemma-3-4b-it",
        "architecture": "dense",
        "num_layers": parts.num_layers,
        "num_tokens_profiled": S,
        "forward_time_s": round(t_forward, 2),
        "per_layer": per_layer,
        "kl_layer0": round(kl_layer0, 4),
        "optimal_exit_95pct": optimal_exit,
        "max_geometric_separation": round(max_geometric_sep, 4),
        "avg_geometric_separation": round(float(avg_geometric_sep), 4),
    }


def multi_prompt_profile(
    parts: GemmaModelParts,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """Run profiling across all canonical prompts and aggregate."""
    from transcender_track_b_cascade import apply_generic_chat_template, build_harmony_messages

    all_profiles = []
    for i, user_prompt in enumerate(PROMPTS):
        prompt_id = f"P{i+1}"
        messages = build_harmony_messages(user_prompt, SYSTEM_PROMPT)
        prompt_text, _ = apply_generic_chat_template(parts.tokenizer, messages)

        print(f"\n  ── {prompt_id}: {user_prompt[:60]}...")
        profile = profile_gemma(parts, prompt_text, max_tokens=max_tokens)
        profile["prompt_id"] = prompt_id
        profile["user_prompt"] = user_prompt
        all_profiles.append(profile)

    # Aggregate: average KL per layer across all prompts
    num_layers = all_profiles[0]["num_layers"]
    aggregated_per_layer = []
    for layer_idx in range(num_layers):
        avg_kls = [p["per_layer"][layer_idx]["avg_kl"] for p in all_profiles]
        median_kls = [p["per_layer"][layer_idx]["median_kl"] for p in all_profiles]
        aggregated_per_layer.append({
            "layer_idx": layer_idx,
            "avg_kl": round(float(np.mean(avg_kls)), 4),
            "median_kl": round(float(np.mean(median_kls)), 4),
            "attn_type": all_profiles[0]["per_layer"][layer_idx]["attn_type"],
        })

    # Find aggregate optimal exit
    kl_layer0 = aggregated_per_layer[0]["avg_kl"]
    optimal_90 = num_layers - 1
    optimal_95 = num_layers - 1
    for entry in aggregated_per_layer:
        reduction = (kl_layer0 - entry["avg_kl"]) / kl_layer0 if kl_layer0 > 0 else 0
        entry["kl_reduction_pct"] = round(reduction * 100, 1)
        if reduction >= 0.90 and optimal_90 == num_layers - 1:
            optimal_90 = entry["layer_idx"]
        if reduction >= 0.95 and optimal_95 == num_layers - 1:
            optimal_95 = entry["layer_idx"]

    # Delta-KL between consecutive layers
    for i, entry in enumerate(aggregated_per_layer):
        if i == 0:
            entry["delta_kl"] = 0.0
        else:
            entry["delta_kl"] = round(
                aggregated_per_layer[i - 1]["avg_kl"] - entry["avg_kl"], 4
            )

    # Geometric separation
    geo_seps = []
    for i in range(1, len(aggregated_per_layer)):
        if aggregated_per_layer[i]["avg_kl"] > 0.001:
            ratio = aggregated_per_layer[i - 1]["avg_kl"] / aggregated_per_layer[i]["avg_kl"]
            geo_seps.append({"from_layer": i - 1, "to_layer": i, "ratio": round(ratio, 4)})

    geo_seps_sorted = sorted(geo_seps, key=lambda x: x["ratio"], reverse=True)

    return {
        "model": "gemma-3-4b-it",
        "architecture": "dense",
        "num_layers": num_layers,
        "num_prompts": len(all_profiles),
        "aggregated_per_layer": aggregated_per_layer,
        "kl_layer0": round(kl_layer0, 4),
        "optimal_exit_90pct": optimal_90,
        "optimal_exit_95pct": optimal_95,
        "top_geometric_separations": geo_seps_sorted[:5],
        "per_prompt_profiles": all_profiles,
    }


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_kl_profile(profile: Dict[str, Any], output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = profile["aggregated_per_layer"]
    x = [e["layer_idx"] for e in layers]
    avg_kl = [e["avg_kl"] for e in layers]
    median_kl = [e["median_kl"] for e in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Track C — Gemma 3 4B-IT KL Profile", fontsize=13, fontweight="bold")

    # Left: KL curve (log scale)
    ax1.semilogy(x, avg_kl, "r-o", markersize=3, label="Mean KL", linewidth=1.5)
    ax1.semilogy(x, median_kl, "b--s", markersize=3, label="Median KL", linewidth=1.0, alpha=0.7)

    # Mark elbow points
    e90 = profile["optimal_exit_90pct"]
    e95 = profile["optimal_exit_95pct"]
    ax1.axvline(e90, color="green", linestyle=":", alpha=0.8, label=f"90% exit: L{e90}")
    ax1.axvline(e95, color="orange", linestyle="--", alpha=0.8, label=f"95% exit: L{e95}")

    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("KL Divergence (nats)")
    ax1.set_title("KL(Deep || Layer_i) — Quality vs Depth")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Delta-KL (information gain per layer)
    delta_kl = [e["delta_kl"] for e in layers]
    colors = ["#2ca02c" if e["attn_type"] == "global" else "#1f77b4" for e in layers]
    ax2.bar(x, delta_kl, color=colors, alpha=0.7, width=0.8)
    ax2.axvline(e90, color="green", linestyle=":", alpha=0.8)
    ax2.axvline(e95, color="orange", linestyle="--", alpha=0.8)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("ΔKL (information gained)")
    ax2.set_title("Per-Layer Information Gain")
    ax2.grid(True, alpha=0.3)

    # Legend for attention types
    from matplotlib.patches import Patch
    ax2.legend(
        handles=[Patch(color="#2ca02c", label="Global attn"), Patch(color="#1f77b4", label="Sliding attn")],
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  KL profile saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Terminal Summary
# ═══════════════════════════════════════════════════════════════════

def print_profile_summary(profile: Dict[str, Any]):
    layers = profile["aggregated_per_layer"]
    num_layers = profile["num_layers"]

    print(f"\n{'='*65}")
    print(f"  Track C — Gemma 3 4B-IT KL Divergence Profile")
    print(f"  {num_layers} layers, {profile['num_prompts']} prompts aggregated")
    print(f"{'='*65}")
    print(f"\n    {'Layer':<8} {'Type':<9} {'Avg KL':>8} {'Med KL':>9} {'ΔKL':>7} {'Red%':>6}")
    print(f"  {'─'*55}")

    for e in layers:
        marker = ""
        if e["layer_idx"] == profile["optimal_exit_90pct"]:
            marker = " ◀ 90%"
        elif e["layer_idx"] == profile["optimal_exit_95pct"]:
            marker = " ◀ 95%"
        elif e["layer_idx"] == num_layers - 1:
            marker = " ◀ DEEP"

        print(
            f"  L {e['layer_idx']:>3}  {e['attn_type']:>8}  "
            f"{e['avg_kl']:>8.2f}  {e['median_kl']:>8.2f}  "
            f"{e['delta_kl']:>+6.2f}  {e['kl_reduction_pct']:>5.1f}%{marker}"
        )

    print(f"\n  {'─'*55}")
    print(f"  Layer 0 KL:           {profile['kl_layer0']:.2f}")
    print(f"  90% reduction at:     Layer {profile['optimal_exit_90pct']}")
    print(f"  95% reduction at:     Layer {profile['optimal_exit_95pct']}")

    if profile["top_geometric_separations"]:
        top = profile["top_geometric_separations"][0]
        print(f"  Max geometric sep:    {top['ratio']:.2f}x (L{top['from_layer']}→L{top['to_layer']})")

    # Candidate exit layers for Phase 2
    e90 = profile["optimal_exit_90pct"]
    e95 = profile["optimal_exit_95pct"]
    early = max(e90 - 4, num_layers // 3)
    mid = e90
    late = e95

    print(f"\n  Candidate exit layers for adaptive benchmark:")
    print(f"    Early:  L{early}")
    print(f"    Mid:    L{mid}")
    print(f"    Late:   L{late}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Track C — Gemma 3 4B-IT KL Profiling"
    )
    parser.add_argument(
        "--model", type=str,
        default=str((Path(__file__).resolve().parent.parent / "gemma-3-4b-it").resolve()),
        help="Path to Gemma 3 4B-IT model",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max input tokens per prompt",
    )
    parser.add_argument(
        "--output", type=str,
        default=str((Path(__file__).resolve().parent / "gemma_kl_profile.json").resolve()),
        help="Output JSON path",
    )
    parser.add_argument(
        "--plot", type=str,
        default=str((Path(__file__).resolve().parent / "gemma_kl_profile.png").resolve()),
        help="Output plot path",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  Track C — Gemma 3 4B-IT KL Profiling")
    print("=" * 65)

    # Load model
    print(f"\n  Loading {args.model}...")
    model, tokenizer = mlx_load(args.model)
    parts = GemmaModelParts.from_loaded(model, tokenizer)
    print(f"  Architecture: dense, {parts.num_layers} layers, hidden=2560")
    print(f"  Tied embeddings: {parts.tie_word_embeddings}")

    # Run multi-prompt profiling
    profile = multi_prompt_profile(parts, max_tokens=args.max_tokens)

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\n  Profile JSON saved to {args.output}")

    # Plot
    plot_kl_profile(profile, args.plot)

    # Summary
    print_profile_summary(profile)

    print(f"\n  Reconnaissance complete.")
    print(f"  Next: run Track C adaptive benchmark with candidate exit layers.")


if __name__ == "__main__":
    main()
