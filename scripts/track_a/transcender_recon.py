"""
Transcender Reconnaissance — KL Profiling + Expert-Skipping on GPT-OSS 20B

Three missions:
  1. KL-Divergence Profiler:  Per-layer quality measurement with token-type analysis
  2. Expert-Skipping Engine:  Run attention (KV-cache intact), skip MoE-MLP for exited tokens
  3. Memory-Efficient Loader: MXFP4 backbone + FP32 lm_head/norm

The Expert-Skipping strategy solves the "Deep KV Semantics" problem:
  ┌────────────────────────────────────────────────────────────────────┐
  │ FULL Layer Skip (BROKEN):                                         │
  │   Token exits → entire layer skipped → KV cache has HOLE          │
  │   → future tokens attend to missing position → garbage output     │
  │                                                                    │
  │ Expert Skip (THIS FILE):                                           │
  │   Token exits → attention runs normally → KV cache INTACT          │
  │                → MoE-MLP skipped → 66% FLOP savings per layer      │
  │   → future tokens see valid attention state → coherent output      │
  └────────────────────────────────────────────────────────────────────┘

Usage:
    # Step 1: Install dependencies
    pip install mlx mlx-lm          # Apple Silicon (optimal)
    pip install matplotlib numpy

    # Step 2: Download model
    huggingface-cli download openai/gpt-oss-20b

    # Step 3: Run reconnaissance
    python transcender_recon.py                    # dry-run (validate structure)
    python transcender_recon.py --load              # load model + KL profile
    python transcender_recon.py --load --generate   # + expert-skipping demo
"""

from __future__ import annotations
import argparse
import math
import sys
import time
from typing import Optional

try:
    import mlx.core as mx
    import mlx.nn as mnn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np

try:
# Import architecture config and MLX runtime from our engine module
    from transcender_engine import (
        GptOssConfig,
        MLXDynamicExpertEngine,
        apply_harmony_template,
        load_resolved_mlx_model,
        load_resolved_transformers_model,
    )
except Exception:
    MLXDynamicExpertEngine = None


# ═══════════════════════════════════════════════════════════════════
# Numerical Helpers (numpy — backend-agnostic)
# ═══════════════════════════════════════════════════════════════════

def softmax_np(x, axis=-1):
    """Numerically stable softmax in numpy."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax_np(x, axis=-1):
    """Numerically stable log-softmax in numpy."""
    return x - np.max(x, axis=axis, keepdims=True) - np.log(
        np.sum(np.exp(x - np.max(x, axis=axis, keepdims=True)),
               axis=axis, keepdims=True)
    )


def kl_divergence_np(deep_logits, early_logits):
    """KL(Deep || Early) per token in numpy. Returns shape (B, S)."""
    deep_probs = softmax_np(deep_logits)
    deep_log = log_softmax_np(deep_logits)
    early_log = log_softmax_np(early_logits)
    return np.sum(deep_probs * (deep_log - early_log), axis=-1)


def to_numpy(x):
    """Convert any tensor (mlx, torch, numpy) to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    if HAS_MLX and isinstance(x, mx.array):
        return np.array(x.astype(mx.float32))
    return np.array(x)


# ═══════════════════════════════════════════════════════════════════
# 1. Memory-Efficient Model Loading
# ═══════════════════════════════════════════════════════════════════

def load_model_for_profiling(
    model_name: str = "openai/gpt-oss-20b",
    backend: str = "auto",
) -> tuple:
    """
    Load GPT-OSS 20B for KL profiling.

    Backend priority:
      1. MLX via mlx-lm  (optimal for Apple Silicon, handles MXFP4 natively)
      2. Transformers     (fallback, uses torch_dtype="auto")

    Precision strategy:
      - Expert weights (MoE MLP):     MXFP4 (as shipped by OpenAI)
      - Attention, MoE router, embed: Native precision (excluded from MXFP4)
      - final_norm + lm_head:         Highest available precision

    Returns:
        (model, tokenizer, backend_name)
        backend_name is "mlx" or "torch"
    """
    # ── Try MLX first (optimal for M1 Pro) ──
    if backend in ("auto", "mlx") and HAS_MLX:
        try:
            print(f"\n  Loading {model_name} via mlx-lm...")
            print(f"  (First load downloads from HuggingFace, ~11 GB)")
            model, tokenizer, resolved_model_name = load_resolved_mlx_model(
                model_name,
                lazy=True,
            )
            if resolved_model_name != model_name:
                print(
                    "  Detected placeholder GPT-OSS metadata in the requested "
                    f"directory; using cached snapshot {resolved_model_name}"
                )

            # Count parameters
            n_params = sum(
                v.size for _, v in model.parameters().items()
                if isinstance(v, mx.array)
            ) if hasattr(model, 'parameters') else 0

            # Get layer info
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                n_layers = len(model.model.layers)
            elif hasattr(model, 'layers'):
                n_layers = len(model.layers)
            else:
                n_layers = -1

            print(f"  Backend:    MLX (Apple Silicon unified memory)")
            print(f"  Layers:     {n_layers}")
            print(f"  Parameters: {n_params/1e9:.1f}B")

            return model, tokenizer, "mlx"
        except Exception as e:
            print(f"  MLX loading failed: {e}")
            if backend == "mlx":
                raise
            print(f"  Falling back to transformers...")

    # ── Fallback: Transformers (torch) ──
    if HAS_TORCH:
        print(f"\n  Loading {model_name} via transformers...")
        model, tokenizer, resolved_model_name = load_resolved_transformers_model(
            model_name,
        )
        if resolved_model_name != model_name:
            print(
                "  Detected placeholder GPT-OSS metadata in the requested "
                f"directory; using cached snapshot {resolved_model_name}"
            )

        # Force lm_head and norm to float32 for precise logit computation
        if hasattr(model, "lm_head"):
            model.lm_head = model.lm_head.float()
        if hasattr(model.model, "norm"):
            model.model.norm = model.model.norm.float()

        model.eval()

        param_bytes = sum(
            p.nelement() * p.element_size() for p in model.parameters()
        )
        print(f"  Backend:    PyTorch (transformers)")
        print(f"  Layers:     {len(model.model.layers)}")
        print(f"  Memory:     {param_bytes / 1e9:.1f} GB")

        return model, tokenizer, "torch"

    raise RuntimeError("No backend available. Install mlx-lm or torch.")


# ═══════════════════════════════════════════════════════════════════
# 2. KL-Divergence Profiler
# ═══════════════════════════════════════════════════════════════════

class KLProfiler:
    """
    Comprehensive per-layer KL divergence profiler for GPT-OSS 20B.

    For each of the 24 layers, computes:
      KL(Deep_Logits || Layer_i_Logits)

    This measures how much information is LOST by exiting at layer i
    instead of running the full 24-layer stack. The layer where KL
    drops below threshold is the Optimal Exit Layer.

    Token-type analysis:
      Classifies tokens into categories and measures whether certain
      types converge faster (lower KL) than others:
        - Function words (the, a, of, is, etc.) → expect early convergence
        - Content words (nouns, verbs, adjectives) → expect late convergence
        - Subword fragments (##ing, ▁the) → expect early convergence
        - Punctuation → expect early convergence
        - Reasoning markers (<reasoning>, </reasoning>) → special analysis
    """

    # Common English function words (high-frequency, low-information)
    FUNCTION_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "of", "in", "to", "for", "with", "on", "at", "from", "by",
        "as", "or", "and", "but", "if", "not", "no", "so", "it",
        "this", "that", "these", "those", "he", "she", "they", "we",
        "I", "you", "his", "her", "its", "our", "their", "my",
    }

    def __init__(self, model, tokenizer, backend: str = "auto",
                 config: GptOssConfig = GptOssConfig()):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Detect backend
        if backend == "auto":
            self._backend = "mlx" if (HAS_MLX and hasattr(model, 'model')
                                      and not HAS_TORCH) else "torch"
            # If model params are mx.array, it's MLX
            if HAS_MLX:
                try:
                    params = dict(model.parameters())
                    if params and isinstance(next(iter(params.values())), mx.array):
                        self._backend = "mlx"
                except Exception:
                    pass
        else:
            self._backend = backend

        # Extract model components (same structure for both backends)
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.final_norm = model.model.norm
        self.lm_head = model.lm_head
        self.num_layers = len(self.layers)
        self.layer_types = list(
            getattr(model.model, "layer_types", config.layer_types)
        )

    @classmethod
    def classify_token(cls, token_str: str) -> str:
        """Classify a token into a semantic category."""
        clean = token_str.strip().lstrip("ĠġâĊ▁ ")

        if not clean:
            return "whitespace"
        if clean in ("<reasoning>", "</reasoning>", "<|system|>",
                     "<|user|>", "<|assistant|>"):
            return "special"
        if clean in (".", ",", "!", "?", ":", ";", "-", "(", ")", '"', "'"):
            return "punctuation"
        if clean.lower() in cls.FUNCTION_WORDS:
            return "function"
        if len(clean) <= 2 and not clean.isalpha():
            return "fragment"
        return "content"

    def _run_layer(self, layer, hidden):
        """Run a single layer, handle both tuple and tensor outputs."""
        if self._backend == "mlx":
            S = hidden.shape[1]
            mask = mnn.MultiHeadAttention.create_additive_causal_mask(S).astype(hidden.dtype)
            output = layer(hidden, mask=mask)
        else:
            output = layer(hidden)
        if isinstance(output, tuple):
            return output[0]
        return output

    def _compute_logits(self, hidden):
        """Project hidden states through norm + lm_head → logits as numpy."""
        logits = self.lm_head(self.final_norm(hidden))
        if self._backend == "mlx":
            mx.eval(logits)
        return to_numpy(logits)

    def _copy_hidden(self, hidden):
        """Copy hidden states (backend-aware)."""
        if self._backend == "mlx":
            # MLX arrays are immutable, so indexing creates a new array.
            # But to be safe with lazy evaluation, evaluate first.
            mx.eval(hidden)
            return hidden
        return hidden.clone()

    def profile(
        self,
        prompt: str,
        max_tokens: int = 512,
        reasoning_effort: str = "medium",
    ) -> dict:
        """
        Run a single forward pass and compute per-layer KL divergence.

        Backend-agnostic: works with both MLX and PyTorch models.
        Forward pass runs in native framework; KL math done in numpy.

        Returns:
            {
                "per_layer": [{layer_idx, avg_kl, median_kl, safe_exit_pct, attn_type}, ...],
                "per_token_type": {type: {layer_idx: avg_kl, ...}, ...},
                "tokens": [token_str, ...],
                "optimal_exit": int,
            }
        """
        prompt_text, messages = apply_harmony_template(
            self.tokenizer,
            user_prompt=prompt,
            reasoning_effort=reasoning_effort,
        )

        # ── Tokenize ──
        if self._backend == "mlx":
            token_ids = self.tokenizer.encode(prompt_text)[:max_tokens]
            input_ids = mx.array(token_ids).reshape(1, -1)
        else:
            tokens_dict = self.tokenizer(
                prompt_text, return_tensors="pt", max_length=max_tokens, truncation=True,
            )
            input_ids = tokens_dict["input_ids"]
            if HAS_TORCH:
                input_ids = input_ids.to(next(self.model.parameters()).device)

        # Get token strings for classification
        ids_list = to_numpy(input_ids)[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids_list)
        S = len(tokens)

        print(f"\n  Profiling {S} Harmony-formatted tokens across {self.num_layers} layers "
              f"(backend: {self._backend}, reasoning={reasoning_effort})...")

        # ── Forward pass: collect logits at every layer ──
        # We compute logits immediately and store as numpy to avoid
        # holding 24 copies of hidden states in GPU/unified memory.
        hidden = self.embed_tokens(input_ids)
        layer_logits_np = []  # list of numpy arrays (1, S, V)

        t0 = time.time()
        for i in range(self.num_layers):
            hidden = self._run_layer(self.layers[i], hidden)
            # Project through norm + lm_head → numpy logits
            logits_np = self._compute_logits(hidden)
            layer_logits_np.append(logits_np)

        elapsed = time.time() - t0
        print(f"  Forward pass: {elapsed:.1f}s ({elapsed/self.num_layers:.2f}s/layer)")

        # ── Deep logits (ground truth from last layer) ──
        deep_logits_np = layer_logits_np[-1]  # (1, S, V)

        # ── Classify tokens ──
        token_types = [self.classify_token(t) for t in tokens]
        type_indices = {}
        for idx, ttype in enumerate(token_types):
            type_indices.setdefault(ttype, []).append(idx)

        # ── Per-layer KL computation (all numpy) ──
        # Safe% is an absolute safety metric: top-1 agreement with layer 23.
        per_layer = []
        per_token_type = {t: {} for t in type_indices}

        deep_top1 = np.argmax(deep_logits_np, axis=-1)

        print(f"\n  {'Layer':>7} {'Type':>10} {'Avg KL':>10} "
              f"{'Median KL':>11} {'Safe%':>8} {'ΔKL':>8}")
        print(f"  {'─' * 60}")

        prev_avg_kl = None
        for i, early_logits_np in enumerate(layer_logits_np):
            # KL(Deep || Early) per token — computed in numpy
            kl = kl_divergence_np(deep_logits_np, early_logits_np)  # (1, S)
            kl_flat = kl[0]  # (S,)

            avg_kl = float(np.mean(kl_flat))
            med_kl = float(np.median(kl_flat))
            early_top1 = np.argmax(early_logits_np, axis=-1)
            safe_pct = float(np.mean(early_top1 == deep_top1)) * 100

            delta = (prev_avg_kl - avg_kl) if prev_avg_kl is not None else 0.0
            prev_avg_kl = avg_kl

            attn_type = (
                "slide" if "sliding" in self.layer_types[i] else "full"
            )
            marker = ""
            if i == 11:
                marker = " ◀ MID"
            elif i == self.num_layers - 1:
                marker = " ◀ DEEP"

            per_layer.append({
                "layer": i,
                "avg_kl": avg_kl,
                "median_kl": med_kl,
                "safe_exit_pct": safe_pct,
                "attn_type": attn_type,
            })

            print(f"  L{i:>4}   {attn_type:>8}   {avg_kl:>10.2f} "
                  f"{med_kl:>11.2f} {safe_pct:>6.1f}% "
                  f"{delta:>+7.2f}{marker}")

            # Per-token-type KL
            for ttype, indices in type_indices.items():
                type_kl = float(np.mean(kl_flat[indices]))
                per_token_type[ttype][i] = type_kl

        # ── Find Optimal Exit Layer ──
        kl_drops = []
        for i in range(1, len(per_layer)):
            drop = per_layer[i - 1]["avg_kl"] - per_layer[i]["avg_kl"]
            kl_drops.append((i, drop))
        kl_drops.sort(key=lambda x: x[1], reverse=True)

        layer0_kl = per_layer[0]["avg_kl"]
        optimal = self.num_layers - 1
        for info in per_layer:
            if info["avg_kl"] < layer0_kl * 0.05:  # 95% KL reduction
                optimal = info["layer"]
                break

        print(f"\n  {'─' * 60}")
        print(f"  Layer 0 KL:       {per_layer[0]['avg_kl']:.2f}")
        print(f"  Steepest drop:    Layer {kl_drops[0][0]} "
              f"(ΔKL = {kl_drops[0][1]:.2f})")
        print(f"  95% reduction at: Layer {optimal}")

        # ── Token-type summary ──
        print(f"\n  Token-Type KL at Layer 12 (mid-stack):")
        print(f"  {'Type':<15} {'Count':>6} {'KL@L12':>10} {'KL@L6':>10}")
        print(f"  {'─' * 45}")
        for ttype, indices in sorted(type_indices.items()):
            kl_12 = per_token_type[ttype].get(12, 0.0)
            kl_6 = per_token_type[ttype].get(6, 0.0)
            print(f"  {ttype:<15} {len(indices):>6} {kl_12:>10.2f} {kl_6:>10.2f}")

        return {
            "per_layer": per_layer,
            "per_token_type": per_token_type,
            "tokens": tokens,
            "token_types": token_types,
            "messages": messages,
            "optimal_exit": optimal,
            "kl_drops": kl_drops,
        }

    def plot_kl_profile(self, profile: dict, save_path: str = "kl_profile.png"):
        """Generate KL divergence profile visualization."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ── Left: Per-layer KL curve ──
        ax1 = axes[0]
        layers = [p["layer"] for p in profile["per_layer"]]
        avg_kls = [p["avg_kl"] for p in profile["per_layer"]]
        med_kls = [p["median_kl"] for p in profile["per_layer"]]

        ax1.plot(layers, avg_kls, "o-", color="#e74c3c", linewidth=2,
                 markersize=6, label="Mean KL")
        ax1.plot(layers, med_kls, "s--", color="#3498db", linewidth=1.5,
                 markersize=5, label="Median KL", alpha=0.7)

        # Mark optimal exit
        opt = profile["optimal_exit"]
        opt_kl = avg_kls[opt]
        ax1.axvline(x=opt, color="#2ecc71", linestyle=":", linewidth=2,
                    label=f"Optimal Exit: L{opt}")
        ax1.scatter([opt], [opt_kl], color="#2ecc71", s=150, zorder=5,
                    marker="*", edgecolors="black")

        # Mark current exit point (Layer 12)
        ax1.axvline(x=12, color="#f39c12", linestyle="--", linewidth=1.5,
                    alpha=0.5, label="Current Exit: L12")

        ax1.set_xlabel("Layer Index", fontsize=12)
        ax1.set_ylabel("KL Divergence (nats)", fontsize=12)
        ax1.set_title("KL(Deep || Layer_i) — Quality vs Depth", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))

        # ── Right: Token-type comparison ──
        ax2 = axes[1]
        type_data = profile["per_token_type"]
        colors = {
            "function": "#3498db",
            "content": "#e74c3c",
            "punctuation": "#2ecc71",
            "fragment": "#9b59b6",
            "special": "#f39c12",
            "whitespace": "#95a5a6",
        }

        for ttype, layer_kls in sorted(type_data.items()):
            if len(layer_kls) < 2:
                continue
            x = sorted(layer_kls.keys())
            y = [layer_kls[l] for l in x]
            color = colors.get(ttype, "#7f8c8d")
            ax2.plot(x, y, "o-", color=color, linewidth=1.5, markersize=4,
                     label=f"{ttype} ({len(profile['token_types'])} tok)",
                     alpha=0.8)

        ax2.set_xlabel("Layer Index", fontsize=12)
        ax2.set_ylabel("KL Divergence (nats)", fontsize=12)
        ax2.set_title("Token-Type KL Convergence", fontsize=13)
        ax2.legend(fontsize=9, loc="upper right")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))

        fig.suptitle("Transcender Reconnaissance — GPT-OSS 20B KL Profile",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  KL profile saved to {save_path}")
        plt.close()


# ═══════════════════════════════════════════════════════════════════
# 3. Expert-Skipping Engine
# ═══════════════════════════════════════════════════════════════════

class ExpertSkippingEngine:
    """
    Modified layer forward that keeps attention intact but skips MoE-MLP.

    Solves the "Deep KV Semantics" problem:
      - Attention runs for ALL tokens → KV-cache always populated
      - RoPE positional encoding always applied
      - Only the expensive MoE-MLP block is skipped for exited tokens

    Per-layer compute savings:
      Full layer:     Attention + MoE(top-4/32 experts, SwiGLU)
      Expert-skip:    Attention only
      Savings:        ~66% of per-layer FLOPS (MoE dominates)

    Compound savings (for exited tokens, layers 12-23):
      12 layers × 66% MLP savings = ~8 layers' worth of compute saved
      Plus MoE was already sparse (4/32 = 12.5% utilization)
    """

    def __init__(self, model, tokenizer, config: GptOssConfig = GptOssConfig()):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Extract components
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.final_norm = model.model.norm
        self.lm_head = model.lm_head
        self.num_layers = len(self.layers)

        # Son Router (reuse from transcender_engine)
        from transcender_engine import SonRouter
        self.router = SonRouter(config)

    @staticmethod
    def _layer_attention_only(layer, hidden_states, **kwargs):
        """
        Run ONLY the attention sub-block of a GptOssDecoderLayer.

        GptOssDecoderLayer internal structure:
          1. input_layernorm → self_attn → residual add
          2. post_attention_layernorm → mlp (MoE) → residual add

        This function executes step 1 only, skipping step 2.
        The MoE-MLP block (32 experts, top-4 gating) is never called.
        """
        # Step 1: Pre-norm + Self-Attention + Residual
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        # Run self-attention (populates KV cache, applies RoPE)
        attn_output = layer.self_attn(normed, **kwargs)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        hidden_states = residual + attn_output
        # Step 2 (MLP) is SKIPPED — hidden_states stays as attention-only output
        return hidden_states

    @staticmethod
    def _layer_full(layer, hidden_states, **kwargs):
        """Run the full layer (attention + MLP)."""
        output = layer(hidden_states, **kwargs)
        return output[0] if isinstance(output, tuple) else output

    def _layer_selective(
        self,
        layer,
        hidden_states: torch.Tensor,
        exit_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run attention for ALL tokens, MLP only for non-exited tokens.

        This is the core Expert-Skipping operation:
          1. Run attention for all tokens (KV cache intact)
          2. Gather non-exited tokens → compact tensor
          3. Run MoE-MLP only on compact tensor (physical savings)
          4. Scatter results back, merge with attention-only states

        Args:
            layer: GptOssDecoderLayer
            hidden_states: (B, S, D)
            exit_mask: (B, S) bool — True means SKIP MLP for this token
        """
        B, S, D = hidden_states.shape

        # ── Step 1: Attention for ALL tokens ──
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)
        attn_output = layer.self_attn(normed, **kwargs)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        attn_states = residual + attn_output  # (B, S, D)

        # If all tokens exit, skip MLP entirely
        if exit_mask.all():
            return attn_states

        # If no tokens exit, run full MLP
        if not exit_mask.any():
            residual2 = attn_states
            normed2 = layer.post_attention_layernorm(attn_states)
            mlp_out = layer.mlp(normed2)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
            return residual2 + mlp_out

        # ── Step 2: Selective MLP (physical compute savings) ──
        # Flatten for indexing
        flat_states = attn_states.view(B * S, D)
        flat_mask = exit_mask.view(B * S)
        non_exit_idx = (~flat_mask).nonzero(as_tuple=True)[0]

        # Gather non-exited tokens into compact tensor
        selected = flat_states[non_exit_idx]  # (N, D) where N < B*S

        # Run MLP only on selected tokens
        # Reshape to (1, N, D) for layer norm compatibility
        selected = selected.unsqueeze(0)
        residual2 = selected
        normed2 = layer.post_attention_layernorm(selected)
        mlp_out = layer.mlp(normed2)
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        mlp_result = (residual2 + mlp_out).squeeze(0)  # (N, D)

        # ── Step 3: Scatter back ──
        output = flat_states.clone()  # Start with attention-only states
        output[non_exit_idx] = mlp_result  # Overwrite non-exited with full states

        return output.view(B, S, D)

    def generate_with_expert_skipping(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        exit_after_layer: int = 12,
        exit_threshold: float = 0.5,
    ) -> dict:
        """
        Autoregressive generation with Expert-Skipping.

        For each new token:
          1. Layers 0-11: Full forward (attention + MLP) for all tokens
          2. Son Router at Layer 12: exit decision
          3. Layers 12-23: For exited tokens → attention only (skip MoE-MLP)
                           For continuing tokens → full forward

        This produces REAL wall-clock speedup because:
          - MoE-MLP is ~66% of per-layer compute
          - Gathering non-exited tokens into a compact tensor means
            the MoE forward runs on a SMALLER tensor
          - MLX/PyTorch only compute what's materialized
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(
            next(self.model.parameters()).device
        )
        generated = input_ids.clone()

        stats = {
            "exit_decisions": [],
            "tokens_generated": 0,
            "total_mlp_calls": 0,      # How many (token, layer) pairs ran MLP
            "total_attn_calls": 0,      # How many (token, layer) pairs ran attention
            "skipped_mlp_calls": 0,     # How many MLP calls were physically skipped
        }

        t_start = time.time()

        for step in range(max_new_tokens):
            B, S = generated.shape

            # ── Embedding ──
            hidden = self.embed_tokens(generated)

            # ── Early layers (0 to exit_layer-1): full forward ──
            for i in range(exit_after_layer):
                hidden = self._layer_full(self.layers[i], hidden)
            stats["total_attn_calls"] += exit_after_layer * S
            stats["total_mlp_calls"] += exit_after_layer * S

            # ── Son Router at exit layer ──
            routing = self.router(hidden)
            exit_mask = routing["exit_probs"] > exit_threshold  # (B, S)
            exit_prob_last = routing["exit_probs"][0, -1].item()
            last_exits = exit_mask[0, -1].item()

            stats["exit_decisions"].append({
                "step": step,
                "exit_prob": exit_prob_last,
                "decision": "EXIT" if last_exits else "CONTINUE",
            })

            # ── Deep layers (exit_layer to 23): expert-skipping ──
            deep_layers = self.num_layers - exit_after_layer
            n_exited = exit_mask.sum().item()
            n_total = B * S

            for i in range(exit_after_layer, self.num_layers):
                hidden = self._layer_selective(
                    self.layers[i], hidden, exit_mask,
                )

            stats["total_attn_calls"] += deep_layers * S
            stats["total_mlp_calls"] += deep_layers * (n_total - n_exited)
            stats["skipped_mlp_calls"] += deep_layers * n_exited

            # ── Logits from final hidden state ──
            logits = self.lm_head(self.final_norm(hidden[:, -1:, :].float()))

            # Greedy decode
            next_token = logits[0, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat(
                [generated, next_token.unsqueeze(0)], dim=1,
            )
            stats["tokens_generated"] += 1

            # Check EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        t_elapsed = time.time() - t_start

        # ── Decode output ──
        output_text = self.tokenizer.decode(
            generated[0, input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # ── Compute savings ──
        total_possible_mlp = stats["total_mlp_calls"] + stats["skipped_mlp_calls"]
        mlp_savings = (stats["skipped_mlp_calls"] / max(total_possible_mlp, 1)) * 100
        exit_rate = sum(
            1 for d in stats["exit_decisions"] if d["decision"] == "EXIT"
        ) / max(stats["tokens_generated"], 1) * 100

        # Print results
        print(f"\n  {'═' * 60}")
        print(f"  Expert-Skipping Generation Results")
        print(f"  {'═' * 60}")
        print(f"  Prompt:     {prompt[:60]}...")
        print(f"  Generated:  {stats['tokens_generated']} tokens in {t_elapsed:.1f}s")
        print(f"  Speed:      {stats['tokens_generated']/t_elapsed:.1f} tok/s")
        print(f"  Exit rate:  {exit_rate:.1f}%")
        print(f"  MLP skips:  {stats['skipped_mlp_calls']:,} / "
              f"{total_possible_mlp:,} ({mlp_savings:.1f}%)")
        print(f"  {'─' * 60}")
        print(f"  Output: {output_text[:200]}")
        print(f"  {'═' * 60}")

        stats["output_text"] = output_text
        stats["elapsed_s"] = t_elapsed
        stats["mlp_savings_pct"] = mlp_savings
        stats["exit_rate_pct"] = exit_rate

        return stats


# ═══════════════════════════════════════════════════════════════════
# 4. Dry-Run Validation (no model required)
# ═══════════════════════════════════════════════════════════════════

def dry_run():
    """Validate all components structurally without loading the 20B model."""
    print("=" * 65)
    print("  Transcender Reconnaissance — Dry Run")
    print("  (Validates structure without model weights)")
    print("=" * 65)

    config = GptOssConfig()

    # Test token classifier
    profiler_tokens = [
        ("Ġthe", "function"), ("Ġcat", "content"), (".", "punctuation"),
        ("Ġ", "whitespace"), ("<reasoning>", "special"), ("ing", "fragment"),
    ]
    print(f"\n  Token Classification Test:")
    # Use a minimal mock to test classification without a full model
    for token, expected in profiler_tokens:
        actual = KLProfiler.classify_token(token)
        status = "✓" if actual == expected else f"✗ (got {actual})"
        print(f"    {token:<20} → {actual:<15} {status}")

    # Test selective MLP logic
    print(f"\n  Expert-Skipping Logic Test:")
    if HAS_TORCH:
        # Simulate exit mask scenarios
        B, S, D = 1, 8, config.hidden_size
        fake_hidden = torch.randn(B, S, D)

        scenarios = [
            ("No exits", torch.zeros(B, S, dtype=torch.bool)),
            ("All exit", torch.ones(B, S, dtype=torch.bool)),
            ("50% exit", torch.tensor([[True, False, True, False,
                                        True, False, True, False]])),
            ("1 exit", torch.tensor([[True, False, False, False,
                                      False, False, False, False]])),
        ]
        for name, mask in scenarios:
            n_exit = mask.sum().item()
            n_mlp = B * S - n_exit
            savings = n_exit / (B * S) * 100
            print(f"    {name:<12} → {n_exit}/{B*S} skip MLP, "
                  f"{n_mlp}/{B*S} run MLP ({savings:.0f}% savings)")

    # Memory estimates
    print(f"\n  {'─' * 50}")
    print(f"  EXPERT-SKIPPING SAVINGS ESTIMATE")
    print(f"  {'─' * 50}")

    # MoE-MLP FLOPS per token per layer:
    #   Router: hidden_size × num_experts = 2880 × 32 = 92K
    #   Expert (top-4): 4 × (2 × hidden_size × intermediate_size) = 4 × 2 × 2880 × 2880 = 66.4M
    #   Total MLP: ~66.5M FLOPS
    # Attention per token per layer:
    #   QKV projection: 3 × hidden_size × hidden_size = 3 × 2880² = 24.9M
    #   Attention logits + softmax: ~seq_len × hidden_size
    #   Output projection: hidden_size × hidden_size = 2880² = 8.3M
    #   Total attention: ~33.2M FLOPS
    # MLP fraction: 66.5 / (66.5 + 33.2) = 66.7%

    mlp_flops = 4 * 2 * config.hidden_size * config.intermediate_size
    attn_flops = 4 * config.hidden_size * config.hidden_size  # approx
    mlp_frac = mlp_flops / (mlp_flops + attn_flops) * 100

    print(f"  Per-layer MLP FLOPS:  {mlp_flops/1e6:.1f}M")
    print(f"  Per-layer Attn FLOPS: {attn_flops/1e6:.1f}M (approx)")
    print(f"  MLP fraction:         {mlp_frac:.1f}%")

    for exit_rate in [25, 50, 75]:
        deep_layers = config.num_hidden_layers // 2  # layers 12-23
        skipped = deep_layers * (exit_rate / 100) * mlp_frac / 100
        total_savings = skipped / config.num_hidden_layers * 100
        print(f"  {exit_rate}% exit rate → "
              f"{total_savings:.1f}% total compute savings")

    print(f"\n  {'─' * 50}")
    print(f"  SETUP COMMANDS")
    print(f"  {'─' * 50}")
    print(f"  # 1. Install MLX (Apple Silicon only)")
    print(f"  pip install mlx mlx-lm matplotlib")
    print(f"")
    print(f"  # 2. Download model (~11 GB from HuggingFace)")
    print(f"  #    NOTE: ollama GGUF can't be used for layer profiling.")
    print(f"  #    mlx-lm downloads from HF automatically on first load.")
    print(f"  python transcender_recon.py --load")
    print(f"")
    print(f"  # 3. Run with expert-skipping generation")
    print(f"  python transcender_recon.py --load --generate")
    print(f"\n{'=' * 65}")


# ═══════════════════════════════════════════════════════════════════
# 5. Main — Execution Modes
# ═══════════════════════════════════════════════════════════════════

DEFAULT_PROFILE_PROMPT = (
    "What is the capital of France and why is it historically significant?"
)


def main():
    parser = argparse.ArgumentParser(
        description="Transcender Reconnaissance — KL Profiling + Expert-Skipping"
    )
    parser.add_argument(
        "--load", action="store_true",
        help="Load the real GPT-OSS 20B model"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Run expert-skipping generation demo (requires --load)"
    )
    parser.add_argument(
        "--model", type=str, default="openai/gpt-oss-20b",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Custom user prompt for Harmony-formatted profiling"
    )
    parser.add_argument(
        "--reasoning-effort", type=str, default="medium",
        choices=["low", "medium", "high"],
        help="Harmony reasoning effort passed into apply_chat_template (default: medium)"
    )
    parser.add_argument(
        "--exit-layer", type=int, default=22,
        help="Hard exit layer for expert-skipping (default: 22)"
    )
    parser.add_argument(
        "--soft-skip-layer", type=int, default=19,
        help="First layer where entropy-based expert-skipping can start (default: 19)"
    )
    parser.add_argument(
        "--entropy-threshold", type=float, default=0.20,
        help="Normalized entropy threshold for soft-skipping (default: 0.20)"
    )
    parser.add_argument(
        "--memory-limit-gb", type=float, default=30.0,
        help="MLX unified-memory budget in GB (default: 30.0)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Benchmark the dynamic engine against full-depth generation"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run the mlx_lm 2048-prefill / 128-gen baseline protocol before dynamic inference"
    )
    parser.add_argument(
        "--plot", action="store_true", default=True,
        help="Generate KL profile plot (default: True)"
    )
    args = parser.parse_args()

    if not args.load:
        dry_run()
        return

    # ── Load model ──
    model, tokenizer, backend = load_model_for_profiling(args.model)

    # ── Mission 1: KL Profile ──
    prompt = args.prompt or DEFAULT_PROFILE_PROMPT
    profiler = KLProfiler(model, tokenizer, backend=backend)
    profile = profiler.profile(prompt, reasoning_effort=args.reasoning_effort)

    if args.plot:
        profiler.plot_kl_profile(profile, "kl_profile.png")

    # ── Mission 2: Expert-Skipping Generation ──
    if args.generate or args.benchmark:
        gen_user_prompt = "Explain quantum entanglement in simple terms."
        gen_prompt, _ = apply_harmony_template(
            tokenizer,
            user_prompt=gen_user_prompt,
            reasoning_effort=args.reasoning_effort,
        )
        if backend == "mlx" and MLXDynamicExpertEngine is not None:
            mlx_engine = MLXDynamicExpertEngine(
                model=model,
                tokenizer=tokenizer,
                config=GptOssConfig(),
                soft_skip_start_layer=args.soft_skip_layer,
                hard_exit_layer=args.exit_layer,
                entropy_threshold=args.entropy_threshold,
                memory_limit_gb=args.memory_limit_gb,
            )

            if args.benchmark:
                protocol = mlx_engine.benchmark_protocol_comparison(
                    prompt_tokens=2048,
                    generation_tokens=128,
                    num_trials=1,
                )
                baseline = protocol["baseline"]
                transcender = protocol["transcender"]

                print(f"\n  {'═' * 60}")
                print(f"  mlx_lm Benchmark Protocol (2048 prefill / 128 gen)")
                print(f"  {'═' * 60}")
                print(f"  Baseline prompt TPS:   {baseline['prompt_tps']:.2f}")
                print(f"  Baseline gen TPS:      {baseline['generation_tps']:.2f}")
                print(f"  Baseline peak memory:  {baseline['peak_memory_gb']:.2f} GB")
                print(f"  {'─' * 60}")
                print(f"  Transcender prompt TPS:{transcender['prompt_tps']:.2f}")
                print(f"  Transcender gen TPS:   {transcender['generation_tps']:.2f}")
                print(f"  Transcender peak mem:  {transcender['peak_memory_gb']:.2f} GB")
                print(f"  Transcender avg layers:{transcender['avg_layers']:.2f}")
                print(f"  Prompt TPS gain:       {protocol['prompt_tps_gain_pct']:.1f}%")
                print(f"  Gen TPS gain:          {protocol['generation_tps_gain_pct']:.1f}%")
                print(f"  {'═' * 60}")

            if args.verify:
                verification = mlx_engine.benchmark_against_full_depth(
                    gen_prompt,
                    max_new_tokens=48,
                )
                baseline = verification["baseline"]
                dynamic = verification["dynamic"]

                print(f"\n  {'═' * 60}")
                print(f"  Dynamic Expert-Skipping Verification")
                print(f"  {'═' * 60}")
                print(f"  Baseline TTFT: {baseline['ttft_s']:.3f}s")
                print(f"  Dynamic TTFT:  {dynamic['ttft_s']:.3f}s")
                print(f"  Improvement:   {verification['ttft_improvement_pct']:.1f}%")
                print(f"  Prefix match:  {verification['prefix_match_tokens']} tokens")
                print(f"  Token match:   {verification['exact_match_rate']*100:.1f}%")
                print(f"  Peak memory:   {dynamic['peak_memory_gb']:.2f} GB")
                print(f"  Avg layers:    {dynamic['avg_layers']:.2f}")
                print(f"  {'─' * 60}")
                print(f"  Dynamic output: {dynamic['output_text'][:200]}")
                print(f"  {'═' * 60}")
            else:
                stats = mlx_engine.generate(
                    gen_prompt,
                    max_new_tokens=100,
                    dynamic=True,
                )
                print(f"\n  {'═' * 60}")
                print(f"  Dynamic Expert-Skipping Generation")
                print(f"  {'═' * 60}")
                print(f"  TTFT:        {stats['ttft_s']:.3f}s")
                print(f"  Throughput:  {stats['tokens_per_s']:.2f} tok/s")
                print(f"  Peak memory: {stats['peak_memory_gb']:.2f} GB")
                print(f"  Avg layers:  {stats['avg_layers']:.2f}")
                print(f"  {'─' * 60}")
                print(f"  Output: {stats['output_text'][:200]}")
                print(f"  {'═' * 60}")
        elif args.generate:
            engine = ExpertSkippingEngine(model, tokenizer)
            stats = engine.generate_with_expert_skipping(
                gen_prompt,
                max_new_tokens=100,
                exit_after_layer=args.exit_layer,
            )
        elif args.benchmark:
            print("\n  mlx_lm benchmark protocol is only available on the MLX backend.")

    print(f"\n  Reconnaissance complete.")
    print(f"  Optimal exit layer: {profile['optimal_exit']}")
    print(f"  Next: train Son Router with KL targets from this profile.")


if __name__ == "__main__":
    main()
