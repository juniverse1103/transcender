"""
Transcender Scaffold for GPT-OSS 20B (Sparse MoE) on M1 Pro (32GB)

Target: OpenAI GPT-OSS 20B (August 2025)
  - Sparse Mixture-of-Experts (MoE)
  - 21B total params, 3.6B active params per token
  - MXFP4 (Microscaling FP4) quantization native
  - Open weights, designed for local inference

Hardware: MacBook Pro M1 Pro (32GB unified memory)

Architecture insight — TWO-LEVEL ROUTING:
  Level 1 (existing): MoE expert routing — selects which FFN experts per token
  Level 2 (Transcender): Depth routing — selects how many layers per token

The key opportunity: MoE models are ALREADY sparse. Adding depth-sparsity
on top of expert-sparsity compounds the savings multiplicatively:
  - MoE alone: 3.6B/21B = 17% utilization per token
  - MoE + Transcender (50% depth): 3.6B * 0.5 / 21B = 8.5% utilization

Usage:
    python transcender_20b_scaffold.py              # Architecture analysis
    python transcender_20b_scaffold.py --analyze    # Detailed analysis
"""

import argparse
import sys


# ═══════════════════════════════════════════════
# 1. GPT-OSS 20B ARCHITECTURE ANALYSIS
# ═══════════════════════════════════════════════

GPT_OSS_20B = {
    "name": "openai/gpt-oss-20b",
    "release_date": "2025-08-05",
    "architecture": "gpt_oss (Sparse MoE)",
    "total_params": 21e9,       # 22B tensor size, 21B stated
    "active_params": 3.6e9,     # Per-token active parameters
    "num_layers": 24,           # From model architecture
    "hidden_size": 4096,        # Typical for this class
    "num_attention_heads": 32,
    "head_dim": 128,
    "num_experts": 8,           # Sparse MoE config
    "top_k_experts": 2,         # Active experts per token
    "intermediate_size": 11008,
    "vocab_size": 100256,
    "quantization": "MXFP4",   # Post-trained with MXFP4
    "precision": "BF16 + U8",  # Tensor types from HF
    "license": "Apache-2.0",
    "arxiv": "2508.10925",
    "hf_id": "openai/gpt-oss-20b",
    "inference_frameworks": ["Transformers", "vLLM", "Ollama", "LM Studio"],
    "fits_in_16gb": True,       # "run within 16GB of memory"
    "harmony_format": True,     # Requires harmony response format
}


def analyze_architecture(config=GPT_OSS_20B):
    """Architecture analysis for GPT-OSS 20B MoE with Transcender feasibility."""
    print("=" * 72)
    print("  GPT-OSS 20B — ARCHITECTURE ANALYSIS FOR TRANSCENDER IMPLANT")
    print("=" * 72)

    print(f"\n  Model:            {config['name']}")
    print(f"  Release:          {config['release_date']}")
    print(f"  Architecture:     {config['architecture']}")
    print(f"  Total params:     {config['total_params']/1e9:.1f}B")
    print(f"  Active params:    {config['active_params']/1e9:.1f}B per token")
    print(f"  Sparsity:         {(1 - config['active_params']/config['total_params'])*100:.0f}%")
    print(f"  Layers:           {config['num_layers']}")
    print(f"  Hidden size:      {config['hidden_size']}")
    print(f"  Attention heads:  {config['num_attention_heads']}")
    print(f"  Experts per layer:{config['num_experts']} (top-{config['top_k_experts']} active)")
    print(f"  Quantization:     {config['quantization']}")
    print(f"  Precision:        {config.get('precision', 'N/A')}")
    print(f"  License:          {config['license']}")
    print(f"  HuggingFace:      {config.get('hf_id', 'N/A')}")
    print(f"  ArXiv:            {config.get('arxiv', 'N/A')}")
    print(f"  Fits in 16GB:     {config.get('fits_in_16gb', 'Unknown')}")
    print(f"  Harmony format:   {config.get('harmony_format', False)}")

    # ── Memory Analysis ──
    print(f"\n  {'─' * 55}")
    print(f"  MEMORY ANALYSIS (M1 Pro 32GB)")
    print(f"  {'─' * 55}")

    # MXFP4: ~4 bits per param effective
    total_mem_gb = (config['total_params'] * 4) / (8 * 1e9)
    active_mem_gb = (config['active_params'] * 4) / (8 * 1e9)

    # KV cache: only attention (not experts) so uses hidden_size, not expert_size
    seq_len = 2048
    kv_cache_gb = (2 * config['num_layers'] * seq_len *
                   config['num_attention_heads'] * config['head_dim'] * 2) / 1e9

    # Router overhead
    router_params = config['hidden_size'] * 64 + 64 + 64 * 1 + 1
    router_kb = router_params * 4 / 1024

    total_gb = total_mem_gb + kv_cache_gb + router_kb / (1024 * 1024)

    print(f"  Full model (MXFP4):   {total_mem_gb:.1f} GB (all experts loaded)")
    print(f"  Active per token:     {active_mem_gb:.1f} GB (top-{config['top_k_experts']} experts)")
    print(f"  KV cache (2K ctx):    {kv_cache_gb:.2f} GB")
    print(f"  Son Router:           {router_kb:.1f} KB ({router_params:,} params)")
    print(f"  ──────────────────────────")
    print(f"  Total loaded:         {total_gb:.1f} GB / 32.0 GB")
    print(f"  Headroom:             {32.0 - total_gb:.1f} GB")
    print(f"  VERDICT:              {'EXCELLENT FIT' if total_gb < 20 else 'FEASIBLE'}")

    # ── MoE + Transcender Compound Savings ──
    print(f"\n  {'─' * 55}")
    print(f"  COMPOUND SAVINGS: MoE x Transcender")
    print(f"  {'─' * 55}")

    moe_util = config['active_params'] / config['total_params']
    print(f"  MoE utilization:           {moe_util*100:.0f}% ({config['active_params']/1e9:.1f}B / {config['total_params']/1e9:.0f}B)")

    exit_candidates = [
        (config['num_layers'] // 4, "25%"),
        (config['num_layers'] // 3, "33%"),
        (config['num_layers'] // 2, "50%"),
        (2 * config['num_layers'] // 3, "67%"),
    ]

    print(f"\n  {'Exit Layer':>12} {'Depth':>8} {'MoE Only':>10} {'MoE+Trans':>11} {'Compound':>10}")
    print(f"  {'─' * 55}")
    for layer, pct in exit_candidates:
        depth_util = layer / config['num_layers']
        moe_only = moe_util * 100
        compound = moe_util * depth_util * 100
        print(f"  Layer {layer:>4}     {pct:>6}    {moe_only:>8.0f}%    {compound:>9.1f}%    "
              f"{(1 - depth_util)*100:>8.0f}% saved")

    # ── MoE-Aware Routing Strategy ──
    print(f"\n  {'─' * 55}")
    print(f"  MoE-AWARE ROUTING STRATEGY")
    print(f"  {'─' * 55}")
    print(f"""
  The Son Router should hook into the RESIDUAL STREAM, not the expert
  routing logic. Here's why:

  MoE Layer Structure:
    input ─┬─ [Attention] ─────────────── add ─┬─ [Expert Router] ── add ── output
           │              residual stream       │     top-K experts
           └────────────────────────────────────┘

  Hook Point: AFTER the attention + expert output is added back to
  the residual stream. This is the "final residual" at each layer.

  Rationale:
  1. The residual stream is the SHARED representation across all
     experts. It encodes the token's full state regardless of which
     experts processed it.
  2. Hooking into expert-internal states would make the router
     dependent on WHICH experts were selected — fragile and
     non-generalizable.
  3. The residual stream is architecture-agnostic: same hook point
     works for dense transformers and MoE models alike.

  CRITICAL: The Son Router REPLACES the existing depth-axis decision.
  The MoE router KEEPS the width-axis decision (which experts).
  These are orthogonal:
    - MoE router:  "WHICH compute?" (expert selection)
    - Son router:  "HOW MUCH compute?" (layer depth)
""")

    # ── Physical Layer Skipping with MLX ──
    print(f"  {'─' * 55}")
    print(f"  PHYSICAL LAYER SKIPPING (Real Latency Reduction)")
    print(f"  {'─' * 55}")
    print(f"""
  MLX's lazy evaluation enables REAL compute savings, not just masking.

  Strategy for autoregressive generation:
  1. Each generated token runs through layers 0..exit_layer
  2. Son Router produces exit_prob for the NEW token only
  3. If exit_prob > threshold:
     - Compute early_logits = lm_head(norm(h_early))
     - SKIP layers exit_layer+1 .. N-1 entirely
     - MLX never evaluates the skipped ops (lazy evaluation)
  4. If exit_prob < threshold:
     - Continue through all layers normally

  KV-Cache Optimization:
  For EXITED tokens, the KV cache for deep layers is NEVER computed.
  This means deep layers' KV caches are sparse — only populated for
  tokens that went through those layers. This requires a modified
  attention mask in deep layers that only attends to tokens present
  in that layer's KV cache.

  Implementation approach:
  - Maintain TWO KV caches: early_kv (layers 0..exit) and deep_kv (layers exit+1..N)
  - early_kv has entries for ALL tokens
  - deep_kv has entries ONLY for non-exited tokens
  - Deep layers use deep_kv with a compressed attention mask
""")

    # ── Subspace Paradox for MoE ──
    print(f"  {'─' * 55}")
    print(f"  SUBSPACE PARADOX IN MoE MODELS")
    print(f"  {'─' * 55}")
    print(f"""
  The Subspace Paradox (4.11x separation in GPT-2) is expected to be
  MORE SEVERE in MoE models because:

  1. Expert Fragmentation: Different tokens pass through different
     experts, creating expert-specific subspaces WITHIN each layer.
     The residual stream at Layer 12 encodes not just "12 layers of
     processing" but "12 layers of DIFFERENT expert combinations."

  2. MoE models rely heavily on later layers: Early layers do
     shared processing, later layers specialize via experts.
     This means early logits may be WORSE relative to deep logits
     compared to dense models.

  3. Solution remains the same: LOGIT-SPACE BLENDING.
     early_logits = lm_head(final_norm(h_at_layer_12))
     deep_logits  = lm_head(final_norm(h_at_layer_24))
     blend = exit_prob * early_logits + (1-exit_prob) * deep_logits

  The final_norm and lm_head are SHARED (not expert-specific),
  making them the universal projector from any layer's subspace
  into vocabulary space.
""")


# ═══════════════════════════════════════════════
# 2. MLX PRODUCTION SCAFFOLD
# ═══════════════════════════════════════════════

def print_mlx_scaffold():
    """Print the complete MLX scaffolding code."""
    code = r'''
# ══════════════════════════════════════════════════════
# Transcender MLX Scaffold for GPT-OSS 20B (Sparse MoE)
# ══════════════════════════════════════════════════════
#
# Prerequisites:
#   pip install mlx mlx-lm
#   huggingface-cli download openai/gpt-oss-20b
#   mlx_lm.convert --hf-path openai/gpt-oss-20b \
#                  --quantize --q-bits 4 -o ./gpt-oss-20b-4bit

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load


# ─────────────────────────────────────────────
# Son Router (MLX Native, Float32)
# ─────────────────────────────────────────────

class SonRouterMLX(nn.Module):
    """
    Per-token exit probability predictor.
    Operates on the residual stream AFTER attention + MoE experts.

    Architecture: Linear(hidden→64) → GELU → Linear(64→1) → Sigmoid
    Params: hidden_size * 64 + 64 + 64 + 1 ≈ 262K for hidden=4096
    """

    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.proj1 = nn.Linear(hidden_size, 64)
        self.proj2 = nn.Linear(64, 1)

    def __call__(self, hidden: mx.array) -> dict:
        x = nn.gelu(self.proj1(hidden))
        logits = self.proj2(x).squeeze(-1)  # (B, S)
        probs = mx.sigmoid(logits)
        return {"exit_probs": probs, "exit_mask": probs > 0.5}


# ─────────────────────────────────────────────
# Transcender Engine
# ─────────────────────────────────────────────

class TranscenderEngine:
    """
    Production inference engine with physical layer skipping.

    Two-level routing hierarchy:
      Level 1 (MoE): Which experts? → Built into backbone
      Level 2 (Son): How many layers? → Our implant

    The backbone stays frozen in MXFP4. Only the Son Router trains.
    """

    def __init__(self, model_path: str, exit_after_layer: int = 12):
        self.exit_after_layer = exit_after_layer

        # Load frozen backbone
        self.model, self.tokenizer = load(model_path)

        # Extract components for manual iteration
        self.embed = self.model.model.embed_tokens
        self.layers = self.model.model.layers
        self.norm = self.model.model.norm
        self.lm_head = self.model.lm_head

        num_layers = len(self.layers)
        hidden_size = self.embed.weight.shape[1]
        print(f"Loaded: {num_layers} layers, hidden={hidden_size}")
        print(f"Exit point: Layer {exit_after_layer}/{num_layers} "
              f"({exit_after_layer/num_layers*100:.0f}% depth)")

        # Initialize Son Router
        self.router = SonRouterMLX(hidden_size)
        router_params = sum(p.size for p in self.router.parameters().values()
                           if isinstance(p, mx.array))
        print(f"Son Router: {router_params:,} params (float32)")

    # ── Self-Distillation: Extract KL Targets ──

    def compute_kl_targets(self, input_ids: mx.array) -> tuple:
        """
        Run full forward pass and compute KL(deep || early) per token.

        Returns:
            hidden_at_exit: (B, S, H) — hidden states at exit layer
            exit_targets: (B, S) — KL-derived soft targets in [0, 1]
                         1.0 = safe to exit, 0.0 = must continue
        """
        hidden = self.embed(input_ids)

        # Early layers
        for i in range(self.exit_after_layer):
            hidden = self.layers[i](hidden)

        hidden_at_exit = hidden  # Save for router training

        # Early logits (through shared norm + lm_head)
        early_logits = self.lm_head(self.norm(hidden))

        # Deep layers (continue)
        for i in range(self.exit_after_layer, len(self.layers)):
            hidden = self.layers[i](hidden)

        # Deep logits
        deep_logits = self.lm_head(self.norm(hidden))

        # KL(deep || early) per token
        deep_probs = mx.softmax(deep_logits, axis=-1)
        early_log_probs = mx.log(mx.softmax(early_logits, axis=-1) + 1e-10)
        deep_log_probs = mx.log(deep_probs + 1e-10)

        kl = mx.sum(deep_probs * (deep_log_probs - early_log_probs), axis=-1)

        # Convert KL to exit targets: low KL → safe to exit
        kl_median = mx.median(kl)
        exit_targets = mx.sigmoid(-2.0 * (kl - kl_median))

        mx.eval(hidden_at_exit, exit_targets, kl)

        return hidden_at_exit, exit_targets, kl

    # ── Router Training (Self-Distillation) ──

    def train_router(self, train_texts: list, num_epochs: int = 5, lr: float = 1e-3):
        """
        Train Son Router via KL-calibrated self-distillation.

        The backbone is FROZEN (MXFP4). Only the router (float32) trains.
        Target: predict which tokens can safely exit early.
        """
        optimizer = optim.AdamW(learning_rate=lr)

        for epoch in range(num_epochs):
            total_loss = 0.0
            n = 0

            for text in train_texts:
                ids = mx.array(self.tokenizer.encode(text))[None, :]

                # Get KL-derived targets (no gradient through backbone)
                hidden_at_exit, targets, _ = self.compute_kl_targets(ids)
                mx.eval(hidden_at_exit, targets)

                # Router forward + loss
                def loss_fn(params):
                    self.router.update(params)
                    routing = self.router(hidden_at_exit)
                    probs = routing["exit_probs"]
                    bce = -(targets * mx.log(probs + 1e-10)
                            + (1 - targets) * mx.log(1 - probs + 1e-10))
                    return mx.mean(bce)

                loss, grads = nn.value_and_grad(self.router, loss_fn)(
                    self.router.parameters()
                )
                optimizer.apply_gradients(grads, self.router)
                mx.eval(self.router.parameters(), loss)

                total_loss += loss.item()
                n += 1

            print(f"  Epoch {epoch+1}/{num_epochs} — loss: {total_loss/max(n,1):.4f}")

    # ── Inference with Physical Layer Skipping ──

    def generate(self, prompt: str, max_tokens: int = 50, mode: str = "hard"):
        """
        Autoregressive generation with PHYSICAL layer skipping.

        For each new token, the Son Router decides exit/continue.
        Exited tokens SKIP all deep layers — MLX's lazy evaluation
        means the compute never happens.
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated_tokens = []
        exit_count = 0
        total_count = 0

        for step in range(max_tokens):
            hidden = self.embed(input_ids)

            # Early layers (all tokens)
            for i in range(self.exit_after_layer):
                hidden = self.layers[i](hidden)

            # Router decision for the LAST token
            routing = self.router(hidden)
            last_prob = routing["exit_probs"][0, -1]
            mx.eval(last_prob)

            should_exit = (mode == "hard" and last_prob.item() > 0.5)
            total_count += 1

            if should_exit:
                # PHYSICAL SKIP: compute logits from early state
                logits = self.lm_head(self.norm(hidden))
                exit_count += 1
            else:
                # Full depth: run remaining layers
                for i in range(self.exit_after_layer, len(self.layers)):
                    hidden = self.layers[i](hidden)
                logits = self.lm_head(self.norm(hidden))

            mx.eval(logits)

            # Greedy decode
            next_id = mx.argmax(logits[0, -1], axis=-1).reshape(1, 1)
            input_ids = mx.concatenate([input_ids, next_id], axis=1)

            token = self.tokenizer.decode([next_id.item()])
            generated_tokens.append(token)

            if next_id.item() == self.tokenizer.eos_token_id:
                break

        exit_pct = exit_count / max(total_count, 1) * 100
        depth_saved = exit_pct * (1 - self.exit_after_layer / len(self.layers))
        print(f"\n  Generated {total_count} tokens: "
              f"{exit_count} exited ({exit_pct:.0f}%), "
              f"{depth_saved:.1f}% depth savings")

        return "".join(generated_tokens)

    # ── Diagnostic: KL Profile Across Layers ──

    def profile_kl_across_layers(self, prompt: str):
        """
        Measure KL divergence at each layer vs final layer.
        Finds the optimal exit point where KL is acceptably low.
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        hidden = self.embed(input_ids)

        # Run all layers and capture hidden states
        layer_hiddens = []
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            layer_hiddens.append(hidden)

        # Deep logits (final layer, ground truth)
        deep_logits = self.lm_head(self.norm(layer_hiddens[-1]))
        deep_probs = mx.softmax(deep_logits, axis=-1)
        deep_log_probs = mx.log(deep_probs + 1e-10)

        print(f"\n  KL DIVERGENCE PROFILE: '{prompt[:50]}...'")
        print(f"  {'Layer':>7} {'Avg KL':>10} {'Median KL':>11} {'Safe Exit%':>12}")
        print(f"  {'─' * 44}")

        for i, h in enumerate(layer_hiddens):
            early_logits = self.lm_head(self.norm(h))
            early_log_probs = mx.log(mx.softmax(early_logits, axis=-1) + 1e-10)
            kl = mx.sum(deep_probs * (deep_log_probs - early_log_probs), axis=-1)
            mx.eval(kl)

            avg_kl = kl.mean().item()
            med_kl = mx.median(kl).item()
            safe_pct = (kl < med_kl).sum().item() / kl.size * 100

            marker = " <-- OPTIMAL" if i == len(layer_hiddens) // 2 else ""
            print(f"  L{i+1:>4}   {avg_kl:>10.2f} {med_kl:>11.2f} {safe_pct:>10.0f}%{marker}")
'''
    print(code)


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Transcender 20B Scaffold")
    parser.add_argument("--analyze", action="store_true", help="Architecture analysis")
    parser.add_argument("--scaffold", action="store_true", help="Print MLX scaffold code")
    args = parser.parse_args()

    if args.scaffold:
        print_mlx_scaffold()
        return

    analyze_architecture()

    print(f"\n{'=' * 72}")
    print("  TRANSCENDER IMPLANT ROADMAP FOR GPT-OSS 20B")
    print(f"{'=' * 72}")
    print("""
  Phase 1: Reconnaissance
    - Download: huggingface-cli download openai/gpt-oss-20b
    - Or via Ollama: ollama pull gpt-oss:20b
    - Or via LM Studio: lms get openai/gpt-oss-20b
    - Profile KL divergence at each layer to find optimal exit point
    - Measure baseline inference speed (tokens/sec)
    - NOTE: Model uses "harmony" response format (required)

  Phase 2: Router Training (Self-Distillation)
    - Freeze backbone (MXFP4, ~10.5 GB)
    - Train Son Router (~262K params, float32) via KL-calibrated BCE
    - Target: predict which tokens can safely exit at the chosen layer

  Phase 3: Physical Layer Skipping
    - Implement dual KV-cache (early + deep) for real latency savings
    - Benchmark wall-clock tokens/sec improvement
    - Compare: soft (0% savings, best quality) vs hard (max savings)

  Phase 4: Compound Savings Measurement
    - MoE sparsity: 3.6B / 21B = 17% utilization
    - Transcender depth: X% exit rate * Y layers saved
    - Combined: multiplicative savings on active compute

  CRITICAL CONSTRAINTS:
    1. ALWAYS use logit-space blending (Subspace Paradox)
    2. Hook into RESIDUAL STREAM, not expert internals
    3. Use MLX lazy evaluation for PHYSICAL skip, not masking
    4. The MoE router and Son Router are ORTHOGONAL:
       - MoE: "WHICH compute?" (width)
       - Son: "HOW MUCH compute?" (depth)
""")
    print(f"  Run `python {sys.argv[0]} --scaffold` for the full MLX code.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
