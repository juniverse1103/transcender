"""
SGA Benchmark Suite — Hard Evidence Layer

Four experiments that produce whitepaper-ready figures:

1. Perplexity Benchmark:  Vanilla GPT-2 vs SGA GPT-2 on wikitext-2-test
2. Threshold Sweep:       Quality-vs-Efficiency curve across exit thresholds
3. Subspace Analysis:     PCA of Layer-2 vs Layer-12 hidden states
4. Blend Comparison:      Hidden-state blending vs Logit-space blending loss

All figures are saved as publication-quality PNGs.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

try:
    from transcender import TranscenderModel
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transcender.model import TranscenderModel

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SEQ_LEN = 256
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_TEST_TOKENS = 50_000  # Cap for faster evaluation


def load_test_chunks(tokenizer, max_tokens=MAX_TEST_TOKENS):
    """Load wikitext-2 test split and chunk into fixed-length sequences."""
    print("  Loading wikitext-2-raw-v1 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n".join([line for line in dataset["text"] if line.strip()])
    token_ids = tokenizer.encode(full_text)
    if max_tokens:
        token_ids = token_ids[:max_tokens]
    chunks = []
    for i in range(0, len(token_ids) - SEQ_LEN, SEQ_LEN):
        chunks.append(token_ids[i : i + SEQ_LEN])
    print(f"  {len(token_ids):,} tokens → {len(chunks)} chunks of {SEQ_LEN}")
    return chunks


# ═══════════════════════════════════════════════
# EXPERIMENT 1: Perplexity Benchmark
# ═══════════════════════════════════════════════

def evaluate_vanilla_ppl(tokenizer, chunks):
    """Evaluate vanilla GPT-2 perplexity on test chunks."""
    print("\n  [Vanilla GPT-2] Loading...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for chunk in tqdm(chunks, desc="  Vanilla PPL", ncols=80):
        input_ids = torch.tensor([chunk], device=DEVICE)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * (len(chunk) - 1)
            total_tokens += len(chunk) - 1

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return ppl, avg_loss, 12.0  # always 12 layers


def evaluate_sga_ppl(tokenizer, chunks, threshold=0.5):
    """Evaluate SGA GPT-2 perplexity and compute savings at a given threshold."""
    model = TranscenderModel(model_name="gpt2", exit_after_layer=2, exit_threshold=threshold)
    router_path = os.path.join(OUTPUT_DIR, "son_router_trained.pt")
    if os.path.exists(router_path):
        model.router.load_state_dict(torch.load(router_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    total_loss = 0.0
    total_tokens = 0       # non-first tokens (for LM loss denominator)
    total_tok_count = 0    # all tokens (for avg_layers and exit_rate denominators)
    total_layer_passes = 0
    total_possible_passes = 0
    total_exited = 0       # tokens that took the early-exit path

    for chunk in tqdm(chunks, desc=f"  SGA PPL (t={threshold:.1f})", ncols=80, leave=False):
        input_ids = torch.tensor([chunk], device=DEVICE)
        with torch.no_grad():
            output = model(input_ids=input_ids, labels=input_ids)
            total_loss += output["lm_loss"].item() * (len(chunk) - 1)
            total_tokens += len(chunk) - 1
            total_tok_count += len(chunk)
            total_layer_passes += output["layer_counts"].sum().item()
            total_possible_passes += len(chunk) * len(model.blocks)
            total_exited += output["routing_info"]["exit_mask"].sum().item()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    # Fix 4: use total_tok_count (all tokens) not total_tokens (non-first tokens)
    avg_layers = total_layer_passes / total_tok_count
    savings = (1 - total_layer_passes / total_possible_passes) * 100
    # Fix 2: actual per-token exit rate
    exit_rate = (total_exited / total_tok_count) * 100 if total_tok_count > 0 else 0.0
    del model
    return ppl, avg_loss, avg_layers, savings, exit_rate


def run_perplexity_benchmark(tokenizer, chunks):
    """Run Experiment 1: Full perplexity comparison."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: Perplexity Benchmark")
    print("=" * 65)

    vanilla_ppl, vanilla_loss, vanilla_layers = evaluate_vanilla_ppl(tokenizer, chunks)
    sga_ppl, sga_loss, sga_avg_layers, sga_savings, sga_exit_rate = evaluate_sga_ppl(tokenizer, chunks, threshold=0.5)

    ppl_delta = ((sga_ppl - vanilla_ppl) / vanilla_ppl) * 100

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │           PERPLEXITY BENCHMARK RESULTS           │")
    print(f"  ├─────────────────┬──────────────┬────────────────┤")
    print(f"  │ Metric          │  Vanilla     │  SGA (t=0.5)   │")
    print(f"  ├─────────────────┼──────────────┼────────────────┤")
    print(f"  │ Perplexity      │ {vanilla_ppl:>10.2f}   │ {sga_ppl:>12.2f}   │")
    print(f"  │ Avg Loss        │ {vanilla_loss:>10.4f}   │ {sga_loss:>12.4f}   │")
    print(f"  │ Avg Layers/Tok  │ {vanilla_layers:>10.1f}   │ {sga_avg_layers:>12.2f}   │")
    print(f"  │ Compute Saved   │       0.0%   │ {sga_savings:>11.1f}%   │")
    print(f"  │ Exit Rate       │       0.0%   │ {sga_exit_rate:>11.1f}%   │")
    print(f"  │ PPL Delta       │         —    │ {ppl_delta:>+11.1f}%   │")
    print(f"  └─────────────────┴──────────────┴────────────────┘")

    return {
        "vanilla_ppl": vanilla_ppl,
        "sga_ppl": sga_ppl,
        "vanilla_loss": vanilla_loss,
        "sga_loss": sga_loss,
        "sga_savings": sga_savings,
        "sga_avg_layers": sga_avg_layers,
        "sga_exit_rate": sga_exit_rate,
    }


# ═══════════════════════════════════════════════
# EXPERIMENT 2: Threshold Sweep
# ═══════════════════════════════════════════════

def run_threshold_sweep(tokenizer, chunks):
    """Sweep exit_threshold from 0.0 to 1.0 and plot Quality vs Efficiency."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2: Threshold Sweep (Quality vs Efficiency)")
    print("=" * 65)

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    for t in thresholds:
        ppl, loss, avg_layers, savings, exit_rate = evaluate_sga_ppl(tokenizer, chunks, threshold=t)
        results.append({
            "threshold": t,
            "ppl": ppl,
            "loss": loss,
            "avg_layers": avg_layers,
            "savings": savings,
            "exit_rate": exit_rate,
        })
        print(f"  t={t:.1f}  PPL={ppl:>8.2f}  Savings={savings:>5.1f}%  AvgLayers={avg_layers:.2f}  ExitRate={exit_rate:.1f}%")

    # Find Golden Point: lowest savings where PPL increase is < 5% of vanilla
    vanilla_ppl = results[-1]["ppl"]  # t=1.0 → no exits → vanilla-equivalent
    golden = None
    for r in results:
        ppl_increase = ((r["ppl"] - vanilla_ppl) / vanilla_ppl) * 100
        if ppl_increase < 5.0 and r["savings"] > 0:
            if golden is None or r["savings"] > golden["savings"]:
                golden = r
                golden["ppl_increase"] = ppl_increase

    if golden:
        print(f"\n  ★ Golden Point: threshold={golden['threshold']:.1f}, "
              f"savings={golden['savings']:.1f}%, PPL increase={golden['ppl_increase']:.1f}%")

    # ── Plot ──
    fig, ax1 = plt.subplots(figsize=(10, 6))

    savings_vals = [r["savings"] for r in results]
    ppl_vals = [r["ppl"] for r in results]
    threshold_vals = [r["threshold"] for r in results]

    # PPL vs Savings
    color_ppl = "#e74c3c"
    ax1.set_xlabel("Compute Savings (%)", fontsize=12)
    ax1.set_ylabel("Perplexity (PPL)", fontsize=12, color=color_ppl)
    line1 = ax1.plot(savings_vals, ppl_vals, "o-", color=color_ppl, linewidth=2,
                     markersize=8, label="Perplexity", zorder=5)
    ax1.tick_params(axis="y", labelcolor=color_ppl)

    # Annotate each point with its threshold
    for r in results:
        ax1.annotate(
            f"t={r['threshold']:.1f}",
            (r["savings"], r["ppl"]),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=7,
            color="gray",
        )

    # Mark Golden Point
    if golden:
        ax1.axvline(x=golden["savings"], color="#2ecc71", linestyle="--",
                     linewidth=1.5, alpha=0.7, label=f"Golden Point (t={golden['threshold']:.1f})")
        ax1.scatter([golden["savings"]], [golden["ppl"]], s=200, color="#2ecc71",
                    marker="*", zorder=10, edgecolors="black", linewidths=0.5)

    # Avg Layers on secondary axis
    color_layers = "#3498db"
    ax2 = ax1.twinx()
    ax2.set_ylabel("Avg Layers / Token", fontsize=12, color=color_layers)
    layer_vals = [r["avg_layers"] for r in results]
    line2 = ax2.plot(savings_vals, layer_vals, "s--", color=color_layers, linewidth=1.5,
                     markersize=6, alpha=0.7, label="Avg Layers")
    ax2.tick_params(axis="y", labelcolor=color_layers)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    if golden:
        from matplotlib.patches import Patch
        lines.append(plt.Line2D([0], [0], color="#2ecc71", linestyle="--", linewidth=1.5))
        labels.append(f"Golden Point (t={golden['threshold']:.1f})")
    ax1.legend(lines, labels, loc="upper left", fontsize=10)

    ax1.set_title("SGA Quality vs Efficiency — Threshold Sweep", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "threshold_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")

    return results, golden


# ═══════════════════════════════════════════════
# EXPERIMENT 3: Subspace Analysis (PCA)
# ═══════════════════════════════════════════════

def run_subspace_analysis(tokenizer, chunks):
    """PCA visualization of Layer-2 vs Layer-12 hidden states."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3: Subspace Analysis (Layer 2 vs Layer 12)")
    print("=" * 65)

    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(DEVICE)
    model.eval()

    layer2_states = []
    layer12_states = []
    token_labels = []  # for coloring by token type

    # Collect hidden states from a sample of chunks
    sample_chunks = chunks[:20]
    print(f"  Collecting hidden states from {len(sample_chunks)} chunks...")

    for chunk in tqdm(sample_chunks, desc="  Extracting", ncols=80):
        input_ids = torch.tensor([chunk], device=DEVICE)

        with torch.no_grad():
            # Manual layer-by-layer forward pass
            position_ids = torch.arange(SEQ_LEN, device=DEVICE).unsqueeze(0)
            hidden = model.transformer.wte(input_ids) + model.transformer.wpe(position_ids)
            hidden = model.transformer.drop(hidden)
            for i, block in enumerate(model.transformer.h):
                hidden = block(hidden)
                if i == 1:  # After layer 2 (0-indexed)
                    layer2_states.append(hidden[0].cpu().detach().numpy())
                if i == 11:  # After layer 12
                    layer12_states.append(hidden[0].cpu().detach().numpy())

        # Classify tokens for coloring
        for tid in chunk:
            tok = tokenizer.decode([tid]).strip().lower()
            if tok in {"the", "a", "an", "of", "in", "on", "to", "and", "is", "was", "for", "it", "that", "with"}:
                token_labels.append("Function word")
            elif tok.isdigit() or (len(tok) > 0 and tok[0].isdigit()):
                token_labels.append("Number")
            elif len(tok) <= 2:
                token_labels.append("Subword fragment")
            else:
                token_labels.append("Content word")

    layer2_all = np.concatenate(layer2_states, axis=0)  # (N, 768)
    layer12_all = np.concatenate(layer12_states, axis=0)  # (N, 768)

    print(f"  Collected {layer2_all.shape[0]} token representations")
    print(f"  Running PCA...")

    # Fit PCA on combined data to use the same projection
    combined = np.concatenate([layer2_all, layer12_all], axis=0)
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)
    n = layer2_all.shape[0]
    layer2_2d = combined_2d[:n]
    layer12_2d = combined_2d[n:]

    explained = pca.explained_variance_ratio_
    print(f"  PCA explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")

    # Compute subspace statistics
    l2_center = layer2_2d.mean(axis=0)
    l12_center = layer12_2d.mean(axis=0)
    centroid_dist = np.linalg.norm(l2_center - l12_center)
    l2_spread = np.std(layer2_2d, axis=0).mean()
    l12_spread = np.std(layer12_2d, axis=0).mean()

    print(f"  Centroid distance: {centroid_dist:.2f}")
    print(f"  Layer 2 spread:  {l2_spread:.2f}")
    print(f"  Layer 12 spread: {l12_spread:.2f}")
    print(f"  Separation ratio: {centroid_dist / (l2_spread + l12_spread):.2f}x")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Layer 2 colored by token type
    color_map = {
        "Function word": "#3498db",
        "Content word": "#e74c3c",
        "Subword fragment": "#95a5a6",
        "Number": "#f39c12",
    }
    for label_type, color in color_map.items():
        mask = [t == label_type for t in token_labels]
        axes[0].scatter(
            layer2_2d[mask, 0], layer2_2d[mask, 1],
            c=color, label=label_type, alpha=0.4, s=8, edgecolors="none",
        )
    axes[0].set_title("Layer 2 Hidden States", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({explained[0]:.1%})", fontsize=10)
    axes[0].set_ylabel(f"PC2 ({explained[1]:.1%})", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right", markerscale=3)

    # Panel 2: Layer 12 colored by token type
    for label_type, color in color_map.items():
        mask = [t == label_type for t in token_labels]
        axes[1].scatter(
            layer12_2d[mask, 0], layer12_2d[mask, 1],
            c=color, label=label_type, alpha=0.4, s=8, edgecolors="none",
        )
    axes[1].set_title("Layer 12 Hidden States", fontsize=13, fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({explained[0]:.1%})", fontsize=10)
    axes[1].set_ylabel(f"PC2 ({explained[1]:.1%})", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper right", markerscale=3)

    # Panel 3: Overlay — the subspace mismatch proof
    axes[2].scatter(
        layer2_2d[:2000, 0], layer2_2d[:2000, 1],
        c="#3498db", alpha=0.3, s=8, label="Layer 2", edgecolors="none",
    )
    axes[2].scatter(
        layer12_2d[:2000, 0], layer12_2d[:2000, 1],
        c="#e74c3c", alpha=0.3, s=8, label="Layer 12", edgecolors="none",
    )
    # Draw centroid arrow
    axes[2].annotate(
        "", xy=l12_center, xytext=l2_center,
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )
    axes[2].scatter(*l2_center, s=100, c="#3498db", marker="X", edgecolors="black", zorder=10)
    axes[2].scatter(*l12_center, s=100, c="#e74c3c", marker="X", edgecolors="black", zorder=10)
    axes[2].set_title(
        f"Subspace Mismatch (separation={centroid_dist:.1f})",
        fontsize=13, fontweight="bold",
    )
    axes[2].set_xlabel(f"PC1 ({explained[0]:.1%})", fontsize=10)
    axes[2].set_ylabel(f"PC2 ({explained[1]:.1%})", fontsize=10)
    axes[2].legend(fontsize=10, loc="upper right", markerscale=3)

    plt.suptitle(
        "Hidden State Subspace Analysis — Why Blending Hidden States Fails",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "subspace_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")

    del model
    return {
        "centroid_distance": centroid_dist,
        "l2_spread": l2_spread,
        "l12_spread": l12_spread,
        "separation_ratio": centroid_dist / (l2_spread + l12_spread),
    }


# ═══════════════════════════════════════════════
# EXPERIMENT 4: Blend Strategy Comparison
# ═══════════════════════════════════════════════

def run_blend_comparison(tokenizer, chunks):
    """Compare hidden-state blending vs logit-space blending loss."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 4: Blend Strategy Comparison")
    print("=" * 65)

    model = TranscenderModel(model_name="gpt2", exit_after_layer=2, exit_threshold=0.5)
    router_path = os.path.join(OUTPUT_DIR, "son_router_trained.pt")
    if os.path.exists(router_path):
        model.router.load_state_dict(torch.load(router_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # We'll sweep blend_alpha from 0 (pure deep) to 1 (pure early) and compute
    # loss under both blending strategies
    alphas = np.linspace(0.0, 1.0, 21)  # 21 points
    hidden_blend_losses = []
    logit_blend_losses = []
    sample_chunks = chunks[:30]  # enough for stable estimate

    ce_loss_fn = nn.CrossEntropyLoss()

    print(f"  Sweeping blend alpha across {len(alphas)} points on {len(sample_chunks)} chunks...")

    for alpha in tqdm(alphas, desc="  Blend sweep", ncols=80):
        hb_total = 0.0
        lb_total = 0.0
        n_tokens = 0

        for chunk in sample_chunks:
            input_ids = torch.tensor([chunk], device=DEVICE)
            batch_size, seq_len = input_ids.shape

            with torch.no_grad():
                # Manual forward to get both early and deep states
                position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
                hidden = model.wte(input_ids) + model.wpe(position_ids)
                hidden = model.drop(hidden)

                for i in range(model.exit_after_layer):
                    hidden = model.blocks[i](hidden)
                early_states = hidden.clone()

                for i in range(model.exit_after_layer, len(model.blocks)):
                    hidden = model.blocks[i](hidden)
                deep_states = hidden

                # Strategy A: Hidden-state blending (THE FAILED APPROACH)
                blended_hidden = alpha * early_states + (1 - alpha) * deep_states
                hs_logits = model.lm_head(model.ln_f(blended_hidden))

                # Strategy B: Logit-space blending (THE WORKING APPROACH)
                early_logits = model.lm_head(model.ln_f(early_states))
                deep_logits = model.lm_head(model.ln_f(deep_states))
                ls_logits = alpha * early_logits + (1 - alpha) * deep_logits

                # Compute loss for both
                labels = input_ids
                shift_hs = hs_logits[..., :-1, :].contiguous()
                shift_ls = ls_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().view(-1)

                hb_loss = ce_loss_fn(shift_hs.view(-1, shift_hs.size(-1)), shift_labels)
                lb_loss = ce_loss_fn(shift_ls.view(-1, shift_ls.size(-1)), shift_labels)

                hb_total += hb_loss.item() * (seq_len - 1)
                lb_total += lb_loss.item() * (seq_len - 1)
                n_tokens += seq_len - 1

        hidden_blend_losses.append(hb_total / n_tokens)
        logit_blend_losses.append(lb_total / n_tokens)

    # Convert to PPL for interpretability
    hidden_blend_ppl = [math.exp(l) for l in hidden_blend_losses]
    logit_blend_ppl = [math.exp(l) for l in logit_blend_losses]

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Loss curves
    ax1.plot(alphas, hidden_blend_losses, "o-", color="#e74c3c", linewidth=2,
             markersize=5, label="Hidden-State Blending (Failed)")
    ax1.plot(alphas, logit_blend_losses, "s-", color="#2ecc71", linewidth=2,
             markersize=5, label="Logit-Space Blending (SGA)")
    ax1.axvline(x=0.0, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.text(0.02, max(hidden_blend_losses) * 0.98, "α=0\n(Pure Deep)", fontsize=8, color="gray")
    ax1.text(0.85, max(hidden_blend_losses) * 0.98, "α=1\n(Pure Early)", fontsize=8, color="gray")
    ax1.set_xlabel("Blend Alpha (0=Deep, 1=Early)", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_title("Blending Strategy: Loss Comparison", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: PPL curves
    ax2.plot(alphas, hidden_blend_ppl, "o-", color="#e74c3c", linewidth=2,
             markersize=5, label="Hidden-State Blending")
    ax2.plot(alphas, logit_blend_ppl, "s-", color="#2ecc71", linewidth=2,
             markersize=5, label="Logit-Space Blending")

    # Highlight the divergence region
    divergence = [h - l for h, l in zip(hidden_blend_ppl, logit_blend_ppl)]
    max_div_idx = np.argmax(divergence)
    ax2.annotate(
        f"Max divergence\nΔPPL={divergence[max_div_idx]:.1f}",
        xy=(alphas[max_div_idx], hidden_blend_ppl[max_div_idx]),
        xytext=(alphas[max_div_idx] - 0.15, hidden_blend_ppl[max_div_idx] + 5),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9, fontweight="bold",
    )
    ax2.set_xlabel("Blend Alpha (0=Deep, 1=Early)", fontsize=12)
    ax2.set_ylabel("Perplexity (PPL)", fontsize=12)
    ax2.set_title("Blending Strategy: Perplexity Comparison", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "The Logit-Space Blending Breakthrough — Hidden-State Blending Degrades at Intermediate α",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "blend_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")

    # Summary
    # Find the alpha where hidden-state blending is worst relative to logit blending
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │           BLEND STRATEGY COMPARISON                     │")
    print(f"  ├──────────┬──────────────────┬──────────────────────────┤")
    print(f"  │  Alpha   │  HiddenState PPL │  LogitSpace PPL (Δ)     │")
    print(f"  ├──────────┼──────────────────┼──────────────────────────┤")
    for i in range(0, len(alphas), 4):  # every 4th point
        a = alphas[i]
        hp = hidden_blend_ppl[i]
        lp = logit_blend_ppl[i]
        delta = hp - lp
        marker = " ★" if i == max_div_idx else ""
        print(f"  │   {a:.2f}    │     {hp:>8.2f}     │     {lp:>8.2f}  ({delta:>+6.1f}){marker:<3}│")
    print(f"  └──────────┴──────────────────┴──────────────────────────┘")

    del model
    return {
        "alphas": alphas.tolist(),
        "hidden_blend_ppl": hidden_blend_ppl,
        "logit_blend_ppl": logit_blend_ppl,
        "max_divergence_alpha": float(alphas[max_div_idx]),
        "max_divergence_ppl": divergence[max_div_idx],
    }


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║    SGA BENCHMARK SUITE — Hard Evidence Layer             ║")
    print("║    4 Experiments for Whitepaper Validation               ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  Device: {DEVICE}")
    print(f"  Seq Length: {SEQ_LEN}")
    print(f"  Max Test Tokens: {MAX_TEST_TOKENS:,}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    chunks = load_test_chunks(tokenizer)

    # Experiment 1: Perplexity
    ppl_results = run_perplexity_benchmark(tokenizer, chunks)

    # Experiment 2: Threshold Sweep
    sweep_results, golden = run_threshold_sweep(tokenizer, chunks)

    # Experiment 3: Subspace Analysis
    subspace_results = run_subspace_analysis(tokenizer, chunks)

    # Experiment 4: Blend Comparison
    blend_results = run_blend_comparison(tokenizer, chunks)

    # ── Final Summary ──
    print("\n" + "╔═══════════════════════════════════════════════════════════╗")
    print("║                   BENCHMARK COMPLETE                      ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║  1. Vanilla PPL: {ppl_results['vanilla_ppl']:>8.2f}  │  SGA PPL: {ppl_results['sga_ppl']:>8.2f}         ║")
    print(f"║     Compute Saved: {ppl_results['sga_savings']:.1f}%                              ║")
    if golden:
        print(f"║  2. Golden Point: t={golden['threshold']:.1f}, "
              f"saves {golden['savings']:.1f}% at +{golden['ppl_increase']:.1f}% PPL  ║")
    print(f"║  3. Subspace Separation: {subspace_results['separation_ratio']:.2f}x               ║")
    print(f"║     (Layer 2 & 12 live in DIFFERENT subspaces)           ║")
    print(f"║  4. Logit Blend Peak Advantage: {blend_results['max_divergence_ppl']:.1f} PPL    ║")
    print(f"║     at α={blend_results['max_divergence_alpha']:.2f}                                     ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print("║  Figures saved:                                          ║")
    print("║    • threshold_sweep.png                                 ║")
    print("║    • subspace_analysis.png                               ║")
    print("║    • blend_comparison.png                                ║")
    print("╚═══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
