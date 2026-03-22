"""
Inference Mode Benchmark — Bridges the Soft-to-Hard Gate Gap

Compares three inference strategies against vanilla GPT-2:
  1. Hard Gate:    exit_prob > 0.5 → pure early logits (max savings, high PPL)
  2. Soft Gate:    weighted logit blend (no savings, training-matched PPL)
  3. Adaptive:     confident exits (>0.9) hard, rest soft (balanced)
  4. Vanilla:      Standard GPT-2 baseline

Outputs:
  - inference_comparison.png    3-way PPL + savings comparison chart
  - Console table for Chapter 4
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
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

try:
    from transcender import TranscenderModel
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transcender.model import TranscenderModel

SEQ_LEN = 256
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_TEST_TOKENS = 50_000


def load_test_chunks(tokenizer):
    print("  Loading wikitext-2 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n".join([line for line in dataset["text"] if line.strip()])
    token_ids = tokenizer.encode(full_text)[:MAX_TEST_TOKENS]
    chunks = [token_ids[i:i + SEQ_LEN] for i in range(0, len(token_ids) - SEQ_LEN, SEQ_LEN)]
    print(f"  {len(token_ids):,} tokens → {len(chunks)} chunks")
    return chunks


def evaluate_vanilla(tokenizer, chunks):
    """Vanilla GPT-2 baseline."""
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for chunk in tqdm(chunks, desc="  Vanilla", ncols=70):
        input_ids = torch.tensor([chunk], device=DEVICE)
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids)
            total_loss += out.loss.item() * (len(chunk) - 1)
            total_tokens += len(chunk) - 1
    avg_loss = total_loss / total_tokens
    del model
    return {"name": "Vanilla GPT-2", "ppl": math.exp(avg_loss), "loss": avg_loss,
            "avg_layers": 12.0, "savings": 0.0, "exit_rate": 0.0}


def evaluate_sga(tokenizer, chunks, inference_mode, threshold=0.5):
    """SGA model in a given inference mode."""
    model = TranscenderModel(
        model_name="gpt2", exit_after_layer=2,
        exit_threshold=threshold, inference_mode=inference_mode,
    )
    router_path = os.path.join(OUTPUT_DIR, "son_router_trained.pt")
    if os.path.exists(router_path):
        model.router.load_state_dict(torch.load(router_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    total_loss, total_tokens = 0.0, 0
    total_layer_passes, total_possible = 0, 0
    total_exited, total_tok_count = 0, 0

    for chunk in tqdm(chunks, desc=f"  SGA-{inference_mode}", ncols=70):
        input_ids = torch.tensor([chunk], device=DEVICE)
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids)
            total_loss += out["lm_loss"].item() * (len(chunk) - 1)
            total_tokens += len(chunk) - 1
            total_layer_passes += out["layer_counts"].sum().item()
            total_possible += len(chunk) * len(model.blocks)

            # Count exited tokens
            if inference_mode == "hard":
                total_exited += out["routing_info"]["exit_mask"].sum().item()
            elif inference_mode == "adaptive":
                total_exited += (out["routing_info"]["exit_probs"] > 0.9).sum().item()
            else:
                total_exited += 0  # soft mode: no actual exits
            total_tok_count += len(chunk)

    avg_loss = total_loss / total_tokens
    savings = (1 - total_layer_passes / total_possible) * 100
    exit_rate = (total_exited / total_tok_count) * 100 if total_tok_count > 0 else 0

    mode_labels = {"hard": "SGA Hard-Gate", "soft": "SGA Soft-Gate", "adaptive": "SGA Adaptive"}
    del model
    return {
        "name": mode_labels[inference_mode],
        "ppl": math.exp(avg_loss), "loss": avg_loss,
        "avg_layers": total_layer_passes / total_tok_count,
        "savings": savings, "exit_rate": exit_rate,
    }


def plot_comparison(results):
    """Generate the 3-way comparison figure for Chapter 4."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    names = [r["name"] for r in results]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    # Panel 1: Perplexity
    ppls = [r["ppl"] for r in results]
    bars1 = axes[0].bar(names, ppls, color=colors[:len(results)], edgecolor="gray")
    axes[0].set_ylabel("Perplexity (PPL) ↓", fontsize=11)
    axes[0].set_title("Quality", fontsize=13, fontweight="bold")
    for bar, val in zip(bars1, ppls):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=20, labelsize=9)

    # Panel 2: Compute Savings
    savings = [r["savings"] for r in results]
    bars2 = axes[1].bar(names, savings, color=colors[:len(results)], edgecolor="gray")
    axes[1].set_ylabel("Compute Savings (%) ↑", fontsize=11)
    axes[1].set_title("Efficiency", fontsize=13, fontweight="bold")
    for bar, val in zip(bars2, savings):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=20, labelsize=9)

    # Panel 3: Avg Layers per Token
    layers = [r["avg_layers"] for r in results]
    bars3 = axes[2].bar(names, layers, color=colors[:len(results)], edgecolor="gray")
    axes[2].set_ylabel("Avg Layers / Token", fontsize=11)
    axes[2].set_title("Depth Utilization", fontsize=13, fontweight="bold")
    axes[2].set_ylim(0, 13)
    axes[2].axhline(y=12, color="gray", linestyle="--", alpha=0.5, label="Full depth")
    for bar, val in zip(bars3, layers):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[2].tick_params(axis="x", rotation=20, labelsize=9)
    axes[2].legend(fontsize=8)

    plt.suptitle(
        "SGA Inference Mode Comparison — Bridging the Soft-to-Hard Gate Gap",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "inference_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {path}")


def plot_pareto(results):
    """Plot the PPL vs Savings Pareto frontier across all modes."""
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {"Vanilla GPT-2": "#3498db", "SGA Hard-Gate": "#e74c3c",
              "SGA Soft-Gate": "#2ecc71", "SGA Adaptive": "#9b59b6"}
    markers = {"Vanilla GPT-2": "o", "SGA Hard-Gate": "X",
               "SGA Soft-Gate": "s", "SGA Adaptive": "*"}

    for r in results:
        ax.scatter(r["savings"], r["ppl"], s=200,
                   c=colors.get(r["name"], "gray"),
                   marker=markers.get(r["name"], "o"),
                   edgecolors="black", linewidths=0.5, zorder=5,
                   label=r["name"])
        ax.annotate(
            f"  PPL={r['ppl']:.1f}\n  Save={r['savings']:.1f}%",
            (r["savings"], r["ppl"]),
            textcoords="offset points", xytext=(15, 5),
            fontsize=8, color="gray",
        )

    ax.set_xlabel("Compute Savings (%)", fontsize=12)
    ax.set_ylabel("Perplexity (PPL) ↓", fontsize=12)
    ax.set_title("SGA Pareto Frontier — Quality vs Efficiency", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "pareto_frontier.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path}")


def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  SGA Inference Benchmark — Soft-to-Hard Gate Gap Analysis   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    chunks = load_test_chunks(tokenizer)

    # Run all four configurations
    results = []
    results.append(evaluate_vanilla(tokenizer, chunks))
    results.append(evaluate_sga(tokenizer, chunks, "soft"))
    results.append(evaluate_sga(tokenizer, chunks, "adaptive"))
    results.append(evaluate_sga(tokenizer, chunks, "hard"))

    # ── Results Table ──
    vanilla_ppl = results[0]["ppl"]
    print("\n" + "═" * 78)
    print("  INFERENCE MODE COMPARISON — Chapter 4 Data")
    print("═" * 78)
    print(f"  {'Mode':<20} {'PPL':>8} {'ΔPPL':>10} {'Savings':>9} {'AvgLayers':>10} {'ExitRate':>9}")
    print(f"  {'─' * 68}")
    for r in results:
        delta = ((r["ppl"] - vanilla_ppl) / vanilla_ppl) * 100
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        print(f"  {r['name']:<20} {r['ppl']:>8.1f} {delta_str:>10} {r['savings']:>8.1f}% "
              f"{r['avg_layers']:>10.2f} {r['exit_rate']:>8.1f}%")
    print("═" * 78)

    # Key finding
    soft = next(r for r in results if r["name"] == "SGA Soft-Gate")
    hard = next(r for r in results if r["name"] == "SGA Hard-Gate")
    adaptive = next(r for r in results if r["name"] == "SGA Adaptive")

    print(f"\n  KEY FINDINGS:")
    print(f"  • Soft-Gate PPL:     {soft['ppl']:.1f}  (Δ from vanilla: "
          f"{((soft['ppl'] - vanilla_ppl) / vanilla_ppl * 100):+.1f}%)")
    print(f"  • Hard-Gate PPL:     {hard['ppl']:.1f}  (Δ from vanilla: "
          f"{((hard['ppl'] - vanilla_ppl) / vanilla_ppl * 100):+.1f}%)")
    print(f"  • Adaptive PPL:      {adaptive['ppl']:.1f}  (Δ from vanilla: "
          f"{((adaptive['ppl'] - vanilla_ppl) / vanilla_ppl * 100):+.1f}%)")
    print(f"  • Soft-to-Hard gap:  {hard['ppl'] - soft['ppl']:.1f} PPL "
          f"({((hard['ppl'] - soft['ppl']) / soft['ppl'] * 100):.0f}% degradation)")

    # Generate figures
    plot_comparison(results)
    plot_pareto(results)

    print("\n  Done. Use these results for Chapter 4.")


if __name__ == "__main__":
    main()
