"""
Layer Comparison Benchmark — The Symbiotic Implant Strategy

Trains Son Routers at Layer 2 and Layer 6, then benchmarks:
  1. Vanilla GPT-2          (baseline: 12 layers, 0% savings)
  2. Transcender-L2          (exit after layer 2: up to 83% savings)
  3. Transcender-L6          (exit after layer 6: up to 50% savings)

Each Transcender config is tested in soft, adaptive, and hard inference modes.

Outputs:
  - pareto_frontier_v2.png   Quality vs Efficiency Pareto frontier
  - layer_comparison.png     Bar chart comparison
  - son_router_l6_trained.pt Trained Layer-6 router weights
  - Console tables for the whitepaper
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

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SEQ_LEN = 256
TRAIN_SEQ_LEN = 128
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
MAX_TRAIN_SAMPLES = 2000
MAX_TEST_TOKENS = 50_000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════

def load_train_chunks(tokenizer):
    """Load WikiText-2 train split for router training."""
    print("  Loading WikiText-2 train split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n".join([line for line in dataset["text"] if line.strip()])
    token_ids = tokenizer.encode(full_text)
    chunks = []
    for i in range(0, len(token_ids) - TRAIN_SEQ_LEN, TRAIN_SEQ_LEN):
        chunks.append(token_ids[i : i + TRAIN_SEQ_LEN])
        if len(chunks) >= MAX_TRAIN_SAMPLES:
            break
    print(f"  {len(chunks)} training chunks of {TRAIN_SEQ_LEN} tokens")
    return chunks


def load_test_chunks(tokenizer):
    """Load WikiText-2 test split for evaluation."""
    print("  Loading WikiText-2 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n".join([line for line in dataset["text"] if line.strip()])
    token_ids = tokenizer.encode(full_text)[:MAX_TEST_TOKENS]
    chunks = [token_ids[i:i + SEQ_LEN] for i in range(0, len(token_ids) - SEQ_LEN, SEQ_LEN)]
    print(f"  {len(token_ids):,} tokens → {len(chunks)} test chunks")
    return chunks


# ═══════════════════════════════════════════════
# ROUTER TRAINING
# ═══════════════════════════════════════════════

def train_router(exit_after_layer, tokenizer, train_chunks, save_name,
                 routing_coeff=0.1):
    """Train a Son Router at a specific layer depth."""
    print(f"\n{'═' * 65}")
    print(f"  TRAINING: Son Router at Layer {exit_after_layer} (coeff={routing_coeff})")
    print(f"{'═' * 65}")

    model = TranscenderModel(
        model_name="gpt2",
        exit_after_layer=exit_after_layer,
        exit_threshold=0.5,
        inference_mode="soft",
        routing_coeff=routing_coeff,
    )

    # Freeze everything except the router
    for param in model.parameters():
        param.requires_grad = False
    for param in model.router.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.router.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Device: {DEVICE}")
    print(f"  Trainable params (router): {trainable:,}")
    print(f"  Frozen params (backbone):  {frozen:,}")

    optimizer = torch.optim.AdamW(model.router.parameters(), lr=LEARNING_RATE)
    model.train()
    model.to(DEVICE)

    for epoch in range(NUM_EPOCHS):
        total_loss, total_routing, total_lm, num_batches = 0, 0, 0, 0
        indices = torch.randperm(len(train_chunks)).tolist()

        pbar = tqdm(
            range(0, len(indices) - BATCH_SIZE, BATCH_SIZE),
            desc=f"  Epoch {epoch + 1}/{NUM_EPOCHS}",
            ncols=90,
        )

        for start in pbar:
            batch_idx = indices[start : start + BATCH_SIZE]
            batch = torch.tensor(
                [train_chunks[i] for i in batch_idx],
                dtype=torch.long, device=DEVICE,
            )
            output = model(input_ids=batch, labels=batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_routing += output["routing_loss"].item()
            total_lm += output["lm_loss"].item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "rout": f"{total_routing / num_batches:.4f}",
                "lm": f"{total_lm / num_batches:.4f}",
            })

        print(f"    Epoch {epoch + 1} — loss: {total_loss / num_batches:.4f}, "
              f"routing: {total_routing / num_batches:.4f}")

    # Save router weights
    save_path = os.path.join(OUTPUT_DIR, save_name)
    torch.save(model.router.state_dict(), save_path)
    print(f"  Router saved: {save_path}")

    del model
    torch.mps.empty_cache() if DEVICE == "mps" else None
    return save_path


# ═══════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════

def evaluate_vanilla(tokenizer, chunks):
    """Vanilla GPT-2 baseline."""
    print("\n  Evaluating: Vanilla GPT-2...")
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
    torch.mps.empty_cache() if DEVICE == "mps" else None
    return {"name": "Vanilla GPT-2", "ppl": math.exp(avg_loss), "loss": avg_loss,
            "avg_layers": 12.0, "savings": 0.0, "exit_rate": 0.0,
            "exit_layer": 12, "mode": "full"}


def evaluate_transcender(tokenizer, chunks, exit_after_layer, inference_mode,
                         router_path, threshold=0.5):
    """Evaluate a Transcender configuration."""
    label = f"Transcender-L{exit_after_layer}"
    mode_suffix = {"hard": "Hard", "soft": "Soft", "adaptive": "Adaptive"}
    full_name = f"{label} ({mode_suffix[inference_mode]})"
    print(f"\n  Evaluating: {full_name}...")

    model = TranscenderModel(
        model_name="gpt2",
        exit_after_layer=exit_after_layer,
        exit_threshold=threshold,
        inference_mode=inference_mode,
    )
    if os.path.exists(router_path):
        model.router.load_state_dict(
            torch.load(router_path, map_location=DEVICE, weights_only=True)
        )
    model.to(DEVICE)
    model.eval()

    total_loss, total_tokens = 0.0, 0
    total_layer_passes, total_possible = 0, 0
    total_exited, total_tok_count = 0, 0

    for chunk in tqdm(chunks, desc=f"  {full_name[:25]}", ncols=70):
        input_ids = torch.tensor([chunk], device=DEVICE)
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids)
            total_loss += out["lm_loss"].item() * (len(chunk) - 1)
            total_tokens += len(chunk) - 1
            total_layer_passes += out["layer_counts"].sum().item()
            total_possible += len(chunk) * len(model.blocks)

            if inference_mode == "hard":
                total_exited += out["routing_info"]["exit_mask"].sum().item()
            elif inference_mode == "adaptive":
                total_exited += (out["routing_info"]["exit_probs"] > 0.9).sum().item()
            total_tok_count += len(chunk)

    avg_loss = total_loss / total_tokens
    savings = (1 - total_layer_passes / total_possible) * 100
    exit_rate = (total_exited / total_tok_count) * 100 if total_tok_count > 0 else 0

    del model
    torch.mps.empty_cache() if DEVICE == "mps" else None
    return {
        "name": full_name,
        "ppl": math.exp(avg_loss), "loss": avg_loss,
        "avg_layers": total_layer_passes / total_tok_count,
        "savings": savings, "exit_rate": exit_rate,
        "exit_layer": exit_after_layer, "mode": inference_mode,
    }


# ═══════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════

def plot_comparison(results):
    """Bar chart comparison of all configurations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    names = [r["name"] for r in results]
    # Color by exit layer
    color_map = {
        "Vanilla GPT-2": "#95a5a6",
        "Transcender-L2": "#e74c3c",
        "Transcender-L6": "#2ecc71",
    }
    colors = []
    for r in results:
        for key, color in color_map.items():
            if key in r["name"]:
                colors.append(color)
                break
        else:
            colors.append("#3498db")

    # Panel 1: Perplexity
    ppls = [r["ppl"] for r in results]
    bars = axes[0].bar(range(len(names)), ppls, color=colors, edgecolor="gray")
    axes[0].set_ylabel("Perplexity (PPL) ↓", fontsize=11)
    axes[0].set_title("Quality", fontsize=13, fontweight="bold")
    for i, (bar, val) in enumerate(zip(bars, ppls)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)

    # Panel 2: Compute Savings
    savings = [r["savings"] for r in results]
    bars = axes[1].bar(range(len(names)), savings, color=colors, edgecolor="gray")
    axes[1].set_ylabel("Compute Savings (%) ↑", fontsize=11)
    axes[1].set_title("Efficiency", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, savings):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)

    # Panel 3: Avg Layers
    layers = [r["avg_layers"] for r in results]
    bars = axes[2].bar(range(len(names)), layers, color=colors, edgecolor="gray")
    axes[2].set_ylabel("Avg Layers / Token", fontsize=11)
    axes[2].set_title("Depth Utilization", fontsize=13, fontweight="bold")
    axes[2].set_ylim(0, 13)
    axes[2].axhline(y=12, color="gray", linestyle="--", alpha=0.5, label="Full depth")
    for bar, val in zip(bars, layers):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[2].legend(fontsize=8)

    plt.suptitle(
        "Transcender Layer Comparison — Symbiotic Implant Strategy",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "layer_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {path}")


def plot_pareto_v2(results):
    """Pareto frontier with L2 and L6 data points."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Auto-assign colors: L2=red family, L6=green family, vanilla=gray
    def get_style(name):
        if "Vanilla" in name:
            return "o", "#95a5a6", 250
        elif "L2" in name:
            if "Hard" in name: return "X", "#a93226", 200
            elif "Adaptive" in name: return "D", "#c0392b", 180
            else: return "s", "#e74c3c", 180
        elif "L6" in name:
            coeff = float(name.split("c=")[1].split(" ")[0]) if "c=" in name else 0.1
            intensity = min(1.0, 0.3 + coeff * 0.7)
            green = f"#{int(30*intensity):02x}{int(200*intensity):02x}{int(100*intensity):02x}"
            if "Hard" in name: return "X", green, 200
            elif "Adaptive" in name: return "D", green, 180
            else: return "s", green, 180
        return "o", "gray", 150

    for r in results:
        marker, color, size = get_style(r["name"])
        ax.scatter(
            r["savings"], r["ppl"], s=size,
            c=color, marker=marker,
            edgecolors="black", linewidths=0.5, zorder=5,
            label=r["name"],
        )
        # Annotate
        offset_x, offset_y = 12, 8
        ax.annotate(
            f"PPL={r['ppl']:.1f}\n{r['savings']:.1f}% saved",
            (r["savings"], r["ppl"]),
            textcoords="offset points", xytext=(offset_x, offset_y),
            fontsize=7, color="gray",
        )

    # Draw the Pareto frontier line (connecting best trade-off points)
    pareto_points = sorted(results, key=lambda r: r["savings"])
    frontier = []
    best_ppl = float("inf")
    for p in pareto_points:
        if p["ppl"] < best_ppl:
            frontier.append(p)
            best_ppl = p["ppl"]
    if len(frontier) > 1:
        ax.plot(
            [p["savings"] for p in frontier],
            [p["ppl"] for p in frontier],
            color="black", linestyle="--", alpha=0.3, linewidth=1.5,
            label="Pareto frontier",
        )

    ax.set_xlabel("Compute Savings (%)", fontsize=12)
    ax.set_ylabel("Perplexity (PPL) ↓", fontsize=12)
    ax.set_title(
        "Transcender Pareto Frontier — Quality vs Efficiency\n"
        "Layer 2 vs Layer 6 Symbiotic Implant",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "pareto_frontier_v2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path}")


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Transcender Benchmark — Symbiotic Implant Layer Comparison     ║")
    print("║  Layer 2 vs Layer 6: Routing Coefficient Sweep                  ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # ── Phase 1: Train routers at multiple routing coefficients ──
    train_chunks = load_train_chunks(tokenizer)

    # Train routers with KL-calibrated loss at both L2 and L6
    # Soft-blend training + KL-calibrated routing loss
    configs = [
        (2,  0.1, "son_router_l2_cal.pt"),
        (6,  0.1, "son_router_l6_cal.pt"),
        (6,  0.3, "son_router_l6_c03.pt"),
    ]
    routers = {}
    for layer, coeff, filename in configs:
        path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(path):
            path = train_router(layer, tokenizer, train_chunks, filename,
                                routing_coeff=coeff)
        else:
            print(f"  Router L{layer} (c={coeff}) already exists: {path}")
        routers[(layer, coeff)] = path

    # ── Phase 2: Benchmark all configurations ──
    test_chunks = load_test_chunks(tokenizer)

    results = []

    # Vanilla baseline
    results.append(evaluate_vanilla(tokenizer, test_chunks))

    # Transcender-L2 (coeff=0.1)
    for mode in ["soft", "adaptive", "hard"]:
        r = evaluate_transcender(
            tokenizer, test_chunks,
            exit_after_layer=2, inference_mode=mode,
            router_path=routers[(2, 0.1)],
        )
        r["name"] = f"T-L2 ({mode.title()})"
        results.append(r)

    # Transcender-L6 at two coefficients
    for coeff in [0.1, 0.3]:
        suffix = f" c={coeff}" if coeff != 0.1 else ""
        for mode in ["soft", "adaptive", "hard"]:
            r = evaluate_transcender(
                tokenizer, test_chunks,
                exit_after_layer=6, inference_mode=mode,
                router_path=routers[(6, coeff)],
            )
            r["name"] = f"T-L6{suffix} ({mode.title()})"
            results.append(r)

    # ── Phase 3: Results Table ──
    vanilla_ppl = results[0]["ppl"]
    print("\n" + "═" * 90)
    print("  TRANSCENDER LAYER COMPARISON — Symbiotic Implant Strategy")
    print("═" * 90)
    print(f"  {'Configuration':<28} {'PPL':>8} {'ΔPPL':>10} {'Savings':>9} "
          f"{'AvgLayers':>10} {'ExitRate':>9}")
    print(f"  {'─' * 78}")
    for r in results:
        delta = ((r["ppl"] - vanilla_ppl) / vanilla_ppl) * 100
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        print(f"  {r['name']:<28} {r['ppl']:>8.1f} {delta_str:>10} {r['savings']:>8.1f}% "
              f"{r['avg_layers']:>10.2f} {r['exit_rate']:>8.1f}%")
    print("═" * 90)

    # ── Key Findings: Best L6 configuration ──
    l6_results = [r for r in results if "L6" in r["name"] and "Hard" in r["name"]]
    l2_hard = next((r for r in results if "L2" in r["name"]), None)

    print(f"\n  KEY FINDINGS — Routing Coefficient Sweep:")
    print(f"  ─────────────────────────────────────────")
    for r in l6_results:
        print(f"  {r['name']:<28} PPL={r['ppl']:.1f}  Savings={r['savings']:.1f}%  "
              f"ExitRate={r['exit_rate']:.1f}%")
    if l2_hard:
        print(f"\n  L2 baseline: PPL={l2_hard['ppl']:.1f}, Savings={l2_hard['savings']:.1f}%")

    # Find best L6 hard config (lowest PPL with > 10% savings, or highest savings overall)
    best_l6 = None
    for r in l6_results:
        if best_l6 is None or (r["savings"] > 10 and r["ppl"] < best_l6.get("ppl", float("inf"))):
            best_l6 = r
    if best_l6 is None and l6_results:
        best_l6 = max(l6_results, key=lambda r: r["savings"])

    if best_l6:
        print(f"\n  BEST L6 CONFIG: {best_l6['name']}")
        print(f"    PPL={best_l6['ppl']:.1f}, Savings={best_l6['savings']:.1f}%")
        target_met = best_l6["ppl"] < 100 and best_l6["savings"] > 25
        if target_met:
            print(f"\n  *** TARGET MET: PPL < 100 with {best_l6['savings']:.1f}% savings ***")
        else:
            print(f"  Target: PPL {'MET' if best_l6['ppl'] < 100 else 'NOT MET (<100)'}, "
                  f"Savings {'MET' if best_l6['savings'] > 25 else 'NOT MET (>25%)'}")

    # ── Generate figures ──
    plot_comparison(results)
    plot_pareto_v2(results)

    print("\n  Benchmark complete. All figures saved.")


if __name__ == "__main__":
    main()
