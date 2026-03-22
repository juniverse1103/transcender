"""
Train & Visualize — Trains the Son Router on WikiText-2 and generates
a Routing Activation Heatmap showing per-token layer depth.

Training strategy:
  - Freeze ALL GPT-2 parameters (embeddings, attention, FFN, LM head).
  - Only train the SonRouter's gate_proj MLP (~300 params).
  - Loss = LM loss (frozen backbone signal) + 0.1 × SonRoutingLoss.
  - The routing loss teaches the gate: "exit early when the model is
    already confident; route deeper when it's uncertain."

After training, we visualize routing on a complex prompt to confirm
that the router allocates variable compute depth per token.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer

try:
    from transcender import TranscenderModel
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transcender.model import TranscenderModel


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BATCH_SIZE = 4
SEQ_LEN = 128
NUM_EPOCHS = 5
LEARNING_RATE = 5e-4
MAX_TRAIN_SAMPLES = 1000  # More data for sharper routing decisions
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_dataset(tokenizer, max_samples=MAX_TRAIN_SAMPLES):
    """
    Load WikiText-2 and chunk it into fixed-length sequences for LM training.

    WikiText-2 is a collection of Wikipedia articles (~2M tokens).
    We concatenate all text, tokenize, then split into SEQ_LEN chunks.
    """
    print("[1/4] Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Concatenate all non-empty lines into one long string
    full_text = "\n".join([line for line in dataset["text"] if line.strip()])

    print(f"      Tokenizing {len(full_text):,} characters...")
    token_ids = tokenizer.encode(full_text)
    print(f"      Total tokens: {len(token_ids):,}")

    # Chunk into sequences of SEQ_LEN
    chunks = []
    for i in range(0, len(token_ids) - SEQ_LEN, SEQ_LEN):
        chunks.append(token_ids[i : i + SEQ_LEN])
        if len(chunks) >= max_samples:
            break

    print(f"      Created {len(chunks)} training chunks of {SEQ_LEN} tokens each")
    return chunks


def train_router(model, tokenizer, chunks):
    """
    Train ONLY the router parameters while keeping GPT-2 frozen.

    The key insight: we still run the full forward pass (all 12 layers)
    during training so the LM loss and entropy signals are accurate.
    The routing loss uses soft exit_probs (not hard exit_mask), so
    gradients flow through the router even though we compute all layers.
    """
    print("\n[2/4] Training Son Router...")
    print(f"      Device: {DEVICE}")
    print(f"      Trainable params: {sum(p.numel() for p in model.router.parameters()):,}")
    print(f"      Frozen params:    {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.router.parameters(), lr=LEARNING_RATE)

    model.train()
    model.to(DEVICE)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        total_routing_loss = 0.0
        total_lm_loss = 0.0
        num_batches = 0

        # Shuffle chunks each epoch
        indices = torch.randperm(len(chunks)).tolist()

        pbar = tqdm(
            range(0, len(indices) - BATCH_SIZE, BATCH_SIZE),
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            ncols=90,
        )

        for start in pbar:
            batch_indices = indices[start : start + BATCH_SIZE]
            batch = torch.tensor(
                [chunks[i] for i in batch_indices],
                dtype=torch.long,
                device=DEVICE,
            )

            # Labels = input_ids (standard causal LM: predict next token)
            output = model(input_ids=batch, labels=batch)

            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_routing_loss += output["routing_loss"].item()
            total_lm_loss += output["lm_loss"].item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "rout": f"{total_routing_loss / num_batches:.4f}",
                "lm": f"{total_lm_loss / num_batches:.4f}",
            })

        avg_loss = total_loss / max(num_batches, 1)
        avg_routing = total_routing_loss / max(num_batches, 1)
        print(f"      Epoch {epoch + 1} — avg loss: {avg_loss:.4f}, "
              f"routing loss: {avg_routing:.4f}")

    print("      Router training complete.")
    return model


def _run_prompt(model, tokenizer, prompt):
    """Run a prompt through the model and return routing data."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(input_ids=inputs["input_ids"])

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    display_tokens = [t.replace("Ġ", "·").replace("Ċ", "\\n") for t in tokens]

    return {
        "tokens": display_tokens,
        "layer_counts": output["layer_counts"][0].cpu().numpy(),
        "son_scores": output["routing_info"]["son_scores"][0].cpu().numpy(),
        "exit_probs": output["routing_info"]["exit_probs"][0].cpu().numpy(),
    }


def visualize_routing(model, tokenizer):
    """
    Generate a Routing Activation Heatmap comparing a simple vs complex prompt.

    The heatmap shows:
      - X-axis: each token in the prompt
      - Y-axis: layer index (0 to 11)
      - Color:  whether the token was processed at that layer
                (bright = processed, dark = skipped via early exit)
    """
    print("\n[3/4] Generating Routing Activation Heatmap...")

    model.eval()
    model.to(DEVICE)

    prompts = {
        "SIMPLE": "The cat sat on the mat and looked at the door",
        "COMPLEX": (
            "The epistemological implications of Gödel's incompleteness "
            "theorems on quantum mechanics are"
        ),
    }

    num_layers = len(model.blocks)
    results = {}
    for label, prompt in prompts.items():
        results[label] = _run_prompt(model, tokenizer, prompt)

    # ── Build combined figure: 2 rows of heatmaps + bar charts ──
    fig, axes = plt.subplots(
        4, 1,
        figsize=(18, 16),
        gridspec_kw={"height_ratios": [3, 1, 3, 1]},
    )

    for idx, (label, data) in enumerate(results.items()):
        ax_heat = axes[idx * 2]
        ax_bar = axes[idx * 2 + 1]

        tokens = data["tokens"]
        layer_counts = data["layer_counts"]
        exit_probs = data["exit_probs"]
        son_scores = data["son_scores"]

        # Build activation matrix
        activation_map = np.zeros((num_layers, len(tokens)))
        for tok_idx in range(len(tokens)):
            depth = int(layer_counts[tok_idx])
            activation_map[:depth, tok_idx] = 1.0

        # Heatmap
        sns.heatmap(
            activation_map,
            ax=ax_heat,
            cmap="YlOrRd",
            vmin=0, vmax=1,
            cbar_kws={"label": "Active / Skipped", "shrink": 0.5},
            xticklabels=tokens,
            yticklabels=[f"L{i}" for i in range(num_layers)],
            linewidths=0.5,
            linecolor="white",
        )
        prompt_text = prompts[label]
        total_possible = len(tokens) * num_layers
        actual = layer_counts.sum()
        savings = (1 - actual / total_possible) * 100
        ax_heat.set_title(
            f"[{label}] \"{prompt_text[:60]}...\"  —  {savings:.0f}% compute saved",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax_heat.set_ylabel("Layer", fontsize=10)
        ax_heat.tick_params(axis="x", rotation=45, labelsize=8)

        # Exit probability bars
        exit_colors = ["#2ecc71" if p > 0.5 else "#e74c3c" for p in exit_probs]
        ax_bar.bar(range(len(tokens)), exit_probs, color=exit_colors,
                   edgecolor="gray", linewidth=0.5)
        ax_bar.axhline(y=0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_bar.set_ylabel("Exit Prob", fontsize=9)
        ax_bar.set_xticks(range(len(tokens)))
        ax_bar.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax_bar.set_xlim(-0.5, len(tokens) - 0.5)
        ax_bar.set_ylim(0, 1)

    plt.suptitle(
        "SGA Routing Activation Map — Dynamic Compute Allocation",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "routing_heatmap.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Heatmap saved to: {output_path}")

    # ── Console Summary ──
    for label, data in results.items():
        tokens = data["tokens"]
        layer_counts = data["layer_counts"]
        son_scores = data["son_scores"]
        exit_probs = data["exit_probs"]

        print(f"\n{'─' * 65}")
        print(f"  [{label}] \"{prompts[label]}\"")
        print(f"{'─' * 65}")
        print(f"  {'Token':<22} {'Son':>7} {'Exit P':>7} {'Layers':>7} {'Exit?':>6}")
        print(f"  {'─' * 51}")
        for j, token in enumerate(tokens):
            print(
                f"  {token:<22} {son_scores[j]:>7.4f} {exit_probs[j]:>7.4f} "
                f"{int(layer_counts[j]):>7} {'  YES' if exit_probs[j] > 0.5 else '   NO':>6}"
            )

        total_possible = len(tokens) * num_layers
        actual = layer_counts.sum()
        savings = (1 - actual / total_possible) * 100
        print(f"\n  Compute: {actual:.0f}/{total_possible} layer-passes ({savings:.1f}% saved)")
    print(f"{'─' * 65}")


def main():
    print("=" * 65)
    print("  SGA — Son-Gated Architecture: Router Training & Visualization")
    print("=" * 65)

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TranscenderModel(model_name="gpt2", exit_after_layer=2, exit_threshold=0.5)

    # Freeze everything except the router
    for param in model.parameters():
        param.requires_grad = False
    for param in model.router.parameters():
        param.requires_grad = True

    # Prepare data
    chunks = prepare_dataset(tokenizer)

    # Train
    model = train_router(model, tokenizer, chunks)

    # Visualize
    visualize_routing(model, tokenizer)

    # Save router weights
    router_path = os.path.join(OUTPUT_DIR, "son_router_trained.pt")
    torch.save(model.router.state_dict(), router_path)
    print(f"\n[4/4] Router weights saved to: {router_path}")
    print("\n" + "=" * 65)
    print("  PoC complete. Inspect routing_heatmap.png for results.")
    print("=" * 65)


if __name__ == "__main__":
    main()
