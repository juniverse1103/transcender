"""
Model Injector — DEPRECATED. Use TranscenderModel from the transcender package.

This file exists only for backward compatibility with existing experiment scripts.
All new code should use:

    from transcender import TranscenderModel

SGAModel is a thin wrapper that delegates to TranscenderModel with GPT-2 defaults.
"""

import warnings
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

from transcender.model import TranscenderModel
from transcender.router import SonRouter, SonRoutingLoss


class SGAModel(nn.Module):
    """
    DEPRECATED: Use TranscenderModel directly.

    Backward-compatible wrapper that delegates to TranscenderModel.
    Preserves the original SGAModel interface (exit_after_layer=2 default,
    GPT-2 specific) while using the canonical implementation.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        exit_after_layer: int = 2,
        exit_threshold: float = 0.5,
        inference_mode: str = "hard",
        routing_coeff: float = 0.1,
    ):
        warnings.warn(
            "SGAModel is deprecated. Use TranscenderModel from the transcender package.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()

        self._model = TranscenderModel(
            model_name=model_name,
            exit_after_layer=exit_after_layer,
            exit_threshold=exit_threshold,
            inference_mode=inference_mode,
            routing_coeff=routing_coeff,
            attn_impl="eager",
        )

        # Expose attributes that experiment scripts access directly
        self.base_model = self._model.base_model
        self.config = self._model.config
        self.exit_after_layer = self._model.exit_after_layer
        self.router = self._model.router
        self.routing_loss_fn = self._model.routing_loss_fn
        self.blocks = self._model.blocks
        self.wte = self._model.embed_tokens
        self.wpe = self._model.embed_positions
        self.drop = self._model.embed_dropout
        self.ln_f = self._model.final_norm
        self.lm_head = self._model.lm_head
        self.inference_mode = self._model.inference_mode

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def train(self, mode=True):
        self._model.train(mode)
        return super().train(mode)

    def eval(self):
        self._model.eval()
        return super().eval()


def demo_routing():
    """Quick demonstration of SGA routing."""
    print("=" * 60)
    print("SGA (Son-Gated Architecture) — Routing Demo")
    print("=" * 60)

    print("\n[1] Loading GPT-2 with Son Router injected after layer 2...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = SGAModel(model_name="gpt2", exit_after_layer=2, exit_threshold=0.5)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    simple_prompt = "The cat sat on the"
    complex_prompt = "The epistemological implications of Gödel's incompleteness theorems on"

    for label, prompt in [("SIMPLE", simple_prompt), ("COMPLEX", complex_prompt)]:
        print(f"\n{'─' * 60}")
        print(f'  [{label}] "{prompt}"')
        print(f"{'─' * 60}")

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model(input_ids=inputs["input_ids"])

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        son_scores = output["routing_info"]["son_scores"][0]
        exit_probs = output["routing_info"]["exit_probs"][0]
        exit_mask = output["routing_info"]["exit_mask"][0]
        layer_counts = output["layer_counts"][0]

        print(f"\n  {'Token':<20} {'Son Score':>10} {'Exit Prob':>10} {'Layers':>8} {'Exit?':>6}")
        print(f"  {'─' * 56}")
        for j, token in enumerate(tokens):
            print(
                f"  {token:<20} {son_scores[j].item():>10.4f} "
                f"{exit_probs[j].item():>10.4f} {int(layer_counts[j].item()):>8} "
                f"{'  Y' if exit_mask[j].item() else '  N':>6}"
            )

        total_possible = len(tokens) * 12
        actual_layers = layer_counts.sum().item()
        savings = (1 - actual_layers / total_possible) * 100
        print(f"\n  FLOPs proxy: {actual_layers:.0f}/{total_possible} layer-passes "
              f"({savings:.1f}% saved)")

    print(f"\n{'=' * 60}")
    print("Demo complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    demo_routing()
