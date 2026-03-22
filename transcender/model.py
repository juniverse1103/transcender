"""
Transcender Model — Surgical Implant for HuggingFace Causal LMs

Takes any HuggingFace causal language model and implants a Son-Gated router
with logit-blending at a configurable layer depth.

Supported architectures:
  - GPT-2 / GPT-Neo / GPT-J (transformer.h blocks)
  - LLaMA / Mistral / Qwen (model.layers blocks)

Usage:
    from transcender import TranscenderModel

    model = TranscenderModel("gpt2", exit_after_layer=6)
    model.freeze_backbone()

    output = model(input_ids=batch, labels=batch)
    loss = output["loss"]
    loss.backward()

    model.set_inference_mode("hard")
    model.eval()
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from transcender.router import SonRouter, SonRoutingLoss


class ArchitectureAdapter:
    """Extracts model components for different HF architectures."""

    @staticmethod
    def detect(model) -> str:
        class_name = model.__class__.__name__.lower()
        if "gpt2" in class_name or "gptneo" in class_name or "gptj" in class_name:
            return "gpt2"
        elif "llama" in class_name or "mistral" in class_name or "qwen" in class_name:
            return "llama"
        raise ValueError(
            f"Unsupported architecture: {model.__class__.__name__}. "
            f"Supported: GPT-2, GPT-Neo, GPT-J, LLaMA, Mistral, Qwen."
        )

    @staticmethod
    def get_components(model, arch: str) -> dict:
        if arch == "gpt2":
            transformer = model.transformer
            return {
                "embed_tokens": transformer.wte,
                "embed_positions": transformer.wpe,
                "embed_dropout": transformer.drop,
                "blocks": transformer.h,
                "final_norm": transformer.ln_f,
                "lm_head": model.lm_head,
                "has_position_embed": True,
            }
        elif arch == "llama":
            llama_model = model.model
            return {
                "embed_tokens": llama_model.embed_tokens,
                "embed_positions": None,
                "embed_dropout": None,
                "blocks": llama_model.layers,
                "final_norm": llama_model.norm,
                "lm_head": model.lm_head,
                "has_position_embed": False,
            }

    @staticmethod
    def get_attn_module(block, arch: str):
        if arch == "gpt2":
            return block.attn
        elif arch == "llama":
            return block.self_attn


class TranscenderModel(nn.Module):
    """
    Wraps any HuggingFace causal LM with Son-Gated early-exit routing.

    Args:
        model_name:       HuggingFace model identifier
        exit_after_layer: Layer after which the router decides on early exit.
        exit_threshold:   Probability threshold for early exit decisions.
        inference_mode:   "hard", "soft", or "adaptive".
        routing_coeff:    Weight of routing loss relative to LM loss.
        attn_impl:        Attention implementation ("eager" for weight capture).
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        exit_after_layer: int = 6,
        exit_threshold: float = 0.5,
        inference_mode: str = "hard",
        routing_coeff: float = 0.1,
        attn_impl: str = "eager",
    ):
        super().__init__()
        self.inference_mode = inference_mode
        self.routing_coeff = routing_coeff
        self.exit_after_layer = exit_after_layer

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation=attn_impl,
        )
        self.config = self.base_model.config
        hidden_size = self.config.hidden_size

        self.arch = ArchitectureAdapter.detect(self.base_model)
        components = ArchitectureAdapter.get_components(self.base_model, self.arch)

        self.embed_tokens = components["embed_tokens"]
        self.embed_positions = components["embed_positions"]
        self.embed_dropout = components["embed_dropout"]
        self.blocks = components["blocks"]
        self.final_norm = components["final_norm"]
        self.lm_head = components["lm_head"]
        self.has_position_embed = components["has_position_embed"]

        num_layers = len(self.blocks)
        if exit_after_layer < 1 or exit_after_layer >= num_layers:
            raise ValueError(
                f"exit_after_layer={exit_after_layer} must be in [1, {num_layers - 1}] "
                f"for a {num_layers}-layer model."
            )

        self.router = SonRouter(
            hidden_size=hidden_size,
            exit_threshold=exit_threshold,
        )
        self.routing_loss_fn = SonRoutingLoss()

        self._captured_attn_weights = None
        self._attn_hooks = []

    def freeze_backbone(self):
        """Freeze all parameters except the router (symbiotic mode)."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.router.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.router.parameters())
        total = sum(p.numel() for p in self.parameters())
        print(f"  Symbiotic mode: {trainable:,} trainable / {total:,} total params "
              f"({trainable / total * 100:.3f}%)")

    def set_inference_mode(self, mode: str):
        assert mode in ("hard", "soft", "adaptive"), f"Unknown mode: {mode}"
        self.inference_mode = mode

    def _register_attn_hook(self, layer_idx: int):
        def hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2:
                self._captured_attn_weights = output[1]
        attn_module = ArchitectureAdapter.get_attn_module(
            self.blocks[layer_idx], self.arch
        )
        handle = attn_module.register_forward_hook(hook_fn)
        self._attn_hooks.append(handle)

    def _remove_attn_hooks(self):
        for h in self._attn_hooks:
            h.remove()
        self._attn_hooks.clear()

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        if self.has_position_embed and self.embed_positions is not None:
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(
                seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
            hidden = hidden + self.embed_positions(position_ids)
        if self.embed_dropout is not None:
            hidden = self.embed_dropout(hidden)
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        num_layers = len(self.blocks)

        hidden_states = self._embed(input_ids)

        router_layer = self.exit_after_layer - 1
        self._register_attn_hook(router_layer)

        for i in range(self.exit_after_layer):
            hidden_states = self.blocks[i](
                hidden_states,
                attention_mask=attention_mask,
            )

        last_attn = self._captured_attn_weights
        self._remove_attn_hooks()
        self._captured_attn_weights = None

        if last_attn is None:
            num_heads = self.config.num_attention_heads
            last_attn = torch.ones(
                batch_size, num_heads, seq_len, seq_len, device=device,
            ) / seq_len

        routing_info = self.router(hidden_states, last_attn)
        exit_mask = routing_info["exit_mask"]
        exit_probs = routing_info["exit_probs"]

        early_exit_states = hidden_states.clone()

        layer_counts = torch.full(
            (batch_size, seq_len), self.exit_after_layer,
            device=device, dtype=torch.float,
        )

        # ALL tokens compute deep states for logit-space blending.
        # exit_mask controls which logits appear in the output.
        for i in range(self.exit_after_layer, num_layers):
            hidden_states = self.blocks[i](
                hidden_states, attention_mask=attention_mask,
            )
            layer_counts[~exit_mask] = i + 1

        early_logits = self.lm_head(self.final_norm(early_exit_states))
        deep_logits = self.lm_head(self.final_norm(hidden_states))

        if self.training:
            blend_weights = exit_probs.unsqueeze(-1)
            logits = blend_weights * early_logits + (1 - blend_weights) * deep_logits
        elif self.inference_mode == "soft":
            blend_weights = exit_probs.unsqueeze(-1)
            logits = blend_weights * early_logits + (1 - blend_weights) * deep_logits
        elif self.inference_mode == "adaptive":
            confident_mask = (exit_probs > 0.9)
            blend_weights = exit_probs.unsqueeze(-1)
            logits = blend_weights * early_logits + (1 - blend_weights) * deep_logits
            confident_expanded = confident_mask.unsqueeze(-1).expand_as(early_logits)
            logits = torch.where(confident_expanded, early_logits, logits)
            layer_counts[confident_mask] = self.exit_after_layer
        else:  # hard
            exit_mask_expanded = exit_mask.unsqueeze(-1).expand_as(early_logits)
            logits = torch.where(exit_mask_expanded, early_logits, deep_logits)

        result = {
            "logits": logits,
            "routing_info": routing_info,
            "layer_counts": layer_counts,
        }

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            routing_loss = self.routing_loss_fn(
                exit_probs,
                early_logits=early_logits.detach(),
                deep_logits=deep_logits.detach(),
            )
            result["loss"] = lm_loss + self.routing_coeff * routing_loss
            result["lm_loss"] = lm_loss
            result["routing_loss"] = routing_loss

        return result

    def get_routing_summary(self) -> dict:
        return {
            "architecture": self.arch,
            "model_class": self.base_model.__class__.__name__,
            "total_layers": len(self.blocks),
            "exit_after_layer": self.exit_after_layer,
            "max_savings_pct": (1 - self.exit_after_layer / len(self.blocks)) * 100,
            "router_params": sum(p.numel() for p in self.router.parameters()),
            "total_params": sum(p.numel() for p in self.parameters()),
            "inference_mode": self.inference_mode,
            "routing_coeff": self.routing_coeff,
        }
