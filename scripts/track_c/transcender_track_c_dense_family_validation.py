"""
Track C Extension — Focused Dense-Family Validation for Llama/Mistral MLX Models

This script runs two pieces of work for one dense decoder-only model:

  1. Recon / KL profile:
     - per-layer KL against final-layer logits on the canonical 5-prompt suite
     - heuristic plateau / resolution summary
     - Subspace Paradox separation ratio between a middle layer and the final layer

  2. Focused Track C-style benchmark:
     - full depth
     - one late fixed exit
     - one compute-both top1_agree mode
     - one real selective-depth entropy mode

The implementation is intentionally narrow:
  - no broad exit sweeps
  - no learned probes
  - no Track A / Track B code-path changes
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm import load as mlx_load
from mlx_lm.models.base import create_attention_mask

from transcender_engine import build_harmony_messages
from transcender_track_b_cascade import (
    PROMPTS,
    WARMUP_PROMPT_INDEX,
    apply_generic_chat_template,
    compare_sequences,
    mean,
    preview_text,
)

SYSTEM_PROMPT = "You are a helpful assistant."


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shift = np.max(x, axis=axis, keepdims=True)
    return x - shift - np.log(np.sum(np.exp(x - shift), axis=axis, keepdims=True))


def kl_divergence_np(deep_logits: np.ndarray, early_logits: np.ndarray) -> np.ndarray:
    deep_probs = softmax_np(deep_logits)
    deep_log = log_softmax_np(deep_logits)
    early_log = log_softmax_np(early_logits)
    return np.sum(deep_probs * (deep_log - early_log), axis=-1)


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, mx.array):
        return np.array(x.astype(mx.float32))
    return np.array(x)


@dataclass(frozen=True)
class ModeConfig:
    key: str
    label: str
    description: str
    kind: str
    exit_layer: Optional[int] = None
    entropy_threshold: Optional[float] = None


@dataclass
class LlamaLikeModelParts:
    model: Any
    tokenizer: Any
    model_id: str
    family: str
    num_layers: int
    hidden_size: int
    layers: Any
    embed_tokens: Any
    final_norm: Any
    tie_word_embeddings: bool
    fa_idx: int
    swa_idx: Optional[int]
    sliding_window: Optional[int]

    @classmethod
    def from_loaded(cls, model, tokenizer, model_id: str, family: str) -> "LlamaLikeModelParts":
        inner = model.model
        args = model.args
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            family=family,
            num_layers=len(inner.layers),
            hidden_size=int(args.hidden_size),
            layers=inner.layers,
            embed_tokens=inner.embed_tokens,
            final_norm=inner.norm,
            tie_word_embeddings=bool(getattr(args, "tie_word_embeddings", True)),
            fa_idx=int(getattr(inner, "fa_idx", 0)),
            swa_idx=getattr(inner, "swa_idx", None),
            sliding_window=getattr(inner, "sliding_window", None),
        )

    def embed(self, input_ids: mx.array) -> mx.array:
        return self.embed_tokens(input_ids)

    def compute_logits(self, hidden: mx.array) -> mx.array:
        normed = self.final_norm(hidden)
        if self.tie_word_embeddings:
            return self.embed_tokens.as_linear(normed)
        return self.model.lm_head(normed)


class LlamaLikeDenseEngine:
    """Focused adaptive-depth engine for Llama-style MLX dense models."""

    def __init__(self, parts: LlamaLikeModelParts):
        self.parts = parts
        self.model = parts.model
        self.tokenizer = parts.tokenizer
        self.layers = parts.layers
        self.num_layers = parts.num_layers
        self.eos_ids = set()
        eos_ids_attr = getattr(self.tokenizer, "eos_token_ids", None)
        if eos_ids_attr:
            self.eos_ids = set(int(x) for x in eos_ids_attr)
        elif hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            self.eos_ids = {int(self.tokenizer.eos_token_id)}

    def _build_masks(self, hidden: mx.array, cache: Optional[Any]) -> Tuple[Any, Any]:
        fa_cache = cache[self.parts.fa_idx] if cache is not None else None
        fa_mask = create_attention_mask(hidden, fa_cache)

        swa_mask = None
        if self.parts.swa_idx is not None and self.parts.sliding_window is not None:
            swa_cache = cache[self.parts.swa_idx] if cache is not None else None
            swa_mask = create_attention_mask(
                hidden,
                swa_cache,
                window_size=self.parts.sliding_window,
            )
        return fa_mask, swa_mask

    def _layer_mask(self, layer_idx: int, fa_mask: Any, swa_mask: Any) -> Any:
        if getattr(self.layers[layer_idx], "use_sliding", False) and swa_mask is not None:
            return swa_mask
        return fa_mask

    def _run_layers(
        self,
        hidden: mx.array,
        cache: Optional[Any],
        start_layer: int,
        end_layer: int,
    ) -> mx.array:
        fa_mask, swa_mask = self._build_masks(hidden, cache)
        for i in range(start_layer, end_layer):
            mask = self._layer_mask(i, fa_mask, swa_mask)
            hidden = self.layers[i](hidden, mask, cache[i] if cache is not None else None)
        mx.eval(hidden)
        return hidden

    def _confidence_probe(self, logits: mx.array) -> Dict[str, float | int]:
        last_logits = logits[0, -1].astype(mx.float32)
        probs = mx.softmax(last_logits, axis=-1)
        entropy = -mx.sum(probs * mx.log(probs + 1e-9), axis=-1)
        normalized_entropy = entropy / math.log(int(last_logits.shape[-1]))
        top2 = mx.topk(probs, 2, axis=-1)
        top2_prob = top2[..., 0]
        top1_prob = top2[..., 1]
        margin = top1_prob - top2_prob
        top1_id = mx.argmax(last_logits, axis=-1)
        mx.eval(probs, entropy, normalized_entropy, top1_prob, top2_prob, margin, top1_id)
        return {
            "top1_id": int(top1_id.item()),
            "top1_prob": float(top1_prob.item()),
            "top2_prob": float(top2_prob.item()),
            "margin": float(margin.item()),
            "normalized_entropy": float(normalized_entropy.item()),
        }

    def _prefill_full_depth(self, prompt_ids: List[int]) -> Dict[str, Any]:
        cache = self.model.make_cache()
        input_ids = mx.array(prompt_ids, dtype=mx.int32).reshape(1, -1)
        hidden = self.parts.embed(input_ids)
        hidden = self._run_layers(hidden, cache, 0, self.num_layers)
        logits = self.parts.compute_logits(hidden)
        mx.eval(logits)
        first_token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
        return {"cache": cache, "first_token_id": first_token_id}

    def _replay_pending_deep(
        self,
        pending_hidden: List[mx.array],
        cache: Any,
        exit_layer: int,
    ) -> int:
        if not pending_hidden:
            return 0
        replayed = 0
        for hidden in pending_hidden:
            self._run_layers(hidden, cache, exit_layer + 1, self.num_layers)
            replayed += int(hidden.shape[1])
        pending_hidden.clear()
        return replayed

    def generate_full_depth(self, prompt_ids: List[int], max_new_tokens: int = 48) -> Dict[str, Any]:
        mx.clear_cache()
        mx.reset_peak_memory()

        cache = self.model.make_cache()
        input_ids = mx.array(prompt_ids, dtype=mx.int32).reshape(1, -1)

        t0 = time.perf_counter()
        hidden = self.parts.embed(input_ids)
        hidden = self._run_layers(hidden, cache, 0, self.num_layers)
        logits = self.parts.compute_logits(hidden)
        mx.eval(logits)
        token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
        generated_ids = [token_id]
        ttft = time.perf_counter() - t0

        if token_id not in self.eos_ids:
            for _ in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                hidden = self.parts.embed(token_input)
                hidden = self._run_layers(hidden, cache, 0, self.num_layers)
                logits = self.parts.compute_logits(hidden)
                mx.eval(logits)
                token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
                generated_ids.append(token_id)
                if token_id in self.eos_ids:
                    break

        elapsed = time.perf_counter() - t0
        return {
            "generated_ids": generated_ids,
            "output_text": self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated_ids),
            "ttft_s": ttft,
            "elapsed_s": elapsed,
            "generation_tps": max(len(generated_ids) - 1, 0) / max(elapsed - ttft, 1e-6),
            "peak_memory_gb": mx.get_peak_memory() / (1024**3),
            "avg_layers": float(self.num_layers),
            "layers_saved": 0.0,
        }

    def generate_early_exit(
        self,
        prompt_ids: List[int],
        exit_layer: int,
        max_new_tokens: int = 48,
    ) -> Dict[str, Any]:
        mx.clear_cache()
        mx.reset_peak_memory()

        cache = self.model.make_cache()
        input_ids = mx.array(prompt_ids, dtype=mx.int32).reshape(1, -1)

        t0 = time.perf_counter()
        hidden = self.parts.embed(input_ids)
        hidden = self._run_layers(hidden, cache, 0, exit_layer + 1)
        logits = self.parts.compute_logits(hidden)
        mx.eval(logits)
        token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
        generated_ids = [token_id]
        ttft = time.perf_counter() - t0

        if token_id not in self.eos_ids:
            for _ in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                hidden = self.parts.embed(token_input)
                hidden = self._run_layers(hidden, cache, 0, exit_layer + 1)
                logits = self.parts.compute_logits(hidden)
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

    def generate_blended(
        self,
        prompt_ids: List[int],
        exit_layer: int,
        max_new_tokens: int = 48,
        blend_alpha: float = 0.10,
        strategy: str = "top1_agree",
    ) -> Dict[str, Any]:
        mx.clear_cache()
        mx.reset_peak_memory()

        cache = self.model.make_cache()
        input_ids = mx.array(prompt_ids, dtype=mx.int32).reshape(1, -1)

        t0 = time.perf_counter()
        hidden = self.parts.embed(input_ids)
        hidden = self._run_layers(hidden, cache, 0, self.num_layers)
        logits = self.parts.compute_logits(hidden)
        mx.eval(logits)
        token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
        generated_ids = [token_id]
        ttft = time.perf_counter() - t0

        agree_count = 0
        total_tokens = 0

        if token_id not in self.eos_ids:
            for _ in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                hidden = self.parts.embed(token_input)
                hidden = self._run_layers(hidden, cache, 0, exit_layer + 1)
                early_logits = self.parts.compute_logits(hidden)
                mx.eval(early_logits)
                early_top1 = int(mx.argmax(early_logits[0, -1], axis=-1).item())

                hidden = self._run_layers(hidden, cache, exit_layer + 1, self.num_layers)
                full_logits = self.parts.compute_logits(hidden)
                mx.eval(full_logits)
                full_top1 = int(mx.argmax(full_logits[0, -1], axis=-1).item())

                total_tokens += 1
                if strategy == "top1_agree" and early_top1 == full_top1:
                    blended = (1.0 - blend_alpha) * full_logits + blend_alpha * early_logits
                    mx.eval(blended)
                    token_id = int(mx.argmax(blended[0, -1], axis=-1).item())
                    agree_count += 1
                else:
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
            "agreement_rate": round(agreement_rate, 4),
            "agree_count": agree_count,
            "total_tokens": total_tokens,
            "avg_layers": float(self.num_layers),
            "layers_saved": 0.0,
        }

    def generate_selective_depth(
        self,
        prompt_ids: List[int],
        exit_layer: int,
        max_new_tokens: int = 48,
        entropy_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        mx.clear_cache()
        mx.reset_peak_memory()

        deep_layers_per_token = self.num_layers - (exit_layer + 1)
        if deep_layers_per_token <= 0:
            raise ValueError("exit_layer must be before the final layer")

        t0 = time.perf_counter()
        prefill = self._prefill_full_depth(prompt_ids)
        cache = prefill["cache"]
        token_id = prefill["first_token_id"]

        generated_ids: List[int] = [token_id]
        ttft = time.perf_counter() - t0

        decision_tokens = 0
        early_accepted_tokens = 0
        continued_tokens = 0
        replayed_accepted_tokens = 0
        pending_hidden: List[mx.array] = []

        if token_id not in self.eos_ids:
            for _ in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                hidden = self.parts.embed(token_input)
                hidden = self._run_layers(hidden, cache, 0, exit_layer + 1)
                early_logits = self.parts.compute_logits(hidden)
                mx.eval(early_logits)
                probe = self._confidence_probe(early_logits)
                decision_tokens += 1

                if float(probe["normalized_entropy"]) <= entropy_threshold:
                    token_id = int(probe["top1_id"])
                    generated_ids.append(token_id)
                    early_accepted_tokens += 1
                    pending_hidden.append(hidden)
                    if token_id in self.eos_ids:
                        break
                    continue

                replayed = self._replay_pending_deep(pending_hidden, cache, exit_layer)
                replayed_accepted_tokens += replayed

                hidden = self._run_layers(hidden, cache, exit_layer + 1, self.num_layers)
                full_logits = self.parts.compute_logits(hidden)
                mx.eval(full_logits)
                token_id = int(mx.argmax(full_logits[0, -1], axis=-1).item())
                generated_ids.append(token_id)
                continued_tokens += 1
                if token_id in self.eos_ids:
                    break

        elapsed = time.perf_counter() - t0
        completion_tokens = len(generated_ids)
        realized_skipped_tokens = max(early_accepted_tokens - replayed_accepted_tokens, 0)
        acceptance_rate = early_accepted_tokens / max(decision_tokens, 1)
        continuation_rate = continued_tokens / max(decision_tokens, 1)
        realized_skip_rate = realized_skipped_tokens / max(decision_tokens, 1)

        total_layer_passes = (
            decision_tokens * (exit_layer + 1)
            + (continued_tokens + replayed_accepted_tokens) * deep_layers_per_token
        )
        avg_realized_depth = (
            total_layer_passes / max(decision_tokens, 1) if decision_tokens > 0 else float(self.num_layers)
        )
        avg_layers_saved = (
            (realized_skipped_tokens * deep_layers_per_token)
            / max(decision_tokens * self.num_layers, 1)
            if decision_tokens > 0 else 0.0
        )

        return {
            "generated_ids": generated_ids,
            "output_text": self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": completion_tokens,
            "ttft_s": ttft,
            "elapsed_s": elapsed,
            "generation_tps": max(completion_tokens - 1, 0) / max(elapsed - ttft, 1e-6),
            "peak_memory_gb": mx.get_peak_memory() / (1024**3),
            "exit_layer": exit_layer,
            "prefill_strategy": "full_depth_prompt_prefill",
            "decision_tokens": decision_tokens,
            "deep_layers_per_token": deep_layers_per_token,
            "early_accepted_tokens": early_accepted_tokens,
            "continued_tokens": continued_tokens,
            "replayed_accepted_tokens": replayed_accepted_tokens,
            "realized_skipped_tokens": realized_skipped_tokens,
            "acceptance_rate": acceptance_rate,
            "continuation_rate": continuation_rate,
            "realized_skip_rate": realized_skip_rate,
            "avg_realized_depth": avg_realized_depth,
            "avg_layers_saved": avg_layers_saved,
            "avg_deep_layers_skipped": realized_skip_rate * deep_layers_per_token,
            "entropy_threshold": entropy_threshold,
        }


def build_prompt_pack(tokenizer, prompt_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    prompts = PROMPTS[:prompt_limit] if prompt_limit is not None else PROMPTS
    pack = []
    for i, user_prompt in enumerate(prompts, start=1):
        messages = build_harmony_messages(user_prompt, SYSTEM_PROMPT)
        try:
            prompt_text, _ = apply_generic_chat_template(tokenizer, messages)
        except Exception as exc:
            # Some instruct templates (for example Mistral) reject an initial
            # system turn and require strict user/assistant alternation.
            if "alternate" not in str(exc).lower():
                raise
            merged_messages = [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                }
            ]
            prompt_text, _ = apply_generic_chat_template(tokenizer, merged_messages)
        prompt_ids = tokenizer.encode(prompt_text)
        pack.append(
            {
                "prompt_id": f"P{i}",
                "user": user_prompt,
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids,
            }
        )
    return pack


def build_modes(late_layer: int, entropy_threshold: float, final_layer: int) -> List[ModeConfig]:
    return [
        ModeConfig(
            key=f"full_depth_L{final_layer}",
            label=f"Full Depth (L{final_layer})",
            description="Full-depth dense baseline",
            kind="full_depth",
        ),
        ModeConfig(
            key=f"fixed_exit_L{late_layer}",
            label=f"Fixed Exit (L{late_layer})",
            description=f"Fixed late exit at layer {late_layer}",
            kind="fixed_exit",
            exit_layer=late_layer,
        ),
        ModeConfig(
            key=f"top1_agree_compute_both_L{late_layer}",
            label=f"top1_agree compute-both (L{late_layer})",
            description="Agreement-aware compute-both quality-control baseline",
            kind="top1_agree_compute_both",
            exit_layer=late_layer,
        ),
        ModeConfig(
            key=f"selective_depth_entropy_L{late_layer}",
            label=f"Selective Depth entropy (L{late_layer})",
            description="Real selective-depth with entropy continuation rule",
            kind="selective_entropy",
            exit_layer=late_layer,
            entropy_threshold=entropy_threshold,
        ),
    ]


def run_mode(
    engine: LlamaLikeDenseEngine,
    mode: ModeConfig,
    prompt_pack: List[Dict[str, Any]],
    reference_results: Optional[Dict[str, List[int]]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt_results = []

    for prompt_def in prompt_pack:
        prompt_ids = prompt_def["prompt_ids"]
        if mode.kind == "full_depth":
            stats = engine.generate_full_depth(prompt_ids=prompt_ids, max_new_tokens=max_new_tokens)
        elif mode.kind == "fixed_exit":
            stats = engine.generate_early_exit(
                prompt_ids=prompt_ids,
                exit_layer=int(mode.exit_layer),
                max_new_tokens=max_new_tokens,
            )
        elif mode.kind == "top1_agree_compute_both":
            stats = engine.generate_blended(
                prompt_ids=prompt_ids,
                exit_layer=int(mode.exit_layer),
                max_new_tokens=max_new_tokens,
                strategy="top1_agree",
            )
        elif mode.kind == "selective_entropy":
            stats = engine.generate_selective_depth(
                prompt_ids=prompt_ids,
                exit_layer=int(mode.exit_layer),
                max_new_tokens=max_new_tokens,
                entropy_threshold=float(mode.entropy_threshold),
            )
        else:
            raise ValueError(f"unknown mode kind: {mode.kind}")

        if reference_results and prompt_def["prompt_id"] in reference_results:
            comparison = compare_sequences(
                stats["generated_ids"],
                reference_results[prompt_def["prompt_id"]],
            )
        else:
            comparison = {
                "exact_match_rate": 1.0,
                "prefix_match_tokens": stats["completion_tokens"],
                "first_divergence_position": None,
                "passed": True,
            }

        prompt_results.append(
            {
                "prompt_id": prompt_def["prompt_id"],
                "user_prompt": prompt_def["user"],
                **stats,
                "comparison": comparison,
                "preview": preview_text(stats["output_text"], 200),
            }
        )

    non_warmup = [r for i, r in enumerate(prompt_results) if i != WARMUP_PROMPT_INDEX]
    aggregate: Dict[str, Any] = {
        "avg_ttft_s": mean([r["ttft_s"] for r in non_warmup]),
        "avg_generation_tps": mean([r["generation_tps"] for r in non_warmup]),
        "avg_elapsed_s": mean([r["elapsed_s"] for r in non_warmup]),
        "avg_peak_memory_gb": mean([r["peak_memory_gb"] for r in non_warmup]),
        "avg_exact_match_rate": mean([r["comparison"]["exact_match_rate"] for r in non_warmup]),
        "avg_prefix_match_tokens": mean([r["comparison"]["prefix_match_tokens"] for r in non_warmup]),
        "avg_completion_tokens": mean([r["completion_tokens"] for r in non_warmup]),
        "avg_layers_saved": mean([r.get("layers_saved", 0.0) for r in non_warmup]),
    }

    if mode.kind == "top1_agree_compute_both":
        aggregate["acceptance_rate"] = mean([r.get("agreement_rate", 0.0) for r in non_warmup])
        aggregate["continuation_rate"] = 1.0 - aggregate["acceptance_rate"]
        aggregate["avg_realized_depth"] = float(engine.num_layers)
        aggregate["realized_skip_rate"] = 0.0
        aggregate["early_accepted_tokens"] = 0
        aggregate["continued_tokens"] = 0
    elif mode.kind == "selective_entropy":
        decision_tokens = sum(r.get("decision_tokens", 0) for r in non_warmup)
        early_accepted_tokens = sum(r.get("early_accepted_tokens", 0) for r in non_warmup)
        continued_tokens = sum(r.get("continued_tokens", 0) for r in non_warmup)
        replayed_accepted_tokens = sum(r.get("replayed_accepted_tokens", 0) for r in non_warmup)
        realized_skipped_tokens = sum(r.get("realized_skipped_tokens", 0) for r in non_warmup)
        total_realized_depth = sum(
            r.get("avg_realized_depth", 0.0) * max(r.get("decision_tokens", 0), 0)
            for r in non_warmup
        )
        total_layers_saved = sum(
            r.get("avg_layers_saved", 0.0) * max(r.get("decision_tokens", 0), 0)
            for r in non_warmup
        )
        aggregate.update(
            {
                "acceptance_rate": early_accepted_tokens / max(decision_tokens, 1),
                "continuation_rate": continued_tokens / max(decision_tokens, 1),
                "avg_realized_depth": total_realized_depth / max(decision_tokens, 1),
                "avg_layers_saved": total_layers_saved / max(decision_tokens, 1),
                "realized_skip_rate": realized_skipped_tokens / max(decision_tokens, 1),
                "early_accepted_tokens": early_accepted_tokens,
                "continued_tokens": continued_tokens,
                "replayed_accepted_tokens": replayed_accepted_tokens,
            }
        )

    return {
        "key": mode.key,
        "label": mode.label,
        "description": mode.description,
        "status": "ok",
        "prompt_results": prompt_results,
        "aggregate_excluding_warmup": aggregate,
    }


def detect_resolution_entry(avg_kls: List[float]) -> int:
    if len(avg_kls) < 4:
        return len(avg_kls) - 1
    deltas = [avg_kls[i - 1] - avg_kls[i] for i in range(1, len(avg_kls))]
    tail = [d for d in deltas[-4:] if d > 0]
    if not tail:
        return len(avg_kls) - 1
    sharp_threshold = float(np.median(tail)) * 0.75
    start = max(len(avg_kls) // 2, len(avg_kls) - 8)
    for layer_idx in range(start, len(avg_kls) - 1):
        tail_deltas = deltas[layer_idx - 1 :]
        if tail_deltas and all(d > 0 for d in tail_deltas) and np.mean(tail_deltas) >= sharp_threshold:
            return layer_idx
    return len(avg_kls) - 1


def detect_plateau_zone(avg_kls: List[float]) -> Optional[Tuple[int, int]]:
    if len(avg_kls) < 8:
        return None
    deltas = [avg_kls[i - 1] - avg_kls[i] for i in range(1, len(avg_kls))]
    positive = [d for d in deltas if d > 0]
    if not positive:
        return None
    small_threshold = float(np.median(positive)) * 0.35
    end_limit = max(len(avg_kls) - 5, 2)
    best: Optional[Tuple[int, int]] = None
    run_start: Optional[int] = None
    for i in range(1, end_limit):
        d = deltas[i - 1]
        if d <= small_threshold:
            if run_start is None:
                run_start = i - 1
        else:
            if run_start is not None and i - run_start >= 3:
                best = (run_start, i - 1)
            run_start = None
    if run_start is not None and end_limit - run_start >= 3:
        best = (run_start, end_limit - 1)
    return best


def compute_separation_ratio(layer_a: np.ndarray, layer_b: np.ndarray) -> Dict[str, float]:
    combined = np.concatenate([layer_a, layer_b], axis=0).astype(np.float32)
    combined -= combined.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(combined, full_matrices=False)
    basis = vt[:2].T
    proj_a = layer_a @ basis
    proj_b = layer_b @ basis
    centroid_a = proj_a.mean(axis=0)
    centroid_b = proj_b.mean(axis=0)
    sigma_a = float(np.std(proj_a, axis=0).mean())
    sigma_b = float(np.std(proj_b, axis=0).mean())
    centroid_distance = float(np.linalg.norm(centroid_a - centroid_b))
    separation_ratio = centroid_distance / max(sigma_a, sigma_b, 1e-6)
    return {
        "centroid_distance": centroid_distance,
        "sigma_a": sigma_a,
        "sigma_b": sigma_b,
        "separation_ratio": separation_ratio,
    }


def run_profile(
    engine: LlamaLikeDenseEngine,
    prompt_pack: List[Dict[str, Any]],
    max_tokens: int,
) -> Dict[str, Any]:
    middle_layer = engine.num_layers // 2
    final_layer = engine.num_layers - 1
    per_prompt_profiles = []
    middle_hidden_all: List[np.ndarray] = []
    final_hidden_all: List[np.ndarray] = []

    for prompt_def in prompt_pack:
        token_ids = prompt_def["prompt_ids"][:max_tokens]
        input_ids = mx.array(token_ids, dtype=mx.int32).reshape(1, -1)
        hidden = engine.parts.embed(input_ids)
        fa_mask, swa_mask = engine._build_masks(hidden, cache=None)

        layer_logits_np: List[np.ndarray] = []
        middle_hidden = None
        final_hidden = None

        for i in range(engine.num_layers):
            mask = engine._layer_mask(i, fa_mask, swa_mask)
            hidden = engine.layers[i](hidden, mask, cache=None)
            mx.eval(hidden)
            logits_i = engine.parts.compute_logits(hidden)
            mx.eval(logits_i)
            layer_logits_np.append(to_numpy(logits_i))
            if i == middle_layer:
                middle_hidden = to_numpy(hidden[0])
            if i == final_layer:
                final_hidden = to_numpy(hidden[0])

        deep_logits = layer_logits_np[-1]
        per_layer = []
        for i in range(engine.num_layers):
            kl = kl_divergence_np(deep_logits, layer_logits_np[i])
            per_layer.append(
                {
                    "layer_idx": i,
                    "avg_kl": float(np.mean(kl)),
                    "median_kl": float(np.median(kl)),
                }
            )

        if middle_hidden is not None:
            middle_hidden_all.append(middle_hidden)
        if final_hidden is not None:
            final_hidden_all.append(final_hidden)

        per_prompt_profiles.append(
            {
                "prompt_id": prompt_def["prompt_id"],
                "user_prompt": prompt_def["user"],
                "num_tokens_profiled": len(token_ids),
                "per_layer": per_layer,
            }
        )

    aggregated_per_layer = []
    for layer_idx in range(engine.num_layers):
        avg_kls = [p["per_layer"][layer_idx]["avg_kl"] for p in per_prompt_profiles]
        med_kls = [p["per_layer"][layer_idx]["median_kl"] for p in per_prompt_profiles]
        aggregated_per_layer.append(
            {
                "layer_idx": layer_idx,
                "avg_kl": round(float(np.mean(avg_kls)), 6),
                "median_kl": round(float(np.mean(med_kls)), 6),
            }
        )

    avg_kls = [entry["avg_kl"] for entry in aggregated_per_layer]
    kl0 = max(avg_kls[0], 1e-9)
    for i, entry in enumerate(aggregated_per_layer):
        entry["delta_kl"] = 0.0 if i == 0 else round(avg_kls[i - 1] - avg_kls[i], 6)
        entry["kl_reduction_pct"] = round((kl0 - entry["avg_kl"]) / kl0 * 100.0, 2)

    resolution_entry = detect_resolution_entry(avg_kls)
    plateau_zone = detect_plateau_zone(avg_kls)
    separation = compute_separation_ratio(
        np.concatenate(middle_hidden_all, axis=0),
        np.concatenate(final_hidden_all, axis=0),
    )

    return {
        "model_id": engine.parts.model_id,
        "family": engine.parts.family,
        "architecture": "dense",
        "num_layers": engine.num_layers,
        "middle_layer": middle_layer,
        "final_layer": final_layer,
        "resolution_entry_layer": resolution_entry,
        "plateau_zone": list(plateau_zone) if plateau_zone is not None else None,
        "separation_ratio_middle_to_final": round(separation["separation_ratio"], 4),
        "centroid_distance_middle_to_final": round(separation["centroid_distance"], 4),
        "middle_sigma": round(separation["sigma_a"], 4),
        "final_sigma": round(separation["sigma_b"], 4),
        "aggregated_per_layer": aggregated_per_layer,
        "per_prompt_profiles": per_prompt_profiles,
    }


def print_profile_summary(profile: Dict[str, Any]) -> None:
    plateau_zone = profile["plateau_zone"]
    plateau_text = "none detected" if plateau_zone is None else f"L{plateau_zone[0]}–L{plateau_zone[1]}"
    print(
        f"  Profile: resolution entry ~L{profile['resolution_entry_layer']} | "
        f"plateau={plateau_text} | "
        f"middle-final separation={profile['separation_ratio_middle_to_final']:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused dense-family validation for Llama/Mistral MLX models"
    )
    parser.add_argument("--model", type=str, required=True, help="Local path or Hugging Face repo ID")
    parser.add_argument("--family", type=str, required=True, help="Family label, e.g. llama or mistral")
    parser.add_argument("--profile-output", type=str, required=True)
    parser.add_argument("--benchmark-output", type=str, required=True)
    parser.add_argument("--max-profile-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--late-layer", type=int, default=None)
    parser.add_argument("--entropy-threshold", type=float, default=0.15)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 96)
    print("  Track C Extension — Focused Dense-Family Validation")
    print("=" * 96)
    print(f"\n  Loading model: {args.model}")
    model, tokenizer = mlx_load(args.model)
    parts = LlamaLikeModelParts.from_loaded(model, tokenizer, args.model, args.family)
    engine = LlamaLikeDenseEngine(parts)

    final_layer = parts.num_layers - 1
    late_layer = args.late_layer if args.late_layer is not None else parts.num_layers - 3
    prompt_pack = build_prompt_pack(tokenizer, prompt_limit=args.prompt_limit)

    print(
        f"  Family: {parts.family} | layers: {parts.num_layers} | hidden: {parts.hidden_size} | "
        f"late checkpoint: L{late_layer}"
    )
    print(f"  Prompts: {len(prompt_pack)} | warmup index: {WARMUP_PROMPT_INDEX}")

    if not args.skip_profile:
        print("\n  ── Running KL recon ──")
        profile = run_profile(engine, prompt_pack, max_tokens=args.max_profile_tokens)
        with open(args.profile_output, "w") as fh:
            json.dump(profile, fh, indent=2)
        print(f"  Recon saved to {args.profile_output}")
        print_profile_summary(profile)

    if not args.skip_benchmark:
        print("\n  ── Running focused benchmark ──")
        modes = build_modes(late_layer=late_layer, entropy_threshold=args.entropy_threshold, final_layer=final_layer)
        reference_results: Optional[Dict[str, List[int]]] = None
        all_modes_data = []
        for mode in modes:
            print(f"\n    Running {mode.label}...")
            result = run_mode(
                engine=engine,
                mode=mode,
                prompt_pack=prompt_pack,
                reference_results=reference_results,
                max_new_tokens=args.max_new_tokens,
            )
            if mode.kind == "full_depth":
                reference_results = {
                    prompt_result["prompt_id"]: prompt_result["generated_ids"]
                    for prompt_result in result["prompt_results"]
                }
            agg = result["aggregate_excluding_warmup"]
            print(
                f"      TPS: {agg['avg_generation_tps']:.2f} | Exact: {agg['avg_exact_match_rate']:.3f} | "
                f"Saved: {agg.get('avg_layers_saved', 0.0):.1%} | "
                f"Accept: {agg.get('acceptance_rate', 0.0):.1%}"
            )
            all_modes_data.append(result)

        benchmark_payload = {
            "experiment": "transcender_track_c_dense_family_validation",
            "model_id": parts.model_id,
            "family": parts.family,
            "architecture": "dense",
            "num_layers": parts.num_layers,
            "late_layer": late_layer,
            "entropy_threshold": args.entropy_threshold,
            "warmup_prompt_index": WARMUP_PROMPT_INDEX,
            "modes": all_modes_data,
        }
        with open(args.benchmark_output, "w") as fh:
            json.dump(benchmark_payload, fh, indent=2)
        print(f"\n  Benchmark saved to {args.benchmark_output}")


if __name__ == "__main__":
    main()
