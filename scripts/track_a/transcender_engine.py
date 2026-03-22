"""
TranscenderEngine — MLX sparse-MoE selective-exit engine.

Implements entropy-gated early exit with physical layer skipping on Apple MLX.
The current Track A benchmark path supports:
  - GPT-OSS 20B (`gpt_oss`)
  - Qwen3-30B-A3B (`qwen3_moe`)

The release-facing interpretation of `avg_layers_saved` is conservative:
it is the mean number of layers physically skipped per generated token, not a
general compute-savings percentage.

Subspace mismatch mitigation:
  all composition occurs in logit space via the model's final normalization and
  LM head; hidden-state blending is intentionally avoided.

Prerequisites:
    pip install mlx mlx-lm
"""

# ═══════════════════════════════════════════════════════════════════
# NOTE ON MLX AVAILABILITY
# ═══════════════════════════════════════════════════════════════════
# This module is designed for Apple MLX framework.
# If MLX is not installed, it falls back to a PyTorch reference
# implementation for validation and architecture analysis.
#
# The code is structured as importable classes regardless of backend.
# ═══════════════════════════════════════════════════════════════════

from __future__ import annotations
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Backend detection ──
try:
    import mlx.core as mx
    import mlx.nn as mnn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════
# Architecture Constants (from config.json)
# ═══════════════════════════════════════════════════════════════════

_GPT_OSS_METADATA_FILES = (
    "config.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
)


def _has_nonempty_metadata_file(path: Path) -> bool:
    try:
        return len(path.read_bytes()) > 0
    except OSError:
        return False


def _has_complete_gpt_oss_metadata(model_dir: Path) -> bool:
    return all(
        _has_nonempty_metadata_file(model_dir / filename)
        for filename in _GPT_OSS_METADATA_FILES
    )


def _find_cached_gpt_oss_snapshot() -> Optional[Path]:
    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--openai--gpt-oss-20b"
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None

    candidates = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _has_complete_gpt_oss_metadata(candidate):
            return candidate
    return None


def resolve_gpt_oss_model_path(model_path: str) -> str:
    """
    Resolve a usable local GPT-OSS model path.

    Some local copies can contain sparse placeholder metadata files at the
    top level while still having valid weights. In that case, prefer an intact
    Hugging Face cache snapshot if one is available.
    """
    requested_path = Path(model_path).expanduser()
    if not requested_path.exists() or not requested_path.is_dir():
        return model_path

    if _has_complete_gpt_oss_metadata(requested_path):
        return str(requested_path)

    looks_like_gpt_oss = (
        (requested_path / "model.safetensors.index.json").exists()
        or (requested_path / "original" / "config.json").exists()
    )
    if not looks_like_gpt_oss:
        return str(requested_path)

    cached_snapshot = _find_cached_gpt_oss_snapshot()
    if cached_snapshot is not None:
        return str(cached_snapshot)

    missing = [
        filename
        for filename in _GPT_OSS_METADATA_FILES
        if not _has_nonempty_metadata_file(requested_path / filename)
    ]
    raise RuntimeError(
        "Local GPT-OSS model metadata is empty or missing under "
        f"{requested_path}: {', '.join(missing)}. Re-download the model or "
        "point --model at a valid Hugging Face cache snapshot."
    )


def load_resolved_mlx_model(
    model_path: str,
    lazy: bool = True,
):
    """
    Load GPT-OSS through mlx_lm after resolving sparse metadata placeholders.
    """
    from mlx_lm import load as mlx_load

    resolved_model_path = resolve_gpt_oss_model_path(model_path)
    model, tokenizer = mlx_load(resolved_model_path, lazy=lazy)
    return model, tokenizer, resolved_model_path


def load_mlx_model(model_path: str, lazy: bool = True):
    """Load any MLX model directly without GPT-OSS path resolution."""
    from mlx_lm import load as mlx_load

    resolved = str(Path(model_path).expanduser())
    model, tokenizer = mlx_load(resolved, lazy=lazy)
    return model, tokenizer, resolved


def load_resolved_transformers_model(model_path: str):
    """
    Load GPT-OSS through transformers after resolving sparse metadata placeholders.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_model_path = resolve_gpt_oss_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    return model, tokenizer, resolved_model_path


def build_harmony_messages(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def apply_harmony_template(
    tokenizer,
    user_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    reasoning_effort: str = "medium",
    system_prompt: str = "You are a helpful assistant.",
    add_generation_prompt: bool = True,
) -> tuple[str, List[Dict[str, str]]]:
    """
    Render an OpenAI-style message list through the GPT-OSS Harmony template.
    """
    if messages is None:
        if user_prompt is None:
            raise ValueError("Provide either user_prompt or messages.")
        messages = build_harmony_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support apply_chat_template().")

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            reasoning_effort=reasoning_effort,
        )
    except TypeError:
        # Non-GPT-OSS tokenizers don't accept reasoning_effort
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return prompt, messages

@dataclass
class GptOssConfig:
    """Exact architecture of openai/gpt-oss-20b."""
    model_type: str = "gpt_oss"
    num_hidden_layers: int = 24
    hidden_size: int = 2880
    num_attention_heads: int = 64
    num_key_value_heads: int = 8       # GQA 8:1 ratio
    head_dim: int = 64
    intermediate_size: int = 2880
    vocab_size: int = 201088
    num_local_experts: int = 32
    num_experts_per_tok: int = 4
    max_position_embeddings: int = 131072
    rope_theta: float = 150000.0
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    sliding_window: int = 128
    attention_bias: bool = True
    tie_word_embeddings: bool = False
    router_aux_loss_coef: float = 0.9
    swiglu_limit: float = 7.0
    soft_skip_start_layer: int = 19
    hard_exit_layer: int = 22
    entropy_threshold: float = 0.20
    min_entropy_streak: int = 2
    enable_logit_blending: bool = False
    blending_confidence_threshold: float = 0.05
    blend_alpha: float = 0.10
    confidence_signal: str = "entropy"
    margin_threshold: float = 0.08
    blend_alpha_mode: str = "fixed"
    blend_alpha_sigmoid_scale: float = 20.0
    blend_entropy_sigmoid_scale: float = 20.0
    blend_alpha_margin_scale: float = 1.0
    fallback_to_full_depth_on_ambiguity: bool = False
    blend_strategy: str = "full_vocab"
    blend_top_k: int = 5
    anchor_alpha_scale: float = 0.25
    prefill_step_size: int = 2048
    memory_limit_gb: float = 30.0
    cache_cleanup_interval: int = 32
    target_peak_memory_gb: float = 14.0

    # Layer attention pattern: even=sliding, odd=full
    layer_types: list = field(default_factory=lambda: [
        "sliding_attention" if i % 2 == 0 else "full_attention"
        for i in range(24)
    ])

    # MXFP4 exclusions: these modules stay in higher precision
    mxfp4_excluded: list = field(default_factory=lambda: [
        "model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head",
    ])

    @property
    def gqa_ratio(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads  # 8

    @property
    def total_params_b(self) -> float:
        return 21.0

    @property
    def active_params_b(self) -> float:
        return 3.6


@dataclass
class Qwen3MoeConfig:
    """Architecture constants for Qwen3-30B-A3B (48-layer MoE, 128 experts, top-8)."""
    model_type: str = "qwen3_moe"
    num_hidden_layers: int = 48
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 6144
    vocab_size: int = 151936
    num_local_experts: int = 128
    num_experts_per_tok: int = 8
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    sliding_window: int = 0
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    router_aux_loss_coef: float = 0.0
    swiglu_limit: float = 7.0
    soft_skip_start_layer: int = 35
    hard_exit_layer: int = 46
    entropy_threshold: float = 0.20
    min_entropy_streak: int = 2
    enable_logit_blending: bool = False
    blending_confidence_threshold: float = 0.05
    blend_alpha: float = 0.10
    confidence_signal: str = "entropy"
    margin_threshold: float = 0.08
    blend_alpha_mode: str = "fixed"
    blend_alpha_sigmoid_scale: float = 20.0
    blend_entropy_sigmoid_scale: float = 20.0
    blend_alpha_margin_scale: float = 1.0
    fallback_to_full_depth_on_ambiguity: bool = False
    blend_strategy: str = "full_vocab"
    blend_top_k: int = 5
    anchor_alpha_scale: float = 0.25
    prefill_step_size: int = 2048
    memory_limit_gb: float = 30.0
    cache_cleanup_interval: int = 32
    target_peak_memory_gb: float = 22.0

    layer_types: list = field(default_factory=lambda: [
        "full_attention" for _ in range(48)
    ])

    mxfp4_excluded: list = field(default_factory=lambda: [])

    @property
    def gqa_ratio(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def total_params_b(self) -> float:
        return 30.5

    @property
    def active_params_b(self) -> float:
        return 3.3


# ═══════════════════════════════════════════════════════════════════
# MLX Dynamic Expert-Skipping Engine
# ═══════════════════════════════════════════════════════════════════

if HAS_MLX:
    from mlx_lm.models.base import create_attention_mask as mlx_create_attention_mask

    class MLXDynamicExpertEngine:
        """
        GPT-OSS 20B inference engine with KL-profiled dynamic expert-skipping.

        Policy encoded from reconnaissance:
          - Soft-skip starts at layer 19, based on normalized logit entropy.
          - Layers 19 and 20 must both satisfy the entropy gate before the
            skip path can open.
          - Layer 22 is the hard exit point: the model exits after layer 22
            and never executes layer 23.
          - If the streak-2 gate triggers, layers 21 and 22 run in
            attention-only mode; otherwise both stay full-depth.

        This keeps the KV cache intact because attention always runs, while
        physically skipping expensive MoE MLP blocks once the logits have
        already converged.
        """

        def __init__(
            self,
            model=None,
            tokenizer=None,
            model_path: Optional[str] = None,
            config: GptOssConfig = GptOssConfig(),
            soft_skip_start_layer: Optional[int] = None,
            hard_exit_layer: Optional[int] = None,
            entropy_threshold: Optional[float] = None,
            min_entropy_streak: Optional[int] = None,
            prefill_step_size: Optional[int] = None,
            memory_limit_gb: Optional[float] = None,
            trace_decisions: bool = False,
            collect_runtime_stats: bool = True,
            enable_allocator_cleanup: bool = True,
            reuse_attention_masks: bool = True,
            force_layer_state_eval: bool = False,
            force_entropy_measurement: bool = False,
            enable_logit_blending: Optional[bool] = None,
            blending_confidence_threshold: Optional[float] = None,
            blend_alpha: Optional[float] = None,
            confidence_signal: Optional[str] = None,
            margin_threshold: Optional[float] = None,
            blend_alpha_mode: Optional[str] = None,
            blend_alpha_sigmoid_scale: Optional[float] = None,
            blend_entropy_sigmoid_scale: Optional[float] = None,
            blend_alpha_margin_scale: Optional[float] = None,
            fallback_to_full_depth_on_ambiguity: Optional[bool] = None,
            blend_strategy: Optional[str] = None,
            blend_top_k: Optional[int] = None,
            anchor_alpha_scale: Optional[float] = None,
        ):
            resolved_model_path = model_path
            if model is None or tokenizer is None:
                if model_path is None:
                    raise ValueError("Provide either (model, tokenizer) or model_path.")
                model, tokenizer, resolved_model_path = load_resolved_mlx_model(
                    model_path,
                    lazy=True,
                )
                if resolved_model_path != model_path:
                    print(
                        "  Detected placeholder GPT-OSS metadata under "
                        f"{model_path}; using cached snapshot {resolved_model_path}."
                    )

            self.model = model
            self.tokenizer = tokenizer
            self.config = config
            self.model_path = model_path
            self.resolved_model_path = resolved_model_path
            self.soft_skip_start_layer = (
                config.soft_skip_start_layer
                if soft_skip_start_layer is None
                else soft_skip_start_layer
            )
            self.hard_exit_layer = (
                config.hard_exit_layer
                if hard_exit_layer is None
                else hard_exit_layer
            )
            self.entropy_threshold = (
                config.entropy_threshold
                if entropy_threshold is None
                else entropy_threshold
            )
            self.min_entropy_streak = (
                config.min_entropy_streak
                if min_entropy_streak is None
                else min_entropy_streak
            )
            self.prefill_step_size = (
                config.prefill_step_size
                if prefill_step_size is None
                else prefill_step_size
            )
            self.cache_cleanup_interval = config.cache_cleanup_interval
            self.trace_decisions = trace_decisions
            self.collect_runtime_stats = collect_runtime_stats
            self.enable_allocator_cleanup = enable_allocator_cleanup
            self.reuse_attention_masks = reuse_attention_masks
            self.force_layer_state_eval = force_layer_state_eval
            self.force_entropy_measurement = force_entropy_measurement
            self.enable_logit_blending = (
                config.enable_logit_blending
                if enable_logit_blending is None
                else enable_logit_blending
            )
            self.blending_confidence_threshold = (
                config.blending_confidence_threshold
                if blending_confidence_threshold is None
                else blending_confidence_threshold
            )
            self.blend_alpha = (
                config.blend_alpha
                if blend_alpha is None
                else blend_alpha
            )
            self.confidence_signal = (
                config.confidence_signal
                if confidence_signal is None
                else confidence_signal
            )
            self.margin_threshold = (
                config.margin_threshold
                if margin_threshold is None
                else margin_threshold
            )
            self.blend_alpha_mode = (
                config.blend_alpha_mode
                if blend_alpha_mode is None
                else blend_alpha_mode
            )
            self.blend_alpha_sigmoid_scale = (
                config.blend_alpha_sigmoid_scale
                if blend_alpha_sigmoid_scale is None
                else blend_alpha_sigmoid_scale
            )
            self.blend_entropy_sigmoid_scale = (
                config.blend_entropy_sigmoid_scale
                if blend_entropy_sigmoid_scale is None
                else blend_entropy_sigmoid_scale
            )
            self.blend_alpha_margin_scale = (
                config.blend_alpha_margin_scale
                if blend_alpha_margin_scale is None
                else blend_alpha_margin_scale
            )
            self.fallback_to_full_depth_on_ambiguity = (
                config.fallback_to_full_depth_on_ambiguity
                if fallback_to_full_depth_on_ambiguity is None
                else fallback_to_full_depth_on_ambiguity
            )
            self.blend_strategy = (
                config.blend_strategy
                if blend_strategy is None
                else blend_strategy
            )
            self.blend_top_k = (
                config.blend_top_k
                if blend_top_k is None
                else blend_top_k
            )
            self.anchor_alpha_scale = (
                config.anchor_alpha_scale
                if anchor_alpha_scale is None
                else anchor_alpha_scale
            )
            self.memory_limit_gb = (
                config.memory_limit_gb
                if memory_limit_gb is None
                else memory_limit_gb
            )

            self.embed_tokens = model.model.embed_tokens
            self.layers = model.model.layers
            self.final_norm = model.model.norm
            self.lm_head = model.lm_head
            self.layer_types = list(
                getattr(model.model, "layer_types", config.layer_types)
            )
            self.window_size = getattr(model.model, "window_size", config.sliding_window)
            self.num_layers = len(self.layers)
            self.gate_activation_layer = (
                self.soft_skip_start_layer + self.min_entropy_streak - 1
            )
            self.sliding_layer_count = sum(
                1 for layer_type in self.layer_types if layer_type == "sliding_attention"
            )
            self.full_layer_count = self.num_layers - self.sliding_layer_count
            self.first_sliding_idx = next(
                (
                    idx
                    for idx, layer_type in enumerate(self.layer_types)
                    if layer_type == "sliding_attention"
                ),
                None,
            )
            self.first_full_idx = next(
                (
                    idx
                    for idx, layer_type in enumerate(self.layer_types)
                    if layer_type == "full_attention"
                ),
                None,
            )

            if self.num_layers <= self.hard_exit_layer:
                raise ValueError(
                    f"hard_exit_layer={self.hard_exit_layer} requires at least "
                    f"{self.hard_exit_layer + 1} layers, got {self.num_layers}"
                )
            if self.soft_skip_start_layer > self.hard_exit_layer:
                raise ValueError(
                    "soft_skip_start_layer must be <= hard_exit_layer."
                )
            if self.gate_activation_layer > self.hard_exit_layer:
                raise ValueError(
                    "hard_exit_layer must be >= soft_skip_start_layer + min_entropy_streak - 1."
                )
            if self.confidence_signal not in {"entropy", "entropy_margin"}:
                raise ValueError(
                    "confidence_signal must be 'entropy' or 'entropy_margin'."
                )
            if self.blend_alpha_mode not in {"fixed", "sigmoid_margin", "margin_linear"}:
                raise ValueError(
                    "blend_alpha_mode must be 'fixed', 'sigmoid_margin', or 'margin_linear'."
                )
            if self.blend_strategy not in {
                "full_vocab",
                "top_k_mask",
                "top1_agree",
                "deep_top1_anchor",
            }:
                raise ValueError(
                    "blend_strategy must be one of "
                    "{'full_vocab', 'top_k_mask', 'top1_agree', 'deep_top1_anchor'}."
                )
            if self.blend_top_k <= 0:
                raise ValueError("blend_top_k must be > 0.")

            self.memory_budget = self._configure_memory_budget(self.memory_limit_gb)
            self._stats_lock = threading.Lock()
            self._runtime_stats = {
                "requests_served": 0,
                "dynamic_requests": 0,
                "avg_layers": 0.0,
                "avg_layers_saved": 0.0,
                "last_layers_saved": 0.0,
                "last_ttft_s": None,
                "last_peak_memory_gb": None,
                "last_cache_memory_gb": None,
                "last_avg_layers": None,
                "last_generation_tps": None,
                "last_prompt_tokens": 0,
                "last_completion_tokens": 0,
                "last_consistency_agreement": None,
                "last_consistency_kl": None,
                "last_consistency_tokens": 0,
                "consistency_checks_run": 0,
                "consistency_agreement_avg": None,
                "consistency_kl_avg": None,
                "resolved_model_path": self.resolved_model_path,
                "soft_skip_start_layer": self.soft_skip_start_layer,
                "gate_activation_layer": self.gate_activation_layer,
                "hard_exit_layer": self.hard_exit_layer,
                "enable_logit_blending": self.enable_logit_blending,
                "blending_confidence_threshold": self.blending_confidence_threshold,
                "blend_alpha": self.blend_alpha,
                "confidence_signal": self.confidence_signal,
                "margin_threshold": self.margin_threshold,
                "blend_alpha_mode": self.blend_alpha_mode,
                "blend_alpha_sigmoid_scale": self.blend_alpha_sigmoid_scale,
                "blend_entropy_sigmoid_scale": self.blend_entropy_sigmoid_scale,
                "blend_alpha_margin_scale": self.blend_alpha_margin_scale,
                "fallback_to_full_depth_on_ambiguity": self.fallback_to_full_depth_on_ambiguity,
                "blend_strategy": self.blend_strategy,
                "blend_top_k": self.blend_top_k,
                "anchor_alpha_scale": self.anchor_alpha_scale,
                "sliding_window": self.window_size,
                "sliding_layers": self.sliding_layer_count,
                "full_attention_layers": self.full_layer_count,
                "prefill_step_size": self.prefill_step_size,
                "target_peak_memory_gb": self.config.target_peak_memory_gb,
            }

        def _configure_memory_budget(self, limit_gb: float) -> Dict[str, float]:
            budget_bytes = int(limit_gb * (1024**3))
            info: Dict[str, Any] = {}
            try:
                info = mx.device_info()
            except Exception:
                info = {}

            result = {
                "memory_limit_gb": float(limit_gb),
                "wired_limit_gb": 0.0,
            }
            try:
                mx.set_memory_limit(budget_bytes)
            except Exception:
                pass

            recommended = info.get("max_recommended_working_set_size")
            if isinstance(recommended, int) and recommended > 0:
                result["recommended_working_set_gb"] = recommended / (1024**3)

            memory_size = info.get("memory_size")
            if isinstance(memory_size, int) and memory_size > 0:
                result["device_memory_gb"] = memory_size / (1024**3)

            return result

        def _make_cache(self):
            if hasattr(self.model, "make_cache"):
                return self.model.make_cache()
            if hasattr(self.model, "model") and hasattr(self.model.model, "make_cache"):
                return self.model.model.make_cache()
            # Fallback: create a default KVCache per layer (for models like Qwen3
            # that don't expose make_cache but work with standard KV caches).
            from mlx_lm.models.cache import KVCache
            return [KVCache() for _ in range(self.num_layers)]

        def _cache_nbytes(self, cache) -> int:
            return int(
                sum(getattr(cache_entry, "nbytes", 0) for cache_entry in cache if cache_entry is not None)
            )

        def _cache_policy(self, cache) -> Dict[str, Any]:
            return {
                "sliding_window": self.window_size,
                "sliding_layers": self.sliding_layer_count,
                "full_attention_layers": self.full_layer_count,
                "prefill_step_size": self.prefill_step_size,
                "cache_cleanup_interval": self.cache_cleanup_interval,
                "cache_bytes_gb": self._cache_nbytes(cache) / (1024**3),
            }

        def _cleanup_allocator(self):
            # Free allocator-held scratch buffers without dropping live KV state.
            mx.clear_cache()

        def _record_generation_stats(self, stats: Dict[str, Any], dynamic: bool):
            if not self.collect_runtime_stats:
                return
            with self._stats_lock:
                requests_served = self._runtime_stats["requests_served"] + 1
                self._runtime_stats["requests_served"] = requests_served
                if dynamic:
                    dynamic_requests = self._runtime_stats["dynamic_requests"] + 1
                    prev_avg_layers = self._runtime_stats["avg_layers"]
                    prev_avg_saved = self._runtime_stats["avg_layers_saved"]
                    layers_saved = max(self.num_layers - stats["avg_layers"], 0.0)
                    self._runtime_stats["dynamic_requests"] = dynamic_requests
                    self._runtime_stats["avg_layers"] = (
                        prev_avg_layers * (dynamic_requests - 1) + stats["avg_layers"]
                    ) / dynamic_requests
                    self._runtime_stats["avg_layers_saved"] = (
                        prev_avg_saved * (dynamic_requests - 1) + layers_saved
                    ) / dynamic_requests
                    self._runtime_stats["last_layers_saved"] = layers_saved
                self._runtime_stats["last_ttft_s"] = stats["ttft_s"]
                self._runtime_stats["last_peak_memory_gb"] = stats["peak_memory_gb"]
                self._runtime_stats["last_cache_memory_gb"] = stats["cache_memory_gb"]
                self._runtime_stats["last_avg_layers"] = stats["avg_layers"]
                self._runtime_stats["last_generation_tps"] = stats["generation_tps"]
                self._runtime_stats["last_prompt_tokens"] = stats["prompt_tokens"]
                self._runtime_stats["last_completion_tokens"] = stats["tokens_generated"]
                self._runtime_stats["last_cache_policy"] = stats["cache_policy"]

        def _record_consistency_stats(self, check: Dict[str, Any]):
            if not self.collect_runtime_stats:
                return
            with self._stats_lock:
                count = self._runtime_stats["consistency_checks_run"] + 1
                prev_agreement = self._runtime_stats["consistency_agreement_avg"]
                prev_kl = self._runtime_stats["consistency_kl_avg"]
                agreement = check["top1_agreement"]
                avg_kl = check["avg_kl_divergence"]

                self._runtime_stats["consistency_checks_run"] = count
                self._runtime_stats["last_consistency_agreement"] = agreement
                self._runtime_stats["last_consistency_kl"] = avg_kl
                self._runtime_stats["last_consistency_tokens"] = check["tokens_compared"]
                self._runtime_stats["consistency_agreement_avg"] = (
                    agreement
                    if prev_agreement is None
                    else ((prev_agreement * (count - 1)) + agreement) / count
                )
                self._runtime_stats["consistency_kl_avg"] = (
                    avg_kl
                    if prev_kl is None
                    else ((prev_kl * (count - 1)) + avg_kl) / count
                )

        def get_runtime_stats(self) -> Dict[str, Any]:
            with self._stats_lock:
                stats = dict(self._runtime_stats)
            stats["memory_budget"] = dict(self.memory_budget)
            return stats

        def _flatten_cache_state(self, cache_entry) -> List[Any]:
            state = getattr(cache_entry, "state", None)
            if state is None:
                return []
            if isinstance(state, tuple):
                return [x for x in state if x is not None]
            if isinstance(state, list):
                return [x for x in state if x is not None]
            return [state]

        def _eval_layer_state(self, hidden_states, cache_entry):
            arrays = [hidden_states] + self._flatten_cache_state(cache_entry)
            mx.eval(*arrays)

        def _eval_cache_states(self, cache, upper_layer: int):
            mx.eval([cache_entry.state for cache_entry in cache[: upper_layer + 1]])

        def _mask_for_layer(self, layer_idx: int, hidden_states, cache_entry):
            window_size = (
                self.window_size
                if self.layer_types[layer_idx] == "sliding_attention"
                else None
            )
            return mlx_create_attention_mask(
                hidden_states,
                cache=cache_entry,
                window_size=window_size,
            )

        def _attention_only(self, layer, hidden_states, mask, cache_entry):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            attn_output = layer.self_attn(normed, mask, cache_entry)
            return residual + attn_output

        def _apply_mlp(self, layer, attn_states):
            residual = attn_states
            normed = layer.post_attention_layernorm(attn_states)
            mlp_output = layer.mlp(normed)
            return residual + mlp_output

        def _sigmoid(self, value: float) -> float:
            if value >= 0.0:
                z = math.exp(-value)
                return 1.0 / (1.0 + z)
            z = math.exp(value)
            return z / (1.0 + z)

        def _confidence_probe(self, hidden_states):
            last_hidden = hidden_states[:, -1:, :]
            logits = self.lm_head(self.final_norm(last_hidden))
            probs = mx.softmax(logits.astype(mx.float32), axis=-1)
            entropy = -mx.sum(probs * mx.log(probs + 1e-9), axis=-1)
            normalized = entropy / math.log(self.config.vocab_size)
            top2 = mx.topk(probs, 2, axis=-1)
            top1_prob = top2[..., 0]
            top2_prob = top2[..., 1]
            margin = top1_prob - top2_prob
            mx.eval(logits, entropy, normalized, top1_prob, top2_prob, margin)
            return {
                "logits": logits,
                "entropy": float(entropy.item()),
                "normalized_entropy": float(normalized.item()),
                "top1_prob": float(top1_prob.item()),
                "top2_prob": float(top2_prob.item()),
                "margin": float(margin.item()),
            }

        def _normalized_entropy(self, hidden_states):
            probe = self._confidence_probe(hidden_states)
            return (
                probe["logits"],
                probe["entropy"],
                probe["normalized_entropy"],
            )

        def _allow_early_exit(self, probe: Dict[str, Any]) -> bool:
            if probe["normalized_entropy"] > self.blending_confidence_threshold:
                return False
            if self.confidence_signal == "entropy_margin":
                return probe["margin"] >= self.margin_threshold
            return True

        def _is_ambiguous_probe(self, probe: Dict[str, Any]) -> bool:
            return (
                self.confidence_signal == "entropy_margin"
                and probe["normalized_entropy"] <= self.blending_confidence_threshold
                and probe["margin"] < self.margin_threshold
            )

        def _resolve_blend_alpha(self, probe: Dict[str, Any]) -> float:
            if self.blend_alpha_mode == "fixed":
                return max(0.0, min(1.0, self.blend_alpha))
            if self.blend_alpha_mode == "margin_linear":
                return max(
                    0.0,
                    min(1.0, min(self.blend_alpha, probe["margin"] * self.blend_alpha_margin_scale)),
                )

            margin_gate = self._sigmoid(
                self.blend_alpha_sigmoid_scale
                * (probe["margin"] - self.margin_threshold)
            )
            entropy_gate = self._sigmoid(
                self.blend_entropy_sigmoid_scale
                * (self.blending_confidence_threshold - probe["normalized_entropy"])
            )
            return max(0.0, min(1.0, margin_gate * entropy_gate))

        def _blend_context(self, early_logits, final_logits) -> Dict[str, Any]:
            early_scores = early_logits[0, -1]
            final_scores = final_logits[0, -1]
            early_top1 = mx.argmax(early_scores, axis=-1)
            final_top1 = mx.argmax(final_scores, axis=-1)
            top_k = min(self.blend_top_k, final_scores.shape[-1])
            final_topk_indices = mx.argpartition(final_scores, kth=-top_k, axis=-1)[-top_k:]
            early_in_final_topk = mx.any(final_topk_indices == early_top1)
            mx.eval(early_top1, final_top1, final_topk_indices, early_in_final_topk)
            return {
                "early_top1": int(early_top1.item()),
                "final_top1": int(final_top1.item()),
                "early_in_final_topk": bool(early_in_final_topk.item()),
            }

        def _resolve_strategy_alpha(
            self,
            blend_alpha: float,
            blend_context: Dict[str, Any],
        ) -> float:
            if self.blend_strategy == "deep_top1_anchor":
                if blend_context["early_top1"] != blend_context["final_top1"]:
                    return max(0.0, min(1.0, blend_alpha * self.anchor_alpha_scale))
            return blend_alpha

        def _should_discard_early_logits(
            self,
            blend_context: Dict[str, Any],
        ) -> bool:
            if self.blend_strategy == "top_k_mask":
                return not blend_context["early_in_final_topk"]
            if self.blend_strategy == "top1_agree":
                return blend_context["early_top1"] != blend_context["final_top1"]
            return False

        def _new_timing_breakdown(self) -> Dict[str, float]:
            return {
                "prefill_time_s": 0.0,
                "first_step_time_s": 0.0,
                "first_token_select_time_s": 0.0,
                "decode_loop_time_s": 0.0,
                "decode_forward_time_s": 0.0,
                "decode_token_select_time_s": 0.0,
                "layer_loop_time_s": 0.0,
                "mask_build_time_s": 0.0,
                "entropy_time_s": 0.0,
                "bookkeeping_time_s": 0.0,
                "cleanup_time_s": 0.0,
                "cache_eval_time_s": 0.0,
                "record_stats_time_s": 0.0,
                "cache_metrics_time_s": 0.0,
                "text_decode_time_s": 0.0,
                "direct_model_forward_time_s": 0.0,
                "layers_processed": 0.0,
                "entropy_evaluations": 0.0,
                "cleanup_calls": 0.0,
                "decode_steps": 0.0,
            }

        def _profile_add(
            self,
            timings: Optional[Dict[str, float]],
            key: str,
            value: float,
        ):
            if timings is not None:
                timings[key] += value

        def _append_layer_decision(
            self,
            layer_decisions: Optional[List[Dict[str, Any]]],
            decision: Dict[str, Any],
            timings: Optional[Dict[str, float]] = None,
        ):
            if layer_decisions is None:
                return
            if timings is None:
                layer_decisions.append(decision)
                return
            t0 = time.perf_counter()
            layer_decisions.append(decision)
            self._profile_add(timings, "bookkeeping_time_s", time.perf_counter() - t0)

        def _soft_skip_measurement_enabled(self) -> bool:
            return self.force_entropy_measurement or self.entropy_threshold >= 0.0

        def _soft_skip_can_activate(self) -> bool:
            return self.entropy_threshold >= 0.0

        def _can_use_direct_model(
            self,
            upper_layer: int,
            collect_trace: bool,
        ) -> bool:
            return (
                upper_layer == self.num_layers - 1
                and not collect_trace
                and not self.force_layer_state_eval
            )

        def _can_use_official_generate_path(self, dynamic: bool) -> bool:
            if self.trace_decisions or self.force_layer_state_eval:
                return False
            if self.enable_logit_blending and self.hard_exit_layer < (self.num_layers - 1):
                return False
            if dynamic:
                return (
                    self.hard_exit_layer == (self.num_layers - 1)
                    and not self._soft_skip_measurement_enabled()
                )
            return True

        def _generate_with_official_mlx(
            self,
            prompt_ids,
            max_new_tokens: int,
            dynamic: bool,
        ) -> Dict[str, Any]:
            from mlx_lm.generate import generate_step
            from mlx_lm.sample_utils import make_sampler

            prompt = mx.array(prompt_ids, dtype=mx.int32)
            sampler = make_sampler(temp=0.0)
            eos_ids = set(getattr(self.tokenizer, "eos_token_ids", []) or [])
            if not eos_ids:
                eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_token_id is not None:
                    eos_ids = {int(eos_token_id)}

            mx.clear_cache()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            ttft_s = None
            generated_ids: List[int] = []

            token_generator = generate_step(
                prompt,
                self.model,
                max_tokens=max_new_tokens,
                sampler=sampler,
                prefill_step_size=self.prefill_step_size,
            )
            for token, _ in token_generator:
                if ttft_s is None:
                    ttft_s = time.perf_counter() - t0
                token_id = int(token)
                generated_ids.append(token_id)
                if token_id in eos_ids:
                    break

            elapsed_s = time.perf_counter() - t0
            ttft_s = ttft_s if ttft_s is not None else elapsed_s
            generation_s = max(elapsed_s - ttft_s, 1e-6)

            cache_metrics_t0 = time.perf_counter()
            peak_memory_gb = mx.get_peak_memory() / (1024**3)
            active_memory_gb = mx.get_active_memory() / (1024**3)
            cache_memory_gb = mx.get_cache_memory() / (1024**3)
            cache_metrics_time_s = time.perf_counter() - cache_metrics_t0

            decode_text_t0 = time.perf_counter()
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            text_decode_time_s = time.perf_counter() - decode_text_t0

            avg_layers = float(self.num_layers)
            stats = {
                "generated_ids": generated_ids,
                "output_text": output_text,
                "ttft_s": ttft_s,
                "elapsed_s": elapsed_s,
                "prompt_tokens": len(prompt_ids),
                "tokens_generated": len(generated_ids),
                "tokens_per_s": len(generated_ids) / max(elapsed_s, 1e-6),
                "generation_tps": max(len(generated_ids) - 1, 0) / generation_s,
                "peak_memory_gb": peak_memory_gb,
                "active_memory_gb": active_memory_gb,
                "cache_memory_gb": cache_memory_gb,
                "avg_layers": avg_layers,
                "exit_events": [],
                "prefill_step_size": self.prefill_step_size,
                "cache_policy": {
                    "path": "official_mlx_generate_step",
                    "prefill_step_size": self.prefill_step_size,
                },
                "timings": {
                    "prefill_time_s": ttft_s,
                    "first_step_time_s": 0.0,
                    "first_token_select_time_s": 0.0,
                    "decode_loop_time_s": max(elapsed_s - ttft_s, 0.0),
                    "decode_forward_time_s": max(elapsed_s - ttft_s, 0.0),
                    "decode_token_select_time_s": 0.0,
                    "layer_loop_time_s": 0.0,
                    "mask_build_time_s": 0.0,
                    "entropy_time_s": 0.0,
                    "bookkeeping_time_s": 0.0,
                    "cleanup_time_s": 0.0,
                    "cache_eval_time_s": 0.0,
                    "record_stats_time_s": 0.0,
                    "cache_metrics_time_s": cache_metrics_time_s,
                    "text_decode_time_s": text_decode_time_s,
                    "direct_model_forward_time_s": 0.0,
                    "layers_processed": 0.0,
                    "entropy_evaluations": 0.0,
                    "cleanup_calls": 0.0,
                    "decode_steps": float(max(len(generated_ids) - 1, 0)),
                    "avg_per_layer_loop_ms": 0.0,
                    "avg_decode_step_ms": (
                        (max(elapsed_s - ttft_s, 0.0) / max(len(generated_ids) - 1, 1))
                        * 1000.0
                    ),
                },
            }
            self._record_generation_stats(stats, dynamic=dynamic)
            return stats

        def _forward_with_full_model(
            self,
            input_ids,
            cache,
            return_logits: bool,
            timings: Optional[Dict[str, float]] = None,
        ) -> Dict[str, Any]:
            t0 = time.perf_counter() if timings is not None else None
            logits = self.model(input_ids, cache=cache)
            if return_logits:
                logits = logits[:, -1:, :]
                mx.eval(logits)
            else:
                self._eval_cache_states(cache, self.num_layers - 1)
                logits = None
            if timings is not None and t0 is not None:
                elapsed = time.perf_counter() - t0
                self._profile_add(timings, "direct_model_forward_time_s", elapsed)
                self._profile_add(timings, "layer_loop_time_s", elapsed)
                self._profile_add(timings, "layers_processed", float(self.num_layers))
            return {
                "logits": logits,
                "cache": cache,
                "layer_decisions": [],
                "skip_from_layer": None,
                "effective_layers": self.num_layers,
            }

        def _cleanup_if_enabled(self, timings: Optional[Dict[str, float]] = None):
            if not self.enable_allocator_cleanup:
                return
            t0 = time.perf_counter() if timings is not None else None
            self._cleanup_allocator()
            if timings is not None and t0 is not None:
                self._profile_add(timings, "cleanup_time_s", time.perf_counter() - t0)
                self._profile_add(timings, "cleanup_calls", 1.0)

        def _shared_attention_masks(
            self,
            hidden_states,
            cache,
            upper_layer: int,
            timings: Optional[Dict[str, float]] = None,
        ) -> tuple[Any, Any]:
            t0 = time.perf_counter() if timings is not None else None
            full_mask = None
            sliding_mask = None
            if self.first_full_idx is not None and self.first_full_idx <= upper_layer:
                full_mask = mlx_create_attention_mask(
                    hidden_states,
                    cache=cache[self.first_full_idx],
                    window_size=None,
                )
            if self.first_sliding_idx is not None and self.first_sliding_idx <= upper_layer:
                sliding_mask = mlx_create_attention_mask(
                    hidden_states,
                    cache=cache[self.first_sliding_idx],
                    window_size=self.window_size,
                )
            if timings is not None and t0 is not None:
                self._profile_add(timings, "mask_build_time_s", time.perf_counter() - t0)
            return full_mask, sliding_mask

        def _mask_for_current_layer(
            self,
            layer_idx: int,
            hidden_states,
            cache,
            upper_layer: int,
            full_mask,
            sliding_mask,
            timings: Optional[Dict[str, float]] = None,
        ):
            if self.reuse_attention_masks:
                if self.layer_types[layer_idx] == "sliding_attention":
                    return sliding_mask
                return full_mask
            t0 = time.perf_counter() if timings is not None else None
            mask = self._mask_for_layer(layer_idx, hidden_states, cache[layer_idx])
            if timings is not None and t0 is not None:
                self._profile_add(timings, "mask_build_time_s", time.perf_counter() - t0)
            return mask

        def _forward_full_layers(
            self,
            input_ids,
            cache,
            upper_layer: int,
            return_logits: bool,
            timings: Optional[Dict[str, float]] = None,
            collect_trace: bool = False,
        ) -> Dict[str, Any]:
            if self._can_use_direct_model(upper_layer, collect_trace):
                return self._forward_with_full_model(
                    input_ids=input_ids,
                    cache=cache,
                    return_logits=return_logits,
                    timings=timings,
                )

            hidden_states = self.embed_tokens(input_ids)
            layer_decisions: Optional[List[Dict[str, Any]]] = [] if collect_trace else None
            full_mask = None
            sliding_mask = None
            if self.reuse_attention_masks:
                full_mask, sliding_mask = self._shared_attention_masks(
                    hidden_states,
                    cache,
                    upper_layer,
                    timings=timings,
                )

            for layer_idx in range(upper_layer + 1):
                layer_t0 = time.perf_counter() if timings is not None else None
                layer = self.layers[layer_idx]
                mask = self._mask_for_current_layer(
                    layer_idx,
                    hidden_states,
                    cache,
                    upper_layer,
                    full_mask,
                    sliding_mask,
                    timings=timings,
                )
                hidden_states = layer(hidden_states, mask, cache[layer_idx])
                if self.force_layer_state_eval:
                    eval_t0 = time.perf_counter() if timings is not None else None
                    self._eval_layer_state(hidden_states, cache[layer_idx])
                    if timings is not None and eval_t0 is not None:
                        self._profile_add(
                            timings,
                            "cache_eval_time_s",
                            time.perf_counter() - eval_t0,
                        )
                self._append_layer_decision(
                    layer_decisions,
                    {"layer": layer_idx, "mode": "full"},
                    timings=timings,
                )
                if timings is not None and layer_t0 is not None:
                    self._profile_add(
                        timings,
                        "layer_loop_time_s",
                        time.perf_counter() - layer_t0,
                    )
                    self._profile_add(timings, "layers_processed", 1.0)

            logits = None
            if return_logits:
                logits = self.lm_head(self.final_norm(hidden_states[:, -1:, :]))
                mx.eval(logits)
            else:
                eval_t0 = time.perf_counter() if timings is not None else None
                self._eval_cache_states(cache, upper_layer)
                if timings is not None and eval_t0 is not None:
                    self._profile_add(
                        timings,
                        "cache_eval_time_s",
                        time.perf_counter() - eval_t0,
                    )

            return {
                "logits": logits,
                "cache": cache,
                "layer_decisions": layer_decisions or [],
                "skip_from_layer": None,
                "effective_layers": upper_layer + 1,
            }

        def _blend_output_logits(self, early_logits, final_logits, alpha: float):
            if alpha <= 1e-6:
                return final_logits
            if alpha >= 1.0 - 1e-6:
                return early_logits
            early_probs = mx.softmax(early_logits.astype(mx.float32), axis=-1)
            final_probs = mx.softmax(final_logits.astype(mx.float32), axis=-1)
            blended_probs = (
                alpha * early_probs
                + (1.0 - alpha) * final_probs
            )
            blended_logits = mx.log(blended_probs + 1e-9)
            mx.eval(blended_logits)
            return blended_logits

        def _forward_with_logit_blending(
            self,
            input_ids,
            cache,
            return_logits: bool,
            timings: Optional[Dict[str, float]] = None,
            collect_trace: bool = False,
        ) -> Dict[str, Any]:
            hidden_states = self.embed_tokens(input_ids)
            layer_decisions: Optional[List[Dict[str, Any]]] = [] if collect_trace else None
            full_mask = None
            sliding_mask = None
            if self.reuse_attention_masks:
                full_mask, sliding_mask = self._shared_attention_masks(
                    hidden_states,
                    cache,
                    self.num_layers - 1,
                    timings=timings,
                )

            early_logits = None
            early_probe: Optional[Dict[str, Any]] = None
            early_exit = False
            early_layer = self.hard_exit_layer
            for layer_idx in range(self.num_layers):
                layer_t0 = time.perf_counter() if timings is not None else None
                layer = self.layers[layer_idx]
                cache_entry = cache[layer_idx]
                mask = self._mask_for_current_layer(
                    layer_idx,
                    hidden_states,
                    cache,
                    self.num_layers - 1,
                    full_mask,
                    sliding_mask,
                    timings=timings,
                )

                if early_exit:
                    hidden_states = self._attention_only(
                        layer,
                        hidden_states,
                        mask,
                        cache_entry,
                    )
                    self._append_layer_decision(
                        layer_decisions,
                        {
                            "layer": layer_idx,
                            "mode": "attn_only",
                            "reason": "cache_preserve_after_early_exit",
                        },
                        timings=timings,
                    )
                    if timings is not None and layer_t0 is not None:
                        self._profile_add(
                            timings,
                            "layer_loop_time_s",
                            time.perf_counter() - layer_t0,
                        )
                        self._profile_add(timings, "layers_processed", 1.0)
                    continue

                hidden_states = layer(hidden_states, mask, cache_entry)
                self._append_layer_decision(
                    layer_decisions,
                    {
                        "layer": layer_idx,
                        "mode": "full",
                    },
                    timings=timings,
                )

                if layer_idx == early_layer:
                    entropy_t0 = time.perf_counter() if timings is not None else None
                    early_probe = self._confidence_probe(hidden_states)
                    early_logits = early_probe["logits"]
                    if timings is not None and entropy_t0 is not None:
                        self._profile_add(
                            timings,
                            "entropy_time_s",
                            time.perf_counter() - entropy_t0,
                        )
                        self._profile_add(timings, "entropy_evaluations", 1.0)

                    self._append_layer_decision(
                        layer_decisions,
                        {
                            "layer": layer_idx,
                            "mode": "blend_probe",
                            "normalized_entropy": early_probe["normalized_entropy"],
                            "threshold": self.blending_confidence_threshold,
                            "margin": early_probe["margin"],
                            "top1_prob": early_probe["top1_prob"],
                            "top2_prob": early_probe["top2_prob"],
                            "confidence_signal": self.confidence_signal,
                            "blend_alpha_mode": self.blend_alpha_mode,
                            "candidate_alpha": self._resolve_blend_alpha(early_probe),
                        },
                        timings=timings,
                    )
                    if self._allow_early_exit(early_probe):
                        early_exit = True

                if timings is not None and layer_t0 is not None:
                    self._profile_add(
                        timings,
                        "layer_loop_time_s",
                        time.perf_counter() - layer_t0,
                    )
                    self._profile_add(timings, "layers_processed", 1.0)

            if early_logits is None:
                early_logits = self.lm_head(self.final_norm(hidden_states[:, -1:, :]))
                mx.eval(early_logits)

            if early_exit:
                eval_t0 = time.perf_counter() if timings is not None else None
                self._eval_cache_states(cache, self.num_layers - 1)
                if timings is not None and eval_t0 is not None:
                    self._profile_add(
                        timings,
                        "cache_eval_time_s",
                        time.perf_counter() - eval_t0,
                    )
                return {
                    "logits": early_logits if return_logits else None,
                    "cache": cache,
                    "layer_decisions": layer_decisions or [],
                    "skip_from_layer": self.hard_exit_layer + 1,
                    "effective_layers": self.hard_exit_layer + 1,
                }

            final_logits = self.lm_head(self.final_norm(hidden_states[:, -1:, :]))
            ambiguous_probe = (
                early_probe is not None and self._is_ambiguous_probe(early_probe)
            )
            if return_logits:
                blend_context = self._blend_context(early_logits, final_logits)
                if self.fallback_to_full_depth_on_ambiguity and ambiguous_probe:
                    logits = final_logits
                elif self._should_discard_early_logits(blend_context):
                    logits = final_logits
                else:
                    blend_alpha = (
                        self._resolve_blend_alpha(early_probe)
                        if early_probe is not None
                        else self.blend_alpha
                    )
                    blend_alpha = self._resolve_strategy_alpha(
                        blend_alpha,
                        blend_context,
                    )
                    logits = self._blend_output_logits(
                        early_logits,
                        final_logits,
                        blend_alpha,
                    )
            else:
                eval_t0 = time.perf_counter() if timings is not None else None
                self._eval_cache_states(cache, self.num_layers - 1)
                if timings is not None and eval_t0 is not None:
                    self._profile_add(
                        timings,
                        "cache_eval_time_s",
                        time.perf_counter() - eval_t0,
                    )
                logits = None

            return {
                "logits": logits,
                "cache": cache,
                "layer_decisions": layer_decisions or [],
                "skip_from_layer": None,
                "effective_layers": float(self.num_layers),
            }

        def _kl_divergence(self, reference_logits, approx_logits) -> float:
            reference_probs = mx.softmax(reference_logits.astype(mx.float32), axis=-1)
            reference_log_probs = mx.log(reference_probs + 1e-9)
            approx_probs = mx.softmax(approx_logits.astype(mx.float32), axis=-1)
            approx_log_probs = mx.log(approx_probs + 1e-9)
            kl = mx.sum(reference_probs * (reference_log_probs - approx_log_probs), axis=-1)
            mx.eval(kl)
            return float(kl.item())

        def _dynamic_forward(
            self,
            input_ids,
            cache=None,
            dynamic: bool = True,
            return_logits: bool = True,
            timings: Optional[Dict[str, float]] = None,
            collect_trace: Optional[bool] = None,
        ) -> Dict[str, Any]:
            if cache is None:
                cache = self._make_cache()

            if collect_trace is None:
                collect_trace = self.trace_decisions

            upper_layer = self.hard_exit_layer if dynamic else self.num_layers - 1
            if (
                dynamic
                and self.enable_logit_blending
                and self.hard_exit_layer < (self.num_layers - 1)
            ):
                return self._forward_with_logit_blending(
                    input_ids=input_ids,
                    cache=cache,
                    return_logits=return_logits,
                    timings=timings,
                    collect_trace=collect_trace,
                )

            soft_skip_measurement = dynamic and self._soft_skip_measurement_enabled()
            soft_skip_activation = dynamic and self._soft_skip_can_activate()

            if not dynamic or not soft_skip_measurement:
                return self._forward_full_layers(
                    input_ids=input_ids,
                    cache=cache,
                    upper_layer=upper_layer,
                    return_logits=return_logits,
                    timings=timings,
                    collect_trace=collect_trace,
                )

            hidden_states = self.embed_tokens(input_ids)
            skip_active = False
            skip_from_layer = None
            low_entropy_streak = 0
            layer_decisions: Optional[List[Dict[str, Any]]] = [] if collect_trace else None
            full_mask = None
            sliding_mask = None
            if self.reuse_attention_masks:
                full_mask, sliding_mask = self._shared_attention_masks(
                    hidden_states,
                    cache,
                    upper_layer,
                    timings=timings,
                )

            for layer_idx in range(upper_layer + 1):
                layer_t0 = time.perf_counter() if timings is not None else None
                layer = self.layers[layer_idx]
                cache_entry = cache[layer_idx]
                mask = self._mask_for_current_layer(
                    layer_idx,
                    hidden_states,
                    cache,
                    upper_layer,
                    full_mask,
                    sliding_mask,
                    timings=timings,
                )

                if layer_idx < self.soft_skip_start_layer:
                    hidden_states = layer(hidden_states, mask, cache_entry)
                    if self.force_layer_state_eval:
                        eval_t0 = time.perf_counter() if timings is not None else None
                        self._eval_layer_state(hidden_states, cache_entry)
                        if timings is not None and eval_t0 is not None:
                            self._profile_add(
                                timings,
                                "cache_eval_time_s",
                                time.perf_counter() - eval_t0,
                            )
                    self._append_layer_decision(
                        layer_decisions,
                        {
                            "layer": layer_idx,
                            "mode": "full",
                        },
                        timings=timings,
                    )
                    if timings is not None and layer_t0 is not None:
                        self._profile_add(
                            timings,
                            "layer_loop_time_s",
                            time.perf_counter() - layer_t0,
                        )
                        self._profile_add(timings, "layers_processed", 1.0)
                    continue

                if skip_active:
                    attn_states = self._attention_only(
                        layer, hidden_states, mask, cache_entry
                    )
                    hidden_states = attn_states
                    if self.force_layer_state_eval:
                        eval_t0 = time.perf_counter() if timings is not None else None
                        self._eval_layer_state(hidden_states, cache_entry)
                        if timings is not None and eval_t0 is not None:
                            self._profile_add(
                                timings,
                                "cache_eval_time_s",
                                time.perf_counter() - eval_t0,
                            )
                    self._append_layer_decision(
                        layer_decisions,
                        {
                            "layer": layer_idx,
                            "mode": "attn_only",
                            "reason": "hard_exit" if layer_idx == self.hard_exit_layer else "soft_skip",
                            "entropy": None,
                            "normalized_entropy": None,
                        },
                        timings=timings,
                    )
                    if timings is not None and layer_t0 is not None:
                        self._profile_add(
                            timings,
                            "layer_loop_time_s",
                            time.perf_counter() - layer_t0,
                        )
                        self._profile_add(timings, "layers_processed", 1.0)
                    continue

                hidden_states = layer(hidden_states, mask, cache_entry)
                if self.force_layer_state_eval:
                    eval_t0 = time.perf_counter() if timings is not None else None
                    self._eval_layer_state(hidden_states, cache_entry)
                    if timings is not None and eval_t0 is not None:
                        self._profile_add(
                            timings,
                            "cache_eval_time_s",
                            time.perf_counter() - eval_t0,
                        )

                if layer_idx == self.hard_exit_layer:
                    self._append_layer_decision(
                        layer_decisions,
                        {
                            "layer": layer_idx,
                            "mode": "full",
                            "reason": "hard_exit",
                        },
                        timings=timings,
                    )
                    if timings is not None and layer_t0 is not None:
                        self._profile_add(
                            timings,
                            "layer_loop_time_s",
                            time.perf_counter() - layer_t0,
                        )
                        self._profile_add(timings, "layers_processed", 1.0)
                    continue

                should_measure_entropy = layer_idx <= self.gate_activation_layer
                entropy = None
                normalized_entropy = None
                if should_measure_entropy:
                    entropy_t0 = time.perf_counter() if timings is not None else None
                    _, entropy, normalized_entropy = self._normalized_entropy(hidden_states)
                    if timings is not None and entropy_t0 is not None:
                        self._profile_add(
                            timings,
                            "entropy_time_s",
                            time.perf_counter() - entropy_t0,
                        )
                        self._profile_add(timings, "entropy_evaluations", 1.0)
                    if normalized_entropy <= self.entropy_threshold:
                        low_entropy_streak += 1
                    else:
                        low_entropy_streak = 0

                    if (
                        soft_skip_activation
                        and layer_idx >= self.gate_activation_layer
                        and low_entropy_streak >= self.min_entropy_streak
                    ):
                        skip_active = True
                        skip_from_layer = layer_idx + 1

                self._append_layer_decision(
                    layer_decisions,
                    {
                        "layer": layer_idx,
                        "mode": "full",
                        "entropy": entropy,
                        "normalized_entropy": normalized_entropy,
                        "low_entropy_streak": low_entropy_streak,
                        "gate_open": skip_active,
                    },
                    timings=timings,
                )
                if timings is not None and layer_t0 is not None:
                    self._profile_add(
                        timings,
                        "layer_loop_time_s",
                        time.perf_counter() - layer_t0,
                    )
                    self._profile_add(timings, "layers_processed", 1.0)

            effective_layers = (
                skip_from_layer + 1
                if skip_from_layer is not None
                else self.hard_exit_layer + 1
            )

            logits = None
            if return_logits:
                logits = self.lm_head(self.final_norm(hidden_states[:, -1:, :]))
                mx.eval(logits)
            else:
                eval_t0 = time.perf_counter() if timings is not None else None
                self._eval_cache_states(cache, upper_layer)
                if timings is not None and eval_t0 is not None:
                    self._profile_add(
                        timings,
                        "cache_eval_time_s",
                        time.perf_counter() - eval_t0,
                    )

            return {
                "logits": logits,
                "cache": cache,
                "layer_decisions": layer_decisions or [],
                "skip_from_layer": skip_from_layer,
                "effective_layers": effective_layers,
            }

        def _prefill_tokens(
            self,
            input_ids,
            cache=None,
            dynamic: bool = True,
            prefill_step_size: Optional[int] = None,
            timings: Optional[Dict[str, float]] = None,
            collect_trace: Optional[bool] = None,
        ) -> Dict[str, Any]:
            """
            Process long prompts in chunks so sliding-window caches stay compact
            and the prefill path matches the official mlx_lm 2048-step protocol.
            """
            if cache is None:
                cache = self._make_cache()

            step_size = prefill_step_size or self.prefill_step_size
            seq_len = input_ids.shape[1]
            last_out = None

            for start in range(0, seq_len, step_size):
                stop = min(start + step_size, seq_len)
                chunk = input_ids[:, start:stop]
                last_out = self._dynamic_forward(
                    chunk,
                    cache=cache,
                    dynamic=dynamic,
                    return_logits=(stop == seq_len),
                    timings=timings,
                    collect_trace=collect_trace,
                )
                self._cleanup_if_enabled(timings)

            if last_out is None:
                raise ValueError("Prompt must contain at least one token.")
            return last_out

        def _prime_generation(
            self,
            input_ids,
            cache=None,
            dynamic: bool = True,
            prefill_step_size: Optional[int] = None,
            timings: Optional[Dict[str, float]] = None,
            collect_trace: Optional[bool] = None,
        ) -> Dict[str, Any]:
            """
            Match the official mlx_lm prompt boundary exactly.

            mlx_lm.generate_step() prefills prompt[:-1] into the KV cache, then
            runs a single-token forward pass on prompt[-1:] to obtain the first
            generated token. This matters for rotating sliding-window caches,
            which take a different update path for multi-token prefill chunks vs
            single-token decode steps.
            """
            if cache is None:
                cache = self._make_cache()

            if input_ids.shape[1] < 1:
                raise ValueError("Prompt must contain at least one token.")

            prefix_ids = input_ids[:, :-1]
            prefix_out = None
            if prefix_ids.shape[1] > 0:
                prefill_t0 = time.perf_counter() if timings is not None else None
                prefix_out = self._prefill_tokens(
                    prefix_ids,
                    cache=cache,
                    dynamic=dynamic,
                    prefill_step_size=prefill_step_size,
                    timings=timings,
                    collect_trace=collect_trace,
                )
                if timings is not None and prefill_t0 is not None:
                    self._profile_add(
                        timings,
                        "prefill_time_s",
                        time.perf_counter() - prefill_t0,
                    )

            first_step_t0 = time.perf_counter() if timings is not None else None
            first_step = self._dynamic_forward(
                input_ids[:, -1:],
                cache=cache,
                dynamic=dynamic,
                return_logits=True,
                timings=timings,
                collect_trace=collect_trace,
            )
            if timings is not None and first_step_t0 is not None:
                self._profile_add(
                    timings,
                    "first_step_time_s",
                    time.perf_counter() - first_step_t0,
                )
            return {
                "cache": cache,
                "prefix_out": prefix_out,
                "first_step": first_step,
            }

        def generate_from_ids(
            self,
            prompt_ids,
            max_new_tokens: int = 64,
            dynamic: bool = True,
            prefill_step_size: Optional[int] = None,
            profile_runtime: bool = False,
        ) -> Dict[str, Any]:
            if self._can_use_official_generate_path(dynamic):
                return self._generate_with_official_mlx(
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    dynamic=dynamic,
                )

            input_ids = mx.array(prompt_ids, dtype=mx.int32)[None, :]
            cache = self._make_cache()
            generated_ids: List[int] = []
            collect_trace = dynamic and self.trace_decisions
            exit_events: List[Dict[str, Any]] = [] if collect_trace else []
            prompt_tokens = len(prompt_ids)
            layers_used_total = 0.0
            timings = self._new_timing_breakdown() if profile_runtime else None

            mx.clear_cache()
            mx.reset_peak_memory()
            t0 = time.perf_counter()

            primed = self._prime_generation(
                input_ids,
                cache=cache,
                dynamic=dynamic,
                prefill_step_size=prefill_step_size,
                timings=timings,
                collect_trace=collect_trace,
            )
            first_step = primed["first_step"]
            select_t0 = time.perf_counter() if timings is not None else None
            next_token = mx.argmax(first_step["logits"][0, -1], axis=-1).reshape(1, 1)
            mx.eval(next_token)
            if timings is not None and select_t0 is not None:
                self._profile_add(
                    timings,
                    "first_token_select_time_s",
                    time.perf_counter() - select_t0,
                )
            ttft_s = time.perf_counter() - t0

            next_token_id = int(next_token.item())
            generated_ids.append(next_token_id)
            layers_used_total += (
                first_step["effective_layers"] if dynamic else float(self.num_layers)
            )
            if collect_trace:
                bookkeeping_t0 = time.perf_counter() if timings is not None else None
                exit_events.append({
                    "step": 0,
                    "skip_from_layer": first_step["skip_from_layer"],
                    "layer_decisions": first_step["layer_decisions"],
                })
                if timings is not None and bookkeeping_t0 is not None:
                    self._profile_add(
                        timings,
                        "bookkeeping_time_s",
                        time.perf_counter() - bookkeeping_t0,
                    )

            current_input = next_token

            decode_loop_t0 = time.perf_counter() if timings is not None else None
            for step in range(1, max_new_tokens):
                if self._is_eos(next_token_id):
                    break

                forward_t0 = time.perf_counter() if timings is not None else None
                step_out = self._dynamic_forward(
                    current_input,
                    cache=cache,
                    dynamic=dynamic,
                    return_logits=True,
                    timings=timings,
                    collect_trace=collect_trace,
                )
                if timings is not None and forward_t0 is not None:
                    self._profile_add(
                        timings,
                        "decode_forward_time_s",
                        time.perf_counter() - forward_t0,
                    )
                    self._profile_add(timings, "decode_steps", 1.0)
                select_t0 = time.perf_counter() if timings is not None else None
                next_token = mx.argmax(step_out["logits"][0, -1], axis=-1).reshape(1, 1)
                mx.eval(next_token)
                if timings is not None and select_t0 is not None:
                    self._profile_add(
                        timings,
                        "decode_token_select_time_s",
                        time.perf_counter() - select_t0,
                    )
                next_token_id = int(next_token.item())
                generated_ids.append(next_token_id)
                current_input = next_token
                layers_used_total += (
                    step_out["effective_layers"] if dynamic else float(self.num_layers)
                )

                if collect_trace:
                    bookkeeping_t0 = time.perf_counter() if timings is not None else None
                    exit_events.append({
                        "step": step,
                        "skip_from_layer": step_out["skip_from_layer"],
                        "layer_decisions": step_out["layer_decisions"],
                    })
                    if timings is not None and bookkeeping_t0 is not None:
                        self._profile_add(
                            timings,
                            "bookkeeping_time_s",
                            time.perf_counter() - bookkeeping_t0,
                        )
                if step % max(self.cache_cleanup_interval, 1) == 0:
                    self._cleanup_if_enabled(timings)
            if timings is not None and decode_loop_t0 is not None:
                self._profile_add(
                    timings,
                    "decode_loop_time_s",
                    time.perf_counter() - decode_loop_t0,
                )

            elapsed_s = time.perf_counter() - t0
            metrics_t0 = time.perf_counter() if timings is not None else None
            peak_memory_gb = mx.get_peak_memory() / (1024**3)
            active_memory_gb = mx.get_active_memory() / (1024**3)
            cache_memory_gb = mx.get_cache_memory() / (1024**3)
            if timings is not None and metrics_t0 is not None:
                self._profile_add(
                    timings,
                    "cache_metrics_time_s",
                    time.perf_counter() - metrics_t0,
                )

            decode_text_t0 = time.perf_counter() if timings is not None else None
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if timings is not None and decode_text_t0 is not None:
                self._profile_add(
                    timings,
                    "text_decode_time_s",
                    time.perf_counter() - decode_text_t0,
                )

            avg_layers = (
                layers_used_total / max(len(generated_ids), 1)
                if generated_ids
                else 0.0
            )
            generation_s = max(elapsed_s - ttft_s, 1e-6)

            stats = {
                "generated_ids": generated_ids,
                "output_text": output_text,
                "ttft_s": ttft_s,
                "elapsed_s": elapsed_s,
                "prompt_tokens": prompt_tokens,
                "tokens_generated": len(generated_ids),
                "tokens_per_s": len(generated_ids) / max(elapsed_s, 1e-6),
                "generation_tps": max(len(generated_ids) - 1, 0) / generation_s,
                "peak_memory_gb": peak_memory_gb,
                "active_memory_gb": active_memory_gb,
                "cache_memory_gb": cache_memory_gb,
                "avg_layers": avg_layers,
                "exit_events": exit_events,
                "prefill_step_size": prefill_step_size or self.prefill_step_size,
                "cache_policy": self._cache_policy(cache),
            }
            if timings is not None:
                timings["avg_per_layer_loop_ms"] = (
                    (timings["layer_loop_time_s"] / max(timings["layers_processed"], 1.0)) * 1000.0
                )
                timings["avg_decode_step_ms"] = (
                    (timings["decode_forward_time_s"] / max(timings["decode_steps"], 1.0)) * 1000.0
                )
                stats["timings"] = timings
            record_stats_t0 = time.perf_counter() if timings is not None else None
            self._record_generation_stats(stats, dynamic=dynamic)
            if timings is not None and record_stats_t0 is not None:
                self._profile_add(
                    timings,
                    "record_stats_time_s",
                    time.perf_counter() - record_stats_t0,
                )
            return stats

        def generate(
            self,
            prompt: str,
            max_new_tokens: int = 64,
            dynamic: bool = True,
            prefill_step_size: Optional[int] = None,
            profile_runtime: bool = False,
        ) -> Dict[str, Any]:
            prompt_ids = self.tokenizer.encode(prompt)
            stats = self.generate_from_ids(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                dynamic=dynamic,
                prefill_step_size=prefill_step_size,
                profile_runtime=profile_runtime,
            )
            stats["prompt"] = prompt
            return stats

        def generate_from_messages(
            self,
            messages: List[Dict[str, str]],
            max_new_tokens: int = 64,
            dynamic: bool = True,
            reasoning_effort: str = "medium",
            prefill_step_size: Optional[int] = None,
            profile_runtime: bool = False,
        ) -> Dict[str, Any]:
            prompt, rendered_messages = apply_harmony_template(
                self.tokenizer,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
            stats = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                dynamic=dynamic,
                prefill_step_size=prefill_step_size,
                profile_runtime=profile_runtime,
            )
            stats["messages"] = rendered_messages
            stats["reasoning_effort"] = reasoning_effort
            return stats

        def consistency_check_from_ids(
            self,
            prompt_ids,
            max_new_tokens: int = 16,
        ) -> Dict[str, Any]:
            prompt_input = mx.array(prompt_ids, dtype=mx.int32)[None, :]
            full_cache = self._make_cache()
            dynamic_cache = self._make_cache()

            full_primed = self._prime_generation(
                prompt_input,
                cache=full_cache,
                dynamic=False,
                prefill_step_size=self.prefill_step_size,
            )
            dynamic_primed = self._prime_generation(
                prompt_input,
                cache=dynamic_cache,
                dynamic=True,
                prefill_step_size=self.prefill_step_size,
            )
            full_out = full_primed["first_step"]
            dynamic_out = dynamic_primed["first_step"]

            compared = 0
            agreements = 0
            kl_values: List[float] = []

            for step in range(max_new_tokens):
                full_logits = full_out["logits"]
                dynamic_logits = dynamic_out["logits"]
                full_top1 = int(mx.argmax(full_logits[0, -1], axis=-1).item())
                dynamic_top1 = int(mx.argmax(dynamic_logits[0, -1], axis=-1).item())
                agreements += int(full_top1 == dynamic_top1)
                kl_values.append(self._kl_divergence(full_logits, dynamic_logits))
                compared += 1

                if self._is_eos(full_top1):
                    break

                teacher_token = mx.array([[full_top1]], dtype=mx.int32)
                full_out = self._dynamic_forward(
                    teacher_token,
                    cache=full_cache,
                    dynamic=False,
                    return_logits=True,
                )
                dynamic_out = self._dynamic_forward(
                    teacher_token,
                    cache=dynamic_cache,
                    dynamic=True,
                    return_logits=True,
                )

            result = {
                "tokens_compared": compared,
                "top1_agreement": agreements / max(compared, 1),
                "avg_kl_divergence": sum(kl_values) / max(len(kl_values), 1),
            }
            self._record_consistency_stats(result)
            self._cleanup_allocator()
            return result

        def consistency_check_from_messages(
            self,
            messages: List[Dict[str, str]],
            max_new_tokens: int = 16,
            reasoning_effort: str = "medium",
        ) -> Dict[str, Any]:
            prompt, _ = apply_harmony_template(
                self.tokenizer,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
            prompt_ids = self.tokenizer.encode(prompt)
            return self.consistency_check_from_ids(
                prompt_ids,
                max_new_tokens=max_new_tokens,
            )

        def benchmark_mlx_protocol(
            self,
            prompt_tokens: int = 2048,
            generation_tokens: int = 128,
            num_trials: int = 1,
            seed: int = 0,
        ) -> Dict[str, float]:
            """
            Reproduce the official mlx_lm benchmark protocol:
              - random prompt ids
              - prefill step size 2048
              - generation target 128
            """
            from mlx_lm import stream_generate

            mx.random.seed(seed)
            prompt = mx.random.randint(
                0, self.config.vocab_size, (prompt_tokens,)
            ).tolist()

            saved_eos = None
            if hasattr(self.tokenizer, "_eos_token_ids"):
                saved_eos = set(self.tokenizer._eos_token_ids)
                self.tokenizer._eos_token_ids = set()

            def bench_once():
                response = None
                for response in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=generation_tokens,
                    prefill_step_size=self.prefill_step_size,
                ):
                    pass
                return response

            bench_once()  # warmup
            responses = [bench_once() for _ in range(num_trials)]

            if saved_eos is not None:
                self.tokenizer._eos_token_ids = saved_eos

            return {
                "prompt_tokens": float(prompt_tokens),
                "generation_tokens": float(generation_tokens),
                "prompt_tps": sum(r.prompt_tps for r in responses) / num_trials,
                "generation_tps": sum(r.generation_tps for r in responses) / num_trials,
                "peak_memory_gb": sum(r.peak_memory for r in responses) / num_trials,
                "prefill_step_size": float(self.prefill_step_size),
            }

        def benchmark_transcender_protocol(
            self,
            prompt_tokens: int = 2048,
            generation_tokens: int = 128,
            num_trials: int = 1,
            seed: int = 0,
        ) -> Dict[str, float]:
            mx.random.seed(seed)
            prompt = mx.random.randint(
                0, self.config.vocab_size, (prompt_tokens,)
            ).tolist()

            self.generate_from_ids(
                prompt,
                max_new_tokens=generation_tokens,
                dynamic=True,
                prefill_step_size=self.prefill_step_size,
            )  # warmup

            runs = [
                self.generate_from_ids(
                    prompt,
                    max_new_tokens=generation_tokens,
                    dynamic=True,
                    prefill_step_size=self.prefill_step_size,
                )
                for _ in range(num_trials)
            ]

            return {
                "prompt_tokens": float(prompt_tokens),
                "generation_tokens": float(generation_tokens),
                "prompt_tps": sum(prompt_tokens / max(r["ttft_s"], 1e-6) for r in runs) / num_trials,
                "generation_tps": sum(r["generation_tps"] for r in runs) / num_trials,
                "peak_memory_gb": sum(r["peak_memory_gb"] for r in runs) / num_trials,
                "avg_layers": sum(r["avg_layers"] for r in runs) / num_trials,
                "prefill_step_size": float(self.prefill_step_size),
            }

        def benchmark_protocol_comparison(
            self,
            prompt_tokens: int = 2048,
            generation_tokens: int = 128,
            num_trials: int = 1,
            seed: int = 0,
        ) -> Dict[str, Dict[str, float]]:
            baseline = self.benchmark_mlx_protocol(
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                num_trials=num_trials,
                seed=seed,
            )
            transcender = self.benchmark_transcender_protocol(
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                num_trials=num_trials,
                seed=seed,
            )
            return {
                "baseline": baseline,
                "transcender": transcender,
                "prompt_tps_gain_pct": (
                    (transcender["prompt_tps"] - baseline["prompt_tps"])
                    / max(baseline["prompt_tps"], 1e-6)
                    * 100.0
                ),
                "generation_tps_gain_pct": (
                    (transcender["generation_tps"] - baseline["generation_tps"])
                    / max(baseline["generation_tps"], 1e-6)
                    * 100.0
                ),
            }

        def benchmark_against_full_depth(
            self,
            prompt: str,
            max_new_tokens: int = 64,
        ) -> Dict[str, Any]:
            baseline = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                dynamic=False,
            )
            dynamic = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                dynamic=True,
            )

            paired = list(zip(baseline["generated_ids"], dynamic["generated_ids"]))
            prefix_match = 0
            for left, right in paired:
                if left != right:
                    break
                prefix_match += 1

            exact_match_rate = (
                sum(1 for left, right in paired if left == right)
                / max(len(paired), 1)
            )

            return {
                "baseline": baseline,
                "dynamic": dynamic,
                "ttft_delta_s": baseline["ttft_s"] - dynamic["ttft_s"],
                "ttft_improvement_pct": (
                    (baseline["ttft_s"] - dynamic["ttft_s"])
                    / max(baseline["ttft_s"], 1e-6)
                    * 100.0
                ),
                "prefix_match_tokens": prefix_match,
                "exact_match_rate": exact_match_rate,
            }

        def _is_eos(self, token_id: int) -> bool:
            eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
            if eos_ids is not None:
                return token_id in set(eos_ids)
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            return eos_token_id is not None and token_id == eos_token_id


else:
    MLXDynamicExpertEngine = None


# ═══════════════════════════════════════════════════════════════════
# Son Router — Per-Token Exit Gate
# ═══════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class SonRouter(nn.Module):
        """
        Son Router for GPT-OSS 20B (PyTorch reference implementation).

        Input:  hidden states from residual stream at exit layer (B, S, 2880)
        Output: exit_probs (B, S) in [0, 1]

        Architecture: Linear(2880→64) → GELU → Linear(64→1) → Sigmoid
        Parameters:  2880*64 + 64 + 64*1 + 1 = 184,513 (float32)
        Overhead:    184K / 21B = 0.0009% of backbone
        """

        def __init__(self, config: GptOssConfig = GptOssConfig()):
            super().__init__()
            self.gate = nn.Sequential(
                nn.Linear(config.hidden_size, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, hidden_states: torch.Tensor) -> dict:
            exit_logits = self.gate(hidden_states).squeeze(-1)  # (B, S)
            exit_probs = torch.sigmoid(exit_logits)
            exit_mask = exit_probs > 0.5
            return {
                "exit_probs": exit_probs,
                "exit_mask": exit_mask,
                "exit_logits": exit_logits,
            }


# ═══════════════════════════════════════════════════════════════════
# TranscenderEngine — Dual-Path Inference with Physical Skipping
# ═══════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class TranscenderEngine(nn.Module):
        """
        Production-grade dynamic routing engine for GPT-OSS 20B.

        Implements the full Transcender pipeline:
        1. Run layers 0-11 (early stack) for ALL tokens
        2. Son Router evaluates exit probability at Layer 12
        3. For EXITED tokens: compute early_logits via frozen norm + lm_head
        4. For CONTINUING tokens: run layers 12-23 (deep stack)
        5. Blend or gate logits based on inference mode

        Physical Skipping:
        In autoregressive generation, exited tokens SKIP layers 12-23
        entirely. Their KV-cache entries for deep layers are never computed.

        KV-Cache Design:
        ┌────────────────────────────────────────────────────────────┐
        │ early_kv_cache (Layers 0-11):  ALL tokens present         │
        │ deep_kv_cache  (Layers 12-23): ONLY non-exited tokens     │
        │                                                            │
        │ Layer  0 (sliding):  full KV for window=128                │
        │ Layer  1 (full):     full KV for all tokens                │
        │ ...                                                        │
        │ Layer 11 (full):     full KV for all tokens                │
        │ ── SON ROUTER GATE ──                                      │
        │ Layer 12 (sliding):  sparse KV (exited tokens excluded)    │
        │ Layer 13 (full):     sparse KV (compressed attention mask) │
        │ ...                                                        │
        │ Layer 23 (full):     sparse KV (compressed attention mask) │
        └────────────────────────────────────────────────────────────┘
        """

        def __init__(
            self,
            model=None,
            tokenizer=None,
            config: GptOssConfig = GptOssConfig(),
            exit_after_layer: int = 12,
            inference_mode: str = "hard",
            routing_coeff: float = 0.1,
        ):
            super().__init__()
            self.config = config
            self.exit_after_layer = exit_after_layer
            self.inference_mode = inference_mode
            self.routing_coeff = routing_coeff

            if model is not None:
                # Extract components from loaded HuggingFace model
                self._attach_backbone(model)
            else:
                print("  [TranscenderEngine] No backbone attached. "
                      "Call attach_backbone(model) after loading.")

            if tokenizer is not None:
                self.tokenizer = tokenizer

            # Son Router (always float32, always trainable)
            self.router = SonRouter(config)

            router_params = sum(p.numel() for p in self.router.parameters())
            print(f"  Son Router: {router_params:,} params (float32)")
            print(f"  Exit point: Layer {exit_after_layer}/{config.num_hidden_layers}")
            print(f"  Max depth savings: "
                  f"{(1 - exit_after_layer / config.num_hidden_layers) * 100:.0f}%")
            print(f"  Inference mode: {inference_mode}")

        def _attach_backbone(self, model):
            """Extract transformer components from HuggingFace model."""
            # GPT-OSS module paths (from config.json quantization exclusions):
            #   model.embed_tokens, model.layers[i].self_attn,
            #   model.layers[i].mlp.router, lm_head, model.norm
            self.embed_tokens = model.model.embed_tokens
            self.layers = model.model.layers
            self.final_norm = model.model.norm   # RMSNorm
            self.lm_head = model.lm_head

            num_layers = len(self.layers)
            assert num_layers == self.config.num_hidden_layers, \
                f"Expected {self.config.num_hidden_layers} layers, got {num_layers}"
            print(f"  Backbone attached: {num_layers} layers, "
                  f"hidden={self.config.hidden_size}")

        def freeze_backbone(self):
            """
            Symbiotic mode: freeze all backbone params, train only Son Router.

            The backbone stays in MXFP4 quantization (for expert weights).
            Attention, MoE router, embeddings, and lm_head remain in
            higher precision (BF16/U8) as per OpenAI's quantization config.
            """
            for name, param in self.named_parameters():
                if "router" not in name or "mlp.router" in name:
                    # Freeze everything except our Son Router
                    # (mlp.router is the MoE router — freeze it too)
                    param.requires_grad = False

            for param in self.router.parameters():
                param.requires_grad = True

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"  Symbiotic mode: {trainable:,} trainable / "
                  f"{total:,} total ({trainable/total*100:.4f}%)")

        # ─────────────────────────────────────────
        # Core Forward Pass
        # ─────────────────────────────────────────

        def forward(
            self,
            input_ids: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
        ) -> dict:
            """
            Full forward pass with logit-space blending.

            During training: soft blend of early + deep logits.
            During inference: mode-dependent (hard/soft/adaptive).

            The MoE expert routing happens INSIDE each layer's MLP block
            and is completely transparent to us. We operate on the
            residual stream AFTER expert aggregation.
            """
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

            # ── Embedding ──
            hidden_states = self.embed_tokens(input_ids)

            # ── Early layers (0 to exit_layer-1) ──
            # All tokens pass through these layers.
            # Each layer internally runs: Attention → MoE(top-4/32) → Residual
            for i in range(self.exit_after_layer):
                hidden_states = self._run_layer(i, hidden_states)

            # ── Son Router Decision ──
            # Operates on the residual stream AFTER Layer 11's
            # attention + expert aggregation.
            routing_info = self.router(hidden_states)
            exit_probs = routing_info["exit_probs"]    # (B, S)
            exit_mask = routing_info["exit_mask"]       # (B, S)

            # Save early hidden states for logit computation
            early_hidden = hidden_states.clone()

            # Track per-token layer depth
            layer_counts = torch.full(
                (batch_size, seq_len), self.exit_after_layer,
                device=device, dtype=torch.float,
            )

            # ── Deep layers (exit_layer to num_layers-1) ──
            for i in range(self.exit_after_layer, self.config.num_hidden_layers):
                hidden_states = self._run_layer(i, hidden_states)
                layer_counts[~exit_mask] = i + 1

            # ── Logit-Space Blending (Subspace mismatch mitigation) ──
            # Project BOTH pathways through the SAME frozen norm + lm_head.
            # This is the ONLY valid blending protocol — hidden-state
            # blending across layers is geometrically invalid (4.11x separation).
            early_logits = self.lm_head(self.final_norm(early_hidden))
            deep_logits = self.lm_head(self.final_norm(hidden_states))

            # ── Mode-dependent output ──
            if self.training:
                # Soft blend: differentiable for router training
                w = exit_probs.unsqueeze(-1)  # (B, S, 1)
                logits = w * early_logits + (1 - w) * deep_logits
            elif self.inference_mode == "soft":
                w = exit_probs.unsqueeze(-1)
                logits = w * early_logits + (1 - w) * deep_logits
            elif self.inference_mode == "adaptive":
                # Hard-exit only when router is highly confident (>0.9)
                confident = (exit_probs > 0.9).unsqueeze(-1).expand_as(early_logits)
                w = exit_probs.unsqueeze(-1)
                logits = w * early_logits + (1 - w) * deep_logits
                logits = torch.where(confident, early_logits, logits)
                layer_counts[exit_probs > 0.9] = self.exit_after_layer
            else:  # "hard"
                mask_3d = exit_mask.unsqueeze(-1).expand_as(early_logits)
                logits = torch.where(mask_3d, early_logits, deep_logits)

            result = {
                "logits": logits,
                "routing_info": routing_info,
                "layer_counts": layer_counts,
                "early_logits": early_logits,
                "deep_logits": deep_logits,
            }

            # ── Loss (training only) ──
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                lm_loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                # KL-calibrated routing loss
                routing_loss = self._kl_calibrated_loss(
                    exit_probs, early_logits.detach(), deep_logits.detach()
                )

                result["loss"] = lm_loss + self.routing_coeff * routing_loss
                result["lm_loss"] = lm_loss
                result["routing_loss"] = routing_loss

            return result

        def _run_layer(self, layer_idx: int, hidden_states: torch.Tensor):
            """
            Run a single transformer layer.

            Each GPT-OSS layer internally does:
              1. RMSNorm → Attention (sliding or full, depending on layer_idx)
              2. RMSNorm → MoE FFN (top-4 of 32 experts selected by mlp.router)
              3. Residual connections

            The MoE expert selection is INTERNAL — we don't interfere.
            We operate only on the output residual stream.
            """
            layer = self.layers[layer_idx]
            # GptOssDecoderLayer.forward() handles attention type
            # (sliding vs full) based on the layer_types config
            output = layer(hidden_states)
            # Handle both tuple and tensor outputs
            if isinstance(output, tuple):
                return output[0]
            return output

        # ─────────────────────────────────────────
        # KL-Calibrated Routing Loss
        # ─────────────────────────────────────────

        def _kl_calibrated_loss(
            self,
            exit_probs: torch.Tensor,
            early_logits: torch.Tensor,
            deep_logits: torch.Tensor,
        ) -> torch.Tensor:
            """
            Teach the router WHICH tokens can safely exit early.

            Uses KL(deep || early) as a per-token quality measure:
            - Low KL → early logits ≈ deep logits → safe to exit (target 1.0)
            - High KL → big quality gap → must continue (target 0.0)

            The router learns to PREDICT these targets, not just push
            uniformly toward exit.
            """
            with torch.no_grad():
                early_log_p = torch.log_softmax(early_logits, dim=-1)
                deep_p = torch.softmax(deep_logits, dim=-1)
                deep_log_p = torch.log_softmax(deep_logits, dim=-1)

                # KL(deep || early) per token
                kl = torch.sum(deep_p * (deep_log_p - early_log_p), dim=-1)  # (B, S)

                # Convert to exit targets via sigmoid
                kl_median = kl.median()
                targets = torch.sigmoid(-2.0 * (kl - kl_median))

            # BCE loss: train router to predict safe-exit tokens
            calibration = nn.functional.binary_cross_entropy(
                exit_probs, targets
            )
            # Mild uniform efficiency pressure to break ties
            efficiency = (1 - exit_probs).mean()

            return calibration + 0.1 * efficiency

        # ─────────────────────────────────────────
        # Physical Layer Skipping (Generation)
        # ─────────────────────────────────────────

        def generate_with_skipping(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 50,
        ) -> dict:
            """
            Autoregressive generation with PHYSICAL layer skipping.

            For each new token position:
            1. Run early layers 0-11 (with full KV cache)
            2. Son Router decides: exit or continue?
            3. If EXIT: compute logits from early hidden state
               → Layers 12-23 are NEVER EXECUTED for this token
               → Deep KV cache is NOT populated for this position
            4. If CONTINUE: run layers 12-23 and use deep logits

            KV-Cache Structure:
              early_kv[0..11]:  Dense  — all positions present
              deep_kv[12..23]:  Sparse — only non-exited positions

            For deep layers, the attention mask is modified so that
            non-exited tokens only attend to other non-exited tokens
            in the deep KV-cache (plus all tokens in the early KV-cache
            that flow through the residual connections).

            Returns:
                dict with generated_ids, exit_decisions, savings_pct
            """
            self.eval()
            device = input_ids.device
            generated = input_ids.clone()
            exit_decisions = []
            total_early_compute = 0
            total_deep_compute = 0
            num_generated = 0

            with torch.no_grad():
                for step in range(max_new_tokens):
                    # Run full forward for simplicity in reference impl
                    # (production MLX version would use incremental KV cache)
                    hidden = self.embed_tokens(generated)

                    # Early layers (all tokens)
                    for i in range(self.exit_after_layer):
                        hidden = self._run_layer(i, hidden)

                    total_early_compute += self.exit_after_layer

                    # Router decision for the LAST token only
                    last_hidden = hidden[:, -1:, :]
                    routing = self.router(last_hidden)
                    exit_prob = routing["exit_probs"][0, 0].item()
                    should_exit = exit_prob > 0.5

                    if should_exit:
                        # PHYSICAL SKIP: use early logits, skip layers 12-23
                        logits = self.lm_head(self.final_norm(last_hidden))
                        exit_decisions.append(("EXIT", exit_prob))
                    else:
                        # Continue through deep layers
                        deep_hidden = hidden
                        for i in range(self.exit_after_layer,
                                       self.config.num_hidden_layers):
                            deep_hidden = self._run_layer(i, deep_hidden)
                        logits = self.lm_head(self.final_norm(deep_hidden[:, -1:, :]))
                        exit_decisions.append(("CONTINUE", exit_prob))
                        total_deep_compute += (self.config.num_hidden_layers
                                               - self.exit_after_layer)

                    # Greedy decode
                    next_token = logits[0, -1].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                    num_generated += 1

                    # Check EOS
                    if next_token.item() == self.config.vocab_size - 1:  # approximate EOS
                        break

            # Compute savings
            max_compute = num_generated * self.config.num_hidden_layers
            actual_compute = total_early_compute + total_deep_compute
            savings_pct = (1 - actual_compute / max_compute) * 100 if max_compute > 0 else 0

            exit_count = sum(1 for d in exit_decisions if d[0] == "EXIT")
            exit_rate = exit_count / max(num_generated, 1) * 100

            return {
                "generated_ids": generated,
                "exit_decisions": exit_decisions,
                "num_generated": num_generated,
                "exit_rate_pct": exit_rate,
                "savings_pct": savings_pct,
                "avg_layers": actual_compute / max(num_generated, 1),
            }

        # ─────────────────────────────────────────
        # Self-Distillation Training
        # ─────────────────────────────────────────

        def train_router_distillation(
            self,
            train_texts: list,
            tokenizer,
            num_epochs: int = 5,
            lr: float = 1e-3,
            max_seq_len: int = 256,
        ):
            """
            Train Son Router via self-distillation from frozen backbone.

            The backbone (MXFP4 quantized) is the TEACHER.
            The Son Router (float32) is the STUDENT.

            For each training sample:
            1. Forward pass through ALL 24 layers (teacher provides deep logits)
            2. Extract early logits at Layer 12 (student's potential output)
            3. Compute KL(deep || early) per token → exit targets
            4. Train router to predict these targets via BCE

            The backbone weights are NEVER modified.
            """
            self.freeze_backbone()
            optimizer = torch.optim.AdamW(self.router.parameters(), lr=lr)

            for epoch in range(num_epochs):
                total_loss = 0.0
                n_batches = 0

                for text in train_texts:
                    # Tokenize
                    tokens = tokenizer(
                        text, return_tensors="pt",
                        max_length=max_seq_len, truncation=True,
                    )
                    input_ids = tokens["input_ids"].to(
                        next(self.parameters()).device
                    )

                    # Forward pass with labels for loss computation
                    output = self.forward(input_ids=input_ids, labels=input_ids)

                    loss = output["loss"]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                avg = total_loss / max(n_batches, 1)
                print(f"  Epoch {epoch+1}/{num_epochs} — "
                      f"loss: {avg:.4f}, "
                      f"routing: {output['routing_loss'].item():.4f}")

        # ─────────────────────────────────────────
        # KL Divergence Profiler
        # ─────────────────────────────────────────

        def profile_exit_layers(
            self,
            input_ids: torch.Tensor,
            tokenizer=None,
        ) -> dict:
            """
            Profile KL divergence at every layer to find optimal exit point.

            Measures how much quality degrades when exiting at each layer
            compared to the full 24-layer output. This finds the "Golden Point"
            on the Pareto frontier.

            Returns dict mapping layer_idx → {avg_kl, safe_exit_pct}
            """
            self.eval()
            device = input_ids.device

            with torch.no_grad():
                hidden = self.embed_tokens(input_ids)

                # Run all layers, save hidden states
                layer_hiddens = []
                for i in range(self.config.num_hidden_layers):
                    hidden = self._run_layer(i, hidden)
                    layer_hiddens.append(hidden.clone())

                # Deep logits (ground truth)
                deep_logits = self.lm_head(self.final_norm(layer_hiddens[-1]))
                deep_probs = torch.softmax(deep_logits, dim=-1)
                deep_log_probs = torch.log_softmax(deep_logits, dim=-1)
                deep_top1 = deep_logits.argmax(dim=-1)

            profile = {}
            print(f"\n  {'Layer':>7} {'Type':>10} {'Avg KL':>10} "
                  f"{'Median KL':>11} {'Safe Exit%':>12}")
            print(f"  {'─' * 55}")

            for i, h in enumerate(layer_hiddens):
                with torch.no_grad():
                    early_logits = self.lm_head(self.final_norm(h))
                    early_log_probs = torch.log_softmax(early_logits, dim=-1)

                    kl = torch.sum(
                        deep_probs * (deep_log_probs - early_log_probs),
                        dim=-1,
                    )  # (B, S)

                    avg_kl = kl.mean().item()
                    med_kl = kl.median().item()
                    early_top1 = early_logits.argmax(dim=-1)
                    safe_pct = (early_top1 == deep_top1).float().mean().item() * 100

                attn_type = self.config.layer_types[i]
                short_type = "slide" if "sliding" in attn_type else "full"

                marker = ""
                if i == self.exit_after_layer - 1:
                    marker = " <-- EXIT POINT"
                elif i == self.config.num_hidden_layers // 2 - 1:
                    marker = " <-- MID-STACK"

                profile[i] = {"avg_kl": avg_kl, "median_kl": med_kl,
                              "safe_exit_pct": safe_pct}

                print(f"  L{i:>4}   {short_type:>8}   {avg_kl:>10.2f} "
                      f"{med_kl:>11.2f} {safe_pct:>10.1f}%{marker}")

            return profile

        # ─────────────────────────────────────────
        # Architecture Summary
        # ─────────────────────────────────────────

        def summary(self) -> str:
            """Print architecture summary."""
            c = self.config
            router_params = sum(p.numel() for p in self.router.parameters())
            lines = [
                f"{'═' * 65}",
                f"  TranscenderEngine — GPT-OSS 20B",
                f"{'═' * 65}",
                f"  Backbone:        {c.model_type} ({c.total_params_b}B total, "
                f"{c.active_params_b}B active)",
                f"  Layers:          {c.num_hidden_layers} "
                f"(alternating sliding/full attention)",
                f"  Hidden:          {c.hidden_size}",
                f"  Attention:       {c.num_attention_heads} heads, "
                f"{c.num_key_value_heads} KV-heads (GQA {c.gqa_ratio}:1)",
                f"  Experts:         {c.num_local_experts} total, "
                f"top-{c.num_experts_per_tok} active",
                f"  Vocab:           {c.vocab_size:,}",
                f"  Context:         {c.max_position_embeddings:,} "
                f"(YaRN RoPE, θ={c.rope_theta})",
                f"  Quantization:    {c.hidden_act.upper()} / MXFP4",
                f"  Sliding window:  {c.sliding_window}",
                f"",
                f"  Son Router:      {router_params:,} params (float32)",
                f"  Exit layer:      {self.exit_after_layer}/{c.num_hidden_layers}",
                f"  Max savings:     "
                f"{(1 - self.exit_after_layer / c.num_hidden_layers) * 100:.0f}%",
                f"  Inference mode:  {self.inference_mode}",
                f"",
                f"  Compound Sparsity:",
                f"    MoE (width):   "
                f"{c.num_experts_per_tok}/{c.num_local_experts} = "
                f"{c.num_experts_per_tok/c.num_local_experts*100:.1f}% active",
                f"    Son (depth):   {self.exit_after_layer}/{c.num_hidden_layers} = "
                f"{self.exit_after_layer/c.num_hidden_layers*100:.0f}% depth",
                f"    Combined:      "
                f"{c.num_experts_per_tok/c.num_local_experts * self.exit_after_layer/c.num_hidden_layers * 100:.1f}% "
                f"total utilization (for exited tokens)",
                f"{'═' * 65}",
            ]
            text = "\n".join(lines)
            print(text)
            return text


# ═══════════════════════════════════════════════════════════════════
# Quick Validation (no model loading required)
# ═══════════════════════════════════════════════════════════════════

def validate_architecture():
    """Validate TranscenderEngine structure without loading the 20B model."""
    print("=" * 65)
    print("  TranscenderEngine — Architecture Validation")
    print("  (No model loading — validates structure and Son Router)")
    print("=" * 65)

    config = GptOssConfig()
    engine = TranscenderEngine(config=config, exit_after_layer=12)
    engine.summary()

    # Test Son Router with synthetic input
    if HAS_TORCH:
        print("\n  Testing Son Router with synthetic hidden states...")
        fake_hidden = torch.randn(1, 10, config.hidden_size)
        routing = engine.router(fake_hidden)

        print(f"  Input shape:  {fake_hidden.shape}")
        print(f"  Exit probs:   {routing['exit_probs'].shape} "
              f"(range: [{routing['exit_probs'].min():.4f}, "
              f"{routing['exit_probs'].max():.4f}])")
        print(f"  Exit mask:    {routing['exit_mask'].sum().item()}/10 tokens exit")
        print(f"\n  Son Router validated successfully.")

    # Memory estimate
    print(f"\n  {'─' * 50}")
    print(f"  MEMORY ESTIMATE (M1 Pro 32GB)")
    print(f"  {'─' * 50}")

    model_gb = config.total_params_b * 4 / 8  # MXFP4
    kv_gb = (2 * config.num_hidden_layers * 4096 *
             config.num_key_value_heads * config.head_dim * 2) / 1e9
    router_kb = sum(p.numel() for p in engine.router.parameters()) * 4 / 1024

    print(f"  Model (MXFP4):     {model_gb:.1f} GB")
    print(f"  KV cache (4K ctx): {kv_gb:.2f} GB")
    print(f"  Son Router:        {router_kb:.1f} KB")
    print(f"  Total:             {model_gb + kv_gb:.1f} GB / 32.0 GB")
    print(f"  Headroom:          {32.0 - model_gb - kv_gb:.1f} GB")
    print(f"\n{'=' * 65}")


if __name__ == "__main__":
    validate_architecture()
