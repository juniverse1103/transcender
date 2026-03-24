"""
GPU validation of the Track A penultimate-layer frontier.

This script now treats the manual forward-pass path as the trustworthy
reference path. The earlier HF implementation was risky for two reasons:

1. It relied on `output_hidden_states` capture semantics from Transformers,
   which are model-implementation dependent and easy to misread.
2. It reported `top1_agree` exact match against full depth even though the
   composed token falls back to the full-depth token on disagreement. That is
   useful as a composition diagnostic, but it is not a valid frontier metric.

The corrected path below:

- runs prefill once
- decodes step-by-step under full-depth shared context
- explicitly runs `embed_tokens -> layers -> final norm -> lm_head`
- extracts candidate logits for requested intermediate layers
- records raw exit tokens, composed tokens, and agreement statistics

This is a validation/debugging tool for GPU external-validity work. It is not
yet a serving benchmark and it is not a claim of exact MLX parity.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

# ---------------------------------------------------------------------------
# Prompts — same suite as MLX Track A. P1 is warmup (excluded from scoring).
# ---------------------------------------------------------------------------
PROMPTS = [
    # --- Original expository (P1-P5) ---
    "Explain quantum entanglement in simple terms.",                          # P1 (warmup)
    "Summarize why the French Revolution was historically important.",        # P2
    "Write a short explanation of recursion for a beginner programmer.",      # P3
    "Explain the difference between TCP and UDP in plain English.",           # P4
    "Describe what photosynthesis does.",                                     # P5
    # --- Code/technical (P6-P7) ---
    "Write a Python function that checks if a string is a palindrome.",      # P6
    "Explain what a hash table is and when you would use one.",              # P7
    # --- Reasoning/logic (P8-P9) ---
    "A farmer has 17 sheep. All but 9 run away. How many are left and why?", # P8
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",  # P9
    # --- Creative/open-ended (P10-P11) ---
    "Write the opening paragraph of a mystery story set in a library.",      # P10
    "Describe a sunset to someone who has never seen one.",                   # P11
    # --- Short-answer/factual (P12-P14) ---
    "What is the capital of Australia?",                                      # P12
    "What does the HTTP status code 404 mean?",                              # P13
    "Name three noble gases.",                                                # P14
    # --- List/structured output (P15-P16) ---
    "List five common sorting algorithms and one sentence about each.",       # P15
    "List the planets of the solar system in order from the Sun.",            # P16
    # --- Expanded expository (P17-P24) ---
    "Explain how a transformer neural network processes a sentence.",         # P17
    "Describe the greenhouse effect and why it matters for climate.",         # P18
    "Explain what an API is to someone with no programming background.",      # P19
    "Describe how vaccines train the immune system.",                         # P20
    "Explain the difference between machine learning and traditional programming.", # P21
    "Describe how a blockchain works at a high level.",                       # P22
    "Explain what inflation is and how it affects purchasing power.",         # P23
    "Describe the water cycle in simple terms.",                              # P24
    # --- Code/technical (P25-P32) ---
    "Write a Python function that reverses a linked list.",                   # P25
    "Explain the difference between a stack and a queue.",                    # P26
    "Describe what a REST API is and give an example endpoint.",              # P27
    "Explain what Big O notation measures and give two examples.",            # P28
    "Write a Python function that counts the vowels in a string.",           # P29
    "Explain the difference between SQL and NoSQL databases.",               # P30
    "Write a Python function that checks if two strings are anagrams.",      # P31
    "Explain what Docker containers are and why developers use them.",        # P32
    # --- Reasoning/logic (P33-P40) ---
    "You have two ropes that each take exactly one hour to burn, but burn unevenly. How can you measure 45 minutes?", # P33
    "Three boxes are labeled Apples, Oranges, and Mixed, but all labels are wrong. You pick one fruit from one box. How do you label all three correctly?", # P34
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", # P35
    "If you flip a fair coin three times, what is the probability of getting exactly two heads?", # P36
    "You are in a room with two doors and two guards. One always lies and one always tells the truth. What single question can you ask to find the safe door?", # P37
    "A snail climbs 3 feet up a wall each day but slides back 2 feet at night. If the wall is 30 feet high, how many days to reach the top?", # P38
    "There are 100 lockers in a hallway, all closed. 100 students walk by; student k toggles every k-th locker. Which lockers are open at the end?", # P39
    "You have 8 identical-looking balls but one is heavier. Using a balance scale, what is the minimum number of weighings to find the heavy ball?", # P40
    # --- Creative/open-ended (P41-P48) ---
    "Write a haiku about a thunderstorm.",                                    # P41
    "Describe an alien planet where the oceans are made of mercury.",         # P42
    "Write a short fable about a tortoise and an eagle.",                     # P43
    "Describe a busy city market to someone who has never visited one.",      # P44
    "Write a limerick about a programmer who debugs in their sleep.",         # P45
    "Describe what it would feel like to walk on the Moon.",                  # P46
    "Write the first paragraph of a detective story set in Tokyo.",           # P47
    "Describe the sound of a rainforest at dawn.",                            # P48
    # --- Short-answer/factual (P49-P56) ---
    "What is the chemical symbol for gold?",                                  # P49
    "What year did the Berlin Wall fall?",                                    # P50
    "What is the speed of light in a vacuum, approximately?",                # P51
    "What programming language was originally developed by Guido van Rossum?", # P52
    "What is the largest organ in the human body?",                           # P53
    "What does DNA stand for?",                                               # P54
    "What is the boiling point of water at sea level in Celsius?",           # P55
    "Name the four fundamental forces in physics.",                           # P56
    # --- List/structured output (P57-P64) ---
    "List five programming paradigms and one sentence about each.",           # P57
    "List the layers of the OSI model from bottom to top.",                   # P58
    "List three differences between Python and JavaScript.",                  # P59
    "List the first ten elements of the periodic table in order.",            # P60
    "Outline the steps to make a peanut butter and jelly sandwich.",          # P61
    "List four types of machine learning and a one-sentence description of each.", # P62
    "List the phases of the Moon in order.",                                  # P63
    "List five renewable energy sources and one advantage of each.",          # P64
]

SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_BLEND_ALPHA = 0.10
WARMUP_INDEX = 0  # P1
SUPPORTED_ORACLE_MODES = [
    "top1_agree",
    "top1_agree_margin",
    "top1_agree_entropy",
    "two_layer_top1_agree",
    "earliest_correct",
]
DEFAULT_ORACLE_MODES = ["top1_agree"]

MODEL_REFERENCE_BY_SUBSTRING = {
    "qwen3-30b-a3b": {
        "reference_type": "mlx_track_a_supplementary_n63",
        "layers": {
            "L46": {"exact_match": 0.837, "perfect": 36, "total": 63},
            "L45": {"exact_match": 0.463, "perfect": 6, "total": 63},
        },
    },
    "gpt-oss-20b": {
        "reference_type": "mlx_track_a_supplementary_n63",
        "layers": {
            "L22": {"exact_match": 0.870, "perfect": 47, "total": 63},
            "L21": {"exact_match": 0.703, "perfect": 27, "total": 63},
        },
    },
}


@dataclass
class ModelParts:
    backbone: Any
    embed_tokens: torch.nn.Module
    layers: Any
    final_norm: torch.nn.Module
    rotary_emb: torch.nn.Module
    lm_head: torch.nn.Module
    config: Any
    device: torch.device
    num_layers: int
    family: str
    architecture: str
    input_hidden_state_scaling: str
    attention_mask_style: str


@dataclass
class OracleConfig:
    modes: List[str]
    margin_threshold: float
    entropy_threshold: float


class NonFiniteTensorError(RuntimeError):
    def __init__(self, details: Dict[str, Any]):
        self.details = details
        stage = details.get("stage", "unknown")
        super().__init__(f"Non-finite tensor detected at {stage}")


def resolve_manual_reference_load_dtype(model_name: str) -> torch.dtype:
    """
    Preserve existing fp16 behavior by default, but honor Gemma 3's configured
    checkpoint dtype. The local Gemma 3 text checkpoints declare bfloat16 and
    forcing fp16 can numerically corrupt the manual-reference path.
    """
    from transformers import AutoConfig

    default_dtype = torch.float16
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return default_dtype

    if getattr(config, "model_type", None) != "gemma3":
        return default_dtype

    configured_dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(configured_dtype, torch.dtype):
        return configured_dtype
    return torch.bfloat16


def load_model_and_tokenizer(model_name: str, quantize: str, device: str):
    """Load the causal LM without relying on hidden-state capture side effects."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_dtype = resolve_manual_reference_load_dtype(model_name)
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": load_dtype,
    }

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=load_dtype,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except ValueError as exc:
        lowered = model_name.lower()
        if "gemma-3" in lowered or "gemma3" in lowered:
            raise RuntimeError(
                "Gemma 3 support in this script is limited to text causal-LM "
                "checkpoints. Multimodal Gemma 3 conditional-generation "
                "checkpoints are not supported by the manual-reference path."
            ) from exc
        raise

    model.eval()
    return model, tokenizer


def build_messages(prompt_text: str, system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]


def render_prompt(tokenizer, prompt_text: str, system_prompt: str) -> torch.Tensor:
    """Render a chat prompt using the tokenizer-native chat template when available."""
    messages = build_messages(prompt_text=prompt_text, system_prompt=system_prompt)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="medium",
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return tokenizer(rendered, return_tensors="pt")["input_ids"]
    return tokenizer(
        f"{system_prompt}\n\n{prompt_text}",
        return_tensors="pt",
    )["input_ids"]


def select_prompts(prompt_offset: int = 0, prompt_limit: Optional[int] = None):
    if prompt_offset < 0:
        raise ValueError("--prompt-offset must be >= 0")
    if prompt_limit is not None and prompt_limit <= 0:
        raise ValueError("--prompt-limit must be > 0 when provided")

    if prompt_limit is None:
        selected = PROMPTS[prompt_offset:]
    else:
        selected = PROMPTS[prompt_offset: prompt_offset + prompt_limit]

    if not selected:
        raise ValueError(
            f"No prompts selected with offset={prompt_offset}, limit={prompt_limit}"
        )
    return list(enumerate(selected, start=prompt_offset))


def resolve_prompt_index(prompt_id: str) -> int:
    if not prompt_id.startswith("P"):
        raise ValueError(f"Prompt id must look like P2, got {prompt_id!r}")
    index = int(prompt_id[1:]) - 1
    if index < 0 or index >= len(PROMPTS):
        raise ValueError(f"Prompt id out of range: {prompt_id}")
    return index


def token_to_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def module_device(module: torch.nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    raise ValueError(f"Cannot infer device for module {type(module).__name__}")


def round_or_none(value: Optional[float], places: int = 6) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, places)


def normalize_oracle_modes(modes: Optional[List[str]]) -> List[str]:
    resolved = DEFAULT_ORACLE_MODES if not modes else modes
    unique_modes: List[str] = []
    for mode in resolved:
        if mode not in SUPPORTED_ORACLE_MODES:
            raise ValueError(
                f"Unsupported oracle mode {mode!r}. "
                f"Expected one of {SUPPORTED_ORACLE_MODES}."
            )
        if mode not in unique_modes:
            unique_modes.append(mode)
    return unique_modes


def oracle_config_dict(oracle_config: OracleConfig, exit_layers: List[int]) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "modes": list(oracle_config.modes),
        "margin_threshold": round_or_none(oracle_config.margin_threshold),
        "entropy_threshold": round_or_none(oracle_config.entropy_threshold),
        "notes": (
            "Oracle summaries are final-aware verifier diagnostics over the "
            "shared-context manual-reference path. They are not deployable "
            "selective-depth policies or speed claims."
        ),
    }
    if "two_layer_top1_agree" in oracle_config.modes:
        config["two_layer_pair"] = (
            [f"L{exit_layers[0]}", f"L{exit_layers[1]}"]
            if len(exit_layers) >= 2
            else None
        )
    return config


def summarize_tensor_numerics(tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
    if tensor is None:
        return {
            "is_none": True,
            "shape": None,
            "dtype": None,
            "all_finite": True,
            "has_nan": False,
            "has_pos_inf": False,
            "has_neg_inf": False,
            "finite_count": 0,
            "nan_count": 0,
            "pos_inf_count": 0,
            "neg_inf_count": 0,
            "min_finite": None,
            "max_finite": None,
            "max_abs_finite": None,
        }

    tensor = tensor.detach()
    flat = tensor.float()
    finite_mask = torch.isfinite(flat)
    nan_mask = torch.isnan(flat)
    pos_inf_mask = torch.isposinf(flat)
    neg_inf_mask = torch.isneginf(flat)
    finite_count = int(finite_mask.sum().item())

    summary: Dict[str, Any] = {
        "is_none": False,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "all_finite": bool(finite_mask.all().item()),
        "has_nan": bool(nan_mask.any().item()),
        "has_pos_inf": bool(pos_inf_mask.any().item()),
        "has_neg_inf": bool(neg_inf_mask.any().item()),
        "finite_count": finite_count,
        "nan_count": int(nan_mask.sum().item()),
        "pos_inf_count": int(pos_inf_mask.sum().item()),
        "neg_inf_count": int(neg_inf_mask.sum().item()),
    }
    if finite_count > 0:
        finite_values = flat[finite_mask]
        summary["min_finite"] = round_or_none(float(finite_values.min().item()))
        summary["max_finite"] = round_or_none(float(finite_values.max().item()))
        summary["max_abs_finite"] = round_or_none(float(finite_values.abs().max().item()))
    else:
        summary["min_finite"] = None
        summary["max_finite"] = None
        summary["max_abs_finite"] = None
    return summary


def ensure_finite_tensor(
    stage: str,
    tensor: torch.Tensor,
    stage_summaries: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = summarize_tensor_numerics(tensor)
    if stage_summaries is not None:
        stage_summaries[stage] = summary
    if not summary["all_finite"]:
        details: Dict[str, Any] = {
            "stage": stage,
            "tensor_summary": summary,
        }
        if stage_summaries is not None:
            details["stage_summaries"] = dict(stage_summaries)
        if extra:
            details.update(extra)
        raise NonFiniteTensorError(details)
    return summary


def ensure_valid_attention_mask(
    stage: str,
    attention_mask: Optional[torch.Tensor],
    stage_summaries: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = summarize_tensor_numerics(attention_mask)
    summary["allows_negative_infinity"] = True
    summary["all_valid_mask_values"] = not summary["has_nan"] and not summary["has_pos_inf"]
    if stage_summaries is not None:
        stage_summaries[stage] = summary
    if not summary["all_valid_mask_values"]:
        details: Dict[str, Any] = {
            "stage": stage,
            "tensor_summary": summary,
        }
        if stage_summaries is not None:
            details["stage_summaries"] = dict(stage_summaries)
        if extra:
            details.update(extra)
        raise NonFiniteTensorError(details)
    return summary


def infer_model_family(backbone: torch.nn.Module) -> str:
    module_name = type(backbone).__module__.lower()
    class_name = type(backbone).__name__.lower()

    families = {
        "gemma3": ("gemma3text", "gemma3"),
        "qwen2_moe": ("qwen2_moe", "qwen2moe"),
        "mixtral": ("mixtral",),
        "llama": ("llama",),
        "mistral": ("mistral",),
        "gemma2": ("gemma2",),
        "gemma": ("gemma",),
        "gpt_oss": ("gpt_oss", "gptoss"),
    }

    for family, markers in families.items():
        if any(marker in module_name or marker in class_name for marker in markers):
            return family

    raise RuntimeError(
        "Unsupported backbone for the manual-reference path: "
        f"{type(backbone).__name__} from {type(backbone).__module__}. "
        "Supported families are qwen2_moe/qwen3_moe, gpt_oss, mixtral, "
        "llama, mistral, gemma, gemma2, and gemma3 text."
    )


def infer_input_hidden_state_scaling(family: str) -> str:
    if family in {"gemma", "gemma2"}:
        return "sqrt_hidden_size"
    return "none"


def infer_attention_mask_style(family: str) -> str:
    if family in {"gpt_oss", "gemma2", "gemma3"}:
        return "per_layer_attention_type"
    return "single_mask"


def unwrap_manual_reference_backbone(model) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Resolve the text decoder stack and lm_head used by the manual-reference path.

    Gemma 3 text checkpoints can be exposed through a conditional-generation
    wrapper whose `.model` contains a `.language_model` text backbone. That is
    still a valid text-only decode path for this script as long as the language
    model exposes the expected decoder-stack attributes.
    """
    backbone = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)

    if backbone is None and hasattr(model, "language_model"):
        backbone = model.language_model

    if backbone is None:
        raise AttributeError(
            f"{type(model).__name__} has no `.model` backbone. "
            "Manual adaptation is required for this architecture."
        )
    if lm_head is None:
        raise AttributeError(f"{type(model).__name__} has no `lm_head`.")

    if hasattr(backbone, "language_model") and not hasattr(backbone, "embed_tokens"):
        candidate = backbone.language_model
        required = ["embed_tokens", "layers", "norm", "rotary_emb"]
        missing = [name for name in required if not hasattr(candidate, name)]
        if missing:
            raise RuntimeError(
                "Loaded Gemma 3 wrapper did not expose a usable text "
                f"language model for manual decode; missing attrs: {missing}"
            )
        return candidate, lm_head

    return backbone, lm_head


def get_model_parts(model) -> ModelParts:
    backbone, lm_head = unwrap_manual_reference_backbone(model)

    required = ["embed_tokens", "layers", "norm", "rotary_emb"]
    missing = [name for name in required if not hasattr(backbone, name)]
    if missing:
        raise AttributeError(
            f"Backbone {type(backbone).__name__} is missing required attrs: {missing}"
        )

    # The trustworthy reference path assumes one GPU with the whole model on it.
    # If the model is sharded, this script should fail loudly rather than silently
    # measuring a partly-wrong path.
    devices = {
        module_device(backbone.embed_tokens),
        module_device(backbone.norm),
        module_device(lm_head),
        module_device(backbone.layers[0]),
        module_device(backbone.layers[-1]),
    }
    if len(devices) != 1:
        raise RuntimeError(
            "Manual debug path expects the model to reside on one device. "
            f"Observed devices: {sorted(str(d) for d in devices)}"
        )

    family = infer_model_family(backbone)

    return ModelParts(
        backbone=backbone,
        embed_tokens=backbone.embed_tokens,
        layers=backbone.layers,
        final_norm=backbone.norm,
        rotary_emb=backbone.rotary_emb,
        lm_head=lm_head,
        config=backbone.config,
        device=devices.pop(),
        num_layers=len(backbone.layers),
        family=family,
        architecture=type(backbone).__name__,
        input_hidden_state_scaling=infer_input_hidden_state_scaling(family),
        attention_mask_style=infer_attention_mask_style(family),
    )


def resolve_reference_for_model(model_name: str) -> Optional[Dict[str, Any]]:
    lowered = model_name.lower()
    for substring, reference in MODEL_REFERENCE_BY_SUBSTRING.items():
        if substring in lowered:
            return reference
    return None


def default_exit_layers(num_layers: int) -> List[int]:
    if num_layers < 3:
        raise ValueError(
            "Manual reference benchmark needs at least 3 layers to compare "
            "penultimate-1, penultimate, and full depth."
        )
    return [num_layers - 3, num_layers - 2]


def normalize_exit_layers(
    exit_layers: Optional[List[int]],
    num_layers: int,
) -> List[int]:
    resolved = default_exit_layers(num_layers) if not exit_layers else sorted(set(exit_layers))
    validate_exit_layers(resolved, num_layers)
    return resolved


def validate_exit_layers(exit_layers: List[int], num_layers: int) -> None:
    bad = [layer for layer in exit_layers if layer < 0 or layer >= num_layers]
    if bad:
        raise ValueError(
            f"Exit layers {bad} out of range for model with {num_layers} layers"
        )


def project_hidden_to_logits(parts: ModelParts, hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Project a decoder-layer hidden state through the same final norm + lm_head
    used by the model head. The earlier script skipped this norm step, which is
    not faithful for Qwen-family decoders.
    """
    normed = apply_final_norm(parts, hidden_state)
    return apply_lm_head(parts, normed)


def apply_final_norm(parts: ModelParts, hidden_state: torch.Tensor) -> torch.Tensor:
    return parts.final_norm(hidden_state.to(parts.device))


def apply_lm_head(parts: ModelParts, hidden_state: torch.Tensor) -> torch.Tensor:
    logits = parts.lm_head(hidden_state)
    softcap = getattr(parts.config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = logits / softcap
        logits = torch.tanh(logits)
        logits = logits * softcap
    return logits[:, -1, :]


def _bidirectional_window_overlay(sliding_window: int):
    """Match the Gemma 3 bidirectional sliding-window mask behavior."""

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return abs(q_idx - kv_idx) < sliding_window

    return inner_mask


def build_position_embeddings(
    parts: ModelParts,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
):
    if parts.family == "gemma3":
        layer_types = getattr(parts.config, "layer_types", None)
        if not layer_types:
            raise RuntimeError(
                "Gemma 3 manual path requires `config.layer_types` to build "
                "layer-specific rotary embeddings."
            )
        position_embeddings = {}
        for layer_type in layer_types:
            if layer_type not in position_embeddings:
                position_embeddings[layer_type] = parts.rotary_emb(
                    hidden_states,
                    position_ids,
                    layer_type,
                )
        return position_embeddings
    try:
        return parts.rotary_emb(hidden_states, position_ids=position_ids)
    except TypeError:
        return parts.rotary_emb(hidden_states, position_ids)


def apply_input_hidden_state_scaling(
    parts: ModelParts,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if parts.input_hidden_state_scaling == "sqrt_hidden_size":
        normalizer = torch.tensor(
            parts.config.hidden_size**0.5,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return hidden_states * normalizer
    return hidden_states


def build_attention_mask_bundle(
    parts: ModelParts,
    inputs_embeds: torch.Tensor,
    cache_position: torch.Tensor,
    cache: DynamicCache,
    position_ids: torch.Tensor,
):
    mask_kwargs = {
        "config": parts.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": None,
        "cache_position": cache_position,
        "past_key_values": cache,
        "position_ids": position_ids,
    }
    if parts.attention_mask_style == "per_layer_attention_type":
        sliding_mask_kwargs = dict(mask_kwargs)
        if (
            parts.family == "gemma3"
            and getattr(parts.config, "use_bidirectional_attention", False)
        ):
            mask_kwargs["or_mask_function"] = (
                lambda *args: torch.tensor(True, dtype=torch.bool)
            )
            sliding_window = getattr(parts.config, "sliding_window", None)
            if sliding_window is None:
                raise RuntimeError(
                    "Gemma 3 bidirectional attention requires `config.sliding_window`."
                )
            sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
                sliding_window
            )
        return {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
        }
    sliding_window = getattr(parts.config, "sliding_window", None)
    mask_fn = (
        create_causal_mask
        if sliding_window is None
        else create_sliding_window_causal_mask
    )
    return mask_fn(**mask_kwargs)


def layer_attention_mask(attention_mask_bundle, decoder_layer):
    if isinstance(attention_mask_bundle, dict):
        attention_type = getattr(decoder_layer, "attention_type", None)
        if attention_type is None:
            raise RuntimeError(
                "Per-layer attention-mask mode requires decoder layers with "
                "`attention_type`."
            )
        if attention_type not in attention_mask_bundle:
            raise RuntimeError(
                f"Unsupported attention_type {attention_type!r}; expected one of "
                f"{sorted(attention_mask_bundle.keys())}"
            )
        return attention_mask_bundle[attention_type]
    return attention_mask_bundle


def layer_position_embeddings(parts: ModelParts, position_embeddings, decoder_layer):
    if parts.family == "gemma3":
        attention_type = getattr(decoder_layer, "attention_type", None)
        if attention_type is None:
            raise RuntimeError(
                "Gemma 3 manual path requires decoder layers with `attention_type`."
            )
        if attention_type not in position_embeddings:
            raise RuntimeError(
                f"Missing position embeddings for Gemma 3 attention_type "
                f"{attention_type!r}; expected one of {sorted(position_embeddings.keys())}"
            )
        return position_embeddings[attention_type]
    return position_embeddings


def extract_hidden_states(layer_output: Any) -> torch.Tensor:
    if torch.is_tensor(layer_output):
        return layer_output
    if isinstance(layer_output, (tuple, list)) and layer_output:
        first = layer_output[0]
        if torch.is_tensor(first):
            return first
    raise TypeError(
        "Decoder layer output is not a tensor or `(hidden_states, ...)` tuple: "
        f"{type(layer_output).__name__}"
    )


def compose_top1_agree_logits(
    full_logits: torch.Tensor,
    exit_logits: torch.Tensor,
    blend_alpha: float,
) -> tuple[torch.Tensor, bool]:
    full_logits = full_logits.float()
    exit_logits = exit_logits.float()
    full_top1 = int(full_logits.argmax(dim=-1).item())
    exit_top1 = int(exit_logits.argmax(dim=-1).item())
    agreed = full_top1 == exit_top1
    if agreed:
        return (1 - blend_alpha) * full_logits + blend_alpha * exit_logits, True
    return full_logits, False


def greedy_token_from_logits(logits: torch.Tensor, stage: str) -> int:
    ensure_finite_tensor(stage, logits)
    return int(logits.float().argmax(dim=-1).item())


def top1_top2_margin(logits: torch.Tensor) -> float:
    values = torch.topk(logits.float(), k=min(2, logits.shape[-1]), dim=-1).values[0]
    if values.numel() == 1:
        return float("inf")
    return float((values[0] - values[1]).item())


def logits_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return float(entropy.item())


def base_oracle_step_summary(
    *,
    accepted: bool,
    selected_token: int,
    full_token: int,
    decision_reason: str,
) -> Dict[str, Any]:
    return {
        "accepted": accepted,
        "fallback_used": not accepted,
        "selected_token_id": selected_token,
        "selected_matches_full_depth": selected_token == full_token,
        "selected_source": "exit" if accepted else "full_depth",
        "decision_reason": decision_reason,
    }


def evaluate_single_layer_oracle(
    *,
    mode: str,
    full_token: int,
    exit_token: int,
    exit_logits: torch.Tensor,
    oracle_config: OracleConfig,
) -> Dict[str, Any]:
    top1_matches = exit_token == full_token
    margin = top1_top2_margin(exit_logits)
    entropy = logits_entropy(exit_logits)

    if mode == "top1_agree":
        accepted = top1_matches
        decision_reason = "accepted" if accepted else "top1_mismatch"
    elif mode == "top1_agree_margin":
        accepted = top1_matches and margin > oracle_config.margin_threshold
        if not top1_matches:
            decision_reason = "top1_mismatch"
        elif not accepted:
            decision_reason = "margin_below_threshold"
        else:
            decision_reason = "accepted"
    elif mode == "top1_agree_entropy":
        accepted = top1_matches and entropy < oracle_config.entropy_threshold
        if not top1_matches:
            decision_reason = "top1_mismatch"
        elif not accepted:
            decision_reason = "entropy_above_threshold"
        else:
            decision_reason = "accepted"
    else:
        raise ValueError(f"Unsupported single-layer oracle mode: {mode}")

    selected_token = exit_token if accepted else full_token
    result = base_oracle_step_summary(
        accepted=accepted,
        selected_token=selected_token,
        full_token=full_token,
        decision_reason=decision_reason,
    )
    result["top1_matches_full_depth"] = top1_matches
    if mode == "top1_agree_margin":
        result["exit_top1_margin"] = round_or_none(margin)
        result["margin_threshold"] = round_or_none(oracle_config.margin_threshold)
    if mode == "top1_agree_entropy":
        result["exit_entropy"] = round_or_none(entropy)
        result["entropy_threshold"] = round_or_none(oracle_config.entropy_threshold)
    return result


def evaluate_two_layer_top1_agree_oracle(
    *,
    full_token: int,
    exit_tokens: Dict[int, int],
    exit_layers: List[int],
) -> Dict[str, Any]:
    if len(exit_layers) < 2:
        return {
            "available": False,
            "reason": "requires at least two exit layers",
        }

    first_layer, second_layer = exit_layers[:2]
    first_label = f"L{first_layer}"
    second_label = f"L{second_layer}"
    first_token = exit_tokens[first_layer]
    second_token = exit_tokens[second_layer]
    pair_agree = first_token == second_token
    accepted = pair_agree and first_token == full_token
    if accepted:
        decision_reason = "accepted"
    elif not pair_agree:
        decision_reason = "pair_top1_mismatch"
    else:
        decision_reason = "pair_matches_but_not_full"

    selected_token = first_token if accepted else full_token
    result = base_oracle_step_summary(
        accepted=accepted,
        selected_token=selected_token,
        full_token=full_token,
        decision_reason=decision_reason,
    )
    result.update(
        {
            "available": True,
            "pair": [first_label, second_label],
            "pair_top1_tokens": {
                first_label: first_token,
                second_label: second_token,
            },
            "pair_top1_agree": pair_agree,
            "pair_matches_full_depth_top1": accepted,
        }
    )
    return result


def evaluate_earliest_correct_oracle(
    *,
    full_token: int,
    exit_tokens: Dict[int, int],
    exit_layers: List[int],
) -> Dict[str, Any]:
    matching_layers = [layer_idx for layer_idx in exit_layers if exit_tokens[layer_idx] == full_token]
    selected_layer = matching_layers[0] if matching_layers else None
    accepted = selected_layer is not None
    selected_token = exit_tokens[selected_layer] if selected_layer is not None else full_token
    result = base_oracle_step_summary(
        accepted=accepted,
        selected_token=selected_token,
        full_token=full_token,
        decision_reason="accepted" if accepted else "fallback_full_depth",
    )
    result.update(
        {
            "matched_layers": [f"L{layer_idx}" for layer_idx in matching_layers],
            "selected_layer": f"L{selected_layer}" if selected_layer is not None else "full_depth",
        }
    )
    return result


def oracle_sequence_summary(
    full_ids: List[int],
    oracle_ids: List[int],
    accepted_steps: int,
    decision_reason_counts: Dict[str, int],
) -> Dict[str, Any]:
    metrics = sequence_metrics(full_ids, oracle_ids)
    total_steps = len(full_ids)
    fallback_steps = total_steps - accepted_steps
    return {
        "accepted_steps": accepted_steps,
        "total_steps": total_steps,
        "acceptance_rate": round(accepted_steps / total_steps, 6) if total_steps else 0.0,
        "fallback_steps": fallback_steps,
        "fallback_rate": round(fallback_steps / total_steps, 6) if total_steps else 0.0,
        "oracle_composed_exact_match_rate": round(metrics["exact_match_rate"], 6),
        "oracle_composed_prefix_match_tokens": metrics["prefix_match_tokens"],
        "oracle_composed_first_divergence_position": metrics["first_divergence_position"],
        "oracle_composed_perfect_match": metrics["perfect_match"],
        "decision_reason_counts": decision_reason_counts,
    }


def merge_reason_counts(count_maps: List[Dict[str, int]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for count_map in count_maps:
        for reason, count in count_map.items():
            merged[reason] = merged.get(reason, 0) + int(count)
    return merged


def build_step_diagnostics(
    full_hidden: torch.Tensor,
    exit_hidden: torch.Tensor,
    full_logits: torch.Tensor,
    exit_logits: torch.Tensor,
    full_top1: int,
    exit_top1: int,
    softcap: Optional[float],
) -> Dict[str, Any]:
    full_hidden_summary = summarize_tensor_numerics(full_hidden)
    exit_hidden_summary = summarize_tensor_numerics(exit_hidden)
    full_logits_summary = summarize_tensor_numerics(full_logits)
    exit_logits_summary = summarize_tensor_numerics(exit_logits)

    hidden_diff_max = None
    logits_diff_max = None
    full_max_abs_logit = full_logits_summary["max_abs_finite"]
    exit_max_abs_logit = exit_logits_summary["max_abs_finite"]

    if full_hidden_summary["all_finite"] and exit_hidden_summary["all_finite"]:
        hidden_diff = (exit_hidden.float() - full_hidden.float()).abs()
        hidden_diff_max = float(hidden_diff.max().item())
    if full_logits_summary["all_finite"] and exit_logits_summary["all_finite"]:
        logits_diff = (exit_logits.float() - full_logits.float()).abs()
        logits_diff_max = float(logits_diff.max().item())

    diagnostics: Dict[str, Any] = {
        "full_hidden_summary": full_hidden_summary,
        "raw_exit_hidden_summary": exit_hidden_summary,
        "full_logits_summary": full_logits_summary,
        "raw_exit_logits_summary": exit_logits_summary,
        "hidden_exact_equal_to_full": bool(torch.equal(exit_hidden, full_hidden)),
        "hidden_max_abs_diff_vs_full": round_or_none(hidden_diff_max, places=8),
        "logits_exact_equal_to_full": bool(torch.equal(exit_logits, full_logits)),
        "logits_max_abs_diff_vs_full": round_or_none(logits_diff_max, places=8),
        "same_argmax_but_logits_differ": (
            bool(full_top1 == exit_top1 and not torch.equal(exit_logits, full_logits))
            if full_logits_summary["all_finite"] and exit_logits_summary["all_finite"]
            else None
        ),
        "full_max_abs_logit": round_or_none(full_max_abs_logit),
        "raw_exit_max_abs_logit": round_or_none(exit_max_abs_logit),
    }

    if softcap is not None and full_max_abs_logit is not None and exit_max_abs_logit is not None:
        diagnostics["final_logit_softcap"] = float(softcap)
        diagnostics["full_max_abs_logit_over_softcap"] = round(
            full_max_abs_logit / softcap, 6
        )
        diagnostics["raw_exit_max_abs_logit_over_softcap"] = round(
            exit_max_abs_logit / softcap, 6
        )

    return diagnostics


def sequence_metrics(reference_ids: List[int], candidate_ids: List[int]) -> Dict[str, Any]:
    n = min(len(reference_ids), len(candidate_ids))
    matches = sum(1 for i in range(n) if reference_ids[i] == candidate_ids[i])
    exact = matches / n if n > 0 else 0.0

    prefix = 0
    for i in range(n):
        if reference_ids[i] == candidate_ids[i]:
            prefix += 1
        else:
            break
    first_div = prefix + 1 if prefix < n else n + 1

    return {
        "exact_match_rate": exact,
        "prefix_match_tokens": prefix,
        "first_divergence_position": first_div,
        "total_tokens": n,
        "perfect_match": prefix == n and len(reference_ids) == len(candidate_ids),
    }


def manual_forward_step(
    parts: ModelParts,
    input_ids: torch.Tensor,
    cache: DynamicCache,
    capture_layers: List[int],
    include_hidden_tensors: bool = False,
) -> Dict[str, Any]:
    """
    Explicit one-step forward pass through embeddings, decoder layers, final
    norm, and lm_head.

    This does not rely on `output_hidden_states` or `generate()` behavior.
    """
    stage_summaries: Dict[str, Any] = {}

    input_ids = input_ids.to(parts.device)
    inputs_embeds = parts.embed_tokens(input_ids)
    if include_hidden_tensors:
        ensure_finite_tensor("inputs_embeds", inputs_embeds, stage_summaries=stage_summaries)

    past_seen_tokens = cache.get_seq_length() if cache is not None else 0
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + inputs_embeds.shape[1],
        device=parts.device,
    )
    position_ids = cache_position.unsqueeze(0)

    attention_mask_bundle = build_attention_mask_bundle(
        parts=parts,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        cache=cache,
        position_ids=position_ids,
    )
    position_embeddings = build_position_embeddings(
        parts=parts,
        hidden_states=inputs_embeds,
        position_ids=position_ids,
    )

    hidden_states = apply_input_hidden_state_scaling(parts, inputs_embeds)
    if include_hidden_tensors:
        ensure_finite_tensor(
            "hidden_states_after_input_scaling",
            hidden_states,
            stage_summaries=stage_summaries,
        )
        if isinstance(attention_mask_bundle, dict):
            for mask_name, attention_mask in attention_mask_bundle.items():
                ensure_valid_attention_mask(
                    f"attention_mask_bundle.{mask_name}",
                    attention_mask,
                    stage_summaries=stage_summaries,
                )
        else:
            ensure_valid_attention_mask(
                "attention_mask_bundle",
                attention_mask_bundle,
                stage_summaries=stage_summaries,
            )

        if isinstance(position_embeddings, dict):
            for layer_type, rotary_pair in position_embeddings.items():
                cos, sin = rotary_pair
                ensure_finite_tensor(
                    f"position_embeddings.{layer_type}.cos",
                    cos,
                    stage_summaries=stage_summaries,
                )
                ensure_finite_tensor(
                    f"position_embeddings.{layer_type}.sin",
                    sin,
                    stage_summaries=stage_summaries,
                )
        else:
            cos, sin = position_embeddings
            ensure_finite_tensor(
                "position_embeddings.cos",
                cos,
                stage_summaries=stage_summaries,
            )
            ensure_finite_tensor(
                "position_embeddings.sin",
                sin,
                stage_summaries=stage_summaries,
            )

    captured_hidden: Dict[int, torch.Tensor] = {}

    for layer_idx, decoder_layer in enumerate(parts.layers[: parts.num_layers]):
        layer_label = f"L{layer_idx}"
        attention_type = getattr(decoder_layer, "attention_type", None)
        layer_output = decoder_layer(
            hidden_states,
            attention_mask=layer_attention_mask(attention_mask_bundle, decoder_layer),
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=layer_position_embeddings(
                parts,
                position_embeddings,
                decoder_layer,
            ),
        )
        hidden_states = extract_hidden_states(layer_output)
        if include_hidden_tensors:
            ensure_finite_tensor(
                f"decoder_layers.{layer_label}.output",
                hidden_states,
                stage_summaries=stage_summaries,
                extra={
                    "layer_idx": layer_idx,
                    "layer_label": layer_label,
                    "attention_type": attention_type,
                },
            )
        if layer_idx in capture_layers:
            # Clone the snapshot to avoid any chance of later in-place layer
            # operations collapsing intermediate captures onto the final state.
            captured_hidden[layer_idx] = hidden_states[:, -1:, :].detach().clone()

    full_hidden = hidden_states[:, -1:, :].detach().clone()
    full_normed = apply_final_norm(parts, full_hidden)
    full_logits = apply_lm_head(parts, full_normed)
    exit_logits = {}
    for layer_idx, hidden in captured_hidden.items():
        exit_normed = apply_final_norm(parts, hidden)
        exit_logits[layer_idx] = apply_lm_head(parts, exit_normed)
        if include_hidden_tensors:
            ensure_finite_tensor(
                f"exit_hidden.L{layer_idx}",
                hidden,
                stage_summaries=stage_summaries,
                extra={"layer_idx": layer_idx, "layer_label": f"L{layer_idx}"},
            )
            ensure_finite_tensor(
                f"exit_final_norm.L{layer_idx}",
                exit_normed,
                stage_summaries=stage_summaries,
                extra={"layer_idx": layer_idx, "layer_label": f"L{layer_idx}"},
            )
            ensure_finite_tensor(
                f"exit_logits.L{layer_idx}",
                exit_logits[layer_idx],
                stage_summaries=stage_summaries,
                extra={"layer_idx": layer_idx, "layer_label": f"L{layer_idx}"},
            )

    if include_hidden_tensors:
        ensure_finite_tensor("full_hidden", full_hidden, stage_summaries=stage_summaries)
        ensure_finite_tensor("final_norm_output", full_normed, stage_summaries=stage_summaries)
        ensure_finite_tensor("full_logits", full_logits, stage_summaries=stage_summaries)

    result = {
        "full_logits": full_logits.detach().cpu(),
        "exit_logits": {k: v.detach().cpu() for k, v in exit_logits.items()},
    }
    if include_hidden_tensors:
        result["full_hidden"] = full_hidden.detach().cpu()
        result["exit_hidden"] = {k: v.detach().cpu() for k, v in captured_hidden.items()}
        result["stage_summaries"] = stage_summaries
    return result


def run_shared_context_decode(
    parts: ModelParts,
    tokenizer,
    prompt_id: str,
    prompt_text: str,
    exit_layers: List[int],
    max_new_tokens: int,
    blend_alpha: float,
    include_trace: bool,
    oracle_config: OracleConfig,
) -> Dict[str, Any]:
    """
    Manual token-by-token decode under shared full-depth context.

    The full-depth token is used to advance the cache at every step. This makes
    the trace diagnostically trustworthy for comparing candidate logits from
    the requested exit layers against the full-depth path without branching
    the generation tree.
    """
    prompt_ids = render_prompt(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        system_prompt=SYSTEM_PROMPT,
    ).to(parts.device)

    cache = DynamicCache(config=parts.config)
    full_ids: List[int] = []
    exit_ids: Dict[int, List[int]] = {layer: [] for layer in exit_layers}
    composed_ids: Dict[int, List[int]] = {layer: [] for layer in exit_layers}
    agreement_counts: Dict[int, int] = {layer: 0 for layer in exit_layers}
    per_layer_oracle_modes = [
        mode
        for mode in oracle_config.modes
        if mode in {"top1_agree", "top1_agree_margin", "top1_agree_entropy"}
    ]
    cross_layer_oracle_modes = [
        mode
        for mode in oracle_config.modes
        if mode in {"two_layer_top1_agree", "earliest_correct"}
    ]
    oracle_ids_by_mode: Dict[str, Dict[int, List[int]]] = {
        mode: {layer: [] for layer in exit_layers}
        for mode in per_layer_oracle_modes
    }
    oracle_accept_counts_by_mode: Dict[str, Dict[int, int]] = {
        mode: {layer: 0 for layer in exit_layers}
        for mode in per_layer_oracle_modes
    }
    oracle_reason_counts_by_mode: Dict[str, Dict[int, Dict[str, int]]] = {
        mode: {layer: {} for layer in exit_layers}
        for mode in per_layer_oracle_modes
    }
    cross_oracle_ids: Dict[str, List[int]] = {
        mode: [] for mode in cross_layer_oracle_modes
    }
    cross_oracle_accept_counts: Dict[str, int] = {
        mode: 0 for mode in cross_layer_oracle_modes
    }
    cross_oracle_reason_counts: Dict[str, Dict[str, int]] = {
        mode: {} for mode in cross_layer_oracle_modes
    }
    earliest_correct_selected_counts: Dict[str, int] = {
        f"L{layer_idx}": 0 for layer_idx in exit_layers
    }
    earliest_correct_selected_counts["full_depth"] = 0
    trace_steps: List[Dict[str, Any]] = []
    non_finite_failure: Optional[Dict[str, Any]] = None

    t0 = time.perf_counter()
    input_ids = prompt_ids

    with torch.no_grad():
        for step_index in range(max_new_tokens):
            try:
                step_out = manual_forward_step(
                    parts=parts,
                    input_ids=input_ids,
                    cache=cache,
                    capture_layers=exit_layers,
                    include_hidden_tensors=include_trace,
                )
            except NonFiniteTensorError as exc:
                non_finite_failure = {"step_index": step_index, **exc.details}
                if include_trace:
                    trace_steps.append(
                        {
                            "step_index": step_index,
                            "numerics_failure": non_finite_failure,
                        }
                    )
                break

            full_logits = step_out["full_logits"]
            try:
                full_token = greedy_token_from_logits(full_logits, "full_logits")
            except NonFiniteTensorError as exc:
                non_finite_failure = {"step_index": step_index, **exc.details}
                if include_trace:
                    trace_steps.append(
                        {
                            "step_index": step_index,
                            "numerics_failure": non_finite_failure,
                            "stage_summaries": step_out.get("stage_summaries"),
                        }
                    )
                break
            full_ids.append(full_token)

            step_record: Dict[str, Any] = {
                "step_index": step_index,
                "full_depth": {
                    "token_id": full_token,
                    "token_text": token_to_text(tokenizer, full_token),
                },
                "layers": {},
            }
            step_exit_tokens: Dict[int, int] = {}
            step_oracles: Dict[str, Any] = {}

            for layer_idx in exit_layers:
                layer_label = f"L{layer_idx}"
                layer_logits = step_out["exit_logits"][layer_idx]
                try:
                    exit_token = greedy_token_from_logits(layer_logits, f"exit_logits.{layer_label}")
                except NonFiniteTensorError as exc:
                    non_finite_failure = {"step_index": step_index, **exc.details}
                    if include_trace:
                        trace_steps.append(
                            {
                                "step_index": step_index,
                                "numerics_failure": non_finite_failure,
                                "stage_summaries": step_out.get("stage_summaries"),
                            }
                        )
                    break
                step_exit_tokens[layer_idx] = exit_token
                exit_ids[layer_idx].append(exit_token)

                composed_logits, agreed = compose_top1_agree_logits(
                    full_logits=full_logits,
                    exit_logits=layer_logits,
                    blend_alpha=blend_alpha,
                )
                composed_token = int(composed_logits.argmax(dim=-1).item())
                composed_ids[layer_idx].append(composed_token)

                agreement_counts[layer_idx] += int(agreed)
                raw_metrics = sequence_metrics(full_ids, exit_ids[layer_idx])
                composed_metrics = sequence_metrics(full_ids, composed_ids[layer_idx])

                step_record["layers"][layer_label] = {
                    "raw_exit": {
                        "token_id": exit_token,
                        "token_text": token_to_text(tokenizer, exit_token),
                        "matches_full_depth": exit_token == full_token,
                        "first_divergence_position_so_far": raw_metrics["first_divergence_position"],
                    },
                    "composed_top1_agree": {
                        "token_id": composed_token,
                        "token_text": token_to_text(tokenizer, composed_token),
                        "matches_full_depth": composed_token == full_token,
                        "first_divergence_position_so_far": composed_metrics["first_divergence_position"],
                    },
                    "top1_agree": {
                        "matches_full_depth_top1": agreed,
                        "agreement_rate_so_far": round(
                            agreement_counts[layer_idx] / len(full_ids), 6
                        ),
                    },
                }
                if include_trace:
                    step_record["layers"][layer_label]["diagnostics"] = build_step_diagnostics(
                        full_hidden=step_out["full_hidden"],
                        exit_hidden=step_out["exit_hidden"][layer_idx],
                        full_logits=full_logits,
                        exit_logits=layer_logits,
                        full_top1=full_token,
                        exit_top1=exit_token,
                        softcap=getattr(parts.config, "final_logit_softcapping", None),
                    )

            if non_finite_failure is not None:
                break

            for mode in per_layer_oracle_modes:
                mode_record = step_oracles.setdefault(mode, {"kind": "single_layer", "per_layer": {}})
                for layer_idx in exit_layers:
                    layer_label = f"L{layer_idx}"
                    oracle_step = evaluate_single_layer_oracle(
                        mode=mode,
                        full_token=full_token,
                        exit_token=step_exit_tokens[layer_idx],
                        exit_logits=step_out["exit_logits"][layer_idx],
                        oracle_config=oracle_config,
                    )
                    oracle_step["selected_token_text"] = token_to_text(
                        tokenizer,
                        oracle_step["selected_token_id"],
                    )
                    mode_record["per_layer"][layer_label] = oracle_step
                    oracle_ids_by_mode[mode][layer_idx].append(oracle_step["selected_token_id"])
                    oracle_accept_counts_by_mode[mode][layer_idx] += int(oracle_step["accepted"])
                    reason = oracle_step["decision_reason"]
                    reason_counts = oracle_reason_counts_by_mode[mode][layer_idx]
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

            if "two_layer_top1_agree" in cross_layer_oracle_modes:
                oracle_step = evaluate_two_layer_top1_agree_oracle(
                    full_token=full_token,
                    exit_tokens=step_exit_tokens,
                    exit_layers=exit_layers,
                )
                if oracle_step.get("available", True):
                    oracle_step["selected_token_text"] = token_to_text(
                        tokenizer,
                        oracle_step["selected_token_id"],
                    )
                    cross_oracle_ids["two_layer_top1_agree"].append(oracle_step["selected_token_id"])
                    cross_oracle_accept_counts["two_layer_top1_agree"] += int(oracle_step["accepted"])
                    reason = oracle_step["decision_reason"]
                    reason_counts = cross_oracle_reason_counts["two_layer_top1_agree"]
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                step_oracles["two_layer_top1_agree"] = oracle_step

            if "earliest_correct" in cross_layer_oracle_modes:
                oracle_step = evaluate_earliest_correct_oracle(
                    full_token=full_token,
                    exit_tokens=step_exit_tokens,
                    exit_layers=exit_layers,
                )
                oracle_step["selected_token_text"] = token_to_text(
                    tokenizer,
                    oracle_step["selected_token_id"],
                )
                cross_oracle_ids["earliest_correct"].append(oracle_step["selected_token_id"])
                cross_oracle_accept_counts["earliest_correct"] += int(oracle_step["accepted"])
                earliest_correct_selected_counts[oracle_step["selected_layer"]] += 1
                reason = oracle_step["decision_reason"]
                reason_counts = cross_oracle_reason_counts["earliest_correct"]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                step_oracles["earliest_correct"] = oracle_step

            if include_trace:
                if step_oracles:
                    step_record["oracles"] = step_oracles
                if "stage_summaries" in step_out:
                    step_record["stage_summaries"] = step_out["stage_summaries"]
                trace_steps.append(step_record)

            input_ids = torch.tensor([[full_token]], device=parts.device)
            if tokenizer.eos_token_id is not None and full_token == tokenizer.eos_token_id:
                break

    elapsed = time.perf_counter() - t0
    generation_tps = len(full_ids) / elapsed if elapsed > 0 else 0.0

    result: Dict[str, Any] = {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "shared_context_mode": (
            "full-depth token chosen at each step is used to advance decode state; "
            "exit-layer tokens are candidate comparisons under the same context."
        ),
        "max_new_tokens": max_new_tokens,
        "blend_alpha": blend_alpha,
        "full_depth_ids": full_ids,
        "full_depth_text": tokenizer.decode(full_ids, skip_special_tokens=False),
        "generation_time_s": round(elapsed, 4),
        "generation_tps": round(generation_tps, 2),
    }
    if non_finite_failure is not None:
        result["status"] = "failed_non_finite"
        result["non_finite_failure"] = non_finite_failure
        if include_trace:
            result["trace"] = trace_steps
        return result

    layer_summaries: Dict[str, Any] = {}
    for layer_idx in exit_layers:
        layer_label = f"L{layer_idx}"
        raw_metrics = sequence_metrics(full_ids, exit_ids[layer_idx])
        composed_metrics = sequence_metrics(full_ids, composed_ids[layer_idx])
        layer_summaries[layer_label] = {
            "exit_layer": layer_idx,
            "raw_exit_exact_match_rate": round(raw_metrics["exact_match_rate"], 6),
            "raw_exit_prefix_match_tokens": raw_metrics["prefix_match_tokens"],
            "raw_exit_first_divergence_position": raw_metrics["first_divergence_position"],
            "raw_exit_perfect_match": raw_metrics["perfect_match"],
            "composed_exact_match_rate": round(composed_metrics["exact_match_rate"], 6),
            "composed_prefix_match_tokens": composed_metrics["prefix_match_tokens"],
            "composed_first_divergence_position": composed_metrics["first_divergence_position"],
            "composed_perfect_match": composed_metrics["perfect_match"],
            "top1_agreement_rate": round(agreement_counts[layer_idx] / len(full_ids), 6),
        }

    result["status"] = "ok"
    result["layer_summaries"] = layer_summaries
    oracle_summaries: Dict[str, Any] = {}
    for mode in per_layer_oracle_modes:
        oracle_summaries[mode] = {
            "kind": "single_layer",
            "per_layer": {},
        }
        if mode == "top1_agree_margin":
            oracle_summaries[mode]["margin_threshold"] = round_or_none(
                oracle_config.margin_threshold
            )
        if mode == "top1_agree_entropy":
            oracle_summaries[mode]["entropy_threshold"] = round_or_none(
                oracle_config.entropy_threshold
            )
        for layer_idx in exit_layers:
            layer_label = f"L{layer_idx}"
            oracle_summaries[mode]["per_layer"][layer_label] = oracle_sequence_summary(
                full_ids=full_ids,
                oracle_ids=oracle_ids_by_mode[mode][layer_idx],
                accepted_steps=oracle_accept_counts_by_mode[mode][layer_idx],
                decision_reason_counts=oracle_reason_counts_by_mode[mode][layer_idx],
            )

    if "two_layer_top1_agree" in cross_layer_oracle_modes:
        if len(exit_layers) < 2:
            oracle_summaries["two_layer_top1_agree"] = {
                "kind": "cross_layer",
                "available": False,
                "reason": "requires at least two exit layers",
            }
        else:
            oracle_summaries["two_layer_top1_agree"] = {
                "kind": "cross_layer",
                "available": True,
                "pair": [f"L{exit_layers[0]}", f"L{exit_layers[1]}"],
                **oracle_sequence_summary(
                    full_ids=full_ids,
                    oracle_ids=cross_oracle_ids["two_layer_top1_agree"],
                    accepted_steps=cross_oracle_accept_counts["two_layer_top1_agree"],
                    decision_reason_counts=cross_oracle_reason_counts["two_layer_top1_agree"],
                ),
            }

    if "earliest_correct" in cross_layer_oracle_modes:
        oracle_summaries["earliest_correct"] = {
            "kind": "cross_layer",
            **oracle_sequence_summary(
                full_ids=full_ids,
                oracle_ids=cross_oracle_ids["earliest_correct"],
                accepted_steps=cross_oracle_accept_counts["earliest_correct"],
                decision_reason_counts=cross_oracle_reason_counts["earliest_correct"],
            ),
            "selected_layer_counts": earliest_correct_selected_counts,
        }

    result["oracle_config"] = oracle_config_dict(oracle_config, exit_layers)
    result["oracle_summaries"] = oracle_summaries
    if include_trace:
        result["trace"] = trace_steps
    return result


def build_prompt_summary(trace_result: Dict[str, Any], is_warmup: bool) -> Dict[str, Any]:
    prompt_result = {
        "prompt_id": trace_result["prompt_id"],
        "prompt_text": trace_result["prompt_text"],
        "is_warmup": is_warmup,
        "full_depth_tokens": len(trace_result["full_depth_ids"]),
        "generation_time_s": trace_result["generation_time_s"],
        "generation_tps": trace_result["generation_tps"],
        "shared_context_mode": trace_result["shared_context_mode"],
        "layer_results": trace_result["layer_summaries"],
    }
    if "oracle_summaries" in trace_result:
        prompt_result["oracle_results"] = trace_result["oracle_summaries"]
    return prompt_result


def aggregate_prompt_results(prompt_results: List[Dict[str, Any]], exit_layers: List[int]) -> Dict[str, Any]:
    scored = [row for row in prompt_results if not row["is_warmup"]]
    aggregate_rows = scored if scored else prompt_results
    scope = (
        "selected prompts excluding warmup"
        if scored
        else "selected prompts (warmup included because no scored prompts remain)"
    )

    aggregates: Dict[str, Any] = {}
    oracle_aggregates: Dict[str, Any] = {}
    for layer_idx in exit_layers:
        layer_label = f"L{layer_idx}"
        raw_ems = [r["layer_results"][layer_label]["raw_exit_exact_match_rate"] for r in aggregate_rows]
        raw_perfect = sum(1 for r in aggregate_rows if r["layer_results"][layer_label]["raw_exit_perfect_match"])
        comp_ems = [r["layer_results"][layer_label]["composed_exact_match_rate"] for r in aggregate_rows]
        comp_perfect = sum(1 for r in aggregate_rows if r["layer_results"][layer_label]["composed_perfect_match"])
        agreements = [r["layer_results"][layer_label]["top1_agreement_rate"] for r in aggregate_rows]

        aggregates[layer_label] = {
            "exit_layer": layer_idx,
            "raw_exit_avg_exact_match": round(sum(raw_ems) / len(raw_ems), 6),
            "raw_exit_perfect_count": raw_perfect,
            "raw_exit_total_scored": len(aggregate_rows),
            "composed_avg_exact_match": round(sum(comp_ems) / len(comp_ems), 6),
            "composed_perfect_count": comp_perfect,
            "composed_total_scored": len(aggregate_rows),
            "avg_top1_agreement_rate": round(sum(agreements) / len(agreements), 6),
        }

    full_tps = [r["generation_tps"] for r in aggregate_rows]
    aggregates["full_depth"] = {
        "avg_exact_match": 1.0,
        "perfect_count": len(aggregate_rows),
        "total_scored": len(aggregate_rows),
        "avg_generation_tps": round(sum(full_tps) / len(full_tps), 2),
    }

    first_oracle_results = aggregate_rows[0].get("oracle_results") if aggregate_rows else None
    if first_oracle_results:
        for mode, mode_summary in first_oracle_results.items():
            if mode_summary.get("kind") == "single_layer":
                oracle_aggregates[mode] = {
                    "kind": "single_layer",
                    "per_layer": {},
                }
                if "margin_threshold" in mode_summary:
                    oracle_aggregates[mode]["margin_threshold"] = mode_summary["margin_threshold"]
                if "entropy_threshold" in mode_summary:
                    oracle_aggregates[mode]["entropy_threshold"] = mode_summary["entropy_threshold"]
                for layer_idx in exit_layers:
                    layer_label = f"L{layer_idx}"
                    layer_rows = [r["oracle_results"][mode]["per_layer"][layer_label] for r in aggregate_rows]
                    acceptance_rates = [row["acceptance_rate"] for row in layer_rows]
                    oracle_ems = [row["oracle_composed_exact_match_rate"] for row in layer_rows]
                    oracle_perfect = sum(1 for row in layer_rows if row["oracle_composed_perfect_match"])
                    accepted_steps_total = sum(int(row["accepted_steps"]) for row in layer_rows)
                    total_steps = sum(int(row["total_steps"]) for row in layer_rows)
                    fallback_steps_total = sum(int(row["fallback_steps"]) for row in layer_rows)
                    oracle_aggregates[mode]["per_layer"][layer_label] = {
                        "avg_acceptance_rate": round(sum(acceptance_rates) / len(acceptance_rates), 6),
                        "accepted_steps_total": accepted_steps_total,
                        "fallback_steps_total": fallback_steps_total,
                        "total_steps": total_steps,
                        "micro_acceptance_rate": round(
                            accepted_steps_total / total_steps, 6
                        )
                        if total_steps
                        else 0.0,
                        "avg_fallback_rate": round(fallback_steps_total / total_steps, 6)
                        if total_steps
                        else 0.0,
                        "avg_oracle_composed_exact_match": round(
                            sum(oracle_ems) / len(oracle_ems), 6
                        ),
                        "oracle_composed_perfect_count": oracle_perfect,
                        "oracle_composed_total_scored": len(layer_rows),
                        "decision_reason_counts": merge_reason_counts(
                            [row["decision_reason_counts"] for row in layer_rows]
                        ),
                    }
            else:
                available = bool(mode_summary.get("available", True))
                if not available:
                    oracle_aggregates[mode] = {
                        "kind": "cross_layer",
                        "available": False,
                        "reason": mode_summary.get("reason"),
                    }
                    continue
                summary_rows = [r["oracle_results"][mode] for r in aggregate_rows]
                acceptance_rates = [row["acceptance_rate"] for row in summary_rows]
                oracle_ems = [row["oracle_composed_exact_match_rate"] for row in summary_rows]
                accepted_steps_total = sum(int(row["accepted_steps"]) for row in summary_rows)
                total_steps = sum(int(row["total_steps"]) for row in summary_rows)
                fallback_steps_total = sum(int(row["fallback_steps"]) for row in summary_rows)
                aggregate_summary: Dict[str, Any] = {
                    "kind": "cross_layer",
                    "available": True,
                    "avg_acceptance_rate": round(sum(acceptance_rates) / len(acceptance_rates), 6),
                    "accepted_steps_total": accepted_steps_total,
                    "fallback_steps_total": fallback_steps_total,
                    "total_steps": total_steps,
                    "micro_acceptance_rate": round(
                        accepted_steps_total / total_steps, 6
                    )
                    if total_steps
                    else 0.0,
                    "avg_fallback_rate": round(fallback_steps_total / total_steps, 6)
                    if total_steps
                    else 0.0,
                    "avg_oracle_composed_exact_match": round(
                        sum(oracle_ems) / len(oracle_ems), 6
                    ),
                    "oracle_composed_perfect_count": sum(
                        1 for row in summary_rows if row["oracle_composed_perfect_match"]
                    ),
                    "oracle_composed_total_scored": len(summary_rows),
                    "decision_reason_counts": merge_reason_counts(
                        [row["decision_reason_counts"] for row in summary_rows]
                    ),
                }
                if mode == "two_layer_top1_agree":
                    aggregate_summary["pair"] = mode_summary["pair"]
                if mode == "earliest_correct":
                    selected_counts: Dict[str, int] = {}
                    for row in summary_rows:
                        for label, count in row["selected_layer_counts"].items():
                            selected_counts[label] = selected_counts.get(label, 0) + int(count)
                    aggregate_summary["selected_layer_counts"] = selected_counts
                oracle_aggregates[mode] = aggregate_summary

    return {
        "aggregates_scope": scope,
        "aggregates": aggregates,
        "oracle_aggregates": oracle_aggregates,
    }


def run_benchmark(
    model,
    tokenizer,
    exit_layers: Optional[List[int]],
    output_path: str,
    model_name: str,
    prompt_offset: int = 0,
    prompt_limit: Optional[int] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    blend_alpha: float = DEFAULT_BLEND_ALPHA,
    oracle_config: Optional[OracleConfig] = None,
):
    parts = get_model_parts(model)
    exit_layers = normalize_exit_layers(exit_layers, parts.num_layers)
    oracle_config = oracle_config or OracleConfig(
        modes=list(DEFAULT_ORACLE_MODES),
        margin_threshold=0.0,
        entropy_threshold=float("inf"),
    )
    selected_prompts = select_prompts(prompt_offset=prompt_offset, prompt_limit=prompt_limit)
    model_reference = resolve_reference_for_model(model_name)

    results: Dict[str, Any] = {
        "experiment": "transcender_gpu_reproduction_manual_reference",
        "model": model_name,
        "model_family": parts.family,
        "model_architecture": parts.architecture,
        "num_layers": parts.num_layers,
        "exit_layers_tested": exit_layers,
        "runtime": "huggingface_transformers_gpu_manual_forward",
        "input_hidden_state_scaling": parts.input_hidden_state_scaling,
        "attention_mask_style": parts.attention_mask_style,
        "max_new_tokens": max_new_tokens,
        "blend_alpha": blend_alpha,
        "oracle_config": oracle_config_dict(oracle_config, exit_layers),
        "warmup_prompt": "P1",
        "prompt_offset": prompt_offset,
        "prompt_limit": len(selected_prompts),
        "notes": (
            "Trustworthy reference path. Manual prefill + step decode through "
            "embed_tokens, decoder layers, final norm, and lm_head. Raw exit "
            "tokens and composed tokens are reported separately. Oracle "
            "summaries are final-aware verifier diagnostics, not deployable "
            "serving policies or speed claims."
        ),
        "prompt_results": [],
    }
    if model_reference is not None:
        results["model_reference"] = model_reference

    print(
        f"Manual-reference path: family={parts.family} "
        f"architecture={parts.architecture} exit_layers={exit_layers}"
    )

    for prompt_index, prompt_text in selected_prompts:
        prompt_id = f"P{prompt_index + 1}"
        is_warmup = prompt_index == WARMUP_INDEX
        print(
            f"\n{'[WARMUP] ' if is_warmup else ''}Processing {prompt_id}: "
            f"{prompt_text[:60]}..."
        )

        trace_result = run_shared_context_decode(
            parts=parts,
            tokenizer=tokenizer,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            exit_layers=exit_layers,
            max_new_tokens=max_new_tokens,
            blend_alpha=blend_alpha,
            include_trace=False,
            oracle_config=oracle_config,
        )
        if trace_result.get("status") != "ok":
            results["status"] = "failed_non_finite"
            results["non_finite_failure"] = {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                **trace_result["non_finite_failure"],
            }
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2))
            failure = results["non_finite_failure"]
            raise RuntimeError(
                "Manual-reference path aborted on non-finite tensors at "
                f"step {failure['step_index']} stage={failure['stage']}. "
                f"Partial results written to {out_path}."
            )
        prompt_result = build_prompt_summary(trace_result, is_warmup=is_warmup)
        results["prompt_results"].append(prompt_result)

        for layer_idx in exit_layers:
            layer_label = f"L{layer_idx}"
            layer_result = prompt_result["layer_results"][layer_label]
            print(
                f"  {layer_label}: raw_exit_EM={layer_result['raw_exit_exact_match_rate']:.3f} "
                f"composed_EM={layer_result['composed_exact_match_rate']:.3f} "
                f"agree={layer_result['top1_agreement_rate']:.3f}"
            )

    aggregate_bundle = aggregate_prompt_results(results["prompt_results"], exit_layers)
    results.update(aggregate_bundle)
    results["status"] = "ok"

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Results written to {out_path}")
    print(f"Aggregates ({results['aggregates_scope']}):")
    for layer_idx in exit_layers:
        layer_label = f"L{layer_idx}"
        agg = results["aggregates"][layer_label]
        print(
            f"  {layer_label}: raw_exit_EM={agg['raw_exit_avg_exact_match']:.3f} "
            f"({agg['raw_exit_perfect_count']}/{agg['raw_exit_total_scored']} perfect), "
            f"composed_EM={agg['composed_avg_exact_match']:.3f} "
            f"({agg['composed_perfect_count']}/{agg['composed_total_scored']} perfect), "
            f"agree={agg['avg_top1_agreement_rate']:.3f}"
        )
    print(f"{'=' * 60}")


def run_debug_trace(
    model,
    tokenizer,
    prompt_id: str,
    exit_layers: Optional[List[int]],
    output_path: str,
    max_new_tokens: int,
    blend_alpha: float,
    oracle_config: Optional[OracleConfig] = None,
):
    parts = get_model_parts(model)
    exit_layers = normalize_exit_layers(exit_layers, parts.num_layers)
    oracle_config = oracle_config or OracleConfig(
        modes=list(DEFAULT_ORACLE_MODES),
        margin_threshold=0.0,
        entropy_threshold=float("inf"),
    )
    model_name = getattr(model.config, "_name_or_path", type(model).__name__)

    prompt_index = resolve_prompt_index(prompt_id)
    prompt_text = PROMPTS[prompt_index]
    trace_result = run_shared_context_decode(
        parts=parts,
        tokenizer=tokenizer,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        exit_layers=exit_layers,
        max_new_tokens=max_new_tokens,
        blend_alpha=blend_alpha,
        include_trace=True,
        oracle_config=oracle_config,
    )
    trace_result["debug_mode"] = True
    trace_result["model"] = model_name
    trace_result["model_family"] = parts.family
    trace_result["model_architecture"] = parts.architecture
    trace_result["input_hidden_state_scaling"] = parts.input_hidden_state_scaling
    trace_result["attention_mask_style"] = parts.attention_mask_style

    model_reference = resolve_reference_for_model(model_name)
    if model_reference is not None:
        trace_result["model_reference"] = model_reference

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace_result, indent=2))

    print(f"\nDebug trace written to {out_path}")
    if trace_result.get("status") != "ok":
        failure = trace_result["non_finite_failure"]
        print(
            "Non-finite failure: "
            f"step={failure['step_index']} stage={failure['stage']}"
        )
        raise RuntimeError(
            "Manual-reference path detected non-finite tensors. "
            f"Diagnostic trace written to {out_path}."
        )
    print(
        f"Manual-reference path: family={parts.family} "
        f"architecture={parts.architecture} exit_layers={exit_layers}"
    )
    for layer_idx in exit_layers:
        layer_label = f"L{layer_idx}"
        summary = trace_result["layer_summaries"][layer_label]
        print(
            f"  {layer_label}: raw_exit_EM={summary['raw_exit_exact_match_rate']:.3f} "
            f"composed_EM={summary['composed_exact_match_rate']:.3f} "
            f"agree={summary['top1_agreement_rate']:.3f} "
            f"first_raw_div={summary['raw_exit_first_divergence_position']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="GPU validation of the Track A penultimate-layer frontier"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "4bit", "8bit"],
        default="4bit",
        help="Quantization mode",
    )
    parser.add_argument(
        "--exit-layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Exit layers to test. Defaults to the model's penultimate-1 and "
            "penultimate layers."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/track_a_gpu/qwen3_gpu_reproduction.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--prompt-offset",
        type=int,
        default=0,
        help="Zero-based prompt offset into the fixed suite",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=0,
        help="Number of prompts to run. 0 means the full suite.",
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        help="Single prompt id like P2. Used by --debug-trace.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Decode length per prompt (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=DEFAULT_BLEND_ALPHA,
        help=f"Blend weight for top1_agree composition (default: {DEFAULT_BLEND_ALPHA})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for non-quantized loading (default: cuda)",
    )
    parser.add_argument(
        "--oracle-modes",
        type=str,
        nargs="+",
        choices=SUPPORTED_ORACLE_MODES,
        default=None,
        help=(
            "Oracle-style final-aware acceptance rules to evaluate. Defaults "
            f"to {DEFAULT_ORACLE_MODES} to preserve current semantics."
        ),
    )
    parser.add_argument(
        "--oracle-margin-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum exit top1-top2 margin for top1_agree_margin. "
            "Only used when that oracle is enabled."
        ),
    )
    parser.add_argument(
        "--oracle-entropy-threshold",
        type=float,
        default=float("inf"),
        help=(
            "Maximum exit entropy for top1_agree_entropy. "
            "Only used when that oracle is enabled."
        ),
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help="Write a single-prompt per-token trace using the manual reference path.",
    )
    args = parser.parse_args()

    print(f"Loading {args.model} (quantize={args.quantize})...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.quantize, args.device)
    print("Model loaded.")
    oracle_config = OracleConfig(
        modes=normalize_oracle_modes(args.oracle_modes),
        margin_threshold=args.oracle_margin_threshold,
        entropy_threshold=args.oracle_entropy_threshold,
    )

    if args.debug_trace:
        if not args.prompt_id:
            raise ValueError("--debug-trace requires --prompt-id, e.g. --prompt-id P2")
        run_debug_trace(
            model=model,
            tokenizer=tokenizer,
            prompt_id=args.prompt_id,
            exit_layers=args.exit_layers,
            output_path=args.output,
            max_new_tokens=args.max_new_tokens,
            blend_alpha=args.blend_alpha,
            oracle_config=oracle_config,
        )
        return

    run_benchmark(
        model=model,
        tokenizer=tokenizer,
        exit_layers=args.exit_layers,
        output_path=args.output,
        model_name=args.model,
        prompt_offset=args.prompt_offset,
        prompt_limit=args.prompt_limit or None,
        max_new_tokens=args.max_new_tokens,
        blend_alpha=args.blend_alpha,
        oracle_config=oracle_config,
    )


if __name__ == "__main__":
    main()
