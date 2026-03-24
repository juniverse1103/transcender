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


def load_model_and_tokenizer(model_name: str, quantize: str, device: str):
    """Load the causal LM without relying on hidden-state capture side effects."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
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

    if hasattr(model, "language_model") and not hasattr(model, "model"):
        raise RuntimeError(
            "Loaded model appears to be a multimodal Gemma 3 conditional-"
            "generation checkpoint. The manual-reference path only supports "
            "text causal-LM checkpoints such as google/gemma-3-4b-it."
        )

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


def get_model_parts(model) -> ModelParts:
    backbone = getattr(model, "model", None)
    if backbone is None:
        if hasattr(model, "language_model"):
            raise RuntimeError(
                "Loaded model appears to be a multimodal Gemma 3 conditional-"
                "generation checkpoint. The manual-reference path only "
                "supports text causal-LM checkpoints such as "
                "google/gemma-3-4b-it."
            )
        raise AttributeError(
            f"{type(model).__name__} has no `.model` backbone. "
            "Manual adaptation is required for this architecture."
        )

    if hasattr(backbone, "language_model") and not hasattr(backbone, "embed_tokens"):
        raise RuntimeError(
            "Loaded backbone appears to be a multimodal Gemma 3 wrapper. "
            "The manual-reference path only supports text causal-LM "
            "checkpoints such as google/gemma-3-4b-it."
        )

    required = ["embed_tokens", "layers", "norm", "rotary_emb"]
    missing = [name for name in required if not hasattr(backbone, name)]
    if missing:
        raise AttributeError(
            f"Backbone {type(backbone).__name__} is missing required attrs: {missing}"
        )
    if not hasattr(model, "lm_head"):
        raise AttributeError(f"{type(model).__name__} has no `lm_head`.")

    # The trustworthy reference path assumes one GPU with the whole model on it.
    # If the model is sharded, this script should fail loudly rather than silently
    # measuring a partly-wrong path.
    devices = {
        module_device(backbone.embed_tokens),
        module_device(backbone.norm),
        module_device(model.lm_head),
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
        lm_head=model.lm_head,
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
    hidden_state = hidden_state.to(parts.device)
    normed = parts.final_norm(hidden_state)
    logits = parts.lm_head(normed)
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
) -> Dict[str, Any]:
    """
    Explicit one-step forward pass through embeddings, decoder layers, final
    norm, and lm_head.

    This does not rely on `output_hidden_states` or `generate()` behavior.
    """
    input_ids = input_ids.to(parts.device)
    inputs_embeds = parts.embed_tokens(input_ids)

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
    captured_hidden: Dict[int, torch.Tensor] = {}

    for layer_idx, decoder_layer in enumerate(parts.layers[: parts.num_layers]):
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
        if layer_idx in capture_layers:
            captured_hidden[layer_idx] = hidden_states[:, -1:, :].detach()

    full_hidden = hidden_states[:, -1:, :].detach()
    full_logits = project_hidden_to_logits(parts, full_hidden)
    exit_logits = {
        layer_idx: project_hidden_to_logits(parts, hidden)
        for layer_idx, hidden in captured_hidden.items()
    }

    return {
        "full_logits": full_logits.detach().cpu(),
        "exit_logits": {k: v.detach().cpu() for k, v in exit_logits.items()},
    }


def run_shared_context_decode(
    parts: ModelParts,
    tokenizer,
    prompt_id: str,
    prompt_text: str,
    exit_layers: List[int],
    max_new_tokens: int,
    blend_alpha: float,
    include_trace: bool,
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
    trace_steps: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    input_ids = prompt_ids

    with torch.no_grad():
        for step_index in range(max_new_tokens):
            step_out = manual_forward_step(
                parts=parts,
                input_ids=input_ids,
                cache=cache,
                capture_layers=exit_layers,
            )

            full_logits = step_out["full_logits"]
            full_token = int(full_logits.argmax(dim=-1).item())
            full_ids.append(full_token)

            step_record: Dict[str, Any] = {
                "step_index": step_index,
                "full_depth": {
                    "token_id": full_token,
                    "token_text": token_to_text(tokenizer, full_token),
                },
                "layers": {},
            }

            for layer_idx in exit_layers:
                layer_label = f"L{layer_idx}"
                layer_logits = step_out["exit_logits"][layer_idx]
                exit_token = int(layer_logits.argmax(dim=-1).item())
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
                trace_steps.append(step_record)

            input_ids = torch.tensor([[full_token]], device=parts.device)
            if tokenizer.eos_token_id is not None and full_token == tokenizer.eos_token_id:
                break

    elapsed = time.perf_counter() - t0
    generation_tps = len(full_ids) / elapsed if elapsed > 0 else 0.0

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
        "layer_summaries": layer_summaries,
    }
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

    return {
        "aggregates_scope": scope,
        "aggregates": aggregates,
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
):
    parts = get_model_parts(model)
    exit_layers = normalize_exit_layers(exit_layers, parts.num_layers)
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
        "warmup_prompt": "P1",
        "prompt_offset": prompt_offset,
        "prompt_limit": len(selected_prompts),
        "notes": (
            "Trustworthy reference path. Manual prefill + step decode through "
            "embed_tokens, decoder layers, final norm, and lm_head. Raw exit "
            "tokens and composed tokens are reported separately."
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
):
    parts = get_model_parts(model)
    exit_layers = normalize_exit_layers(exit_layers, parts.num_layers)
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
        "--debug-trace",
        action="store_true",
        help="Write a single-prompt per-token trace using the manual reference path.",
    )
    args = parser.parse_args()

    print(f"Loading {args.model} (quantize={args.quantize})...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.quantize, args.device)
    print("Model loaded.")

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
    )


if __name__ == "__main__":
    main()
