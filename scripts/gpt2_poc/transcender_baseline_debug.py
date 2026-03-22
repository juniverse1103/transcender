"""
Strict baseline-equivalence debug harness for the Transcender MLX engine.

This mode isolates manual baseline correctness only:
  - shared Harmony prompt rendering
  - reasoning_effort="medium"
  - deterministic greedy decoding
  - dynamic=False on the manual engine path

It compares the official mlx_lm greedy path against the manual engine path and
also records the legacy full-prompt-prefill behavior that previously diverged.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models import cache as mlx_cache
from mlx_lm.sample_utils import make_sampler

from transcender_engine import (
    GptOssConfig,
    MLXDynamicExpertEngine,
    apply_harmony_template,
    load_resolved_mlx_model,
)


DEFAULT_MODEL_PATH = str(
    (Path(__file__).resolve().parent.parent / "gpt-oss-20b-raw").resolve()
)
DEFAULT_OUTPUT_PATH = str(
    (Path(__file__).resolve().parent / "baseline_equivalence_debug.json").resolve()
)
DEFAULT_MAX_NEW_TOKENS = 64
TOKEN_PREVIEW_COUNT = 64
TOP_K = 10

SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "Explain quantum entanglement in simple terms."
REASONING_EFFORT = "medium"


def first_divergence_position(left: List[int], right: List[int]) -> Optional[int]:
    for idx, (left_tok, right_tok) in enumerate(zip(left, right)):
        if left_tok != right_tok:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compare_sequences(left: List[int], right: List[int]) -> Dict[str, Any]:
    paired = list(zip(left, right))
    prefix = 0
    for left_tok, right_tok in paired:
        if left_tok != right_tok:
            break
        prefix += 1

    exact_match_rate = (
        sum(1 for left_tok, right_tok in paired if left_tok == right_tok)
        / max(len(paired), 1)
    )
    divergence = first_divergence_position(left, right)
    return {
        "prefix_match_tokens": prefix,
        "exact_match_rate": exact_match_rate,
        "first_divergence_position": divergence,
        "passed": left == right,
        "divergence_window": divergence_window(left, right, divergence),
    }


def divergence_window(
    left: List[int],
    right: List[int],
    divergence: Optional[int],
    radius: int = 4,
) -> Optional[Dict[str, Any]]:
    if divergence is None:
        return None
    start = max(divergence - radius, 0)
    stop = min(max(len(left), len(right)), divergence + radius + 1)
    return {
        "start": start,
        "stop": stop,
        "manual": left[start:stop],
        "official": right[start:stop],
    }


def preview_text(text: str, limit: int = 240) -> str:
    return text[:limit]


def get_eos_ids(tokenizer) -> List[int]:
    eos_ids = list(getattr(tokenizer, "eos_token_ids", []) or [])
    if not eos_ids:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            eos_ids = [int(eos_token_id)]
    return sorted(int(token_id) for token_id in eos_ids)


def topk_from_scores(scores, label: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    if len(scores.shape) != 1:
        scores = scores.reshape(-1)
    k = min(k, scores.shape[0])
    indices = mx.argsort(-scores, axis=-1)[:k]
    values = scores[indices]
    mx.eval(indices, values)
    return [
        {
            "rank": rank + 1,
            "token_id": int(indices[rank].item()),
            label: float(values[rank].item()),
        }
        for rank in range(k)
    ]


def flatten_cache_state(cache_entry) -> List[Any]:
    state = getattr(cache_entry, "state", None)
    if state is None:
        return []
    if isinstance(state, tuple):
        return [item for item in state if item is not None]
    if isinstance(state, list):
        return [item for item in state if item is not None]
    return [state]


def summarize_cache(cache) -> Dict[str, Any]:
    layers = []
    total_nbytes = 0
    for layer_idx, cache_entry in enumerate(cache):
        nbytes = int(getattr(cache_entry, "nbytes", 0) or 0)
        total_nbytes += nbytes
        size = None
        if hasattr(cache_entry, "size"):
            try:
                size = int(cache_entry.size())
            except Exception:
                size = None
        layers.append(
            {
                "layer": layer_idx,
                "cache_type": type(cache_entry).__name__,
                "offset": int(getattr(cache_entry, "offset", 0) or 0),
                "size": size,
                "nbytes": nbytes,
                "state_shapes": [
                    list(array.shape)
                    for array in flatten_cache_state(cache_entry)
                    if hasattr(array, "shape")
                ],
            }
        )
    return {
        "total_layers": len(layers),
        "total_nbytes": total_nbytes,
        "total_gb": total_nbytes / (1024**3),
        "layers": layers,
    }


def tokenizer_debug(tokenizer) -> Dict[str, Any]:
    return {
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "eos_token_ids": get_eos_ids(tokenizer),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }


def legacy_full_prompt_prefill_generate(
    engine: MLXDynamicExpertEngine,
    prompt_ids: List[int],
    max_new_tokens: int,
) -> Dict[str, Any]:
    """
    Reproduce the old manual baseline behavior for root-cause isolation.

    This path incorrectly prefills the entire prompt, including the final
    prompt token, before selecting the first generated token.
    """
    input_ids = mx.array(prompt_ids, dtype=mx.int32)[None, :]
    cache = engine._make_cache()
    t0 = time.perf_counter()
    prefill = engine._prefill_tokens(
        input_ids,
        cache=cache,
        dynamic=False,
        prefill_step_size=engine.prefill_step_size,
    )
    first_logits = prefill["logits"][0, -1]
    first_logprobs = first_logits - mx.logsumexp(first_logits, keepdims=True)
    next_token = mx.argmax(first_logits, axis=-1).reshape(1, 1)
    mx.eval(first_logits, first_logprobs, next_token)
    ttft_s = time.perf_counter() - t0

    generated_ids = [int(next_token.item())]
    current_input = next_token
    for _ in range(1, max_new_tokens):
        if engine._is_eos(generated_ids[-1]):
            break
        step_out = engine._dynamic_forward(
            current_input,
            cache=cache,
            dynamic=False,
            return_logits=True,
        )
        current_input = mx.argmax(step_out["logits"][0, -1], axis=-1).reshape(1, 1)
        mx.eval(current_input)
        generated_ids.append(int(current_input.item()))

    elapsed_s = time.perf_counter() - t0
    generation_window = max(elapsed_s - ttft_s, 1e-6)
    return {
        "generated_ids": generated_ids,
        "output_text": engine.tokenizer.decode(generated_ids, skip_special_tokens=True),
        "ttft_s": ttft_s,
        "elapsed_s": elapsed_s,
        "generation_tps": max(len(generated_ids) - 1, 0) / generation_window,
        "first_token_id": generated_ids[0] if generated_ids else None,
        "first_8_token_ids": generated_ids[:8],
        "first_64_token_ids": generated_ids[:TOKEN_PREVIEW_COUNT],
        "first_token_logits_top10": topk_from_scores(first_logits, "logit"),
        "first_token_logprobs_top10": topk_from_scores(first_logprobs, "logprob"),
    }


def official_prefill_cache_debug(model, prompt_ids: List[int], prefill_step_size: int):
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    prompt_cache = mlx_cache.make_prompt_cache(model)
    prompt_processed_tokens = 0
    total_prompt_tokens = len(prompt_ids)
    remaining = prompt

    while total_prompt_tokens - prompt_processed_tokens > 1:
        n_to_process = min(prefill_step_size, (total_prompt_tokens - prompt_processed_tokens) - 1)
        model(remaining[:n_to_process][None], cache=prompt_cache)
        mx.eval([cache_entry.state for cache_entry in prompt_cache])
        prompt_processed_tokens += n_to_process
        remaining = remaining[n_to_process:]
        mx.clear_cache()

    return {
        "prompt_tokens_processed": prompt_processed_tokens,
        "cache": summarize_cache(prompt_cache),
    }


def official_greedy_generate(
    model,
    tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    prefill_step_size: int,
) -> Dict[str, Any]:
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    sampler = make_sampler(temp=0.0)
    generated_ids: List[int] = []
    first_logprobs = None

    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    ttft_s: Optional[float] = None

    token_generator = generate_step(
        prompt,
        model,
        max_tokens=max_new_tokens,
        sampler=sampler,
        logits_processors=None,
        prefill_step_size=prefill_step_size,
    )
    for token, logprobs in token_generator:
        if ttft_s is None:
            ttft_s = time.perf_counter() - t0
            first_logprobs = logprobs
        token_id = int(token)
        generated_ids.append(token_id)
        if token_id in get_eos_ids(tokenizer):
            break

    elapsed_s = time.perf_counter() - t0
    ttft_s = ttft_s if ttft_s is not None else elapsed_s
    generation_window = max(elapsed_s - ttft_s, 1e-6)
    return {
        "generated_ids": generated_ids,
        "output_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "ttft_s": ttft_s,
        "elapsed_s": elapsed_s,
        "generation_tps": max(len(generated_ids) - 1, 0) / generation_window,
        "peak_memory_gb": mx.get_peak_memory() / (1024**3),
        "first_token_id": generated_ids[0] if generated_ids else None,
        "first_8_token_ids": generated_ids[:8],
        "first_64_token_ids": generated_ids[:TOKEN_PREVIEW_COUNT],
        "first_token_logprobs_top10": (
            topk_from_scores(first_logprobs, "logprob")
            if first_logprobs is not None
            else []
        ),
    }


def manual_baseline_generate(
    engine: MLXDynamicExpertEngine,
    prompt_ids: List[int],
    max_new_tokens: int,
) -> Dict[str, Any]:
    input_ids = mx.array(prompt_ids, dtype=mx.int32)[None, :]
    cache = engine._make_cache()
    prefix_ids = input_ids[:, :-1]
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()

    prefix_cache_debug = None
    if prefix_ids.shape[1] > 0:
        engine._prefill_tokens(
            prefix_ids,
            cache=cache,
            dynamic=False,
            prefill_step_size=engine.prefill_step_size,
        )
        prefix_cache_debug = {
            "prompt_tokens_processed": int(prefix_ids.shape[1]),
            "cache": summarize_cache(cache),
        }

    first_step = engine._dynamic_forward(
        input_ids[:, -1:],
        cache=cache,
        dynamic=False,
        return_logits=True,
    )
    first_logits = first_step["logits"][0, -1]
    first_logprobs = first_logits - mx.logsumexp(first_logits, keepdims=True)
    next_token = mx.argmax(first_logits, axis=-1).reshape(1, 1)
    mx.eval(first_logits, first_logprobs, next_token)
    ttft_s = time.perf_counter() - t0

    generated_ids = [int(next_token.item())]
    current_input = next_token
    for _ in range(1, max_new_tokens):
        if engine._is_eos(generated_ids[-1]):
            break
        step_out = engine._dynamic_forward(
            current_input,
            cache=cache,
            dynamic=False,
            return_logits=True,
        )
        current_input = mx.argmax(step_out["logits"][0, -1], axis=-1).reshape(1, 1)
        mx.eval(current_input)
        generated_ids.append(int(current_input.item()))

    elapsed_s = time.perf_counter() - t0
    generation_window = max(elapsed_s - ttft_s, 1e-6)
    return {
        "generated_ids": generated_ids,
        "output_text": engine.tokenizer.decode(generated_ids, skip_special_tokens=True),
        "ttft_s": ttft_s,
        "elapsed_s": elapsed_s,
        "generation_tps": max(len(generated_ids) - 1, 0) / generation_window,
        "peak_memory_gb": mx.get_peak_memory() / (1024**3),
        "first_token_id": generated_ids[0] if generated_ids else None,
        "first_8_token_ids": generated_ids[:8],
        "first_64_token_ids": generated_ids[:TOKEN_PREVIEW_COUNT],
        "first_token_logits_top10": topk_from_scores(first_logits, "logit"),
        "first_token_logprobs_top10": topk_from_scores(first_logprobs, "logprob"),
        "prefill_cache_debug": prefix_cache_debug,
    }


def run_debug(
    model_path: str,
    output_path: str,
    max_new_tokens: int,
    memory_limit_gb: float,
):
    model, tokenizer, resolved_model_path = load_resolved_mlx_model(
        model_path,
        lazy=True,
    )
    prompt_text, messages = apply_harmony_template(
        tokenizer,
        user_prompt=USER_PROMPT,
        system_prompt=SYSTEM_PROMPT,
        reasoning_effort=REASONING_EFFORT,
    )
    prompt_ids = tokenizer.encode(prompt_text)

    engine = MLXDynamicExpertEngine(
        model=model,
        tokenizer=tokenizer,
        config=GptOssConfig(
            hard_exit_layer=22,
            entropy_threshold=-1.0,
            min_entropy_streak=2,
            memory_limit_gb=memory_limit_gb,
        ),
        hard_exit_layer=22,
        entropy_threshold=-1.0,
        min_entropy_streak=2,
        memory_limit_gb=memory_limit_gb,
    )

    manual = manual_baseline_generate(
        engine=engine,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
    )
    official = official_greedy_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        prefill_step_size=engine.prefill_step_size,
    )
    legacy = legacy_full_prompt_prefill_generate(
        engine=engine,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
    )

    equivalence = compare_sequences(manual["generated_ids"], official["generated_ids"])
    legacy_comparison = compare_sequences(legacy["generated_ids"], official["generated_ids"])

    payload = {
        "experiment": "transcender_manual_baseline_equivalence_debug",
        "resolved_model_path": resolved_model_path,
        "prompt": {
            "system": SYSTEM_PROMPT,
            "user": USER_PROMPT,
            "reasoning_effort": REASONING_EFFORT,
            "rendered_harmony_prompt": prompt_text,
            "prompt_token_ids": prompt_ids,
            "prompt_length": len(prompt_ids),
            "messages": messages,
        },
        "generation_settings": {
            "max_new_tokens": max_new_tokens,
            "prefill_step_size": engine.prefill_step_size,
            "dynamic": False,
            "sampler": "greedy_argmax_temp_0",
            "logits_processors": None,
            "soft_skip_status": "disabled by dynamic=False",
            "hard_exit_status": "ignored by dynamic=False full-depth baseline",
        },
        "tokenizer": tokenizer_debug(tokenizer),
        "eos_and_stop_handling": {
            "shared_eos_ids": get_eos_ids(tokenizer),
            "manual_stops_on_eos": True,
            "official_generate_step_stops_on_eos": False,
            "official_harness_stops_on_eos": True,
            "stop_token_suppression": "none",
            "tokenizer_special_token_config_differs": False,
        },
        "manual_baseline": manual,
        "official_mlx": official,
        "official_prefill_cache_debug": official_prefill_cache_debug(
            model=model,
            prompt_ids=prompt_ids,
            prefill_step_size=engine.prefill_step_size,
        ),
        "legacy_full_prompt_prefill": legacy,
        "equivalence": equivalence,
        "legacy_vs_official": legacy_comparison,
        "root_cause": {
            "bucket": "prefill mismatch",
            "status": "fixed" if equivalence["passed"] else "still_failing",
            "summary": (
                "The old manual baseline diverged because it prefills the entire "
                "prompt before sampling the first generated token. Official "
                "mlx_lm.generate_step() instead prefills prompt[:-1] and then "
                "runs a single-token forward pass on the last prompt token. "
                "That difference changes rotating sliding-window KV-cache update "
                "behavior and caused the divergence that began at token 17."
            ),
        },
        "official_path_audit": {
            "sampler": "make_sampler(temp=0.0) -> argmax",
            "logits_processors": "None",
            "repetition_penalty": "None",
            "presence_penalty": "None",
            "frequency_penalty": "None",
            "prompt_prefill_behavior": "prefill prompt[:-1], then step on prompt[-1:]",
            "cache_update_order": (
                "official path updates cache during multi-token prefill chunks, "
                "then performs the first decode step as a single-token cache update"
            ),
            "eos_behavior": (
                "generate_step itself does not stop on EOS; this debug harness "
                "stops both paths on the same EOS set for fair comparison"
            ),
        },
    }

    Path(output_path).write_text(json.dumps(payload, indent=2))
    print_summary(payload, output_path)


def print_summary(payload: Dict[str, Any], output_path: str):
    equivalence = payload["equivalence"]
    legacy = payload["legacy_vs_official"]
    manual = payload["manual_baseline"]
    official = payload["official_mlx"]

    print("\nBaseline Equivalence Debug")
    print(
        f"Prompt: system={SYSTEM_PROMPT!r} | user={USER_PROMPT!r} | "
        f"reasoning_effort={REASONING_EFFORT}"
    )
    print(f"JSON output: {output_path}")

    print("\nEquivalence Result")
    print(
        f"status={'PASS' if equivalence['passed'] else 'FAIL'} | "
        f"prefix_match={equivalence['prefix_match_tokens']} | "
        f"exact_match_rate={equivalence['exact_match_rate']:.3f} | "
        f"first_divergence={equivalence['first_divergence_position']}"
    )

    print("\nPerformance")
    print(
        f"manual ttft={manual['ttft_s']:.3f}s | manual tps={manual['generation_tps']:.2f} | "
        f"manual peak_mem={manual['peak_memory_gb']:.2f} GB"
    )
    print(
        f"official ttft={official['ttft_s']:.3f}s | official tps={official['generation_tps']:.2f} | "
        f"official peak_mem={official['peak_memory_gb']:.2f} GB"
    )

    print("\nToken Comparison")
    print(f"manual first 64 ids:   {manual['first_64_token_ids']}")
    print(f"official first 64 ids: {official['first_64_token_ids']}")
    print(f"manual preview:   {preview_text(manual['output_text'])}")
    print(f"official preview: {preview_text(official['output_text'])}")

    print("\nLegacy Failure Reproduction")
    print(
        f"legacy prefix_match={legacy['prefix_match_tokens']} | "
        f"legacy exact_match_rate={legacy['exact_match_rate']:.3f} | "
        f"legacy first_divergence={legacy['first_divergence_position']}"
    )
    print(f"legacy first 64 ids: {payload['legacy_full_prompt_prefill']['first_64_token_ids']}")

    print("\nRoot Cause")
    print(payload["root_cause"]["summary"])


def main():
    parser = argparse.ArgumentParser(
        description="Debug baseline equivalence between manual MLX generation and official mlx_lm greedy decoding."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--memory-limit-gb", type=float, default=30.0)
    args = parser.parse_args()

    run_debug(
        model_path=args.model,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        memory_limit_gb=args.memory_limit_gb,
    )


if __name__ == "__main__":
    main()
