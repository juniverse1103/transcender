"""
Track B — Minimal small-model-first cascade baseline for Transcender.

This module is intentionally isolated from the Track A engine logic. It provides:
  - local draft-model auto-detection
  - model-specific prompt rendering
  - official greedy single-model generation helpers
  - a naive chunked draft-then-verify cascade loop

The cascade is deliberately simple and interpretable:
  1. The draft model proposes K tokens on its own prompt/template path.
  2. The verifier (GPT-OSS 20B) generates the same span on the same accepted text.
  3. Matching verifier tokens are accepted.
  4. At the first divergence, the verifier's token is emitted as correction.

There is no KV sharing, no tree attention, and no speculative kernel work here.
This is a cost-structure benchmark, not a production speculative decoder.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from transcender_engine import (
    apply_harmony_template,
    build_harmony_messages,
    load_resolved_mlx_model,
)


DEFAULT_LARGE_MODEL_PATH = str(
    (Path(__file__).resolve().parent.parent / "gpt-oss-20b-raw").resolve()
)
DEFAULT_OUTPUT_PATH = str(
    (Path(__file__).resolve().parent / "transcender_track_b_benchmark.json").resolve()
)
SYSTEM_PROMPT = "You are a helpful assistant."
REASONING_EFFORT = "medium"
WARMUP_PROMPT_INDEX = 0
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_DRAFT_CHUNK_TOKENS = 8
DEFAULT_PREFILL_STEP_SIZE = 2048

PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "Summarize why the French Revolution was historically important.",
    "Write a short explanation of recursion for a beginner programmer.",
    "Explain the difference between TCP and UDP in plain English.",
    "Describe what photosynthesis does.",
]

# Ordered by preference: Gemma/Llama/Qwen first, practical local fallback last.
DRAFT_MODEL_CANDIDATE_SUBSTRINGS = [
    "gemma-3-text-4b-it",
    "gemma-3-4b-text-it",
    "gemma-3-4b-it",
    "gemma-3-4b",
    "meta-llama-3.1-8b-instruct",
    "llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-4b-instruct",
    "mistral-7b-instruct-v0.2",
]


@dataclass(frozen=True)
class CascadeConfig:
    draft_chunk_tokens: int = DEFAULT_DRAFT_CHUNK_TOKENS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def preview_text(text: str, limit: int = 180) -> str:
    return text[:limit]


def get_eos_ids(tokenizer) -> List[int]:
    eos_ids = list(getattr(tokenizer, "eos_token_ids", []) or [])
    if not eos_ids:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            eos_ids = [int(eos_token_id)]
    return sorted(int(token_id) for token_id in eos_ids)


def first_divergence_position(left: List[int], right: List[int]) -> Optional[int]:
    for idx, (left_tok, right_tok) in enumerate(zip(left, right)):
        if left_tok != right_tok:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def prefix_match_length(left: List[int], right: List[int]) -> int:
    prefix = 0
    for left_tok, right_tok in zip(left, right):
        if left_tok != right_tok:
            break
        prefix += 1
    return prefix


def compare_sequences(left: List[int], right: List[int]) -> Dict[str, Any]:
    paired = list(zip(left, right))
    exact_match_rate = (
        sum(1 for left_tok, right_tok in paired if left_tok == right_tok)
        / max(len(paired), 1)
    )
    return {
        "prefix_match_tokens": prefix_match_length(left, right),
        "exact_match_rate": exact_match_rate,
        "first_divergence_position": first_divergence_position(left, right),
        "passed": left == right,
    }


def compare_text_against_reference_ids(
    candidate_text: str,
    reference_ids: List[int],
    reference_tokenizer,
) -> Dict[str, Any]:
    candidate_ids = reference_tokenizer.encode(candidate_text)
    comparison = compare_sequences(candidate_ids, reference_ids)
    comparison["comparison_space"] = "gpt_oss_reference_token_space"
    comparison["candidate_reference_ids"] = candidate_ids[:64]
    return comparison


def latest_snapshot_for_model(repo_dir: Path) -> Optional[Path]:
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def auto_detect_draft_model(explicit_path: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.exists():
            return str(candidate), {
                "status": "found_explicit",
                "path": str(candidate),
            }
        return None, {
            "status": "missing_explicit",
            "requested_path": str(candidate),
        }

    searched_roots: List[str] = []
    search_roots = [
        Path.home() / "Documents",
        Path.home() / "Documents" / "GitHub",
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / "llama-models",
    ]
    candidates: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        searched_roots.append(str(root))
        try:
            for path in root.rglob("*"):
                if not path.is_dir():
                    continue
                name = path.name.lower()
                if any(needle in name for needle in DRAFT_MODEL_CANDIDATE_SUBSTRINGS):
                    candidates.append(path)
        except OSError:
            continue

    def normalized_candidate(path: Path) -> Optional[Path]:
        if path.name.startswith("models--"):
            snapshot = latest_snapshot_for_model(path)
            return snapshot
        return path if path.is_dir() else None

    # Preserve declared preference order.
    for needle in DRAFT_MODEL_CANDIDATE_SUBSTRINGS:
        for path in candidates:
            if needle in path.name.lower():
                candidate = normalized_candidate(path)
                if candidate is not None:
                    return str(candidate), {
                        "status": "found_local_candidate",
                        "path": str(candidate),
                        "matched_substring": needle,
                    }

    return None, {
        "status": "not_found",
        "searched_roots": searched_roots,
        "candidate_substrings": DRAFT_MODEL_CANDIDATE_SUBSTRINGS,
    }


def apply_generic_chat_template(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> Tuple[str, str]:
    if not hasattr(tokenizer, "apply_chat_template"):
        prompt = (
            f"System: {messages[0]['content']}\n"
            f"User: {messages[-1]['content']}\n"
            "Assistant:"
        )
        return prompt, "plain_fallback"

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return prompt, "tokenizer_chat_template"
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        return prompt, "tokenizer_chat_template_no_add_generation_prompt"


def build_prompt_pack(large_tokenizer) -> List[Dict[str, Any]]:
    prompt_pack: List[Dict[str, Any]] = []
    for prompt_index, user_prompt in enumerate(PROMPTS, start=1):
        prompt_id = f"P{prompt_index}"
        messages = build_harmony_messages(
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        large_prompt_text, _ = apply_harmony_template(
            large_tokenizer,
            messages=messages,
            reasoning_effort=REASONING_EFFORT,
        )
        prompt_pack.append(
            {
                "prompt_id": prompt_id,
                "system": SYSTEM_PROMPT,
                "user": user_prompt,
                "messages": messages,
                "large_prompt_text": large_prompt_text,
                "large_prompt_ids": large_tokenizer.encode(large_prompt_text),
            }
        )
    return prompt_pack


def enrich_prompt_pack_for_draft(prompt_pack: List[Dict[str, Any]], draft_tokenizer):
    for prompt_def in prompt_pack:
        draft_prompt_text, template_mode = apply_generic_chat_template(
            draft_tokenizer,
            prompt_def["messages"],
        )
        prompt_def["draft_prompt_text"] = draft_prompt_text
        prompt_def["draft_prompt_ids"] = draft_tokenizer.encode(draft_prompt_text)
        prompt_def["draft_prompt_template"] = template_mode


def release_models(*objects: Any):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


def greedy_generate_core(
    model,
    tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    prefill_step_size: int,
) -> Dict[str, Any]:
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    sampler = make_sampler(temp=0.0)
    eos_ids = set(get_eos_ids(tokenizer))

    t0 = time.perf_counter()
    ttft_s: Optional[float] = None
    generated_ids: List[int] = []

    token_generator = generate_step(
        prompt,
        model,
        max_tokens=max_new_tokens,
        sampler=sampler,
        prefill_step_size=prefill_step_size,
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
    return {
        "generated_ids": generated_ids,
        "output_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "ttft_s": ttft_s,
        "elapsed_s": elapsed_s,
        "generation_tps": max(len(generated_ids) - 1, 0) / max(elapsed_s - ttft_s, 1e-6),
        "prompt_tokens": len(prompt_ids),
        "tokens_generated": len(generated_ids),
    }


def official_greedy_generate(
    model,
    tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    prefill_step_size: int,
) -> Dict[str, Any]:
    mx.clear_cache()
    mx.reset_peak_memory()
    stats = greedy_generate_core(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        prefill_step_size=prefill_step_size,
    )
    runtime_peak_gb = mx.get_peak_memory() / (1024**3)
    active_memory_gb = mx.get_active_memory() / (1024**3)
    cache_memory_gb = mx.get_cache_memory() / (1024**3)
    stats.update(
        {
            "runtime_peak_memory_gb": runtime_peak_gb,
            "active_memory_gb": active_memory_gb,
            "cache_memory_gb": cache_memory_gb,
            # Use the larger of runtime peak and active memory so the metric
            # reflects loaded-model residency as well as execution-time peak.
            "peak_memory_gb": max(runtime_peak_gb, active_memory_gb),
        }
    )
    return stats


def naive_cascade_generate(
    draft_model,
    draft_tokenizer,
    verifier_model,
    verifier_tokenizer,
    draft_prompt_text: str,
    verifier_prompt_text: str,
    config: CascadeConfig,
) -> Dict[str, Any]:
    verifier_eos_ids = set(get_eos_ids(verifier_tokenizer))
    accepted_large_ids: List[int] = []
    accepted_text = ""
    ttft_s: Optional[float] = None

    draft_iterations = 0
    verifier_iterations = 0
    correction_tokens = 0
    accepted_by_draft_alignment = 0
    total_proposed_large_tokens = 0

    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()

    while len(accepted_large_ids) < config.max_new_tokens:
        remaining = config.max_new_tokens - len(accepted_large_ids)
        draft_iterations += 1

        draft_prompt_ids = draft_tokenizer.encode(draft_prompt_text + accepted_text)
        draft_stats = greedy_generate_core(
            model=draft_model,
            tokenizer=draft_tokenizer,
            prompt_ids=draft_prompt_ids,
            max_new_tokens=min(config.draft_chunk_tokens, remaining),
            prefill_step_size=config.prefill_step_size,
        )
        if not draft_stats["generated_ids"]:
            break

        draft_chunk_text = draft_stats["output_text"]
        proposed_large_ids = verifier_tokenizer.encode(draft_chunk_text)[:remaining]
        total_proposed_large_tokens += len(proposed_large_ids)

        verifier_prompt_ids = verifier_tokenizer.encode(verifier_prompt_text + accepted_text)
        verifier_iterations += 1
        verifier_budget = max(len(proposed_large_ids), 1)
        verifier_stats = greedy_generate_core(
            model=verifier_model,
            tokenizer=verifier_tokenizer,
            prompt_ids=verifier_prompt_ids,
            max_new_tokens=verifier_budget,
            prefill_step_size=config.prefill_step_size,
        )
        verifier_ids = verifier_stats["generated_ids"]
        if not verifier_ids:
            break

        matched = prefix_match_length(proposed_large_ids, verifier_ids)
        if matched:
            accepted_slice = verifier_ids[:matched]
            if len(accepted_large_ids) + len(accepted_slice) > config.max_new_tokens:
                accepted_slice = accepted_slice[: config.max_new_tokens - len(accepted_large_ids)]
            accepted_large_ids.extend(accepted_slice)
            accepted_by_draft_alignment += len(accepted_slice)
            if ttft_s is None and accepted_slice:
                ttft_s = time.perf_counter() - t0
            if accepted_slice and accepted_slice[-1] in verifier_eos_ids:
                break

        # Correct at first divergence, or if draft produced no verifier-space tokens.
        if matched < len(verifier_ids) and len(accepted_large_ids) < config.max_new_tokens:
            correction_token = verifier_ids[matched]
            accepted_large_ids.append(correction_token)
            correction_tokens += 1
            if ttft_s is None:
                ttft_s = time.perf_counter() - t0
            if correction_token in verifier_eos_ids:
                break

        accepted_text = verifier_tokenizer.decode(
            accepted_large_ids,
            skip_special_tokens=True,
        )
        if accepted_large_ids and accepted_large_ids[-1] in verifier_eos_ids:
            break

    elapsed_s = time.perf_counter() - t0
    ttft_s = ttft_s if ttft_s is not None else elapsed_s
    runtime_peak_gb = mx.get_peak_memory() / (1024**3)
    active_memory_gb = mx.get_active_memory() / (1024**3)
    cache_memory_gb = mx.get_cache_memory() / (1024**3)

    return {
        "generated_ids": accepted_large_ids,
        "output_text": verifier_tokenizer.decode(
            accepted_large_ids,
            skip_special_tokens=True,
        ),
        "ttft_s": ttft_s,
        "elapsed_s": elapsed_s,
        "generation_tps": max(len(accepted_large_ids) - 1, 0) / max(elapsed_s - ttft_s, 1e-6),
        "prompt_tokens": len(verifier_tokenizer.encode(verifier_prompt_text)),
        "tokens_generated": len(accepted_large_ids),
        "runtime_peak_memory_gb": runtime_peak_gb,
        "active_memory_gb": active_memory_gb,
        "cache_memory_gb": cache_memory_gb,
        "peak_memory_gb": max(runtime_peak_gb, active_memory_gb),
        "stage_metrics": {
            "draft_iterations": draft_iterations,
            "verifier_iterations": verifier_iterations,
            "accepted_by_draft_alignment_tokens": accepted_by_draft_alignment,
            "correction_tokens": correction_tokens,
            "total_proposed_large_tokens": total_proposed_large_tokens,
            "acceptance_rate": (
                accepted_by_draft_alignment / max(total_proposed_large_tokens, 1)
            ),
        },
        "system_complexity_note": (
            "dual_model_naive_chunked_verify_no_kv_sharing_cross_tokenizer_retokenization"
        ),
    }


def load_large_model_for_track_b(model_path: str):
    return load_resolved_mlx_model(model_path, lazy=True)


def load_draft_model_for_track_b(model_path: str):
    model, tokenizer = mlx_load(model_path, lazy=True)
    return model, tokenizer
