"""
Minimal comparison benchmark between:
  - gpt-oss-20b full-depth baseline
  - current best same-model adaptive-depth Transcender path
  - small-model draft baseline (if available)
  - simple serial draft-then-verify cascade (if available)

This is intentionally minimal. It is not a production speculative decoder.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from transcender_engine import (
    GptOssConfig,
    MLXDynamicExpertEngine,
    apply_harmony_template,
    build_harmony_messages,
    load_resolved_mlx_model,
)


DEFAULT_LARGE_MODEL_PATH = str(
    (Path(__file__).resolve().parent.parent / "gpt-oss-20b-raw").resolve()
)
DEFAULT_OUTPUT_PATH = str(
    (Path(__file__).resolve().parent / "transcender_cascade_benchmark.json").resolve()
)
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_VERIFY_TOKENS = 8
SYSTEM_PROMPT = "You are a helpful assistant."
REASONING_EFFORT = "medium"
WARMUP_PROMPT_INDEX = 0

PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "Summarize why the French Revolution was historically important.",
    "Write a short explanation of recursion for a beginner programmer.",
    "Explain the difference between TCP and UDP in plain English.",
    "Describe what photosynthesis does.",
]

DRAFT_MODEL_CANDIDATE_SUBSTRINGS = [
    "gemma-3-text-4b-it",
    "gemma-3-4b-it",
    "gemma-3-4b",
    "meta-llama-3.1-8b-instruct",
    "llama-3.1-8b-instruct",
]


@dataclass(frozen=True)
class ModeSpec:
    key: str
    label: str
    description: str
    mode_type: str


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


def compare_sequences(left: List[int], right: List[int]) -> Dict[str, Any]:
    prefix = 0
    for left_tok, right_tok in zip(left, right):
        if left_tok != right_tok:
            break
        prefix += 1

    paired = list(zip(left, right))
    exact_match_rate = (
        sum(1 for left_tok, right_tok in paired if left_tok == right_tok)
        / max(len(paired), 1)
    )
    return {
        "prefix_match_tokens": prefix,
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


def official_greedy_generate(
    model,
    tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    prefill_step_size: int,
) -> Dict[str, Any]:
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    sampler = make_sampler(temp=0.0)
    eos_ids = set(get_eos_ids(tokenizer))

    mx.clear_cache()
    mx.reset_peak_memory()
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
    generation_window = max(elapsed_s - ttft_s, 1e-6)
    return {
        "generated_ids": generated_ids,
        "output_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "ttft_s": ttft_s,
        "elapsed_s": elapsed_s,
        "generation_tps": max(len(generated_ids) - 1, 0) / generation_window,
        "peak_memory_gb": mx.get_peak_memory() / (1024**3),
        "prompt_tokens": len(prompt_ids),
        "tokens_generated": len(generated_ids),
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

    searched = []
    roots = [
        Path.home() / "Documents" / "GitHub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in root.iterdir():
            name = path.name.lower()
            searched.append(str(path))
            for needle in DRAFT_MODEL_CANDIDATE_SUBSTRINGS:
                if needle in name:
                    if path.is_dir() and path.name.startswith("models--"):
                        snapshot = latest_snapshot_for_model(path)
                        if snapshot is not None:
                            return str(snapshot), {
                                "status": "found_cache_snapshot",
                                "path": str(snapshot),
                                "repo_dir": str(path),
                            }
                    elif path.is_dir():
                        return str(path), {
                            "status": "found_local_dir",
                            "path": str(path),
                        }
    return None, {
        "status": "not_found",
        "searched_roots": [str(root) for root in roots],
        "candidate_substrings": DRAFT_MODEL_CANDIDATE_SUBSTRINGS,
    }


def aggregate_prompt_results(prompt_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    divergences = [
        float(item["comparison"]["first_divergence_position"])
        for item in prompt_results
        if item["comparison"]["first_divergence_position"] is not None
    ]
    return {
        "avg_ttft_s": mean([item["ttft_s"] for item in prompt_results]),
        "avg_generation_tps": mean([item["generation_tps"] for item in prompt_results]),
        "avg_elapsed_s": mean([item["elapsed_s"] for item in prompt_results]),
        "avg_peak_memory_gb": mean([item["peak_memory_gb"] for item in prompt_results]),
        "avg_exact_match_rate": mean([item["comparison"]["exact_match_rate"] for item in prompt_results]),
        "avg_prefix_match_tokens": mean([item["comparison"]["prefix_match_tokens"] for item in prompt_results]),
        "avg_first_divergence_position": mean(divergences),
        "avg_prompt_tokens": mean([item["prompt_tokens"] for item in prompt_results]),
        "avg_completion_tokens": mean([item["completion_tokens"] for item in prompt_results]),
    }


def print_summary(payload: Dict[str, Any], output_path: str):
    print("\nTranscender Cascade Comparison")
    print(f"JSON output: {output_path}")
    print(
        f"Warmup-corrected aggregate excludes prompt "
        f"{payload['prompts'][WARMUP_PROMPT_INDEX]['prompt_id']}"
    )
    print("\nAggregate Summary")
    header = (
        f"{'mode':<28} {'status':<12} {'ttft':>8} {'elapsed':>8} "
        f"{'gen_tps':>10} {'exact':>8} {'prefix':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in payload["modes"]:
        if item["status"] != "ok":
            print(
                f"{item['label']:<28} {item['status']:<12} {'-':>8} {'-':>8} "
                f"{'-':>10} {'-':>8} {'-':>8}"
            )
            continue
        agg = item["aggregate_excluding_warmup"]
        print(
            f"{item['label']:<28} {item['status']:<12} "
            f"{agg['avg_ttft_s']:>8.3f} "
            f"{agg['avg_elapsed_s']:>8.3f} "
            f"{agg['avg_generation_tps']:>10.2f} "
            f"{agg['avg_exact_match_rate']:>8.3f} "
            f"{agg['avg_prefix_match_tokens']:>8.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large-model", default=DEFAULT_LARGE_MODEL_PATH)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--verify-tokens", type=int, default=DEFAULT_VERIFY_TOKENS)
    parser.add_argument("--memory-limit-gb", type=float, default=30.0)
    args = parser.parse_args()

    large_model, large_tokenizer, resolved_large_model_path = load_resolved_mlx_model(
        args.large_model,
        lazy=True,
    )
    messages_by_prompt = []
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
        messages_by_prompt.append(
            {
                "prompt_id": prompt_id,
                "system": SYSTEM_PROMPT,
                "user": user_prompt,
                "messages": messages,
                "large_prompt_text": large_prompt_text,
                "large_prompt_ids": large_tokenizer.encode(large_prompt_text),
            }
        )

    transcender_engine = MLXDynamicExpertEngine(
        model=large_model,
        tokenizer=large_tokenizer,
        config=GptOssConfig(
            hard_exit_layer=22,
            entropy_threshold=-1.0,
            enable_logit_blending=True,
            blending_confidence_threshold=0.035,
            blend_alpha=0.10,
            confidence_signal="entropy",
            blend_alpha_mode="fixed",
            blend_strategy="top1_agree",
            memory_limit_gb=args.memory_limit_gb,
        ),
        enable_logit_blending=True,
        blending_confidence_threshold=0.035,
        blend_alpha=0.10,
        confidence_signal="entropy",
        blend_alpha_mode="fixed",
        blend_strategy="top1_agree",
        memory_limit_gb=args.memory_limit_gb,
    )

    draft_model_path, draft_detection = auto_detect_draft_model(args.draft_model)
    draft_model = None
    draft_tokenizer = None
    draft_prompt_template = None
    if draft_model_path is not None:
        draft_model, draft_tokenizer = mlx_load(draft_model_path, lazy=True)

    mode_specs = [
        ModeSpec(
            key="gpt_oss_full_depth",
            label="GPT-OSS Full-Depth",
            description="Corrected official GPT-OSS greedy baseline.",
            mode_type="large_baseline",
        ),
        ModeSpec(
            key="gpt_oss_transcender",
            label="GPT-OSS Transcender",
            description="Current best same-model adaptive-depth config.",
            mode_type="same_model_adaptive",
        ),
        ModeSpec(
            key="draft_model_only",
            label="Draft Model Only",
            description="Small-model draft baseline.",
            mode_type="draft_only",
        ),
        ModeSpec(
            key="draft_verify_cascade",
            label="Draft -> Verify Cascade",
            description="Simple serial draft then large-prefix verify, with fallback to full GPT-OSS.",
            mode_type="cascade",
        ),
    ]

    prompt_results_by_mode: Dict[str, List[Dict[str, Any]]] = {
        spec.key: [] for spec in mode_specs
    }
    mode_status: Dict[str, Dict[str, Any]] = {
        "gpt_oss_full_depth": {"status": "ok"},
        "gpt_oss_transcender": {"status": "ok"},
        "draft_model_only": {"status": "ok" if draft_model is not None else "skipped_missing_model"},
        "draft_verify_cascade": {"status": "ok" if draft_model is not None else "skipped_missing_model"},
    }

    if draft_model is not None:
        for prompt_def in messages_by_prompt:
            draft_prompt_text, template_mode = apply_generic_chat_template(
                draft_tokenizer,
                prompt_def["messages"],
            )
            prompt_def["draft_prompt_text"] = draft_prompt_text
            prompt_def["draft_prompt_ids"] = draft_tokenizer.encode(draft_prompt_text)
            prompt_def["draft_prompt_template"] = template_mode

    for prompt_def in messages_by_prompt:
        baseline_stats = official_greedy_generate(
            model=large_model,
            tokenizer=large_tokenizer,
            prompt_ids=prompt_def["large_prompt_ids"],
            max_new_tokens=args.max_new_tokens,
            prefill_step_size=2048,
        )
        baseline_entry = {
            "prompt_id": prompt_def["prompt_id"],
            "user_prompt": prompt_def["user"],
            "prompt_tokens": baseline_stats["prompt_tokens"],
            "completion_tokens": baseline_stats["tokens_generated"],
            "ttft_s": baseline_stats["ttft_s"],
            "generation_tps": baseline_stats["generation_tps"],
            "elapsed_s": baseline_stats["elapsed_s"],
            "peak_memory_gb": baseline_stats["peak_memory_gb"],
            "output_text": baseline_stats["output_text"],
            "text_preview": preview_text(baseline_stats["output_text"]),
            "generated_ids": baseline_stats["generated_ids"],
            "comparison": {
                "prefix_match_tokens": len(baseline_stats["generated_ids"]),
                "exact_match_rate": 1.0,
                "first_divergence_position": None,
                "passed": True,
            },
        }
        prompt_results_by_mode["gpt_oss_full_depth"].append(baseline_entry)

        transcender_stats = transcender_engine.generate_from_ids(
            prompt_def["large_prompt_ids"],
            max_new_tokens=args.max_new_tokens,
            dynamic=True,
            profile_runtime=False,
        )
        transcender_entry = {
            "prompt_id": prompt_def["prompt_id"],
            "user_prompt": prompt_def["user"],
            "prompt_tokens": transcender_stats["prompt_tokens"],
            "completion_tokens": transcender_stats["tokens_generated"],
            "ttft_s": transcender_stats["ttft_s"],
            "generation_tps": transcender_stats["generation_tps"],
            "elapsed_s": transcender_stats["elapsed_s"],
            "peak_memory_gb": transcender_stats["peak_memory_gb"],
            "output_text": transcender_stats["output_text"],
            "text_preview": preview_text(transcender_stats["output_text"]),
            "generated_ids": transcender_stats["generated_ids"],
            "comparison": compare_sequences(
                transcender_stats["generated_ids"],
                baseline_stats["generated_ids"],
            ),
            "avg_layers": transcender_stats.get("avg_layers", 24.0),
            "avg_layers_saved": max(24.0 - transcender_stats.get("avg_layers", 24.0), 0.0),
        }
        prompt_results_by_mode["gpt_oss_transcender"].append(transcender_entry)

        if draft_model is None:
            continue

        draft_stats = official_greedy_generate(
            model=draft_model,
            tokenizer=draft_tokenizer,
            prompt_ids=prompt_def["draft_prompt_ids"],
            max_new_tokens=args.max_new_tokens,
            prefill_step_size=2048,
        )
        draft_comparison = compare_text_against_reference_ids(
            draft_stats["output_text"],
            baseline_stats["generated_ids"],
            large_tokenizer,
        )
        draft_entry = {
            "prompt_id": prompt_def["prompt_id"],
            "user_prompt": prompt_def["user"],
            "prompt_tokens": draft_stats["prompt_tokens"],
            "completion_tokens": draft_stats["tokens_generated"],
            "ttft_s": draft_stats["ttft_s"],
            "generation_tps": draft_stats["generation_tps"],
            "elapsed_s": draft_stats["elapsed_s"],
            "peak_memory_gb": draft_stats["peak_memory_gb"],
            "output_text": draft_stats["output_text"],
            "text_preview": preview_text(draft_stats["output_text"]),
            "generated_ids": draft_stats["generated_ids"][:64],
            "comparison": draft_comparison,
            "prompt_template_mode": prompt_def["draft_prompt_template"],
        }
        prompt_results_by_mode["draft_model_only"].append(draft_entry)

        verifier_stats = official_greedy_generate(
            model=large_model,
            tokenizer=large_tokenizer,
            prompt_ids=prompt_def["large_prompt_ids"],
            max_new_tokens=args.verify_tokens,
            prefill_step_size=2048,
        )
        draft_in_large_ids = large_tokenizer.encode(draft_stats["output_text"])
        verifier_comparison = compare_sequences(
            draft_in_large_ids[: args.verify_tokens],
            verifier_stats["generated_ids"],
        )
        accepted = (
            verifier_comparison["passed"]
            and len(verifier_stats["generated_ids"]) >= min(args.verify_tokens, len(draft_in_large_ids))
        )

        if accepted:
            cascade_output_text = draft_stats["output_text"]
            cascade_comparison = draft_comparison
            cascade_completion_tokens = len(draft_in_large_ids)
            cascade_ttft = draft_stats["ttft_s"]
            cascade_generation_tps = draft_stats["generation_tps"]
            cascade_peak_memory = max(
                draft_stats["peak_memory_gb"],
                verifier_stats["peak_memory_gb"],
            )
            cascade_elapsed = draft_stats["elapsed_s"] + verifier_stats["elapsed_s"]
            cascade_source = "draft_accepted"
        else:
            cascade_output_text = baseline_stats["output_text"]
            cascade_comparison = {
                "prefix_match_tokens": len(baseline_stats["generated_ids"]),
                "exact_match_rate": 1.0,
                "first_divergence_position": None,
                "passed": True,
                "comparison_space": "gpt_oss_reference_token_space",
            }
            cascade_completion_tokens = len(baseline_stats["generated_ids"])
            cascade_ttft = draft_stats["elapsed_s"] + verifier_stats["elapsed_s"] + baseline_stats["ttft_s"]
            cascade_generation_tps = baseline_stats["generation_tps"]
            cascade_peak_memory = max(
                draft_stats["peak_memory_gb"],
                verifier_stats["peak_memory_gb"],
                baseline_stats["peak_memory_gb"],
            )
            cascade_elapsed = draft_stats["elapsed_s"] + verifier_stats["elapsed_s"] + baseline_stats["elapsed_s"]
            cascade_source = "fallback_large_regen"

        cascade_entry = {
            "prompt_id": prompt_def["prompt_id"],
            "user_prompt": prompt_def["user"],
            "prompt_tokens": (
                draft_stats["prompt_tokens"]
                + verifier_stats["prompt_tokens"]
                + (baseline_stats["prompt_tokens"] if not accepted else 0)
            ),
            "completion_tokens": cascade_completion_tokens,
            "ttft_s": cascade_ttft,
            "generation_tps": cascade_generation_tps,
            "elapsed_s": cascade_elapsed,
            "peak_memory_gb": cascade_peak_memory,
            "output_text": cascade_output_text,
            "text_preview": preview_text(cascade_output_text),
            "comparison": cascade_comparison,
            "accepted_by_verifier": accepted,
            "verifier_prefix_match_tokens": verifier_comparison["prefix_match_tokens"],
            "verifier_first_divergence_position": verifier_comparison["first_divergence_position"],
            "cascade_source": cascade_source,
            "stage_metrics": {
                "draft_elapsed_s": draft_stats["elapsed_s"],
                "verifier_elapsed_s": verifier_stats["elapsed_s"],
                "fallback_large_elapsed_s": 0.0 if accepted else baseline_stats["elapsed_s"],
            },
            "prompt_template_mode": {
                "draft": prompt_def["draft_prompt_template"],
                "large": "harmony",
            },
        }
        prompt_results_by_mode["draft_verify_cascade"].append(cascade_entry)

    mode_results = []
    for spec in mode_specs:
        prompt_results = prompt_results_by_mode[spec.key]
        status = mode_status[spec.key]["status"]
        payload: Dict[str, Any] = {
            "key": spec.key,
            "label": spec.label,
            "description": spec.description,
            "status": status,
            "prompt_results": prompt_results,
        }
        if status == "ok":
            included = [
                item for index, item in enumerate(prompt_results) if index != WARMUP_PROMPT_INDEX
            ]
            payload["aggregate_including_warmup"] = aggregate_prompt_results(prompt_results)
            payload["aggregate_excluding_warmup"] = aggregate_prompt_results(included)
        else:
            payload["aggregate_including_warmup"] = None
            payload["aggregate_excluding_warmup"] = None
            payload["reason"] = "No local draft model available."
        mode_results.append(payload)

    available_modes = [item for item in mode_results if item["status"] == "ok"]
    comparison_summary = {
        "large_baseline_mode": "GPT-OSS Full-Depth",
        "same_model_adaptive_mode": "GPT-OSS Transcender",
        "draft_model_detected": draft_model_path is not None,
        "draft_model_detection": draft_detection,
    }
    if draft_model_path is not None:
        comparison_summary["draft_model_path"] = draft_model_path
    payload = {
        "experiment": "transcender_cascade_benchmark",
        "resolved_large_model_path": resolved_large_model_path,
        "draft_model_detection": draft_detection,
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "verify_tokens": args.verify_tokens,
            "reasoning_effort_large_model": REASONING_EFFORT,
            "warmup_discard_prompt_id": messages_by_prompt[WARMUP_PROMPT_INDEX]["prompt_id"],
            "notes": (
                "GPT-OSS uses Harmony. Auxiliary models use their own tokenizer chat template path. "
                "Cross-model exact-match is computed in GPT-OSS token space by re-tokenizing decoded text."
            ),
        },
        "prompts": [
            {
                "prompt_id": prompt_def["prompt_id"],
                "system": prompt_def["system"],
                "user": prompt_def["user"],
            }
            for prompt_def in messages_by_prompt
        ],
        "modes": mode_results,
        "comparison_summary": comparison_summary,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print_summary(payload, args.output)


if __name__ == "__main__":
    main()
