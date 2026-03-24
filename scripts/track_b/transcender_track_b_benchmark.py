"""
Track B benchmark runner.

Compares four modes on the same selected prompt suite:
  1. Draft model alone
  2. GPT-OSS 20B full-depth baseline
  3. Naive Track B cascade (draft + verifier)
  4. Track A frontier (L22 top1_agree)

Default behavior preserves the original five-prompt expository subset. The CLI
also supports canonical Track A prompt reuse and prompt-id filtering for
matched-scope reruns.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

TRACK_A_DIR = Path(__file__).resolve().parents[1] / "track_a"
if str(TRACK_A_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_A_DIR))
from transcender_track_b_cascade import (
    DEFAULT_LARGE_MODEL_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PROMPT_SUITE,
    CascadeConfig,
    REASONING_EFFORT,
    SYSTEM_PROMPT,
    WARMUP_PROMPT_INDEX,
    auto_detect_draft_model,
    build_prompt_pack,
    compare_sequences,
    compare_text_against_reference_ids,
    enrich_prompt_pack_for_draft,
    load_draft_model_for_track_b,
    load_large_model_for_track_b,
    mean,
    naive_cascade_generate,
    official_greedy_generate,
    preview_text,
    release_models,
    resolve_prompt_definitions,
)


@dataclass(frozen=True)
class ModeSpec:
    key: str
    label: str
    description: str
    system_complexity_note: str


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
        "avg_runtime_peak_memory_gb": mean(
            [item.get("runtime_peak_memory_gb", item["peak_memory_gb"]) for item in prompt_results]
        ),
        "avg_active_memory_gb": mean(
            [item.get("active_memory_gb", item["peak_memory_gb"]) for item in prompt_results]
        ),
        "avg_cache_memory_gb": mean([item.get("cache_memory_gb", 0.0) for item in prompt_results]),
        "avg_exact_match_rate": mean([item["comparison"]["exact_match_rate"] for item in prompt_results]),
        "avg_prefix_match_tokens": mean([item["comparison"]["prefix_match_tokens"] for item in prompt_results]),
        "avg_first_divergence_position": mean(divergences),
        "avg_prompt_tokens": mean([item["prompt_tokens"] for item in prompt_results]),
        "avg_completion_tokens": mean([item["completion_tokens"] for item in prompt_results]),
    }


def build_mode_specs() -> List[ModeSpec]:
    return [
        ModeSpec(
            key="draft_model_only",
            label="Draft Model Only",
            description="Small dense draft-model baseline.",
            system_complexity_note="single_model_greedy_draft_template_path",
        ),
        ModeSpec(
            key="gpt_oss_full_depth",
            label="GPT-OSS Full-Depth",
            description="Corrected official GPT-OSS greedy baseline.",
            system_complexity_note="single_model_greedy_harmony_reference",
        ),
        ModeSpec(
            key="track_b_naive_cascade",
            label="Track B Naive Cascade",
            description="Chunked draft-then-verify with first-divergence correction.",
            system_complexity_note="dual_model_chunked_verify_first_divergence_no_kv_sharing",
        ),
        ModeSpec(
            key="track_a_l22_top1_agree",
            label="Track A L22 top1_agree",
            description="Canonical same-model adaptive-depth frontier.",
            system_complexity_note="single_model_adaptive_depth_layer22_top1_agree",
        ),
    ]


def print_summary(payload: Dict[str, Any], output_path: str):
    print("\nTranscender Track B Comparison")
    print(f"JSON output: {output_path}")
    prompt_scope = payload.get("prompt_scope", {})
    print(
        f"Prompt suite: {prompt_scope.get('prompt_suite', 'unknown')} | "
        f"selected prompts: {prompt_scope.get('selected_prompt_count', 0)} | "
        f"scored after warmup: {prompt_scope.get('scored_prompt_count_after_warmup', 0)}"
    )
    print(
        f"Warmup-corrected aggregate excludes prompt "
        f"{payload['prompts'][WARMUP_PROMPT_INDEX]['prompt_id']}"
    )
    print("\nAggregate Summary")
    header = (
        f"{'mode':<28} {'status':<12} {'ttft':>8} {'gen_tps':>10} "
        f"{'peak_gb':>9} {'exact':>8} {'prefix':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in payload["modes"]:
        if item["status"] != "ok":
            print(
                f"{item['label']:<28} {item['status']:<12} "
                f"{'-':>8} {'-':>10} {'-':>9} {'-':>8} {'-':>8}"
            )
            continue
        agg = item["aggregate_excluding_warmup"]
        print(
            f"{item['label']:<28} {item['status']:<12} "
            f"{agg['avg_ttft_s']:>8.3f} "
            f"{agg['avg_generation_tps']:>10.2f} "
            f"{agg['avg_peak_memory_gb']:>9.2f} "
            f"{agg['avg_exact_match_rate']:>8.3f} "
            f"{agg['avg_prefix_match_tokens']:>8.1f}"
        )
    print("\nRecommendation")
    summary = payload["comparison_summary"]
    print(summary["recommendation"])
    print(
        f"Track B faster on generation TPS: {summary['track_b_faster_than_track_a_on_generation_tps']}"
    )
    print(f"Track B TTFT penalty vs Track A: {summary['track_b_ttft_delta_s_vs_track_a']:.3f}s")
    print(
        f"Track B memory delta vs Track A: {summary['track_b_peak_memory_delta_gb_vs_track_a']:.2f} GB"
    )


def run_large_baseline(
    large_model_path: str,
    max_new_tokens: int,
    prompt_definitions: List[Dict[str, str]],
) -> Dict[str, Any]:
    large_model, large_tokenizer, resolved_large_model_path = load_large_model_for_track_b(
        large_model_path
    )
    prompt_pack = build_prompt_pack(large_tokenizer, prompt_definitions)
    prompt_results: List[Dict[str, Any]] = []
    for prompt_def in prompt_pack:
        stats = official_greedy_generate(
            model=large_model,
            tokenizer=large_tokenizer,
            prompt_ids=prompt_def["large_prompt_ids"],
            max_new_tokens=max_new_tokens,
            prefill_step_size=2048,
        )
        prompt_results.append(
            {
                "prompt_id": prompt_def["prompt_id"],
                "user_prompt": prompt_def["user"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["tokens_generated"],
                "ttft_s": stats["ttft_s"],
                "generation_tps": stats["generation_tps"],
                "elapsed_s": stats["elapsed_s"],
                "peak_memory_gb": stats["peak_memory_gb"],
                "runtime_peak_memory_gb": stats["runtime_peak_memory_gb"],
                "active_memory_gb": stats["active_memory_gb"],
                "cache_memory_gb": stats["cache_memory_gb"],
                "output_text": stats["output_text"],
                "text_preview": preview_text(stats["output_text"]),
                "generated_ids": stats["generated_ids"],
                "comparison": {
                    "prefix_match_tokens": len(stats["generated_ids"]),
                    "exact_match_rate": 1.0,
                    "first_divergence_position": None,
                    "passed": True,
                    "comparison_space": "gpt_oss_verifier_token_space",
                },
            }
        )
    release_models(large_model)
    return {
        "resolved_large_model_path": resolved_large_model_path,
        "reference_tokenizer": large_tokenizer,
        "prompt_pack": prompt_pack,
        "prompt_results": prompt_results,
    }


def run_track_a_mode(
    large_model_path: str,
    max_new_tokens: int,
    baseline_results_by_prompt: Dict[str, Dict[str, Any]],
    prompt_pack_seed: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from transcender_engine import GptOssConfig, MLXDynamicExpertEngine

    large_model, large_tokenizer, resolved_large_model_path = load_large_model_for_track_b(
        large_model_path
    )
    engine = MLXDynamicExpertEngine(
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
        ),
        enable_logit_blending=True,
        blending_confidence_threshold=0.035,
        blend_alpha=0.10,
        confidence_signal="entropy",
        blend_alpha_mode="fixed",
        blend_strategy="top1_agree",
    )

    prompt_results: List[Dict[str, Any]] = []
    for prompt_def in prompt_pack_seed:
        stats = engine.generate_from_ids(
            prompt_def["large_prompt_ids"],
            max_new_tokens=max_new_tokens,
            dynamic=True,
            profile_runtime=False,
        )
        baseline = baseline_results_by_prompt[prompt_def["prompt_id"]]
        prompt_results.append(
            {
                "prompt_id": prompt_def["prompt_id"],
                "user_prompt": prompt_def["user"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["tokens_generated"],
                "ttft_s": stats["ttft_s"],
                "generation_tps": stats["generation_tps"],
                "elapsed_s": stats["elapsed_s"],
                "peak_memory_gb": stats["peak_memory_gb"],
                "runtime_peak_memory_gb": stats.get("peak_memory_gb", 0.0),
                "active_memory_gb": stats.get("active_memory_gb", stats["peak_memory_gb"]),
                "cache_memory_gb": stats.get("cache_memory_gb", 0.0),
                "output_text": stats["output_text"],
                "text_preview": preview_text(stats["output_text"]),
                "generated_ids": stats["generated_ids"],
                "comparison": compare_sequences(
                    stats["generated_ids"],
                    baseline["generated_ids"],
                ),
                "avg_layers": stats.get("avg_layers", 24.0),
                "avg_layers_saved": max(24.0 - stats.get("avg_layers", 24.0), 0.0),
            }
        )
        prompt_results[-1]["comparison"]["comparison_space"] = "gpt_oss_verifier_token_space"
    release_models(engine, large_model, large_tokenizer)
    return {
        "resolved_large_model_path": resolved_large_model_path,
        "prompt_results": prompt_results,
    }


def run_draft_mode(
    draft_model_path: str,
    prompt_pack_seed: List[Dict[str, Any]],
    max_new_tokens: int,
    baseline_results_by_prompt: Dict[str, Dict[str, Any]],
    baseline_tokenizer,
) -> Dict[str, Any]:
    draft_model, draft_tokenizer = load_draft_model_for_track_b(draft_model_path)
    prompt_pack = [
        {
            "prompt_id": item["prompt_id"],
            "system": item["system"],
            "user": item["user"],
            "messages": item["messages"],
        }
        for item in prompt_pack_seed
    ]
    enrich_prompt_pack_for_draft(prompt_pack, draft_tokenizer)

    prompt_results: List[Dict[str, Any]] = []
    for prompt_def in prompt_pack:
        stats = official_greedy_generate(
            model=draft_model,
            tokenizer=draft_tokenizer,
            prompt_ids=prompt_def["draft_prompt_ids"],
            max_new_tokens=max_new_tokens,
            prefill_step_size=2048,
        )
        baseline = baseline_results_by_prompt[prompt_def["prompt_id"]]
        prompt_results.append(
            {
                "prompt_id": prompt_def["prompt_id"],
                "user_prompt": prompt_def["user"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["tokens_generated"],
                "ttft_s": stats["ttft_s"],
                "generation_tps": stats["generation_tps"],
                "elapsed_s": stats["elapsed_s"],
                "peak_memory_gb": stats["peak_memory_gb"],
                "runtime_peak_memory_gb": stats["runtime_peak_memory_gb"],
                "active_memory_gb": stats["active_memory_gb"],
                "cache_memory_gb": stats["cache_memory_gb"],
                "output_text": stats["output_text"],
                "text_preview": preview_text(stats["output_text"]),
                "comparison": compare_text_against_reference_ids(
                    stats["output_text"],
                    baseline["generated_ids"],
                    baseline_tokenizer,
                ),
                "prompt_template_mode": prompt_def["draft_prompt_template"],
            }
        )
    release_models(draft_model, draft_tokenizer)
    return {"prompt_results": prompt_results}


def run_track_b_mode(
    draft_model_path: str,
    large_model_path: str,
    prompt_pack_seed: List[Dict[str, Any]],
    max_new_tokens: int,
    draft_chunk_tokens: int,
    baseline_results_by_prompt: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    large_model, large_tokenizer, resolved_large_model_path = load_large_model_for_track_b(
        large_model_path
    )
    draft_model, draft_tokenizer = load_draft_model_for_track_b(draft_model_path)
    prompt_pack = [
        {
            "prompt_id": item["prompt_id"],
            "system": item["system"],
            "user": item["user"],
            "messages": item["messages"],
            "large_prompt_text": item["large_prompt_text"],
            "large_prompt_ids": item["large_prompt_ids"],
        }
        for item in prompt_pack_seed
    ]
    enrich_prompt_pack_for_draft(prompt_pack, draft_tokenizer)

    config = CascadeConfig(
        draft_chunk_tokens=draft_chunk_tokens,
        max_new_tokens=max_new_tokens,
        prefill_step_size=2048,
    )
    prompt_results: List[Dict[str, Any]] = []
    for prompt_def in prompt_pack:
        stats = naive_cascade_generate(
            draft_model=draft_model,
            draft_tokenizer=draft_tokenizer,
            verifier_model=large_model,
            verifier_tokenizer=large_tokenizer,
            draft_prompt_text=prompt_def["draft_prompt_text"],
            verifier_prompt_text=prompt_def["large_prompt_text"],
            config=config,
        )
        baseline = baseline_results_by_prompt[prompt_def["prompt_id"]]
        prompt_results.append(
            {
                "prompt_id": prompt_def["prompt_id"],
                "user_prompt": prompt_def["user"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["tokens_generated"],
                "ttft_s": stats["ttft_s"],
                "generation_tps": stats["generation_tps"],
                "elapsed_s": stats["elapsed_s"],
                "peak_memory_gb": stats["peak_memory_gb"],
                "runtime_peak_memory_gb": stats["runtime_peak_memory_gb"],
                "active_memory_gb": stats["active_memory_gb"],
                "cache_memory_gb": stats["cache_memory_gb"],
                "output_text": stats["output_text"],
                "text_preview": preview_text(stats["output_text"]),
                "generated_ids": stats["generated_ids"],
                "comparison": compare_sequences(
                    stats["generated_ids"],
                    baseline["generated_ids"],
                ),
                "stage_metrics": stats["stage_metrics"],
                "prompt_template_mode": {
                    "draft": prompt_def["draft_prompt_template"],
                    "large": "harmony",
                },
            }
        )
        prompt_results[-1]["comparison"]["comparison_space"] = "gpt_oss_verifier_token_space"
    release_models(draft_model, draft_tokenizer, large_model, large_tokenizer)
    return {
        "resolved_large_model_path": resolved_large_model_path,
        "prompt_results": prompt_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large-model", default=DEFAULT_LARGE_MODEL_PATH)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--prompt-suite",
        choices=("expository_5", "canonical_64"),
        default=DEFAULT_PROMPT_SUITE,
        help=(
            "Prompt scope for the benchmark. "
            "'expository_5' preserves legacy behavior. "
            "'canonical_64' reuses the canonical Track A prompt definitions from the checked-in artifact."
        ),
    )
    parser.add_argument(
        "--prompt-ids",
        default=None,
        help=(
            "Optional comma-separated prompt ids to filter within the selected suite, "
            "for example 'P1,P2,P3,P4,P5'. Warmup still excludes the first selected prompt."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-chunk-tokens", type=int, default=8)
    args = parser.parse_args()

    prompt_definitions, prompt_scope = resolve_prompt_definitions(
        prompt_suite=args.prompt_suite,
        prompt_ids_arg=args.prompt_ids,
    )
    draft_model_path, draft_detection = auto_detect_draft_model(args.draft_model)

    # Baseline first so every other mode has the same source of truth.
    baseline_run = run_large_baseline(
        large_model_path=args.large_model,
        max_new_tokens=args.max_new_tokens,
        prompt_definitions=prompt_definitions,
    )
    baseline_prompt_results = baseline_run["prompt_results"]
    baseline_results_by_prompt = {
        item["prompt_id"]: item for item in baseline_prompt_results
    }

    baseline_prompt_pack = baseline_run["prompt_pack"]

    mode_specs = build_mode_specs()
    mode_results: List[Dict[str, Any]] = []

    # Draft-only mode.
    draft_mode_payload: Dict[str, Any] = {
        "key": "draft_model_only",
        "label": "Draft Model Only",
        "description": next(spec.description for spec in mode_specs if spec.key == "draft_model_only"),
        "system_complexity_note": next(
            spec.system_complexity_note for spec in mode_specs if spec.key == "draft_model_only"
        ),
    }
    if draft_model_path is None:
        draft_mode_payload["status"] = "skipped_missing_model"
        draft_mode_payload["reason"] = "No local dense draft model detected."
        draft_mode_payload["prompt_results"] = []
        draft_mode_payload["aggregate_including_warmup"] = None
        draft_mode_payload["aggregate_excluding_warmup"] = None
    else:
        draft_mode = run_draft_mode(
            draft_model_path=draft_model_path,
            prompt_pack_seed=baseline_prompt_pack,
            max_new_tokens=args.max_new_tokens,
            baseline_results_by_prompt=baseline_results_by_prompt,
            baseline_tokenizer=baseline_run["reference_tokenizer"],
        )
        prompt_results = draft_mode["prompt_results"]
        included = [item for index, item in enumerate(prompt_results) if index != WARMUP_PROMPT_INDEX]
        draft_mode_payload.update(
            {
                "status": "ok",
                "prompt_results": prompt_results,
                "aggregate_including_warmup": aggregate_prompt_results(prompt_results),
                "aggregate_excluding_warmup": aggregate_prompt_results(included),
            }
        )
    mode_results.append(draft_mode_payload)

    # Baseline mode payload.
    baseline_included = [
        item for index, item in enumerate(baseline_prompt_results) if index != WARMUP_PROMPT_INDEX
    ]
    mode_results.append(
        {
            "key": "gpt_oss_full_depth",
            "label": "GPT-OSS Full-Depth",
            "description": next(spec.description for spec in mode_specs if spec.key == "gpt_oss_full_depth"),
            "system_complexity_note": next(
                spec.system_complexity_note for spec in mode_specs if spec.key == "gpt_oss_full_depth"
            ),
            "status": "ok",
            "prompt_results": baseline_prompt_results,
            "aggregate_including_warmup": aggregate_prompt_results(baseline_prompt_results),
            "aggregate_excluding_warmup": aggregate_prompt_results(baseline_included),
        }
    )

    # Track B mode.
    track_b_payload: Dict[str, Any] = {
        "key": "track_b_naive_cascade",
        "label": "Track B Naive Cascade",
        "description": next(spec.description for spec in mode_specs if spec.key == "track_b_naive_cascade"),
        "system_complexity_note": next(
            spec.system_complexity_note for spec in mode_specs if spec.key == "track_b_naive_cascade"
        ),
    }
    if draft_model_path is None:
        track_b_payload["status"] = "skipped_missing_model"
        track_b_payload["reason"] = "No local dense draft model detected."
        track_b_payload["prompt_results"] = []
        track_b_payload["aggregate_including_warmup"] = None
        track_b_payload["aggregate_excluding_warmup"] = None
    else:
        track_b_run = run_track_b_mode(
            draft_model_path=draft_model_path,
            large_model_path=args.large_model,
            prompt_pack_seed=baseline_prompt_pack,
            max_new_tokens=args.max_new_tokens,
            draft_chunk_tokens=args.draft_chunk_tokens,
            baseline_results_by_prompt=baseline_results_by_prompt,
        )
        prompt_results = track_b_run["prompt_results"]
        included = [item for index, item in enumerate(prompt_results) if index != WARMUP_PROMPT_INDEX]
        track_b_payload.update(
            {
                "status": "ok",
                "prompt_results": prompt_results,
                "aggregate_including_warmup": aggregate_prompt_results(prompt_results),
                "aggregate_excluding_warmup": aggregate_prompt_results(included),
            }
        )
    mode_results.append(track_b_payload)

    # Track A canonical frontier.
    track_a_run = run_track_a_mode(
        large_model_path=args.large_model,
        max_new_tokens=args.max_new_tokens,
        baseline_results_by_prompt=baseline_results_by_prompt,
        prompt_pack_seed=baseline_prompt_pack,
    )
    track_a_prompt_results = track_a_run["prompt_results"]
    track_a_included = [
        item for index, item in enumerate(track_a_prompt_results) if index != WARMUP_PROMPT_INDEX
    ]
    mode_results.append(
        {
            "key": "track_a_l22_top1_agree",
            "label": "Track A L22 top1_agree",
            "description": next(spec.description for spec in mode_specs if spec.key == "track_a_l22_top1_agree"),
            "system_complexity_note": next(
                spec.system_complexity_note for spec in mode_specs if spec.key == "track_a_l22_top1_agree"
            ),
            "status": "ok",
            "prompt_results": track_a_prompt_results,
            "aggregate_including_warmup": aggregate_prompt_results(track_a_prompt_results),
            "aggregate_excluding_warmup": aggregate_prompt_results(track_a_included),
        }
    )

    modes_by_key = {item["key"]: item for item in mode_results if item["status"] == "ok"}
    track_a_agg = modes_by_key["track_a_l22_top1_agree"]["aggregate_excluding_warmup"]
    baseline_agg = modes_by_key["gpt_oss_full_depth"]["aggregate_excluding_warmup"]
    track_b_agg = modes_by_key.get("track_b_naive_cascade", {}).get("aggregate_excluding_warmup")
    draft_agg = modes_by_key.get("draft_model_only", {}).get("aggregate_excluding_warmup")

    generation_speed_mode = max(
        modes_by_key.values(),
        key=lambda item: item["aggregate_excluding_warmup"]["avg_generation_tps"],
    )["label"]
    lowest_memory_mode = min(
        modes_by_key.values(),
        key=lambda item: item["aggregate_excluding_warmup"]["avg_peak_memory_gb"],
    )["label"]
    best_quality_mode = max(
        modes_by_key.values(),
        key=lambda item: item["aggregate_excluding_warmup"]["avg_exact_match_rate"],
    )["label"]

    if track_b_agg is None:
        recommendation = (
            "Track B could not be benchmarked locally because no dense draft model was detected. "
            "Track A remains the only runnable frontier on this machine right now."
        )
        track_b_faster = False
        track_b_ttft_delta = 0.0
        track_b_memory_delta = 0.0
        track_b_exact_delta = 0.0
    else:
        track_b_faster = track_b_agg["avg_generation_tps"] > track_a_agg["avg_generation_tps"]
        track_b_ttft_delta = track_b_agg["avg_ttft_s"] - track_a_agg["avg_ttft_s"]
        track_b_memory_delta = track_b_agg["avg_peak_memory_gb"] - track_a_agg["avg_peak_memory_gb"]
        track_b_exact_delta = track_b_agg["avg_exact_match_rate"] - track_a_agg["avg_exact_match_rate"]
        if track_b_faster and track_b_memory_delta <= 0.5 and track_b_ttft_delta <= 0.05:
            recommendation = (
                "Track B is operationally competitive on this machine, but it still carries dual-model "
                "orchestration complexity. Keep it as a serious comparison track, not a replacement."
            )
        else:
            recommendation = (
                "Track B does not beat the current Track A frontier cleanly once TTFT, memory, and "
                "orchestration overhead are included. Keep Track A as the main line and Track B as a "
                "comparison baseline."
            )

    payload = {
        "experiment": "transcender_track_b_benchmark",
        "resolved_large_model_path": baseline_run["resolved_large_model_path"],
        "draft_model_detection": draft_detection,
        "draft_model_path": draft_model_path,
        "prompt_scope": prompt_scope,
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "draft_chunk_tokens": args.draft_chunk_tokens,
            "reasoning_effort_large_model": REASONING_EFFORT,
            "warmup_discard_prompt_id": baseline_prompt_pack[WARMUP_PROMPT_INDEX]["prompt_id"],
            "prompt_suite": prompt_scope["prompt_suite"],
            "prompt_scope_label": prompt_scope["scope_label"],
            "selected_prompt_ids": prompt_scope["selected_prompt_ids"],
            "selected_prompt_count": prompt_scope["selected_prompt_count"],
            "scored_prompt_count_after_warmup": prompt_scope["scored_prompt_count_after_warmup"],
            "notes": (
                "GPT-OSS uses Harmony. The auxiliary draft model uses its own tokenizer chat template path. "
                "Track B is a naive chunked verifier loop with no KV sharing."
            ),
        },
        "prompts": [
            {
                "prompt_id": prompt_def["prompt_id"],
                "system": SYSTEM_PROMPT,
                "user": prompt_def["user"],
            }
            for prompt_def in baseline_prompt_pack
        ],
        "modes": mode_results,
        "comparison_summary": {
            "best_quality_mode": best_quality_mode,
            "best_generation_speed_mode": generation_speed_mode,
            "lowest_memory_mode": lowest_memory_mode,
            "track_a_mode": "Track A L22 top1_agree",
            "track_b_mode": "Track B Naive Cascade",
            "track_b_faster_than_track_a_on_generation_tps": track_b_faster,
            "track_b_ttft_delta_s_vs_track_a": track_b_ttft_delta,
            "track_b_peak_memory_delta_gb_vs_track_a": track_b_memory_delta,
            "track_b_exact_match_delta_vs_track_a": track_b_exact_delta,
            "baseline_exact_match_rate": baseline_agg["avg_exact_match_rate"],
            "draft_exact_match_rate": 0.0 if draft_agg is None else draft_agg["avg_exact_match_rate"],
            "recommendation": recommendation,
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print_summary(payload, args.output)
    release_models(baseline_run["reference_tokenizer"])


if __name__ == "__main__":
    main()
