"""
Warmup-corrected quality calibration benchmark for Transcender v0.2.1.

Compares conservative layer-22 blending variants against the official
mlx_lm greedy baseline on the shared Harmony prompt path.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from transcender_engine import (
    GptOssConfig,
    MLXDynamicExpertEngine,
    apply_harmony_template,
    build_harmony_messages,
    load_resolved_mlx_model,
)


DEFAULT_MODEL_PATH = str(
    (Path(__file__).resolve().parent.parent / "gpt-oss-20b-raw").resolve()
)
DEFAULT_OUTPUT_PATH = str(
    (Path(__file__).resolve().parent / "transcender_quality_calibration.json").resolve()
)
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_CONSISTENCY_TOKENS = 16
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


@dataclass(frozen=True)
class PolicySpec:
    key: str
    label: str
    description: str
    config: GptOssConfig


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def preview_text(text: str, limit: int = 180) -> str:
    return text[:limit]


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


def get_eos_ids(tokenizer) -> List[int]:
    eos_ids = list(getattr(tokenizer, "eos_token_ids", []) or [])
    if not eos_ids:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            eos_ids = [int(eos_token_id)]
    return sorted(int(token_id) for token_id in eos_ids)


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
        "cache_memory_gb": mx.get_cache_memory() / (1024**3),
        "prompt_tokens": len(prompt_ids),
        "tokens_generated": len(generated_ids),
    }


def build_engine(
    model,
    tokenizer,
    spec: PolicySpec,
    memory_limit_gb: float,
) -> MLXDynamicExpertEngine:
    config = GptOssConfig(
        soft_skip_start_layer=spec.config.soft_skip_start_layer,
        hard_exit_layer=spec.config.hard_exit_layer,
        entropy_threshold=spec.config.entropy_threshold,
        min_entropy_streak=spec.config.min_entropy_streak,
        enable_logit_blending=spec.config.enable_logit_blending,
        blending_confidence_threshold=spec.config.blending_confidence_threshold,
        blend_alpha=spec.config.blend_alpha,
        confidence_signal=spec.config.confidence_signal,
        margin_threshold=spec.config.margin_threshold,
        blend_alpha_mode=spec.config.blend_alpha_mode,
        blend_alpha_sigmoid_scale=spec.config.blend_alpha_sigmoid_scale,
        blend_entropy_sigmoid_scale=spec.config.blend_entropy_sigmoid_scale,
        fallback_to_full_depth_on_ambiguity=spec.config.fallback_to_full_depth_on_ambiguity,
        memory_limit_gb=memory_limit_gb,
    )
    return MLXDynamicExpertEngine(
        model=model,
        tokenizer=tokenizer,
        config=config,
        enable_logit_blending=config.enable_logit_blending,
        blending_confidence_threshold=config.blending_confidence_threshold,
        blend_alpha=config.blend_alpha,
        confidence_signal=config.confidence_signal,
        margin_threshold=config.margin_threshold,
        blend_alpha_mode=config.blend_alpha_mode,
        blend_alpha_sigmoid_scale=config.blend_alpha_sigmoid_scale,
        blend_entropy_sigmoid_scale=config.blend_entropy_sigmoid_scale,
        fallback_to_full_depth_on_ambiguity=config.fallback_to_full_depth_on_ambiguity,
        memory_limit_gb=memory_limit_gb,
    )


def aggregate_prompt_results(prompt_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    divergence_positions = [
        float(item["comparison"]["first_divergence_position"])
        for item in prompt_results
        if item["comparison"]["first_divergence_position"] is not None
    ]
    return {
        "avg_ttft_s": mean([item["ttft_s"] for item in prompt_results]),
        "avg_generation_tps": mean([item["generation_tps"] for item in prompt_results]),
        "avg_peak_memory_gb": mean([item["peak_memory_gb"] for item in prompt_results]),
        "avg_cache_memory_gb": mean([item.get("cache_memory_gb", 0.0) for item in prompt_results]),
        "avg_exact_match_rate": mean([item["comparison"]["exact_match_rate"] for item in prompt_results]),
        "avg_prefix_match_tokens": mean([item["comparison"]["prefix_match_tokens"] for item in prompt_results]),
        "avg_first_divergence_position": mean(divergence_positions),
        "parity_pass_rate": mean(
            [1.0 if item["comparison"]["passed"] else 0.0 for item in prompt_results]
        ),
        "avg_avg_layers": mean([item.get("avg_layers", 24.0) for item in prompt_results]),
        "avg_avg_layers_saved": mean(
            [item.get("avg_layers_saved", 0.0) for item in prompt_results]
        ),
        "avg_consistency_top1_agreement": mean(
            [item.get("consistency", {}).get("top1_agreement", 0.0) for item in prompt_results]
        ),
        "avg_consistency_kl_divergence": mean(
            [item.get("consistency", {}).get("avg_kl_divergence", 0.0) for item in prompt_results]
        ),
    }


def quality_sort_key(item: Dict[str, Any]):
    agg = item["aggregate_excluding_warmup"]
    return (
        agg["avg_exact_match_rate"],
        agg["avg_prefix_match_tokens"],
        agg["avg_consistency_top1_agreement"],
        -agg["avg_consistency_kl_divergence"],
        agg["avg_generation_tps"],
    )


def print_summary(payload: Dict[str, Any], output_path: str):
    print("\nTranscender v0.2.1 Quality Calibration")
    print(f"JSON output: {output_path}")
    print(
        f"Warmup-corrected aggregate excludes prompt "
        f"{payload['prompts'][WARMUP_PROMPT_INDEX]['prompt_id']}"
    )

    print("\nAggregate Summary")
    header = (
        f"{'config':<22} {'ttft':>8} {'gen_tps':>10} {'exact':>8} "
        f"{'prefix':>8} {'saved':>8} {'top1':>8} {'kl':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in payload["configs"]:
        agg = item["aggregate_excluding_warmup"]
        print(
            f"{item['label']:<22} "
            f"{agg['avg_ttft_s']:>8.3f} "
            f"{agg['avg_generation_tps']:>10.2f} "
            f"{agg['avg_exact_match_rate']:>8.3f} "
            f"{agg['avg_prefix_match_tokens']:>8.1f} "
            f"{agg['avg_avg_layers_saved']:>8.2f} "
            f"{agg['avg_consistency_top1_agreement']:>8.3f} "
            f"{agg['avg_consistency_kl_divergence']:>8.3f}"
        )

    ranking = payload["ranking_summary"]
    print("\nRecommendation")
    print(f"Best quality: {ranking['best_quality_config']}")
    print(f"Best balanced: {ranking['best_balanced_config']}")
    print(f">0.80 exact reached: {ranking['reached_exact_080']}")
    print(f">0.90 exact reached: {ranking['reached_exact_090']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--consistency-tokens", type=int, default=DEFAULT_CONSISTENCY_TOKENS)
    parser.add_argument("--memory-limit-gb", type=float, default=30.0)
    args = parser.parse_args()

    model, tokenizer, resolved_model_path = load_resolved_mlx_model(
        args.model,
        lazy=True,
    )

    prompt_defs = []
    for prompt_index, user_prompt in enumerate(PROMPTS, start=1):
        prompt_id = f"P{prompt_index}"
        rendered_messages = build_harmony_messages(
            user_prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        prompt_text, _ = apply_harmony_template(
            tokenizer,
            messages=rendered_messages,
            reasoning_effort=REASONING_EFFORT,
        )
        prompt_defs.append(
            {
                "prompt_id": prompt_id,
                "system": SYSTEM_PROMPT,
                "user": user_prompt,
                "messages": rendered_messages,
                "prompt_text": prompt_text,
                "prompt_ids": tokenizer.encode(prompt_text),
            }
        )

    policy_specs = [
        PolicySpec(
            key="A_current_v02",
            label="A_current_v02",
            description="Current v0.2 entropy-only policy.",
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                enable_logit_blending=True,
                blending_confidence_threshold=0.05,
                blend_alpha=0.10,
                confidence_signal="entropy",
                blend_alpha_mode="fixed",
            ),
        ),
        PolicySpec(
            key="B_strict_threshold",
            label="B_strict_threshold",
            description="Entropy-only with stricter early-exit threshold.",
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                enable_logit_blending=True,
                blending_confidence_threshold=0.035,
                blend_alpha=0.10,
                confidence_signal="entropy",
                blend_alpha_mode="fixed",
            ),
        ),
        PolicySpec(
            key="C_lower_alpha",
            label="C_lower_alpha",
            description="Entropy-only with lower fixed blend alpha.",
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                enable_logit_blending=True,
                blending_confidence_threshold=0.05,
                blend_alpha=0.05,
                confidence_signal="entropy",
                blend_alpha_mode="fixed",
            ),
        ),
        PolicySpec(
            key="D_strict_plus_low_alpha",
            label="D_strict_plus_low_alpha",
            description="Stricter entropy threshold with lower blend alpha.",
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                enable_logit_blending=True,
                blending_confidence_threshold=0.035,
                blend_alpha=0.05,
                confidence_signal="entropy",
                blend_alpha_mode="fixed",
            ),
        ),
        PolicySpec(
            key="E_margin_fallback",
            label="E_margin_fallback",
            description="Entropy + margin signal, with ambiguous cases forced to full depth.",
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                enable_logit_blending=True,
                blending_confidence_threshold=0.05,
                blend_alpha=0.05,
                confidence_signal="entropy_margin",
                margin_threshold=0.08,
                blend_alpha_mode="fixed",
                fallback_to_full_depth_on_ambiguity=True,
            ),
        ),
        PolicySpec(
            key="F_margin_sigmoid",
            label="F_margin_sigmoid",
            description="Entropy + margin trigger with adaptive sigmoid alpha.",
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                enable_logit_blending=True,
                blending_confidence_threshold=0.05,
                blend_alpha=0.10,
                confidence_signal="entropy_margin",
                margin_threshold=0.08,
                blend_alpha_mode="sigmoid_margin",
                blend_alpha_sigmoid_scale=20.0,
                blend_entropy_sigmoid_scale=20.0,
                fallback_to_full_depth_on_ambiguity=False,
            ),
        ),
    ]

    engine_by_key = {
        spec.key: build_engine(
            model=model,
            tokenizer=tokenizer,
            spec=spec,
            memory_limit_gb=args.memory_limit_gb,
        )
        for spec in policy_specs
    }

    mode_order = ["official_baseline"] + [spec.key for spec in policy_specs]
    mode_metadata = {
        "official_baseline": {
            "key": "official_baseline",
            "label": "official_baseline",
            "description": "Official mlx_lm greedy generation.",
            "settings": {
                "dynamic": False,
                "path": "official_mlx_generate_step",
            },
        }
    }
    for spec in policy_specs:
        mode_metadata[spec.key] = {
            "key": spec.key,
            "label": spec.label,
            "description": spec.description,
            "settings": {
                "dynamic": True,
                "hard_exit_layer": spec.config.hard_exit_layer,
                "enable_logit_blending": spec.config.enable_logit_blending,
                "blending_confidence_threshold": spec.config.blending_confidence_threshold,
                "blend_alpha": spec.config.blend_alpha,
                "confidence_signal": spec.config.confidence_signal,
                "margin_threshold": spec.config.margin_threshold,
                "blend_alpha_mode": spec.config.blend_alpha_mode,
                "blend_alpha_sigmoid_scale": spec.config.blend_alpha_sigmoid_scale,
                "blend_entropy_sigmoid_scale": spec.config.blend_entropy_sigmoid_scale,
                "fallback_to_full_depth_on_ambiguity": spec.config.fallback_to_full_depth_on_ambiguity,
            },
        }

    prompt_results_by_mode: Dict[str, List[Dict[str, Any]]] = {
        mode_key: [] for mode_key in mode_order
    }
    policy_by_key = {spec.key: spec for spec in policy_specs}

    for prompt_index, prompt_def in enumerate(prompt_defs):
        rotated_order = (
            mode_order[prompt_index % len(mode_order):]
            + mode_order[: prompt_index % len(mode_order)]
        )
        prompt_run_data: Dict[str, Dict[str, Any]] = {}
        consistency_by_key: Dict[str, Dict[str, Any]] = {}

        for mode_key in rotated_order:
            if mode_key == "official_baseline":
                prompt_run_data[mode_key] = official_greedy_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_def["prompt_ids"],
                    max_new_tokens=args.max_new_tokens,
                    prefill_step_size=2048,
                )
                continue

            engine = engine_by_key[mode_key]
            prompt_run_data[mode_key] = engine.generate_from_ids(
                prompt_def["prompt_ids"],
                max_new_tokens=args.max_new_tokens,
                dynamic=True,
                profile_runtime=False,
            )
            consistency_by_key[mode_key] = engine.consistency_check_from_ids(
                prompt_def["prompt_ids"],
                max_new_tokens=args.consistency_tokens,
            )

        official = prompt_run_data["official_baseline"]
        prompt_results_by_mode["official_baseline"].append(
            {
                "prompt_id": prompt_def["prompt_id"],
                "user_prompt": prompt_def["user"],
                "ttft_s": official["ttft_s"],
                "generation_tps": official["generation_tps"],
                "peak_memory_gb": official["peak_memory_gb"],
                "cache_memory_gb": official.get("cache_memory_gb", 0.0),
                "prompt_tokens": official["prompt_tokens"],
                "completion_tokens": official["tokens_generated"],
                "generated_ids": official["generated_ids"],
                "output_text": official["output_text"],
                "text_preview": preview_text(official["output_text"]),
                "comparison": {
                    "prefix_match_tokens": len(official["generated_ids"]),
                    "exact_match_rate": 1.0,
                    "first_divergence_position": None,
                    "passed": True,
                },
                "consistency": {
                    "tokens_compared": args.consistency_tokens,
                    "top1_agreement": 1.0,
                    "avg_kl_divergence": 0.0,
                },
                "avg_layers": 24.0,
                "avg_layers_saved": 0.0,
            }
        )

        for spec in policy_specs:
            stats = prompt_run_data[spec.key]
            comparison = compare_sequences(
                stats["generated_ids"],
                official["generated_ids"],
            )
            prompt_results_by_mode[spec.key].append(
                {
                    "prompt_id": prompt_def["prompt_id"],
                    "user_prompt": prompt_def["user"],
                    "ttft_s": stats["ttft_s"],
                    "generation_tps": stats["generation_tps"],
                    "peak_memory_gb": stats["peak_memory_gb"],
                    "cache_memory_gb": stats.get("cache_memory_gb", 0.0),
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["tokens_generated"],
                    "generated_ids": stats["generated_ids"],
                    "output_text": stats["output_text"],
                    "text_preview": preview_text(stats["output_text"]),
                    "comparison": comparison,
                    "consistency": consistency_by_key[spec.key],
                    "avg_layers": stats.get("avg_layers", 24.0),
                    "avg_layers_saved": max(24.0 - stats.get("avg_layers", 24.0), 0.0),
                }
            )

    config_results = []
    for mode_key in mode_order:
        prompt_results = prompt_results_by_mode[mode_key]
        included = [
            item for index, item in enumerate(prompt_results) if index != WARMUP_PROMPT_INDEX
        ]
        config_results.append(
            {
                **mode_metadata[mode_key],
                "prompt_results": prompt_results,
                "aggregate_including_warmup": aggregate_prompt_results(prompt_results),
                "aggregate_excluding_warmup": aggregate_prompt_results(included),
            }
        )

    candidate_results = [item for item in config_results if item["key"] != "official_baseline"]
    ranked_quality = sorted(candidate_results, key=quality_sort_key, reverse=True)
    official_aggregate = next(
        item["aggregate_excluding_warmup"]
        for item in config_results
        if item["key"] == "official_baseline"
    )
    for item in candidate_results:
        agg = item["aggregate_excluding_warmup"]
        generation_ratio = agg["avg_generation_tps"] / max(official_aggregate["avg_generation_tps"], 1e-6)
        ttft_ratio = official_aggregate["avg_ttft_s"] / max(agg["avg_ttft_s"], 1e-6)
        item["balanced_score"] = (
            agg["avg_exact_match_rate"] * generation_ratio * ttft_ratio
        )
    ranked_balanced = sorted(
        candidate_results,
        key=lambda item: (
            item["balanced_score"],
            item["aggregate_excluding_warmup"]["avg_exact_match_rate"],
            item["aggregate_excluding_warmup"]["avg_generation_tps"],
        ),
        reverse=True,
    )
    prompt_difficulty = []
    for prompt_index, prompt_def in enumerate(prompt_defs):
        prompt_rows = []
        for item in candidate_results:
            row = item["prompt_results"][prompt_index]
            prompt_rows.append(
                {
                    "config": item["label"],
                    "exact_match_rate": row["comparison"]["exact_match_rate"],
                    "prefix_match_tokens": row["comparison"]["prefix_match_tokens"],
                    "first_divergence_position": row["comparison"]["first_divergence_position"],
                }
            )
        prompt_difficulty.append(
            {
                "prompt_id": prompt_def["prompt_id"],
                "user_prompt": prompt_def["user"],
                "avg_exact_match_rate": mean([row["exact_match_rate"] for row in prompt_rows]),
                "worst_config": min(prompt_rows, key=lambda row: row["exact_match_rate"]),
                "best_config": max(prompt_rows, key=lambda row: row["exact_match_rate"]),
                "config_results": prompt_rows,
            }
        )

    ranking_summary = {
        "quality_ranking": [item["label"] for item in ranked_quality],
        "balanced_ranking": [item["label"] for item in ranked_balanced],
        "best_quality_config": ranked_quality[0]["label"],
        "best_balanced_config": ranked_balanced[0]["label"],
        "best_balanced_score": ranked_balanced[0]["balanced_score"],
        "reached_exact_080": ranked_quality[0]["aggregate_excluding_warmup"]["avg_exact_match_rate"] >= 0.80,
        "reached_exact_090": ranked_quality[0]["aggregate_excluding_warmup"]["avg_exact_match_rate"] >= 0.90,
    }

    payload = {
        "experiment": "transcender_quality_calibration",
        "resolved_model_path": resolved_model_path,
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "consistency_tokens": args.consistency_tokens,
            "reasoning_effort": REASONING_EFFORT,
            "warmup_discard_prompt_id": prompt_defs[WARMUP_PROMPT_INDEX]["prompt_id"],
            "notes": "Warmup-corrected aggregate excludes the first prompt result for every config.",
        },
        "prompts": [
            {
                "prompt_id": prompt_def["prompt_id"],
                "system": prompt_def["system"],
                "user": prompt_def["user"],
                "reasoning_effort": REASONING_EFFORT,
            }
            for prompt_def in prompt_defs
        ],
        "configs": config_results,
        "prompt_difficulty": prompt_difficulty,
        "ranking_summary": ranking_summary,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print_summary(payload, args.output)


if __name__ == "__main__":
    main()
