"""
Post-fix dynamic benchmark harness for the Transcender MLX engine.

This benchmark treats the corrected full-depth manual path as the source of
truth and evaluates a conservative policy sweep around layer 22.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from transcender_engine import (
    GptOssConfig,
    MLXDynamicExpertEngine,
    build_harmony_messages,
    load_resolved_mlx_model,
)


DEFAULT_MODEL_PATH = str(
    (Path(__file__).resolve().parent.parent / "gpt-oss-20b-raw").resolve()
)
DEFAULT_OUTPUT_PATH = str(
    (Path(__file__).resolve().parent / "postfix_dynamic_benchmark.json").resolve()
)
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_CONSISTENCY_TOKENS = 16
DEFAULT_CONSISTENCY_PROMPTS = 3
SYSTEM_PROMPT = "You are a helpful assistant."
REASONING_EFFORT = "medium"
ANALYSIS_MARKERS = (
    "analysis",
    "assistantfinal",
    "assistant final",
)

PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "Summarize why the French Revolution was historically important.",
    "Write a short explanation of recursion for a beginner programmer.",
    "Explain the difference between TCP and UDP in plain English.",
    "Describe what photosynthesis does.",
]


@dataclass(frozen=True)
class PolicyConfig:
    key: str
    label: str
    description: str
    soft_skip_start_layer: int
    hard_exit_layer: int
    entropy_threshold: float
    min_entropy_streak: int
    is_current_default: bool = False


POLICY_CONFIGS = [
    PolicyConfig(
        key="A_h23_no_skip",
        label="A",
        description=(
            "hard_exit=23, soft skip off. Control path: entropy_threshold=-1.0 "
            "guarantees the gate never opens."
        ),
        soft_skip_start_layer=19,
        hard_exit_layer=23,
        entropy_threshold=-1.0,
        min_entropy_streak=2,
    ),
    PolicyConfig(
        key="B_h22_no_skip",
        label="B",
        description=(
            "hard_exit=22, soft skip off. Isolates hard exit at layer 22 only."
        ),
        soft_skip_start_layer=19,
        hard_exit_layer=22,
        entropy_threshold=-1.0,
        min_entropy_streak=2,
    ),
    PolicyConfig(
        key="C_h22_default_gate",
        label="C",
        description=(
            "hard_exit=22, current default streak-2 gate on: "
            "soft_skip_start=19, entropy_threshold=0.20, min_entropy_streak=2."
        ),
        soft_skip_start_layer=19,
        hard_exit_layer=22,
        entropy_threshold=0.20,
        min_entropy_streak=2,
        is_current_default=True,
    ),
    PolicyConfig(
        key="D_h22_strict_gate",
        label="D",
        description=(
            "hard_exit=22, stricter gate: same layer schedule as default but "
            "requires lower entropy (threshold=0.15)."
        ),
        soft_skip_start_layer=19,
        hard_exit_layer=22,
        entropy_threshold=0.15,
        min_entropy_streak=2,
    ),
    PolicyConfig(
        key="E_h22_very_conservative_gate",
        label="E",
        description=(
            "hard_exit=22, very conservative gate: starts at layer 20, "
            "threshold=0.10, streak=2. If it opens at all, it can only skip "
            "layer 22."
        ),
        soft_skip_start_layer=20,
        hard_exit_layer=22,
        entropy_threshold=0.10,
        min_entropy_streak=2,
    ),
    PolicyConfig(
        key="F_h21_no_skip",
        label="F",
        description=(
            "hard_exit=21, soft skip off. Reference only for the more aggressive "
            "hard-exit depth."
        ),
        soft_skip_start_layer=19,
        hard_exit_layer=21,
        entropy_threshold=-1.0,
        min_entropy_streak=2,
    ),
]


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

    divergence = None
    for idx, (left_tok, right_tok) in enumerate(paired):
        if left_tok != right_tok:
            divergence = idx
            break
    if divergence is None and len(left) != len(right):
        divergence = min(len(left), len(right))

    return {
        "prefix_match_tokens": prefix,
        "exact_match_rate": exact_match_rate,
        "first_divergence_position": divergence,
    }


def preview_text(text: str, limit: int = 180) -> str:
    return text[:limit]


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def detect_analysis_markers(text: str) -> Dict[str, Any]:
    lower = text.lower()
    markers = [marker for marker in ANALYSIS_MARKERS if marker in lower]
    return {
        "has_analysis_like_trace": bool(markers),
        "markers": markers,
    }


def style_shift_vs_baseline(
    baseline_style: Dict[str, Any],
    dynamic_style: Dict[str, Any],
) -> bool:
    baseline_markers = set(baseline_style["markers"])
    dynamic_markers = set(dynamic_style["markers"])
    return bool(dynamic_markers - baseline_markers)


def build_engine(
    model,
    tokenizer,
    policy: PolicyConfig,
    memory_limit_gb: float,
) -> MLXDynamicExpertEngine:
    config = GptOssConfig(
        soft_skip_start_layer=policy.soft_skip_start_layer,
        hard_exit_layer=policy.hard_exit_layer,
        entropy_threshold=policy.entropy_threshold,
        min_entropy_streak=policy.min_entropy_streak,
        memory_limit_gb=memory_limit_gb,
    )
    return MLXDynamicExpertEngine(
        model=model,
        tokenizer=tokenizer,
        config=config,
        soft_skip_start_layer=policy.soft_skip_start_layer,
        hard_exit_layer=policy.hard_exit_layer,
        entropy_threshold=policy.entropy_threshold,
        min_entropy_streak=policy.min_entropy_streak,
        memory_limit_gb=memory_limit_gb,
    )


def warmup_engine(
    engine: MLXDynamicExpertEngine,
    messages: List[Dict[str, str]],
    dynamic: bool,
):
    engine.generate_from_messages(
        messages=messages,
        max_new_tokens=1,
        dynamic=dynamic,
        reasoning_effort=REASONING_EFFORT,
    )


def benchmark_prompt_pair(
    baseline_engine: MLXDynamicExpertEngine,
    dynamic_engine: MLXDynamicExpertEngine,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    baseline = baseline_engine.generate_from_messages(
        messages=messages,
        max_new_tokens=max_new_tokens,
        dynamic=False,
        reasoning_effort=REASONING_EFFORT,
    )
    dynamic = dynamic_engine.generate_from_messages(
        messages=messages,
        max_new_tokens=max_new_tokens,
        dynamic=True,
        reasoning_effort=REASONING_EFFORT,
    )
    comparison = compare_sequences(
        baseline["generated_ids"],
        dynamic["generated_ids"],
    )

    baseline_style = detect_analysis_markers(baseline["output_text"])
    dynamic_style = detect_analysis_markers(dynamic["output_text"])
    avg_layers_saved = max(dynamic_engine.num_layers - dynamic["avg_layers"], 0.0)

    return {
        "baseline": {
            "ttft_s": baseline["ttft_s"],
            "generation_tps": baseline["generation_tps"],
            "peak_memory_gb": baseline["peak_memory_gb"],
            "text_preview": preview_text(baseline["output_text"]),
            "generated_ids": baseline["generated_ids"],
            "analysis_style": baseline_style,
        },
        "dynamic": {
            "ttft_s": dynamic["ttft_s"],
            "generation_tps": dynamic["generation_tps"],
            "peak_memory_gb": dynamic["peak_memory_gb"],
            "avg_layers": dynamic["avg_layers"],
            "avg_layers_saved": avg_layers_saved,
            "text_preview": preview_text(dynamic["output_text"]),
            "generated_ids": dynamic["generated_ids"],
            "analysis_style": dynamic_style,
        },
        "ttft_improvement_pct": (
            (baseline["ttft_s"] - dynamic["ttft_s"])
            / max(baseline["ttft_s"], 1e-6)
            * 100.0
        ),
        "generation_tps_delta_pct": (
            (dynamic["generation_tps"] - baseline["generation_tps"])
            / max(baseline["generation_tps"], 1e-6)
            * 100.0
        ),
        "memory_delta_gb": dynamic["peak_memory_gb"] - baseline["peak_memory_gb"],
        "analysis_style_shift": style_shift_vs_baseline(baseline_style, dynamic_style),
        **comparison,
    }


def aggregate_prompt_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "avg_exact_match_rate": mean([item["exact_match_rate"] for item in results]),
        "avg_prefix_match_tokens": mean([item["prefix_match_tokens"] for item in results]),
        "avg_ttft_improvement_pct": mean([item["ttft_improvement_pct"] for item in results]),
        "avg_generation_tps_delta_pct": mean(
            [item["generation_tps_delta_pct"] for item in results]
        ),
        "avg_memory_delta_gb": mean([item["memory_delta_gb"] for item in results]),
        "avg_layers_saved": mean([item["dynamic"]["avg_layers_saved"] for item in results]),
        "avg_dynamic_layers": mean([item["dynamic"]["avg_layers"] for item in results]),
        "avg_baseline_ttft_s": mean([item["baseline"]["ttft_s"] for item in results]),
        "avg_dynamic_ttft_s": mean([item["dynamic"]["ttft_s"] for item in results]),
        "avg_baseline_generation_tps": mean(
            [item["baseline"]["generation_tps"] for item in results]
        ),
        "avg_dynamic_generation_tps": mean(
            [item["dynamic"]["generation_tps"] for item in results]
        ),
        "dynamic_analysis_trace_rate": mean(
            [
                1.0 if item["dynamic"]["analysis_style"]["has_analysis_like_trace"] else 0.0
                for item in results
            ]
        ),
        "baseline_analysis_trace_rate": mean(
            [
                1.0 if item["baseline"]["analysis_style"]["has_analysis_like_trace"] else 0.0
                for item in results
            ]
        ),
        "analysis_style_shift_rate": mean(
            [1.0 if item["analysis_style_shift"] else 0.0 for item in results]
        ),
    }


def classify_flags(aggregate: Dict[str, Any], kl_spike_threshold: float) -> Dict[str, bool]:
    return {
        "exact_match_collapse": (
            aggregate["avg_exact_match_rate"] < 0.35
            or aggregate["avg_prefix_match_tokens"] < 12.0
        ),
        "kl_spike": aggregate["avg_kl_divergence"] > kl_spike_threshold,
        "style_degradation": aggregate["analysis_style_shift_rate"] > 0.0,
        "commercially_meaningful": (
            aggregate["avg_exact_match_rate"] >= 0.90
            and aggregate["avg_ttft_improvement_pct"] >= 5.0
            and aggregate["avg_generation_tps_delta_pct"] >= -5.0
            and aggregate["avg_memory_delta_gb"] <= -0.25
            and aggregate["avg_top1_agreement"] >= 0.95
        ),
    }


def rank_configs(config_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_quality = max(
        config_results,
        key=lambda item: (
            item["aggregate"]["avg_exact_match_rate"],
            item["aggregate"]["avg_prefix_match_tokens"],
            item["aggregate"]["avg_top1_agreement"],
            -item["aggregate"]["avg_kl_divergence"],
        ),
    )
    best_speed = max(
        config_results,
        key=lambda item: (
            item["aggregate"]["avg_ttft_improvement_pct"],
            item["aggregate"]["avg_layers_saved"],
            item["aggregate"]["avg_exact_match_rate"],
        ),
    )

    viable = [
        item
        for item in config_results
        if not item["flags"]["exact_match_collapse"]
        and not item["flags"]["kl_spike"]
        and not item["flags"]["style_degradation"]
    ]
    overall_pool = viable or config_results
    best_overall = max(
        overall_pool,
        key=lambda item: (
            item["aggregate"]["avg_exact_match_rate"],
            item["aggregate"]["avg_prefix_match_tokens"],
            item["aggregate"]["avg_ttft_improvement_pct"],
            item["aggregate"]["avg_layers_saved"],
        ),
    )

    commercially_meaningful = [
        item for item in config_results if item["flags"]["commercially_meaningful"]
    ]

    recommended = None
    if (
        best_overall["aggregate"]["avg_exact_match_rate"] >= 0.90
        and best_overall["aggregate"]["avg_ttft_improvement_pct"] > 0.0
        and best_overall["aggregate"]["avg_generation_tps_delta_pct"] >= -10.0
    ):
        recommended = {
            "mode": "dynamic_enabled",
            "policy_key": best_overall["policy"]["key"],
            "reason": (
                "This policy is the best quality-preserving option that still "
                "offers some prompt-side latency benefit."
            ),
        }
    else:
        recommended = {
            "mode": "dynamic_disabled_baseline",
            "policy_key": best_quality["policy"]["key"],
            "reason": (
                "No dynamic configuration is strong enough to recommend as the "
                "default. Use full-depth baseline for now; if a policy must stay "
                "configured in code, keep the near-no-op layer-23 control."
            ),
        }

    return {
        "best_quality_config": best_quality["policy"]["key"],
        "best_speed_config": best_speed["policy"]["key"],
        "best_overall_quality_first_speed_second": best_overall["policy"]["key"],
        "commercially_meaningful_configs": [
            item["policy"]["key"] for item in commercially_meaningful
        ],
        "recommended_next_default": recommended,
    }


def print_summary(payload: Dict[str, Any], output_path: str):
    print("\nPost-Fix Dynamic Benchmark")
    print(f"JSON output: {output_path}")
    print(
        f"Prompt family: {len(payload['prompts'])} Harmony-formatted prompts | "
        f"reasoning_effort={payload['settings']['reasoning_effort']}"
    )

    print("\nAggregate Config Summary")
    header = (
        f"{'cfg':<18} {'exact':>8} {'prefix':>8} {'ttft%':>8} {'gen%':>8} "
        f"{'mem_gb':>8} {'saved':>8} {'agree':>8} {'kl':>8} {'flags':<24}"
    )
    print(header)
    print("-" * len(header))
    for config in payload["config_results"]:
        aggregate = config["aggregate"]
        flag_names = [name for name, active in config["flags"].items() if active]
        flags = ",".join(flag_names) if flag_names else "-"
        print(
            f"{config['policy']['key']:<18} "
            f"{aggregate['avg_exact_match_rate']:>8.3f} "
            f"{aggregate['avg_prefix_match_tokens']:>8.1f} "
            f"{aggregate['avg_ttft_improvement_pct']:>8.2f} "
            f"{aggregate['avg_generation_tps_delta_pct']:>8.2f} "
            f"{aggregate['avg_memory_delta_gb']:>8.2f} "
            f"{aggregate['avg_layers_saved']:>8.2f} "
            f"{aggregate['avg_top1_agreement']:>8.3f} "
            f"{aggregate['avg_kl_divergence']:>8.3f} "
            f"{flags:<24}"
        )

    print("\nPer-Prompt Results")
    prompt_header = (
        f"{'cfg':<18} {'prompt':<7} {'exact':>8} {'prefix':>8} {'ttft%':>8} "
        f"{'gen%':>8} {'saved':>8} {'div':>6}"
    )
    print(prompt_header)
    print("-" * len(prompt_header))
    for config in payload["config_results"]:
        for prompt_result in config["prompt_results"]:
            print(
                f"{config['policy']['key']:<18} "
                f"{prompt_result['prompt_id']:<7} "
                f"{prompt_result['exact_match_rate']:>8.3f} "
                f"{prompt_result['prefix_match_tokens']:>8} "
                f"{prompt_result['ttft_improvement_pct']:>8.2f} "
                f"{prompt_result['generation_tps_delta_pct']:>8.2f} "
                f"{prompt_result['dynamic']['avg_layers_saved']:>8.2f} "
                f"{str(prompt_result['first_divergence_position']):>6}"
            )

    print("\nRanking Summary")
    ranking = payload["ranking_summary"]
    print(f"best quality: {ranking['best_quality_config']}")
    print(f"best speed: {ranking['best_speed_config']}")
    print(
        "best overall (quality first, speed second): "
        f"{ranking['best_overall_quality_first_speed_second']}"
    )
    commercially_meaningful = ranking["commercially_meaningful_configs"] or ["none"]
    print(f"commercially meaningful configs: {', '.join(commercially_meaningful)}")
    print(
        "recommended next default: "
        f"{ranking['recommended_next_default']['mode']} "
        f"({ranking['recommended_next_default']['policy_key']})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Post-fix dynamic benchmark for the Transcender MLX engine."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--consistency-tokens",
        type=int,
        default=DEFAULT_CONSISTENCY_TOKENS,
    )
    parser.add_argument(
        "--consistency-prompts",
        type=int,
        default=DEFAULT_CONSISTENCY_PROMPTS,
    )
    parser.add_argument("--memory-limit-gb", type=float, default=30.0)
    args = parser.parse_args()

    model, tokenizer, resolved_model_path = load_resolved_mlx_model(
        args.model,
        lazy=True,
    )
    baseline_engine = build_engine(
        model=model,
        tokenizer=tokenizer,
        policy=POLICY_CONFIGS[0],
        memory_limit_gb=args.memory_limit_gb,
    )
    warmup_messages = build_harmony_messages(
        user_prompt=PROMPTS[0],
        system_prompt=SYSTEM_PROMPT,
    )
    warmup_engine(baseline_engine, warmup_messages, dynamic=False)

    prompt_defs = []
    baseline_cache: Dict[str, Dict[str, Any]] = {}
    for index, user_prompt in enumerate(PROMPTS, start=1):
        prompt_id = f"P{index}"
        messages = build_harmony_messages(
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        baseline_cache[prompt_id] = baseline_engine.generate_from_messages(
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            dynamic=False,
            reasoning_effort=REASONING_EFFORT,
        )
        prompt_defs.append(
            {
                "prompt_id": prompt_id,
                "user_prompt": user_prompt,
                "messages": messages,
            }
        )

    consistency_prompt_defs = prompt_defs[: max(args.consistency_prompts, 0)]
    config_results: List[Dict[str, Any]] = []
    for policy in POLICY_CONFIGS:
        engine = build_engine(
            model=model,
            tokenizer=tokenizer,
            policy=policy,
            memory_limit_gb=args.memory_limit_gb,
        )
        warmup_engine(engine, warmup_messages, dynamic=True)
        prompt_results = []
        for prompt_def in prompt_defs:
            baseline = baseline_cache[prompt_def["prompt_id"]]
            dynamic = engine.generate_from_messages(
                messages=prompt_def["messages"],
                max_new_tokens=args.max_new_tokens,
                dynamic=True,
                reasoning_effort=REASONING_EFFORT,
            )
            comparison = compare_sequences(
                baseline["generated_ids"],
                dynamic["generated_ids"],
            )
            baseline_style = detect_analysis_markers(baseline["output_text"])
            dynamic_style = detect_analysis_markers(dynamic["output_text"])
            avg_layers_saved = max(engine.num_layers - dynamic["avg_layers"], 0.0)
            prompt_results.append(
                {
                    "prompt_id": prompt_def["prompt_id"],
                    "user_prompt": prompt_def["user_prompt"],
                    "baseline": {
                        "ttft_s": baseline["ttft_s"],
                        "generation_tps": baseline["generation_tps"],
                        "peak_memory_gb": baseline["peak_memory_gb"],
                        "text_preview": preview_text(baseline["output_text"]),
                        "analysis_style": baseline_style,
                    },
                    "dynamic": {
                        "ttft_s": dynamic["ttft_s"],
                        "generation_tps": dynamic["generation_tps"],
                        "peak_memory_gb": dynamic["peak_memory_gb"],
                        "avg_layers": dynamic["avg_layers"],
                        "avg_layers_saved": avg_layers_saved,
                        "text_preview": preview_text(dynamic["output_text"]),
                        "analysis_style": dynamic_style,
                    },
                    "ttft_improvement_pct": (
                        (baseline["ttft_s"] - dynamic["ttft_s"])
                        / max(baseline["ttft_s"], 1e-6)
                        * 100.0
                    ),
                    "generation_tps_delta_pct": (
                        (dynamic["generation_tps"] - baseline["generation_tps"])
                        / max(baseline["generation_tps"], 1e-6)
                        * 100.0
                    ),
                    "memory_delta_gb": dynamic["peak_memory_gb"] - baseline["peak_memory_gb"],
                    "analysis_style_shift": style_shift_vs_baseline(
                        baseline_style,
                        dynamic_style,
                    ),
                    **comparison,
                }
            )

        consistency_checks = []
        for prompt_def in consistency_prompt_defs:
            check = engine.consistency_check_from_messages(
                messages=prompt_def["messages"],
                max_new_tokens=args.consistency_tokens,
                reasoning_effort=REASONING_EFFORT,
            )
            consistency_checks.append(
                {
                    "prompt_id": prompt_def["prompt_id"],
                    "user_prompt": prompt_def["user_prompt"],
                    **check,
                }
            )

        aggregate = aggregate_prompt_results(prompt_results)
        aggregate["avg_top1_agreement"] = mean(
            [item["top1_agreement"] for item in consistency_checks]
        )
        aggregate["avg_kl_divergence"] = mean(
            [item["avg_kl_divergence"] for item in consistency_checks]
        )
        aggregate["consistency_prompts_evaluated"] = len(consistency_checks)
        config_results.append(
            {
                "policy": {
                    "key": policy.key,
                    "label": policy.label,
                    "description": policy.description,
                    "soft_skip_start_layer": policy.soft_skip_start_layer,
                    "hard_exit_layer": policy.hard_exit_layer,
                    "entropy_threshold": policy.entropy_threshold,
                    "min_entropy_streak": policy.min_entropy_streak,
                    "is_current_default": policy.is_current_default,
                },
                "prompt_results": prompt_results,
                "consistency_checks": consistency_checks,
                "aggregate": aggregate,
            }
        )

    kl_values = [item["aggregate"]["avg_kl_divergence"] for item in config_results]
    positive_kl_values = [value for value in kl_values if value > 0.0]
    median_kl = statistics.median(positive_kl_values) if positive_kl_values else 0.0
    kl_spike_threshold = max(0.25, median_kl * 2.0)
    for item in config_results:
        item["flags"] = classify_flags(item["aggregate"], kl_spike_threshold)

    ranking_summary = rank_configs(config_results)
    current_default = next(
        item for item in config_results if item["policy"]["is_current_default"]
    )
    payload = {
        "experiment": "transcender_postfix_dynamic_benchmark",
        "resolved_model_path": resolved_model_path,
        "prompts": [
            {
                "prompt_id": prompt_def["prompt_id"],
                "system": SYSTEM_PROMPT,
                "user": prompt_def["user_prompt"],
                "reasoning_effort": REASONING_EFFORT,
            }
            for prompt_def in prompt_defs
        ],
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "consistency_tokens": args.consistency_tokens,
            "consistency_prompts": len(consistency_prompt_defs),
            "reasoning_effort": REASONING_EFFORT,
            "baseline_source_of_truth": "corrected_manual_full_depth_dynamic_false",
            "deterministic_decoding": True,
            "policy_sweep": [
                {
                    "key": policy.key,
                    "label": policy.label,
                    "description": policy.description,
                    "soft_skip_start_layer": policy.soft_skip_start_layer,
                    "hard_exit_layer": policy.hard_exit_layer,
                    "entropy_threshold": policy.entropy_threshold,
                    "min_entropy_streak": policy.min_entropy_streak,
                }
                for policy in POLICY_CONFIGS
            ],
            "kl_spike_threshold": kl_spike_threshold,
        },
        "current_default_policy": current_default["policy"]["key"],
        "config_results": config_results,
        "ranking_summary": ranking_summary,
        "best_quality_config": ranking_summary["best_quality_config"],
        "best_speed_config": ranking_summary["best_speed_config"],
        "recommended_next_default": ranking_summary["recommended_next_default"],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print_summary(payload, args.output)


if __name__ == "__main__":
    main()
