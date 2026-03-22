"""
Warmup-corrected benchmark for top1_agree family extensions.

Focus: determine whether agreement-aware alpha shaping can improve the
remaining recursion-style failure (P3) without sacrificing meaningful
adaptive savings at layer 22.
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
    (Path(__file__).resolve().parent / "transcender_top1_agree_benchmark.json").resolve()
)
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_CONSISTENCY_TOKENS = 16
SYSTEM_PROMPT = "You are a helpful assistant."
REASONING_EFFORT = "medium"
WARMUP_PROMPT_INDEX = 0
MEANINGFUL_SAVINGS_THRESHOLD = 0.40
P3_PROMPT_ID = "P3"

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
    hypothesis: str
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
        blend_alpha_margin_scale=spec.config.blend_alpha_margin_scale,
        fallback_to_full_depth_on_ambiguity=spec.config.fallback_to_full_depth_on_ambiguity,
        blend_strategy=spec.config.blend_strategy,
        blend_top_k=spec.config.blend_top_k,
        anchor_alpha_scale=spec.config.anchor_alpha_scale,
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
        blend_alpha_margin_scale=config.blend_alpha_margin_scale,
        fallback_to_full_depth_on_ambiguity=config.fallback_to_full_depth_on_ambiguity,
        blend_strategy=config.blend_strategy,
        blend_top_k=config.blend_top_k,
        anchor_alpha_scale=config.anchor_alpha_scale,
        memory_limit_gb=memory_limit_gb,
    )


def aggregate_prompt_results(prompt_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    divergences = [
        float(item["comparison"]["first_divergence_position"])
        for item in prompt_results
        if item["comparison"]["first_divergence_position"] is not None
    ]
    return {
        "avg_ttft_s": mean([item["ttft_s"] for item in prompt_results]),
        "avg_generation_tps": mean([item["generation_tps"] for item in prompt_results]),
        "avg_peak_memory_gb": mean([item["peak_memory_gb"] for item in prompt_results]),
        "avg_exact_match_rate": mean([item["comparison"]["exact_match_rate"] for item in prompt_results]),
        "avg_prefix_match_tokens": mean([item["comparison"]["prefix_match_tokens"] for item in prompt_results]),
        "avg_first_divergence_position": mean(divergences),
        "avg_avg_layers_saved": mean([item.get("avg_layers_saved", 0.0) for item in prompt_results]),
        "avg_consistency_top1_agreement": mean(
            [item.get("consistency", {}).get("top1_agreement", 0.0) for item in prompt_results]
        ),
        "avg_consistency_kl_divergence": mean(
            [item.get("consistency", {}).get("avg_kl_divergence", 0.0) for item in prompt_results]
        ),
    }


def quality_sort_key(item: Dict[str, Any]):
    agg = item["aggregate_excluding_warmup"]
    p3 = item["p3_summary"]
    return (
        agg["avg_exact_match_rate"],
        p3["exact_match_rate"],
        p3["prefix_match_tokens"],
        agg["avg_prefix_match_tokens"],
        agg["avg_generation_tps"],
    )


def characterize_config(
    aggregate: Dict[str, Any],
    control_exact: float,
) -> str:
    if aggregate["avg_avg_layers_saved"] < 0.05:
        return "mostly fallback-to-full-depth"
    if (
        aggregate["avg_exact_match_rate"] > control_exact
        and aggregate["avg_avg_layers_saved"] >= MEANINGFUL_SAVINGS_THRESHOLD
    ):
        return "true adaptive improvement"
    return "mixed / ambiguous"


def print_summary(payload: Dict[str, Any], output_path: str):
    print("\nTranscender top1_agree Benchmark")
    print(f"JSON output: {output_path}")
    print(
        f"Warmup-corrected aggregate excludes prompt "
        f"{payload['prompts'][WARMUP_PROMPT_INDEX]['prompt_id']}"
    )
    print("\nAggregate Summary")
    header = (
        f"{'config':<28} {'ttft':>8} {'gen_tps':>10} {'exact':>8} "
        f"{'prefix':>8} {'saved':>8} {'p3_exact':>9} {'p3_pref':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in payload["configs"]:
        agg = item["aggregate_excluding_warmup"]
        p3 = item["p3_summary"]
        print(
            f"{item['label']:<28} "
            f"{agg['avg_ttft_s']:>8.3f} "
            f"{agg['avg_generation_tps']:>10.2f} "
            f"{agg['avg_exact_match_rate']:>8.3f} "
            f"{agg['avg_prefix_match_tokens']:>8.1f} "
            f"{agg['avg_avg_layers_saved']:>8.2f} "
            f"{p3['exact_match_rate']:>9.3f} "
            f"{p3['prefix_match_tokens']:>8.1f}"
        )
    print("\nRecommendation")
    print(f"Best raw quality: {payload['ranking_summary']['best_raw_quality_config']}")
    print(f"Best adaptive frontier: {payload['ranking_summary']['best_adaptive_quality_frontier_config']}")
    print(f"Improved on control: {payload['ranking_summary']['improved_on_control']}")
    print(f"Improved P3 with meaningful savings: {payload['ranking_summary']['improved_p3_with_meaningful_savings']}")


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
        messages = build_harmony_messages(
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        prompt_text, _ = apply_harmony_template(
            tokenizer,
            messages=messages,
            reasoning_effort=REASONING_EFFORT,
        )
        prompt_defs.append(
            {
                "prompt_id": prompt_id,
                "system": SYSTEM_PROMPT,
                "user": user_prompt,
                "messages": messages,
                "prompt_text": prompt_text,
                "prompt_ids": tokenizer.encode(prompt_text),
            }
        )

    policy_specs = [
        PolicySpec(
            key="H_top1_agree",
            label="H_top1_agree",
            description="Current control.",
            hypothesis="Only blend when early and deep top-1 match, with fixed alpha.",
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
        ),
        PolicySpec(
            key="K_top1_agree_dynamic_margin_fast",
            label="K_top1_agree_dynamic_margin_fast",
            description="Linear margin-scaled alpha, fast ramp.",
            hypothesis="When L22 agrees but its margin is weak, reduce alpha sharply to avoid perturbing recursive continuations.",
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                enable_logit_blending=True,
                blending_confidence_threshold=0.035,
                blend_alpha=0.10,
                confidence_signal="entropy",
                blend_alpha_mode="margin_linear",
                blend_alpha_margin_scale=2.0,
                blend_strategy="top1_agree",
            ),
        ),
        PolicySpec(
            key="L_top1_agree_dynamic_margin_slow",
            label="L_top1_agree_dynamic_margin_slow",
            description="Linear margin-scaled alpha, slower ramp.",
            hypothesis="Use a more conservative alpha curve so low-margin agreements lean much harder on L24 while preserving some early signal on very confident cases.",
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                enable_logit_blending=True,
                blending_confidence_threshold=0.035,
                blend_alpha=0.10,
                confidence_signal="entropy",
                blend_alpha_mode="margin_linear",
                blend_alpha_margin_scale=1.0,
                blend_strategy="top1_agree",
            ),
        ),
        PolicySpec(
            key="M_top1_agree_sigmoid_margin",
            label="M_top1_agree_sigmoid_margin",
            description="Sigmoid margin alpha inside top1_agree.",
            hypothesis="Smoothly taper alpha near the ambiguity boundary without making the change purely linear in the raw margin.",
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                enable_logit_blending=True,
                blending_confidence_threshold=0.035,
                blend_alpha=0.10,
                confidence_signal="entropy",
                margin_threshold=0.08,
                blend_alpha_mode="sigmoid_margin",
                blend_alpha_sigmoid_scale=20.0,
                blend_entropy_sigmoid_scale=20.0,
                blend_strategy="top1_agree",
            ),
        ),
        PolicySpec(
            key="N_top1_agree_dynamic_margin_lowcap",
            label="N_top1_agree_dynamic_margin_lowcap",
            description="Linear margin alpha with lower cap.",
            hypothesis="If fixed alpha is still too strong under agreement, reduce the maximum early contribution while keeping margin sensitivity.",
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                enable_logit_blending=True,
                blending_confidence_threshold=0.035,
                blend_alpha=0.08,
                confidence_signal="entropy",
                blend_alpha_mode="margin_linear",
                blend_alpha_margin_scale=2.0,
                blend_strategy="top1_agree",
            ),
        ),
    ]

    engines = {
        spec.key: build_engine(
            model=model,
            tokenizer=tokenizer,
            spec=spec,
            memory_limit_gb=args.memory_limit_gb,
        )
        for spec in policy_specs
    }

    mode_order = ["official_baseline"] + [spec.key for spec in policy_specs]
    prompt_results_by_mode: Dict[str, List[Dict[str, Any]]] = {
        mode_key: [] for mode_key in mode_order
    }

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

            engine = engines[mode_key]
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
                    "avg_layers_saved": max(24.0 - stats.get("avg_layers", 24.0), 0.0),
                }
            )

    config_results = []
    for spec in policy_specs:
        prompt_results = prompt_results_by_mode[spec.key]
        included = [
            item for index, item in enumerate(prompt_results) if index != WARMUP_PROMPT_INDEX
        ]
        p3_entry = next(item for item in prompt_results if item["prompt_id"] == P3_PROMPT_ID)
        config_results.append(
            {
                "key": spec.key,
                "label": spec.label,
                "description": spec.description,
                "hypothesis": spec.hypothesis,
                "settings": {
                    "hard_exit_layer": spec.config.hard_exit_layer,
                    "blending_confidence_threshold": spec.config.blending_confidence_threshold,
                    "blend_alpha": spec.config.blend_alpha,
                    "blend_alpha_mode": spec.config.blend_alpha_mode,
                    "blend_alpha_margin_scale": spec.config.blend_alpha_margin_scale,
                    "blend_alpha_sigmoid_scale": spec.config.blend_alpha_sigmoid_scale,
                    "margin_threshold": spec.config.margin_threshold,
                    "blend_strategy": spec.config.blend_strategy,
                },
                "prompt_results": prompt_results,
                "aggregate_including_warmup": aggregate_prompt_results(prompt_results),
                "aggregate_excluding_warmup": aggregate_prompt_results(included),
                "p3_summary": {
                    "exact_match_rate": p3_entry["comparison"]["exact_match_rate"],
                    "prefix_match_tokens": p3_entry["comparison"]["prefix_match_tokens"],
                    "first_divergence_position": p3_entry["comparison"]["first_divergence_position"],
                    "avg_layers_saved": p3_entry["avg_layers_saved"],
                    "consistency_top1_agreement": p3_entry["consistency"]["top1_agreement"],
                    "consistency_kl_divergence": p3_entry["consistency"]["avg_kl_divergence"],
                },
            }
        )

    control = next(item for item in config_results if item["key"] == "H_top1_agree")
    control_exact = control["aggregate_excluding_warmup"]["avg_exact_match_rate"]
    control_p3_exact = control["p3_summary"]["exact_match_rate"]
    for item in config_results:
        item["adaptive_characterization"] = characterize_config(
            item["aggregate_excluding_warmup"],
            control_exact,
        )
        item["delta_vs_control"] = {
            "avg_exact_match_rate": (
                item["aggregate_excluding_warmup"]["avg_exact_match_rate"] - control_exact
            ),
            "p3_exact_match_rate": item["p3_summary"]["exact_match_rate"] - control_p3_exact,
            "avg_layers_saved": (
                item["aggregate_excluding_warmup"]["avg_avg_layers_saved"]
                - control["aggregate_excluding_warmup"]["avg_avg_layers_saved"]
            ),
        }

    ranked_quality = sorted(config_results, key=quality_sort_key, reverse=True)
    adaptive_candidates = [
        item
        for item in config_results
        if item["aggregate_excluding_warmup"]["avg_avg_layers_saved"] >= MEANINGFUL_SAVINGS_THRESHOLD
    ]
    ranked_adaptive = sorted(adaptive_candidates, key=quality_sort_key, reverse=True)
    p3_summary = {
        item["label"]: item["p3_summary"] for item in config_results
    }
    ranking_summary = {
        "best_raw_quality_config": ranked_quality[0]["label"],
        "best_adaptive_quality_frontier_config": ranked_adaptive[0]["label"] if ranked_adaptive else None,
        "quality_ranking": [item["label"] for item in ranked_quality],
        "adaptive_ranking": [item["label"] for item in ranked_adaptive],
        "improved_on_control": any(
            item["aggregate_excluding_warmup"]["avg_exact_match_rate"] > control_exact
            for item in config_results
            if item["key"] != control["key"]
        ),
        "improved_p3": any(
            item["p3_summary"]["exact_match_rate"] > control_p3_exact
            for item in config_results
            if item["key"] != control["key"]
        ),
        "improved_p3_with_meaningful_savings": any(
            item["p3_summary"]["exact_match_rate"] > control_p3_exact
            and item["aggregate_excluding_warmup"]["avg_avg_layers_saved"] >= MEANINGFUL_SAVINGS_THRESHOLD
            for item in config_results
            if item["key"] != control["key"]
        ),
        "meaningful_savings_threshold": MEANINGFUL_SAVINGS_THRESHOLD,
    }

    payload = {
        "experiment": "transcender_top1_agree_benchmark",
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
        "p3_focused_summary": p3_summary,
        "ranking_summary": ranking_summary,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print_summary(payload, args.output)


if __name__ == "__main__":
    main()
