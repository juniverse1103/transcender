"""
Warmup-corrected exit-layer comparison benchmark for Transcender.

Purpose:
  Compare the current layer-22 top1_agree frontier against moving the same
  agreement-aware family one layer deeper to layer 23.
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
    (Path(__file__).resolve().parent / "transcender_exit_layer_benchmark.json").resolve()
)
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_CONSISTENCY_TOKENS = 16
SYSTEM_PROMPT = "You are a helpful assistant."
REASONING_EFFORT = "medium"
WARMUP_PROMPT_INDEX = 0
P3_PROMPT_ID = "P3"
MEANINGFUL_SAVINGS_THRESHOLD = 0.10

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
    dynamic: bool
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
        hard_exit_layer=spec.config.hard_exit_layer,
        entropy_threshold=spec.config.entropy_threshold,
        enable_logit_blending=spec.config.enable_logit_blending,
        blending_confidence_threshold=spec.config.blending_confidence_threshold,
        blend_alpha=spec.config.blend_alpha,
        confidence_signal=spec.config.confidence_signal,
        blend_alpha_mode=spec.config.blend_alpha_mode,
        blend_strategy=spec.config.blend_strategy,
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
        blend_alpha_mode=config.blend_alpha_mode,
        blend_strategy=config.blend_strategy,
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


def classify_exit_shift(
    aggregate: Dict[str, Any],
    control: Dict[str, Any],
) -> str:
    control_agg = control["aggregate_excluding_warmup"]
    if aggregate["avg_exact_match_rate"] <= control_agg["avg_exact_match_rate"]:
        return "no meaningful frontier improvement"
    if aggregate["avg_avg_layers_saved"] < MEANINGFUL_SAVINGS_THRESHOLD:
        return "quality win but mostly full-depth drift"
    return "better quality with acceptable savings loss"


def quality_sort_key(item: Dict[str, Any]):
    agg = item["aggregate_excluding_warmup"]
    p3 = item["p3_summary"]
    return (
        agg["avg_exact_match_rate"],
        p3["exact_match_rate"],
        p3["prefix_match_tokens"],
        agg["avg_avg_layers_saved"],
        agg["avg_generation_tps"],
    )


def print_summary(payload: Dict[str, Any], output_path: str):
    print("\nTranscender Exit-Layer Benchmark")
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
    print(f"Best config: {payload['ranking_summary']['best_config']}")
    print(f"Layer 23 beats layer 22: {payload['ranking_summary']['layer23_beats_layer22']}")
    print(
        f"Layer 23 gain justifies savings loss: "
        f"{payload['ranking_summary']['layer23_gain_justifies_savings_loss']}"
    )


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
            key="L22_top1_agree",
            label="L22_top1_agree",
            description="Current layer-22 frontier control.",
            hypothesis="Agreement-aware blend at layer 22 is the current best adaptive-quality frontier.",
            dynamic=True,
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
            key="L23_full_depth_reference",
            label="L23_full_depth_reference",
            description="Near-control reference at layer 23/full depth.",
            hypothesis="Moving to the final layer should recover full-depth quality but remove adaptive savings.",
            dynamic=False,
            config=GptOssConfig(
                hard_exit_layer=23,
                entropy_threshold=-1.0,
                enable_logit_blending=False,
            ),
        ),
        PolicySpec(
            key="L23_top1_agree",
            label="L23_top1_agree",
            description="Agreement-aware family shifted one layer deeper.",
            hypothesis="If the remaining failure is a layer-22 representation limit, using layer 23 should improve P3, but may drift toward full depth.",
            dynamic=True,
            config=GptOssConfig(
                hard_exit_layer=23,
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
            key="L23_top1_agree_low_alpha",
            label="L23_top1_agree_low_alpha",
            description="Conservative layer-23 agreement-aware variant.",
            hypothesis="A conservative alpha at layer 23 should not matter if the path has already collapsed to full depth; include it to verify that explicitly.",
            dynamic=True,
            config=GptOssConfig(
                hard_exit_layer=23,
                entropy_threshold=-1.0,
                enable_logit_blending=True,
                blending_confidence_threshold=0.035,
                blend_alpha=0.05,
                confidence_signal="entropy",
                blend_alpha_mode="fixed",
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

    prompt_results_by_mode: Dict[str, List[Dict[str, Any]]] = {
        spec.key: [] for spec in policy_specs
    }

    for prompt_index, prompt_def in enumerate(prompt_defs):
        baseline_stats = official_greedy_generate(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_def["prompt_ids"],
            max_new_tokens=args.max_new_tokens,
            prefill_step_size=2048,
        )
        for spec in policy_specs:
            engine = engines[spec.key]
            stats = engine.generate_from_ids(
                prompt_def["prompt_ids"],
                max_new_tokens=args.max_new_tokens,
                dynamic=spec.dynamic,
                profile_runtime=False,
            )
            consistency = engine.consistency_check_from_ids(
                prompt_def["prompt_ids"],
                max_new_tokens=args.consistency_tokens,
            )
            comparison = compare_sequences(
                stats["generated_ids"],
                baseline_stats["generated_ids"],
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
                    "consistency": consistency,
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
                    "dynamic": spec.dynamic,
                    "hard_exit_layer": spec.config.hard_exit_layer,
                    "blending_confidence_threshold": spec.config.blending_confidence_threshold,
                    "blend_alpha": spec.config.blend_alpha,
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

    control = next(item for item in config_results if item["key"] == "L22_top1_agree")
    for item in config_results:
        item["exit_layer_classification"] = classify_exit_shift(
            item["aggregate_excluding_warmup"],
            control,
        )
        item["delta_vs_layer22_control"] = {
            "avg_exact_match_rate": (
                item["aggregate_excluding_warmup"]["avg_exact_match_rate"]
                - control["aggregate_excluding_warmup"]["avg_exact_match_rate"]
            ),
            "avg_layers_saved": (
                item["aggregate_excluding_warmup"]["avg_avg_layers_saved"]
                - control["aggregate_excluding_warmup"]["avg_avg_layers_saved"]
            ),
            "avg_generation_tps": (
                item["aggregate_excluding_warmup"]["avg_generation_tps"]
                - control["aggregate_excluding_warmup"]["avg_generation_tps"]
            ),
            "avg_ttft_s": (
                item["aggregate_excluding_warmup"]["avg_ttft_s"]
                - control["aggregate_excluding_warmup"]["avg_ttft_s"]
            ),
            "p3_exact_match_rate": (
                item["p3_summary"]["exact_match_rate"] - control["p3_summary"]["exact_match_rate"]
            ),
        }

    ranked = sorted(config_results, key=quality_sort_key, reverse=True)
    layer23_candidates = [
        item for item in config_results if item["settings"]["hard_exit_layer"] == 23
    ]
    best_layer23 = sorted(layer23_candidates, key=quality_sort_key, reverse=True)[0]
    ranking_summary = {
        "best_config": ranked[0]["label"],
        "best_layer23_config": best_layer23["label"],
        "layer23_beats_layer22": (
            best_layer23["aggregate_excluding_warmup"]["avg_exact_match_rate"]
            > control["aggregate_excluding_warmup"]["avg_exact_match_rate"]
        ),
        "layer23_improves_p3": (
            best_layer23["p3_summary"]["exact_match_rate"]
            > control["p3_summary"]["exact_match_rate"]
        ),
        "layer23_gain_justifies_savings_loss": (
            best_layer23["aggregate_excluding_warmup"]["avg_exact_match_rate"]
            > control["aggregate_excluding_warmup"]["avg_exact_match_rate"]
            and best_layer23["aggregate_excluding_warmup"]["avg_avg_layers_saved"] >= MEANINGFUL_SAVINGS_THRESHOLD
        ),
        "quality_ranking": [item["label"] for item in ranked],
    }

    payload = {
        "experiment": "transcender_exit_layer_benchmark",
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
        "p3_comparison_summary": {
            item["label"]: item["p3_summary"] for item in config_results
        },
        "ranking_summary": ranking_summary,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print_summary(payload, args.output)


if __name__ == "__main__":
    main()
