"""
Warmup-corrected benchmark for Transcender Engine v0.2.

Compares:
  - official mlx_lm greedy generation
  - optimized Transcender full-depth baseline
  - Transcender v0.1 legacy layer-22 hard exit
  - Transcender v0.2 optimized + logit-space blending
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
    (Path(__file__).resolve().parent / "transcender_v02_benchmark.json").resolve()
)
DEFAULT_MAX_NEW_TOKENS = 48
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
class EngineSpec:
    key: str
    label: str
    description: str
    dynamic: bool
    config: GptOssConfig
    trace_decisions: bool = False
    collect_runtime_stats: bool = True
    enable_allocator_cleanup: bool = True
    reuse_attention_masks: bool = True
    force_layer_state_eval: bool = False
    force_entropy_measurement: bool = False


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


def build_engine(
    model,
    tokenizer,
    spec: EngineSpec,
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
        memory_limit_gb=memory_limit_gb,
    )
    return MLXDynamicExpertEngine(
        model=model,
        tokenizer=tokenizer,
        config=config,
        soft_skip_start_layer=config.soft_skip_start_layer,
        hard_exit_layer=config.hard_exit_layer,
        entropy_threshold=config.entropy_threshold,
        min_entropy_streak=config.min_entropy_streak,
        trace_decisions=spec.trace_decisions,
        collect_runtime_stats=spec.collect_runtime_stats,
        enable_allocator_cleanup=spec.enable_allocator_cleanup,
        reuse_attention_masks=spec.reuse_attention_masks,
        force_layer_state_eval=spec.force_layer_state_eval,
        force_entropy_measurement=spec.force_entropy_measurement,
        enable_logit_blending=config.enable_logit_blending,
        blending_confidence_threshold=config.blending_confidence_threshold,
        blend_alpha=config.blend_alpha,
        memory_limit_gb=memory_limit_gb,
    )


def aggregate_prompt_results(prompt_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "avg_ttft_s": mean([item["ttft_s"] for item in prompt_results]),
        "avg_generation_tps": mean([item["generation_tps"] for item in prompt_results]),
        "avg_peak_memory_gb": mean([item["peak_memory_gb"] for item in prompt_results]),
        "avg_exact_match_rate": mean([item["comparison"]["exact_match_rate"] for item in prompt_results]),
        "avg_prefix_match_tokens": mean([item["comparison"]["prefix_match_tokens"] for item in prompt_results]),
        "avg_first_divergence_position": mean(
            [
                float(item["comparison"]["first_divergence_position"])
                for item in prompt_results
                if item["comparison"]["first_divergence_position"] is not None
            ]
        ),
        "parity_pass_rate": mean(
            [1.0 if item["comparison"]["passed"] else 0.0 for item in prompt_results]
        ),
        "avg_avg_layers": mean(
            [item.get("avg_layers", 24.0) for item in prompt_results]
        ),
        "avg_avg_layers_saved": mean(
            [item.get("avg_layers_saved", 0.0) for item in prompt_results]
        ),
        "avg_prefill_time_s": mean(
            [item.get("timings", {}).get("prefill_time_s", 0.0) for item in prompt_results]
        ),
        "avg_first_step_time_s": mean(
            [item.get("timings", {}).get("first_step_time_s", 0.0) for item in prompt_results]
        ),
        "avg_decode_loop_time_s": mean(
            [item.get("timings", {}).get("decode_loop_time_s", 0.0) for item in prompt_results]
        ),
        "avg_layer_loop_time_s": mean(
            [item.get("timings", {}).get("layer_loop_time_s", 0.0) for item in prompt_results]
        ),
        "avg_entropy_time_s": mean(
            [item.get("timings", {}).get("entropy_time_s", 0.0) for item in prompt_results]
        ),
        "avg_direct_model_forward_time_s": mean(
            [item.get("timings", {}).get("direct_model_forward_time_s", 0.0) for item in prompt_results]
        ),
    }


def print_summary(payload: Dict[str, Any], output_path: str):
    print("\nTranscender v0.2 Benchmark")
    print(f"JSON output: {output_path}")
    print(
        f"Warmup-corrected aggregate excludes prompt "
        f"{payload['prompts'][WARMUP_PROMPT_INDEX]['prompt_id']}"
    )

    print("\nAggregate Summary")
    header = (
        f"{'mode':<28} {'ttft':>8} {'gen_tps':>10} {'exact':>8} {'prefix':>8} "
        f"{'layers':>8} {'saved':>8} {'parity':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in payload["modes"]:
        aggregate = item["aggregate_excluding_warmup"]
        print(
            f"{item['label']:<28} "
            f"{aggregate['avg_ttft_s']:>8.3f} "
            f"{aggregate['avg_generation_tps']:>10.2f} "
            f"{aggregate['avg_exact_match_rate']:>8.3f} "
            f"{aggregate['avg_prefix_match_tokens']:>8.1f} "
            f"{aggregate['avg_avg_layers']:>8.2f} "
            f"{aggregate['avg_avg_layers_saved']:>8.2f} "
            f"{aggregate['parity_pass_rate']:>8.3f}"
        )

    print("\nPer-Prompt")
    prompt_header = (
        f"{'mode':<28} {'prompt':<7} {'ttft':>8} {'gen_tps':>10} "
        f"{'exact':>8} {'prefix':>8} {'div':>6}"
    )
    print(prompt_header)
    print("-" * len(prompt_header))
    for item in payload["modes"]:
        for prompt_result in item["prompt_results"]:
            print(
                f"{item['label']:<28} "
                f"{prompt_result['prompt_id']:<7} "
                f"{prompt_result['ttft_s']:>8.3f} "
                f"{prompt_result['generation_tps']:>10.2f} "
                f"{prompt_result['comparison']['exact_match_rate']:>8.3f} "
                f"{prompt_result['comparison']['prefix_match_tokens']:>8} "
                f"{str(prompt_result['comparison']['first_divergence_position']):>6}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Warmup-corrected benchmark for Transcender Engine v0.2."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--memory-limit-gb", type=float, default=30.0)
    args = parser.parse_args()

    model, tokenizer, resolved_model_path = load_resolved_mlx_model(
        args.model,
        lazy=True,
    )

    prompt_defs = []
    for index, user_prompt in enumerate(PROMPTS, start=1):
        prompt_id = f"P{index}"
        messages = build_harmony_messages(
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        prompt_text, rendered_messages = apply_harmony_template(
            tokenizer,
            messages=messages,
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

    mode_specs = [
        EngineSpec(
            key="engine_full_depth",
            label="Engine Full-Depth",
            description="Optimized Transcender full-depth path; must keep official parity.",
            dynamic=False,
            config=GptOssConfig(
                hard_exit_layer=23,
                entropy_threshold=-1.0,
                min_entropy_streak=2,
            ),
        ),
        EngineSpec(
            key="transcender_v01_legacy",
            label="Transcender v0.1 Legacy",
            description="Legacy layer-22 hard exit with old-style Python overhead switches re-enabled.",
            dynamic=True,
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                min_entropy_streak=2,
                enable_logit_blending=False,
            ),
            trace_decisions=True,
            force_layer_state_eval=True,
            force_entropy_measurement=True,
            reuse_attention_masks=False,
        ),
        EngineSpec(
            key="transcender_v02_blend",
            label="Transcender v0.2 Blend",
            description="Optimized layer-22 path with confidence-gated logit-space blending.",
            dynamic=True,
            config=GptOssConfig(
                hard_exit_layer=22,
                entropy_threshold=-1.0,
                min_entropy_streak=2,
                enable_logit_blending=True,
                blending_confidence_threshold=0.05,
                blend_alpha=0.10,
            ),
        ),
    ]

    engine_by_key = {}
    for spec in mode_specs:
        engine_by_key[spec.key] = build_engine(
            model=model,
            tokenizer=tokenizer,
            spec=spec,
            memory_limit_gb=args.memory_limit_gb,
        )
    mode_metadata = {
        "official_baseline": {
            "key": "official_baseline",
            "label": "Official Baseline",
            "description": "Official mlx_lm greedy generation.",
            "settings": {
                "dynamic": False,
                "path": "official_mlx_generate_step",
            },
        }
    }
    for spec in mode_specs:
        mode_metadata[spec.key] = {
            "key": spec.key,
            "label": spec.label,
            "description": spec.description,
            "settings": {
                "dynamic": spec.dynamic,
                "hard_exit_layer": spec.config.hard_exit_layer,
                "soft_skip_start_layer": spec.config.soft_skip_start_layer,
                "entropy_threshold": spec.config.entropy_threshold,
                "min_entropy_streak": spec.config.min_entropy_streak,
                "enable_logit_blending": spec.config.enable_logit_blending,
                "blending_confidence_threshold": spec.config.blending_confidence_threshold,
                "blend_alpha": spec.config.blend_alpha,
                "trace_decisions": spec.trace_decisions,
                "force_layer_state_eval": spec.force_layer_state_eval,
                "force_entropy_measurement": spec.force_entropy_measurement,
                "reuse_attention_masks": spec.reuse_attention_masks,
            },
        }

    mode_order = ["official_baseline"] + [spec.key for spec in mode_specs]
    prompt_results_by_mode: Dict[str, List[Dict[str, Any]]] = {
        mode_key: [] for mode_key in mode_order
    }

    spec_by_key = {spec.key: spec for spec in mode_specs}
    for prompt_index, prompt_def in enumerate(prompt_defs):
        rotated_order = (
            mode_order[prompt_index % len(mode_order):]
            + mode_order[: prompt_index % len(mode_order)]
        )
        prompt_run_data: Dict[str, Dict[str, Any]] = {}
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

            spec = spec_by_key[mode_key]
            engine = engine_by_key[mode_key]
            prompt_run_data[mode_key] = engine.generate_from_ids(
                prompt_def["prompt_ids"],
                max_new_tokens=args.max_new_tokens,
                dynamic=spec.dynamic,
                profile_runtime=True,
            )

        official = prompt_run_data["official_baseline"]
        prompt_results_by_mode["official_baseline"].append(
            {
                "prompt_id": prompt_def["prompt_id"],
                "user_prompt": prompt_def["user"],
                "ttft_s": official["ttft_s"],
                "generation_tps": official["generation_tps"],
                "peak_memory_gb": official["peak_memory_gb"],
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
                "timings": {},
                "avg_layers": 24.0,
                "avg_layers_saved": 0.0,
            }
        )

        for spec in mode_specs:
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
                    "cache_memory_gb": stats["cache_memory_gb"],
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["tokens_generated"],
                    "generated_ids": stats["generated_ids"],
                    "output_text": stats["output_text"],
                    "text_preview": preview_text(stats["output_text"]),
                    "comparison": comparison,
                    "timings": stats.get("timings", {}),
                    "avg_layers": stats.get("avg_layers", 24.0),
                    "avg_layers_saved": max(24.0 - stats.get("avg_layers", 24.0), 0.0),
                }
            )

    mode_results = []
    for mode_key in mode_order:
        prompt_results = prompt_results_by_mode[mode_key]
        included = [
            item for index, item in enumerate(prompt_results) if index != WARMUP_PROMPT_INDEX
        ]
        mode_results.append(
            {
                **mode_metadata[mode_key],
                "prompt_results": prompt_results,
                "aggregate_including_warmup": aggregate_prompt_results(prompt_results),
                "aggregate_excluding_warmup": aggregate_prompt_results(included),
            }
        )

    payload = {
        "experiment": "transcender_v02_benchmark",
        "resolved_model_path": resolved_model_path,
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "reasoning_effort": REASONING_EFFORT,
            "warmup_discard_prompt_id": prompt_defs[WARMUP_PROMPT_INDEX]["prompt_id"],
            "notes": (
                "Warmup-corrected aggregate excludes the first prompt result for every mode."
            ),
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
        "modes": mode_results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print_summary(payload, args.output)


if __name__ == "__main__":
    main()
