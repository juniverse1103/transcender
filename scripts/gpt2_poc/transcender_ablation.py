"""
Hard-exit-only layer ablation for the Transcender MLX engine.

This experiment isolates hard-exit depth by explicitly disabling all
entropy-based soft skipping:
  - soft_skip_start_layer remains at layer 19
  - min_entropy_streak remains 2
  - entropy_threshold is forced to -1.0

Because normalized entropy is always >= 0, the threshold -1.0 guarantees that
the entropy gate never opens. The dynamic path therefore differs only by the
selected hard exit layer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx_lm.generate import generate_step
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
    (Path(__file__).resolve().parent / "hard_exit_ablation.json").resolve()
)

ABLATION_EXIT_LAYERS = [23, 22, 21]
HARD_EXIT_ONLY_SOFT_SKIP_START_LAYER = 19
HARD_EXIT_ONLY_ENTROPY_THRESHOLD = -1.0
HARD_EXIT_ONLY_MIN_STREAK = 2
DEFAULT_MAX_NEW_TOKENS = 48
TOKEN_PREVIEW_COUNT = 32

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

    return {
        "prefix_match_tokens": prefix,
        "exact_match_rate": exact_match_rate,
        "first_divergence_position": first_divergence_position(left, right),
    }


def preview_text(text: str, limit: int = 200) -> str:
    return text[:limit]


def official_mlx_greedy_generate(
    model,
    tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    prefill_step_size: int,
) -> Dict[str, Any]:
    """
    Collect greedy tokens from the official mlx_lm generation step path.

    We use generate_step() directly with temp=0 greedy argmax sampling so this
    is the official MLX decode path, but we stop on EOS using the same rule as
    the manual baseline to make the token comparison fair.
    """
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    sampler = make_sampler(temp=0.0)
    eos_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    if not eos_ids:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            eos_ids = {eos_token_id}

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
    mx.clear_cache()

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


def build_hard_exit_only_engine(
    model,
    tokenizer,
    hard_exit_layer: int,
    memory_limit_gb: float,
) -> MLXDynamicExpertEngine:
    config = GptOssConfig(
        soft_skip_start_layer=HARD_EXIT_ONLY_SOFT_SKIP_START_LAYER,
        hard_exit_layer=hard_exit_layer,
        entropy_threshold=HARD_EXIT_ONLY_ENTROPY_THRESHOLD,
        min_entropy_streak=HARD_EXIT_ONLY_MIN_STREAK,
        memory_limit_gb=memory_limit_gb,
    )
    return MLXDynamicExpertEngine(
        model=model,
        tokenizer=tokenizer,
        config=config,
        soft_skip_start_layer=HARD_EXIT_ONLY_SOFT_SKIP_START_LAYER,
        hard_exit_layer=hard_exit_layer,
        entropy_threshold=HARD_EXIT_ONLY_ENTROPY_THRESHOLD,
        min_entropy_streak=HARD_EXIT_ONLY_MIN_STREAK,
        memory_limit_gb=memory_limit_gb,
    )


def summarize_run(stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ttft_s": stats["ttft_s"],
        "generation_tps": stats["generation_tps"],
        "peak_memory_gb": stats["peak_memory_gb"],
        "avg_layers": stats.get("avg_layers"),
        "first_32_token_ids": stats["generated_ids"][:TOKEN_PREVIEW_COUNT],
        "text_preview": preview_text(stats["output_text"]),
    }


def run_ablation(
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

    control_engine = build_hard_exit_only_engine(
        model=model,
        tokenizer=tokenizer,
        hard_exit_layer=23,
        memory_limit_gb=memory_limit_gb,
    )
    official_baseline = official_mlx_greedy_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        prefill_step_size=control_engine.prefill_step_size,
    )
    manual_baseline_for_equivalence = control_engine.generate_from_ids(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        dynamic=False,
    )
    baseline_equivalence = compare_sequences(
        manual_baseline_for_equivalence["generated_ids"],
        official_baseline["generated_ids"],
    )
    baseline_equivalence.update(
        {
            "passed": (
                manual_baseline_for_equivalence["generated_ids"]
                == official_baseline["generated_ids"]
            ),
            "manual_baseline": summarize_run(manual_baseline_for_equivalence),
            "official_mlx": summarize_run(official_baseline),
            "manual_generated_ids": manual_baseline_for_equivalence["generated_ids"],
            "official_generated_ids": official_baseline["generated_ids"],
        }
    )

    runs = []
    for exit_layer in ABLATION_EXIT_LAYERS:
        engine = build_hard_exit_only_engine(
            model=model,
            tokenizer=tokenizer,
            hard_exit_layer=exit_layer,
            memory_limit_gb=memory_limit_gb,
        )
        baseline = engine.generate_from_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            dynamic=False,
        )
        dynamic = engine.generate_from_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            dynamic=True,
        )
        comparison = compare_sequences(
            baseline["generated_ids"],
            dynamic["generated_ids"],
        )
        runs.append(
            {
                "hard_exit_layer": exit_layer,
                "hard_exit_only": {
                    "soft_skip_start_layer": HARD_EXIT_ONLY_SOFT_SKIP_START_LAYER,
                    "min_entropy_streak": HARD_EXIT_ONLY_MIN_STREAK,
                    "entropy_threshold": HARD_EXIT_ONLY_ENTROPY_THRESHOLD,
                    "explanation": (
                        "Hard-exit-only ablation: entropy threshold is set to -1.0, "
                        "so the normalized entropy gate can never activate."
                    ),
                },
                "baseline": summarize_run(baseline),
                "dynamic": summarize_run(dynamic),
                "baseline_generated_ids": baseline["generated_ids"],
                "dynamic_generated_ids": dynamic["generated_ids"],
                **comparison,
            }
        )

    payload = {
        "experiment": "transcender_hard_exit_only_ablation",
        "prompt": {
            "system": SYSTEM_PROMPT,
            "user": USER_PROMPT,
            "reasoning_effort": REASONING_EFFORT,
            "rendered_harmony_prompt": prompt_text,
            "prompt_token_count": len(prompt_ids),
            "messages": messages,
        },
        "settings": {
            "max_new_tokens": max_new_tokens,
            "exit_layers": ABLATION_EXIT_LAYERS,
            "soft_skip_disabled": True,
            "soft_skip_disable_method": {
                "soft_skip_start_layer": HARD_EXIT_ONLY_SOFT_SKIP_START_LAYER,
                "min_entropy_streak": HARD_EXIT_ONLY_MIN_STREAK,
                "entropy_threshold": HARD_EXIT_ONLY_ENTROPY_THRESHOLD,
            },
            "resolved_model_path": resolved_model_path,
        },
        "baseline_equivalence": baseline_equivalence,
        "runs": runs,
    }

    Path(output_path).write_text(json.dumps(payload, indent=2))
    print_human_summary(payload, output_path)


def print_human_summary(payload: Dict[str, Any], output_path: str):
    print("\nHard-Exit-Only Ablation")
    print(
        "Soft skip disabled explicitly: "
        f"entropy_threshold={HARD_EXIT_ONLY_ENTROPY_THRESHOLD}, "
        f"soft_skip_start_layer={HARD_EXIT_ONLY_SOFT_SKIP_START_LAYER}, "
        f"min_entropy_streak={HARD_EXIT_ONLY_MIN_STREAK}"
    )
    print(
        f"Prompt: system={SYSTEM_PROMPT!r} | user={USER_PROMPT!r} | "
        f"reasoning_effort={REASONING_EFFORT}"
    )
    print(f"JSON output: {output_path}")

    control = payload["baseline_equivalence"]
    print("\nBaseline Equivalence Control")
    print(
        f"status={'PASS' if control['passed'] else 'FAIL'} | "
        f"prefix_match={control['prefix_match_tokens']} | "
        f"exact_match_rate={control['exact_match_rate']:.3f} | "
        f"first_divergence={control['first_divergence_position']}"
    )
    if not control["passed"]:
        print("BASELINE-EQUIVALENCE FAILURE")
        print(f"manual baseline tokens: {control['manual_generated_ids']}")
        print(f"official mlx tokens:    {control['official_generated_ids']}")

    print("\nSummary Table")
    header = (
        f"{'exit':>4} {'base_ttft':>10} {'base_tps':>10} {'base_mem':>10} "
        f"{'dyn_ttft':>10} {'dyn_tps':>10} {'dyn_mem':>10} {'dyn_layers':>10} "
        f"{'prefix':>8} {'exact':>8} {'div':>6}"
    )
    print(header)
    print("-" * len(header))
    for run in payload["runs"]:
        print(
            f"{run['hard_exit_layer']:>4} "
            f"{run['baseline']['ttft_s']:>10.3f} "
            f"{run['baseline']['generation_tps']:>10.2f} "
            f"{run['baseline']['peak_memory_gb']:>10.2f} "
            f"{run['dynamic']['ttft_s']:>10.3f} "
            f"{run['dynamic']['generation_tps']:>10.2f} "
            f"{run['dynamic']['peak_memory_gb']:>10.2f} "
            f"{run['dynamic']['avg_layers']:>10.2f} "
            f"{run['prefix_match_tokens']:>8} "
            f"{run['exact_match_rate']:>8.3f} "
            f"{str(run['first_divergence_position']):>6}"
        )

    print("\nToken / Text Previews")
    for run in payload["runs"]:
        print(f"\nExit layer {run['hard_exit_layer']}")
        print(f"baseline first 32 ids: {run['baseline']['first_32_token_ids']}")
        print(f"dynamic  first 32 ids: {run['dynamic']['first_32_token_ids']}")
        print(f"baseline preview: {run['baseline']['text_preview']}")
        print(f"dynamic  preview: {run['dynamic']['text_preview']}")


def main():
    parser = argparse.ArgumentParser(
        description="Strict hard-exit-only ablation for the Transcender MLX engine."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--memory-limit-gb", type=float, default=30.0)
    args = parser.parse_args()

    run_ablation(
        model_path=args.model,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        memory_limit_gb=args.memory_limit_gb,
    )


if __name__ == "__main__":
    main()
