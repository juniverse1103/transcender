"""
Warmup-corrected Track A exit-layer frontier benchmark for Transcender.

Purpose:
  Measure the local penultimate-layer frontier on the supported sparse-MoE
  families:
    - GPT-OSS 20B (`gpt_oss`)
    - Qwen3-30B-A3B (`qwen3_moe`)

  The script emits per-prompt and aggregate metrics for the configured policy
  set and its matched full-depth control.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from transcender_engine import (
    GptOssConfig,
    MLXDynamicExpertEngine,
    Qwen3MoeConfig,
    apply_harmony_template,
    build_harmony_messages,
    load_mlx_model,
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
    # =====================================================================
    # Original suite (P1-P16) — kept in place for backward compatibility.
    # P1 is always the warmup prompt (discarded from scored aggregates).
    # =====================================================================
    # --- Original expository (P1-P5) ---
    "Explain quantum entanglement in simple terms.",                          # P1 (warmup)
    "Summarize why the French Revolution was historically important.",        # P2
    "Write a short explanation of recursion for a beginner programmer.",      # P3
    "Explain the difference between TCP and UDP in plain English.",           # P4
    "Describe what photosynthesis does.",                                     # P5
    # --- Code/technical (P6-P7) ---
    "Write a Python function that checks if a string is a palindrome.",      # P6
    "Explain what a hash table is and when you would use one.",              # P7
    # --- Reasoning/logic (P8-P9) ---
    "A farmer has 17 sheep. All but 9 run away. How many are left and why?", # P8
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",  # P9
    # --- Creative/open-ended (P10-P11) ---
    "Write the opening paragraph of a mystery story set in a library.",      # P10
    "Describe a sunset to someone who has never seen one.",                   # P11
    # --- Short-answer/factual (P12-P14) ---
    "What is the capital of Australia?",                                      # P12
    "What does the HTTP status code 404 mean?",                              # P13
    "Name three noble gases.",                                                # P14
    # --- List/structured output (P15-P16) ---
    "List five common sorting algorithms and one sentence about each.",       # P15
    "List the planets of the solar system in order from the Sun.",            # P16
    # =====================================================================
    # Expanded suite (P17-P64) — added to increase N from 15 to 63.
    # =====================================================================
    # --- Expository (P17-P24) ---
    "Explain how a transformer neural network processes a sentence.",         # P17
    "Describe the greenhouse effect and why it matters for climate.",         # P18
    "Explain what an API is to someone with no programming background.",      # P19
    "Describe how vaccines train the immune system.",                         # P20
    "Explain the difference between machine learning and traditional programming.",  # P21
    "Describe how a blockchain works at a high level.",                       # P22
    "Explain what inflation is and how it affects purchasing power.",         # P23
    "Describe the water cycle in four or five sentences.",                    # P24
    # --- Code/technical (P25-P32) ---
    "Write a Python function that reverses a linked list.",                   # P25
    "Explain the difference between a stack and a queue.",                    # P26
    "Write a Python function that finds the second largest number in a list.",  # P27
    "Explain what Big O notation measures and give two examples.",            # P28
    "Write a Python function that counts the vowels in a string.",           # P29
    "Explain the difference between SQL and NoSQL databases.",               # P30
    "Write a Python function that checks if two strings are anagrams.",      # P31
    "Explain what a REST API endpoint is and how it handles requests.",      # P32
    # --- Reasoning/logic (P33-P40) ---
    "You have two ropes that each take exactly one hour to burn, but burn unevenly. How can you measure 45 minutes?",  # P33
    "Three boxes are labeled Apples, Oranges, and Mixed, but all labels are wrong. You pick one fruit from one box. How do you label all three correctly?",  # P34
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",  # P35
    "If you flip a fair coin three times, what is the probability of getting exactly two heads?",  # P36
    "You are in a room with two doors and two guards. One always lies and one always tells the truth. What single question can you ask to find the safe door?",  # P37
    "A snail climbs 3 feet up a wall each day and slides back 2 feet each night. If the wall is 10 feet tall, how many days does it take to reach the top?",  # P38
    "What is the minimum number of weighings needed to find one counterfeit coin among 8 coins using a balance scale?",  # P39
    "You have 12 identical-looking coins. One is heavier or lighter than the rest. With three weighings on a balance scale, how would you find it?",  # P40
    # --- Creative/open-ended (P41-P48) ---
    "Write the opening paragraph of a science fiction story set on Mars.",    # P41
    "Describe the sound of rain on a tin roof.",                              # P42
    "Write a short fable about a tortoise and an eagle.",                     # P43
    "Describe a busy city market to someone who has never visited one.",      # P44
    "Write a haiku about a winter morning.",                                  # P45
    "Describe the feeling of solving a difficult puzzle.",                    # P46
    "Write the first paragraph of a detective story set in Tokyo.",           # P47
    "Describe the smell of a bakery early in the morning.",                   # P48
    # --- Short-answer/factual (P49-P56) ---
    "What is the chemical formula for table salt?",                           # P49
    "What year did the Berlin Wall fall?",                                    # P50
    "What is the speed of light in a vacuum, approximately?",                # P51
    "What programming language was originally developed by Guido van Rossum?",  # P52
    "What is the largest organ in the human body?",                           # P53
    "What does DNA stand for?",                                               # P54
    "What is the boiling point of water at sea level in Celsius?",           # P55
    "Name the four fundamental forces in physics.",                           # P56
    # --- List/structured output (P57-P64) ---
    "List five programming paradigms and one sentence about each.",           # P57
    "List the first ten elements of the periodic table in order.",            # P58
    "List four types of machine learning and give one example of each.",      # P59
    "List the seven continents in order of land area from largest to smallest.",  # P60
    "List five common data structures and when you would use each.",          # P61
    "List three differences between Python and JavaScript.",                  # P62
    "List the phases of the Moon in order.",                                  # P63
    "List five renewable energy sources and one advantage of each.",          # P64
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
    config = replace(spec.config, memory_limit_gb=memory_limit_gb)
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


def parse_config_keys_arg(config_keys_arg: Optional[str]) -> Optional[List[str]]:
    if config_keys_arg is None:
        return None
    keys = [key.strip() for key in config_keys_arg.split(",") if key.strip()]
    return keys or None


def select_policy_specs(
    policy_specs: List[PolicySpec],
    requested_keys: Optional[List[str]],
) -> List[PolicySpec]:
    if requested_keys is None:
        return policy_specs

    requested = set(requested_keys)
    selected = [spec for spec in policy_specs if spec.key in requested]
    missing = [key for key in requested_keys if key not in {spec.key for spec in selected}]
    if missing:
        raise ValueError(f"Unknown --config-keys entries: {missing}")
    return selected


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
    print(f"Full depth beats control: {payload['ranking_summary']['full_depth_beats_control']}")
    print(
        f"Full depth gain justifies savings loss: "
        f"{payload['ranking_summary']['full_depth_gain_justifies_savings_loss']}"
    )


def _qwen3_top1_agree(exit_layer: int) -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hard_exit_layer=exit_layer,
        entropy_threshold=-1.0,
        enable_logit_blending=True,
        blending_confidence_threshold=0.035,
        blend_alpha=0.10,
        confidence_signal="entropy",
        blend_alpha_mode="fixed",
        blend_strategy="top1_agree",
    )


def _qwen3_raw_exit(exit_layer: int) -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hard_exit_layer=exit_layer,
        entropy_threshold=-1.0,
        enable_logit_blending=False,
    )


QWEN3_POLICY_SPECS = [
    PolicySpec(
        key="L40_top1_agree",
        label="L40_top1_agree",
        description="Aggressive probe (7 layers skipped).",
        hypothesis="Tests whether a deeper skip budget exists on Qwen3-30B-A3B.",
        dynamic=True,
        config=_qwen3_top1_agree(40),
    ),
    PolicySpec(
        key="L44_top1_agree",
        label="L44_top1_agree",
        description="Moderate exit (3 layers skipped).",
        hypothesis="Tests quality at a mid-range exit point.",
        dynamic=True,
        config=_qwen3_top1_agree(44),
    ),
    PolicySpec(
        key="L45_top1_agree",
        label="L45_top1_agree",
        description="One below penultimate (2 layers skipped).",
        hypothesis="Tests the frontier boundary one step below the primary candidate.",
        dynamic=True,
        config=_qwen3_top1_agree(45),
    ),
    PolicySpec(
        key="L45_raw_exit",
        label="L45_raw_exit",
        description="One below penultimate raw-exit ablation (2 layers skipped).",
        hypothesis="Tests whether Qwen3 L45 quality depends on composition rather than plain raw exit.",
        dynamic=True,
        config=_qwen3_raw_exit(45),
    ),
    PolicySpec(
        key="L46_top1_agree",
        label="L46_top1_agree",
        description="Penultimate exit — primary Qwen3 frontier candidate.",
        hypothesis="Analogous to GPT-OSS L22: skip exactly one layer.",
        dynamic=True,
        config=_qwen3_top1_agree(46),
    ),
    PolicySpec(
        key="L46_raw_exit",
        label="L46_raw_exit",
        description="Penultimate raw-exit ablation (1 layer skipped).",
        hypothesis="Tests whether the Qwen3 penultimate frontier remains viable without composition.",
        dynamic=True,
        config=_qwen3_raw_exit(46),
    ),
    PolicySpec(
        key="L47_full_depth_reference",
        label="L47_full_depth_reference",
        description="Full-depth control reference for Qwen3-30B-A3B.",
        hypothesis="Full depth should recover exact match 1.000 as sanity gate.",
        dynamic=False,
        config=Qwen3MoeConfig(
            hard_exit_layer=47,
            entropy_threshold=-1.0,
            enable_logit_blending=False,
        ),
    ),
]


def _gpt_oss_top1_agree(exit_layer: int) -> GptOssConfig:
    return GptOssConfig(
        hard_exit_layer=exit_layer,
        entropy_threshold=-1.0,
        enable_logit_blending=True,
        blending_confidence_threshold=0.035,
        blend_alpha=0.10,
        confidence_signal="entropy",
        blend_alpha_mode="fixed",
        blend_strategy="top1_agree",
    )


def _gpt_oss_raw_exit(exit_layer: int) -> GptOssConfig:
    return GptOssConfig(
        hard_exit_layer=exit_layer,
        entropy_threshold=-1.0,
        enable_logit_blending=False,
    )


GPT_OSS_POLICY_SPECS = [
    PolicySpec(
        key="L20_top1_agree",
        label="L20_top1_agree",
        description="Most aggressive valid exit (gate_activation_layer floor).",
        hypothesis="L20 is the earliest layer where the entropy gate can fire. Tests the mechanism at its limit.",
        dynamic=True,
        config=_gpt_oss_top1_agree(20),
    ),
    PolicySpec(
        key="L21_top1_agree",
        label="L21_top1_agree",
        description="One layer below current best. Fills the L20-L22 gap.",
        hypothesis="If L21 ~ L22, the frontier is a stable band. If L21 << L22, the frontier is narrow.",
        dynamic=True,
        config=_gpt_oss_top1_agree(21),
    ),
    PolicySpec(
        key="L21_raw_exit",
        label="L21_raw_exit",
        description="One layer below current best raw-exit ablation.",
        hypothesis="Tests whether GPT-OSS L21 quality depends on composition rather than plain raw exit.",
        dynamic=True,
        config=_gpt_oss_raw_exit(21),
    ),
    PolicySpec(
        key="L22_top1_agree",
        label="L22_top1_agree",
        description="Current layer-22 frontier control.",
        hypothesis="Agreement-aware blend at layer 22 is the current best adaptive-quality frontier.",
        dynamic=True,
        config=_gpt_oss_top1_agree(22),
    ),
    PolicySpec(
        key="L22_raw_exit",
        label="L22_raw_exit",
        description="Layer-22 raw-exit ablation.",
        hypothesis="Tests whether the GPT-OSS penultimate frontier remains viable without composition.",
        dynamic=True,
        config=_gpt_oss_raw_exit(22),
    ),
    PolicySpec(
        key="L23_full_depth_reference",
        label="L23_full_depth_reference",
        description="Full-depth control reference (N=15 update).",
        hypothesis="Moving to the final layer should recover full-depth quality but remove adaptive savings.",
        dynamic=False,
        config=GptOssConfig(
            hard_exit_layer=23,
            entropy_threshold=-1.0,
            enable_logit_blending=False,
        ),
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-type", choices=["gpt_oss", "qwen3_moe"], default="gpt_oss")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--consistency-tokens", type=int, default=DEFAULT_CONSISTENCY_TOKENS)
    parser.add_argument("--memory-limit-gb", type=float, default=30.0)
    parser.add_argument(
        "--config-keys",
        help="Optional comma-separated config keys to run, e.g. L21_raw_exit,L22_raw_exit",
    )
    args = parser.parse_args()

    if args.model_type == "qwen3_moe":
        model, tokenizer, resolved_model_path = load_mlx_model(args.model, lazy=True)
        policy_specs = QWEN3_POLICY_SPECS
        control_key = "L46_top1_agree"
        full_depth_layer = 47
    else:
        model, tokenizer, resolved_model_path = load_resolved_mlx_model(args.model, lazy=True)
        policy_specs = GPT_OSS_POLICY_SPECS
        control_key = "L22_top1_agree"
        full_depth_layer = 23

    requested_config_keys = parse_config_keys_arg(args.config_keys)
    policy_specs = select_policy_specs(policy_specs, requested_config_keys)

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
                    "avg_layers_saved": max(float(engine.num_layers) - stats.get("avg_layers", float(engine.num_layers)), 0.0),
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
                "enable_logit_blending": spec.config.enable_logit_blending,
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

    control = next((item for item in config_results if item["key"] == control_key), None)
    for item in config_results:
        if control is None:
            item["exit_layer_classification"] = None
            item["delta_vs_control"] = None
            continue

        item["exit_layer_classification"] = classify_exit_shift(
            item["aggregate_excluding_warmup"],
            control,
        )
        item["delta_vs_control"] = {
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
    full_depth_candidates = [
        item for item in config_results if item["settings"]["hard_exit_layer"] == full_depth_layer
    ]
    best_full_depth = (
        sorted(full_depth_candidates, key=quality_sort_key, reverse=True)[0]
        if full_depth_candidates
        else None
    )
    ranking_summary = {
        "best_config": ranked[0]["label"],
        "best_full_depth_config": best_full_depth["label"] if best_full_depth is not None else None,
        "full_depth_beats_control": (
            best_full_depth["aggregate_excluding_warmup"]["avg_exact_match_rate"]
            > control["aggregate_excluding_warmup"]["avg_exact_match_rate"]
            if best_full_depth is not None and control is not None
            else None
        ),
        "full_depth_improves_p3": (
            best_full_depth["p3_summary"]["exact_match_rate"]
            > control["p3_summary"]["exact_match_rate"]
            if best_full_depth is not None and control is not None
            else None
        ),
        "full_depth_gain_justifies_savings_loss": (
            best_full_depth["aggregate_excluding_warmup"]["avg_exact_match_rate"]
            > control["aggregate_excluding_warmup"]["avg_exact_match_rate"]
            and best_full_depth["aggregate_excluding_warmup"]["avg_avg_layers_saved"] >= MEANINGFUL_SAVINGS_THRESHOLD
            if best_full_depth is not None and control is not None
            else None
        ),
        "quality_ranking": [item["label"] for item in ranked],
    }

    payload = {
        "experiment": "transcender_exit_layer_benchmark",
        "model_type": args.model_type,
        "resolved_model_path": resolved_model_path,
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "consistency_tokens": args.consistency_tokens,
            "reasoning_effort": REASONING_EFFORT,
            "warmup_discard_prompt_id": prompt_defs[WARMUP_PROMPT_INDEX]["prompt_id"],
            "config_keys_filter": requested_config_keys,
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
