"""
GPU reproduction of Track A penultimate-layer frontier.

Purpose:
  Test whether the penultimate-layer quality cliff observed on MLX
  reproduces on a GPU runtime using HuggingFace Transformers.

  This is an external-validity experiment, not a speedup benchmark.
  The primary question: is the frontier a model effect or a runtime effect?

Target model:
  Qwen3-30B-A3B (Qwen/Qwen3-30B-A3B) — 48 layers, 128 experts, top-8.
  This model was already tested in Track A on MLX (L46 top1_agree: 0.837 EM).

Method:
  1. Run full-depth greedy decode (48 tokens) as baseline.
  2. Extract penultimate-layer (L46) hidden states, project through LM head
     to get early logits, and apply top1_agree composition.
  3. Extract L45 hidden states similarly (one-layer-earlier cliff test).
  4. Compare: does the quality cliff reproduce?

Runtime:
  HuggingFace Transformers on GPU. Not vLLM or TRT-LLM.
  This is a scoped first step — clearly labeled as such.

Usage:
  python transcender_gpu_reproduction.py \
    --model Qwen/Qwen3-30B-A3B \
    --quantize 4bit \
    --prompt-limit 3 \
    --max-new-tokens 16 \
    --output artifacts/track_a_gpu/qwen3_gpu_smoke.json

  python transcender_gpu_reproduction.py \
    --model Qwen/Qwen3-30B-A3B \
    --quantize 4bit \
    --output artifacts/track_a_gpu/qwen3_gpu_reproduction.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Prompts — same suite as MLX Track A.  P1 is warmup (excluded from scoring).
# ---------------------------------------------------------------------------
PROMPTS = [
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
    # --- Expanded expository (P17-P24) ---
    "Explain how a transformer neural network processes a sentence.",         # P17
    "Describe the greenhouse effect and why it matters for climate.",         # P18
    "Explain what an API is to someone with no programming background.",      # P19
    "Describe how vaccines train the immune system.",                         # P20
    "Explain the difference between machine learning and traditional programming.", # P21
    "Describe how a blockchain works at a high level.",                       # P22
    "Explain what inflation is and how it affects purchasing power.",         # P23
    "Describe the water cycle in simple terms.",                              # P24
    # --- Code/technical (P25-P32) ---
    "Write a Python function that reverses a linked list.",                   # P25
    "Explain the difference between a stack and a queue.",                    # P26
    "Describe what a REST API is and give an example endpoint.",              # P27
    "Explain what Big O notation measures and give two examples.",            # P28
    "Write a Python function that counts the vowels in a string.",           # P29
    "Explain the difference between SQL and NoSQL databases.",               # P30
    "Write a Python function that checks if two strings are anagrams.",      # P31
    "Explain what Docker containers are and why developers use them.",        # P32
    # --- Reasoning/logic (P33-P40) ---
    "You have two ropes that each take exactly one hour to burn, but burn unevenly. How can you measure 45 minutes?", # P33
    "Three boxes are labeled Apples, Oranges, and Mixed, but all labels are wrong. You pick one fruit from one box. How do you label all three correctly?", # P34
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", # P35
    "If you flip a fair coin three times, what is the probability of getting exactly two heads?", # P36
    "You are in a room with two doors and two guards. One always lies and one always tells the truth. What single question can you ask to find the safe door?", # P37
    "A snail climbs 3 feet up a wall each day but slides back 2 feet at night. If the wall is 30 feet high, how many days to reach the top?", # P38
    "There are 100 lockers in a hallway, all closed. 100 students walk by; student k toggles every k-th locker. Which lockers are open at the end?", # P39
    "You have 8 identical-looking balls but one is heavier. Using a balance scale, what is the minimum number of weighings to find the heavy ball?", # P40
    # --- Creative/open-ended (P41-P48) ---
    "Write a haiku about a thunderstorm.",                                    # P41
    "Describe an alien planet where the oceans are made of mercury.",         # P42
    "Write a short fable about a tortoise and an eagle.",                     # P43
    "Describe a busy city market to someone who has never visited one.",      # P44
    "Write a limerick about a programmer who debugs in their sleep.",         # P45
    "Describe what it would feel like to walk on the Moon.",                  # P46
    "Write the first paragraph of a detective story set in Tokyo.",           # P47
    "Describe the sound of a rainforest at dawn.",                            # P48
    # --- Short-answer/factual (P49-P56) ---
    "What is the chemical symbol for gold?",                                  # P49
    "What year did the Berlin Wall fall?",                                    # P50
    "What is the speed of light in a vacuum, approximately?",                # P51
    "What programming language was originally developed by Guido van Rossum?", # P52
    "What is the largest organ in the human body?",                           # P53
    "What does DNA stand for?",                                               # P54
    "What is the boiling point of water at sea level in Celsius?",           # P55
    "Name the four fundamental forces in physics.",                           # P56
    # --- List/structured output (P57-P64) ---
    "List five programming paradigms and one sentence about each.",           # P57
    "List the layers of the OSI model from bottom to top.",                   # P58
    "List three differences between Python and JavaScript.",                  # P59
    "List the first ten elements of the periodic table in order.",            # P60
    "Outline the steps to make a peanut butter and jelly sandwich.",          # P61
    "List four types of machine learning and a one-sentence description of each.", # P62
    "List the phases of the Moon in order.",                                  # P63
    "List five renewable energy sources and one advantage of each.",          # P64
]

SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_BLEND_ALPHA = 0.10
WARMUP_INDEX = 0  # P1


@dataclass
class LayerResult:
    """Result of evaluating one prompt at one exit layer."""
    prompt_id: str
    prompt_text: str
    exit_layer: int
    full_depth_tokens: List[int]
    exit_tokens: List[int]
    composed_tokens: List[int]
    exact_match_rate: float
    prefix_match_tokens: int
    first_divergence_position: int
    top1_agreement_rate: float
    generation_time_s: float


def load_model_and_tokenizer(model_name: str, quantize: str, device: str):
    """Load model with optional quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "output_hidden_states": True,
    }

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    return model, tokenizer


def build_messages(
    prompt_text: str,
    system_prompt: str,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]


def render_prompt(
    tokenizer,
    prompt_text: str,
    system_prompt: str,
) -> torch.Tensor:
    """Render a chat prompt in the closest available tokenizer-native form."""
    messages = build_messages(prompt_text=prompt_text, system_prompt=system_prompt)

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="medium",
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return tokenizer(rendered, return_tensors="pt")["input_ids"]

    return tokenizer(
        f"{system_prompt}\n\n{prompt_text}",
        return_tensors="pt",
    )["input_ids"]


def get_lm_head(model) -> torch.nn.Module:
    """Extract the language model head for projecting hidden states to logits."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise AttributeError(
        f"Cannot find lm_head on {type(model).__name__}. "
        "Manual adaptation needed for this model architecture."
    )


def greedy_decode_with_hidden_states(
    model,
    tokenizer,
    prompt_text: str,
    system_prompt: str,
    max_new_tokens: int,
    exit_layers: List[int],
) -> Dict[str, Any]:
    """
    Greedy-decode max_new_tokens, collecting hidden states at specified layers.

    Returns:
        {
            "full_depth_ids": List[int],
            "hidden_states_per_layer": {layer_idx: List[Tensor]},
            "full_logits": List[Tensor],
            "elapsed_s": float,
        }
    """
    prompt_ids = render_prompt(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        system_prompt=system_prompt,
    )

    device = next(model.parameters()).device
    input_ids = prompt_ids.to(device)

    generated_ids: List[int] = []
    hidden_per_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in exit_layers}
    full_logits_list: List[torch.Tensor] = []

    t0 = time.perf_counter()

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(input_ids, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]  # (1, vocab)
            full_logits_list.append(logits.cpu())

            # Collect hidden states at exit layers
            # outputs.hidden_states is a tuple of (n_layers+1) tensors
            # Index 0 = embedding, index i = output of layer i-1
            for layer_idx in exit_layers:
                # hidden_states[layer_idx + 1] = output of layer layer_idx
                hs = outputs.hidden_states[layer_idx + 1][:, -1, :]  # (1, hidden)
                hidden_per_layer[layer_idx].append(hs.cpu())

            # Greedy select
            next_token = logits.argmax(dim=-1)  # (1,)
            generated_ids.append(next_token.item())

            # Append for next step
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    elapsed = time.perf_counter() - t0

    return {
        "full_depth_ids": generated_ids,
        "hidden_states_per_layer": hidden_per_layer,
        "full_logits": full_logits_list,
        "elapsed_s": elapsed,
    }


def select_prompts(prompt_offset: int = 0, prompt_limit: Optional[int] = None):
    """Return [(prompt_index, prompt_text), ...] for the requested slice."""
    if prompt_offset < 0:
        raise ValueError("--prompt-offset must be >= 0")
    if prompt_limit is not None and prompt_limit <= 0:
        raise ValueError("--prompt-limit must be > 0 when provided")

    if prompt_limit is None:
        selected = PROMPTS[prompt_offset:]
    else:
        selected = PROMPTS[prompt_offset: prompt_offset + prompt_limit]

    if not selected:
        raise ValueError(
            f"No prompts selected with offset={prompt_offset}, limit={prompt_limit}"
        )

    return list(enumerate(selected, start=prompt_offset))


def compose_top1_agree(
    full_logits: List[torch.Tensor],
    exit_logits: List[torch.Tensor],
    blend_alpha: float = 0.10,
) -> List[int]:
    """
    Apply top1_agree composition: blend logits only when shallow and deep
    agree on top-1 token identity; otherwise use full-depth logits.

    Returns the composed token sequence.
    """
    composed_ids: List[int] = []
    n = min(len(full_logits), len(exit_logits))

    for i in range(n):
        full_l = full_logits[i].float()
        exit_l = exit_logits[i].float()

        full_top1 = full_l.argmax(dim=-1).item()
        exit_top1 = exit_l.argmax(dim=-1).item()

        if full_top1 == exit_top1:
            # Blend
            blended = (1 - blend_alpha) * full_l + blend_alpha * exit_l
            composed_ids.append(blended.argmax(dim=-1).item())
        else:
            # Fall back to full depth
            composed_ids.append(full_top1)

    return composed_ids


def compute_metrics(
    full_ids: List[int],
    composed_ids: List[int],
    full_logits: List[torch.Tensor],
    exit_logits: List[torch.Tensor],
) -> Dict[str, Any]:
    """Compute exact match, prefix match, divergence, and agreement metrics."""
    n = min(len(full_ids), len(composed_ids))

    # Exact match rate
    matches = sum(1 for i in range(n) if full_ids[i] == composed_ids[i])
    exact_match_rate = matches / n if n > 0 else 0.0

    # Prefix match
    prefix = 0
    for i in range(n):
        if full_ids[i] == composed_ids[i]:
            prefix += 1
        else:
            break
    else:
        prefix = n

    # First divergence
    first_div = prefix + 1 if prefix < n else n + 1

    # Top-1 agreement between full and exit logits
    agreements = 0
    m = min(len(full_logits), len(exit_logits))
    for i in range(m):
        if full_logits[i].argmax(dim=-1).item() == exit_logits[i].argmax(dim=-1).item():
            agreements += 1
    agreement_rate = agreements / m if m > 0 else 0.0

    return {
        "exact_match_rate": exact_match_rate,
        "prefix_match_tokens": prefix,
        "first_divergence_position": first_div,
        "top1_agreement_rate": agreement_rate,
        "total_tokens": n,
    }


def run_benchmark(
    model,
    tokenizer,
    exit_layers: List[int],
    output_path: str,
    model_name: str,
    num_layers: int,
    prompt_offset: int = 0,
    prompt_limit: Optional[int] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    blend_alpha: float = DEFAULT_BLEND_ALPHA,
):
    """Run the full benchmark and write results."""
    lm_head = get_lm_head(model)
    selected_prompts = select_prompts(
        prompt_offset=prompt_offset,
        prompt_limit=prompt_limit,
    )
    results = {
        "experiment": "transcender_gpu_reproduction",
        "model": model_name,
        "num_layers": num_layers,
        "exit_layers_tested": exit_layers,
        "runtime": "huggingface_transformers_gpu",
        "max_new_tokens": max_new_tokens,
        "blend_alpha": blend_alpha,
        "warmup_prompt": "P1",
        "prompt_offset": prompt_offset,
        "prompt_limit": len(selected_prompts),
        "notes": (
            "GPU reproduction of MLX Track A. "
            "Uses output_hidden_states to extract intermediate logits. "
            "This is a HuggingFace Transformers path, not vLLM/TRT-LLM."
        ),
        "prompt_results": [],
        "aggregates": {},
    }

    for prompt_index, prompt_text in selected_prompts:
        prompt_id = f"P{prompt_index + 1}"
        is_warmup = prompt_index == WARMUP_INDEX

        print(f"\n{'[WARMUP] ' if is_warmup else ''}Processing {prompt_id}: "
              f"{prompt_text[:60]}...")

        decode_result = greedy_decode_with_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            system_prompt=SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
            exit_layers=exit_layers,
        )

        full_ids = decode_result["full_depth_ids"]
        elapsed = decode_result["elapsed_s"]
        tps = len(full_ids) / elapsed if elapsed > 0 else 0.0

        prompt_result: Dict[str, Any] = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "is_warmup": is_warmup,
            "full_depth_tokens": len(full_ids),
            "generation_time_s": round(elapsed, 4),
            "generation_tps": round(tps, 2),
            "layer_results": {},
        }

        # For each exit layer, project hidden states through LM head
        for layer_idx in exit_layers:
            hidden_states = decode_result["hidden_states_per_layer"][layer_idx]
            exit_logits = []

            with torch.no_grad():
                for hs in hidden_states:
                    # Project hidden state through LM head to get logits
                    logit = lm_head(hs.to(next(lm_head.parameters()).device))
                    exit_logits.append(logit.cpu())

            # Compose with top1_agree
            composed_ids = compose_top1_agree(
                decode_result["full_logits"],
                exit_logits,
                blend_alpha=blend_alpha,
            )

            metrics = compute_metrics(
                full_ids, composed_ids,
                decode_result["full_logits"], exit_logits,
            )

            layer_label = f"L{layer_idx}"
            prompt_result["layer_results"][layer_label] = {
                "exit_layer": layer_idx,
                "exact_match_rate": round(metrics["exact_match_rate"], 6),
                "prefix_match_tokens": metrics["prefix_match_tokens"],
                "first_divergence_position": metrics["first_divergence_position"],
                "top1_agreement_rate": round(metrics["top1_agreement_rate"], 6),
            }

            status = "PASS" if metrics["exact_match_rate"] == 1.0 else "FAIL"
            print(f"  {layer_label} top1_agree: EM={metrics['exact_match_rate']:.3f} "
                  f"prefix={metrics['prefix_match_tokens']} "
                  f"agree={metrics['top1_agreement_rate']:.3f} [{status}]")

        results["prompt_results"].append(prompt_result)

    # ---- Compute aggregates (excluding warmup) ----
    scored = [r for r in results["prompt_results"] if not r["is_warmup"]]
    aggregate_rows = scored if scored else results["prompt_results"]
    warmup_excluded = bool(scored)
    results["aggregates_scope"] = (
        "selected prompts excluding warmup"
        if warmup_excluded
        else "selected prompts (warmup included because no scored prompts remain)"
    )

    for layer_idx in exit_layers:
        layer_label = f"L{layer_idx}"
        ems = [r["layer_results"][layer_label]["exact_match_rate"] for r in aggregate_rows]
        agreements = [r["layer_results"][layer_label]["top1_agreement_rate"] for r in aggregate_rows]
        prefixes = [r["layer_results"][layer_label]["prefix_match_tokens"] for r in aggregate_rows]
        perfect = sum(1 for e in ems if e == 1.0)

        results["aggregates"][f"{layer_label}_top1_agree"] = {
            "exit_layer": layer_idx,
            "avg_exact_match": round(sum(ems) / len(ems), 6),
            "perfect_count": perfect,
            "total_scored": len(aggregate_rows),
            "avg_top1_agreement": round(sum(agreements) / len(agreements), 6),
            "avg_prefix_match": round(sum(prefixes) / len(prefixes), 2),
        }

    # Full-depth reference (trivially 1.0)
    full_tps = [r["generation_tps"] for r in aggregate_rows]
    results["aggregates"]["full_depth"] = {
        "exit_layer": num_layers - 1,
        "avg_exact_match": 1.0,
        "perfect_count": len(aggregate_rows),
        "total_scored": len(aggregate_rows),
        "avg_generation_tps": round(sum(full_tps) / len(full_tps), 2),
    }

    # ---- MLX comparison reference ----
    results["mlx_reference"] = {
        "L46_top1_agree": {"exact_match": 0.837, "perfect": 36, "total": 63},
        "L45_top1_agree": {"exact_match": 0.463, "perfect": 6, "total": 63},
        "note": "MLX Track A results on Qwen3-30B-A3B for direct comparison.",
    }

    # ---- Write output ----
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results written to {out_path}")
    print(f"\nAggregates ({results['aggregates_scope']}, N={len(aggregate_rows)}):")
    for key, agg in results["aggregates"].items():
        if key == "full_depth":
            print(f"  Full depth: EM=1.000 (baseline)")
        else:
            print(f"  {key}: EM={agg['avg_exact_match']:.3f} "
                  f"({agg['perfect_count']}/{agg['total_scored']} perfect) "
                  f"agree={agg['avg_top1_agreement']:.3f}")

    print(f"\nMLX reference (for comparison):")
    print(f"  L46 top1_agree: EM=0.837 (36/63 perfect)")
    print(f"  L45 top1_agree: EM=0.463 (6/63 perfect)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="GPU reproduction of Transcender Track A penultimate-layer frontier"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--quantize", type=str, choices=["none", "4bit", "8bit"], default="4bit",
        help="Quantization mode (4bit recommended for 48GB GPU)"
    )
    parser.add_argument(
        "--exit-layers", type=int, nargs="+", default=[45, 46],
        help="Exit layers to test (default: 45 46 for penultimate and penultimate-1)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=48,
        help="Total layers in model (default: 48 for Qwen3-30B-A3B)"
    )
    parser.add_argument(
        "--output", type=str,
        default="artifacts/track_a_gpu/qwen3_gpu_reproduction.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--prompt-offset", type=int, default=0,
        help="Zero-based prompt offset into the fixed suite (default: 0)"
    )
    parser.add_argument(
        "--prompt-limit", type=int, default=0,
        help="Number of prompts to run. 0 means the full fixed suite."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Decode length per prompt (default: {DEFAULT_MAX_NEW_TOKENS})"
    )
    parser.add_argument(
        "--blend-alpha", type=float, default=DEFAULT_BLEND_ALPHA,
        help=f"Blend weight for top1_agree composition (default: {DEFAULT_BLEND_ALPHA})"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)"
    )
    args = parser.parse_args()

    print(f"Loading {args.model} (quantize={args.quantize})...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.quantize, args.device
    )
    print(f"Model loaded. Testing exit layers: {args.exit_layers}")

    run_benchmark(
        model=model,
        tokenizer=tokenizer,
        exit_layers=args.exit_layers,
        output_path=args.output,
        model_name=args.model,
        num_layers=args.num_layers,
        prompt_offset=args.prompt_offset,
        prompt_limit=args.prompt_limit or None,
        max_new_tokens=args.max_new_tokens,
        blend_alpha=args.blend_alpha,
    )


if __name__ == "__main__":
    main()
