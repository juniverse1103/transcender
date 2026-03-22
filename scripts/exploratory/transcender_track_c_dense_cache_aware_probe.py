"""
Track C Extension — Dense Cache-Aware Continuation Probe

This script tests one narrow next step beyond the current dense selective-depth
position: can chunked cache repair reduce replay overhead enough to improve the
late-checkpoint selective-depth runtime on a dense model family?

The benchmark is intentionally scoped:
  - one model family per run (default: Llama 3.1 8B Instruct 4bit)
  - one late checkpoint
  - one existing selective-depth entropy baseline
  - one new chunk-replay selective-depth entropy variant

This is not a new continuation criterion. It isolates the replay/cache-repair
bottleneck directly.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx_lm import load as mlx_load

from transcender_track_b_cascade import (
    WARMUP_PROMPT_INDEX,
    compare_sequences,
    mean,
    preview_text,
)
from transcender_track_c_dense_family_validation import (
    LlamaLikeDenseEngine,
    LlamaLikeModelParts,
    build_prompt_pack,
)


DEFAULT_MODEL_PATH = str(
    (Path(__file__).resolve().parent.parent / "models" / "llama-3.1-8b-instruct-4bit").resolve()
)
DEFAULT_OUTPUT_PATH = str(
    (Path(__file__).resolve().parent / "transcender_track_c_llama3_8b_cache_aware_probe.json").resolve()
)


@dataclass(frozen=True)
class ModeConfig:
    key: str
    label: str
    description: str
    kind: str
    exit_layer: Optional[int] = None
    entropy_threshold: Optional[float] = None


class ChunkRepairDenseEngine(LlamaLikeDenseEngine):
    """Adds a chunked cache-repair variant on top of the existing dense engine."""

    def generate_selective_depth_chunk_repair(
        self,
        prompt_ids: List[int],
        exit_layer: int,
        max_new_tokens: int = 48,
        entropy_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        mx.clear_cache()
        mx.reset_peak_memory()

        deep_layers_per_token = self.num_layers - (exit_layer + 1)
        if deep_layers_per_token <= 0:
            raise ValueError("exit_layer must be before the final layer")

        t0 = time.perf_counter()
        prefill = self._prefill_full_depth(prompt_ids)
        cache = prefill["cache"]
        token_id = prefill["first_token_id"]

        generated_ids: List[int] = [token_id]
        ttft = time.perf_counter() - t0

        decision_tokens = 0
        early_accepted_tokens = 0
        continued_tokens = 0
        replayed_accepted_tokens = 0
        repair_events = 0
        pending_hidden: List[mx.array] = []

        if token_id not in self.eos_ids:
            for _ in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                hidden = self.parts.embed(token_input)
                hidden = self._run_layers(hidden, cache, 0, exit_layer + 1)
                early_logits = self.parts.compute_logits(hidden)
                mx.eval(early_logits)
                probe = self._confidence_probe(early_logits)
                decision_tokens += 1

                if float(probe["normalized_entropy"]) <= entropy_threshold:
                    token_id = int(probe["top1_id"])
                    generated_ids.append(token_id)
                    early_accepted_tokens += 1
                    pending_hidden.append(hidden)
                    if token_id in self.eos_ids:
                        break
                    continue

                if pending_hidden:
                    repair_events += 1
                    replay_chunk = mx.concatenate(pending_hidden + [hidden], axis=1)
                    replayed_accepted_tokens += sum(int(h.shape[1]) for h in pending_hidden)
                    pending_hidden.clear()
                    deep_hidden = self._run_layers(
                        replay_chunk,
                        cache,
                        exit_layer + 1,
                        self.num_layers,
                    )
                    full_logits = self.parts.compute_logits(deep_hidden)
                    mx.eval(full_logits)
                    token_id = int(mx.argmax(full_logits[0, -1], axis=-1).item())
                else:
                    hidden = self._run_layers(hidden, cache, exit_layer + 1, self.num_layers)
                    full_logits = self.parts.compute_logits(hidden)
                    mx.eval(full_logits)
                    token_id = int(mx.argmax(full_logits[0, -1], axis=-1).item())

                generated_ids.append(token_id)
                continued_tokens += 1
                if token_id in self.eos_ids:
                    break

        elapsed = time.perf_counter() - t0
        completion_tokens = len(generated_ids)
        realized_skipped_tokens = max(early_accepted_tokens - replayed_accepted_tokens, 0)
        acceptance_rate = early_accepted_tokens / max(decision_tokens, 1)
        continuation_rate = continued_tokens / max(decision_tokens, 1)
        realized_skip_rate = realized_skipped_tokens / max(decision_tokens, 1)

        total_layer_passes = (
            decision_tokens * (exit_layer + 1)
            + (continued_tokens + replayed_accepted_tokens) * deep_layers_per_token
        )
        avg_realized_depth = (
            total_layer_passes / max(decision_tokens, 1) if decision_tokens > 0 else float(self.num_layers)
        )
        avg_layers_saved = (
            (realized_skipped_tokens * deep_layers_per_token)
            / max(decision_tokens * self.num_layers, 1)
            if decision_tokens > 0 else 0.0
        )

        return {
            "generated_ids": generated_ids,
            "output_text": self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": completion_tokens,
            "ttft_s": ttft,
            "elapsed_s": elapsed,
            "generation_tps": max(completion_tokens - 1, 0) / max(elapsed - ttft, 1e-6),
            "peak_memory_gb": mx.get_peak_memory() / (1024**3),
            "exit_layer": exit_layer,
            "prefill_strategy": "full_depth_prompt_prefill",
            "repair_strategy": "chunk_repair_with_current_token",
            "decision_tokens": decision_tokens,
            "deep_layers_per_token": deep_layers_per_token,
            "early_accepted_tokens": early_accepted_tokens,
            "continued_tokens": continued_tokens,
            "replayed_accepted_tokens": replayed_accepted_tokens,
            "repair_events": repair_events,
            "realized_skipped_tokens": realized_skipped_tokens,
            "acceptance_rate": acceptance_rate,
            "continuation_rate": continuation_rate,
            "realized_skip_rate": realized_skip_rate,
            "avg_realized_depth": avg_realized_depth,
            "avg_layers_saved": avg_layers_saved,
            "avg_deep_layers_skipped": realized_skip_rate * deep_layers_per_token,
            "entropy_threshold": entropy_threshold,
        }


def build_modes(late_layer: int, entropy_threshold: float, final_layer: int) -> List[ModeConfig]:
    return [
        ModeConfig(
            key=f"full_depth_L{final_layer}",
            label=f"Full Depth (L{final_layer})",
            description="Full-depth dense baseline",
            kind="full_depth",
        ),
        ModeConfig(
            key=f"fixed_exit_L{late_layer}",
            label=f"Fixed Exit (L{late_layer})",
            description=f"Fixed late exit at layer {late_layer}",
            kind="fixed_exit",
            exit_layer=late_layer,
        ),
        ModeConfig(
            key=f"top1_agree_compute_both_L{late_layer}",
            label=f"top1_agree compute-both (L{late_layer})",
            description="Agreement-aware compute-both quality-control baseline",
            kind="top1_agree_compute_both",
            exit_layer=late_layer,
        ),
        ModeConfig(
            key=f"selective_depth_entropy_L{late_layer}",
            label=f"Selective Depth entropy (L{late_layer})",
            description="Reference selective-depth with one-by-one replay",
            kind="selective_entropy_standard",
            exit_layer=late_layer,
            entropy_threshold=entropy_threshold,
        ),
        ModeConfig(
            key=f"selective_depth_entropy_chunk_repair_L{late_layer}",
            label=f"Selective Depth entropy + chunk repair (L{late_layer})",
            description="Selective-depth with chunked cache repair",
            kind="selective_entropy_chunk_repair",
            exit_layer=late_layer,
            entropy_threshold=entropy_threshold,
        ),
    ]


def run_mode(
    engine: ChunkRepairDenseEngine,
    mode: ModeConfig,
    prompt_pack: List[Dict[str, Any]],
    reference_results: Optional[Dict[str, List[int]]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt_results = []

    for prompt_def in prompt_pack:
        prompt_ids = prompt_def["prompt_ids"]
        if mode.kind == "full_depth":
            stats = engine.generate_full_depth(prompt_ids=prompt_ids, max_new_tokens=max_new_tokens)
        elif mode.kind == "fixed_exit":
            stats = engine.generate_early_exit(
                prompt_ids=prompt_ids,
                exit_layer=int(mode.exit_layer),
                max_new_tokens=max_new_tokens,
            )
        elif mode.kind == "top1_agree_compute_both":
            stats = engine.generate_blended(
                prompt_ids=prompt_ids,
                exit_layer=int(mode.exit_layer),
                max_new_tokens=max_new_tokens,
                strategy="top1_agree",
            )
        elif mode.kind == "selective_entropy_standard":
            stats = engine.generate_selective_depth(
                prompt_ids=prompt_ids,
                exit_layer=int(mode.exit_layer),
                max_new_tokens=max_new_tokens,
                entropy_threshold=float(mode.entropy_threshold),
            )
        elif mode.kind == "selective_entropy_chunk_repair":
            stats = engine.generate_selective_depth_chunk_repair(
                prompt_ids=prompt_ids,
                exit_layer=int(mode.exit_layer),
                max_new_tokens=max_new_tokens,
                entropy_threshold=float(mode.entropy_threshold),
            )
        else:
            raise ValueError(f"unknown mode kind: {mode.kind}")

        if reference_results and prompt_def["prompt_id"] in reference_results:
            comparison = compare_sequences(
                stats["generated_ids"],
                reference_results[prompt_def["prompt_id"]],
            )
        else:
            comparison = {
                "exact_match_rate": 1.0,
                "prefix_match_tokens": stats["completion_tokens"],
                "first_divergence_position": None,
                "passed": True,
            }

        prompt_results.append(
            {
                "prompt_id": prompt_def["prompt_id"],
                "user_prompt": prompt_def["user"],
                **stats,
                "comparison": comparison,
                "preview": preview_text(stats["output_text"], 200),
            }
        )

    non_warmup = [r for i, r in enumerate(prompt_results) if i != WARMUP_PROMPT_INDEX]
    aggregate: Dict[str, Any] = {
        "avg_ttft_s": mean([r["ttft_s"] for r in non_warmup]),
        "avg_generation_tps": mean([r["generation_tps"] for r in non_warmup]),
        "avg_elapsed_s": mean([r["elapsed_s"] for r in non_warmup]),
        "avg_peak_memory_gb": mean([r["peak_memory_gb"] for r in non_warmup]),
        "avg_exact_match_rate": mean([r["comparison"]["exact_match_rate"] for r in non_warmup]),
        "avg_prefix_match_tokens": mean([r["comparison"]["prefix_match_tokens"] for r in non_warmup]),
        "avg_completion_tokens": mean([r["completion_tokens"] for r in non_warmup]),
    }

    if mode.kind == "top1_agree_compute_both":
        aggregate["acceptance_rate"] = mean([r.get("agreement_rate", 0.0) for r in non_warmup])
        aggregate["continuation_rate"] = 1.0 - aggregate["acceptance_rate"]
        aggregate["avg_realized_depth"] = mean([r.get("avg_layers", engine.num_layers) for r in non_warmup])
        aggregate["avg_layers_saved"] = mean([r.get("layers_saved", 0.0) for r in non_warmup])
        aggregate["realized_skip_rate"] = 0.0
    elif mode.kind.startswith("selective_"):
        decision_tokens = sum(r.get("decision_tokens", 0) for r in non_warmup)
        early_accepted_tokens = sum(r.get("early_accepted_tokens", 0) for r in non_warmup)
        continued_tokens = sum(r.get("continued_tokens", 0) for r in non_warmup)
        replayed_accepted_tokens = sum(r.get("replayed_accepted_tokens", 0) for r in non_warmup)
        realized_skipped_tokens = sum(r.get("realized_skipped_tokens", 0) for r in non_warmup)
        total_realized_depth = sum(
            r.get("avg_realized_depth", 0.0) * max(r.get("decision_tokens", 0), 0)
            for r in non_warmup
        )
        total_layers_saved = sum(
            r.get("avg_layers_saved", 0.0) * max(r.get("decision_tokens", 0), 0)
            for r in non_warmup
        )
        aggregate.update(
            {
                "acceptance_rate": early_accepted_tokens / max(decision_tokens, 1),
                "continuation_rate": continued_tokens / max(decision_tokens, 1),
                "avg_realized_depth": (
                    total_realized_depth / max(decision_tokens, 1)
                    if decision_tokens > 0 else float(engine.num_layers)
                ),
                "avg_layers_saved": (
                    total_layers_saved / max(decision_tokens, 1)
                    if decision_tokens > 0 else 0.0
                ),
                "realized_skip_rate": realized_skipped_tokens / max(decision_tokens, 1),
                "early_accepted_tokens": early_accepted_tokens,
                "continued_tokens": continued_tokens,
                "replayed_accepted_tokens": replayed_accepted_tokens,
                "realized_skipped_tokens": realized_skipped_tokens,
                "repair_events": sum(r.get("repair_events", 0) for r in non_warmup),
            }
        )
    else:
        aggregate["avg_layers_saved"] = mean([r.get("layers_saved", 0.0) for r in non_warmup])

    return {
        "key": mode.key,
        "label": mode.label,
        "description": mode.description,
        "status": "ok",
        "prompt_results": prompt_results,
        "aggregate_excluding_warmup": aggregate,
    }


def print_results(payload: Dict[str, Any]) -> None:
    print(f"\n{'=' * 96}")
    print("  Track C Extension — Dense Cache-Aware Continuation Probe")
    print(f"{'=' * 96}")
    header = (
        f"  {'Mode':<42} {'TPS':>8} {'Exact':>7} {'Saved':>7} "
        f"{'Accept':>7} {'Skip':>7} {'Replay':>7}"
    )
    print(f"\n{header}")
    print(f"  {'-' * 92}")
    for mode in payload["modes"]:
        agg = mode["aggregate_excluding_warmup"]
        print(
            f"  {mode['label']:<42} "
            f"{agg['avg_generation_tps']:>8.2f} "
            f"{agg['avg_exact_match_rate']:>7.3f} "
            f"{agg.get('avg_layers_saved', 0.0):>7.1%} "
            f"{agg.get('acceptance_rate', 0.0):>7.1%} "
            f"{agg.get('realized_skip_rate', 0.0):>7.1%} "
            f"{agg.get('replayed_accepted_tokens', 0):>7}"
        )

    baseline = next(
        m["aggregate_excluding_warmup"] for m in payload["modes"] if m["key"] == payload["baseline_key"]
    )
    print(f"\n  Baseline full-depth TPS: {baseline['avg_generation_tps']:.2f}")
    for mode in payload["modes"]:
        if not mode["key"].startswith("selective_depth_entropy"):
            continue
        agg = mode["aggregate_excluding_warmup"]
        delta = agg["avg_generation_tps"] - baseline["avg_generation_tps"]
        print(
            f"  {mode['key']}: TPS_delta={delta:+.2f}, "
            f"exact={agg['avg_exact_match_rate']:.3f}, "
            f"saved={agg.get('avg_layers_saved', 0.0):.2%}, "
            f"skip={agg.get('realized_skip_rate', 0.0):.1%}, "
            f"replayed={agg.get('replayed_accepted_tokens', 0)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track C extension — dense cache-aware continuation probe"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--family", type=str, default="llama")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--late-layer", type=int, default=29)
    parser.add_argument("--entropy-threshold", type=float, default=0.15)
    args = parser.parse_args()

    print("=" * 96)
    print("  Track C Extension — Dense Cache-Aware Continuation Probe")
    print("=" * 96)
    print(f"\n  Loading model: {args.model}")
    model, tokenizer = mlx_load(args.model)
    parts = LlamaLikeModelParts.from_loaded(model, tokenizer, args.model, args.family)
    engine = ChunkRepairDenseEngine(parts)
    print(f"  Loaded: {engine.num_layers} layers | family={args.family}")

    prompt_pack = build_prompt_pack(tokenizer, prompt_limit=args.prompt_limit)
    print(f"  Prompts: {len(prompt_pack)} | warmup index: {WARMUP_PROMPT_INDEX}")

    modes = build_modes(args.late_layer, args.entropy_threshold, engine.num_layers - 1)
    reference_results: Optional[Dict[str, List[int]]] = None
    all_modes: List[Dict[str, Any]] = []

    for mode in modes:
        print(f"\n  ── Running: {mode.label} ──")
        result = run_mode(
            engine=engine,
            mode=mode,
            prompt_pack=prompt_pack,
            reference_results=reference_results,
            max_new_tokens=args.max_new_tokens,
        )
        if mode.kind == "full_depth":
            reference_results = {
                pr["prompt_id"]: pr["generated_ids"]
                for pr in result["prompt_results"]
            }
        agg = result["aggregate_excluding_warmup"]
        print(
            f"     TPS: {agg['avg_generation_tps']:.2f} | "
            f"Exact: {agg['avg_exact_match_rate']:.3f} | "
            f"Saved: {agg.get('avg_layers_saved', 0.0):.1%} | "
            f"Skip: {agg.get('realized_skip_rate', 0.0):.1%}"
        )
        all_modes.append(result)

    payload = {
        "experiment": "transcender_track_c_dense_cache_aware_probe",
        "model": args.model,
        "family": args.family,
        "num_layers": engine.num_layers,
        "max_new_tokens": args.max_new_tokens,
        "warmup_prompt_index": WARMUP_PROMPT_INDEX,
        "late_layer": args.late_layer,
        "entropy_threshold": args.entropy_threshold,
        "baseline_key": f"full_depth_L{engine.num_layers - 1}",
        "modes": all_modes,
    }

    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n  Results saved to {args.output}")
    print_results(payload)


if __name__ == "__main__":
    main()
