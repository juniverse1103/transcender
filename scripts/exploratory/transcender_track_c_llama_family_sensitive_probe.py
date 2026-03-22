"""
Track C Extension — Llama Family-Sensitive Late-Window Probe

This script runs one focused dense-model next step from the current locked
research position:

  - model family: Llama
  - checkpoint: L29 on a 32-layer stack
  - runtime: real selective-depth
  - continuation: chunk-repair cache restoration
  - new rule: late-window readiness over L27/L28/L29

The goal is narrow: test whether a stronger, explicit late-window readiness
criterion can preserve more quality than the plain entropy rule without
collapsing all realized skipping.
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
from transcender_track_c_dense_cache_aware_probe import ChunkRepairDenseEngine
from transcender_track_c_dense_family_validation import (
    LlamaLikeModelParts,
    build_prompt_pack,
)


DEFAULT_MODEL_PATH = str(
    (Path(__file__).resolve().parent.parent / "models" / "llama-3.1-8b-instruct-4bit").resolve()
)
DEFAULT_OUTPUT_PATH = str(
    (Path(__file__).resolve().parent / "transcender_track_c_llama3_8b_family_sensitive_probe.json").resolve()
)


@dataclass(frozen=True)
class ModeConfig:
    key: str
    label: str
    description: str
    kind: str
    exit_layer: Optional[int] = None
    entropy_threshold: Optional[float] = None
    margin_threshold: Optional[float] = None
    topk: Optional[int] = None
    min_topk_overlap: Optional[int] = None


class LlamaFamilySensitiveEngine(ChunkRepairDenseEngine):
    def _topk_indices(self, logits: mx.array, top_k: int) -> List[int]:
        scores = logits[0, -1]
        k = min(int(top_k), int(scores.shape[-1]))
        idx = mx.argsort(-scores, axis=-1)[:k]
        mx.eval(idx)
        return [int(x) for x in idx.tolist()]

    def _late_window_readiness(
        self,
        logits_l27: mx.array,
        logits_l28: mx.array,
        logits_l29: mx.array,
        *,
        margin_threshold: float,
        entropy_threshold: float,
        topk: int,
        min_topk_overlap: int,
    ) -> Dict[str, Any]:
        top1_l27 = mx.argmax(logits_l27[0, -1], axis=-1)
        top1_l28 = mx.argmax(logits_l28[0, -1], axis=-1)
        top1_l29 = mx.argmax(logits_l29[0, -1], axis=-1)
        mx.eval(top1_l27, top1_l28, top1_l29)

        top1_l27_i = int(top1_l27.item())
        top1_l28_i = int(top1_l28.item())
        top1_l29_i = int(top1_l29.item())

        probe_l29 = self._confidence_probe(logits_l29)
        topk_l28 = self._topk_indices(logits_l28, topk)
        topk_l29 = self._topk_indices(logits_l29, topk)
        topk_overlap = len(set(topk_l28) & set(topk_l29))
        topk_order_match = topk_l28 == topk_l29

        rolling_agree = top1_l27_i == top1_l28_i == top1_l29_i
        margin_ok = float(probe_l29["margin"]) >= margin_threshold
        entropy_ok = float(probe_l29["normalized_entropy"]) <= entropy_threshold
        topk_ok = topk_overlap >= min_topk_overlap and topk_order_match

        accept = rolling_agree and margin_ok and entropy_ok and topk_ok
        return {
            "accept": accept,
            "top1_l27": top1_l27_i,
            "top1_l28": top1_l28_i,
            "top1_l29": top1_l29_i,
            "rolling_agree": rolling_agree,
            "margin": float(probe_l29["margin"]),
            "normalized_entropy": float(probe_l29["normalized_entropy"]),
            "margin_ok": margin_ok,
            "entropy_ok": entropy_ok,
            "topk_overlap": topk_overlap,
            "topk_order_match": topk_order_match,
            "topk_ok": topk_ok,
            "top1_id": int(probe_l29["top1_id"]),
        }

    def generate_family_sensitive_probe(
        self,
        prompt_ids: List[int],
        *,
        exit_layer: int,
        max_new_tokens: int = 48,
        margin_threshold: float = 0.02,
        entropy_threshold: float = 0.15,
        topk: int = 4,
        min_topk_overlap: int = 3,
    ) -> Dict[str, Any]:
        mx.clear_cache()
        mx.reset_peak_memory()

        deep_layers_per_token = self.num_layers - (exit_layer + 1)
        if deep_layers_per_token <= 0:
            raise ValueError("exit_layer must be before the final layer")
        if exit_layer != 29:
            raise ValueError("family-sensitive Llama probe is scoped to exit_layer=29")

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
        rolling_agree_tokens = 0
        topk_stable_tokens = 0
        pending_hidden: List[mx.array] = []

        if token_id not in self.eos_ids:
            for _ in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                hidden = self.parts.embed(token_input)

                hidden = self._run_layers(hidden, cache, 0, 28)
                logits_l27 = self.parts.compute_logits(hidden)
                mx.eval(logits_l27)

                hidden = self._run_layers(hidden, cache, 28, 29)
                logits_l28 = self.parts.compute_logits(hidden)
                mx.eval(logits_l28)

                hidden = self._run_layers(hidden, cache, 29, 30)
                logits_l29 = self.parts.compute_logits(hidden)
                mx.eval(logits_l29)

                readiness = self._late_window_readiness(
                    logits_l27,
                    logits_l28,
                    logits_l29,
                    margin_threshold=margin_threshold,
                    entropy_threshold=entropy_threshold,
                    topk=topk,
                    min_topk_overlap=min_topk_overlap,
                )
                decision_tokens += 1
                if readiness["rolling_agree"]:
                    rolling_agree_tokens += 1
                if readiness["topk_order_match"]:
                    topk_stable_tokens += 1

                if readiness["accept"]:
                    token_id = int(readiness["top1_id"])
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
        rolling_agree_rate = rolling_agree_tokens / max(decision_tokens, 1)
        topk_stable_rate = topk_stable_tokens / max(decision_tokens, 1)

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
            "rolling_agree_rate": rolling_agree_rate,
            "topk_stable_rate": topk_stable_rate,
            "margin_threshold": margin_threshold,
            "entropy_threshold": entropy_threshold,
            "topk": topk,
            "min_topk_overlap": min_topk_overlap,
        }


def build_modes(
    late_layer: int,
    entropy_threshold: float,
    margin_threshold: float,
    topk: int,
    min_topk_overlap: int,
    final_layer: int,
) -> List[ModeConfig]:
    return [
        ModeConfig(
            key=f"full_depth_L{final_layer}",
            label=f"Full Depth (L{final_layer})",
            description="Full-depth Llama baseline",
            kind="full_depth",
        ),
        ModeConfig(
            key=f"fixed_exit_L{late_layer}",
            label=f"Fixed Exit (L{late_layer})",
            description="Fixed late exit baseline",
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
            description="Reference selective-depth with chunked cache repair",
            kind="selective_entropy_chunk_repair",
            exit_layer=late_layer,
            entropy_threshold=entropy_threshold,
        ),
        ModeConfig(
            key=f"selective_depth_family_sensitive_probe_L{late_layer}",
            label=f"Family-Sensitive Probe (L{late_layer})",
            description="Llama late-window readiness probe with chunk repair",
            kind="family_sensitive_probe",
            exit_layer=late_layer,
            entropy_threshold=entropy_threshold,
            margin_threshold=margin_threshold,
            topk=topk,
            min_topk_overlap=min_topk_overlap,
        ),
    ]


def run_mode(
    engine: LlamaFamilySensitiveEngine,
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
        elif mode.kind == "family_sensitive_probe":
            stats = engine.generate_family_sensitive_probe(
                prompt_ids=prompt_ids,
                exit_layer=int(mode.exit_layer),
                max_new_tokens=max_new_tokens,
                margin_threshold=float(mode.margin_threshold),
                entropy_threshold=float(mode.entropy_threshold),
                topk=int(mode.topk),
                min_topk_overlap=int(mode.min_topk_overlap),
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
    elif mode.kind.startswith("selective_") or mode.kind == "family_sensitive_probe":
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
                "rolling_agree_rate": mean([r.get("rolling_agree_rate", 0.0) for r in non_warmup]),
                "topk_stable_rate": mean([r.get("topk_stable_rate", 0.0) for r in non_warmup]),
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
    print(f"\n{'=' * 104}")
    print("  Track C Extension — Llama Family-Sensitive Late-Window Probe")
    print(f"{'=' * 104}")
    header = (
        f"  {'Mode':<42} {'TPS':>8} {'Exact':>7} {'Saved':>7} "
        f"{'Accept':>7} {'Skip':>7} {'Replay':>7}"
    )
    print(f"\n{header}")
    print(f"  {'-' * 100}")
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
        if not mode["key"].startswith("selective_depth_"):
            continue
        agg = mode["aggregate_excluding_warmup"]
        delta = agg["avg_generation_tps"] - baseline["avg_generation_tps"]
        extra = ""
        if mode["key"].startswith("selective_depth_family_sensitive_probe"):
            extra = (
                f", rolling_agree={agg.get('rolling_agree_rate', 0.0):.1%}, "
                f"topk_stable={agg.get('topk_stable_rate', 0.0):.1%}"
            )
        print(
            f"  {mode['key']}: TPS_delta={delta:+.2f}, "
            f"exact={agg['avg_exact_match_rate']:.3f}, "
            f"saved={agg.get('avg_layers_saved', 0.0):.2%}, "
            f"skip={agg.get('realized_skip_rate', 0.0):.1%}, "
            f"replayed={agg.get('replayed_accepted_tokens', 0)}{extra}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track C extension — Llama family-sensitive late-window probe"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--family", type=str, default="llama")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--late-layer", type=int, default=29)
    parser.add_argument("--entropy-threshold", type=float, default=0.15)
    parser.add_argument("--margin-threshold", type=float, default=0.02)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--min-topk-overlap", type=int, default=3)
    args = parser.parse_args()

    print("=" * 104)
    print("  Track C Extension — Llama Family-Sensitive Late-Window Probe")
    print("=" * 104)
    print(f"\n  Loading model: {args.model}")
    model, tokenizer = mlx_load(args.model)
    parts = LlamaLikeModelParts.from_loaded(model, tokenizer, args.model, args.family)
    engine = LlamaFamilySensitiveEngine(parts)
    print(f"  Loaded: {engine.num_layers} layers | family={args.family}")

    prompt_pack = build_prompt_pack(tokenizer, prompt_limit=args.prompt_limit)
    print(f"  Prompts: {len(prompt_pack)} | warmup index: {WARMUP_PROMPT_INDEX}")

    modes = build_modes(
        late_layer=args.late_layer,
        entropy_threshold=args.entropy_threshold,
        margin_threshold=args.margin_threshold,
        topk=args.topk,
        min_topk_overlap=args.min_topk_overlap,
        final_layer=engine.num_layers - 1,
    )

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
        "experiment": "transcender_track_c_llama_family_sensitive_probe",
        "model": args.model,
        "family": args.family,
        "num_layers": engine.num_layers,
        "max_new_tokens": args.max_new_tokens,
        "warmup_prompt_index": WARMUP_PROMPT_INDEX,
        "late_layer": args.late_layer,
        "entropy_threshold": args.entropy_threshold,
        "margin_threshold": args.margin_threshold,
        "topk": args.topk,
        "min_topk_overlap": args.min_topk_overlap,
        "baseline_key": f"full_depth_L{engine.num_layers - 1}",
        "modes": all_modes,
    }

    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n  Results saved to {args.output}")
    print_results(payload)


if __name__ == "__main__":
    main()
