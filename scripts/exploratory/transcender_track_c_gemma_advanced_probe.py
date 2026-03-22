"""
Track C Extension — Plateau-Aware Multi-Layer Consistency Probe for Gemma 3 4B-IT

This script isolates a stronger L20 selective-depth probe built on the existing
continuation-safe selective-depth runtime.

Probe design:
  - Prompt prefill remains full-depth so deep caches are valid at decode start.
  - During decode, each token runs layers 0..20 first.
  - The probe inspects the top-1 token at layers 18, 19, and 20.
  - Early acceptance requires rolling multi-layer agreement and optional L20
    confidence conditions.
  - Accepted tokens physically skip layers 21..33.
  - If a later token requires continuation, previously skipped decode tokens are
    replayed through the skipped layers to repair deep-cache state.

This is real selective-depth. Replay overhead is measured rather than ignored.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
from mlx_lm import load as mlx_load

from transcender_track_b_cascade import (
    WARMUP_PROMPT_INDEX,
    compare_sequences,
    mean,
    preview_text,
)
from transcender_track_c_gemma_selective_depth import (
    DEFAULT_MODEL_PATH,
    GemmaSelectiveDepthEngine,
    build_prompt_pack,
)

DEFAULT_OUTPUT_PATH = str(
    (
        Path(__file__).resolve().parent
        / "transcender_track_c_gemma_advanced_probe_L20_results.json"
    ).resolve()
)
OBSERVATION_LAYERS: Tuple[int, int, int] = (18, 19, 20)


@dataclass(frozen=True)
class ProbeModeConfig:
    key: str
    label: str
    description: str
    kind: str
    exit_layer: int = 20
    margin_threshold: Optional[float] = None
    entropy_threshold: Optional[float] = None


class GemmaAdvancedProbeEngine(GemmaSelectiveDepthEngine):
    """Adds a multi-layer consistency probe on top of selective-depth decode."""

    def _top1_from_logits(self, logits: mx.array) -> int:
        top1 = mx.argmax(logits[0, -1], axis=-1)
        mx.eval(top1)
        return int(top1.item())

    def _run_to_exit_with_observation_window(
        self,
        token_input: mx.array,
        cache: Any,
        *,
        exit_layer: int,
        observation_layers: Sequence[int],
    ) -> Tuple[mx.array, Dict[int, int], Dict[str, float | int]]:
        hidden = self._embed(token_input)
        observation_layer_set = set(observation_layers)
        rolling_top1: Dict[int, int] = {}
        exit_probe: Optional[Dict[str, float | int]] = None

        for i in range(exit_layer + 1):
            is_global = self._is_global_layer(i)
            global_mask, sliding_mask = self._build_masks(hidden, cache[i])
            mask = global_mask if is_global else sliding_mask
            hidden = self.layers[i](hidden, mask, cache[i])
            mx.eval(hidden)

            if i not in observation_layer_set:
                continue

            logits = self._compute_logits(hidden)
            mx.eval(logits)

            if i == exit_layer:
                exit_probe = self._confidence_probe(logits)
                rolling_top1[i] = int(exit_probe["top1_id"])
            else:
                rolling_top1[i] = self._top1_from_logits(logits)

        if exit_probe is None:
            raise RuntimeError("failed to capture exit-layer probe")

        return hidden, rolling_top1, exit_probe

    def _rolling_agreement(self, rolling_top1: Dict[int, int], observation_layers: Sequence[int]) -> bool:
        ids = [rolling_top1[layer] for layer in observation_layers]
        return all(token_id == ids[0] for token_id in ids[1:])

    def generate_multilayer_probe(
        self,
        prompt_ids: List[int],
        *,
        exit_layer: int = 20,
        observation_layers: Sequence[int] = OBSERVATION_LAYERS,
        max_new_tokens: int = 48,
        require_entropy: bool = False,
        require_margin: bool = False,
        margin_threshold: float = 0.02,
        entropy_threshold: float = 0.65,
    ) -> Dict[str, Any]:
        mx.clear_cache()
        mx.reset_peak_memory()

        deep_layers_per_token = self.num_layers - (exit_layer + 1)
        if deep_layers_per_token <= 0:
            raise ValueError("exit_layer must be before the final layer")

        prefill_t0 = mx.array(0)  # sentinel to keep local style consistent with inherited runtime
        del prefill_t0

        import time

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
        rolling_agreement_tokens = 0
        pending_hidden: List[mx.array] = []

        if token_id not in self.eos_ids:
            for _ in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                hidden, rolling_top1, exit_probe = self._run_to_exit_with_observation_window(
                    token_input,
                    cache,
                    exit_layer=exit_layer,
                    observation_layers=observation_layers,
                )

                decision_tokens += 1
                agreement = self._rolling_agreement(rolling_top1, observation_layers)
                if agreement:
                    rolling_agreement_tokens += 1

                accept = agreement
                if require_margin:
                    accept = accept and float(exit_probe["margin"]) >= margin_threshold
                if require_entropy:
                    accept = accept and float(exit_probe["normalized_entropy"]) <= entropy_threshold

                if accept:
                    token_id = int(exit_probe["top1_id"])
                    generated_ids.append(token_id)
                    early_accepted_tokens += 1
                    pending_hidden.append(hidden)
                    if token_id in self.eos_ids:
                        break
                    continue

                replayed = self._replay_pending_deep(pending_hidden, cache, exit_layer)
                replayed_accepted_tokens += replayed

                hidden = self._run_layers(hidden, cache, exit_layer + 1, self.num_layers)
                full_logits = self._compute_logits(hidden)
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
        realized_skip_rate_all_completion_tokens = realized_skipped_tokens / max(completion_tokens, 1)
        rolling_agreement_rate = rolling_agreement_tokens / max(decision_tokens, 1)

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
            "observation_layers": list(observation_layers),
            "decision_tokens": decision_tokens,
            "deep_layers_per_token": deep_layers_per_token,
            "rolling_agreement_tokens": rolling_agreement_tokens,
            "rolling_agreement_rate": rolling_agreement_rate,
            "early_accepted_tokens": early_accepted_tokens,
            "continued_tokens": continued_tokens,
            "replayed_accepted_tokens": replayed_accepted_tokens,
            "realized_skipped_tokens": realized_skipped_tokens,
            "acceptance_rate": acceptance_rate,
            "continuation_rate": continuation_rate,
            "realized_skip_rate": realized_skip_rate,
            "realized_skip_rate_all_completion_tokens": realized_skip_rate_all_completion_tokens,
            "avg_realized_depth": avg_realized_depth,
            "avg_layers_saved": avg_layers_saved,
            "avg_deep_layers_skipped": realized_skip_rate * deep_layers_per_token,
            "margin_threshold": margin_threshold if require_margin else None,
            "entropy_threshold": entropy_threshold if require_entropy else None,
            "require_margin": require_margin,
            "require_entropy": require_entropy,
        }


def build_modes(
    *,
    exit_layer: int,
    margin_threshold: float,
    entropy_threshold: float,
) -> List[ProbeModeConfig]:
    return [
        ProbeModeConfig(
            key="full_depth_L33",
            label="Full Depth (L33)",
            description="Gemma full 34-layer baseline",
            kind="full_depth",
            exit_layer=exit_layer,
        ),
        ProbeModeConfig(
            key=f"fixed_exit_L{exit_layer}",
            label=f"Fixed Exit (L{exit_layer})",
            description=f"Fixed exit at layer {exit_layer}",
            kind="fixed_exit",
            exit_layer=exit_layer,
        ),
        ProbeModeConfig(
            key=f"top1_agree_compute_both_L{exit_layer}",
            label=f"top1_agree compute-both (L{exit_layer})",
            description="Agreement-aware compute-both baseline",
            kind="top1_agree_compute_both",
            exit_layer=exit_layer,
        ),
        ProbeModeConfig(
            key=f"selective_depth_entropy_L{exit_layer}",
            label=f"Selective Depth entropy baseline (L{exit_layer})",
            description="Prior L20 entropy threshold baseline for direct comparison",
            kind="selective_entropy_baseline",
            exit_layer=exit_layer,
            entropy_threshold=entropy_threshold,
        ),
        ProbeModeConfig(
            key=f"selective_depth_multilayer_agree_L{exit_layer}",
            label=f"Multi-layer agreement only (L{exit_layer})",
            description="Rolling L18-L20 agreement only",
            kind="probe_agreement_only",
            exit_layer=exit_layer,
        ),
        ProbeModeConfig(
            key=f"selective_depth_multilayer_agree_entropy_L{exit_layer}",
            label=f"Multi-layer agreement + entropy (L{exit_layer})",
            description="Rolling L18-L20 agreement plus L20 entropy threshold",
            kind="probe_agreement_entropy",
            exit_layer=exit_layer,
            entropy_threshold=entropy_threshold,
        ),
        ProbeModeConfig(
            key=f"selective_depth_multilayer_agree_margin_entropy_L{exit_layer}",
            label=f"Multi-layer agreement + margin + entropy (L{exit_layer})",
            description="Rolling L18-L20 agreement plus L20 margin and entropy thresholds",
            kind="probe_agreement_margin_entropy",
            exit_layer=exit_layer,
            margin_threshold=margin_threshold,
            entropy_threshold=entropy_threshold,
        ),
    ]


def run_mode(
    engine: GemmaAdvancedProbeEngine,
    mode: ProbeModeConfig,
    prompt_pack: List[Dict[str, Any]],
    reference_results: Optional[Dict[str, List[int]]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt_results: List[Dict[str, Any]] = []

    for prompt_def in prompt_pack:
        prompt_id = prompt_def["prompt_id"]
        prompt_ids = prompt_def["prompt_ids"]

        if mode.kind == "full_depth":
            stats = engine.generate_full_depth(prompt_ids=prompt_ids, max_new_tokens=max_new_tokens)
        elif mode.kind == "fixed_exit":
            stats = engine.generate_early_exit(
                prompt_ids=prompt_ids,
                exit_layer=mode.exit_layer,
                max_new_tokens=max_new_tokens,
            )
        elif mode.kind == "top1_agree_compute_both":
            stats = engine.generate_blended(
                prompt_ids=prompt_ids,
                exit_layer=mode.exit_layer,
                max_new_tokens=max_new_tokens,
                blend_alpha=0.10,
                strategy="top1_agree",
            )
        elif mode.kind == "selective_entropy_baseline":
            stats = engine.generate_selective_depth(
                prompt_ids=prompt_ids,
                exit_layer=mode.exit_layer,
                max_new_tokens=max_new_tokens,
                entropy_threshold=mode.entropy_threshold,
            )
        elif mode.kind == "probe_agreement_only":
            stats = engine.generate_multilayer_probe(
                prompt_ids=prompt_ids,
                exit_layer=mode.exit_layer,
                max_new_tokens=max_new_tokens,
            )
        elif mode.kind == "probe_agreement_entropy":
            stats = engine.generate_multilayer_probe(
                prompt_ids=prompt_ids,
                exit_layer=mode.exit_layer,
                max_new_tokens=max_new_tokens,
                require_entropy=True,
                entropy_threshold=float(mode.entropy_threshold),
            )
        elif mode.kind == "probe_agreement_margin_entropy":
            stats = engine.generate_multilayer_probe(
                prompt_ids=prompt_ids,
                exit_layer=mode.exit_layer,
                max_new_tokens=max_new_tokens,
                require_entropy=True,
                require_margin=True,
                margin_threshold=float(mode.margin_threshold),
                entropy_threshold=float(mode.entropy_threshold),
            )
        else:
            raise ValueError(f"unknown mode kind: {mode.kind}")

        if reference_results and prompt_id in reference_results:
            comparison = compare_sequences(stats["generated_ids"], reference_results[prompt_id])
        else:
            comparison = {
                "exact_match_rate": 1.0,
                "prefix_match_tokens": stats["completion_tokens"],
                "first_divergence_position": None,
                "passed": True,
            }

        prompt_results.append(
            {
                "prompt_id": prompt_id,
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
        aggregate["early_accepted_tokens"] = 0
        aggregate["continued_tokens"] = 0
    elif mode.kind.startswith("selective_") or mode.kind.startswith("probe_"):
        decision_tokens = sum(r.get("decision_tokens", 0) for r in non_warmup)
        early_accepted_tokens = sum(r.get("early_accepted_tokens", 0) for r in non_warmup)
        continued_tokens = sum(r.get("continued_tokens", 0) for r in non_warmup)
        realized_skipped_tokens = sum(r.get("realized_skipped_tokens", 0) for r in non_warmup)
        replayed_accepted_tokens = sum(r.get("replayed_accepted_tokens", 0) for r in non_warmup)
        rolling_agreement_tokens = sum(r.get("rolling_agreement_tokens", 0) for r in non_warmup)
        total_completion_tokens = sum(r["completion_tokens"] for r in non_warmup)
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
                "realized_skip_rate_all_completion_tokens": (
                    realized_skipped_tokens / max(total_completion_tokens, 1)
                ),
                "early_accepted_tokens": early_accepted_tokens,
                "continued_tokens": continued_tokens,
                "decision_tokens": decision_tokens,
                "realized_skipped_tokens": realized_skipped_tokens,
                "replayed_accepted_tokens": replayed_accepted_tokens,
                "rolling_agreement_rate": rolling_agreement_tokens / max(decision_tokens, 1),
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


def annotate_against_baselines(payload: Dict[str, Any]) -> None:
    baseline = next(
        mode["aggregate_excluding_warmup"] for mode in payload["modes"] if mode["key"] == "full_depth_L33"
    )
    fixed_exit = next(
        mode["aggregate_excluding_warmup"]
        for mode in payload["modes"]
        if mode["key"] == f"fixed_exit_L{payload['exit_layer']}"
    )

    for mode in payload["modes"]:
        if mode["status"] != "ok":
            continue
        agg = mode["aggregate_excluding_warmup"]
        agg["wall_clock_tps_delta_vs_full_depth"] = (
            agg["avg_generation_tps"] - baseline["avg_generation_tps"]
        )
        agg["quality_lift_vs_fixed_exit"] = (
            agg["avg_exact_match_rate"] - fixed_exit["avg_exact_match_rate"]
        )


def print_results(payload: Dict[str, Any]) -> None:
    print(f"\n{'=' * 100}")
    print("  Track C Extension — Gemma 3 4B-IT Plateau-Aware Multi-Layer Probe (L20)")
    print(f"{'=' * 100}")

    header = (
        f"  {'Mode':<40} {'TPS':>8} {'Exact':>7} {'Saved':>7} "
        f"{'Accept':>7} {'Skip':>7} {'dTPS':>8}"
    )
    print(f"\n{header}")
    print(f"  {'-' * 94}")

    for mode in payload["modes"]:
        if mode["status"] != "ok":
            print(f"  {mode['label']:<40} {'FAIL':>8}")
            continue
        agg = mode["aggregate_excluding_warmup"]
        print(
            f"  {mode['label']:<40} "
            f"{agg['avg_generation_tps']:>8.2f} "
            f"{agg['avg_exact_match_rate']:>7.3f} "
            f"{agg.get('avg_layers_saved', 0.0):>7.1%} "
            f"{agg.get('acceptance_rate', 0.0):>7.1%} "
            f"{agg.get('realized_skip_rate', 0.0):>7.1%} "
            f"{agg.get('wall_clock_tps_delta_vs_full_depth', 0.0):>+8.2f}"
        )

    print("\n  Quality lift vs fixed exit:")
    for mode in payload["modes"]:
        if mode["status"] != "ok" or mode["key"] == f"fixed_exit_L{payload['exit_layer']}":
            continue
        agg = mode["aggregate_excluding_warmup"]
        print(
            f"  {mode['key']}: "
            f"quality_lift={agg.get('quality_lift_vs_fixed_exit', 0.0):+.3f}, "
            f"skip={agg.get('realized_skip_rate', 0.0):.1%}, "
            f"accept={agg.get('acceptance_rate', 0.0):.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track C Extension — Gemma 3 4B-IT plateau-aware multi-layer probe benchmark"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--late-layer", type=int, default=20)
    parser.add_argument("--margin-threshold", type=float, default=0.02)
    parser.add_argument("--entropy-threshold", type=float, default=0.65)
    parser.add_argument("--prompt-limit", type=int, default=None)
    args = parser.parse_args()

    print("=" * 100)
    print("  Track C Extension — Gemma 3 4B-IT Plateau-Aware Multi-Layer Probe (L20)")
    print("=" * 100)
    print(f"\n  Loading model: {args.model}")
    model, tokenizer = mlx_load(args.model)
    engine = GemmaAdvancedProbeEngine(model, tokenizer)
    print(f"  Loaded: {engine.num_layers} layers")

    prompt_pack = build_prompt_pack(tokenizer, prompt_limit=args.prompt_limit)
    print(f"  Prompts: {len(prompt_pack)} | warmup index: {WARMUP_PROMPT_INDEX}")

    modes = build_modes(
        exit_layer=args.late_layer,
        margin_threshold=args.margin_threshold,
        entropy_threshold=args.entropy_threshold,
    )

    reference_results: Optional[Dict[str, List[int]]] = None
    all_modes_data: List[Dict[str, Any]] = []

    for mode in modes:
        print(f"\n  ── Running: {mode.label} ──")
        try:
            result = run_mode(
                engine=engine,
                mode=mode,
                prompt_pack=prompt_pack,
                reference_results=reference_results,
                max_new_tokens=args.max_new_tokens,
            )
            if mode.key == "full_depth_L33":
                reference_results = {
                    prompt_result["prompt_id"]: prompt_result["generated_ids"]
                    for prompt_result in result["prompt_results"]
                }
            all_modes_data.append(result)
            agg = result["aggregate_excluding_warmup"]
            print(
                f"     TPS: {agg['avg_generation_tps']:.2f} | "
                f"Exact: {agg['avg_exact_match_rate']:.3f} | "
                f"Saved: {agg.get('avg_layers_saved', 0.0):.1%} | "
                f"Accept: {agg.get('acceptance_rate', 0.0):.1%} | "
                f"Skip: {agg.get('realized_skip_rate', 0.0):.1%}"
            )
        except Exception as exc:
            print(f"     FAILED: {exc}")
            all_modes_data.append(
                {
                    "key": mode.key,
                    "label": mode.label,
                    "description": mode.description,
                    "status": "error",
                    "error": str(exc),
                }
            )

    payload = {
        "experiment": "transcender_track_c_gemma_advanced_probe",
        "model": "gemma-3-4b-it",
        "architecture": "dense",
        "num_layers": engine.num_layers,
        "max_new_tokens": args.max_new_tokens,
        "warmup_prompt_index": WARMUP_PROMPT_INDEX,
        "prompt_prefill_strategy": "full_depth_prompt_prefill",
        "exit_layer": args.late_layer,
        "observation_layers": list(OBSERVATION_LAYERS),
        "margin_threshold": args.margin_threshold,
        "entropy_threshold": args.entropy_threshold,
        "modes": all_modes_data,
    }
    annotate_against_baselines(payload)

    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n  Results saved to {args.output}")
    print_results(payload)


if __name__ == "__main__":
    main()
