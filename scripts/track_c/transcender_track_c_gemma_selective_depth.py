"""
Track C Extension — Real Selective-Depth Runtime for Gemma 3 4B-IT

This script benchmarks a continuation-safe selective-depth runtime on Gemma.
It preserves the existing Track C benchmark script and isolates the new logic
here.

Modes:
  1. full_depth_L33
  2. fixed_exit_L<exit_layer>
  3. top1_agree_compute_both_L<exit_layer>
  4. selective_depth_margin_L<exit_layer>
  5. selective_depth_entropy_L<exit_layer>
  6. optional selective_depth_hybrid_L<exit_layer>

Selective-depth design:
  - Prompt prefill remains full-depth so deep caches are valid at decode start.
  - During autoregressive decode, each new token first runs layers 0..exit_layer.
  - If the early token is accepted, the remaining deep layers are physically skipped.
  - If a later token needs continuation, any previously skipped tokens are
    replayed through the skipped deep layers to restore deep-cache consistency before
    continuing the current token.

This means:
  - skipping is real for accepted tokens that are never replayed later
  - replay overhead is measured rather than ignored
  - correctness is prioritized over optimistic speed claims
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

mx = None
mlx_load = None

TRACK_A_DIR = Path(__file__).resolve().parent.parent / "track_a"
TRACK_B_DIR = Path(__file__).resolve().parent.parent / "track_b"
for path in (TRACK_A_DIR, TRACK_B_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from transcender_engine import build_harmony_messages
from transcender_track_b_cascade import (
    DEFAULT_PROMPT_SUITE,
    WARMUP_PROMPT_INDEX,
    apply_generic_chat_template,
    compare_sequences,
    mean,
    preview_text,
    resolve_prompt_definitions,
)
from transcender_track_c_gemma_benchmark import GemmaAdaptiveEngine

SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MODEL_PATH = str(
    (Path(__file__).resolve().parent.parent / "gemma-3-4b-it").resolve()
)
DEFAULT_OUTPUT_PATH = str(
    (Path(__file__).resolve().parent / "transcender_track_c_gemma_selective_depth_results.json").resolve()
)


def require_mlx_runtime():
    global mx, mlx_load
    if mx is None:
        import mlx.core as _mx
        from mlx_lm import load as _mlx_load

        mx = _mx
        mlx_load = _mlx_load
    return mx, mlx_load


@dataclass(frozen=True)
class ModeConfig:
    key: str
    label: str
    description: str
    kind: str
    exit_layer: int = 31
    margin_threshold: Optional[float] = None
    entropy_threshold: Optional[float] = None


class GemmaSelectiveDepthEngine(GemmaAdaptiveEngine):
    """Adds real selective-depth decode to the existing Gemma engine."""

    def _run_layers(
        self,
        hidden: mx.array,
        cache: Any,
        start_layer: int,
        end_layer: int,
    ) -> mx.array:
        for i in range(start_layer, end_layer):
            is_global = self._is_global_layer(i)
            global_mask, sliding_mask = self._build_masks(hidden, cache[i])
            mask = global_mask if is_global else sliding_mask
            hidden = self.layers[i](hidden, mask, cache[i])
        mx.eval(hidden)
        return hidden

    def _confidence_probe(self, logits: mx.array) -> Dict[str, float | int]:
        last_logits = logits[0, -1].astype(mx.float32)
        probs = mx.softmax(last_logits, axis=-1)
        entropy = -mx.sum(probs * mx.log(probs + 1e-9), axis=-1)
        normalized_entropy = entropy / math.log(int(last_logits.shape[-1]))
        top2 = mx.topk(probs, 2, axis=-1)
        # MLX returns the selected values in ascending order here.
        top2_prob = top2[..., 0]
        top1_prob = top2[..., 1]
        margin = top1_prob - top2_prob
        top1_id = mx.argmax(last_logits, axis=-1)
        mx.eval(
            probs,
            entropy,
            normalized_entropy,
            top1_prob,
            top2_prob,
            margin,
            top1_id,
        )
        return {
            "top1_id": int(top1_id.item()),
            "top1_prob": float(top1_prob.item()),
            "top2_prob": float(top2_prob.item()),
            "margin": float(margin.item()),
            "normalized_entropy": float(normalized_entropy.item()),
        }

    def _accept_early(
        self,
        probe: Dict[str, float | int],
        *,
        margin_threshold: Optional[float],
        entropy_threshold: Optional[float],
    ) -> bool:
        if margin_threshold is not None and entropy_threshold is not None:
            return (
                float(probe["margin"]) >= margin_threshold
                and float(probe["normalized_entropy"]) <= entropy_threshold
            )
        if margin_threshold is not None:
            return float(probe["margin"]) >= margin_threshold
        if entropy_threshold is not None:
            return float(probe["normalized_entropy"]) <= entropy_threshold
        raise ValueError("either margin_threshold or entropy_threshold must be set")

    def _prefill_full_depth(self, prompt_ids: List[int]) -> Dict[str, Any]:
        cache = self.model.make_cache()
        input_ids = mx.array(prompt_ids, dtype=mx.int32).reshape(1, -1)
        hidden = self._embed(input_ids)
        hidden = self._run_layers(hidden, cache, 0, self.num_layers)
        logits = self._compute_logits(hidden)
        mx.eval(logits)
        first_token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
        return {
            "cache": cache,
            "first_token_id": first_token_id,
        }

    def _replay_pending_deep(
        self,
        pending_hidden: List[mx.array],
        cache: Any,
        exit_layer: int,
    ) -> int:
        if not pending_hidden:
            return 0

        replayed = 0
        for hidden in pending_hidden:
            self._run_layers(hidden, cache, exit_layer + 1, self.num_layers)
            replayed += int(hidden.shape[1])
        pending_hidden.clear()
        return replayed

    def generate_selective_depth(
        self,
        prompt_ids: List[int],
        *,
        exit_layer: int = 31,
        max_new_tokens: int = 48,
        margin_threshold: Optional[float] = None,
        entropy_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Continuation-safe selective-depth decode.

        Prompt prefill is full-depth so decode starts with valid deep caches.
        Selective-depth is then applied to autoregressive decode tokens
        (tokens after the first emitted token).
        """
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
        pending_hidden: List[mx.array] = []

        if token_id not in self.eos_ids:
            for _ in range(max_new_tokens - 1):
                token_input = mx.array([[token_id]], dtype=mx.int32)
                hidden = self._embed(token_input)
                hidden = self._run_layers(hidden, cache, 0, exit_layer + 1)

                early_logits = self._compute_logits(hidden)
                mx.eval(early_logits)
                probe = self._confidence_probe(early_logits)

                decision_tokens += 1

                if self._accept_early(
                    probe,
                    margin_threshold=margin_threshold,
                    entropy_threshold=entropy_threshold,
                ):
                    token_id = int(probe["top1_id"])
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
            "decision_tokens": decision_tokens,
            "deep_layers_per_token": deep_layers_per_token,
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
            "margin_threshold": margin_threshold,
            "entropy_threshold": entropy_threshold,
        }


def build_prompt_pack(
    tokenizer,
    prompt_definitions: Optional[List[Dict[str, str]]] = None,
    prompt_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    pack = []
    if prompt_definitions is None:
        prompt_definitions, _ = resolve_prompt_definitions(DEFAULT_PROMPT_SUITE, None)
    if prompt_limit is not None:
        prompt_definitions = prompt_definitions[:prompt_limit]
    for prompt_def in prompt_definitions:
        prompt_id = prompt_def["prompt_id"]
        user_prompt = prompt_def["user"]
        system_prompt = prompt_def.get("system", SYSTEM_PROMPT)
        messages = build_harmony_messages(user_prompt, system_prompt)
        prompt_text, _ = apply_generic_chat_template(tokenizer, messages)
        prompt_ids = tokenizer.encode(prompt_text)
        pack.append(
            {
                "prompt_id": prompt_id,
                "user": user_prompt,
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids,
            }
        )
    return pack


def build_modes(
    *,
    late_layer: int,
    margin_threshold: float,
    entropy_threshold: float,
    include_entropy_mode: bool,
    include_hybrid_mode: bool,
) -> List[ModeConfig]:
    modes = [
        ModeConfig(
            key="full_depth_L33",
            label="Full Depth (L33)",
            description="Gemma full 34-layer baseline",
            kind="full_depth",
            exit_layer=late_layer,
        ),
        ModeConfig(
            key=f"fixed_exit_L{late_layer}",
            label=f"Fixed Exit (L{late_layer})",
            description=f"Fixed exit at layer {late_layer}",
            kind="fixed_exit",
            exit_layer=late_layer,
        ),
        ModeConfig(
            key=f"top1_agree_compute_both_L{late_layer}",
            label=f"top1_agree compute-both (L{late_layer})",
            description="Agreement-aware compute-both baseline",
            kind="top1_agree_compute_both",
            exit_layer=late_layer,
        ),
        ModeConfig(
            key=f"selective_depth_margin_L{late_layer}",
            label=f"Selective Depth margin (L{late_layer})",
            description="Real selective-depth with probability-margin gate",
            kind="selective_margin",
            exit_layer=late_layer,
            margin_threshold=margin_threshold,
        ),
    ]
    if include_entropy_mode:
        modes.append(
            ModeConfig(
                key=f"selective_depth_entropy_L{late_layer}",
                label=f"Selective Depth entropy (L{late_layer})",
                description="Real selective-depth with normalized-entropy gate",
                kind="selective_entropy",
                exit_layer=late_layer,
                entropy_threshold=entropy_threshold,
            )
        )
    if include_hybrid_mode:
        modes.append(
            ModeConfig(
                key=f"selective_depth_hybrid_L{late_layer}",
                label=f"Selective Depth hybrid (L{late_layer})",
                description="Real selective-depth with both margin and entropy gates",
                kind="selective_hybrid",
                exit_layer=late_layer,
                margin_threshold=margin_threshold,
                entropy_threshold=entropy_threshold,
            )
        )
    return modes


def run_mode(
    engine: GemmaSelectiveDepthEngine,
    mode: ModeConfig,
    prompt_pack: List[Dict[str, Any]],
    reference_results: Optional[Dict[str, List[int]]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt_results = []

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
        elif mode.kind == "selective_margin":
            stats = engine.generate_selective_depth(
                prompt_ids=prompt_ids,
                exit_layer=mode.exit_layer,
                max_new_tokens=max_new_tokens,
                margin_threshold=mode.margin_threshold,
            )
        elif mode.kind == "selective_entropy":
            stats = engine.generate_selective_depth(
                prompt_ids=prompt_ids,
                exit_layer=mode.exit_layer,
                max_new_tokens=max_new_tokens,
                entropy_threshold=mode.entropy_threshold,
            )
        elif mode.kind == "selective_hybrid":
            stats = engine.generate_selective_depth(
                prompt_ids=prompt_ids,
                exit_layer=mode.exit_layer,
                max_new_tokens=max_new_tokens,
                margin_threshold=mode.margin_threshold,
                entropy_threshold=mode.entropy_threshold,
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

    if mode.kind.startswith("selective_"):
        decision_tokens = sum(r.get("decision_tokens", 0) for r in non_warmup)
        early_accepted_tokens = sum(r.get("early_accepted_tokens", 0) for r in non_warmup)
        continued_tokens = sum(r.get("continued_tokens", 0) for r in non_warmup)
        realized_skipped_tokens = sum(r.get("realized_skipped_tokens", 0) for r in non_warmup)
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
    print(f"\n{'=' * 88}")
    print("  Track C Extension — Gemma 3 4B-IT Real Selective-Depth Benchmark")
    print(f"{'=' * 88}")

    header = (
        f"  {'Mode':<34} {'TTFT':>7} {'TPS':>8} {'Exact':>7} "
        f"{'Saved':>7} {'Accept':>7} {'Skip':>7}"
    )
    print(f"\n{header}")
    print(f"  {'-' * 82}")

    for mode in payload["modes"]:
        if mode["status"] != "ok":
            print(f"  {mode['label']:<34} {'FAIL':>7}")
            continue
        agg = mode["aggregate_excluding_warmup"]
        print(
            f"  {mode['label']:<34} "
            f"{agg['avg_ttft_s']:>7.3f} "
            f"{agg['avg_generation_tps']:>8.2f} "
            f"{agg['avg_exact_match_rate']:>7.3f} "
            f"{agg.get('avg_layers_saved', 0.0):>7.1%} "
            f"{agg.get('acceptance_rate', 0.0):>7.1%} "
            f"{agg.get('realized_skip_rate', 0.0):>7.1%}"
        )

    baseline = next(
        mode["aggregate_excluding_warmup"] for mode in payload["modes"] if mode["key"] == "full_depth_L33"
    )
    print(f"\n  Baseline full-depth TPS: {baseline['avg_generation_tps']:.2f}")

    for mode in payload["modes"]:
        if not mode["key"].startswith("selective_depth_") or mode["status"] != "ok":
            continue
        agg = mode["aggregate_excluding_warmup"]
        delta = agg["avg_generation_tps"] - baseline["avg_generation_tps"]
        print(
            f"  {mode['key']}: "
            f"accept={agg['acceptance_rate']:.1%}, "
            f"skip={agg['realized_skip_rate']:.1%}, "
            f"depth={agg['avg_realized_depth']:.2f}, "
            f"TPS_delta={delta:+.2f}, "
            f"exact={agg['avg_exact_match_rate']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track C Extension — Gemma 3 4B-IT real selective-depth benchmark"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--prompt-suite",
        choices=("expository_5", "canonical_64"),
        default=DEFAULT_PROMPT_SUITE,
        help=(
            "Prompt scope for the benchmark. "
            "'expository_5' preserves the legacy Track C behavior. "
            "'canonical_64' reuses the canonical Track A prompt definitions."
        ),
    )
    parser.add_argument(
        "--prompt-ids",
        default=None,
        help=(
            "Optional comma-separated prompt ids to filter within the selected suite, "
            "for example 'P1,P2,P3,P4,P5'. Warmup still excludes the first selected prompt."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--late-layer", type=int, default=31)
    parser.add_argument("--margin-threshold", type=float, default=0.50)
    parser.add_argument("--entropy-threshold", type=float, default=0.15)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument(
        "--skip-entropy-mode",
        action="store_true",
        help="Run only the margin selective-depth mode.",
    )
    parser.add_argument(
        "--include-hybrid-mode",
        action="store_true",
        help="Also run a stricter mode requiring both margin and entropy gates.",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("  Track C Extension — Gemma 3 4B-IT Real Selective-Depth Benchmark")
    print("=" * 88)
    print(f"\n  Loading model: {args.model}")
    require_mlx_runtime()
    model, tokenizer = mlx_load(args.model)
    engine = GemmaSelectiveDepthEngine(model, tokenizer)
    print(f"  Loaded: {engine.num_layers} layers")

    prompt_definitions, prompt_scope = resolve_prompt_definitions(
        prompt_suite=args.prompt_suite,
        prompt_ids_arg=args.prompt_ids,
    )
    prompt_pack = build_prompt_pack(
        tokenizer,
        prompt_definitions=prompt_definitions,
        prompt_limit=args.prompt_limit,
    )
    selected_prompt_ids = [prompt["prompt_id"] for prompt in prompt_pack]
    if not selected_prompt_ids:
        raise ValueError("Prompt selection resolved to zero prompts")
    prompt_scope = {
        **prompt_scope,
        "selected_prompt_ids": selected_prompt_ids,
        "selected_prompt_count": len(selected_prompt_ids),
        "warmup_prompt_id": selected_prompt_ids[WARMUP_PROMPT_INDEX],
        "scored_prompt_count_after_warmup": max(len(selected_prompt_ids) - 1, 0),
        "scope_label": (
            prompt_scope["scope_label"]
            if args.prompt_limit is None
            else f"{prompt_scope['scope_label']}__limit_{len(selected_prompt_ids)}"
        ),
    }
    print(
        "  Prompt suite:"
        f" {prompt_scope['prompt_suite']} | selected prompts: {prompt_scope['selected_prompt_count']}"
        f" | scored after warmup: {prompt_scope['scored_prompt_count_after_warmup']}"
        f" | warmup prompt: {prompt_scope['warmup_prompt_id']}"
    )

    modes = build_modes(
        late_layer=args.late_layer,
        margin_threshold=args.margin_threshold,
        entropy_threshold=args.entropy_threshold,
        include_entropy_mode=not args.skip_entropy_mode,
        include_hybrid_mode=args.include_hybrid_mode,
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
                    pr["prompt_id"]: pr["generated_ids"]
                    for pr in result["prompt_results"]
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
        "experiment": "transcender_track_c_gemma_selective_depth",
        "model": "gemma-3-4b-it",
        "architecture": "dense",
        "num_layers": engine.num_layers,
        "max_new_tokens": args.max_new_tokens,
        "warmup_prompt_index": WARMUP_PROMPT_INDEX,
        "prompt_scope": prompt_scope,
        "prompt_prefill_strategy": "full_depth_prompt_prefill",
        "exit_layer": args.late_layer,
        "margin_threshold": args.margin_threshold,
        "entropy_threshold": args.entropy_threshold,
        "include_hybrid_mode": args.include_hybrid_mode,
        "modes": all_modes_data,
    }

    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n  Results saved to {args.output}")
    print_results(payload)


if __name__ == "__main__":
    main()
