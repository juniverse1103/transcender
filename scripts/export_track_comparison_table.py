#!/usr/bin/env python3
"""
Export a compact paper-facing cross-track comparison table.

This script is intentionally narrow. It reads the checked-in benchmark JSON
artifacts and emits either:
  - the current artifact-level paper scope, or
  - a matched-scope view on the shared expository subset P2-P5

The matched view does not invent new experiments. It recomputes Track A
aggregates on the same prompt subset already used by the checked-in Track B and
Track C artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_MATCHED_PROMPT_IDS = ("P2", "P3", "P4", "P5")


@dataclass(frozen=True)
class RowSpec:
    track: str
    model: str
    condition: str
    artifact_path: str
    container_key: str
    item_key: str
    comparison_space: str
    notes: str
    main_prompt_scope: str


ROW_SPECS: Sequence[RowSpec] = (
    RowSpec(
        track="A",
        model="GPT-OSS 20B",
        condition="L22 top1_agree",
        artifact_path="artifacts/track_a/transcender_exit_layer_benchmark_n63.json",
        container_key="configs",
        item_key="L22_top1_agree",
        comparison_space="same_model_native_token_space",
        notes="Canonical local MLX Track A operating point.",
        main_prompt_scope="canonical_64_prompt_suite_63_scored",
    ),
    RowSpec(
        track="A",
        model="Qwen3-30B-A3B",
        condition="L46 top1_agree",
        artifact_path="artifacts/track_a/qwen3_30b_a3b_exit_layer_benchmark_n63.json",
        container_key="configs",
        item_key="L46_top1_agree",
        comparison_space="same_model_native_token_space",
        notes="Canonical local MLX Track A cross-family validation point.",
        main_prompt_scope="canonical_64_prompt_suite_63_scored",
    ),
    RowSpec(
        track="B",
        model="Gemma 3 4B-IT -> GPT-OSS 20B",
        condition="naive cascade",
        artifact_path="artifacts/track_b/transcender_track_b_benchmark.json",
        container_key="modes",
        item_key="track_b_naive_cascade",
        comparison_space="gpt_oss_reference_token_space",
        notes="Scoped negative baseline; cross-model comparison is normalized in GPT-OSS token space.",
        main_prompt_scope="legacy_expository_subset_5_prompts_4_scored",
    ),
    RowSpec(
        track="C",
        model="Gemma 3 4B-IT",
        condition="fixed exit L31",
        artifact_path="artifacts/track_c/transcender_track_c_gemma_results.json",
        container_key="modes",
        item_key="late_exit_L31",
        comparison_space="same_model_native_token_space",
        notes="Dense boundary evidence: late fixed exit still degrades heavily.",
        main_prompt_scope="legacy_expository_subset_5_prompts_4_scored",
    ),
    RowSpec(
        track="C",
        model="Gemma 3 4B-IT",
        condition="top1_agree compute-both L31",
        artifact_path="artifacts/track_c/transcender_track_c_gemma_results.json",
        container_key="modes",
        item_key="top1_agree_L31",
        comparison_space="same_model_native_token_space",
        notes="Dense compute-both composition recovers quality but does not skip layers.",
        main_prompt_scope="legacy_expository_subset_5_prompts_4_scored",
    ),
    RowSpec(
        track="C",
        model="Gemma 3 4B-IT",
        condition="selective entropy L31",
        artifact_path="artifacts/track_c/transcender_track_c_gemma_selective_depth_results.json",
        container_key="modes",
        item_key="selective_depth_entropy_L31",
        comparison_space="same_model_native_token_space",
        notes="Real selective depth is operationally correct but still weak.",
        main_prompt_scope="legacy_expository_subset_5_prompts_4_scored",
    ),
    RowSpec(
        track="C",
        model="Llama 3.1 8B",
        condition="fixed exit L29",
        artifact_path="artifacts/dense_followup/transcender_track_c_llama3_8b_results.json",
        container_key="modes",
        item_key="fixed_exit_L29",
        comparison_space="same_model_native_token_space",
        notes="Late fixed exit remains weak on dense follow-up.",
        main_prompt_scope="legacy_expository_subset_5_prompts_4_scored",
    ),
    RowSpec(
        track="C",
        model="Llama 3.1 8B",
        condition="top1_agree compute-both L29",
        artifact_path="artifacts/dense_followup/transcender_track_c_llama3_8b_results.json",
        container_key="modes",
        item_key="top1_agree_compute_both_L29",
        comparison_space="same_model_native_token_space",
        notes="Compute-both composition recovers full-depth output on dense follow-up.",
        main_prompt_scope="legacy_expository_subset_5_prompts_4_scored",
    ),
    RowSpec(
        track="C",
        model="Llama 3.1 8B",
        condition="selective entropy L29",
        artifact_path="artifacts/dense_followup/transcender_track_c_llama3_8b_results.json",
        container_key="modes",
        item_key="selective_depth_entropy_L29",
        comparison_space="same_model_native_token_space",
        notes="Real selective depth skips some work but remains far below compute-both quality.",
        main_prompt_scope="legacy_expository_subset_5_prompts_4_scored",
    ),
    RowSpec(
        track="C",
        model="Mistral 7B v0.3",
        condition="fixed exit L29",
        artifact_path="artifacts/dense_followup/transcender_track_c_mistral7b_results.json",
        container_key="modes",
        item_key="fixed_exit_L29",
        comparison_space="same_model_native_token_space",
        notes="Late fixed exit remains weak on dense follow-up.",
        main_prompt_scope="legacy_expository_subset_5_prompts_4_scored",
    ),
    RowSpec(
        track="C",
        model="Mistral 7B v0.3",
        condition="top1_agree compute-both L29",
        artifact_path="artifacts/dense_followup/transcender_track_c_mistral7b_results.json",
        container_key="modes",
        item_key="top1_agree_compute_both_L29",
        comparison_space="same_model_native_token_space",
        notes="Compute-both composition recovers full-depth output on dense follow-up.",
        main_prompt_scope="legacy_expository_subset_5_prompts_4_scored",
    ),
    RowSpec(
        track="C",
        model="Mistral 7B v0.3",
        condition="selective entropy L29",
        artifact_path="artifacts/dense_followup/transcender_track_c_mistral7b_results.json",
        container_key="modes",
        item_key="selective_depth_entropy_L29",
        comparison_space="same_model_native_token_space",
        notes="Real selective depth skips some work but remains far below compute-both quality.",
        main_prompt_scope="legacy_expository_subset_5_prompts_4_scored",
    ),
)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def find_item(payload: Dict[str, Any], container_key: str, item_key: str) -> Dict[str, Any]:
    for item in payload.get(container_key, []):
        if item.get("key") == item_key:
            return item
    raise KeyError(f"missing {item_key!r} in {container_key!r}")


def prompt_rows(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(item.get("prompt_results", []))


def scored_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("prompt_id") != "P1"]


def matched_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = set(SHARED_MATCHED_PROMPT_IDS)
    return [row for row in rows if row.get("prompt_id") in wanted]


def prompt_layers_saved(row: Dict[str, Any]) -> float:
    if "avg_layers_saved" in row:
        return float(row["avg_layers_saved"])
    if "layers_saved" in row:
        return float(row["layers_saved"])
    return 0.0


def aggregate_layers_saved(aggregate: Dict[str, Any]) -> float:
    if "avg_avg_layers_saved" in aggregate:
        return float(aggregate["avg_avg_layers_saved"])
    if "avg_layers_saved" in aggregate:
        return float(aggregate["avg_layers_saved"])
    return 0.0


def derive_main_row(spec: RowSpec, item: Dict[str, Any]) -> Dict[str, Any]:
    rows = scored_rows(prompt_rows(item))
    aggregate = item.get("aggregate_excluding_warmup", {})
    perfect = sum(1 for row in rows if row.get("comparison", {}).get("passed"))
    return {
        "view": "paper_main",
        "track": spec.track,
        "model": spec.model,
        "condition": spec.condition,
        "prompt_scope": spec.main_prompt_scope,
        "scored_prompts": len(rows),
        "exact_match": aggregate.get("avg_exact_match_rate", mean(
            row.get("comparison", {}).get("exact_match_rate", 0.0) for row in rows
        )),
        "gen_tps": aggregate.get("avg_generation_tps", mean(row.get("generation_tps", 0.0) for row in rows)),
        "avg_layers_saved": aggregate_layers_saved(aggregate),
        "perfect_prompts": perfect,
        "comparison_space": spec.comparison_space,
        "notes": spec.notes,
        "artifact_path": spec.artifact_path,
    }


def derive_matched_row(spec: RowSpec, item: Dict[str, Any]) -> Dict[str, Any]:
    rows = matched_rows(prompt_rows(item))
    perfect = sum(1 for row in rows if row.get("comparison", {}).get("passed"))
    return {
        "view": "matched_p2_p5",
        "track": spec.track,
        "model": spec.model,
        "condition": spec.condition,
        "prompt_scope": "shared_expository_subset_P2_P5_after_warmup",
        "scored_prompts": len(rows),
        "exact_match": mean(row.get("comparison", {}).get("exact_match_rate", 0.0) for row in rows),
        "gen_tps": mean(row.get("generation_tps", 0.0) for row in rows),
        "avg_layers_saved": mean(prompt_layers_saved(row) for row in rows),
        "perfect_prompts": perfect,
        "comparison_space": spec.comparison_space,
        "notes": spec.notes,
        "artifact_path": spec.artifact_path,
    }


def collect_rows(scope: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cache: Dict[Path, Dict[str, Any]] = {}
    for spec in ROW_SPECS:
        path = REPO_ROOT / spec.artifact_path
        payload = cache.setdefault(path, load_json(path))
        item = find_item(payload, spec.container_key, spec.item_key)
        if scope in {"main", "all"}:
            rows.append(derive_main_row(spec, item))
        if scope in {"matched", "all"}:
            rows.append(derive_matched_row(spec, item))
    return rows


def fmt_float(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(rows: Sequence[Dict[str, Any]], scope: str) -> str:
    groups: List[tuple[str, List[Dict[str, Any]]]] = []
    if scope == "all":
        groups = [
            ("paper_main", [row for row in rows if row["view"] == "paper_main"]),
            ("matched_p2_p5", [row for row in rows if row["view"] == "matched_p2_p5"]),
        ]
    else:
        groups = [(rows[0]["view"] if rows else scope, list(rows))]

    sections: List[str] = []
    for view, group_rows in groups:
        if not group_rows:
            continue
        if sections:
            sections.append("")
        title = "Paper Main Scope" if view == "paper_main" else "Matched P2-P5 Scope"
        sections.append(f"## {title}")
        sections.append("")
        sections.append(
            "| track | model | condition | prompt_scope | scored_prompts | exact_match | gen_tps | avg_layers_saved | perfect_prompts | comparison_space | notes |"
        )
        sections.append(
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"
        )
        for row in group_rows:
            sections.append(
                "| {track} | {model} | {condition} | {prompt_scope} | {scored_prompts} | {exact_match} | {gen_tps} | {avg_layers_saved} | {perfect_prompts} | {comparison_space} | {notes} |".format(
                    track=row["track"],
                    model=row["model"],
                    condition=row["condition"],
                    prompt_scope=row["prompt_scope"],
                    scored_prompts=row["scored_prompts"],
                    exact_match=fmt_float(float(row["exact_match"])),
                    gen_tps=fmt_float(float(row["gen_tps"])),
                    avg_layers_saved=fmt_float(float(row["avg_layers_saved"])),
                    perfect_prompts=row["perfect_prompts"],
                    comparison_space=row["comparison_space"],
                    notes=row["notes"],
                )
            )
    return "\n".join(sections) + "\n"


def write_csv(rows: Sequence[Dict[str, Any]], out) -> None:
    fieldnames = [
        "view",
        "track",
        "model",
        "condition",
        "prompt_scope",
        "scored_prompts",
        "exact_match",
        "gen_tps",
        "avg_layers_saved",
        "perfect_prompts",
        "comparison_space",
        "notes",
        "artifact_path",
    ]
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                **row,
                "exact_match": fmt_float(float(row["exact_match"])),
                "gen_tps": fmt_float(float(row["gen_tps"])),
                "avg_layers_saved": fmt_float(float(row["avg_layers_saved"])),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("main", "matched", "all"),
        default="all",
        help="Which scope view to export.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output format.",
    )
    args = parser.parse_args()

    rows = collect_rows(args.scope)
    if args.format == "markdown":
        sys.stdout.write(render_markdown(rows, args.scope))
    else:
        write_csv(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
