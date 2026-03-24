# Track Matching Plan

This note records the current Track A / B / C comparison mismatch and the conservative matched-scope reporting plan for the paper.

## Current Scope

| track | what it currently measures | prompt scope | scored prompts | matching convention | aggregation |
| --- | --- | --- | ---: | --- | --- |
| `Track A` | same-model adaptive-depth frontier on MLX | canonical 64-prompt suite across six categories | 63 | same-model token comparison against full-depth reference in the model's native tokenizer space | warmup-corrected mean over scored prompts |
| `Track B` | scoped naive cross-model cascade baseline | legacy five-prompt expository subset | 4 | cross-model outputs normalized into GPT-OSS reference token space | warmup-corrected mean over scored prompts |
| `Track C` | dense-model limitation / boundary evidence | legacy five-prompt expository subset | 4 | same-model token comparison against dense full-depth reference in each model's native tokenizer space | warmup-corrected mean over scored prompts |

## Exact Mismatch Found

- `Track A` main paper numbers are `N=63` on the full six-category suite.
- `Track B` and the checked-in `Track C` artifacts are `N=4` scored prompts after excluding `P1` from the older five-prompt expository subset.
- `Track B` is not only smaller-scope; it also uses a different comparison space for the draft/cascade rows: GPT-OSS reference token space rather than same-model native tokenizer space.
- The aggregation rule is mostly aligned across tracks: warmup-corrected mean exact match over scored prompts. The main mismatch is denominator, task mix, and comparison space.
- The repo also contained a stale summary problem: `docs/BENCHMARK_SUMMARY.md` was still mixing older Track A numbers instead of the canonical `N=63` results.
- The paper had one direct apples-to-oranges presentation issue: the Track B table used the canonical Track A `N=63` row as context inside a five-prompt Track B table. That should be matched-scope context instead.

## What Matched Scope Should Mean

- Keep `Track A` as the main table on the canonical `N=63` suite.
- For direct cross-track comparison, slice `Track A` to the same shared expository subset already used by `Track B` and `Track C`: `P2`-`P5`, with `P1` excluded as warmup.
- Do not pool `Track A N=63` and `Track B/C N=4` into a single denominator or imply that the tables are directly comparable without that slice.
- Keep the main table vs supporting-table split explicit:
  - `Track A`: main frontier evidence
  - `Track B`: scoped negative baseline
  - `Track C`: dense limitation / boundary evidence
  - `GPU Stage B / karma`: additive offline diagnostic extension

## Existing Matched-Scope Snapshot

The shared matched subset is the expository `P2`-`P5` slice. Track~A is sliced from the canonical `N=63` artifacts. Track~B now also has a dedicated matched-scope rerun artifact:

- `artifacts/track_b/transcender_track_b_benchmark_matched_p1_p5_chunk16.json`
- `artifacts/track_c/transcender_track_c_gemma_results_matched_p1_p5.json`
- `artifacts/track_c/transcender_track_c_gemma_selective_depth_results_matched_p1_p5.json`

| track | model | condition | scope | exact match | gen TPS | note |
| --- | --- | --- | --- | ---: | ---: | --- |
| `A` | `GPT-OSS 20B` | `L22 top1_agree` | `P2-P5` matched subset | `1.000` | `23.764` | canonical Track A artifact sliced to the shared subset |
| `A` | `Qwen3-30B-A3B` | `L46 top1_agree` | `P2-P5` matched subset | `0.760` | `35.456` | same shared subset, weaker than GPT-OSS |
| `B` | `Gemma 3 4B-IT -> GPT-OSS 20B` | `naive cascade` | `P2-P5` matched subset | `0.021` | `0.177` | matched-scope rerun using canonical prompt definitions filtered to `P1-P5`; compared in GPT-OSS reference token space |
| `C` | `Gemma 3 4B-IT` | `top1_agree compute-both L31` | `P2-P5` matched subset | `0.807` | `14.493` | matched-scope Gemma rerun confirms dense compute-both quality recovery |
| `C` | `Llama 3.1 8B` | `top1_agree compute-both L29` | `P2-P5` matched subset | `1.000` | `17.295` | dense follow-up quality recovery |
| `C` | `Mistral 7B v0.3` | `top1_agree compute-both L29` | `P2-P5` matched subset | `1.000` | `17.674` | dense follow-up quality recovery |

These numbers do not erase the role differences between the tracks. They only provide the smallest grounded matched-scope comparison already recoverable from the repo snapshot.

## Recommended Next Run

No new GPU Stage B run is required for the matching problem.

If a next real run is scheduled, the minimal high-value target is now narrower:

1. Do **not** rerun `Track A` just to recover matched scope. The canonical `N=63` artifacts already support a matched `P2-P5` slice.
2. Do **not** rerun `Track B` for the current matched-scope table unless the model pair or generation settings change. The dedicated matched-scope artifact now exists.
3. Do **not** rerun the matched-scope Gemma Track~C scripts unless the model, prompt subset, or runtime settings change. The dedicated matched-scope artifacts now exist.
4. Do **not** rerun GPU `Stage B / karma` for this paper-structure issue.
5. The next optional target, only if the paper needs fuller same-denominator dense evidence, is a canonical-scope Gemma Track~C rerun:
   - `scripts/track_c/transcender_track_c_gemma_benchmark.py`
   - `scripts/track_c/transcender_track_c_gemma_selective_depth.py`
6. Defer full-scope reruns of the Llama and Mistral dense follow-up until the Gemma canonical-scope rerun is stable and paper-necessary.

## Reproducible Export

Use the compact helper to regenerate the comparison tables from the checked-in JSON artifacts:

```bash
python3 scripts/export_track_comparison_table.py --scope main --format markdown
python3 scripts/export_track_comparison_table.py --scope matched --format markdown
python3 scripts/export_track_comparison_table.py --scope all --format csv
```
