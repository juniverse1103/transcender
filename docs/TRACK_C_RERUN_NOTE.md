# Track C Rerun Note

## What Track C Measures

Track C is the dense-model boundary / limitation track.

- Gemma 3 4B-IT benchmark:
  - full depth
  - fixed exits
  - `top1_agree` compute-both
  - naive blend control
- Gemma 3 4B-IT selective-depth follow-up:
  - full depth
  - fixed exit
  - `top1_agree` compute-both
  - real selective-depth gates

Aggregation is a warmup-corrected mean over scored prompts.

## Why It Needs Scope Control

The checked-in Track C artifacts were produced on the older expository five-prompt subset with `P1` excluded as warmup.

For paper alignment, Track C should be runnable under an explicit prompt scope so that:

- the legacy subset remains reproducible
- the matched `P2-P5` subset is explicit
- a canonical full-suite rerun is possible without changing Track A logic

## Commands

These commands assume an MLX-capable Python environment.

### Legacy Gemma benchmark behavior

```bash
python scripts/track_c/transcender_track_c_gemma_benchmark.py \
  --model /path/to/gemma-3-4b-it \
  --prompt-suite expository_5 \
  --output artifacts/track_c/transcender_track_c_gemma_results_expository5.json
```

### Matched-scope Gemma benchmark rerun

```bash
python scripts/track_c/transcender_track_c_gemma_benchmark.py \
  --model /path/to/gemma-3-4b-it \
  --prompt-suite canonical_64 \
  --prompt-ids P1,P2,P3,P4,P5 \
  --output artifacts/track_c/transcender_track_c_gemma_results_matched_p1_p5.json
```

### Canonical full-suite Gemma benchmark rerun

```bash
python scripts/track_c/transcender_track_c_gemma_benchmark.py \
  --model /path/to/gemma-3-4b-it \
  --prompt-suite canonical_64 \
  --output artifacts/track_c/transcender_track_c_gemma_results_canonical64.json
```

### Matched-scope selective-depth rerun

```bash
python scripts/track_c/transcender_track_c_gemma_selective_depth.py \
  --model /path/to/gemma-3-4b-it \
  --prompt-suite canonical_64 \
  --prompt-ids P1,P2,P3,P4,P5 \
  --output artifacts/track_c/transcender_track_c_gemma_selective_depth_results_matched_p1_p5.json
```

## Output Guidance

The Gemma Track C JSON payloads now record:

- `prompt_scope.prompt_suite`
- `prompt_scope.selected_prompt_ids`
- `prompt_scope.selected_prompt_count`
- `prompt_scope.scored_prompt_count_after_warmup`
- `prompt_scope.scope_label`

Recommended filenames:

- `transcender_track_c_gemma_results_expository5.json`
- `transcender_track_c_gemma_results_matched_p1_p5.json`
- `transcender_track_c_gemma_results_canonical64.json`
- `transcender_track_c_gemma_selective_depth_results_matched_p1_p5.json`

## Paper Use

- Compare matched-scope Track C numbers against the matched Track A slice and the matched Track B rerun, not against the canonical `N=63` Track A table directly.
- Keep Track C framed as dense-model limitation / boundary evidence, not as a competing main result.
- The dense follow-up Llama and Mistral artifacts can remain unchanged unless the paper later requires same-denominator full-scope reruns for those models too.
