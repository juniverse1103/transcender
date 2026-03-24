# Track B Rerun Note

## What Track B Measures

Track B is the scoped negative baseline for a naive cross-model cascade:

- draft model: small dense model on its own tokenizer/template path
- verifier: GPT-OSS 20B full-depth reference
- comparison space for draft and cascade rows: GPT-OSS reference token space
- aggregate convention: warmup-corrected mean over scored prompts

## Why It Needs Rerun

The canonical Track A result uses the checked-in 64-prompt suite with `P1` excluded as warmup, yielding `63` scored prompts. The checked-in Track B artifact uses the legacy five-prompt expository subset, yielding `4` scored prompts after warmup.

For paper alignment, Track B should be rerun under an explicit prompt scope so that:

- the matched expository subset is recorded unambiguously
- the artifact metadata shows which prompt suite was used
- a canonical full-suite rerun is available when needed, without changing Track A logic

## Commands

Use explicit output names so rerun artifacts are self-describing.
These commands assume an MLX-capable Python environment.

### Legacy behavior

This preserves the original five-prompt expository suite.

```bash
python scripts/track_b/transcender_track_b_benchmark.py \
  --large-model /path/to/gpt-oss-20b \
  --draft-model /path/to/gemma-3-4b-it \
  --prompt-suite expository_5 \
  --output artifacts/track_b/transcender_track_b_benchmark_expository5.json
```

### Matched subset rerun

This uses the canonical Track A prompt source but restricts to `P1`-`P5`, keeping `P1` as warmup so the scored set is the shared expository `P2`-`P5` subset.

```bash
python scripts/track_b/transcender_track_b_benchmark.py \
  --large-model /path/to/gpt-oss-20b \
  --draft-model /path/to/gemma-3-4b-it \
  --prompt-suite canonical_64 \
  --prompt-ids P1,P2,P3,P4,P5 \
  --output artifacts/track_b/transcender_track_b_benchmark_matched_p1_p5.json
```

### Canonical full-suite rerun

This is now supported cleanly by reusing the prompt definitions from the checked-in canonical Track A artifact. It is a larger run and should only be launched when the paper actually needs full-denominator Track B numbers.

```bash
python scripts/track_b/transcender_track_b_benchmark.py \
  --large-model /path/to/gpt-oss-20b \
  --draft-model /path/to/gemma-3-4b-it \
  --prompt-suite canonical_64 \
  --output artifacts/track_b/transcender_track_b_benchmark_canonical64.json
```

## Output Guidance

The Track B JSON now records:

- `prompt_scope.prompt_suite`
- `prompt_scope.selected_prompt_ids`
- `prompt_scope.selected_prompt_count`
- `prompt_scope.scored_prompt_count_after_warmup`
- `settings.prompt_scope_label`

Use filenames that match the scope label:

- `transcender_track_b_benchmark_expository5.json`
- `transcender_track_b_benchmark_matched_p1_p5.json`
- `transcender_track_b_benchmark_canonical64.json`

## What To Compare In The Paper

After the rerun:

- compare Track B matched-scope numbers against the matched Track A slice, not against the canonical `N=63` Track A table directly
- keep Track B framed as a scoped negative baseline
- keep Track C as the dense-model limitation track; no Track C rerun is required for the existing matched expository subset basis
