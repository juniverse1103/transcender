# Transcender

**Research repository for the Transcender paper, experiments, and reporting artifacts.**

This repository contains the code, paper-facing materials, and benchmark artifacts for the Transcender research program. It includes the canonical MLX Track A benchmark path, the Track B cascade baseline, the Track C dense-model evaluation work, and the GPU diagnostic workflow used for token-row export and offline Stage B analysis.

**Transcender: Cross-layer composition control for adaptive transformer inference.**

Transcender is best understood not as a generic early-exit method, but as a cross-layer composition control framework for adaptive transformer inference. The central problem is not early exit — it is how to combine information from different depth levels without corrupting the output distribution.

---

## Canonical Results (Track A)

| Model | Config | Exact Match | Avg Layers Saved | Gen TPS | N |
|-------|--------|-------------|------------------|---------|---|
| GPT-OSS 20B (MoE, 24L) | L22 `top1_agree` + entropy-gated exit | **0.870** | 0.490 | 18.6 | 63 |
| Qwen3-30B-A3B (MoE, 48L) | L46 `top1_agree` + entropy-gated exit | **0.837** | 0.760 | 32.1 | 63 |

Hardware: Apple M1 Pro, 32 GB unified memory, Apple MLX runtime.

**Note on avg_layers_saved:** This metric reports the mean number of layers physically skipped per generated token, not the percentage of total compute saved. At GPT-OSS L22 and Qwen3 L46, each early-exiting token skips exactly one layer, so `0.490` means roughly 49% of GPT-OSS decode tokens triggered the entropy gate and exited early.

---

## What Is Proven (Three Tracks)

**Track A — Same-Model Adaptive Depth (Canonical)**
- GPT-OSS 20B at L22 `top1_agree`: 0.870 exact match (47/63 perfect), 0.490 avg layers saved per token
- Qwen3-30B-A3B at L46 `top1_agree`: 0.837 exact match (36/63 perfect), 0.760 avg layers saved per token
- Both MoE families show a single viable penultimate-layer operating point with a sharp quality cliff one layer earlier

**Track B — Cross-Model Cascade (Scoped Negative Baseline)**
- Gemma 3 4B-IT + GPT-OSS 20B naive cascade: 0.026 exact match, 0.28 TPS, 20.19 GB
- Vocabulary/distribution mismatch destroyed acceptance rate
- Result is specific to this cascade implementation, model pair, and local runtime; does not invalidate cascade strategies in general

**Track C — Dense Model Validation**
- Fixed early exit is catastrophic on all three tested dense families (Gemma 3 4B-IT, Llama 3.1 8B, Mistral 7B)
- Agreement-aware compute-both composition recovers substantial quality across all three: 0.807 on Gemma at L31, 1.000 on Llama at L29, 1.000 on Mistral at L29
- Real selective-depth was operationally correct but practically weak on all tested dense runs on this local Apple MLX runtime
- The composition *methodology* generalizes; no practical dense speed frontier is established

**Cross-Architecture**
- The Subspace Mismatch confirmed at 4.11x (MoE) and 2.46x (Dense) geometric separation
- Hidden-state blending is structurally broken; logit-space composition is mandatory
- KL-profile shape is family-sensitive, not universal

---

## How To Read The Evidence

- `Track A` is the primary empirical result in the paper: the main same-model adaptive-depth frontier on the local MLX path.
- `Track B` remains paper-relevant as a scoped negative baseline showing that the naive cross-model cascade setup fails in this configuration.
- `Track C` remains paper-relevant as boundary-condition evidence on dense models: fixed exit degrades, compute-both recovers quality, and real selective-depth remains weak on this runtime.
- `GPU Track A + Stage B / karma` is an offline diagnostic extension. It helps interpret penultimate acceptance, but it does not replace the core roles of Tracks A, B, and C.
- Direct cross-track numeric comparison should either use the shared `P2`-`P5` expository subset or keep the canonical `Track A N=63` tables separate from the smaller `Track B` and `Track C` tables.

---

## GPU Diagnostic Path And Offline Stage B

- `scripts/track_a_gpu/` provides a trustworthy manual-reference decode path, oracle-style final-aware summaries, token-level relation-row export, and offline proxy evaluation helpers.
- Token rows can capture already-exported adjacent-layer features such as entropy, margin, rank, overlap, and logit-delta signals for the previous-vs-penultimate comparison.
- The Stage B evaluation set on the GPU path includes Gemma 3 `4B`, `12B`, and `27B`, GPT-OSS 20B, Mixtral 8x7B, and Mistral 7B.
- A main negative result is that naive adjacent-layer agreement is not a viable Stage B acceptance signal across this cross-family set.
- Penultimate entropy remains a crude offline baseline, and the framing is correction-vs-shared-failure rather than simple agreement.
- The offline interpretation is that `karma` materially improves Stage B decision quality across multiple model families, but this remains a held-out offline evaluation result rather than an online policy claim.
- `karma = probability_of_need_full_depth`, so lower is safer to accept at penultimate.
- This Stage B work strengthens the interpretation of penultimate acceptance, but it does not eliminate the need for Track B and Track C in the paper.
- This is not an online serving policy and not a claim of final-free inference.
- See [`docs/karma_stage_b_summary.md`](docs/karma_stage_b_summary.md) for the compact paper-facing Stage B note and export instructions.

---

## Stage A / Stage B Terminology

- `Stage A`: decide whether the earlier candidate was already correct relative to full depth. In the oracle labels this is `earlier_correct`.
- `Stage B`: condition on Stage A already missing, then decide whether the penultimate layer is sufficient or whether full depth is still needed. In the oracle labels this is `need_penultimate` versus `need_full_depth`.
- `penultimate_entropy`: a confidence-style baseline on the penultimate logits. Lower entropy usually means a sharper distribution, but sharpness alone does not tell us whether the penultimate layer is still wrong in a way that only full depth will fix.
- `penultimate_margin`: another confidence-style baseline using the penultimate top-1 vs top-2 gap. It is useful as a reference baseline, not as the main interpretation of the problem.
- `adjacent_top1_agree` and related adjacent-agreement variants: relation heuristics based on whether the earlier and penultimate layers agree. The offline result here is negative: agreement alone does not reliably separate `need_penultimate` from `need_full_depth`.
- `karma`: a small offline logistic model over penultimate and adjacent-relation features. The intended interpretation is risk, not confidence: `karma = probability_of_need_full_depth`, and lower is safer.
- Confidence and risk are not the same in this workflow. A token can look locally confident at penultimate and still be in a regime where the final layer is likely to correct it.

---

## Repository Structure

```
transcender/              # Installable Python package
  __init__.py             # Exports: TranscenderModel, SonRouter, SonRoutingLoss
  router.py               # Son Router + KL-calibrated routing loss
  model.py                # Multi-arch HF model wrapper (canonical)
  policies.py             # Dense selective-depth acceptance policies (exploratory Track C)
  engine/                 # Sparse-MoE engine components

scripts/
  track_a/                # Canonical: MLX engine, exit layer benchmark, recon, server
  track_a_gpu/            # Diagnostic: manual-reference GPU validation, relation-row export, proxy evaluation
  track_b/                # Scoped negative baseline: cascade engine + benchmark
  track_c/                # Validated: Gemma profile/benchmark/selective-depth, dense family validation
  exploratory/            # Probes: advanced Gemma probe, Llama family-sensitive, cache-aware
  gpt2_poc/               # Original GPT-2 proof-of-concept scripts (historical)

artifacts/
  track_a/                # Track A benchmark JSONs
  track_a_gpu/            # Local GPU diagnostic traces, benchmark summaries, and offline proxy inputs/outputs
  track_b/                # Track B benchmark JSONs
  track_c/                # Gemma benchmark + selective-depth JSONs
  dense_followup/         # Llama/Mistral late-checkpoint validation + exploratory probe JSONs
  recon/                  # KL reconnaissance profiles (Llama, Mistral)
  gpt2_poc/               # GPT-2 PoC benchmark artifacts (historical)

paper/
  main.tex                # LaTeX draft
  references.bib          # Bibliography
  figures/                # All visualization PNGs
  Transcender_Final_Whitepaper_v2.md   # Superseded markdown narrative; see paper/main.tex
  Transcender_Final_Whitepaper_v1.md   # Superseded markdown narrative (historical)
  Transcender_Whitepaper_v1.md         # Original GPT-2 PoC whitepaper (historical)

docs/
  BENCHMARK_SUMMARY.md    # Cross-track comparison table
  ROADMAP.md              # Archived execution roadmap
  NEXT_EXPERIMENTS.md     # Archived optional research backlog
  chapter3_subspace_paradox.md    # Subspace mismatch chapter (historical filename retained)
  chapter4_logit_blend_solution.md # Logit-space blending solution

models/                   # Son Router checkpoint files (.pt, GPT-2 PoC only)
tests/                    # pytest smoke tests
```

See [REPO_MAP.md](REPO_MAP.md) for detailed file classification.
See [RESULTS_INDEX.md](RESULTS_INDEX.md) for artifact inventory.

---

## Installation

```bash
pip install -e .           # Core (torch + transformers)
pip install -e ".[mlx]"    # + Apple MLX support
pip install -e ".[dev]"    # + pytest
pip install -e ".[all]"    # Everything
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Scripts

| Script | Track | Status | Purpose |
|--------|-------|--------|---------|
| `scripts/track_a/transcender_engine.py` | A | Canonical | MLX runtime engine for GPT-OSS 20B and Qwen3-30B-A3B |
| `scripts/track_a/transcender_exit_layer_benchmark.py` | A | Canonical | Exit-layer frontier benchmark (GPT-OSS and Qwen3) |
| `scripts/track_a/transcender_top1_agree_benchmark.py` | A | Supplementary | Earlier GPT-OSS blend strategy benchmark |
| `scripts/track_a/transcender_recon.py` | A | Canonical | GPT-OSS KL reconnaissance profiler |
| `scripts/track_a/transcender_server.py` | A | Canonical | FastAPI inference server |
| `scripts/track_a_gpu/transcender_gpu_reproduction.py` | A | Diagnostic | Manual-reference GPU shared-context validation; can also emit oracle summaries and token-level relation rows |
| `scripts/track_a_gpu/analyze_debug_trace.py` | A | Diagnostic | Summarize a single-prompt GPU trace before trusting aggregate results |
| `scripts/track_a_gpu/summarize_benchmark.py` | A | Diagnostic | Summarize aggregate GPU validation JSON by layer |
| `scripts/track_a_gpu/compare_benchmarks.py` | A | Diagnostic | Compare multiple GPU benchmark JSON files across models |
| `scripts/track_a_gpu/evaluate_relation_proxies.py` | A | Diagnostic | Offline Stage A / Stage B proxy evaluation on token rows, including optional `karma` logistic fitting |
| `scripts/export_track_comparison_table.py` | A/B/C | Reporting | Export compact paper-facing Track A / B / C comparison tables for main scope or matched `P2-P5` scope |
| `scripts/track_b/transcender_track_b_cascade.py` | B | Negative baseline | Cross-model cascade engine |
| `scripts/track_b/transcender_track_b_benchmark.py` | B | Negative baseline | Track A vs Track B comparison |
| `scripts/track_c/transcender_track_c_gemma_profile.py` | C | Validated | Gemma KL depth profiling |
| `scripts/track_c/transcender_track_c_gemma_benchmark.py` | C | Validated | Gemma adaptive-depth benchmark |
| `scripts/track_c/transcender_track_c_gemma_selective_depth.py` | C | Validated | Real selective-depth speed validation |
| `scripts/track_c/transcender_track_c_dense_family_validation.py` | C | Validated | Llama/Mistral dense-family validation |
| `scripts/track_c/transcender_dense_exit_sweep.py` | C | Exploratory | Dense exit-layer agreement sweep for frontier discovery |
| `scripts/exploratory/transcender_track_c_gemma_advanced_probe.py` | C | Exploratory | Gemma L20 advanced probe |
| `scripts/exploratory/transcender_track_c_llama_family_sensitive_probe.py` | C | Exploratory | Llama family-sensitive probe |
| `scripts/exploratory/transcender_track_c_dense_cache_aware_probe.py` | C | Exploratory | Llama cache-aware continuation probe |

## Running the Server

```bash
python scripts/track_a/transcender_server.py --model /path/to/gpt-oss-20b --port 8000
```

## What Is Experimental

- Real dense-model selective-depth remains practically weak across all tested families
- The GPU Track A path in `scripts/track_a_gpu/` is a diagnostic off-MLX validation flow, not a canonical paper benchmark or serving benchmark
- The offline Stage B proxy path in `scripts/track_a_gpu/` is for analysis only; `karma` is an internal offline risk score, not a deployable routing policy
- Optional post-release directions: family-sensitive continuation criteria, token-difficulty-aware routing, cache-aware continuation
- See [docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md) for the archived post-release research backlog

## GPU Track A Validation

`scripts/track_a_gpu/` is the repo's off-MLX validation path for Track A. It exists to test whether the penultimate-layer frontier survives on GPU under a trustworthy manual-reference decode path. It is not the canonical paper evidence and it should not be read as a serving-speed benchmark. It also does not implement MLX-style entropy-gated physical skipping, so its aggregates are structural diagnostics rather than direct replacements for the MLX Track A release numbers.

Within the paper framing, this path is additive rather than substitutive: the local MLX Track A result remains the main empirical claim, Track B remains the scoped negative cascade baseline, and Track C remains the dense-model limitation evidence.

This path now also supports oracle-style final-aware summaries, token-level relation-row export, and offline proxy evaluation. The present Stage B conclusion is narrow: naive adjacent-layer agreement has not held up as a cross-family penultimate-acceptance signal, penultimate entropy is a crude live baseline, and the next step is offline interpretable risk modeling rather than another oracle mode.

**Two axes**
- Architecture class:
  - sparse MoE
  - dense
- Validation status:
  - empirically validated on this GPU path
  - code-path supported, pending real runs
  - blocked by model access

**Architecture + validation status**
- Sparse MoE:
  - `Qwen/Qwen3-30B-A3B` — empirically validated GPU baseline
  - `openai/gpt-oss-20b` — empirically validated sparse external-validity result
  - `mistralai/Mixtral-8x7B-Instruct-v0.1` — empirically validated sparse-MoE extension
- Dense:
  - `mistralai/Mistral-7B-Instruct-v0.3` — empirically validated dense control
  - `meta-llama/Llama-3.1-8B-Instruct` — blocked by gated Hugging Face access; not run on this GPU Track A path yet
  - `google/gemma-3-4b-it` — empirically validated dense-family supporting evidence
  - `google/gemma-3-12b-it` — empirically validated but materially weaker dense-family evidence
  - `google/gemma-3-27b-it` — empirically validated strongest current Gemma 3 result
  - older Gemma / Gemma2 text checkpoints — code-path supported, pending real runs

Gemma 3 is a dense family, not a sparse-MoE family. In this GPU validation flow it should be discussed at the checkpoint level, especially `4B`, `12B`, and `27B`, because those sizes imply different memory/runtime costs and make Gemma 3 a size-aware dense comparison axis rather than one undifferentiated model. Smaller `270M` and `1B` Gemma 3 text checkpoints exist, but they are not the main recommended dense-control targets here unless there is a specific reason to probe very small-scale behavior.

The explicit manual path supports Qwen3/Qwen2-MoE, GPT-OSS, Mixtral, Llama, Mistral, Gemma, Gemma2, and Gemma 3 text checkpoints, with fail-fast checks for unsupported architectures. Multimodal Gemma 3 checkpoints are intentionally out of scope for this path.

For command examples, note that `openai/gpt-oss-20b` should currently be run with `--quantize none`, not the BitsAndBytes `4bit` path used in the other example commands.

**Currently verified GPU manual-reference outcomes**
- `openai/gpt-oss-20b` with `--quantize none --exit-layers 21 22`: sane trace; N=63 raw-exit exact match is `0.840` at `L21` and `0.885` at `L22`; `L22` is at least as strong as `L21` on `55/63` prompts and strictly better on `53/63`. This is the canonical sparse supporting evidence on this GPU path.
- `mistralai/Mixtral-8x7B-Instruct-v0.1` with `--quantize 4bit --exit-layers 29 30`: sane trace; N=63 raw-exit exact match is `0.667` at `L29` and `0.837` at `L30`; `L30` beats `L29` on all `63/63` scored prompts. This is strong sparse-family generalization with a very clean penultimate advantage.
- `mistralai/Mistral-7B-Instruct-v0.3` with `--quantize 4bit --exit-layers 29 30`: sane trace; N=63 raw-exit exact match is `0.715` at `L29` and `0.864` at `L30`; `L30` is at least as strong on `63/63` prompts and strictly better on `62/63`. This is the strongest first dense control on the GPU path so far.
- `google/gemma-3-4b-it` with `--device cuda --quantize none --exit-layers 31 32`: the current result is numerically sane and supersedes the earlier invalid all-`1.000` run; the verified `P2` trace has finite hidden states and logits throughout, and the N=63 benchmark gives raw-exit exact match `0.833` at `L31` and `0.900` at `L32`, with `L32` at least as strong on `58/63` prompts and strictly better on `54/63`. This is genuine dense-family supporting evidence under the GPU manual-reference path, not yet a size-scaling conclusion.
- `google/gemma-3-12b-it` with `--device cuda --quantize none --exit-layers 41 42`: the trace is sane and the logits are finite, but the benchmark evidence is materially weaker than the 4B and 27B checkpoints. N=63 raw-exit exact match is `0.528` at `L41` and `0.562` at `L42`, with `L42` at least as strong on `54/63` prompts and strictly better on `46/63`. This is valid checkpoint-specific evidence, but not especially strong dense-family support on its own.
- `google/gemma-3-27b-it` with `--device cuda --quantize none --exit-layers 59 60`: the trace is sane, the logits are finite, and the benchmark is the strongest current Gemma 3 result. N=63 raw-exit exact match is `0.813` at `L59` and `0.932` at `L60`, with `L60` at least as strong on `63/63` prompts and strictly better on `63/63`. This is the cleanest Gemma 3 supporting evidence in the current GPU Track A set.
- `meta-llama/Llama-3.1-8B-Instruct` has not been run on this GPU Track A path yet. The model remains blocked by gated Hugging Face access, so it should not be described as empirically validated here.

Taken together, the current dense-side GPU evidence is checkpoint-specific rather than a universal dense-family law: Gemma 3 `4B` is decent, `12B` is weak, and `27B` is strong; Mistral 7B remains the clean first dense control; Llama is still pending because of access rather than because the code path is missing.

**Metric semantics**
- `raw_exit_*`: raw intermediate-layer candidate tokens compared against full depth under shared full-depth context. This is the primary diagnostic metric for whether the tested exit layers genuinely diverge from the final layer.
- `composed_*`: conservative `top1_agree` composition outputs. Because disagreement falls back to the full-depth token, these metrics are a composition diagnostic, not a substitute for raw-exit divergence.
- `avg_top1_agreement_rate`: per-token fraction of raw top-1 agreement between the exit layer and full depth.

**Trust rule**
- Run a single-prompt debug trace first.
- Inspect the trace directly or run the trace-analysis helper.
- Only trust aggregate benchmark output after at least one trace shows real raw divergence and sane fallback behavior.
- Do not treat `composed_exact_match = 1.0` as the frontier claim. The main signal is raw-exit divergence and penultimate-vs-earlier-layer behavior.

By default, the GPU script tests the model's penultimate-minus-one and penultimate layers. Keep explicit layer numbers for model-specific checks in the runbook; use the default behavior for the top-level flow.

See [scripts/track_a_gpu/RUNBOOK.md](scripts/track_a_gpu/RUNBOOK.md) for the full GPU validation flow.

```bash
# 1) Single-prompt debug trace
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model Qwen/Qwen3-30B-A3B \
  --quantize 4bit \
  --debug-trace \
  --prompt-id P2 \
  --max-new-tokens 16 \
  --output artifacts/track_a_gpu/qwen3_trace_p2.json

# 2) Trace analysis helper
python scripts/track_a_gpu/analyze_debug_trace.py \
  artifacts/track_a_gpu/qwen3_trace_p2.json

# 3) Full benchmark run
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model Qwen/Qwen3-30B-A3B \
  --quantize 4bit \
  --max-new-tokens 48 \
  --output artifacts/track_a_gpu/qwen3_gpu_reproduction_n63.json

# 4) Benchmark summary helper
python scripts/track_a_gpu/summarize_benchmark.py \
  artifacts/track_a_gpu/qwen3_gpu_reproduction_n63.json

# 5) Optional token-row export for offline proxy analysis
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model openai/gpt-oss-20b \
  --quantize none \
  --emit-token-rows \
  --token-rows-output artifacts/track_a_gpu/gpt_oss_20b_token_rows.jsonl \
  --output artifacts/track_a_gpu/gpt_oss_20b_gpu_reproduction_n63.json

# 6) Offline proxy evaluation, including optional karma fitting
python scripts/track_a_gpu/evaluate_relation_proxies.py \
  artifacts/track_a_gpu/gpt_oss_20b_token_rows.jsonl \
  --fit-karma-logistic

# 7) Compact markdown / CSV export for paper notes
python scripts/track_a_gpu/summarize_karma_results.py \
  --format markdown \
  artifacts/track_a_gpu/*karma*.json
```

## Papers / Docs

- [`paper/main.tex`](paper/main.tex) — Canonical arXiv draft and release source of truth
- [`paper/snippets/offline_stage_b_karma_note.tex`](paper/snippets/offline_stage_b_karma_note.tex) — Ready-to-paste cautious LaTeX wording for the offline Stage B framing
- [`paper/Transcender_Final_Whitepaper_v2.md`](paper/Transcender_Final_Whitepaper_v2.md) — Superseded markdown narrative retained with a status notice
- [`docs/BENCHMARK_SUMMARY.md`](docs/BENCHMARK_SUMMARY.md) — Cross-track benchmark comparison
- [`docs/TRACK_MATCHING_PLAN.md`](docs/TRACK_MATCHING_PLAN.md) — Exact Track A / B / C scope mismatch and matched-scope reporting plan
- [`docs/karma_stage_b_summary.md`](docs/karma_stage_b_summary.md) — Compact Stage B summary, limitations, and export commands
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — Archived execution roadmap with release-status note
- [`docs/NEXT_EXPERIMENTS.md`](docs/NEXT_EXPERIMENTS.md) — Archived optional post-release research notes

## License

See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
