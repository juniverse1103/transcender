# Transcender-MLX

**MLX reference implementation and benchmark suite for the Transcender paper.**

`transcender-mlx` is the public Apple MLX implementation repo for the Transcender paper. Transcender remains the method and paper identity; `transcender-mlx` is the repository identity for the MLX reference implementation and release artifacts.

**Transcender: Cross-layer composition control for adaptive transformer inference.**

Transcender is best understood not as a generic early-exit method, but as a cross-layer composition control framework for adaptive transformer inference. The central problem is not early exit — it is how to combine information from different depth levels without corrupting the output distribution.

---

## Canonical Results (Track A)

| Model | Config | Exact Match | Avg Layers Saved | Gen TPS | N |
|-------|--------|-------------|------------------|---------|---|
| GPT-OSS 20B (MoE, 24L) | L22 `top1_agree` + entropy-gated exit | **0.941** | 0.528 | 21.1 | 15 |
| Qwen3-30B-A3B (MoE, 48L) | L46 `top1_agree` + entropy-gated exit | **0.868** | 0.735 | 34.5 | 15 |

Hardware: Apple M1 Pro, 32 GB unified memory, Apple MLX runtime.

**Note on avg_layers_saved:** This metric reports the mean number of layers physically skipped per generated token, not the percentage of total compute saved. At L22 on a 24-layer model, each early-exiting token skips 1 layer; the 0.528 figure means ~53% of tokens triggered the entropy gate.

---

## What Is Proven (Three Tracks)

**Track A — Same-Model Adaptive Depth (Canonical)**
- GPT-OSS 20B at L22 `top1_agree`: 0.941 exact match (13/15 perfect), 0.528 avg layers saved per token
- Qwen3-30B-A3B at L46 `top1_agree`: 0.868 exact match (11/15 perfect), 0.735 avg layers saved per token
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
  track_a_gpu/            # Diagnostic: manual-reference GPU validation + helpers
  track_b/                # Scoped negative baseline: cascade engine + benchmark
  track_c/                # Validated: Gemma profile/benchmark/selective-depth, dense family validation
  exploratory/            # Probes: advanced Gemma probe, Llama family-sensitive, cache-aware
  gpt2_poc/               # Original GPT-2 proof-of-concept scripts (historical)

artifacts/
  track_a/                # Track A benchmark JSONs
  track_a_gpu/            # Local GPU diagnostic traces and benchmark summaries (non-canonical)
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
| `scripts/track_a_gpu/transcender_gpu_reproduction.py` | A | Diagnostic | Manual-reference GPU shared-context validation for raw exit vs composed `top1_agree` |
| `scripts/track_a_gpu/analyze_debug_trace.py` | A | Diagnostic | Summarize a single-prompt GPU trace before trusting aggregate results |
| `scripts/track_a_gpu/summarize_benchmark.py` | A | Diagnostic | Summarize aggregate GPU validation JSON by layer |
| `scripts/track_a_gpu/compare_benchmarks.py` | A | Diagnostic | Compare multiple GPU benchmark JSON files across models |
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
- Optional post-release directions: family-sensitive continuation criteria, token-difficulty-aware routing, cache-aware continuation
- See [docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md) for the archived post-release research backlog

## GPU Track A Validation

`scripts/track_a_gpu/` is the repo's off-MLX validation path for Track A. It exists to test whether the penultimate-layer frontier survives on GPU under a trustworthy manual-reference decode path. It is not the canonical paper evidence and it should not be read as a serving-speed benchmark. It also does not implement MLX-style entropy-gated physical skipping, so its aggregates are structural diagnostics rather than direct replacements for the MLX Track A release numbers.

**Two axes**
- Architecture class:
  - sparse MoE
  - dense
- Validation status:
  - empirically validated on this GPU path
  - code-path supported, pending real runs

**Architecture + validation status**
- Sparse MoE:
  - `Qwen/Qwen3-30B-A3B` — empirically validated GPU baseline
  - `openai/gpt-oss-20b` — code-path supported, pending real run
  - `mistralai/Mixtral-8x7B-Instruct-v0.1` — code-path supported, pending real run
- Dense:
  - `mistralai/Mistral-7B-Instruct-v0.3` — code-path supported, pending real run
  - `meta-llama/Llama-3.1-8B-Instruct` — code-path supported, pending real run
  - `google/gemma-3-4b-it` — code-path supported, pending real run
  - `google/gemma-3-12b-it` — code-path supported, pending real run
  - `google/gemma-3-27b-it` — code-path supported, pending real run
  - older Gemma / Gemma2 text checkpoints — code-path supported, pending real runs

Gemma 3 is a dense family, not a sparse-MoE family. In this GPU validation flow it should be discussed at the checkpoint level, especially `4B`, `12B`, and `27B`, because those sizes imply different memory/runtime costs and make Gemma 3 a size-aware dense comparison axis rather than one undifferentiated model. Smaller `270M` and `1B` Gemma 3 text checkpoints exist, but they are not the main recommended dense-control targets here unless there is a specific reason to probe very small-scale behavior.

The explicit manual path supports Qwen3/Qwen2-MoE, GPT-OSS, Mixtral, Llama, Mistral, Gemma, Gemma2, and Gemma 3 text checkpoints, with fail-fast checks for unsupported architectures. Multimodal Gemma 3 checkpoints are intentionally out of scope for this path.

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
```

## Papers / Docs

- [`paper/main.tex`](paper/main.tex) — Canonical arXiv draft and release source of truth
- [`paper/Transcender_Final_Whitepaper_v2.md`](paper/Transcender_Final_Whitepaper_v2.md) — Superseded markdown narrative retained with a status notice
- [`docs/BENCHMARK_SUMMARY.md`](docs/BENCHMARK_SUMMARY.md) — Cross-track benchmark comparison
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — Archived execution roadmap with release-status note
- [`docs/NEXT_EXPERIMENTS.md`](docs/NEXT_EXPERIMENTS.md) — Archived optional post-release research notes

## License

See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
