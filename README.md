# Transcender

**Cross-layer composition control for adaptive transformer inference.**

Transcender is best understood not as a generic early-exit method, but as a cross-layer composition control framework for adaptive transformer inference. The central problem is not early exit — it is how to combine information from different depth levels without corrupting the output distribution.

---

## Canonical Result (Track A)

| Model | Config | Exact Match | Layers Saved | Gen TPS | Peak Mem |
|-------|--------|-------------|--------------|---------|----------|
| GPT-OSS 20B (MoE, 24L) | L22 `top1_agree` + entropy-gated exit | **0.969** | **~49.5%** | 27.41 | 12.96 GB |

Hardware: Apple M1 Pro, 32 GB unified memory, Apple MLX runtime.

---

## What Is Proven (Three Tracks)

**Track A — Same-Model Adaptive Depth (Canonical)**
- GPT-OSS 20B at Layer 22 with `top1_agree` and entropy-gated exit: 0.969 exact match, ~49.5% real layer savings
- The strongest validated practical same-model adaptive-depth result in this repository

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
- The Subspace Paradox confirmed at 4.11x (MoE) and 2.46x (Dense) geometric separation
- Hidden-state blending is structurally broken; logit-space composition is mandatory
- KL-profile shape is family-sensitive, not universal

---

## Repository Structure

```
transcender/              # Installable Python package
  __init__.py             # Exports: TranscenderModel, SonRouter, SonRoutingLoss
  router.py               # Son Router + KL-calibrated routing loss
  model.py                # Multi-arch HF model wrapper (canonical)
  engine/                 # GPT-OSS 20B engine components

scripts/
  track_a/                # Canonical: MLX engine, exit layer benchmark, recon, server
  track_b/                # Scoped negative baseline: cascade engine + benchmark
  track_c/                # Validated: Gemma profile/benchmark/selective-depth, dense family validation
  exploratory/            # Probes: advanced Gemma probe, Llama family-sensitive, cache-aware
  gpt2_poc/               # Original GPT-2 proof-of-concept scripts (historical)

artifacts/
  track_a/                # Track A benchmark JSONs
  track_b/                # Track B benchmark JSONs
  track_c/                # Gemma benchmark + selective-depth JSONs
  dense_followup/         # Llama/Mistral late-checkpoint validation + exploratory probe JSONs
  recon/                  # KL reconnaissance profiles (Llama, Mistral)
  gpt2_poc/               # GPT-2 PoC benchmark artifacts (historical)

paper/
  main.tex                # LaTeX draft
  references.bib          # Bibliography
  figures/                # All visualization PNGs
  Transcender_Final_Whitepaper_v2.md   # Current evidence-audited narrative
  Transcender_Final_Whitepaper_v1.md   # Superseded by v2
  Transcender_Whitepaper_v1.md         # Original GPT-2 PoC whitepaper (historical)

docs/
  BENCHMARK_SUMMARY.md    # Cross-track comparison table
  ROADMAP.md              # Execution roadmap with status
  NEXT_EXPERIMENTS.md     # Forward-looking research plan
  chapter3_subspace_paradox.md    # The Subspace Paradox proof
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
| `scripts/track_a/transcender_engine.py` | A | Canonical | MLX runtime engine for GPT-OSS 20B |
| `scripts/track_a/transcender_exit_layer_benchmark.py` | A | Canonical | L22/L23 exit layer comparison |
| `scripts/track_a/transcender_top1_agree_benchmark.py` | A | Canonical | Blend strategy benchmark |
| `scripts/track_a/transcender_recon.py` | A | Canonical | GPT-OSS KL reconnaissance profiler |
| `scripts/track_a/transcender_server.py` | A | Canonical | FastAPI inference server |
| `scripts/track_b/transcender_track_b_cascade.py` | B | Negative baseline | Cross-model cascade engine |
| `scripts/track_b/transcender_track_b_benchmark.py` | B | Negative baseline | Track A vs Track B comparison |
| `scripts/track_c/transcender_track_c_gemma_profile.py` | C | Validated | Gemma KL depth profiling |
| `scripts/track_c/transcender_track_c_gemma_benchmark.py` | C | Validated | Gemma adaptive-depth benchmark |
| `scripts/track_c/transcender_track_c_gemma_selective_depth.py` | C | Validated | Real selective-depth speed validation |
| `scripts/track_c/transcender_track_c_dense_family_validation.py` | C | Validated | Llama/Mistral dense-family validation |
| `scripts/exploratory/transcender_track_c_gemma_advanced_probe.py` | C | Exploratory | Gemma L20 advanced probe |
| `scripts/exploratory/transcender_track_c_llama_family_sensitive_probe.py` | C | Exploratory | Llama family-sensitive probe |
| `scripts/exploratory/transcender_track_c_dense_cache_aware_probe.py` | C | Exploratory | Llama cache-aware continuation probe |

## Running the Server

```bash
python scripts/track_a/transcender_server.py --model /path/to/gpt-oss-20b --port 8000
```

## What Is Experimental

- Real dense-model selective-depth remains practically weak across all tested families
- Next directions: family-sensitive continuation criteria, token-difficulty-aware routing, cache-aware continuation
- See [docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md) for the forward-looking research plan

## Papers / Docs

- [`paper/Transcender_Final_Whitepaper_v2.md`](paper/Transcender_Final_Whitepaper_v2.md) — Current evidence-audited whitepaper
- [`paper/main.tex`](paper/main.tex) — LaTeX draft for arXiv submission
- [`docs/BENCHMARK_SUMMARY.md`](docs/BENCHMARK_SUMMARY.md) — Cross-track benchmark comparison
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — Execution roadmap
- [`docs/NEXT_EXPERIMENTS.md`](docs/NEXT_EXPERIMENTS.md) — Forward research plan

## License

See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
