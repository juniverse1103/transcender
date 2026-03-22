# Repository Map

This document explains the top-level structure and classifies every major component by its role in the research.

---

## Classification Key

| Status | Meaning |
|--------|---------|
| **Canonical** | Supports the paper's core argument directly |
| **Validated follow-up** | Measured result that extends the core argument |
| **Negative baseline** | Scoped comparison that establishes what does not work in the tested configuration |
| **Exploratory** | Probes that informed research direction but are not core evidence |
| **Historical (GPT-2 PoC)** | Original proof-of-concept; not part of the current paper |

---

## Top-Level Directories

| Directory | Contents |
|-----------|----------|
| `transcender/` | Installable Python package: `TranscenderModel`, `SonRouter`, `SonRoutingLoss`, GPT-OSS engine components |
| `scripts/` | All experiment and benchmark scripts, organized by track |
| `artifacts/` | All JSON result files, organized by track |
| `paper/` | LaTeX draft, bibliography, figures, whitepaper narratives |
| `docs/` | Benchmark summary, roadmap, next experiments, supplementary chapters |
| `models/` | Son Router checkpoint files (GPT-2 PoC, ~195 KB each) |
| `tests/` | pytest smoke tests for the `transcender` package |

---

## scripts/

### scripts/track_a/ — Canonical

| File | Purpose |
|------|---------|
| `transcender_engine.py` | MLX runtime engine for GPT-OSS 20B with entropy-gated exit |
| `transcender_exit_layer_benchmark.py` | L22/L23 exit layer comparison benchmark |
| `transcender_top1_agree_benchmark.py` | Blend strategy benchmark (top1_agree vs naive blend) |
| `transcender_recon.py` | GPT-OSS KL reconnaissance depth profiler |
| `transcender_server.py` | FastAPI inference server |
| `transcender_20b_scaffold.py` | GPT-OSS 20B model scaffold and configuration |

### scripts/track_b/ — Negative Baseline

| File | Purpose |
|------|---------|
| `transcender_track_b_cascade.py` | Cross-model cascade engine (Gemma draft + GPT-OSS verify) |
| `transcender_track_b_benchmark.py` | Track A vs Track B comparison benchmark |

### scripts/track_c/ — Validated Follow-Up

| File | Purpose |
|------|---------|
| `transcender_track_c_gemma_profile.py` | Gemma 3 4B-IT KL depth profiling |
| `transcender_track_c_gemma_benchmark.py` | Gemma six-mode adaptive-depth benchmark |
| `transcender_track_c_gemma_selective_depth.py` | Real selective-depth speed validation (L31 and L20) |
| `transcender_track_c_dense_family_validation.py` | Focused Llama/Mistral dense-family late-checkpoint validation + KL recon |

### scripts/exploratory/ — Exploratory Probes

| File | Purpose |
|------|---------|
| `transcender_track_c_gemma_advanced_probe.py` | Gemma L20 advanced selective-depth probe with replay repair |
| `transcender_track_c_llama_family_sensitive_probe.py` | Llama family-sensitive continuation probe |
| `transcender_track_c_dense_cache_aware_probe.py` | Llama chunk-repair cache-aware continuation probe |

### scripts/gpt2_poc/ — Historical (GPT-2 PoC)

| File | Purpose |
|------|---------|
| `benchmark.py` | Original GPT-2 perplexity benchmark |
| `benchmark_inference.py` | GPT-2 inference benchmark |
| `benchmark_layer_comparison.py` | GPT-2 layer comparison benchmark |
| `train_and_visualize.py` | Son Router training + visualization |
| `model_injector.py` | Original SGAModel (deprecated, now a shim) |
| `sga_router.py` | Original SGA router (re-export shim) |
| `transcender_injector.py` | Original TranscenderModel source (superseded by package) |
| `transcender_ablation.py` | GPT-2 ablation study |
| `transcender_baseline_debug.py` | Baseline equivalence debugging |
| `transcender_blend_strategy_benchmark.py` | GPT-2 blend strategy benchmark |
| `transcender_dynamic_benchmark.py` | GPT-2 dynamic benchmark |
| `transcender_quality_calibration.py` | GPT-2 quality calibration |
| `transcender_v02_benchmark.py` | GPT-2 v0.2 benchmark |
| `transcender_cascade_benchmark.py` | GPT-2 cascade benchmark (pre-Track B) |

---

## artifacts/

### artifacts/track_a/ — Canonical

| File | Supports Paper |
|------|----------------|
| `transcender_exit_layer_benchmark.json` | Yes — Track A Tables 1 |
| `transcender_top1_agree_benchmark.json` | Yes — Track A blend comparison |
| `hard_exit_ablation.json` | Supporting — hard exit ablation |
| `hard_exit_ablation_post_fix.json` | Supporting — post-fix ablation |
| `postfix_dynamic_benchmark.json` | Supporting — dynamic benchmark |

### artifacts/track_b/ — Negative Baseline

| File | Supports Paper |
|------|----------------|
| `transcender_track_b_benchmark.json` | Yes — Track B Table 2 |

### artifacts/track_c/ — Validated (Gemma)

| File | Supports Paper |
|------|----------------|
| `transcender_track_c_gemma_results.json` | Yes — Track C Table 3 |
| `transcender_track_c_gemma_selective_depth_results.json` | Yes — Track C Table 4 (L31) |
| `transcender_track_c_gemma_selective_depth_L20_results.json` | Yes — Track C Table 5 (L20) |
| `gemma_kl_profile.json` | Yes — KL profiling section |

### artifacts/dense_followup/ — Validated + Exploratory

| File | Supports Paper |
|------|----------------|
| `transcender_track_c_llama3_8b_results.json` | Yes — Track C Table 6 (multi-family) |
| `transcender_track_c_mistral7b_results.json` | Yes — Track C Table 6 (multi-family) |
| `transcender_track_c_gemma_advanced_probe_L20_results.json` | Exploratory — Gemma L20 advanced probe |
| `transcender_track_c_llama3_8b_family_sensitive_probe.json` | Exploratory — Llama family-sensitive probe |
| `transcender_track_c_llama3_8b_cache_aware_probe.json` | Exploratory — Llama cache-aware probe |

### artifacts/recon/ — KL Reconnaissance

| File | Supports Paper |
|------|----------------|
| `recon_llama3_8b.json` | Yes — family-sensitive KL profiles |
| `recon_mistral7b.json` | Yes — family-sensitive KL profiles |

### artifacts/gpt2_poc/ — Historical

| File | Supports Paper |
|------|----------------|
| `transcender_blend_strategy_benchmark.json` | No — GPT-2 PoC only |
| `transcender_quality_calibration.json` | No — GPT-2 PoC only |
| `transcender_v02_benchmark.json` | No — GPT-2 PoC only |
| `baseline_equivalence_debug.json` | No — debugging artifact |
