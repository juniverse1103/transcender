# Repository Map

This document maps `transcender-mlx`, the Apple MLX reference implementation and benchmark suite for the Transcender paper, and classifies every major component by its role in the research.

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
| `transcender/` | Installable Python package: `TranscenderModel`, `SonRouter`, `SonRoutingLoss`, MLX engine components for the current Track A sparse-MoE runs, and shared dense selective-depth policy utilities |
| `scripts/` | All experiment and benchmark scripts, organized by track |
| `artifacts/` | All JSON result files, organized by track |
| `paper/` | Canonical LaTeX draft, bibliography, figures, and retained markdown whitepapers |
| `docs/` | Benchmark summary, archived roadmap/backlog notes, supplementary chapters |
| `models/` | Son Router checkpoint files (GPT-2 PoC, ~195 KB each) |
| `tests/` | pytest smoke tests for the `transcender` package |

---

## scripts/

### scripts/track_a/ — Canonical Sparse-MoE Benchmarks

| File | Purpose |
|------|---------|
| `transcender_engine.py` | MLX runtime engine for GPT-OSS 20B and Qwen3-30B-A3B with entropy-gated exit |
| `transcender_exit_layer_benchmark.py` | Exit-layer frontier benchmark for GPT-OSS (`gpt_oss`) and Qwen3 (`qwen3_moe`) |
| `transcender_top1_agree_benchmark.py` | Earlier GPT-OSS blend-strategy benchmark retained for reference |
| `transcender_recon.py` | GPT-OSS KL reconnaissance depth profiler |
| `transcender_server.py` | FastAPI inference server |
| `transcender_20b_scaffold.py` | GPT-OSS 20B model scaffold and configuration |

### scripts/track_a_gpu/ — Diagnostic GPU Validation

| File | Purpose |
|------|---------|
| `transcender_gpu_reproduction.py` | Manual-reference GPU validation path for Track A under shared full-depth context; reports raw exit metrics separately from composed `top1_agree` metrics |
| `analyze_debug_trace.py` | Summarizes a single-prompt trace JSON to confirm raw divergence and sane fallback behavior |
| `summarize_benchmark.py` | Summarizes aggregate GPU benchmark JSON by layer (`raw_exit_avg_exact_match`, `composed_avg_exact_match`, `avg_top1_agreement_rate`) |
| `compare_benchmarks.py` | Compares multiple benchmark JSON files across models using the same raw/composed aggregate metrics |
| `RUNBOOK.md` | Exact cloud-GPU execution and validation workflow |

This GPU path spans both architecture classes:

- sparse MoE: Qwen3/Qwen2-MoE, GPT-OSS, Mixtral
- dense: Mistral, Llama, Gemma, Gemma2, and Gemma 3 text checkpoints such as `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, and `google/gemma-3-27b-it`

Validation status is narrower than code-path support:

- empirically validated baseline on this GPU path: Qwen3-30B-A3B
- code-path supported, pending real runs: GPT-OSS, Mixtral, Mistral, Llama, older Gemma/Gemma2 text checkpoints, and Gemma 3 text checkpoints by size

Gemma 3 should be read as a dense family with checkpoint-specific size variants such as `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, and `google/gemma-3-27b-it`, not as one undifferentiated model. Multimodal Gemma 3 checkpoints are intentionally out of scope for this manual-reference path.

### scripts/track_b/ — Negative Baseline

| File | Purpose |
|------|---------|
| `transcender_track_b_cascade.py` | Cross-model cascade engine (Gemma draft + GPT-OSS verify) |
| `transcender_track_b_benchmark.py` | Track B benchmark for draft-only, full-depth, and naive cascade modes |

### scripts/track_c/ — Validated Follow-Up

| File | Purpose |
|------|---------|
| `transcender_track_c_gemma_profile.py` | Gemma 3 4B-IT KL depth profiling |
| `transcender_track_c_gemma_benchmark.py` | Gemma six-mode adaptive-depth benchmark |
| `transcender_track_c_gemma_selective_depth.py` | Real selective-depth speed validation (L31 and L20) |
| `transcender_track_c_dense_family_validation.py` | Focused Llama/Mistral dense-family late-checkpoint validation + KL recon |
| `transcender_dense_exit_sweep.py` | Exploratory dense exit-layer agreement sweep for frontier discovery |

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

### artifacts/track_a/ — Current Canonical + Retained Reference

| File | Status | Purpose |
|------|--------|---------|
| `transcender_exit_layer_benchmark_n15.json` | Paper-supporting | Current GPT-OSS 20B N=15 exit-layer frontier artifact |
| `qwen3_30b_a3b_exit_layer_benchmark.json` | Paper-supporting | Current Qwen3-30B-A3B N=15 cross-model MoE frontier artifact |
| `transcender_exit_layer_benchmark.json` | Historical / superseded | Earlier GPT-OSS exit-layer benchmark superseded by `_n15` |
| `transcender_top1_agree_benchmark.json` | Supplementary | Earlier GPT-OSS blend-strategy comparison retained for reference |
| `hard_exit_ablation.json` | Historical / supplementary | GPT-OSS hard-exit ablation (pre-fix) |
| `hard_exit_ablation_post_fix.json` | Historical / supplementary | GPT-OSS hard-exit ablation (post-fix) |
| `postfix_dynamic_benchmark.json` | Historical / supplementary | Earlier GPT-OSS dynamic benchmark |

### artifacts/track_a_gpu/ — Local Diagnostic Outputs

These files are generated locally by the GPU validation flow and are not canonical paper-supporting artifacts.

| File Pattern | Purpose |
|--------------|---------|
| `*.json` | Single-prompt traces and multi-prompt shared-context benchmark summaries from `scripts/track_a_gpu/transcender_gpu_reproduction.py` |

### artifacts/track_b/ — Negative Baseline

| File | Supports Paper |
|------|----------------|
| `transcender_track_b_benchmark.json` | Yes — Track B table and scoped baseline claims |

### artifacts/track_c/ — Validated (Gemma)

| File | Supports Paper |
|------|----------------|
| `transcender_track_c_gemma_results.json` | Yes — Gemma adaptive benchmark |
| `transcender_track_c_gemma_selective_depth_results.json` | Yes — Gemma L31 selective-depth validation |
| `transcender_track_c_gemma_selective_depth_L20_results.json` | Yes — Gemma L20 selective-depth probe |
| `gemma_kl_profile.json` | Yes — Gemma KL profiling section |

### artifacts/dense_followup/ — Validated + Exploratory

| File | Supports Paper |
|------|----------------|
| `transcender_track_c_llama3_8b_results.json` | Yes — Llama dense-family validation |
| `transcender_track_c_mistral7b_results.json` | Yes — Mistral dense-family validation |
| `llama31_8b_exit_sweep.json` | Exploratory — Llama exit-layer agreement sweep |
| `llama31_8b_exit_sweep_summary.csv` | Exploratory — CSV summary for the Llama exit-layer agreement sweep |
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
