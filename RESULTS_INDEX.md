# Results Index

All measured artifacts in `transcender-mlx`, the MLX benchmark and artifact repo for the Transcender paper. JSON files are preserved benchmark outputs.

---

## Paper-Supporting Artifacts

These files directly support tables or claims in `paper/main.tex`.

| Artifact | Track | What It Measures |
|----------|-------|------------------|
| `artifacts/track_a/transcender_exit_layer_benchmark_n15.json` | A | GPT-OSS 20B N=15 exit-layer frontier (L20, L21, L22, L23 full depth) |
| `artifacts/track_a/qwen3_30b_a3b_exit_layer_benchmark.json` | A | Qwen3-30B-A3B N=15 exit-layer frontier (L40, L44, L45, L46, L47 full depth) |
| `artifacts/track_b/transcender_track_b_benchmark.json` | B | Draft-only, full-depth, naive cascade, and Track B aggregate metrics for Gemma + GPT-OSS |
| `artifacts/track_c/transcender_track_c_gemma_results.json` | C | Six-mode adaptive benchmark on Gemma 3 4B-IT (fixed exit, compute-both top1_agree, naive blend) |
| `artifacts/track_c/transcender_track_c_gemma_selective_depth_results.json` | C | Real selective-depth speed validation at L31 on Gemma 3 4B-IT |
| `artifacts/track_c/transcender_track_c_gemma_selective_depth_L20_results.json` | C | Dedicated L20 selective-depth probe on Gemma 3 4B-IT |
| `artifacts/track_c/gemma_kl_profile.json` | C | Per-layer KL divergence across the 34-layer Gemma 3 4B-IT stack |
| `artifacts/dense_followup/transcender_track_c_llama3_8b_results.json` | C | Late-checkpoint (L29) dense-family validation on Llama 3.1 8B Instruct 4bit |
| `artifacts/dense_followup/transcender_track_c_mistral7b_results.json` | C | Late-checkpoint (L29) dense-family validation on Mistral 7B Instruct v0.3 4bit |
| `artifacts/recon/recon_llama3_8b.json` | C | KL reconnaissance profile for Llama 3.1 8B Instruct 4bit |
| `artifacts/recon/recon_mistral7b.json` | C | KL reconnaissance profile for Mistral 7B Instruct v0.3 4bit |

---

## Exploratory / Supplementary Artifacts

These files informed development or provide retained reference data, but they are not the canonical paper-supporting Track A artifacts. This includes locally generated GPU validation outputs from the manual-reference off-MLX diagnostic path.

| Artifact | What It Measures |
|----------|------------------|
| `artifacts/track_a/transcender_top1_agree_benchmark.json` | Earlier GPT-OSS blend-strategy comparison retained for reference |
| `artifacts/track_a_gpu/gpt_oss_20b_gpu_reproduction_n63_bf16.json` | GPT-OSS 20B N=63 GPU validation on NVIDIA H200 (bfloat16, MXFP4→bf16 dequant). L21=0.808 L22=0.879 raw-exit EM. Definitive bf16 run superseding earlier fp16 results. |
| `artifacts/track_a_gpu/qwen3_30b_a3b_gpu_reproduction_n63.json` | Qwen3-30B-A3B N=63 GPU validation on NVIDIA H200 (float16). L45=0.832 L46=0.916 raw-exit EM. Penultimate advantage holds but quality cliff is softer than on MLX. |
| `artifacts/track_a_gpu/*.json` | Local GPU validation traces and shared-context benchmark summaries. These are diagnostic off-MLX outputs, not direct replacements for the canonical MLX Track A artifacts. Run a single-prompt debug trace first; do not interpret aggregate GPU output before verifying raw-exit divergence and sane composed fallback behavior. |
| `artifacts/dense_followup/llama31_8b_exit_sweep.json` | Llama 3.1 8B exit-layer agreement sweep used for dense frontier discovery |
| `artifacts/dense_followup/llama31_8b_exit_sweep_summary.csv` | CSV summary for the Llama 3.1 8B exit-layer agreement sweep |
| `artifacts/dense_followup/transcender_track_c_gemma_advanced_probe_L20_results.json` | Gemma L20 advanced probe with replay repair; quality improved but realized skipping collapsed |
| `artifacts/dense_followup/transcender_track_c_llama3_8b_family_sensitive_probe.json` | Llama family-sensitive continuation probe; preserved quality but realized skipping remained negligible |
| `artifacts/dense_followup/transcender_track_c_llama3_8b_cache_aware_probe.json` | Llama chunk-repair cache-aware probe; some runtime recovery but no practical frontier |

---

## Historical / Superseded Artifacts

These files are kept for auditability but should not be cited as the current public-release evidence.

| Artifact | What It Measures |
|----------|------------------|
| `artifacts/track_a/transcender_exit_layer_benchmark.json` | Earlier GPT-OSS exit-layer benchmark superseded by `transcender_exit_layer_benchmark_n15.json` |
| `artifacts/track_a/hard_exit_ablation.json` | GPT-OSS hard-exit ablation (pre-fix) |
| `artifacts/track_a/hard_exit_ablation_post_fix.json` | GPT-OSS hard-exit ablation (post-fix) |
| `artifacts/track_a/postfix_dynamic_benchmark.json` | Earlier GPT-OSS dynamic benchmark |
| `artifacts/gpt2_poc/transcender_blend_strategy_benchmark.json` | GPT-2 proof-of-concept blend-strategy comparison |
| `artifacts/gpt2_poc/transcender_quality_calibration.json` | GPT-2 proof-of-concept quality calibration |
| `artifacts/gpt2_poc/transcender_v02_benchmark.json` | GPT-2 proof-of-concept v0.2 benchmark |
| `artifacts/gpt2_poc/baseline_equivalence_debug.json` | GPT-2 proof-of-concept baseline-equivalence debugging |

---

## Figures

All PNGs are in `paper/figures/`. They were generated during profiling and benchmarking runs.

| File | Source |
|------|--------|
| `blend_comparison.png` | GPT-2 PoC blend comparison |
| `gemma_kl_profile.png` | Gemma 3 4B-IT KL depth profile |
| `inference_comparison.png` | GPT-2 PoC inference comparison |
| `kl_profile.png` | GPT-OSS 20B KL depth profile |
| `layer_comparison.png` | GPT-2 PoC layer comparison |
| `pareto_frontier.png` | GPT-2 PoC Pareto frontier |
| `pareto_frontier_v2.png` | GPT-2 PoC Pareto frontier v2 |
| `routing_heatmap.png` | GPT-2 PoC routing heatmap |
| `subspace_analysis.png` | Cross-layer subspace mismatch PCA analysis |
| `threshold_sweep.png` | GPT-2 PoC threshold sweep |
