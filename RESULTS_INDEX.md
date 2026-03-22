# Results Index

All measured artifacts in this repository. No result has been altered from its original benchmark output.

---

## Paper-Supporting Artifacts

These files directly support tables and claims in `paper/main.tex`.

| Artifact | Track | What It Measures |
|----------|-------|------------------|
| `artifacts/track_a/transcender_exit_layer_benchmark.json` | A | Per-prompt and aggregate metrics for L22 and L23 configurations on GPT-OSS 20B |
| `artifacts/track_a/transcender_top1_agree_benchmark.json` | A | Blend strategy comparison (top1_agree, naive blend, fixed exit) on GPT-OSS 20B |
| `artifacts/track_b/transcender_track_b_benchmark.json` | B | Draft-only, full-depth, naive cascade, and Track A comparison on Gemma + GPT-OSS |
| `artifacts/track_c/transcender_track_c_gemma_results.json` | C | Six-mode adaptive benchmark on Gemma 3 4B-IT (fixed exit, top1_agree, naive blend) |
| `artifacts/track_c/transcender_track_c_gemma_selective_depth_results.json` | C | Real selective-depth speed validation at L31 on Gemma 3 4B-IT |
| `artifacts/track_c/transcender_track_c_gemma_selective_depth_L20_results.json` | C | Dedicated L20 selective-depth probe on Gemma 3 4B-IT |
| `artifacts/track_c/gemma_kl_profile.json` | C | Per-layer KL divergence across 34-layer Gemma 3 4B-IT stack |
| `artifacts/dense_followup/transcender_track_c_llama3_8b_results.json` | C | Late-checkpoint (L29) validation on Llama 3.1 8B Instruct 4bit |
| `artifacts/dense_followup/transcender_track_c_mistral7b_results.json` | C | Late-checkpoint (L29) validation on Mistral 7B Instruct v0.3 4bit |
| `artifacts/recon/recon_llama3_8b.json` | C | KL reconnaissance profile for Llama 3.1 8B Instruct 4bit |
| `artifacts/recon/recon_mistral7b.json` | C | KL reconnaissance profile for Mistral 7B Instruct v0.3 4bit |

---

## Exploratory Artifacts

These files document probes that informed research direction but are not cited in the paper.

| Artifact | What It Measures |
|----------|------------------|
| `artifacts/dense_followup/transcender_track_c_gemma_advanced_probe_L20_results.json` | Gemma L20 advanced probe with replay repair — quality improved but realized skipping collapsed |
| `artifacts/dense_followup/transcender_track_c_llama3_8b_family_sensitive_probe.json` | Llama family-sensitive continuation — preserved quality but ~0.03% avg layers saved |
| `artifacts/dense_followup/transcender_track_c_llama3_8b_cache_aware_probe.json` | Llama chunk-repair cache-aware probe — some runtime recovery, did not beat full depth |

---

## Historical Artifacts (GPT-2 PoC)

| Artifact | What It Measures |
|----------|------------------|
| `artifacts/track_a/hard_exit_ablation.json` | GPT-OSS hard exit ablation (pre-fix) |
| `artifacts/track_a/hard_exit_ablation_post_fix.json` | GPT-OSS hard exit ablation (post-fix) |
| `artifacts/track_a/postfix_dynamic_benchmark.json` | GPT-OSS dynamic benchmark (post-fix) |
| `artifacts/gpt2_poc/transcender_blend_strategy_benchmark.json` | GPT-2 blend strategy comparison |
| `artifacts/gpt2_poc/transcender_quality_calibration.json` | GPT-2 quality calibration |
| `artifacts/gpt2_poc/transcender_v02_benchmark.json` | GPT-2 v0.2 benchmark |
| `artifacts/gpt2_poc/baseline_equivalence_debug.json` | GPT-2 baseline equivalence debugging |

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
| `subspace_analysis.png` | Subspace Paradox PCA analysis |
| `threshold_sweep.png` | GPT-2 PoC threshold sweep |
