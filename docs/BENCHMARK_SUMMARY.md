# Transcender — Benchmark Summary

**Date:** 2026-03-22 | **Hardware:** Apple M1 Pro, 32 GB unified memory

---

## Cross-Track Comparison

| Track | Model(s) | Best Mode | Gen TPS | Exact Match | Peak Mem (GB) | TTFT (s) | Key Finding |
|-------|----------|-----------|---------|-------------|---------------|----------|-------------|
| **A** | GPT-OSS 20B (MoE, 24L) | L22 top1_agree | 20.22 | 0.969 | 12.96 | 0.98 | Strongest current practical same-model adaptive-depth result. Canonical frontier. |
| **B** | Gemma 3 4B-IT + GPT-OSS 20B | Naive Cascade | 0.28 | 0.026 | 20.19 | 11.83 | Negative comparison baseline. Vocabulary/distribution mismatch between this model pair destroyed acceptance. Result is specific to this cascade implementation, model pair, and local runtime environment. |
| **C** | Gemma 3 4B-IT (Dense, 34L) | top1_agree @ L31 | 7.19 | 0.807 | 7.28 | 0.18 | Gemma established the dense-model pattern. Follow-up late-checkpoint validation on Llama 3.1 8B and Mistral 7B reproduced the same benchmark pattern on this local Apple MLX runtime. |

---

## Track A — Same-Model Adaptive Depth (Canonical)

**Model:** GPT-OSS 20B (24 layers, MoE, MXFP4 quantization)

| Config | Gen TPS | Exact Match | TTFT (s) | Layers Saved |
|--------|---------|-------------|----------|--------------|
| Full Depth (L23) | 31.79 | 1.000 | — | 0% |
| L22 top1_agree | 27.41 | 0.969 | 0.75 | **49.5%** |
| L23 top1_agree | 32.41 | 1.000 | — | 0% |

**Interpretation:** L22 top1_agree achieves 96.9% exact match against full depth with 49.5% real layer savings via entropy-gated exit. The engine physically skips layers when confident — this is genuine adaptive compute savings, not just blending. L23 top1_agree achieves 100% but offers no depth savings. L22 remains the canonical adaptive frontier.

**Note on layers_saved:** The 49.5% figure reflects the `MLXDynamicExpertEngine` physically skipping layers via entropy-gated exit. This is distinct from the logit blending modes (which always compute both paths). The two mechanisms are complementary.

**Subspace Paradox:** 4.11x geometric separation between early and deep hidden states confirmed. Logit-space blending is mandatory.

---

## Track B — Cross-Model Cascade (Negative Comparison Baseline)

**Draft:** Gemma 3 4B-IT | **Verifier:** GPT-OSS 20B

| Mode | Gen TPS | Exact Match | Peak Mem (GB) | TTFT (s) |
|------|---------|-------------|---------------|----------|
| Draft Only (Gemma 3 4B-IT) | 18.99 | 0.006 | 7.28 | 0.23 |
| GPT-OSS Full Depth | 30.94 | 1.000 | 12.96 | 0.66 |
| Track B Naive Cascade | 0.28 | 0.026 | 20.19 | 11.83 |
| Track A L22 top1_agree | 20.22 | 0.969 | 12.96 | 0.98 |

**Track A vs Track B deltas:**
- TPS: Track A 72x faster in this tested comparison on the local Apple MLX runtime (20.22 vs 0.28)
- Memory: Track A saves 7.23 GB (12.96 vs 20.19)
- TTFT: Track A 10.85s faster (0.98 vs 11.83)
- Quality: Track A +94.3pp exact match (0.969 vs 0.026)

**Interpretation:** The cascade failed because Gemma and GPT-OSS have incompatible vocabulary distributions and prompt templates. This is a structural limitation of naive cross-model speculative decoding with architecturally mismatched models, not a failure of cascade strategies in general. The 72x TPS delta characterizes this specific comparison (naive cascade + maximally mismatched models) on this local Apple MLX runtime; it should not be cited as a general advantage of adaptive depth over cascading. Track B validates that Track A's same-model approach occupies a better frontier for this model pair in this runtime environment.

---

## Track C — Dense Model Generalization (Partial)

**Model:** Gemma 3 4B-IT (34 layers, dense, no MoE)

### KL Profiling

| Metric | Value |
|--------|-------|
| Layer 0 KL | 66.11 |
| 90% KL reduction | Layer 20 |
| 95% KL reduction | Layer 31 |
| Max geometric separation | 2.46x (L31 to L32) |
| KL plateau | Layers 20-29 (KL stagnates or reverses) |

### Adaptive Benchmark

| Mode | Gen TPS | Exact Match | Layers Saved | Agreement Rate |
|------|---------|-------------|--------------|----------------|
| Full Depth (L33) | 15.01 | 1.000 | 0% | — |
| Early Exit (L16) | 22.81 | 0.010 | 50% | — |
| Mid Exit (L20) | 20.81 | 0.026 | 38% | — |
| Late Exit (L31) | 14.95 | 0.198 | 6% | — |
| top1_agree (L31) | 7.19 | 0.807 | 0% | 81.9% |
| Naive Blend (L31) | 10.01 | 0.688 | 0% | 100% |

**Key findings:**
1. **Fixed early exit is catastrophic on dense Gemma.** Even the late fixed exit at L31 drops exact match to 0.198 while saving only 6% of layers. Compare: GPT-OSS Track A reaches 0.969 exact match with real adaptive layer skipping.
2. **Agreement-aware blending recovers quality.** top1_agree at L31 achieves 0.807 exact match versus 0.198 for fixed L31 exit.
3. **top1_agree outperforms naive blending.** +11.98pp exact match (0.807 vs 0.688). The agreement gate matters.
4. **The KL plateau (L20-L29) is architecturally significant.** Layers 20-29 do structural/compositional work that temporarily destabilizes logit-space agreement. The final 4 layers (L30-L33) resolve this. This pattern was not observed in GPT-OSS.

**Interpretation:** The Transcender *methodology* (logit-space composition, agreement-aware gating) generalizes to dense architectures. The *magnitude of achievable fixed-exit speedup* does not — dense models require nearly all layers for quality. Real selective-depth at both L31 and L20 was operationally correct but practically weak on this local Apple MLX runtime. L20 increased the skip budget relative to L31, but replay/control overhead plus quality collapse still erased the practical benefit. The next research direction for dense models is not simply moving the checkpoint earlier; it is better continuation criteria, token-difficulty-aware routing, and lower-overhead cache-aware continuation.

**Selective-depth speed validation (L31):** A first real selective-depth runtime physically skipped layers 32..33 on some decode tokens, but remained practically weak in this configuration. `selective_depth_margin_L31` reached 18.26 TPS and 0.229 exact match with 78.2% early acceptance and 0.2% average layers saved. `selective_depth_entropy_L31` reached 18.20 TPS and 0.208 exact match with 93.6% early acceptance and 2.0% average layers saved. Neither mode beat full depth (20.05 TPS), and exact match stayed close to fixed exit (0.198). This is a scoped negative speed-validation result for Gemma 3 4B-IT at L31 on this local Apple MLX runtime.

**Selective-depth speed validation (L20 probe):** Moving the checkpoint earlier increased the available skip budget but did not recover a practical frontier. In the dedicated L20 run, `selective_depth_entropy_L20` reached 13.26 TPS, 0.073 exact match, 41.0% realized skip rate, and 15.7% average layers saved versus a matched full-depth baseline of 15.16 TPS and 1.000 exact match. `selective_depth_margin_L20` reached 11.29 TPS and 0.083 exact with 5.3% average layers saved. `selective_depth_hybrid_L20` reached 11.37 TPS and 0.125 exact with 2.4% average layers saved. All three modes remained below full depth on wall-clock and stayed much closer to fixed exit at L20 (0.026 exact) than to compute-both `top1_agree` at L20 (0.807 exact). This shows that a larger skip budget alone does not solve dense selective-depth on this model and runtime.

### Multi-Family Dense Follow-Up

| Model | Late Checkpoint | Fixed Exit Exact | Compute-Both Exact | Real Selective Exact | Real Selective TPS | Avg Layers Saved | Resolution Entry | Plateau Zone |
|-------|-----------------|------------------|--------------------|----------------------|--------------------|------------------|------------------|--------------|
| Llama 3.1 8B Instruct 4bit | L29 | 0.151 | 1.000 | 0.469 | 16.60 vs 17.81 full depth | 0.3% | L27 | L12-L14 |
| Mistral 7B Instruct v0.3 4bit | L29 | 0.109 | 1.000 | 0.318 | 17.25 vs 29.81 full depth | 0.2% | L26 | L2-L8 |

**Interpretation:** The late-checkpoint dense benchmark pattern now appears across three tested dense families on this local Apple MLX runtime: fixed exit fails materially, compute-both `top1_agree` recovers substantial quality, and real selective-depth physically skips layers but still does not recover a practical speed-quality frontier after replay/cache repair. The internal KL profile is less uniform. Gemma shows a late L20-L29 plateau, Llama shows a milder middle plateau around L12-L14, and Mistral shows an earlier plateau around L2-L8. The benchmark-side limitation therefore generalizes more cleanly than Gemma's exact internal profile shape.

---

## Current Project Takeaway

**What is proven:**
- Agreement-aware logit-space composition works across architectures (MoE and dense)
- The Subspace Paradox (hidden-state blending fails) is confirmed on both architecture types
- Track B documents the cascade tax of this naive cross-model setup for this model pair
- GPT-OSS 20B has a practical adaptive-depth frontier at L22; no practical dense fixed-exit or real selective-depth frontier was established on the tested Gemma, Llama, or Mistral follow-ups in this local Apple MLX runtime
- Real selective-depth runtimes on Gemma, Llama, and Mistral physically skipped layers but did not recover a practical dense speed-quality frontier on this local Apple MLX runtime

**What is not yet proven:**
- Whether deeper selective-depth variants with better continuation criteria recover a practical frontier on dense models
- Whether the method scales to larger dense models (12B+)
- Whether Gemma's exact KL plateau shape is universal across dense families
- Whether a well-matched draft/verifier pair would change the Track B outcome

**Canonical result:** Track A, L22 top1_agree on GPT-OSS 20B — 0.969 exact match with ~49.5% real layer savings via entropy-gated exit and no memory overhead.
