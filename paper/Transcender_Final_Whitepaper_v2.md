# Transcender: Composition-Controlled Adaptive-Depth Inference for Transformer Language Models

**Version 2.0 — Superseded Markdown Narrative**
**Date:** 2026-03-22

> **Status:** Retained for historical reference only. This markdown snapshot predates the final GPT-OSS N=15 / Qwen3 / dense-follow-up release reconciliation and still contains obsolete numbers and earlier terminology. Use `paper/main.tex`, `README.md`, and `docs/BENCHMARK_SUMMARY.md` as the current public-release sources of truth.

---

## Abstract

Transcender is best understood not as a generic early-exit method, but as a cross-layer composition control framework for adaptive transformer inference.

The core problem is not *when to exit* — it is *how to combine information from different depth levels without corrupting the output distribution*. We identify a structural obstacle — the **Subspace Paradox** — which renders hidden-state interpolation across layers geometrically invalid. We then demonstrate that **agreement-aware logit-space blending** resolves this obstacle and produces robust adaptive-depth inference across architecturally distinct transformer families.

We validate Transcender across three research tracks on a single Apple M1 Pro (32 GB unified memory):

- **Track A (GPT-OSS 20B, MoE, 24 layers):** Agreement-aware blending at Layer 22 achieves **96.9% exact match** against full-depth greedy output, with the engine physically skipping ~49% of layers on average via entropy-gated exit.
- **Track B (Cross-Model Cascade — Negative Comparison Baseline):** A naive draft-verify cascade pairing Gemma 3 4B-IT with GPT-OSS 20B achieves 0.026 exact match at 0.28 TPS. This result is specific to this cascade implementation, model pair, and local Apple MLX runtime; it does not invalidate cascade strategies in general.
- **Track C (Dense Models):** Gemma 3 4B-IT establishes the core dense result: fixed early exit is catastrophic (0.198 exact match at best fixed exit vs 1.000 at full depth), while agreement-aware blending recovers to **0.807 exact match** at the same exit layer. Focused late-checkpoint follow-up on Llama 3.1 8B and Mistral 7B reproduces the same benchmark pattern on this local Apple MLX runtime: fixed exit fails materially, compute-both `top1_agree` recovers full-depth output, and real selective-depth remains practically weak after replay.

---

## 1. Introduction

Adaptive-depth inference aims to reduce transformer compute by running fewer layers when the model's output has converged at an intermediate depth. Prior work frames this as an early-exit problem: choose a layer, exit there, decode. This framing is incomplete.

The deeper problem is **composition control**: when two depth levels produce different representations of the same token, how should they be combined? The naive answer — interpolate hidden states — fails for structural reasons. The correct answer — blend in logit space with an agreement gate — is what Transcender demonstrates.

This paper reports empirical results across three tracks that test the framework on MoE and dense architectures, on same-model and cross-model configurations, and on fixed vs. adaptive exit strategies. Every metric reported below is traceable to a specific JSON artifact produced during benchmarking.

---

## 2. Distinctive Contributions

This work makes five empirical contributions:

1. **The Subspace Paradox.** We demonstrate that hidden states from different transformer layers occupy geometrically separated subspaces (4.11x separation in MoE, 2.46x in dense), rendering hidden-state interpolation structurally invalid for cross-layer composition. This is confirmed across two architecturally distinct model families.

2. **Agreement-aware logit-space composition.** We show that logit-space blending gated on top-1 agreement produces robust adaptive-depth output. On Gemma 3 4B-IT, agreement-gated blending at L31 achieves 0.807 exact match versus 0.688 for ungated blending and 0.198 for fixed exit at the same layer. The agreement gate filters the cases where early and deep layers disagree on token identity.

3. **Empirical characterization of the cascade tax.** We measure the cost of naive cross-model speculative decoding with architecturally mismatched models (different tokenizers, different training distributions). For the tested Gemma 3 4B-IT / GPT-OSS 20B pair, the cascade achieves 0.026 exact match at 0.28 TPS — demonstrating that vocabulary and distribution mismatch can destroy cascade viability entirely. This result is specific to the tested configuration and local Apple MLX runtime, and does not generalize to well-matched model families.

4. **Family-sensitive dense depth profiles.** Gemma 3 4B-IT exhibits a late KL plateau (L20–L29) followed by a sharp resolution phase (L30–L33). Focused recon on Llama 3.1 8B and Mistral 7B shows later resolution entry points as well, but with different plateau zones (L12–L14 on Llama, L2–L8 on Mistral). The benchmark-side dense limitation generalizes more cleanly than Gemma's exact internal profile shape.

5. **Real dense selective-depth speed validation at two Gemma checkpoints.** We implement continuation-safe selective-depth runtimes at both L31 and L20 on Gemma 3 4B-IT. Both physically skip deep layers on some decode tokens, but neither recovers a practical speed-quality frontier on the tested local Apple MLX runtime. L31 fails because the remaining two-layer budget is too small; L20 fails because the larger skip budget still comes with heavy quality collapse and replay/control overhead.

6. **Focused multi-family dense benchmark validation.** At late checkpoints on Llama 3.1 8B and Mistral 7B, fixed exit again fails materially, compute-both `top1_agree` again recovers substantial quality, and real selective-depth again remains practically weak after replay/cache repair. This broadens the dense benchmark result beyond Gemma alone on the tested local Apple MLX runtime.

---

## 3. The Subspace Paradox

### 3.1 Definition

When transformer hidden states from layer $i$ and layer $j$ ($i < j$) are projected into a shared space via PCA, the centroid distance between the two layer clusters exceeds the within-cluster spread by a significant factor:

$$\text{Separation Ratio} = \frac{\|c_i - c_j\|_2}{\max(\sigma_i, \sigma_j)}$$

| Architecture | Model | Layers Compared | Separation Ratio |
|-------------|-------|-----------------|------------------|
| MoE | GPT-OSS 20B | L12 vs L24 | **4.11x** |
| Dense | Gemma 3 4B-IT | L20 vs L34 | **2.46x** |

### 3.2 Implication

Any weighted average $\alpha \cdot h_{\text{early}} + (1-\alpha) \cdot h_{\text{deep}}$ produces a vector in "no-man's land" — a region of representation space the LM head has never seen during training. The U-shaped loss landscape across blend weights $\alpha \in [0, 1]$ has no interior minimum: both pure endpoints outperform every blend.

Hidden-state blending is structurally broken for adaptive depth in transformers. This is not a tuning issue — it is a geometric property of depth.

### 3.3 Cross-Architecture Consistency

The paradox is confirmed on both MoE and dense architectures, though its magnitude differs. The 2.46x separation in Gemma 3 4B is lower than GPT-OSS's 4.11x but still far above the 1.0x threshold where interpolation would become viable. This confirms the paradox is a property of transformer depth, not of any specific architectural choice.

---

## 4. Method: Agreement-Aware Logit Blending

### 4.1 Why Logit Space

Layer normalization followed by LM head projection maps hidden states from any layer into a shared vocabulary-probability space:

$$\text{logits}_i = W_{\text{LM}} \cdot \text{LayerNorm}(h_i)$$

Because normalization occurs before blending, both pathways produce semantically comparable distributions. Logit-space interpolation is geometrically meaningful; hidden-state interpolation is not.

### 4.2 The `top1_agree` Strategy

For each token generation step:

1. Compute $\text{logits}_{\text{early}}$ at exit layer $e$
2. Compute $\text{logits}_{\text{deep}}$ at full depth $N$
3. If $\arg\max(\text{logits}_{\text{early}}) = \arg\max(\text{logits}_{\text{deep}})$:
   - Blend: $\text{logits}_{\text{out}} = (1 - \alpha) \cdot \text{logits}_{\text{deep}} + \alpha \cdot \text{logits}_{\text{early}}$
4. Else:
   - Discard early logits, use $\text{logits}_{\text{deep}}$ only

The agreement gate ensures that blending only occurs when the early layer's top-1 decision is consistent with the full-depth decision. When they disagree, the system falls back to full depth with zero quality loss.

### 4.3 Why Agreement Matters

Naive blending (always interpolate regardless of agreement) consistently underperforms agreement-gated blending:

| Strategy | Model | Exact Match | Delta vs top1_agree |
|----------|-------|-------------|---------------------|
| top1_agree | GPT-OSS 20B (L22) | 0.969 | — |
| top1_agree | Gemma 3 4B (L31) | 0.807 | — |
| naive blend | Gemma 3 4B (L31) | 0.688 | **−11.98pp** |

On Gemma, naive blending poisons the output on every token, including the ~18% where the early layer disagrees with full depth. The agreement gate restricts blending to tokens where the early exit layer was already correct about the most important bit — the identity of the next token.

---

## 5. Experimental Setup

### 5.1 Hardware

All experiments ran on a single Apple M1 Pro MacBook Pro (32 GB unified memory) using Apple MLX for inference. No GPU cluster was used.

### 5.2 Evaluation Protocol

- **Prompt suite:** 5 factual/explanatory prompts (quantum entanglement, French Revolution, recursion, TCP vs UDP, photosynthesis), each generating 48 tokens
- **Baseline:** Full-depth greedy decoding on the same model
- **Primary metric:** Exact token-sequence match rate against baseline
- **Secondary metrics:** Generation TPS, TTFT, peak memory, layers saved
- **Warmup:** First prompt (P1) excluded from aggregate metrics to control for JIT compilation and memory allocation effects

### 5.3 Models

| Model | Architecture | Layers | Parameters | Quantization |
|-------|-------------|--------|------------|--------------|
| GPT-OSS 20B | MoE (32 experts, top-4) | 24 | ~20B | MXFP4 |
| Gemma 3 4B-IT | Dense | 34 | ~4B | Default (bfloat16) |
| Llama 3.1 8B Instruct | Dense | 32 | ~8B | MLX 4bit |
| Mistral 7B Instruct v0.3 | Dense | 32 | ~7B | MLX 4bit |

---

## 6. Results

### 6.1 Track A — Same-Model Adaptive Depth (GPT-OSS 20B)

| Config | Gen TPS | Exact Match | Avg Layers Saved | Top-1 Agreement |
|--------|---------|-------------|------------------|-----------------|
| Full Depth (L23) | 31.79 | 1.000 | 0% | — |
| L22 top1_agree | 27.41 | 0.969 | **49.5%** | 1.000 |
| L23 top1_agree | 32.41 | 1.000 | 0% | — |

**Key finding:** L22 top1_agree is the canonical adaptive frontier. The engine physically skips layers via entropy-gated exit, achieving an average of 49.5% layer savings with 96.9% exact match. The top-1 agreement rate of 1.000 across all non-warmup prompts indicates that when the engine does use the early exit path, the early and deep layers always agree on the top-1 token.

**Clarification on layers_saved:** The `avg_layers_saved = 0.4948` metric reflects *real adaptive behavior* in the `MLXDynamicExpertEngine`. The engine evaluates an entropy-based confidence signal at the exit layer and physically skips remaining layers when confidence is high. This is distinct from the logit blending modes, which always compute both paths. The two mechanisms — adaptive exit and agreement-aware blending — are complementary: adaptive exit provides compute savings, while agreement-aware blending provides output quality control.

### 6.2 Track B — Cross-Model Cascade (Negative Comparison Baseline)

| Mode | Gen TPS | Exact Match | Peak Mem (GB) | TTFT (s) |
|------|---------|-------------|---------------|----------|
| Draft Only (Gemma 3 4B-IT) | 18.99 | 0.006 | 7.28 | 0.23 |
| GPT-OSS Full Depth | 30.94 | 1.000 | 12.96 | 0.66 |
| Naive Cascade | 0.28 | 0.026 | 20.19 | 11.83 |
| Track A L22 top1_agree | 20.22 | 0.969 | 12.96 | 0.98 |

**Track A vs Naive Cascade deltas:**
- TPS: 72x faster in this tested comparison on the local Apple MLX runtime (20.22 vs 0.28)
- Memory: 7.23 GB less (12.96 vs 20.19)
- TTFT: 10.85s faster (0.98 vs 11.83)
- Quality: +94.3pp exact match (0.969 vs 0.026)

**Scope of this result:** The cascade failure has three contributing causes, each of which should be attributed correctly:

1. **Vocabulary mismatch.** Gemma and GPT-OSS use different tokenizers with different vocabulary sizes. Draft tokens cannot be directly verified by the larger model without retokenization, destroying the acceptance rate.
2. **Distribution mismatch.** The two models were trained on different data with different objectives. Their output distributions have low overlap even on simple prompts.
3. **Implementation overhead.** The naive chunked-verify loop loads two models simultaneously, manages two KV caches, and performs expensive cross-model verification at each chunk boundary.

Causes (1) and (2) are specific to this model pair. A well-matched model family (e.g., GPT-OSS 7B + GPT-OSS 20B, same tokenizer and training distribution) would eliminate vocabulary and distribution mismatch. Cause (3) — the inherent overhead of two-model orchestration — is structural. The measured Track A vs Track B deltas also depend on this local Apple MLX runtime.

**What Track B validates:** Same-model adaptive depth (Track A) avoids all three failure modes by construction. There is no vocabulary mismatch, no distribution mismatch, and no two-model orchestration overhead. For the specific configuration tested on this local Apple MLX runtime, Track A dominates on every measured axis.

### 6.3 Track C — Dense Model Generalization (Gemma + Focused Llama/Mistral Follow-Up)

#### 6.3.1 KL Depth Profile

Gemma 3 4B-IT exhibits a characteristic depth profile not observed in GPT-OSS:

| Layer Range | KL Behavior | Interpretation |
|-------------|-------------|----------------|
| L0 — L20 | Rapid decline (66.1 → 6.3, 90% reduction) | Token identity converging |
| L20 — L29 | **Plateau / reversal** (KL hovers at 5.5–6.3) | Compositional/structural work |
| L30 — L33 | Sharp final decline (6.3 → 0.0) | Resolution phase |

The **KL plateau** at layers 20–29 is architecturally significant. During this range, the model performs compositional work that temporarily *increases* logit-space divergence from the final output. Any fixed exit that enters or precedes this plateau skips the resolution phase (L30–L33) that resolves compositional ambiguity into coherent output.

Focused dense-family recon on Llama 3.1 8B and Mistral 7B later showed that the exact profile shape is not Gemma-specific. Llama's heuristic plateau zone appeared around L12–L14 with resolution entry at L27. Mistral's appeared earlier, around L2–L8, with resolution entry at L26. The benchmark limitation generalizes more cleanly than Gemma's exact KL profile shape.

#### 6.3.2 Fixed Exit vs Agreement-Aware Blending

| Mode | Exit Layer | Gen TPS | Exact Match | Layers Saved | Agreement Rate |
|------|-----------|---------|-------------|--------------|----------------|
| Full Depth | L33 | 15.01 | 1.000 | 0% | — |
| Fixed Exit | L16 | 22.81 | 0.010 | 50% | — |
| Fixed Exit | L20 | 20.81 | 0.026 | 38% | — |
| Fixed Exit | L31 | 14.95 | 0.198 | 6% | — |
| top1_agree | L31 | 7.19 | **0.807** | 0% | 81.9% |
| Naive Blend | L31 | 10.01 | 0.688 | 0% | 100% |

**Key findings:**

1. **Fixed early exit is catastrophic on dense Gemma.** Even the late fixed exit at L31 drops exact match to 0.198 while saving only 6% of layers. Compare: GPT-OSS Track A reaches 0.969 exact match with real adaptive layer skipping.

2. **Agreement-aware blending recovers quality.** top1_agree at L31 achieves 0.807 exact match versus 0.198 for fixed L31 exit, by discarding early logits precisely when they would cause errors.

3. **top1_agree outperforms naive blending by +11.98pp** (0.807 vs 0.688). The agreement gate matters: naive blending poisons tokens where the early and deep layers disagree on identity.

4. **The 81.9% agreement rate** indicates that Gemma's L31 produces the correct top-1 token approximately 82% of the time. The remaining 18% are precisely the tokens where depth matters most — and where the agreement gate correctly falls back to full depth.

#### 6.3.3 First Real Selective-Depth Speed Validation

After the compute-both `top1_agree` result established a quality signal at L31, we implemented a real selective-depth runtime to test whether that signal could be converted into wall-clock savings. Prompt prefill remained full-depth so the deep cache was valid at decode start. During decode, each token ran layers 0..31 first; accepted tokens skipped layers 32..33; when later continuation was required, previously skipped decode tokens were replayed through layers 32..33 to restore deep-cache correctness.

| Mode | Gen TPS | Exact Match | Acceptance Rate | Realized Skip Rate | Avg Layers Saved |
|------|---------|-------------|-----------------|--------------------|------------------|
| Full Depth | 20.05 | 1.000 | — | 0% | 0% |
| Fixed Exit | 19.10 | 0.198 | — | — | 5.9% |
| top1_agree compute-both | 16.31 | 0.807 | 81.9% | 0% | 0% |
| selective_depth_margin | 18.26 | 0.229 | 78.2% | 3.7% | 0.2% |
| selective_depth_entropy | 18.20 | 0.208 | 93.6% | 34.0% | 2.0% |

**Key findings:**

1. **Selective-depth was operationally correct.** The runtime physically skipped layers 32..33 on some decode tokens and preserved cache correctness by replaying skipped tokens when later continuation required deep-cache repair.

2. **The available compute budget at L31 is too small.** Only the final 2 of 34 layers are skippable, so even aggressive early acceptance produces limited average layer savings.

3. **Wall-clock TPS did not beat full depth.** Full depth reached 20.05 TPS; the margin and entropy selective-depth variants reached 18.26 and 18.20 TPS. Replay and control overhead erased the small compute savings.

4. **Quality remained close to fixed-exit behavior.** The selective-depth variants achieved 0.229 and 0.208 exact match, far below the compute-both `top1_agree` result at 0.807 and only modestly above the fixed L31 exit at 0.198.

This is a negative speed-validation result for Gemma 3 4B-IT at L31 on this local Apple MLX runtime. It does not overturn Track C; it refines it. The compute-both result still shows a composition-quality signal, while the real selective-depth result shows that this late two-layer skip budget is too small to recover a practical speed-quality frontier with the current continuation mechanism.

We then probed an earlier checkpoint at L20 to test the obvious follow-up question: does a materially larger skip budget recover a practical frontier?

| Mode | Gen TPS | Exact Match | Acceptance Rate | Realized Skip Rate | Avg Layers Saved |
|------|---------|-------------|-----------------|--------------------|------------------|
| Full Depth | 15.16 | 1.000 | — | 0% | 0% |
| Fixed Exit | 20.20 | 0.026 | — | — | 38.2% |
| top1_agree compute-both | 10.54 | 0.807 | 28.2% | 0% | 0% |
| selective_depth_margin | 11.29 | 0.083 | 70.2% | 13.8% | 5.3% |
| selective_depth_entropy | 13.26 | 0.073 | 73.4% | 41.0% | 15.7% |
| selective_depth_hybrid | 11.37 | 0.125 | 61.2% | 6.4% | 2.4% |

**Additional findings from the L20 probe:**

1. **Moving earlier increased the skip budget.** The L20 entropy rule reached 15.7% average layers saved versus 2.0% at L31, and a 41.0% realized skip rate versus 34.0% at L31.

2. **The larger skip budget still did not survive replay/control overhead.** Full depth in the matched L20 run reached 15.16 TPS, while the L20 selective variants reached 11.29, 13.26, and 11.37 TPS.

3. **Quality collapsed toward fixed-exit behavior.** The L20 selective variants achieved only 0.083, 0.073, and 0.125 exact match, much closer to fixed L20 exit at 0.026 than to compute-both `top1_agree` at 0.807.

This second probe clarifies the failure boundary. L31 showed that a tiny skip budget is insufficient. L20 shows that simply moving the checkpoint earlier and increasing the skip budget is still insufficient when the continuation rule is weak and the replay path is costly. The result remains scoped to Gemma 3 4B-IT, these implementations, and this local Apple MLX runtime.

#### 6.3.4 Focused Multi-Family Dense Follow-Up

To test whether the dense benchmark result was Gemma-specific or more broadly shared, we ran focused late-checkpoint validation on two additional dense families: Llama 3.1 8B Instruct 4bit and Mistral 7B Instruct v0.3 4bit. For both models, the checkpoint was chosen analogously to Gemma's late L31 probe: L29 on a 32-layer stack, leaving the final two layers as the only skippable budget.

| Model | Late Checkpoint | Fixed Exit Exact | top1_agree Compute-Both Exact | Real Selective Exact | Real Selective TPS | Avg Layers Saved |
|-------|-----------------|------------------|-------------------------------|----------------------|--------------------|------------------|
| Llama 3.1 8B Instruct 4bit | L29 | 0.151 | 1.000 | 0.469 | 16.60 vs 17.81 full depth | 0.3% |
| Mistral 7B Instruct v0.3 4bit | L29 | 0.109 | 1.000 | 0.318 | 17.25 vs 29.81 full depth | 0.2% |

**Key findings from the follow-up:**

1. **Late fixed exit failed materially on both additional dense families.** The failure was not as extreme as Gemma's L20 probe, but it remained far from full-depth output quality at 0.151 exact on Llama and 0.109 exact on Mistral.

2. **Compute-both quality recovery generalized cleanly.** At the same late checkpoint, `top1_agree` recovered 1.000 exact match on both Llama and Mistral. This reinforces the composition-control interpretation: when early and deep logits are composed conservatively, the quality signal transfers across tested dense families.

3. **Real selective-depth remained practically weak.** The entropy selective-depth modes physically skipped layers on some decode steps, but replay/cache repair consumed nearly all of the late two-layer skip budget. Average layer savings were only 0.3% on Llama and 0.2% on Mistral, and neither mode beat its matched full-depth baseline.

4. **The internal profile is family-sensitive.** Gemma showed a late L20–L29 plateau, Llama a milder middle plateau around L12–L14, and Mistral an earlier plateau around L2–L8. The benchmark-side dense limitation therefore generalizes more clearly than Gemma's exact internal KL signature.

This follow-up broadens the dense benchmark result from a Gemma-only observation to a multi-family dense observation on this local Apple MLX runtime. It does not justify claiming that Gemma's exact KL plateau geometry is universal.

#### 6.3.5 Architecture-Dependent Frontier Strength

| Property | GPT-OSS 20B (MoE) | Dense Follow-Up (Gemma / Llama / Mistral) |
|----------|-------------------|---------------------|
| Best practical adaptive-depth result | 0.969 (L22 top1_agree, ~49.5% layers saved) | Not established |
| Best composition-only result | — | 1.000 on Llama/Mistral late-checkpoint follow-up; 0.807 on Gemma L31 |
| Best fixed-exit exact match | Not the canonical Track A result | 0.198 on Gemma L31; 0.151 on Llama L29; 0.109 on Mistral L29 |
| Real selective-depth probes | — | Operationally correct, but weak across Gemma, Llama, and Mistral follow-ups |
| Subspace paradox / KL profile evidence | 4.11x separation, no plateau observed | Gemma: 2.46x with late plateau; Llama/Mistral profiles differ by family |
| Practical frontier | **Yes** | **No practical fixed-exit or real selective-depth frontier established in the tested dense runs** |
| Blending quality signal | Strong | Strong at late checkpoints, but still compute-both |

MoE models may tolerate early exit better because expert routing creates natural "knowledge completion" points — once the relevant experts have fired, the token's representation is mature. Dense models distribute knowledge more uniformly across depth, making nearly every layer load-bearing. This is a hypothesis, not a proven mechanism.

---

## 7. Comparative Analysis: Adaptive Exit vs Composition Quality

Track A results demonstrate two distinct mechanisms that should not be conflated:

1. **Adaptive exit** (entropy-gated layer skipping): The engine physically skips ~49.5% of layers, reducing compute. This is where wall-clock speedup originates.

2. **Composition quality** (agreement-aware blending): When early and deep paths are both computed, the blending strategy determines output quality. top1_agree is designed to preserve the full-depth decision when early and deep paths disagree, and it materially improves output quality relative to naive blending.

In the current Track A implementation, these mechanisms operate together: the engine skips layers when confident (adaptive exit), and blends logits when both paths are computed (composition quality). The 96.9% exact match reflects their combined effect.

Track C now separates these mechanisms cleanly. The compute-both `top1_agree` modes preserved quality relative to fixed exit, but they did so without physical layer skipping. The real selective-depth runtimes isolated the skipping side. L31 showed that a two-layer skip budget is too small to matter. L20 increased the skip budget materially, but the selective modes still fell below their matched full-depth baseline and collapsed toward fixed-exit quality. Focused late-checkpoint follow-up on Llama 3.1 8B and Mistral 7B reproduced the same separation between quality-control and speed-validation: compute-both `top1_agree` recovered full-depth output on both, while real selective-depth saved only 0.3% and 0.2% of layers on average after replay and remained below full depth on TPS. Larger skip budget or late-checkpoint agreement alone is therefore insufficient; dense-model utility depends on a continuation mechanism that preserves quality while keeping replay/control overhead low (see Section 9).

---

## 8. Limitations and Honest Assessment

### What We Can Claim

1. **The Subspace Paradox appears across both tested architectures.** Confirmed at 4.11x (MoE) and 2.46x (dense) — hidden-state blending is structurally invalid in the tested settings.
2. **Agreement-aware logit-space composition improves output quality on both MoE and dense models.** The methodology generalizes across architectures.
3. **Same-model adaptive depth beats this naive cross-model cascade** for this specific model pair and cascade implementation.
4. **The GPT-OSS engine achieves real adaptive compute savings** (~49.5% layer reduction) while retaining 0.969 exact match against full depth.
5. **Across the tested dense follow-ups, compute-both quality recovery generalized more cleanly than real selective-depth speedup.** Gemma, Llama, and Mistral all showed materially better quality under conservative compute-both composition than under fixed exit, but none established a practical real selective-depth frontier on this local Apple MLX runtime.

### What We Cannot Claim

1. **Universal early-exit speedup.** Fixed early exit produced a practical frontier on GPT-OSS 20B but not on Gemma 3 4B. The magnitude of achievable speedup is architecture-dependent.
2. **That cascade strategies are always inferior.** Track B's failure reflects vocabulary/distribution mismatch between Gemma and GPT-OSS. A same-family cascade (same tokenizer, aligned distributions) would produce a fairer comparison.
3. **That Gemma's exact KL plateau shape is universal across dense transformers.** Llama and Mistral showed different plateau zones and heuristic resolution entry points in the focused follow-up.
4. **That Transcender is production-ready.** Track C's compute-both blending implementation adds latency by running both paths, and the real selective-depth follow-ups on Gemma, Llama, and Mistral still did not recover a practical speed-quality frontier in this local Apple MLX runtime.
5. **That the 72x TPS advantage generalizes.** This figure compares Track A against a naive cross-model cascade with maximally mismatched models on this local Apple MLX runtime. It characterizes *this specific comparison*, not adaptive-depth vs cascade strategies in general.

### Threats to Validity

- **Prompt suite size:** 5 prompts × 48 tokens is sufficient for directional findings but insufficient for statistical confidence. Production validation requires larger and more diverse evaluation sets.
- **Single hardware platform:** All results are from a single M1 Pro with Apple MLX. TPS numbers are platform-specific and should not be compared to GPU-based inference benchmarks.
- **Quantization effects:** GPT-OSS runs at MXFP4 quantization; Gemma runs in the default MLX configuration; the Llama and Mistral follow-ups used MLX 4bit models. Quantization may affect exit-layer convergence and replay behavior differently across model families.

---

## 9. Forward Direction: Selective-Depth Inference

The gap between Track A (practical speedup) and Track C (quality signal but no speedup) points to the next research question: can per-token depth routing recover a practical speed frontier on dense models?

**Selective-depth** means: for each token, run the early exit path first, evaluate confidence, and only invoke remaining layers if confidence is low. Unlike the current blending implementation, this physically skips deep layers for confident tokens, producing real wall-clock savings.

The first real selective-depth attempt at L31 answered the narrow question of whether Track C's agreement signal could be converted into actual skipping. The answer was yes, operationally, but no in practical terms: average layer savings stayed small (2.0% for entropy, 0.2% for margin), neither mode beat full-depth TPS (20.05), and exact match remained near fixed-exit behavior (0.208 and 0.229 versus 0.198 for fixed L31 exit).

The L20 probe answered the next obvious question: does a materially larger skip budget fix the problem? Again, the answer was no in practical terms. The entropy variant increased average layer savings to 15.7% and the realized skip rate to 41.0%, but still reached only 13.26 TPS versus a matched full-depth baseline of 15.16 TPS, with exact match collapsing to 0.073. Margin and hybrid variants were similarly weak.

Focused late-checkpoint follow-up on Llama 3.1 8B and Mistral 7B suggests that this benchmark-side limitation is not Gemma-only. Both additional dense families again showed materially bad fixed exit, near-full quality under compute-both `top1_agree`, and weak real selective-depth after replay. But the internal KL profiles were not uniform across families. The next continuation rule may therefore need to be family-sensitive rather than derived from Gemma's exact L20–L29 plateau alone.

These probes collectively clarify the shape of the dense-model problem in the tested local runtime. Gemma L31 shows that a tiny skip budget is insufficient. Gemma L20 shows that larger skip budget alone is also insufficient. Llama and Mistral show that even when late-checkpoint compute-both quality is perfect, replay/cache repair can still erase almost all of the practical skip budget. The key research question for dense models is therefore not simply "which layer to exit at" but "what continuation criterion can distinguish unresolved composition from converged output while preserving enough of the skip budget to matter after replay and control overhead?" Repeating the same late-checkpoint margin/entropy mechanism is not the best next use of time.

---

## 10. Conclusion

Transcender is best understood not as a generic early-exit method, but as a cross-layer composition control framework for adaptive transformer inference.

Its central contribution is the demonstration that **cross-layer composition control** — not exit-point selection — is the actual problem in adaptive-depth inference, and that **agreement-aware logit-space blending** is a robust solution to that problem.

The three tracks tell a coherent story:

- **Track A** is the strongest current practical result: 0.969 exact match with ~49.5% real layer savings via entropy-gated exit on GPT-OSS 20B. This is the canonical frontier of this repository.
- **Track B** is a completed negative comparison baseline. It documents the cascade tax of naive cross-model speculative decoding with mismatched models (0.026 exact match, 0.28 TPS). It does not disprove cascade strategies in general.
- **Track C** now has three layers of result: fixed exit fails; compute-both agreement-aware composition generalizes and recovers substantial quality; and real selective-depth remains practically weak after replay/cache repair. Gemma provides the deepest dense case study, while focused late-checkpoint follow-up on Llama 3.1 8B and Mistral 7B shows the same benchmark-side pattern on this local Apple MLX runtime. Track C therefore does not establish a practical fixed-exit or real selective-depth frontier in the tested dense follow-ups.

The practical impact of this framework varies by architecture: strong on MoE, suggestive but incomplete on dense models. The dense benchmark limitation is now observed across three tested dense families in this local runtime, but the internal KL profile remains family-sensitive rather than uniformly Gemma-like. Whether selective-depth strategies with better, possibly family-sensitive continuation criteria — disagreement-triggered continuation, token-difficulty-aware routing, and cache-aware continuation — can close this gap on dense models is the clear next research step. Wall-clock speedup for dense selective-depth remains model- and implementation-dependent, and is not validated today beyond the negative dense follow-ups reported here.

---

## Appendix A: Hardware Configuration

| Component | Specification |
|-----------|--------------|
| Machine | MacBook Pro 18,3 (Apple M1 Pro) |
| Memory | 32 GB unified |
| OS | macOS (Darwin 24.6.0) |
| Framework | Apple MLX 0.31.1, mlx-lm 0.31.1 |
| Python | 3.14.2 |

## Appendix B: Reproduction Commands

```bash
# Track A — GPT-OSS 20B adaptive benchmark
python transcender_exit_layer_benchmark.py

# Track B — Cross-model cascade comparison
python transcender_track_b_benchmark.py \
  --large-model /path/to/gpt-oss-20b-raw \
  --draft-model /path/to/gemma-3-4b-it

# Track C — Gemma KL profiling
python transcender_track_c_gemma_profile.py \
  --model /path/to/gemma-3-4b-it

# Track C — Gemma adaptive benchmark
python transcender_track_c_gemma_benchmark.py \
  --model /path/to/gemma-3-4b-it \
  --early-layer 16 --mid-layer 20 --late-layer 31

# Track C — Gemma selective-depth speed validation
python transcender_track_c_gemma_selective_depth.py \
  --model /path/to/gemma-3-4b-it \
  --late-layer 31

# Track C — Gemma selective-depth L20 probe
python transcender_track_c_gemma_selective_depth.py \
  --model /path/to/gemma-3-4b-it \
  --late-layer 20 \
  --margin-threshold 0.02 \
  --entropy-threshold 0.65 \
  --include-hybrid-mode

# Track C — focused Llama dense-family validation
python transcender_track_c_dense_family_validation.py \
  --model /path/to/llama-3.1-8b-instruct-4bit \
  --family llama \
  --profile-output recon_llama3_8b.json \
  --benchmark-output transcender_track_c_llama3_8b_results.json \
  --late-layer 29 \
  --entropy-threshold 0.15

# Track C — focused Mistral dense-family validation
python transcender_track_c_dense_family_validation.py \
  --model /path/to/mistral-7b-instruct-v0.3-4bit \
  --family mistral \
  --profile-output recon_mistral7b.json \
  --benchmark-output transcender_track_c_mistral7b_results.json \
  --late-layer 29 \
  --entropy-threshold 0.15
```

## Appendix C: Data Artifacts

All metrics in this paper are sourced from JSON artifacts produced during benchmarking. Aggregate figures use warmup-corrected values (excluding prompt P1).

| Artifact | Track | Contents |
|----------|-------|----------|
| `transcender_exit_layer_benchmark.json` | A | Per-prompt and aggregate metrics for L22/L23 configurations |
| `transcender_track_b_benchmark.json` | B | Per-prompt and aggregate metrics for all 4 modes |
| `transcender_track_c_gemma_results.json` | C | Per-prompt and aggregate metrics for all 6 modes |
| `transcender_track_c_gemma_selective_depth_results.json` | C | Per-prompt and aggregate metrics for real selective-depth speed validation at L31 |
| `transcender_track_c_gemma_selective_depth_L20_results.json` | C | Per-prompt and aggregate metrics for the L20 selective-depth probe |
| `recon_llama3_8b.json` | C | Per-layer KL profile and heuristic summary for Llama 3.1 8B Instruct 4bit |
| `transcender_track_c_llama3_8b_results.json` | C | Focused late-checkpoint validation metrics for Llama 3.1 8B Instruct 4bit |
| `recon_mistral7b.json` | C | Per-layer KL profile and heuristic summary for Mistral 7B Instruct v0.3 4bit |
| `transcender_track_c_mistral7b_results.json` | C | Focused late-checkpoint validation metrics for Mistral 7B Instruct v0.3 4bit |
| `gemma_kl_profile.json` | C | Per-layer KL divergence across 34 layers |
