# Chapter 4: The Logit-Blend Solution

## Bridging the Soft-to-Hard Gate Gap Through Vocabulary-Space Routing

### 4.1 From Subspace Mismatch to Logit-Space Blending

Chapter 3 demonstrated that hidden-state blending fails because Layer 2 and Layer 12 representations occupy geometrically distinct subspaces (4.11x separation ratio). The breakthrough insight is that while *hidden states* are layer-specific, *logits* are universal — both pathways produce distributions over the same 50,257-token vocabulary.

Instead of blending representations and then decoding:

$$\text{logits} = W_{\text{LM}} \cdot [\alpha \cdot h_2 + (1-\alpha) \cdot h_{12}] \quad \text{(fails)}$$

We decode first, then blend in logit space:

$$\text{logits} = \alpha \cdot W_{\text{LM}} \cdot h_2 + (1-\alpha) \cdot W_{\text{LM}} \cdot h_{12} \quad \text{(works)}$$

Mathematically, if $W_{\text{LM}}$ were a linear map and we used the same normalization, these would be identical. But the critical difference is the layer norm $\text{LN}_f$ applied before the LM head:

$$\text{logits}_{\text{early}} = W_{\text{LM}} \cdot \text{LN}_f(h_2)$$
$$\text{logits}_{\text{deep}} = W_{\text{LM}} \cdot \text{LN}_f(h_{12})$$

Layer normalization is a *nonlinear* operation — it normalizes each vector independently to unit variance. This projection brings both Layer 2 and Layer 12 representations onto the same manifold before the LM head decodes them. The resulting logits are in the same space (log-probabilities over vocabulary) regardless of which layer produced them, making blending geometrically valid.

### 4.2 Soft Routing: Differentiable Training

With logit-space blending, SGA uses the Son Router's exit probability $p_{\text{exit}}$ as a differentiable blending weight during training:

$$\text{logits} = p_{\text{exit}} \cdot \text{logits}_{\text{early}} + (1 - p_{\text{exit}}) \cdot \text{logits}_{\text{deep}}$$

This has two crucial properties:

1. **End-to-end differentiability**: The LM loss $\mathcal{L}_{\text{LM}}$ flows gradients through *both* pathways and through $p_{\text{exit}}$ itself. If routing a token early hurts prediction quality, the LM loss pushes $p_{\text{exit}}$ toward 0. If the early logits are already good enough, the efficiency pressure from $\mathcal{L}_{\text{routing}} = (1 - p_{\text{exit}})_{\text{mean}}$ pushes $p_{\text{exit}}$ toward 1.

2. **Decoupled loss functions**: Quality is enforced by $\mathcal{L}_{\text{LM}}$; efficiency is encouraged by $\mathcal{L}_{\text{routing}}$. The combined loss $\mathcal{L} = \mathcal{L}_{\text{LM}} + 0.1 \cdot \mathcal{L}_{\text{routing}}$ lets the router discover the natural boundary between "exit-safe" and "route-deeper" tokens without explicit supervision.

### 4.3 The Inference Reality Gap

Training uses soft blending, but deployment demands hard gates for compute savings — if we compute both $\text{logits}_{\text{early}}$ and $\text{logits}_{\text{deep}}$ for every token, there is no computational benefit. This creates a train-test mismatch:

| Mode | Mechanism | Quality | Savings |
|------|-----------|---------|---------|
| **Training** (soft) | $p \cdot L_{\text{early}} + (1-p) \cdot L_{\text{deep}}$ | Optimized | 0% |
| **Inference** (hard) | $\text{if } p > 0.5: L_{\text{early}} \text{ else } L_{\text{deep}}$ | Degraded | 16.7% |

To quantify this gap, we evaluated three inference strategies on WikiText-2 (50,000 tokens):

| Mode | PPL | $\Delta$ from Vanilla | Compute Savings | Avg Layers/Token |
|------|-----|----------------------|-----------------|------------------|
| **Vanilla GPT-2** | 50.2 | — | 0% | 12.0 |
| **SGA Soft-Gate** | 471.2 | +838.6% | 0% | 12.0 |
| **SGA Adaptive** | 472.2 | +840.6% | 3.3% | ~11.6 |
| **SGA Hard-Gate** | 677.9 | +1250.3% | 16.7% | ~10.0 |

### 4.4 Decomposing the PPL Gap

The results reveal two distinct sources of perplexity degradation:

**Gap 1: Routing overhead (Vanilla → Soft-Gate)**
$$50.2 \to 471.2 \quad (+421.0 \text{ PPL})$$

This gap exists even with soft routing (identical to training mode). It reflects the intrinsic cost of blending Layer 2 logits into the output. Layer 2 predictions are poor — the early layers have only performed 2 of 12 attention/FFN passes and lack the deep contextual processing needed for accurate next-token prediction. Any nonzero $p_{\text{exit}}$ contaminates the output.

This gap is *not* a flaw of the routing mechanism — it is a fundamental property of using Layer 2 as the exit point. Deeper exit layers (e.g., Layer 6) would reduce this gap substantially because mid-network representations carry more predictive signal.

**Gap 2: Train-test mismatch (Soft-Gate → Hard-Gate)**
$$471.2 \to 677.9 \quad (+206.7 \text{ PPL, } +44\%)$$

This gap is caused by the transition from soft to hard gating. During training, a token with $p_{\text{exit}} = 0.6$ gets a 60/40 blend — mostly early logits with some deep correction. At hard inference, that same token gets *pure* early logits. The 40% deep-logit correction is lost entirely, and PPL spikes.

The adaptive strategy (hard-exit only when $p_{\text{exit}} > 0.9$) nearly eliminates Gap 2 while recovering a fraction of the compute savings. At the 0.9 confidence threshold, only the tokens where the router is highly certain exit hard; the rest use soft blending.

### 4.5 The R = I x P Ontology as Theoretical Anchor

The Son metric $R = I \times P$ provides the interpretive framework for understanding routing decisions:

- **I (Information Amount)**: The L2 norm of the hidden state, normalized by $\sqrt{d}$. High $I$ tokens carry rich embedding information — they are syntactically and semantically well-formed at Layer 2.
- **P (Probability)**: The mean attention weight received by the token. High $P$ tokens are contextually important — other tokens attend to them.
- **R = I x P (Son Score)**: Tokens with high $R$ are both informationally rich *and* contextually central. The router learns that these tokens are safe to exit early — their Layer 2 representation is already "mature."

In our trained model, we observe the expected pattern:
- **Function words** ("the", "of", "and") receive high exit probabilities — their Layer 2 logits are already accurate because function words are highly predictable from local context alone.
- **Content words** ("quantum", "implications", "epistemological") receive low exit probabilities — their predictions require deep contextual integration across the full 12-layer stack.
- **Subword fragments** ("Ġ", "Ċ", partial BPE tokens) cluster at high exit probability — they are syntactically determined by their prefix and require minimal additional computation.

### 4.6 Path Forward: Closing Gap 1

The dominant source of PPL degradation is Gap 1 (routing overhead), not Gap 2 (gate mismatch). Three strategies for closing it:

1. **Deeper exit point**: Moving the router from Layer 2 to Layer 6 would give early-exit tokens 50% of the full compute stack. Layer 6 representations carry substantially more predictive signal, reducing the quality penalty of early exit. The trade-off is smaller maximum compute savings (50% vs 83%).

2. **Temperature annealing**: During training, gradually sharpen exit probabilities from soft (temperature=1.0) toward hard (temperature→0). This smooths the transition from soft blending to hard gating, reducing Gap 2.

3. **Calibrated routing loss**: Replace the uniform efficiency pressure $\mathcal{L}_{\text{routing}} = (1 - p_{\text{exit}})_{\text{mean}}$ with a token-conditional loss that only encourages exit when the early logits are empirically close to the deep logits (small KL divergence).

### 4.7 Implications for Early-Exit Architectures

The logit-blend framework reveals a general principle for early-exit architectures:

> **Cross-subspace communication must occur in output space, not representation space.**

Any architecture that routes tokens through different numbers of layers faces the subspace paradox. The SGA solution — decode each pathway independently, then blend in the shared vocabulary space — is not specific to our architecture. It applies to any early-exit transformer, any mixture-of-depths model, and any system that combines representations from different layers.

The key insight is that layer normalization + LM head projection acts as a *subspace normalizer*, mapping layer-specific representations onto a common manifold. By blending *after* this projection, we respect the geometric structure of the representation space rather than forcing a fictitious interpolation across disjoint subspaces.

---

*The logit-blend solution transforms SGA from a brittle binary gate into a differentiable routing system. The remaining PPL gap (50.2 → 471.2) is dominated by the intrinsic quality of Layer 2 predictions — a challenge addressable through deeper exit points, not routing mechanism changes. The soft-to-hard gate gap (471.2 → 677.9) is a solvable engineering problem via adaptive thresholding and temperature annealing.*
