# Chapter 3: The Subspace Paradox

## Why Early-Exit Hidden States Cannot Be Blended with Deep Hidden States

### 3.1 The Intuitive Approach — and Its Failure

The most natural implementation of a soft-gated early-exit architecture is to blend hidden states:

$$h_{\text{blend}} = p_{\text{exit}} \cdot h_{\text{early}} + (1 - p_{\text{exit}}) \cdot h_{\text{deep}}$$

where $h_{\text{early}}$ is the hidden state after layer 2 and $h_{\text{deep}}$ is the hidden state after layer 12. The exit probability $p_{\text{exit}} \in [0, 1]$ is produced by the Son Router. This approach is differentiable, simple, and — catastrophically wrong.

When we implemented hidden-state blending in SGA, the router immediately collapsed. Regardless of initialization, training drove $p_{\text{exit}}$ to one of two extremes: either all tokens exit early (quality collapse) or no tokens exit (efficiency collapse). The loss landscape was U-shaped with no stable interior minimum. Gradient descent rolled to the nearest edge every time.

This was not a hyperparameter problem. It was a *geometric* problem.

### 3.2 The Subspace Separation Proof

To understand the failure, we collected 5,120 token representations from WikiText-2 at two points in GPT-2's forward pass:

- **Layer 2 states** ($h_2$): representations after the second transformer block
- **Layer 12 states** ($h_{12}$): representations after the final transformer block

Both live in $\mathbb{R}^{768}$, but they do not occupy the same region of that space. We applied PCA to the combined set of representations, projecting onto a shared 2D basis to make the separation visually and metrically apparent.

**Empirical findings:**

| Metric | Value |
|--------|-------|
| Centroid distance (L2 norm between cluster centers) | 432.45 |
| Layer 2 spread (mean std across PCs) | 56.28 |
| Layer 12 spread (mean std across PCs) | 49.54 |
| **Separation ratio** (centroid distance / combined spread) | **4.11x** |

A separation ratio of 4.11x means the two clusters are more than four times farther apart than they are wide. In statistical terms, these are not overlapping distributions — they are distinct populations occupying different subspaces of $\mathbb{R}^{768}$.

### 3.3 Why Separation Kills Blending

Consider what happens when we compute $h_{\text{blend}} = 0.5 \cdot h_2 + 0.5 \cdot h_{12}$. The resulting vector sits at the *midpoint* between the two subspaces — a region of the 768-dimensional space that neither layer 2 nor layer 12 ever produces naturally. The language model head ($W_{\text{LM}} \in \mathbb{R}^{50257 \times 768}$) was trained to decode representations from the Layer 12 subspace. Feed it a vector from no-man's land, and it produces nonsensical logits.

This is not an edge case. For *any* blending weight $\alpha \in (0, 1)$:

$$h_{\text{blend}}(\alpha) = \alpha \cdot h_2 + (1 - \alpha) \cdot h_{12}$$

the resulting vector follows a linear path through the 768-dimensional space. Because the two clusters are separated by 4.11x their combined spread, the interior of this path passes through a region that is:

1. **Out-of-distribution** for the LM head — it was never trained on these intermediate representations
2. **Semantically incoherent** — it encodes neither early-layer syntactic patterns nor late-layer semantic patterns
3. **High-loss** — producing a U-shaped loss curve where both endpoints ($\alpha = 0$ and $\alpha = 1$) have lower loss than any interior point

Our Experiment 4 confirmed this empirically. Sweeping $\alpha$ from 0 to 1 for hidden-state blending produced a convex loss curve with the minimum at $\alpha = 0$ (pure deep states). There is no "sweet spot" in hidden-state space — the best you can do is not blend at all.

### 3.4 The Deeper Lesson: Representations Are Layer-Specific

This result has implications beyond SGA. The transformer does not simply refine a single representation across layers — it *transforms* the representation into progressively different subspaces. Each layer's output is the "native format" for the next layer's input. The LM head expects the final layer's format.

This is why approaches like hidden-state interpolation (common in some early-exit literature) face fundamental limits. The assumption that intermediate and final representations are "close enough" to blend is falsified by the 4.11x separation we measured. Any early-exit architecture that blends hidden states must contend with this subspace mismatch.

The PCA variance ratios tell us the geometry is dominated by a few principal directions: PC1 captures 68.0% and PC2 captures 14.8% of total variance. The separation is not diffuse — it is concentrated along the primary axis of variation, meaning Layer 2 and Layer 12 representations differ in their most important features, not just noise dimensions.

### 3.5 The Son Metric as a Subspace Classifier

The Son metric $R = I \times P$ provides an interpretable signal *within* each subspace. At Layer 2, $I$ (embedding norm) reflects how much syntactic information the token has accumulated, while $P$ (attention probability) reflects how much contextual attention the token receives. Their product identifies tokens whose representations are already "mature enough" at Layer 2 — tokens that don't need the subspace transformation of deeper layers because their prediction is already confident in the early subspace.

This reframes early exit not as "skipping computation" but as *recognizing that some tokens' representations are already in a decodable state after Layer 2*. The LM head can decode Layer 2 representations — it just does so less accurately than Layer 12 representations. The Son Router identifies the tokens where this accuracy gap is smallest.

---

*The subspace paradox reveals that early-exit routing cannot operate in hidden-state space. Chapter 4 presents the solution: logit-space blending, where both pathways' outputs are projected into a shared vocabulary space before combination.*
