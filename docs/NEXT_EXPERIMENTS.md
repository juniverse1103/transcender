# Next Experiments — Post-Track C Selective-Depth Follow-Up

**Date:** 2026-03-22 | **Status:** Proposed

---

## Why Fixed Exit Failed on Dense Models

Track C demonstrated that fixed early exit is catastrophic on Gemma 3 4B-IT:

- Exiting at L31: **19.8% exact match** with only 6% layers saved
- Compare GPT-OSS Track A: **0.969 exact match** with real adaptive layer skipping

The root cause is the **KL plateau** observed in Gemma's depth profile: layers 20-29 perform compositional work that temporarily *increases* logit divergence from the final output. Any fixed cutoff that enters this plateau skips the resolution phase (L30-L33) that resolves compositional ambiguity into coherent output.

Dense models may distribute knowledge more uniformly across depth than MoE models. In MoE architectures, expert routing creates natural "knowledge completion" points — when the relevant experts have fired, the token's representation is mature. Dense models may lack this structure. This is a hypothesis, not a validated mechanism.

## Why Blending Still Showed Signal

Despite fixed-exit failure, agreement-aware compute-both blending at L31 recovered to **0.807 exact match** (versus 0.198 for fixed exit at the same layer). This tells us:

1. The early exit layer (L31) produces the *correct top-1 token* 82% of the time
2. When it agrees with full depth, the agreement is reliable
3. The 18% disagreement cases are precisely the tokens where depth matters most

This means the exit layer contains useful signal about token difficulty — it just cannot be trusted unconditionally.

## What the Real Selective-Depth Tests Showed

Real selective-depth runtimes were implemented and measured at both L31 and L20 on Gemma 3 4B-IT. These were not compute-both blending modes. They physically skipped deep layers on some decode tokens and replayed previously skipped decode tokens through the deep layers only when later continuation was required for cache repair.

- `selective_depth_margin_L31`: 18.26 TPS, 0.229 exact match, 78.2% early acceptance, 3.7% realized skip rate, 0.2% average layers saved
- `selective_depth_entropy_L31`: 18.20 TPS, 0.208 exact match, 93.6% early acceptance, 34.0% realized skip rate, 2.0% average layers saved
- `selective_depth_margin_L20`: 11.29 TPS, 0.083 exact match, 70.2% early acceptance, 13.8% realized skip rate, 5.3% average layers saved
- `selective_depth_entropy_L20`: 13.26 TPS, 0.073 exact match, 73.4% early acceptance, 41.0% realized skip rate, 15.7% average layers saved
- `selective_depth_hybrid_L20`: 11.37 TPS, 0.125 exact match, 61.2% early acceptance, 6.4% realized skip rate, 2.4% average layers saved

These are negative speed-validation results for Gemma 3 4B-IT on the local Apple MLX runtime. L31 showed that a two-layer skip budget is too small. L20 showed that simply moving the checkpoint earlier and increasing the skip budget is still not enough: wall-clock TPS remained below full depth and exact match collapsed toward fixed-exit behavior.

## Why This Is No Longer Gemma-Only

Focused late-checkpoint follow-up on two additional dense families reproduced the same benchmark-side pattern on the local Apple MLX runtime:

- **Llama 3.1 8B Instruct 4bit (L29):** fixed exit reached 0.151 exact match; compute-both `top1_agree` reached 1.000 exact; real selective-depth entropy reached 16.60 TPS and 0.469 exact with 0.3% average layers saved versus a matched full-depth baseline of 17.81 TPS
- **Mistral 7B Instruct v0.3 4bit (L29):** fixed exit reached 0.109 exact match; compute-both `top1_agree` reached 1.000 exact; real selective-depth entropy reached 17.25 TPS and 0.318 exact with 0.2% average layers saved versus a matched full-depth baseline of 29.81 TPS

This means the dense benchmark limitation is no longer just a Gemma observation. Late fixed exit failure, strong compute-both quality recovery, and weak real selective-depth now appear across three tested dense families on this runtime.

The internal profile is less uniform. Gemma shows a late L20-L29 plateau, Llama shows a milder middle plateau near L12-L14, and Mistral shows an earlier plateau near L2-L8. Future continuation criteria should therefore be treated as potentially family-sensitive rather than copied directly from Gemma's exact depth profile.

## Proposed Direction: Selective-Depth Beyond Current L20/L31 Probes

### What Is NOT the Next Step

The following approaches have been exhausted or are not productive for dense-model adaptive depth:

- **More brute-force fixed-exit sweeps.** Track C tested exit at L16, L20, and L31. All are catastrophic (0.010–0.198 exact match). Sweeping additional fixed exit points on this model will not produce a different outcome.
- **Threshold fiddling without a new mechanism.** Adjusting KL thresholds or confidence parameters on the same fixed-exit architecture does not address the structural problem (the KL plateau).
- **Re-running the same Gemma fixed-exit benchmark with different parameters.** The benchmark infrastructure is validated. The bottleneck is the exit strategy, not the measurement.
- **Repeating L31 selective-depth with the same mechanism.** This has now been measured. The runtime was correct, but it did not beat full depth and stayed near fixed-exit quality.
- **Repeating L20 selective-depth with the same margin/entropy mechanism.** This has also now been measured. The larger skip budget did not recover a practical frontier because quality collapsed and replay/control overhead still consumed the gain.

### What IS the Next Step

The next direction is still **selective-depth**, but not as a repeat of the current L31 or L20 mechanisms. The completed probes narrow the question: future dense-model work should focus on better continuation rules and lower continuation overhead, not on moving the checkpoint earlier by itself.

Concretely, the next dense-model work should focus on:
- deeper early checkpoints only if they are paired with a stronger, family-sensitive continuation criterion
- token-difficulty-aware routing
- continuation criteria that distinguish plateau vs convergence
- cache-aware continuation improvements that reduce replay overhead

### Experiment A: Deeper Selective-Depth with Plateau-Aware Continuation

**Hypothesis:** The completed Gemma, Llama, and Mistral follow-ups show that larger skip budget or late-checkpoint confidence alone is not enough. A future checkpoint probe is only worth testing if the continuation rule can distinguish "still in the model's unresolved compositional regime" from "truly converged."

**Design:**
1. Choose a checkpoint only if it is paired with a continuation signal stronger than plain margin or entropy gating
2. Compute a continuation criterion designed to detect whether the model is still in the KL plateau
3. Continue deeper only when the criterion signals unresolved composition
4. Measure whether the larger skip budget survives the added routing overhead

**Why this is different from the completed L20/L31 tests:** The existing probes already showed that real skipping alone is not enough, whether the skip budget is small (L31) or materially larger (L20), when the continuation mechanism is weak.

### Experiment B: Cache-Aware Continuation

**Hypothesis:** The completed L31 and L20 selective-depth runs exposed replay and control overhead as first-order bottlenecks. A cache-aware continuation path that reduces or eliminates replay cost may recover part of the lost wall-clock budget.

**Design:** This is analogous to speculative decoding, but within a single model across depth rather than across two models. The "draft" is the early-exit path; the "verifier" is the full-depth path of the same model.

**Advantage over cross-model speculation (Track B):** Same tokenizer, same vocabulary, same prompt template. The compatibility problems that destroyed Track B's cascade are structurally impossible.

### Experiment C: Token-Difficulty-Aware Depth Allocation

**Hypothesis:** Token classes may differ in whether they need continuation through the KL plateau. Routing by token difficulty may be more effective than a single global confidence threshold.

**Design:**
1. Profile per-token-type KL convergence using the existing profiler infrastructure
2. Build a simple token-type classifier (function word, content word, punctuation, subword fragment)
3. Assign depth budgets per type: function words exit at L20, content words run full depth
4. Measure quality vs compute savings tradeoff

**Note:** The Track A KL profiler already showed that token types converge at similar rates on GPT-OSS. This experiment tests whether Gemma's plateau creates more differentiated convergence patterns across token types.

### Experiment D: Larger Dense Model Validation

**Hypothesis:** The KL plateau may narrow or shift in larger dense models, potentially enabling a practical fixed-exit frontier at scale.

**Design:** Run the Track C profiler and benchmark on a larger dense model (for example Gemma 3 12B or 27B) only after the next continuation criterion is stronger than the current margin/entropy probes. If the late-resolution dependence weakens with scale, it would suggest dense-model adaptive depth becomes more viable as models grow. If it persists across larger models and additional families, it would strengthen the case that dense continuation rules need to be family-sensitive but broadly stronger than fixed confidence gating.

## Priority Order

1. **Experiment A** (selective-depth with plateau-aware continuation) — most direct follow-up to the negative Gemma, Llama, and Mistral dense results
2. **Experiment C** (token-difficulty depth allocation) — tests whether routing signal is more class-dependent than threshold-dependent
3. **Experiment B** (cache-aware continuation) — addresses the replay/control overhead exposed by the L20/L31 selective-depth runs
4. **Experiment D** (larger model) — resource-dependent, validates whether findings are scale-dependent

## Success Criteria

Selective-depth on Gemma 3 4B succeeds if:
- Exact match rate remains above 0.95 against full-depth baseline
- At least 20% of tokens exit early (physically skip remaining layers)
- Net TPS improvement is measurable on wall-clock (even if modest)

**Important caveat:** The 81.9% agreement rate observed in Gemma Track C was a hypothesis-generating signal, not a validated speedup prediction. The completed Gemma, Llama, and Mistral dense benchmarks showed that confidence-based skipping alone does not guarantee a wall-clock win, even when the skip budget increases materially or the late checkpoint agreement rate is high. Future variants still need to measure whether their continuation criterion, cache management, and control flow preserve enough of the theoretical compute savings to matter on a given model family.

The completed L31 and L20 probes already showed that real skipping alone is not enough. Future dense-model claims must beat both replay/control overhead and quality collapse in measured runtime, not just in layer-count arithmetic.

If a deeper selective-depth variant does not reach 0.95 exact match on Gemma 3 4B-IT, the finding is still valuable: it would suggest that dense 4B models are too small or too shallow for practical adaptive depth, and that the method may require either MoE structure or larger dense scale.
