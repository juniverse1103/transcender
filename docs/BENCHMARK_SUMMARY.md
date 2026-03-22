# Transcender — Benchmark Summary

**Date:** 2026-03-23 | **Hardware:** Apple M1 Pro, 32 GB unified memory

---

## Cross-Track Snapshot

| Track | Model(s) | Configuration | Exact Match | Avg Layers Saved | Gen TPS | Key Finding |
|-------|----------|---------------|-------------|------------------|---------|-------------|
| **A** | GPT-OSS 20B (MoE, 24L) | L22 `top1_agree` | **0.941** | 0.528 | 21.1 | Strong positive penultimate-layer frontier on one tested sparse MoE family |
| **A** | Qwen3-30B-A3B (MoE, 48L) | L46 `top1_agree` | **0.868** | 0.735 | 34.5 | Same frontier structure reproduces on a second tested sparse MoE family, but at weaker quality |
| **B** | Gemma 3 4B-IT + GPT-OSS 20B | Naive cascade | 0.026 | — | 0.28 | Scoped negative baseline for this model pair and local MLX runtime |
| **C** | Gemma 3 4B-IT, Llama 3.1 8B, Mistral 7B | Dense compute-both + selective-depth follow-up | Mixed | Mixed | Mixed | Compute-both quality recovery is real; no practical dense selective-depth frontier was established on this runtime |

---

## Track A — Same-Model Adaptive Depth

`avg_layers_saved` is the mean number of layers physically skipped per generated token, not a total-compute-savings percentage. At GPT-OSS L22 and Qwen3 L46, each early-exiting token skips exactly one layer.

### GPT-OSS 20B

| Config | Exact Match | Perfect Prompts | Avg Layers Saved | Gen TPS |
|--------|-------------|-----------------|------------------|---------|
| Full depth (L23) | 1.000 | 15/15 | 0.000 | 25.2 |
| L22 `top1_agree` | **0.941** | **13/15** | **0.528** | **21.1** |
| L21 `top1_agree` | 0.745 | 6/15 | 1.156 | 22.2 |
| L20 `top1_agree` | 0.379 | 0/15 | 1.257 | 20.0 |

**Interpretation:** GPT-OSS shows a narrow but practical penultimate-layer frontier. L22 is the only viable selective-exit operating point in the measured frontier; one layer earlier the quality drops sharply.

### Qwen3-30B-A3B

| Config | Exact Match | Perfect Prompts | Avg Layers Saved | Gen TPS |
|--------|-------------|-----------------|------------------|---------|
| Full depth (L47) | 1.000 | 15/15 | 0.000 | 40.9 |
| L46 `top1_agree` | **0.868** | **11/15** | **0.735** | **34.5** |
| L45 `top1_agree` | 0.432 | 0/15 | 1.481 | 35.3 |

**Interpretation:** Qwen3 reproduces the same structural frontier pattern as GPT-OSS: a viable penultimate exit and a sharp cliff one layer earlier. The operating-point quality is weaker than GPT-OSS, so the result should be framed as a qualified positive rather than as uniform MoE generalization.

**Cross-architecture note:** Cross-layer subspace mismatch remains confirmed at 4.11x on GPT-OSS and 2.46x on Gemma. The public-release terminology is **Subspace Mismatch**.

---

## Track B — Cross-Model Cascade (Scoped Negative Baseline)

**Draft:** Gemma 3 4B-IT | **Verifier:** GPT-OSS 20B

| Mode | Gen TPS | Exact Match | Peak Mem (GB) | TTFT (s) |
|------|---------|-------------|---------------|----------|
| Draft only (Gemma 3 4B-IT) | 18.99 | 0.006 | 7.28 | 0.23 |
| GPT-OSS full depth | 30.94 | 1.000 | 12.96 | 0.66 |
| Naive cascade | **0.28** | **0.026** | 20.19 | 11.83 |

**Interpretation:** This is a negative comparison baseline for one naive cascade implementation, one mismatched model pair, and one local runtime. It does not support a general claim about cascade methods as a class.

---

## Track C — Dense Models

### Gemma 3 4B-IT

| Mode | Gen TPS | Exact Match | Layers Saved | Agreement Rate |
|------|---------|-------------|--------------|----------------|
| Full depth (L33) | 15.01 | 1.000 | 0% | — |
| Fixed exit (L16) | 22.81 | 0.010 | 50% | — |
| Fixed exit (L20) | 20.81 | 0.026 | 38% | — |
| Fixed exit (L31) | 14.95 | 0.198 | 6% | — |
| `top1_agree` compute-both (L31) | 7.19 | **0.807** | 0% | 81.9% |
| Naive blend (L31) | 10.01 | 0.688 | 0% | 100% |

**Interpretation:** Dense Gemma validates the composition thesis in compute-both mode, but not a practical selective-depth frontier. Fixed exit is catastrophic; agreement-aware compute-both composition recovers quality.

**Selective-depth validation:** At L31, `selective_depth_entropy_L31` reached 18.20 TPS and 0.208 exact match with 2.0% average layers saved. At L20, `selective_depth_entropy_L20` reached 13.26 TPS and 0.073 exact match with 15.7% average layers saved versus a matched full-depth baseline of 15.16 TPS and 1.000 exact match. Larger skip budget did not recover a practical frontier.

### Dense-Family Follow-Up

| Model | Late Checkpoint | Fixed Exit Exact | Compute-Both Exact | Real Selective Exact | Real Selective TPS | Avg Layers Saved |
|-------|-----------------|------------------|--------------------|----------------------|--------------------|------------------|
| Llama 3.1 8B Instruct 4bit | L29 | 0.151 | 1.000 | 0.469 | 16.60 vs 17.81 full depth | 0.3% |
| Mistral 7B Instruct v0.3 4bit | L29 | 0.109 | 1.000 | 0.318 | 17.25 vs 29.81 full depth | 0.2% |

**Interpretation:** The dense benchmark-side limitation now appears across three tested dense families on this runtime: fixed exit fails materially, compute-both `top1_agree` recovers full-depth output, and real selective-depth remains practically weak after replay/cache repair.

---

## Current Release Takeaway

- A viable penultimate-layer selective-exit frontier appears on both tested sparse MoE families.
- The quality of that frontier is model-dependent: GPT-OSS is strong; Qwen3 is qualified but weaker.
- Track B remains a scoped negative baseline, not a general statement about cascading.
- Dense models support the composition thesis in compute-both mode, but the current runtime did not produce a practical dense selective-depth frontier.
