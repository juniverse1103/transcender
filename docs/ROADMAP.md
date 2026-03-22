# Transcender / SGA-POC — Execution Roadmap

**Date:** 2026-03-22 | **Stage:** Research PoC → Usable Runtime

---

## 0. COMPLETED — Package Restructure (2026-03-22)

All items below are done and verified:

- **Package structure**: `transcender/` package with `router.py`, `model.py`, `engine/` sub-package
- **`pyproject.toml`**: Project is pip-installable via `pip install -e .`
- **SGAModel deprecated**: `model_injector.py` is now a thin shim with `DeprecationWarning`
- **Shims**: `sga_router.py` and `transcender_injector.py` are re-export shims
- **Imports cleaned**: All experiment scripts use `from transcender import TranscenderModel`
- **sys.path hacks removed**: Replaced with `try/except` fallback pattern
- **FastAPI modernized**: `@app.on_event("startup")` → `asynccontextmanager` lifespan
- **Smoke tests added**: 20 pytest tests covering imports, router, model forward pass, loss
- **Fix 1–5**: All applied and verified (see Section 1 below for original descriptions)
- **Engine split**: Config, loading, and prompt helpers extracted to `transcender/engine/`
- **Backward compat verified**: `SGAModel`, `sga_router`, `transcender_injector` imports work

---

## 1. CRITICAL FIXES — ALL APPLIED

### Fix 1 — `output_attentions` silent tuple bug

**File:** `model_injector.py` lines 129–135  
**Also in:** `transcender_injector.py` lines 236–241

**Exact problem:**
```python
hidden_states = self.blocks[i](
    hidden_states,
    attention_mask=attention_mask,
    output_attentions=(i == router_layer),   # ← the problem
)
```
When `output_attentions=True`, `GPT2Block.forward` returns `(hidden_states_tensor, attn_weights_tensor)`. The assignment silently sets `hidden_states` to a tuple. The next block call receives that tuple as input, which crashes inside attention's `torch.matmul`. In practice it doesn't crash because the forward hook fires first and captures attention weights — but if `eager` attention is swapped for SDPA or flash attention, or if HuggingFace changes `GPT2Block`'s return convention again, this silently corrupts hidden states without an informative error.

**Why it matters technically:** Latent correctness hazard that depends on hook timing. The workaround (the hook) already captures what we need — the `output_attentions` argument is dead code.

**Minimal safe fix:** Remove `output_attentions` from the block call entirely.

```python
# BEFORE (model_injector.py:129)
for i in range(self.exit_after_layer):
    hidden_states = self.blocks[i](
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=(i == router_layer),
    )

# AFTER
for i in range(self.exit_after_layer):
    hidden_states = self.blocks[i](
        hidden_states,
        attention_mask=attention_mask,
    )
```

Apply the identical change to `transcender_injector.py` line 240.

**Regression risk:** Near-zero.

**Verify:**
```bash
python -c "
from model_injector import SGAModel; import torch
m = SGAModel('gpt2', exit_after_layer=2); m.eval()
ids = torch.randint(0, 50257, (1, 16))
out = m(input_ids=ids)
assert out['routing_info']['exit_probs'].shape == (1, 16)
assert isinstance(out['logits'], torch.Tensor)
print('Fix 1: PASS')
"
```

---

### Fix 2 — Broken exit rate in `benchmark.py:evaluate_sga_ppl`

**File:** `benchmark.py` lines 111–116

**Exact problem:**
```python
exit_rate = sum(
    1 for chunk in chunks[:10]
    for _ in range(1)  # just sample
)   # → always returns 10
```
Always returns 10. Never reflects actual router behavior. The variable is also dropped from the return tuple, so callers never see it.

**Minimal safe fix:** Add proper accumulation tracking inside the loop:

```python
# Add before the loop:
total_exited = 0
total_tok_count = 0

# Add inside the for-chunk loop, after `output = model(...)`:
total_exited += output["routing_info"]["exit_mask"].sum().item()
total_tok_count += len(chunk)

# Replace the broken lines 111-114:
exit_rate = (total_exited / total_tok_count) * 100 if total_tok_count > 0 else 0.0

# Update return to include exit_rate:
return ppl, avg_loss, avg_layers, savings, exit_rate
```

Update `run_perplexity_benchmark` to unpack and print the exit rate.

**Regression risk:** Low. Changes return tuple arity — one call site to update.

---

### Fix 3 — Unfilled template variables in `chapter3_subspace_paradox.md`

**File:** `chapter3_subspace_paradox.md` line 59

**Exact problem:**
```
PC1 captures {explained_var_1} and PC2 captures {explained_var_2} of total variance.
```
Python f-string tokens that were never substituted. Looks broken to any reader.

**Fix:** Run `python benchmark.py` (experiment 3, the subspace analysis), capture the printed PCA output (`PC1=X.X%, PC2=X.X%`), and hard-code those values directly into line 59.

**Verify:** `grep -n "{" chapter3_subspace_paradox.md` returns no results.

---

### Fix 4 — `avg_layers` denominator off-by-one in `benchmark.py`

**File:** `benchmark.py` line 109

**Exact problem:**
```python
avg_layers = total_layer_passes / (total_tokens + 1)
```
`total_tokens` counts non-first tokens (`len(chunk) - 1` per chunk), but `total_layer_passes` is summed over `len(chunk)` — all tokens. The denominators don't match.

**Fix:**
```python
# Track total_tok_count separately (add inside loop):
total_tok_count += len(chunk)

# Then:
avg_layers = total_layer_passes / total_tok_count
```

**Verify:** At `threshold=1.0` (no exits), `avg_layers` must equal exactly `12.0`.

---

### Fix 5 — Misleading comment in deep-layer loop

**File:** `model_injector.py` line 165 / `transcender_injector.py` line 266

**Exact problem:** Comment says `"only for non-exiting tokens"` but the loop runs ALL tokens through deep layers. Exit mask only controls which logits are used in the output, not which tokens are computed.

**Fix:** Update the comment (no code changes):
```python
# --- Deep layers --- (ALL tokens compute deep states for logit-space blending.
# exit_mask controls which logits are used in the output, not which tokens are computed.
# Actual FLOP savings only materialize in hard inference mode via separate batching.)
```

---

## 2. ARCHITECTURE CONSOLIDATION PLAN

### Decision: `TranscenderModel` is canonical, `SGAModel` is deprecated.

`SGAModel` and `TranscenderModel` implement the same forward pass. `TranscenderModel` is strictly more capable: multi-arch support, `freeze_backbone()`, `get_routing_summary()`, better arg validation. Every bug exists in both. This stops now.

### Post-Refactor Class Structure

```
transcender/
    router.py     → SonRouter, SonRoutingLoss (from sga_router.py, unchanged)
    adapters.py   → ArchitectureAdapter (extracted from transcender_injector.py)
    model.py      → TranscenderModel ONLY (canonical)
    engine.py     → TranscenderEngine PyTorch class
    mlx_engine.py → MLXDynamicExpertEngine
```

`SGAModel` becomes a one-liner deprecation shim in `model.py`:
```python
def SGAModel(model_name="gpt2", exit_after_layer=2, **kwargs):
    """Deprecated. Use TranscenderModel directly."""
    import warnings
    warnings.warn("SGAModel is deprecated. Use TranscenderModel.", DeprecationWarning)
    return TranscenderModel(model_name=model_name, exit_after_layer=exit_after_layer, **kwargs)
```

### Methods to Extract from the Duplicated Forward Pass

Both classes contain identical logic for these — extract to private methods on `TranscenderModel`:

| Extracted Method | Replaces |
|------------------|----------|
| `_blend_logits(early_logits, deep_logits, exit_probs, exit_mask)` | The `if self.training / elif soft / elif adaptive / else hard` block |
| `_compute_loss(logits, early_logits, deep_logits, exit_probs, labels)` | The loss computation block |

### Migration Steps (Safest Order)

1. Fix all bugs first (Section 1) — never migrate with known bugs.
2. Extract `_blend_logits()` into `TranscenderModel`. Verify output unchanged.
3. Extract `_compute_loss()` into `TranscenderModel`. Verify loss values unchanged.
4. Add `SGAModel` deprecation shim. Import it from `transcender_injector.py` in `model_injector.py`.
5. Update all experiment scripts to import from `transcender`.
6. Run all benchmarks — numbers must match prior run.
7. Delete `model_injector.py` body, keep only the deprecation import.

**Rule: Each step is a separate commit. Steps 1–3 are isolated refactors. Steps 4–7 are the migration.**

---

## 3. ROUTING STRATEGY UNIFICATION

### Two Different Algorithms

| Property | Son Router (GPT-2 PoC) | Entropy Gate (20B MLX) |
|----------|------------------------|------------------------|
| Signal | Learned MLP on hidden state | Normalized logit entropy |
| Supervision | KL-calibrated BCE | None — zero-shot heuristic |
| Training required | Yes (49K params, ~10 epochs) | No |
| Decision granularity | Per-token | Per-generation-step |
| Compute overhead | 49K-param forward pass | One softmax + entropy scalar |
| Generalization | Retrain per model | Works immediately |

The entropy gate asks "has the model converged on an answer?" The Son Router asks "is this specific token's early representation mature?" They solve different sub-problems and can be composed.

### What the Project Should Officially Claim

> **Transcender is a framework for depth-axis sparsity, supporting two routing primitives:**
>
> 1. **Learned Son Router** — trained per model, per exit layer. Best per-token granularity. Requires a short training phase (0.04% of backbone params).
> 2. **Entropy Gate** — zero-shot, sequence-level. No training required. Deploy immediately.
>
> These are orthogonal and composable: Entropy Gate handles sequence-level convergence detection; Son Router handles per-token compute allocation within a sequence.

### Definitive Comparison Experiment

**Iso-savings duel on LLaMA-3 8B.** Fix compute savings at 20%. Compare:
- Son Router at Layer 26 of 32 (trained, KL-calibrated loss, WikiText-103)  
- Entropy Gate threshold swept to produce ~20% step-skips

Metric: PPL on WikiText-103 test. If Son Router wins, training is justified. If entropy gate matches, emphasize zero-shot path for deployment story.

---

## 4. PACKAGE / REPO RESTRUCTURE

### Target Directory Structure

```
sga-poc/
├── pyproject.toml
├── README.md
├── ROADMAP.md                        ← this file
├── Transcender_Whitepaper_v1.md
├── docs/
│   ├── chapter3_subspace_paradox.md
│   └── chapter4_logit_blend_solution.md
├── transcender/
│   ├── __init__.py                   # exports TranscenderModel, SonRouter, SonRoutingLoss
│   ├── router.py                     # ← sga_router.py (renamed)
│   ├── model.py                      # ← transcender_injector.py (canonical)
│   ├── adapters.py                   # ← ArchitectureAdapter extracted
│   ├── engine.py                     # ← TranscenderEngine (PyTorch)
│   └── mlx_engine.py                 # ← MLXDynamicExpertEngine
├── experiments/
│   ├── __init__.py
│   ├── train.py                      # ← train_and_visualize.py
│   ├── benchmark_ppl.py              # ← benchmark.py
│   ├── benchmark_inference.py        # unchanged
│   └── benchmark_layers.py          # ← benchmark_layer_comparison.py
├── checkpoints/                      # ← *.pt files
├── figures/                          # ← *.png files
└── scripts/
    └── recon.py                      # ← transcender_recon.py
```

### Minimum Viable `pyproject.toml`

```toml
[project]
name = "transcender"
version = "0.1.0"
description = "Per-token depth-axis routing for transformer language models"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "datasets>=2.14.0",
    "tqdm>=4.65.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

[project.optional-dependencies]
mlx = ["mlx>=0.16.0", "mlx-lm>=0.16.0"]
dev = ["pytest", "pyright"]

[project.scripts]
transcender-train = "experiments.train:main"
transcender-bench = "experiments.benchmark_ppl:main"
transcender-bench-layers = "experiments.benchmark_layers:main"

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends.legacy:build"

[tool.setuptools.packages.find]
where = ["."]
include = ["transcender*", "experiments*"]
```

### Import Change

```python
# BEFORE (in any experiment file):
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_injector import SGAModel

# AFTER:
from transcender import TranscenderModel
```

---

## 5. EXPERIMENT ROADMAP

### Experiment 1 — Learned Router vs Entropy Gate (Iso-Savings Duel)

**Hypothesis:** At identical compute savings, the trained Son Router achieves lower PPL than the zero-shot entropy gate, because it uses the full hidden state rather than just output entropy.

**Code area:** Add `EntropyGate` class to `transcender/router.py`. Benchmark using `experiments/benchmark_layers.py` infrastructure.

**Metrics:** PPL on WikiText-2 test; savings %; exit rate %

**Success criterion:** Son Router PPL ≤ Entropy Gate PPL at iso-savings (±1 PPL).

**Failure interpretation:** If entropy gate ties or wins, the KL-calibrated loss is not learning useful signal from GPT-2's shallow stack. Re-run at Layer 6 before drawing conclusions.

---

### Experiment 2 — Layer-Ablation for Hard Exit (Full Sweep L1–L11)

**Hypothesis:** There is a "sweet spot" exit layer for GPT-2, not Layer 2 or Layer 6 — likely Layer 8–9.

**Code area:** Generalize the `configs` list in `experiments/benchmark_layers.py` to sweep all 11 candidate layers. Train each router for 5 epochs.

**Metrics:** PPL vs savings scatter for all exit points. Pareto-optimal layer.

**Success criterion:** Identify the layer where Hard-Gate PPL first crosses below 2× vanilla PPL.

**Failure interpretation:** If no layer achieves <2× vanilla PPL in hard mode, confirm the GPT-2 scale limitation and pivot the architecture narrative entirely to LLaMA-scale experiments.

---

### Experiment 3 — Manual Forward vs Official Path Equivalence Check

**Hypothesis:** `TranscenderModel` in soft mode with `exit_probs → 0` produces logits numerically identical to `GPT2LMHeadModel.forward()`.

**Code area:** New script `experiments/validate_equivalence.py`.

**Metric:** Max absolute logit difference between the two models on the same input.

**Success criterion:** Max absolute difference < 1e-4.

**Failure interpretation:** Dropout not disabled, position_ids mismatch, or attention mask format discrepancy. Common culprits to check first.

```python
# experiments/validate_equivalence.py skeleton:
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transcender import TranscenderModel

tok = GPT2Tokenizer.from_pretrained("gpt2")
ids = tok("The quick brown fox", return_tensors="pt")["input_ids"]

ref = GPT2LMHeadModel.from_pretrained("gpt2").eval()
with torch.no_grad():
    ref_logits = ref(ids).logits

# Force exit_probs = 0 by zeroing router weights
model = TranscenderModel("gpt2", exit_after_layer=2, inference_mode="soft").eval()
with torch.no_grad():
    model.router.gate_proj[0].weight.data.zero_()
    model.router.gate_proj[2].weight.data.zero_()  # → sigmoid(0) = 0.5 blend
    # Note: for true equivalence you want exit_probs=0, so override forward
    out = model(ids)

diff = (ref_logits - out["logits"]).abs().max()
print(f"Max diff: {diff:.2e}  {'PASS' if diff < 1e-4 else 'FAIL'}")
```

---

### Experiment 4 — Temperature Annealing for Soft-to-Hard Gap

**Hypothesis:** Training with exit probabilities sharpened by temperature annealing (τ: 1.0→0.1 over epochs) reduces Gap 2 (Soft→Hard PPL delta) by teaching more binary routing decisions.

**Code area:** Modify `experiments/train.py` to accept `--temperature-schedule linear|cosine|none`. Apply: `exit_probs = torch.sigmoid(exit_logits / tau)` with tau decreasing per epoch.

**Metrics:** Soft-Gate PPL; Hard-Gate PPL; gap between them. Track per-epoch.

**Success criterion:** Gap 2 (currently ~206 PPL at L2) reduces by ≥30%.

**Failure interpretation:** If annealing collapses diversity (all exit_probs → 0 or 1 early), use a two-phase schedule: soft-only for epochs 1–7, annealing for epochs 8–10.

---

### Experiment 5 — Scaling Beyond GPT-2: LLaMA-3 8B at Layer 16

**Hypothesis:** At 32 layers, Layer 16 exit achieves a much smaller PPL penalty relative to baseline than GPT-2's Layer 6 exit, because mid-stack LLaMA representations are richer than 6-of-12-layer GPT-2 representations.

**Code area:** `TranscenderModel("meta-llama/Llama-3.1-8B", exit_after_layer=16)` — the architecture adapter already supports LLaMA. Train the Son Router on WikiText-103 (richer dataset, appropriate for 8B scale).

**Metrics:** PPL on WikiText-103 test; savings %; token routing distribution.

**Success criterion:** Hard-Gate PPL < 1.5× vanilla LLaMA-3-8B baseline at >30% savings.

**Failure interpretation:** If PPL > 3× vanilla, re-run at Layer 22–24. The KL calibration may need tuning for GQA+SwiGLU attention pattern.

---

### Experiment 6 — Token-Semantic Routing Interpretation Study

**Hypothesis:** The trained Son Router's exit probabilities correlate with token POS class: function words have higher mean exit_prob than content words, which have higher exit_prob than named entities.

**Code area:** New script `experiments/routing_semantics.py`. Use `nltk.pos_tag` to classify tokens. Compute mean ± std exit_prob per POS bin across 1000 WikiText-2 sentences.

**Metrics:** Mean exit_prob per POS class; ANOVA p-value across groups; function-word vs content-word t-test.

**Success criterion:** p < 0.01 for function vs content word separation. Mean exit_prob difference ≥ 0.1. This converts the qualitative claim in chapter4 into a quantitative result for the whitepaper.

**Failure interpretation:** Router is routing on surface-form features (token position, byte length) rather than semantic class. Visualize R=I×P distribution per class as a diagnostic.

---

## 5b. TRACK B — DUAL-MODEL CASCADE COMPARISON (2026-03-22)

### What Track B Is

A minimal dual-model cascade benchmark comparing a small draft model (Gemma 3 4B IT or any local dense model) against GPT-OSS 20B as verifier. The question Track B answers:

> **Is Track A (same-model adaptive depth at Layer 22) a better quality/speed/memory frontier than a simple dual-model cascade?**

### Files

| File | Purpose |
|------|---------|
| `transcender_track_b_cascade.py` | Cascade engine: auto-detection of local draft models, per-model prompt formatting (Harmony for GPT-OSS, native chat template for draft), naive chunked draft-then-verify loop with first-divergence correction |
| `transcender_track_b_benchmark.py` | Benchmark runner comparing 4 modes: draft-only, GPT-OSS full-depth baseline, Track B cascade, Track A L22 top1_agree |

### Four Benchmark Modes

1. **Draft Model Only** — small dense model baseline (fast, low memory, lower quality)
2. **GPT-OSS Full-Depth** — all 24 layers, reference quality
3. **Track B Naive Cascade** — chunked draft-then-verify with first-divergence correction
4. **Track A L22 top1_agree** — canonical same-model adaptive-depth frontier

### Key Metrics

- **Quality**: exact match rate vs full-depth baseline (token-level agreement)
- **Speed**: TTFT (time to first token) and generation TPS
- **Memory**: peak memory per mode (draft-only vs single large model vs dual-model)
- **Cascade-specific**: acceptance rate, correction tokens, draft iterations

### Design Constraints

- **Isolated from Track A engine**: `transcender_engine.py` is NOT modified
- **Same 5 evaluation prompts**: quantum entanglement, French Revolution, recursion, TCP vs UDP, photosynthesis
- **Warmup-corrected**: P1 (quantum entanglement) excluded from aggregate stats
- **Per-model prompt formatting**: Harmony template for GPT-OSS, native chat template for draft
- **No KV sharing**: Naive chunked verify, not production speculative decoding

### Positioning

Track B is a **comparison baseline**, not a pivot. Its purpose is to validate that Track A's same-model approach occupies a meaningfully different point on the quality/speed/memory frontier than the obvious alternative of "just use a smaller model for easy prompts."

**Expected outcome**: Track B will likely show:
- Draft-only is faster and uses less memory, but lower quality
- Cascade adds orchestration overhead (TTFT penalty, dual-model memory)
- In this naive comparison on the local runtime, Track A may occupy a better frontier: single model, no orchestration, quality near-parity with full depth

**If Track B wins cleanly**: Track A's value proposition weakens — simpler dual-model routing may dominate same-model adaptive depth. This would be an honest and important finding.

### Running

```bash
python transcender_track_b_benchmark.py \
  --large-model /path/to/gpt-oss-20b \
  --draft-model /path/to/gemma-3-4b-it \
  --output transcender_track_b_benchmark.json
```

Draft model auto-detection searches common local paths if `--draft-model` is not specified.

### Track B Result (2026-03-22)

Track B did not beat Track A for this model pair. The naive cascade achieved 0.28 TPS and 0.026 exact match versus Track A's 20.22 TPS and 0.969 exact match. Memory was 7.23 GB higher (20.19 vs 12.96 GB). The vocabulary/distribution mismatch between Gemma and GPT-OSS destroyed the acceptance rate. These deltas are specific to this naive cascade implementation, this architecturally mismatched model pair, and this local Apple MLX runtime; they do not characterize cascade strategies in general. Track B remains a completed negative comparison baseline.

---

## 5c. TRACK C — DENSE MODEL GENERALIZATION (2026-03-22)

### What Track C Is

A methodology validation track testing whether Transcender's depth-frontier observations reproduce on dense models. Gemma 3 4B-IT is the primary dense case study; focused late-checkpoint follow-up on Llama 3.1 8B and Mistral 7B extends the benchmark result beyond Gemma alone in this local Apple MLX runtime.

### Files

| File | Purpose |
|------|---------|
| `transcender_track_c_gemma_profile.py` | KL depth profiling for Gemma |
| `transcender_track_c_gemma_benchmark.py` | Adaptive-depth benchmark (6 modes) |
| `transcender_track_c_gemma_selective_depth.py` | Real selective-depth speed-validation benchmark |
| `transcender_track_c_dense_family_validation.py` | Focused Llama/Mistral dense-family validation |
| `gemma_kl_profile.json` / `.png` | Profiling artifacts |
| `transcender_track_c_gemma_results.json` | Benchmark results |
| `transcender_track_c_gemma_selective_depth_results.json` | Selective-depth results artifact |
| `transcender_track_c_gemma_selective_depth_L20_results.json` | L20 selective-depth probe artifact |
| `recon_llama3_8b.json` / `recon_mistral7b.json` | Dense-family KL recon artifacts |
| `transcender_track_c_llama3_8b_results.json` / `transcender_track_c_mistral7b_results.json` | Dense-family late-checkpoint validation artifacts |

### Track C Result (2026-03-22)

**KL Profiling:**
- 90% KL reduction at Layer 20, 95% at Layer 31
- KL plateau from L20-L29 (compositional work, no logit convergence)
- Geometric separation 2.46x (weaker than GPT-OSS's 4.11x)

**Adaptive Benchmark:**
- Fixed exit catastrophic at all layers (L16: 1%, L20: 2.6%, L31: 19.8% exact match)
- top1_agree blending at L31: 0.807 exact match (vs 0.198 for fixed exit at same layer)
- Naive blending at L31: 0.688 exact match (top1_agree outperforms by +11.98pp)
- Real selective-depth at L31 was operationally correct but weak: margin mode reached 18.26 TPS / 0.229 exact / 0.2% average layers saved; entropy mode reached 18.20 TPS / 0.208 exact / 2.0% average layers saved. Neither beat full depth (20.05 TPS).
- L20 selective-depth increased the skip budget but still failed practically: entropy mode reached 13.26 TPS / 0.073 exact / 15.7% average layers saved; margin mode reached 11.29 TPS / 0.083 exact / 5.3% average layers saved; hybrid mode reached 11.37 TPS / 0.125 exact / 2.4% average layers saved. None beat the matched L20 full-depth baseline (15.16 TPS).
- Focused dense-family follow-up reproduced the same benchmark pattern at a late checkpoint on two additional dense families. Llama 3.1 8B Instruct 4bit at L29: fixed exit reached 0.151 exact; compute-both `top1_agree` reached 1.000 exact; real selective-depth entropy reached 16.60 TPS / 0.469 exact / 0.3% average layers saved versus a matched full-depth baseline of 17.81 TPS. Mistral 7B Instruct v0.3 4bit at L29: fixed exit reached 0.109 exact; compute-both `top1_agree` reached 1.000 exact; real selective-depth entropy reached 17.25 TPS / 0.318 exact / 0.2% average layers saved versus a matched full-depth baseline of 29.81 TPS.
- The dense-family KL profiles were not Gemma-identical: Gemma resolved at L30-L33 after a late L20-L29 plateau; Llama's heuristic resolution entry was L27 with a mild plateau at L12-L14; Mistral's heuristic resolution entry was L26 with an earlier plateau at L2-L8.

**Interpretation:** Track C now has a stronger cross-family dense benchmark result. Fixed exit fails materially on all three tested dense families in this local Apple MLX runtime. Compute-both agreement-aware composition recovers substantial quality on all three. Real selective-depth was operationally correct on Gemma, Llama, and Mistral late-checkpoint probes, but remained practically weak after replay/cache repair. This broadens the dense-model limitation from a Gemma-only benchmark result to a multi-family dense benchmark result, while the KL-profile evidence remains family-sensitive rather than uniformly Gemma-like. The next direction is not repeating the same late-checkpoint margin/entropy mechanism, but improving continuation criteria, token-difficulty-aware routing, and cache-aware continuation. See `NEXT_EXPERIMENTS.md`.

---

## 6. "IF I WANTED TO MAKE THIS SELLABLE"

### Already Valuable (Right Now)

- **The Subspace Paradox documentation** is a genuine contribution applicable to all early-exit transformer architectures. Should be submitted as a short preprint. No further experiments needed.
- **Logit-space blending** is the correct implementation pattern. Teams implementing MoD or early-exit and using hidden-state blending are making a verifiable error.
- **`TranscenderModel` routing diagnostics** — exit_probs, layer_counts, son_scores on any HF causal LM — are useful for model interpretability today, independent of any efficiency claim.

### Research-Only (Not Product-Ready)

- PPL numbers on GPT-2 are proof-of-mechanism only. Nobody ships GPT-2.
- Trained Son Router checkpoints are GPT-2-specific and non-transferable.
- The MLX engine for GPT-OSS 20B is unvalidated against ground truth.

### What Could Become a Tool (~3 Days of Work)

**Transcender Profiler** — takes any HF model + sample corpus, outputs:
- Per-layer logit convergence profile (when does the model "know" the answer?)
- Token-class routing distribution (% of function words that could exit early)
- Pareto frontier: quality-vs-savings curve for 5 exit points

Ship as `pip install transcender-profiler`. Useful to any ML team evaluating whether their model is a good candidate for early-exit.

### What Could Become a Service (1–3 Months)

Transcender-as-inference-layer: a trained `TranscenderModel` export deployable as a drop-in inference server. Requires:
1. Demonstrated PPL + TTFT results on LLaMA-3 8B or larger.
2. Wall-clock latency benchmarks showing real speedup (not just layer-count proxy).
3. Token budget mode: "stay within X% of baseline PPL" as a runtime constraint.

### Before Calling It a Product

1. **End-to-end wall-clock speedup** — actual tokens/second on real hardware. Physical layer skipping requires sparse batching (not just logit-space blending).
2. **Standardized training recipe** — `transcender-train --model gpt2 --exit-layer 6` in <1 hour.
3. **Validation on one non-GPT-2 model** — LLaMA-3 8B is the minimum.
4. **A quality contract**: "Son Router at Layer 16 of LLaMA-3-8B, hard mode, achieves X% savings with PPL increase ≤ Y%." Users need a commitment, not a research caveat.

---

## 7. PATCH ORDER — 2-WEEK EXECUTION SEQUENCE

### Week 1: Stabilize and Consolidate

| Day | Task | Risk |
|-----|------|------|
| **Day 1 (Mon)** | Fix 1: Remove `output_attentions` from block calls in both files | Low |
| **Day 1 (Mon)** | Fix 5: Update misleading comment in deep-layer loop | Zero |
| **Day 2 (Tue)** | Fix 2: Correct exit rate computation in `benchmark.py` | Low |
| **Day 2 (Tue)** | Fix 4: Correct `avg_layers` denominator | Low |
| **Day 3 (Wed)** | Experiment 3: Run equivalence check, confirm <1e-4 max diff | — |
| **Day 3 (Wed)** | Fix 3: Run subspace benchmark, hard-code PCA values in chapter3 | — |
| **Day 4 (Thu)** | Architecture consolidation: extract `_blend_logits`, `_compute_loss` into `TranscenderModel` | Medium |
| **Day 4 (Thu)** | Add `SGAModel` deprecation shim | Low |
| **Day 5 (Fri)** | Package restructure: create `transcender/` package, add `pyproject.toml`, `pip install -e .` | Low |
| **Day 5 (Fri)** | Update all experiment scripts to use package imports. Verify all benchmarks run. | Medium |

### Week 2: Research Experiments

| Day | Task |
|-----|------|
| **Day 8 (Mon)** | Experiment 2: Layer-ablation sweep L1–L11, 5 epochs each. Set overnight. |
| **Day 9 (Tue)** | Experiment 6: Routing semantics study. Interpret layer-ablation results from Day 8. |
| **Day 10 (Wed)** | Experiment 4: Temperature annealing. Train with τ schedule at best exit layer from Day 8. |
| **Day 11 (Thu)** | Experiment 1: Add `EntropyGate` class + iso-savings duel vs Son Router. |
| **Day 12 (Fri)** | Experiment 5: LLaMA-3 8B if hardware available. Else: build Profiler CLI. |

---

## DO THIS FIRST TOMORROW MORNING

These three, in this order, before touching anything else:

### Step 1 — Confirm the hook works correctly (3 minutes)

```bash
cd /Users/junson/Documents/GitHub/sga-poc
python -c "
import torch
from model_injector import SGAModel
m = SGAModel('gpt2', exit_after_layer=2); m.eval()
ids = torch.randint(0, 50257, (1, 8))
out = m(ids)
print('exit_probs shape:', out['routing_info']['exit_probs'].shape)
print('logits type:', type(out['logits']))
print('Hook path: CONFIRMED OK')
"
```

If this errors or output shows a tuple instead of a tensor, the hook is broken. Fix that before anything else.

### Step 2 — Remove `output_attentions` from block calls

In `model_injector.py` line 134, delete the line:
```python
output_attentions=(i == router_layer),
```

In `transcender_injector.py` line 240, delete the same. Then re-run Step 1 to confirm.

### Step 3 — Fix the broken exit rate and run benchmark experiment 1

Apply Fix 2 (the exit rate accumulation), then:
```bash
python benchmark.py 2>&1 | head -60
```

Confirm the printed table shows a non-constant exit rate column. This is your first sanity check that the benchmark infrastructure is reliable before running any further experiments.
