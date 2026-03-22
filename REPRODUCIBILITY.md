# Reproducibility Guide

Instructions for reproducing the validated results in this repository.

---

## Environment

**Hardware:** Apple M1 Pro, 32 GB unified memory
**Runtime:** Apple MLX
**Python:** >= 3.10

```bash
# Clone and install
git clone <repo-url> && cd transcender
pip install -e ".[mlx]"
```

All commands below assume execution from the repository root.

---

## Track A — Canonical Same-Model Adaptive Depth

**Model required:** GPT-OSS 20B (MXFP4 quantization, local path)

```bash
# Exit layer comparison (L22 vs L23)
python scripts/track_a/transcender_exit_layer_benchmark.py

# Blend strategy benchmark
python scripts/track_a/transcender_top1_agree_benchmark.py
```

**Expected result:** L22 `top1_agree` achieves ~0.969 exact match with ~49.5% real layer savings.

**Scope warning:** TPS and TTFT numbers are hardware-specific. Exact match and layers-saved are deterministic with greedy decoding and fixed prompts.

---

## Track B — Scoped Negative Cascade Baseline

**Models required:** GPT-OSS 20B + Gemma 3 4B-IT (local paths)

```bash
python scripts/track_b/transcender_track_b_benchmark.py \
  --large-model /path/to/gpt-oss-20b-raw \
  --draft-model /path/to/gemma-3-4b-it
```

**Expected result:** Naive cascade reaches ~0.026 exact match, ~0.28 TPS. Track A dominates on all metrics for this model pair.

**Scope warning:** This result is specific to this naive cascade implementation, this model pair, and this local Apple MLX runtime. It does not characterize cascade strategies in general.

---

## Track C — Dense Model Validation

### Gemma 3 4B-IT

**Model required:** Gemma 3 4B-IT (local path)

```bash
# KL depth profiling
python scripts/track_c/transcender_track_c_gemma_profile.py \
  --model /path/to/gemma-3-4b-it

# Six-mode adaptive benchmark
python scripts/track_c/transcender_track_c_gemma_benchmark.py \
  --model /path/to/gemma-3-4b-it \
  --early-layer 16 --mid-layer 20 --late-layer 31

# Selective-depth speed validation (L31)
python scripts/track_c/transcender_track_c_gemma_selective_depth.py \
  --model /path/to/gemma-3-4b-it --late-layer 31

# Selective-depth probe (L20)
python scripts/track_c/transcender_track_c_gemma_selective_depth.py \
  --model /path/to/gemma-3-4b-it --late-layer 20 \
  --margin-threshold 0.02 --entropy-threshold 0.65 --include-hybrid-mode
```

### Dense Family Follow-Up (Llama + Mistral)

**Models required:** Llama 3.1 8B Instruct 4bit, Mistral 7B Instruct v0.3 4bit (local paths)

```bash
# Llama 3.1 8B
python scripts/track_c/transcender_track_c_dense_family_validation.py \
  --model /path/to/llama-3.1-8b-instruct-4bit \
  --family llama \
  --profile-output artifacts/recon/recon_llama3_8b.json \
  --benchmark-output artifacts/dense_followup/transcender_track_c_llama3_8b_results.json \
  --late-layer 29 --entropy-threshold 0.15

# Mistral 7B
python scripts/track_c/transcender_track_c_dense_family_validation.py \
  --model /path/to/mistral-7b-instruct-v0.3-4bit \
  --family mistral \
  --profile-output artifacts/recon/recon_mistral7b.json \
  --benchmark-output artifacts/dense_followup/transcender_track_c_mistral7b_results.json \
  --late-layer 29 --entropy-threshold 0.15
```

**Expected results:**
- Fixed exit fails materially on all three dense families
- Compute-both `top1_agree` recovers substantial quality (0.807 on Gemma, 1.000 on Llama and Mistral)
- Real selective-depth physically skips layers but does not recover a practical speed-quality frontier

**Scope warning:** All results are on a single Apple M1 Pro with MLX. TPS numbers are not comparable to GPU inference. The dense benchmark limitation generalizes across the tested families on this runtime, but the internal KL profile is family-sensitive.

---

## Exploratory Probes

These are not paper-supporting benchmarks. They are preserved for completeness.

```bash
# Gemma L20 advanced probe
python scripts/exploratory/transcender_track_c_gemma_advanced_probe.py \
  --model /path/to/gemma-3-4b-it

# Llama family-sensitive probe
python scripts/exploratory/transcender_track_c_llama_family_sensitive_probe.py \
  --model /path/to/llama-3.1-8b-instruct-4bit

# Llama cache-aware probe
python scripts/exploratory/transcender_track_c_dense_cache_aware_probe.py \
  --model /path/to/llama-3.1-8b-instruct-4bit
```

---

## Artifact Verification

All JSON artifacts in `artifacts/` are original benchmark outputs. To verify:
1. Re-run the corresponding script
2. Compare the new JSON output against the stored artifact
3. Exact match and layer savings should be deterministic; TPS will vary by hardware state

See [RESULTS_INDEX.md](RESULTS_INDEX.md) for the complete artifact inventory.
