# Track A GPU Validation Runbook

## Goal

Run the cheapest credible off-MLX reproduction of the Track A penultimate-layer frontier without weakening the manual-reference trust model.

This is an external-validity test, not a serving benchmark:

- first empirically validated GPU baseline: Qwen3-30B-A3B
- canonical sparse supporting evidence beyond Qwen3: GPT-OSS 20B
- strong sparse-MoE generalization: Mixtral-8x7B-Instruct-v0.1
- strong first dense control: Mistral-7B-Instruct-v0.3
- Gemma 3 dense controls should be treated as checkpoint-specific size variants:
  - `google/gemma-3-4b-it` now provides genuine dense-family supporting evidence on this path
  - `google/gemma-3-12b-it` for a mid-scale dense control
  - `google/gemma-3-27b-it` for a large dense control
- runtime: HuggingFace Transformers on CUDA
- comparison: full depth vs penultimate layer vs one layer earlier
- first step: single-prompt debug trace
- second step: 3-prompt smoke test
- third step: full N=63 run only if the trace and smoke test are clean

Paper role of this path:

- Track~A on MLX remains the primary empirical result.
- Track~B remains the scoped negative cascade baseline.
- Track~C remains the dense-model limitation and boundary-condition evidence.
- This GPU path is an offline diagnostic extension that strengthens the interpretation of penultimate acceptance; it does not replace the Track~A/B/C structure.

## Measurement Fix

The earlier HF scaffold was not trustworthy enough to interpret frontier numbers.

Two specific problems:

- it relied on black-box hidden-state capture behavior instead of an explicit layer runner
- it reported `top1_agree` exact match against full depth even though the composed token falls back to the full-depth token on disagreement, which can make the metric look artificially perfect

The corrected script now uses a manual reference path:

- prefill once
- decode step-by-step
- run `embed_tokens -> decoder layers -> final norm -> lm_head` explicitly
- extract raw candidate logits for the requested exit layers and full depth
- report raw exit tokens and composed tokens separately

Use the debug trace first. Do not trust aggregate benchmark output until the trace shows that the tested exit-layer candidates truly differ from full depth on at least some steps.

## Metric Semantics

- `raw_exit_*`: compares the raw intermediate-layer candidate sequence against the full-depth sequence under shared full-depth context. This is the primary diagnostic metric for whether the penultimate-layer frontier is really present off-MLX.
- `composed_*`: measures the conservative `top1_agree` composition path. Because disagreement falls back to the full-depth token, these metrics are a composition diagnostic, not a substitute for raw divergence.
- `avg_top1_agreement_rate`: the per-token fraction of raw top-1 agreement between the exit layer and full depth.

## Stage A And Stage B

The token-row export and offline proxy scripts separate two different questions:

- `Stage A`: was the earlier layer already correct relative to full depth?
  - oracle label: `earlier_correct`
  - typical baselines: `earlier_entropy`, `earlier_margin`
- `Stage B`: conditional on Stage A already missing, is the penultimate layer enough or is full depth still needed?
  - oracle labels: `need_penultimate`, `need_full_depth`
  - typical baselines: `penultimate_entropy`, `penultimate_margin`, and adjacent-agreement variants

This split matters because the current negative result is specifically about Stage B. Adjacent agreement is not the same thing as penultimate safety. A token can show strong local agreement or low entropy and still be in a regime where the final layer would correct the penultimate token.

Current Stage B interpretation:

- `penultimate_entropy`: a crude confidence-style baseline
- `penultimate_margin`: another confidence-style baseline
- `adjacent_top1_agree` and overlap / logit-delta variants: relation heuristics that have not held up as a reliable cross-family acceptance rule
- `karma`: a small offline logistic model over penultimate and adjacent-relation features, interpreted as `probability_of_need_full_depth`

In other words, the current framing is closer to risk estimation than to raw confidence thresholding. `karma` is not a serving policy. It is an offline held-out score used to test whether Stage B is better modeled as risk of still needing full depth.

## Two-Axis Framing

Keep two axes separate:

- architecture class:
  - sparse MoE
  - dense
- validation status:
  - empirically validated on this GPU manual-reference path
  - code-path supported, pending real runs
  - blocked by model access

The GPU path is for structural validation only. It is not canonical paper evidence, it is not a serving-speed benchmark, and `composed_exact_match = 1.0` is not the main frontier claim.

## Architecture Classes And Validation Status

### Sparse MoE

| Model / checkpoint | Status on this GPU path | Notes |
|---|---|---|
| `Qwen/Qwen3-30B-A3B` | Empirically validated | Current GPU manual-reference baseline; use this to confirm the path is sane before expanding |
| `openai/gpt-oss-20b` | Empirically validated | Verified with `--quantize none --exit-layers 21 22` on NVIDIA H200 (bfloat16); canonical sparse supporting evidence on this GPU path |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | Empirically validated | Verified with `--quantize 4bit --exit-layers 29 30`; very clean monotonic penultimate advantage |

### Dense

| Model family / checkpoint | Status on this GPU path | Notes |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | Empirically validated | Verified with `--quantize 4bit --exit-layers 29 30`; strong penultimate advantage for the first dense control |
| `meta-llama/Llama-3.1-8B-Instruct` | Blocked by gated access | Code path is supported, but the current token still cannot download the checkpoint from Hugging Face |
| `google/gemma-3-4b-it` | Empirically validated | Verified with `--device cuda --quantize none --exit-layers 31 32`; numerically sane dense-family supporting evidence |
| `google/gemma-3-12b-it` | Empirically validated | Verified with `--device cuda --quantize none --exit-layers 41 42`; valid but materially weaker than the 4B and 27B checkpoints |
| `google/gemma-3-27b-it` | Empirically validated | Verified with `--device cuda --quantize none --exit-layers 59 60`; strongest current Gemma 3 result with a clean monotonic penultimate advantage |
| older Gemma / Gemma2 text checkpoints | Code-path supported, pending real run | Dense-family follow-up controls, not first-run targets |

Gemma 3 should be discussed as a dense family with checkpoint-specific size variants, not as one undifferentiated model. For this Track A validation flow, the practical checkpoints are `4B`, `12B`, and `27B`. Smaller `270M` and `1B` Gemma 3 text checkpoints exist, but they are not the main dense-control targets here unless there is a specific reason to probe very small-scale behavior.

## Current Verified Outcomes

- `openai/gpt-oss-20b` with `--quantize none --exit-layers 21 22` on NVIDIA H200 (bfloat16): sane trace; N=63 raw-exit exact match `0.808` at `L21` and `0.879` at `L22`. The model ships MXFP4-quantized weights that transformers dequantizes to bfloat16; `resolve_manual_reference_load_dtype` now detects this and matches the dtype automatically. Earlier fp16 runs hit a `grouped_mm` dtype mismatch in the MoE path.
- `mistralai/Mixtral-8x7B-Instruct-v0.1` with `--quantize 4bit --exit-layers 29 30`: sane trace; N=63 raw-exit exact match `0.667` at `L29` and `0.837` at `L30`; `L30` is at least as strong as `L29` on `63/63` prompts and strictly better on `63/63`.
- `mistralai/Mistral-7B-Instruct-v0.3` with `--quantize 4bit --exit-layers 29 30`: sane trace; N=63 raw-exit exact match `0.715` at `L29` and `0.864` at `L30`; `L30` is at least as strong as `L29` on `63/63` prompts and strictly better on `62/63`.
- `google/gemma-3-4b-it` with `--device cuda --quantize none --exit-layers 31 32`: sane trace and finite hidden/logit diagnostics; N=63 raw-exit exact match `0.833` at `L31` and `0.900` at `L32`; `L32` is at least as strong as `L31` on `58/63` prompts and strictly better on `54/63`.
- `google/gemma-3-12b-it` with `--device cuda --quantize none --exit-layers 41 42`: sane trace and finite hidden/logit diagnostics; N=63 raw-exit exact match `0.528` at `L41` and `0.562` at `L42`; `L42` is at least as strong as `L41` on `54/63` prompts and strictly better on `46/63`. This is valid but comparatively weak dense-family evidence.
- `google/gemma-3-27b-it` with `--device cuda --quantize none --exit-layers 59 60`: sane trace and finite hidden/logit diagnostics; N=63 raw-exit exact match `0.813` at `L59` and `0.932` at `L60`; `L60` is at least as strong as `L59` on `63/63` prompts and strictly better on `63/63`. This is the cleanest Gemma 3 result in the current GPU Track A set.
- `meta-llama/Llama-3.1-8B-Instruct` remains unrun on this GPU Track A path because Hugging Face access is still gated and pending review.

These are structural off-MLX validation results under the manual-reference path. They do not turn this script into a serving-speed benchmark, and `composed_exact_match = 1.0` remains a fallback-path diagnostic rather than the main claim.

## What The Script Supports

The manual-reference path now supports these HF backbone families explicitly:

- sparse MoE:
  - `qwen2_moe` / Qwen3
  - `gpt_oss`
  - `mixtral`
- dense:
  - `llama`
  - `mistral`
  - `gemma`
  - `gemma2`
  - `gemma3` text checkpoints

What is architecture-specific:

- Gemma and Gemma2 require the same hidden-state scaling used by the HF backbone before the decoder stack.
- GPT-OSS examples should use `--quantize none`, not the BitsAndBytes `4bit` path used for the other example checkpoints. The Hugging Face GPT-OSS checkpoint already carries its own MXFP4 quantization config, and overriding it with `BitsAndBytesConfig` fails. The script automatically detects this and loads in bfloat16 to match the dequantized weight dtype.
- GPT-OSS, Gemma2, and Gemma 3 use per-layer attention-mask selection (`full_attention` vs `sliding_attention`), and the manual path now mirrors that explicitly.
- Gemma 3 text checkpoints use layer-type-specific rotary embeddings, so the manual path selects position embeddings per decoder layer instead of reusing a single generic object.
- Gemma 2 and Gemma 3 can apply final logit softcapping; the manual path now mirrors that before top-1 comparison.
- Some decoder layers return tuples instead of a bare hidden-state tensor; the manual path now extracts the hidden state explicitly instead of assuming a single return type.
- Multimodal Gemma 3 checkpoints are intentionally rejected. This path only supports text causal-LM checkpoints such as `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, and `google/gemma-3-27b-it`.

If a loaded model does not match one of the explicitly supported backbones, the script fails fast with a clear architecture message. If the model is sharded across multiple devices, it also fails fast rather than silently measuring a partly-wrong path.

## Recommended Test Order

Use this order to build confidence without over-interpreting one run:

1. `Qwen/Qwen3-30B-A3B`
   Empirically validated sparse-MoE baseline for the manual-reference path.
2. `openai/gpt-oss-20b`
   Highest-priority sparse-MoE external-validity target because there is an MLX Track A reference.
3. `mistralai/Mixtral-8x7B-Instruct-v0.1`
   Second sparse-MoE target. Useful for checking whether the frontier shape extends to a third sparse family.
4. `mistralai/Mistral-7B-Instruct-v0.3`
   First dense control. It is still the cleanest dense baseline because it avoids Gemma-family checkpoint and chat-template complications on the first pass.
5. `google/gemma-3-4b-it`
   Small Gemma 3 dense control. Use this first if you want a Gemma-family dense comparison without jumping to a large checkpoint.
6. `google/gemma-3-12b-it`
   Mid-scale Gemma 3 dense control. Use this to add a size-aware dense comparison axis inside one family.
7. `google/gemma-3-27b-it`
   Large Gemma 3 dense control. Only move here if the smaller Gemma 3 checkpoints are already behaving sensibly and the hardware budget is acceptable.
8. `meta-llama/Llama-3.1-8B-Instruct`
   Additional dense control if gated access is available.

The script defaults to the model's penultimate-minus-one and penultimate layers. For a model with `N` layers, the default tested exits are `L(N-3)` and `L(N-2)`, with full depth at `L(N-1)`.

## What To Inspect

The important outputs are:

- debug trace JSON:
  - whether raw penultimate and penultimate-minus-one tokens ever differ from full depth
  - whether the composed `top1_agree` token simply returns to full depth on raw disagreement
  - whether the penultimate layer diverges later or less often than the earlier layer
- benchmark JSON:
  - `raw_exit_exact_match_rate`
  - `raw_exit_prefix_match_tokens`
  - `raw_exit_first_divergence_position`
  - `top1_agreement_rate`
  - penultimate vs penultimate-minus-one comparison

Do not treat `composed_exact_match = 1.0` as the frontier claim. That is often expected because `top1_agree` falls back to the full-depth token on disagreement.

## Trust Order

Use the GPU path in this order:

1. Run a single-prompt debug trace.
2. Run `analyze_debug_trace.py` on that JSON.
3. If the verdict is `sane`, run the multi-prompt smoke test or full benchmark.
4. If the verdict is `inconclusive`, try another prompt before moving on.
5. Do not interpret aggregate benchmark output if the trace verdict is `suspicious`.

## Provider Recommendation

### Pick: RunPod

RunPod is the best first-run choice for this exact experiment because it is the cleanest combination of:

- low setup friction
- plain SSH access to a single GPU box
- low hourly cost on a 48 GB card
- simple destroy path
- low risk of leaving background infrastructure around

### Why not the others for this first run

- **Vast.ai:** often cheaper, but this is a marketplace with host-to-host variability. Billing also includes storage and bandwidth details that are easier to get wrong on a first run. Good second choice, not the safest first one.
- **Modal:** operationally elegant, and the free-credit story is attractive, but the default workflow is app/sandbox oriented rather than "SSH into one box and run one existing Python script." That is extra adaptation for no scientific upside on this first pass.
- **Lambda Cloud:** straightforward SSH workflow, but the official on-demand A6000 price is higher than RunPod for the same memory class. It also uses card-based account billing rather than the tighter prepaid-credit model that helps cap first-run risk.

### Billing and signup

- **Yes**, you need billing set up up front on RunPod.
- RunPod requires funded account credits before Pod launch.
- Official billing docs say you can load as little as **$10**.
- Credits are non-refundable, so keep the first deposit small.

## Safest First Instance

### Use: 1x RTX A6000 48 GB, on-demand

Why this is enough:

- Qwen3-30B-A3B in 4-bit should fit on a 48 GB card with comfortable headroom.
- The manual reference path keeps decoder caches and explicit per-layer projections live, so 24 GB is the wrong place to start.
- A6000 is much cheaper than an A100 while still giving enough VRAM for a conservative first reproduction.

Preferred tier:

- **RunPod Secure Cloud A6000** if available and the price delta is small
- otherwise **RunPod Community Cloud A6000**

## Current Provider Facts To Use

These were the operative facts used for this runbook:

- RunPod official A6000 page lists **RTX A6000 from $0.49/hr** and also shows a lower community-cloud price point.
- RunPod Pod pricing docs describe on-demand Pods as the non-interruptible option and note that stopped Pods can still incur persistent volume-disk charges.
- RunPod billing docs say credits must be funded before launch and can be loaded in small amounts.
- Lambda official pricing lists **1x A6000 48 GB at $0.80/hr**.
- Modal official pricing lists **A100 40 GB at $0.000583/sec** and **RTX PRO 6000 at $0.000842/sec**, which is materially more expensive than RunPod for this task.
- Vast official billing docs confirm prepaid credits plus continuing storage charges for stopped instances.

## Launch Steps

### 1. Create the account

1. Sign up at RunPod.
2. Verify email.
3. Add a payment method or other supported funding method.
4. Load only **$10-$15** for the first run.

### 2. Deploy the Pod

In the RunPod web console:

1. Open **Pods**.
2. Click **Deploy**.
3. Pricing mode: **On-Demand**.
4. GPU: **RTX A6000 48 GB**.
5. Template: current official **RunPod PyTorch** template.
6. Container disk: **50 GB**.
7. Volume disk: **0 GB**.
8. Network volume: **none**.
9. Deploy.

Why these storage choices matter:

- container disk is enough for the repo, Python environment, and model cache
- no volume disk means no persistent-disk surprise charge after you terminate

### 3. Connect by SSH

From the Pod detail page, open **Connect** and copy the SSH command.

Typical shape:

```bash
ssh root@<POD_IP> -p <SSH_PORT> -i ~/.ssh/id_ed25519
```

## Environment Setup

Run these on the pod after SSH login:

```bash
nvidia-smi

git clone <repo-url>
cd transcender-mlx

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[gpu]"
```

### Verify CUDA and PyTorch

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

### Verify model load before any benchmark

```bash
python - <<'PY'
import torch
from scripts.track_a_gpu.transcender_gpu_reproduction import load_model_and_tokenizer, get_model_parts

model, tokenizer = load_model_and_tokenizer(
    "Qwen/Qwen3-30B-A3B",
    "4bit",
    "cuda",
)
parts = get_model_parts(model)
print("tokenizer_ok", tokenizer.__class__.__name__)
print("layers", parts.num_layers)
print("device", parts.device)
print("lm_head_ok", parts.lm_head.__class__.__name__)
print("cuda_available", torch.cuda.is_available())
print("first_param_device", next(model.parameters()).device)
PY
```

If that fails, stop there. Do not start the smoke test.

## Single-Prompt Debug Trace

Run this before trusting any aggregate benchmark output:

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model Qwen/Qwen3-30B-A3B \
  --quantize 4bit \
  --debug-trace \
  --prompt-id P2 \
  --max-new-tokens 16 \
  --output artifacts/track_a_gpu/qwen3_trace_p2.json
```

What this gives you:

- one prompt only
- one JSON file
- per-token records for:
  - full depth
  - raw L45 candidate
  - raw L46 candidate
  - composed `top1_agree` token
- running first-divergence position and top-1 agreement rate for each layer

Only after that trace looks sane should you move to the multi-prompt smoke test or full N=63 run.

### Trace Analysis Helper

After writing a trace JSON, summarize it with:

```bash
python scripts/track_a_gpu/analyze_debug_trace.py \
  artifacts/track_a_gpu/qwen3_trace_p2.json
```

This reports:

- first raw divergence position for L45 and L46
- how many steps raw L45 and raw L46 differ from full depth
- whether composed `top1_agree` simply falls back to full depth on raw disagreement
- a concise verdict: `sane`, `inconclusive`, or `suspicious`

The helper auto-detects whichever exit layers are present in the trace JSON. You do not need to hard-code `L45/L46` for other models.

## Multi-Prompt Smoke Test

This confirms:

- model load works
- the manual layer runner works
- the default previous-layer and penultimate-layer candidate projection paths run
- full-depth, raw exit-layer candidates, and composed tokens are all emitted separately
- JSON artifact writing works

Command:

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model Qwen/Qwen3-30B-A3B \
  --quantize 4bit \
  --prompt-limit 3 \
  --max-new-tokens 16 \
  --output artifacts/track_a_gpu/qwen3_gpu_smoke.json
```

Quick verification:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/track_a_gpu/qwen3_gpu_smoke.json")
data = json.loads(path.read_text())
print("exists", path.exists())
print("prompt_count", len(data["prompt_results"]))
print("aggregate_keys", sorted(data["aggregates"].keys()))
print("scope", data["aggregates_scope"])
PY
```

Expected operational result:

- the script finishes
- `qwen3_gpu_smoke.json` is written
- output contains raw-exit and composed summaries for the two tested exit layers

## Full Run

Run this only after the smoke test is clean:

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model Qwen/Qwen3-30B-A3B \
  --quantize 4bit \
  --max-new-tokens 48 \
  --output artifacts/track_a_gpu/qwen3_gpu_reproduction_n63.json
```

This uses the full fixed prompt suite in the script:

- P1 warmup
- 63 scored prompts

### Benchmark Summary Helper

After the benchmark JSON is written, summarize it with:

```bash
python scripts/track_a_gpu/summarize_benchmark.py \
  artifacts/track_a_gpu/qwen3_gpu_reproduction_n63.json
```

This reports:

- `raw_exit_avg_exact_match` by layer
- `composed_avg_exact_match` by layer
- `avg_top1_agreement_rate` by layer
- whether the penultimate layer is consistently at least as strong as the previous layer on per-prompt raw-exit exact match

### Multi-Model Comparison Helper

To compare multiple benchmark JSON files side by side:

```bash
python scripts/track_a_gpu/compare_benchmarks.py \
  artifacts/track_a_gpu/qwen3_gpu_reproduction_n63.json \
  artifacts/track_a_gpu/gpt_oss_20b_gpu_reproduction_n63.json \
  artifacts/track_a_gpu/mixtral_8x7b_gpu_reproduction_n63.json \
  artifacts/track_a_gpu/mistral_7b_gpu_reproduction_n63.json \
  artifacts/track_a_gpu/gemma3_4b_gpu_reproduction_n63.json \
  artifacts/track_a_gpu/gemma3_12b_gpu_reproduction_n63.json \
  artifacts/track_a_gpu/gemma3_27b_gpu_reproduction_n63.json
```

This prints one compact per-model summary with:

- raw/composed aggregate exact match by tested layer
- aggregate top-1 agreement by tested layer
- whether the penultimate layer is consistently at least as strong as the previous layer

## Example Commands By Model

These examples pin the exact verified command paths for the currently checked models. The main signal remains raw-exit divergence, first-divergence position, raw-exit exact match, top-1 agreement, and penultimate-vs-earlier-layer behavior, not composed exact match.

### Qwen3-30B-A3B (Sparse MoE, validated baseline)

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model Qwen/Qwen3-30B-A3B \
  --quantize 4bit \
  --debug-trace \
  --prompt-id P2 \
  --max-new-tokens 16 \
  --output artifacts/track_a_gpu/qwen3_trace_p2.json

python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model Qwen/Qwen3-30B-A3B \
  --quantize 4bit \
  --max-new-tokens 48 \
  --output artifacts/track_a_gpu/qwen3_gpu_reproduction_n63.json

python scripts/track_a_gpu/summarize_benchmark.py \
  artifacts/track_a_gpu/qwen3_gpu_reproduction_n63.json
```

### GPT-OSS 20B (Sparse MoE, highest-priority external validity)

Use `--quantize none` here. Do not reuse the BitsAndBytes `4bit` example path for GPT-OSS.

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model openai/gpt-oss-20b \
  --quantize none \
  --exit-layers 21 22 \
  --debug-trace \
  --prompt-id P2 \
  --max-new-tokens 16 \
  --output artifacts/track_a_gpu/gpt_oss_20b_trace_p2.json

python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model openai/gpt-oss-20b \
  --quantize none \
  --exit-layers 21 22 \
  --max-new-tokens 48 \
  --output artifacts/track_a_gpu/gpt_oss_20b_gpu_reproduction_n63.json

python scripts/track_a_gpu/summarize_benchmark.py \
  artifacts/track_a_gpu/gpt_oss_20b_gpu_reproduction_n63.json
```

### Mixtral-8x7B-Instruct-v0.1 (Sparse MoE, new family)

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --quantize 4bit \
  --exit-layers 29 30 \
  --debug-trace \
  --prompt-id P2 \
  --max-new-tokens 16 \
  --output artifacts/track_a_gpu/mixtral_8x7b_trace_p2.json

python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --quantize 4bit \
  --exit-layers 29 30 \
  --max-new-tokens 48 \
  --output artifacts/track_a_gpu/mixtral_8x7b_gpu_reproduction_n63.json

python scripts/track_a_gpu/summarize_benchmark.py \
  artifacts/track_a_gpu/mixtral_8x7b_gpu_reproduction_n63.json
```

### Mistral-7B-Instruct-v0.3 (Dense control)

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --quantize 4bit \
  --exit-layers 29 30 \
  --debug-trace \
  --prompt-id P2 \
  --max-new-tokens 16 \
  --output artifacts/track_a_gpu/mistral_7b_trace_p2.json

python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --quantize 4bit \
  --exit-layers 29 30 \
  --max-new-tokens 48 \
  --output artifacts/track_a_gpu/mistral_7b_gpu_reproduction_n63.json

python scripts/track_a_gpu/summarize_benchmark.py \
  artifacts/track_a_gpu/mistral_7b_gpu_reproduction_n63.json
```

### Other Dense Controls

- `meta-llama/Llama-3.1-8B-Instruct` should follow the same command pattern if you have access to the gated model.
- At the moment, Llama is still blocked by gated Hugging Face access. Do not describe it as empirically validated on this GPU Track A path until the checkpoint can actually be downloaded and run.
- Gemma 3 is a dense family, and the useful comparison axis here is checkpoint scale:
  - `google/gemma-3-4b-it` as a small dense control
  - `google/gemma-3-12b-it` as a mid-scale dense control
  - `google/gemma-3-27b-it` as a large dense control
- Mistral is still the cleaner first dense control because it avoids Gemma-family vocabulary and checkpoint-scaling confounds on the first pass.
- Current Gemma 3 checkpoint behavior is not uniform: `4B` is decent, `12B` is weak, and `27B` is strong. That is useful dense-family evidence, but it is still checkpoint-specific rather than a universal dense-family law.
- Older Gemma and Gemma2 instruction checkpoints are also supported by the explicit manual path, but they are lower-priority follow-up controls here.
- Do not point this script at multimodal Gemma 3 checkpoints. The manual-reference path only supports text causal-LM checkpoints.

### Gemma 3 4B-IT (Dense control, text only)

Use `--quantize none` here. The verified numerically sane path for `google/gemma-3-4b-it` is the non-BitsAndBytes load with explicit `--exit-layers 31 32`.

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model google/gemma-3-4b-it \
  --device cuda \
  --quantize none \
  --exit-layers 31 32 \
  --debug-trace \
  --prompt-id P2 \
  --max-new-tokens 32 \
  --output artifacts/track_a_gpu/gemma3_4b_trace_p2.json

python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model google/gemma-3-4b-it \
  --device cuda \
  --quantize none \
  --exit-layers 31 32 \
  --max-new-tokens 48 \
  --output artifacts/track_a_gpu/gemma3_4b_gpu_reproduction_n63.json

python scripts/track_a_gpu/summarize_benchmark.py \
  artifacts/track_a_gpu/gemma3_4b_gpu_reproduction_n63.json
```

### Gemma 3 12B-IT And 27B-IT (Dense controls, text only)

These are now empirically run on the GPU manual-reference path, but they should still be treated as separate checkpoints with different memory/runtime costs and different strength of evidence.

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model google/gemma-3-12b-it \
  --device cuda \
  --quantize none \
  --exit-layers 41 42 \
  --debug-trace \
  --prompt-id P2 \
  --max-new-tokens 32 \
  --output artifacts/track_a_gpu/gemma3_12b_trace_p2.json

python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model google/gemma-3-12b-it \
  --device cuda \
  --quantize none \
  --exit-layers 41 42 \
  --max-new-tokens 48 \
  --output artifacts/track_a_gpu/gemma3_12b_gpu_reproduction_n63.json

python scripts/track_a_gpu/summarize_benchmark.py \
  artifacts/track_a_gpu/gemma3_12b_gpu_reproduction_n63.json

python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model google/gemma-3-27b-it \
  --device cuda \
  --quantize none \
  --exit-layers 59 60 \
  --debug-trace \
  --prompt-id P2 \
  --max-new-tokens 32 \
  --output artifacts/track_a_gpu/gemma3_27b_trace_p2.json

python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model google/gemma-3-27b-it \
  --device cuda \
  --quantize none \
  --exit-layers 59 60 \
  --max-new-tokens 48 \
  --output artifacts/track_a_gpu/gemma3_27b_gpu_reproduction_n63.json

python scripts/track_a_gpu/summarize_benchmark.py \
  artifacts/track_a_gpu/gemma3_27b_gpu_reproduction_n63.json
```

## Download the Artifact

From your local machine:

```bash
scp -P <SSH_PORT> \
  root@<POD_IP>:/root/transcender-mlx/artifacts/track_a_gpu/qwen3_gpu_reproduction_n63.json \
  .
```

## Shutdown / Destroy

Do this immediately after downloading the artifact.

1. In RunPod, open the Pod page.
2. Click **Terminate**.
3. Do **not** leave the Pod in a stopped state.

Why terminate, not stop:

- on RunPod, stopped Pods can still incur persistent volume-disk charges
- this runbook deliberately uses **0 GB volume disk** so termination should leave no ongoing storage bill

## Billing Verification

After termination:

1. Confirm the Pod no longer appears as active.
2. Open RunPod billing/usage.
3. Confirm there is no active GPU charge.
4. Confirm you did not leave behind a volume disk or network volume.

## Methodological Scope

This script is deliberately narrow.

It does:

- full-depth shared-context decode
- explicit layer-by-layer hidden-state extraction at L45 and L46
- final-norm-plus-lm-head projection for intermediate candidates
- raw exit-token reporting
- separate `top1_agree` composition reporting

It does not do:

- MLX-style entropy-gated physical skipping
- serving throughput benchmarking
- vLLM or TRT-LLM
- quantization parity with MLX

So exact numerical parity with MLX is **not** expected. The first question is structural:

> Does the viable penultimate-layer / sharp one-layer-earlier cliff pattern survive off-MLX?

## Interpretation Templates

### If the frontier reproduces

> The off-MLX GPU reproduction preserves the same structural pattern seen on Apple MLX: a viable penultimate-layer operating point and a sharp quality cliff one layer earlier. This supports the interpretation that the frontier is at least partly model-level rather than purely an MLX runtime artifact.

### If it partially reproduces

> The GPU run preserves the penultimate-layer vs one-layer-earlier ordering, but the absolute values shift materially. The structure appears to transfer, while the operating-point quality remains runtime- and implementation-sensitive.

### If it does not reproduce

> The GPU run does not preserve the MLX frontier cleanly. This raises the possibility that the Track A frontier is partly contingent on runtime, quantization, or template details rather than reflecting a runtime-robust architectural property.
