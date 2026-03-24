# Track A GPU Reproduction Runbook

## Goal

Run the cheapest credible first off-MLX reproduction of the Track A penultimate-layer frontier.

This is an external-validity test, not a serving benchmark:

- target model: Qwen3-30B-A3B
- runtime: HuggingFace Transformers on CUDA
- comparison: full depth vs penultimate layer vs one layer earlier
- first step: 3-prompt smoke test
- second step: full N=63 run only if the smoke test is clean

## Measurement Fix

The earlier HF scaffold was not trustworthy enough to interpret frontier numbers.

Two specific problems:

- it relied on black-box hidden-state capture behavior instead of an explicit layer runner
- it reported `top1_agree` exact match against full depth even though the composed token falls back to the full-depth token on disagreement, which can make the metric look artificially perfect

The corrected script now uses a manual reference path:

- prefill once
- decode step-by-step
- run `embed_tokens -> decoder layers -> final norm -> lm_head` explicitly
- extract raw candidate logits for L45, L46, and full depth
- report raw exit tokens and composed tokens separately

Use the debug trace first. Do not trust aggregate benchmark output until the trace shows that L45/L46 candidate tokens truly differ from full depth on at least some steps.

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

git clone https://github.com/juniverse1103/transcender.git
cd transcender

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

## Smoke Test

This confirms:

- model load works
- the manual layer runner works
- L46 and L45 candidate projection paths run
- full-depth, raw L45/L46, and composed tokens are all emitted separately
- JSON artifact writing works

Command:

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model Qwen/Qwen3-30B-A3B \
  --quantize 4bit \
  --exit-layers 45 46 \
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
- output contains raw-exit and composed summaries for `L45` and `L46`

## Single-Prompt Debug Trace

Run this before trusting any benchmark aggregate:

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model Qwen/Qwen3-30B-A3B \
  --quantize 4bit \
  --debug-trace \
  --prompt-id P2 \
  --exit-layers 45 46 \
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

## Full Run

Run this only after the smoke test is clean:

```bash
python scripts/track_a_gpu/transcender_gpu_reproduction.py \
  --model Qwen/Qwen3-30B-A3B \
  --quantize 4bit \
  --exit-layers 45 46 \
  --max-new-tokens 48 \
  --output artifacts/track_a_gpu/qwen3_gpu_reproduction_n63.json
```

This uses the full fixed prompt suite in the script:

- P1 warmup
- 63 scored prompts

## Download the Artifact

From your local machine:

```bash
scp -P <SSH_PORT> \
  root@<POD_IP>:/root/transcender/artifacts/track_a_gpu/qwen3_gpu_reproduction_n63.json \
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
