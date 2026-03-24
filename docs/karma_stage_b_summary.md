# Karma Stage B Summary

This note is the compact paper-facing summary for the offline Stage B work inside the Track A GPU diagnostic path.

Scope:

- offline only
- held-out split based
- diagnostic only, not a serving policy
- intended for paper drafting, appendix notes, and next-run hygiene

## Stage B Problem

The token-row export induces three oracle labels:

- `earlier_correct`: the earlier layer already matches full depth
- `need_penultimate`: the earlier layer is wrong, but the penultimate layer matches full depth
- `need_full_depth`: both earlier and penultimate are wrong relative to full depth

`Stage A` asks whether the earlier layer was already enough. `Stage B` starts only after Stage A has already failed, and asks whether the penultimate layer is still safe to accept or whether full depth is required.

Why Stage B exists:

- the Track A frontier is a penultimate-layer story, not a generic shallow-layer story
- once the earlier layer has already missed, the remaining decision is specifically penultimate versus full depth
- this is a correction-versus-shared-failure problem, not just an agreement problem

Why adjacent agreement fails:

- adjacent-layer agreement only tells us that the earlier and penultimate layers look similar to each other
- it does not tell us whether they are jointly wrong in a way that the final layer would still correct
- the current cross-family conclusion is negative: naive adjacent agreement is not a reliable Stage B acceptance rule

## Interpreting `karma`

The current offline interpretation is:

- `karma = probability_of_need_full_depth`
- lower `karma` is safer
- `karma` should be read as a risk-style score, not just as confidence

This matters because confidence and risk are not the same here. A penultimate token can have low entropy or high margin and still be in a regime where the final layer would correct it. The working interpretation is therefore: Stage B is better framed as estimating the risk that the penultimate layer is still unsafe than as thresholding a generic confidence scalar.

## Cross-Model Paper Table

At edit time, the checked-in repo snapshot did not include the model-level Stage B JSON outputs under `artifacts/track_a_gpu/`, and `/workspace/artifacts/track_a_gpu/` was also absent. The table below therefore preserves the intended paper schema without inventing numbers.

| model | karma accepted_precision | karma positive_recall | karma accepted_error_rate | penultimate_entropy accepted_precision | penultimate_entropy positive_recall | penultimate_entropy accepted_error_rate | status |
|---|---:|---:|---:|---:|---:|---:|---|
| `google/gemma-3-4b-it` | pending | pending | pending | pending | pending | pending | model is in the current Stage B evaluation set, but the result JSON is not present in this repo snapshot |
| `google/gemma-3-12b-it` | pending | pending | pending | pending | pending | pending | model is in the current Stage B evaluation set, but the result JSON is not present in this repo snapshot |
| `google/gemma-3-27b-it` | pending | pending | pending | pending | pending | pending | model is in the current Stage B evaluation set, but the result JSON is not present in this repo snapshot |
| `openai/gpt-oss-20b` | pending | pending | pending | pending | pending | pending | model is in the current Stage B evaluation set, but the result JSON is not present in this repo snapshot |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | pending | pending | pending | pending | pending | pending | model is in the current Stage B evaluation set, but the result JSON is not present in this repo snapshot |
| `mistralai/Mistral-7B-Instruct-v0.3` | pending | pending | pending | pending | pending | pending | model is in the current Stage B evaluation set, but the result JSON is not present in this repo snapshot |

## Current Takeaways

- The current claim is not that Stage B has a deployable threshold. The current claim is that Stage B appears to be a risk-estimation problem.
- Adjacent agreement is the main negative result. It is too weak to serve as the acceptance signal on its own.
- Penultimate entropy remains a live baseline, but only as a crude confidence proxy.
- The current internal interpretation is that `karma` materially improves accepted-error behavior at useful recall on multiple model families. That claim should be backed by the exported cross-model table once the model-level JSONs are copied into the repo.
- Gemma 3 `12B` is already the weakest checkpoint in the checked-in raw penultimate benchmark notes. It should be treated as the first likely weak or pathological regime when the Stage B paper table is populated. Do not generalize that checkpoint-specific weakness into a universal dense-model statement.

## Limitations

- offline only
- held-out split based
- not an online serving policy
- threshold dependent
- still needs stronger validation across more splits, prompts, and deployment conditions
- the current checked-in repo snapshot lacks the model-level Stage B summary JSONs needed for a numeric appendix table

## Repro / Export

Generate one JSON per model from token rows:

```bash
python3 scripts/track_a_gpu/evaluate_relation_proxies.py \
  artifacts/track_a_gpu/<model_token_rows>.jsonl \
  --fit-karma-logistic \
  --as-json \
  > artifacts/track_a_gpu/<model>_stage_b_karma.json
```

Export the compact paper table once those JSONs exist:

```bash
python3 scripts/track_a_gpu/summarize_karma_results.py \
  --format markdown \
  artifacts/track_a_gpu/*stage_b_karma*.json

python3 scripts/track_a_gpu/summarize_karma_results.py \
  --format csv \
  artifacts/track_a_gpu/*stage_b_karma*.json
```

If threshold sweeps or seed sweeps are present, select the one JSON per model that you actually want to cite in the paper before exporting the final table.
