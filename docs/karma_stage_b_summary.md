# Karma Stage B Summary

This note is the compact paper-facing summary for the offline Stage B work inside the Track A GPU diagnostic path.

Scope:

- offline only
- held-out split based
- diagnostic only, not a serving policy
- intended for paper drafting, appendix notes, and next-run hygiene

## Position In The Paper

- `Track A` remains the primary empirical result: the main same-model adaptive-depth frontier.
- `Track B` still matters as the scoped negative cascade baseline.
- `Track C` still matters as dense-model limitation and boundary-condition evidence.
- `GPU Track A + Stage B / karma` is additive. It strengthens the interpretation of penultimate acceptance, but it does not replace the Track A/B/C structure.

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

This should be read as one layer of the story, not the whole story. Stage B helps explain when penultimate acceptance is risky, but it does not eliminate the need for the Track B cascade baseline or the Track C dense-model boundary evidence.

## Cross-Model Paper Table

The table below was exported from the current pod artifacts under `/workspace/artifacts/track_a_gpu/*_karma_eval.json` using `scripts/track_a_gpu/summarize_karma_results.py --format markdown`. All rows below use the current default offline operating point with the full feature set and threshold `0.5`.

| model | karma accepted_precision | karma positive_recall | karma accepted_error_rate | penultimate_entropy accepted_precision | penultimate_entropy positive_recall | penultimate_entropy accepted_error_rate |
|---|---:|---:|---:|---:|---:|---:|
| `google/gemma-3-4b-it` | 0.859 | 0.938 | 0.097 | 0.645 | 0.923 | 0.320 |
| `google/gemma-3-12b-it` | 0.538 | 0.700 | 0.062 | 0.109 | 1.000 | 0.845 |
| `google/gemma-3-27b-it` | 0.920 | 0.976 | 0.061 | 0.726 | 0.988 | 0.270 |
| `openai/gpt-oss-20b` | 0.863 | 1.000 | 0.070 | 0.469 | 0.864 | 0.430 |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 0.828 | 0.964 | 0.110 | 0.560 | 0.936 | 0.405 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 0.861 | 0.961 | 0.094 | 0.694 | 0.903 | 0.240 |

## Current Takeaways

- The current claim is not that Stage B has a deployable threshold. The current claim is that Stage B appears to be a risk-estimation problem.
- Adjacent agreement is the main negative result. It is too weak to serve as the acceptance signal on its own.
- Penultimate entropy remains a live baseline, but only as a crude confidence proxy.
- At the current `0.5` operating point, `karma` improves accepted-error behavior versus entropy on all six exported models while keeping high recall on five of the six. The strongest regimes are `gemma-3-27b-it`, `gpt-oss-20b`, `gemma-3-4b-it`, `Mistral-7B`, and `Mixtral-8x7B`.
- `gemma-3-12b-it` remains the weak or pathological regime in the current set. `karma` is still much better than entropy there, but the acceptance rate is low and recall is only `0.700` at threshold `0.5`. The seed sweep is also less stable there than on the other checkpoints.
- This strengthens the interpretation of penultimate acceptance inside Track A, but it does not subsume the role of Track B or Track C in the paper.
- Threshold sweeps reinforce the same story: the useful checkpoints keep a workable precision-recall tradeoff over a range of thresholds, while `gemma-3-12b-it` stays visibly weaker and more threshold-sensitive.

## Limitations

- offline only
- held-out split based
- not an online serving policy
- threshold dependent
- still needs stronger validation across more splits, prompts, and deployment conditions
- the numeric table above comes from pod-local artifact exports rather than checked-in result JSONs in this repo snapshot
- seed sensitivity is non-trivial in the weakest regime, especially `gemma-3-12b-it`

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
  artifacts/track_a_gpu/*_karma_eval.json

python3 scripts/track_a_gpu/summarize_karma_results.py \
  --format csv \
  artifacts/track_a_gpu/*_karma_eval.json
```

Threshold and seed sweep summaries:

```bash
python3 scripts/track_a_gpu/summarize_karma_threshold_sweeps.py \
  artifacts/track_a_gpu/*_karma_th_*.json

python3 scripts/track_a_gpu/summarize_karma_seed_sweeps.py \
  artifacts/track_a_gpu/*_karma_seed_*.json
```

If threshold sweeps or seed sweeps are present, keep the paper table fixed to one explicit operating point and treat the sweep summaries as robustness notes rather than as a replacement for the main cross-model table.
