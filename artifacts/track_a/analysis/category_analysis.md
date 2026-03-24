# Track A Category-Level Analysis (N=63)

## GPT-OSS 20B

### Penultimate-Layer Exact Match by Category

| Category | Mean EM | Perfect / Total | Mean Prefix | Mean Layers Saved |
|----------|---------|-----------------|-------------|-------------------|
| Expository | 0.7760 | 8/12 | 34.5 | 0.4427 |
| Code/technical | 0.7646 | 7/10 | 36.4 | 0.4625 |
| Reasoning/logic | 0.8542 | 6/10 | 40.6 | 0.5000 |
| Creative/open-ended | 1.0000 | 10/10 | 48.0 | 0.3500 |
| Short-answer/factual | 0.8659 | 7/11 | 38.3 | 0.6121 |
| List/structured | 0.9771 | 9/10 | 46.9 | 0.5604 |

### Penultimate-1 Layer Mean EM by Category

| Category | Mean EM |
|----------|---------|
| Expository | 0.6094 |
| Code/technical | 0.6083 |
| Reasoning/logic | 0.6875 |
| Creative/open-ended | 0.7792 |
| Short-answer/factual | 0.7674 |
| List/structured | 0.7792 |

### Failures at Penultimate Layer (16 total)

| Prompt | Category | EM | Prefix | Div Pos | Class |
|--------|----------|----|--------|---------|-------|
| P18 | Expository | 0.7083 | 7 | 7 | early |
| P19 | Expository | 0.1458 | 7 | 7 | early |
| P22 | Expository | 0.2708 | 9 | 9 | early |
| P23 | Expository | 0.1875 | 7 | 7 | early |
| P26 | Code/technical | 0.2500 | 9 | 9 | early |
| P28 | Code/technical | 0.1458 | 7 | 7 | early |
| P29 | Code/technical | 0.2500 | 12 | 12 | mid |
| P8 | Reasoning/logic | 0.7917 | 38 | 38 | late |
| P34 | Reasoning/logic | 0.3750 | 18 | 18 | mid |
| P35 | Reasoning/logic | 0.5417 | 22 | 22 | mid |
| P37 | Reasoning/logic | 0.8333 | 40 | 40 | late |
| P12 | Short-answer/factual | 0.9211 | 35 | 35 | late |
| P53 | Short-answer/factual | 0.2292 | 11 | 11 | mid |
| P54 | Short-answer/factual | 0.4167 | 20 | 20 | mid |
| P56 | Short-answer/factual | 0.9583 | 23 | 23 | mid |
| P64 | List/structured | 0.7708 | 37 | 37 | late |

**Divergence summary:** Early (1-10): 6, Mid (11-24): 6, Late (25+): 4

**Failures by category:**

| Category | Failures | Early | Mid | Late |
|----------|----------|-------|-----|------|
| Expository | 4 | 4 | 0 | 0 |
| Code/technical | 3 | 2 | 1 | 0 |
| Reasoning/logic | 4 | 0 | 2 | 2 |
| Short-answer/factual | 4 | 0 | 3 | 1 |
| List/structured | 1 | 0 | 0 | 1 |

## Qwen3-30B-A3B

### Penultimate-Layer Exact Match by Category

| Category | Mean EM | Perfect / Total | Mean Prefix | Mean Layers Saved |
|----------|---------|-----------------|-------------|-------------------|
| Expository | 0.6562 | 4/12 | 29.3 | 0.7587 |
| Code/technical | 0.8562 | 7/10 | 41.0 | 0.8313 |
| Reasoning/logic | 0.9500 | 7/10 | 43.6 | 0.8146 |
| Creative/open-ended | 0.9083 | 7/10 | 43.4 | 0.7375 |
| Short-answer/factual | 0.7538 | 3/11 | 31.5 | 0.7083 |
| List/structured | 0.9437 | 8/10 | 45.2 | 0.7125 |

### Penultimate-1 Layer Mean EM by Category

| Category | Mean EM |
|----------|---------|
| Expository | 0.4358 |
| Code/technical | 0.4500 |
| Reasoning/logic | 0.4646 |
| Creative/open-ended | 0.5771 |
| Short-answer/factual | 0.3352 |
| List/structured | 0.5312 |

### Failures at Penultimate Layer (27 total)

| Prompt | Category | EM | Prefix | Div Pos | Class |
|--------|----------|----|--------|---------|-------|
| P3 | Expository | 0.9167 | 44 | 44 | late |
| P5 | Expository | 0.1250 | 6 | 6 | early |
| P17 | Expository | 0.1250 | 6 | 6 | early |
| P18 | Expository | 0.9167 | 44 | 44 | late |
| P19 | Expository | 0.4792 | 14 | 14 | mid |
| P20 | Expository | 0.6250 | 30 | 30 | late |
| P21 | Expository | 0.4792 | 6 | 6 | early |
| P22 | Expository | 0.2083 | 10 | 10 | early |
| P7 | Code/technical | 0.5625 | 27 | 27 | late |
| P29 | Code/technical | 0.4792 | 22 | 22 | mid |
| P30 | Code/technical | 0.5208 | 25 | 25 | late |
| P33 | Reasoning/logic | 0.7708 | 37 | 37 | late |
| P35 | Reasoning/logic | 0.9792 | 27 | 27 | late |
| P36 | Reasoning/logic | 0.7500 | 36 | 36 | late |
| P43 | Creative/open-ended | 0.5000 | 24 | 24 | mid |
| P44 | Creative/open-ended | 0.7708 | 35 | 35 | late |
| P47 | Creative/open-ended | 0.8125 | 39 | 39 | late |
| P12 | Short-answer/factual | 0.4167 | 20 | 20 | mid |
| P50 | Short-answer/factual | 0.8750 | 32 | 32 | late |
| P51 | Short-answer/factual | 0.1875 | 8 | 8 | early |
| P52 | Short-answer/factual | 0.9583 | 46 | 46 | late |
| P53 | Short-answer/factual | 0.9375 | 45 | 45 | late |
| P54 | Short-answer/factual | 0.4583 | 22 | 22 | mid |
| P55 | Short-answer/factual | 0.9792 | 8 | 8 | early |
| P56 | Short-answer/factual | 0.4792 | 22 | 22 | mid |
| P57 | List/structured | 0.6042 | 28 | 28 | late |
| P63 | List/structured | 0.8333 | 40 | 40 | late |

**Divergence summary:** Early (1-10): 6, Mid (11-24): 6, Late (25+): 15

**Failures by category:**

| Category | Failures | Early | Mid | Late |
|----------|----------|-------|-----|------|
| Expository | 8 | 4 | 1 | 3 |
| Code/technical | 3 | 0 | 1 | 2 |
| Reasoning/logic | 3 | 0 | 0 | 3 |
| Creative/open-ended | 3 | 0 | 1 | 2 |
| Short-answer/factual | 8 | 2 | 3 | 3 |
| List/structured | 2 | 0 | 0 | 2 |
