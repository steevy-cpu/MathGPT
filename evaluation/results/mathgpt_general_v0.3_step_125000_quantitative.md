# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-step-125000`
- Checkpoint: `mathgpt_step_125000.npz`
- Training step: `125,000`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 11.11%
- Median first-target rank: 1699.0
- Preference accuracy: 70.00%
- Mean preference margin: +0.740251

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 882 | 6.410145 | 607.98 |
| target_astronomy_01 | factual | ` Jupiter.` | 1504 | 5.035166 | 153.73 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 1699 | 4.586124 | 98.11 |
| target_animal_01 | semantic | ` mammal.` | 11237 | 8.350263 | 4231.29 |
| target_opposite_01 | semantic | ` cold.` | 4097 | 7.525944 | 1855.56 |
| target_math_01 | math | ` four.` | 1378 | 8.453162 | 4689.88 |
| target_algebra_01 | math | ` 7.` | 91 | 7.277050 | 1446.71 |
| target_language_01 | language | ` room.` | 2623 | 7.949651 | 2834.59 |
| target_direction_01 | factual | ` east.` | 4145 | 9.030621 | 8355.05 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.410145 | 8.567126 | +2.156982 | True |
| preference_astronomy_01 | factual | 5.035166 | 6.485495 | +1.450329 | True |
| preference_science_01 | factual | 4.586124 | 4.388375 | -0.197749 | False |
| preference_animal_01 | semantic | 8.350263 | 11.564255 | +3.213992 | True |
| preference_opposite_01 | semantic | 7.525944 | 7.452468 | -0.073476 | False |
| preference_math_01 | math | 8.453162 | 8.501605 | +0.048443 | True |
| preference_algebra_01 | math | 7.277050 | 5.952361 | -1.324689 | False |
| preference_grammar_01 | language | 6.821363 | 7.947416 | +1.126053 | True |
| preference_context_01 | context_tracking | 5.768625 | 6.224202 | +0.455577 | True |
| preference_language_01 | language | 7.949651 | 8.496704 | +0.547052 | True |