# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.4-300M`
- Checkpoint: `mathgpt_step_292969.npz`
- Training step: `292,969`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 0.00%
- Median first-target rank: 1977.0
- Preference accuracy: 70.00%
- Mean preference margin: +0.520722

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 3602 | 6.899181 | 991.46 |
| target_astronomy_01 | factual | ` Jupiter.` | 4385 | 5.704555 | 300.23 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 1977 | 4.732037 | 113.53 |
| target_animal_01 | semantic | ` mammal.` | 9801 | 7.703833 | 2216.83 |
| target_opposite_01 | semantic | ` cold.` | 11511 | 7.311058 | 1496.76 |
| target_math_01 | math | ` four.` | 268 | 7.668313 | 2139.47 |
| target_algebra_01 | math | ` 7.` | 259 | 7.961486 | 2868.33 |
| target_language_01 | language | ` room.` | 264 | 7.392604 | 1623.93 |
| target_direction_01 | factual | ` east.` | 497 | 7.117096 | 1232.87 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.899181 | 9.428111 | +2.528930 | True |
| preference_astronomy_01 | factual | 5.704555 | 7.077366 | +1.372812 | True |
| preference_science_01 | factual | 4.732037 | 4.769701 | +0.037664 | True |
| preference_animal_01 | semantic | 7.703833 | 10.285429 | +2.581596 | True |
| preference_opposite_01 | semantic | 7.311058 | 7.983756 | +0.672698 | True |
| preference_math_01 | math | 7.668313 | 7.124109 | -0.544205 | False |
| preference_algebra_01 | math | 7.961486 | 6.094533 | -1.866952 | False |
| preference_grammar_01 | language | 6.311009 | 6.985464 | +0.674455 | True |
| preference_context_01 | context_tracking | 7.244745 | 7.355191 | +0.110446 | True |
| preference_language_01 | language | 7.392604 | 7.032377 | -0.360227 | False |