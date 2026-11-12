# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-step-100000`
- Checkpoint: `mathgpt_step_100000.npz`
- Training step: `100,000`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 11.11%
- Median first-target rank: 1579.0
- Preference accuracy: 70.00%
- Mean preference margin: +0.752885

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 1240 | 6.549928 | 699.19 |
| target_astronomy_01 | factual | ` Jupiter.` | 1100 | 5.438831 | 230.17 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 3903 | 4.880326 | 131.67 |
| target_animal_01 | semantic | ` mammal.` | 13650 | 8.501783 | 4923.54 |
| target_opposite_01 | semantic | ` cold.` | 2083 | 6.892987 | 985.34 |
| target_math_01 | math | ` four.` | 1579 | 8.098020 | 3287.95 |
| target_algebra_01 | math | ` 7.` | 93 | 6.991036 | 1086.85 |
| target_language_01 | language | ` room.` | 1514 | 7.456003 | 1730.22 |
| target_direction_01 | factual | ` east.` | 2669 | 8.038852 | 3099.05 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.549928 | 8.802538 | +2.252610 | True |
| preference_astronomy_01 | factual | 5.438831 | 6.690769 | +1.251938 | True |
| preference_science_01 | factual | 4.880326 | 4.736002 | -0.144324 | False |
| preference_animal_01 | semantic | 8.501783 | 11.750100 | +3.248317 | True |
| preference_opposite_01 | semantic | 6.892987 | 6.677253 | -0.215733 | False |
| preference_math_01 | math | 8.098020 | 9.179046 | +1.081026 | True |
| preference_algebra_01 | math | 6.991036 | 5.013773 | -1.977262 | False |
| preference_grammar_01 | language | 6.647651 | 7.697396 | +1.049745 | True |
| preference_context_01 | context_tracking | 5.918304 | 6.391728 | +0.473424 | True |
| preference_language_01 | language | 7.456003 | 7.965112 | +0.509109 | True |