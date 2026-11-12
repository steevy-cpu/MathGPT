# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-step-175000`
- Checkpoint: `mathgpt_step_175000.npz`
- Training step: `175,000`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 0.00%
- Median first-target rank: 2913.0
- Preference accuracy: 80.00%
- Mean preference margin: +0.707588

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 992 | 6.308110 | 549.01 |
| target_astronomy_01 | factual | ` Jupiter.` | 2526 | 5.309295 | 202.21 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 3296 | 4.485420 | 88.71 |
| target_animal_01 | semantic | ` mammal.` | 11557 | 8.170748 | 3535.99 |
| target_opposite_01 | semantic | ` cold.` | 3981 | 7.662815 | 2127.74 |
| target_math_01 | math | ` four.` | 589 | 7.655400 | 2112.02 |
| target_algebra_01 | math | ` 7.` | 180 | 8.458018 | 4712.71 |
| target_language_01 | language | ` room.` | 2913 | 7.462347 | 1741.23 |
| target_direction_01 | factual | ` east.` | 3216 | 8.206463 | 3664.56 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.308110 | 8.556261 | +2.248151 | True |
| preference_astronomy_01 | factual | 5.309295 | 6.292137 | +0.982842 | True |
| preference_science_01 | factual | 4.485420 | 4.630931 | +0.145511 | True |
| preference_animal_01 | semantic | 8.170748 | 11.726482 | +3.555735 | True |
| preference_opposite_01 | semantic | 7.662815 | 7.490487 | -0.172328 | False |
| preference_math_01 | math | 7.655400 | 8.105215 | +0.449815 | True |
| preference_algebra_01 | math | 8.458018 | 6.148547 | -2.309471 | False |
| preference_grammar_01 | language | 6.717652 | 8.085828 | +1.368176 | True |
| preference_context_01 | context_tracking | 5.877053 | 6.261517 | +0.384464 | True |
| preference_language_01 | language | 7.462347 | 7.885328 | +0.422981 | True |