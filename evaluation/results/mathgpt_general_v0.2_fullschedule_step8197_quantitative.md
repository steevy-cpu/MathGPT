# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.2-FullSchedule-Step8197`
- Checkpoint: `mathgpt_step_008197.npz`
- Training step: `8,197`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 11.11%
- Median first-target rank: 844.0
- Preference accuracy: 40.00%
- Mean preference margin: +0.246688

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 477 | 5.760905 | 317.64 |
| target_astronomy_01 | factual | ` Jupiter.` | 741 | 7.381348 | 1605.75 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 1638 | 7.121526 | 1238.34 |
| target_animal_01 | semantic | ` mammal.` | 1735 | 7.506291 | 1819.45 |
| target_opposite_01 | semantic | ` cold.` | 844 | 6.352416 | 573.88 |
| target_math_01 | math | ` four.` | 260 | 6.670909 | 789.11 |
| target_algebra_01 | math | ` 7.` | 79 | 7.369007 | 1586.06 |
| target_language_01 | language | ` room.` | 892 | 8.414129 | 4510.34 |
| target_direction_01 | factual | ` east.` | 1449 | 8.688122 | 5932.03 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 5.760905 | 7.345576 | +1.584672 | True |
| preference_astronomy_01 | factual | 7.381348 | 6.709481 | -0.671867 | False |
| preference_science_01 | factual | 7.121526 | 6.879837 | -0.241689 | False |
| preference_animal_01 | semantic | 7.506291 | 9.179341 | +1.673050 | True |
| preference_opposite_01 | semantic | 6.352416 | 7.145178 | +0.792762 | True |
| preference_math_01 | math | 6.670909 | 6.493023 | -0.177886 | False |
| preference_algebra_01 | math | 7.369007 | 5.670237 | -1.698770 | False |
| preference_grammar_01 | language | 6.766253 | 8.044269 | +1.278017 | True |
| preference_context_01 | context_tracking | 6.983907 | 6.926467 | -0.057440 | False |
| preference_language_01 | language | 8.414129 | 8.400165 | -0.013964 | False |