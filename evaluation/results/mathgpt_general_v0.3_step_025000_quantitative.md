# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-step-025000`
- Checkpoint: `mathgpt_step_025000.npz`
- Training step: `25,000`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 0.00%
- Median first-target rank: 1307.0
- Preference accuracy: 60.00%
- Mean preference margin: +0.326716

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 737 | 6.534134 | 688.24 |
| target_astronomy_01 | factual | ` Jupiter.` | 298 | 6.189902 | 487.80 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 799 | 5.249649 | 190.50 |
| target_animal_01 | semantic | ` mammal.` | 2671 | 7.156098 | 1281.90 |
| target_opposite_01 | semantic | ` cold.` | 1442 | 7.203808 | 1344.54 |
| target_math_01 | math | ` four.` | 1307 | 8.629217 | 5592.70 |
| target_algebra_01 | math | ` 7.` | 206 | 7.743312 | 2306.10 |
| target_language_01 | language | ` room.` | 1873 | 7.952809 | 2843.55 |
| target_direction_01 | factual | ` east.` | 5351 | 9.249369 | 10398.00 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.534134 | 8.450181 | +1.916048 | True |
| preference_astronomy_01 | factual | 6.189902 | 6.363578 | +0.173675 | True |
| preference_science_01 | factual | 5.249649 | 5.297010 | +0.047361 | True |
| preference_animal_01 | semantic | 7.156098 | 9.553815 | +2.397717 | True |
| preference_opposite_01 | semantic | 7.203808 | 6.928339 | -0.275469 | False |
| preference_math_01 | math | 8.629217 | 8.532886 | -0.096331 | False |
| preference_algebra_01 | math | 7.743312 | 5.730705 | -2.012607 | False |
| preference_grammar_01 | language | 6.427821 | 7.740750 | +1.312929 | True |
| preference_context_01 | context_tracking | 6.612218 | 5.774587 | -0.837631 | False |
| preference_language_01 | language | 7.952809 | 8.594275 | +0.641466 | True |