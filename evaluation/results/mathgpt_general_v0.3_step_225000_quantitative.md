# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-step-225000`
- Checkpoint: `mathgpt_step_225000.npz`
- Training step: `225,000`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 0.00%
- Median first-target rank: 2003.0
- Preference accuracy: 80.00%
- Mean preference margin: +0.744805

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 1479 | 6.625871 | 754.36 |
| target_astronomy_01 | factual | ` Jupiter.` | 1653 | 4.991809 | 147.20 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 2003 | 4.663052 | 105.96 |
| target_animal_01 | semantic | ` mammal.` | 9605 | 7.578440 | 1955.58 |
| target_opposite_01 | semantic | ` cold.` | 5025 | 7.539449 | 1880.79 |
| target_math_01 | math | ` four.` | 1283 | 8.288584 | 3978.20 |
| target_algebra_01 | math | ` 7.` | 152 | 8.217536 | 3705.36 |
| target_language_01 | language | ` room.` | 3974 | 8.134409 | 3409.80 |
| target_direction_01 | factual | ` east.` | 6300 | 9.251564 | 10420.86 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.625871 | 8.827997 | +2.202126 | True |
| preference_astronomy_01 | factual | 4.991809 | 6.083746 | +1.091937 | True |
| preference_science_01 | factual | 4.663052 | 4.592417 | -0.070635 | False |
| preference_animal_01 | semantic | 7.578440 | 11.445712 | +3.867272 | True |
| preference_opposite_01 | semantic | 7.539449 | 7.751536 | +0.212087 | True |
| preference_math_01 | math | 8.288584 | 8.309022 | +0.020439 | True |
| preference_algebra_01 | math | 8.217536 | 6.074603 | -2.142933 | False |
| preference_grammar_01 | language | 6.605392 | 7.871447 | +1.266055 | True |
| preference_context_01 | context_tracking | 6.104268 | 6.397831 | +0.293563 | True |
| preference_language_01 | language | 8.134409 | 8.842548 | +0.708139 | True |