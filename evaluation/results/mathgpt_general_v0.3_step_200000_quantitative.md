# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-step-200000`
- Checkpoint: `mathgpt_step_200000.npz`
- Training step: `200,000`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 0.00%
- Median first-target rank: 2183.0
- Preference accuracy: 70.00%
- Mean preference margin: +0.657765

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 1405 | 6.906258 | 998.50 |
| target_astronomy_01 | factual | ` Jupiter.` | 1644 | 5.273564 | 195.11 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 2183 | 4.857085 | 128.65 |
| target_animal_01 | semantic | ` mammal.` | 11585 | 7.850487 | 2566.99 |
| target_opposite_01 | semantic | ` cold.` | 4562 | 7.742256 | 2303.66 |
| target_math_01 | math | ` four.` | 1098 | 8.292027 | 3991.92 |
| target_algebra_01 | math | ` 7.` | 229 | 8.374926 | 4336.95 |
| target_language_01 | language | ` room.` | 3217 | 7.920057 | 2751.93 |
| target_direction_01 | factual | ` east.` | 4825 | 8.589517 | 5375.02 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.906258 | 8.949607 | +2.043348 | True |
| preference_astronomy_01 | factual | 5.273564 | 6.154068 | +0.880504 | True |
| preference_science_01 | factual | 4.857085 | 4.655870 | -0.201215 | False |
| preference_animal_01 | semantic | 7.850487 | 11.445223 | +3.594735 | True |
| preference_opposite_01 | semantic | 7.742256 | 7.872159 | +0.129903 | True |
| preference_math_01 | math | 8.292027 | 8.093432 | -0.198595 | False |
| preference_algebra_01 | math | 8.374926 | 6.383942 | -1.990984 | False |
| preference_grammar_01 | language | 6.473863 | 7.726405 | +1.252542 | True |
| preference_context_01 | context_tracking | 5.861302 | 6.317040 | +0.455738 | True |
| preference_language_01 | language | 7.920057 | 8.531729 | +0.611672 | True |