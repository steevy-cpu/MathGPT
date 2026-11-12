# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-step-250000`
- Checkpoint: `mathgpt_step_250000.npz`
- Training step: `250,000`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 0.00%
- Median first-target rank: 2391.0
- Preference accuracy: 80.00%
- Mean preference margin: +0.769520

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 1527 | 6.775264 | 875.91 |
| target_astronomy_01 | factual | ` Jupiter.` | 1079 | 5.065244 | 158.42 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 2391 | 4.494799 | 89.55 |
| target_animal_01 | semantic | ` mammal.` | 10238 | 7.816540 | 2481.31 |
| target_opposite_01 | semantic | ` cold.` | 6221 | 7.868939 | 2614.79 |
| target_math_01 | math | ` four.` | 1135 | 7.879321 | 2642.08 |
| target_algebra_01 | math | ` 7.` | 154 | 8.233381 | 3764.54 |
| target_language_01 | language | ` room.` | 2835 | 7.732300 | 2280.84 |
| target_direction_01 | factual | ` east.` | 4689 | 8.721591 | 6133.93 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.775264 | 9.326717 | +2.551453 | True |
| preference_astronomy_01 | factual | 5.065244 | 6.391813 | +1.326569 | True |
| preference_science_01 | factual | 4.494799 | 4.487900 | -0.006899 | False |
| preference_animal_01 | semantic | 7.816540 | 11.055927 | +3.239387 | True |
| preference_opposite_01 | semantic | 7.868939 | 7.945745 | +0.076807 | True |
| preference_math_01 | math | 7.879321 | 7.999277 | +0.119956 | True |
| preference_algebra_01 | math | 8.233381 | 6.013518 | -2.219864 | False |
| preference_grammar_01 | language | 6.566697 | 7.836177 | +1.269480 | True |
| preference_context_01 | context_tracking | 5.948257 | 6.412856 | +0.464599 | True |
| preference_language_01 | language | 7.732300 | 8.606010 | +0.873710 | True |