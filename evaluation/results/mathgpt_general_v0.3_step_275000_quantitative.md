# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-step-275000`
- Checkpoint: `mathgpt_step_275000.npz`
- Training step: `275,000`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 0.00%
- Median first-target rank: 2662.0
- Preference accuracy: 60.00%
- Mean preference margin: +0.758296

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 2375 | 6.707881 | 818.83 |
| target_astronomy_01 | factual | ` Jupiter.` | 1805 | 4.805746 | 122.21 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 2662 | 4.459133 | 86.41 |
| target_animal_01 | semantic | ` mammal.` | 10525 | 7.971551 | 2897.35 |
| target_opposite_01 | semantic | ` cold.` | 5550 | 7.853874 | 2575.69 |
| target_math_01 | math | ` four.` | 1833 | 7.983307 | 2931.61 |
| target_algebra_01 | math | ` 7.` | 204 | 8.442516 | 4640.22 |
| target_language_01 | language | ` room.` | 3135 | 7.744528 | 2308.90 |
| target_direction_01 | factual | ` east.` | 5126 | 8.876638 | 7162.67 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.707881 | 9.148309 | +2.440428 | True |
| preference_astronomy_01 | factual | 4.805746 | 6.388782 | +1.583036 | True |
| preference_science_01 | factual | 4.459133 | 4.388301 | -0.070832 | False |
| preference_animal_01 | semantic | 7.971551 | 11.320284 | +3.348733 | True |
| preference_opposite_01 | semantic | 7.853874 | 7.827626 | -0.026248 | False |
| preference_math_01 | math | 7.983307 | 7.763443 | -0.219864 | False |
| preference_algebra_01 | math | 8.442516 | 6.217694 | -2.224823 | False |
| preference_grammar_01 | language | 6.864790 | 8.076891 | +1.212101 | True |
| preference_context_01 | context_tracking | 5.571437 | 6.384918 | +0.813481 | True |
| preference_language_01 | language | 7.744528 | 8.471471 | +0.726943 | True |