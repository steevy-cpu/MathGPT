# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.2-Matched8197`
- Checkpoint: `mathgpt_step_008197.npz`
- Training step: `8,197`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 11.11%
- Median first-target rank: 629.0
- Preference accuracy: 40.00%
- Mean preference margin: -0.185466

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 396 | 5.640048 | 281.48 |
| target_astronomy_01 | factual | ` Jupiter.` | 823 | 7.583111 | 1964.73 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 858 | 7.106960 | 1220.43 |
| target_animal_01 | semantic | ` mammal.` | 629 | 8.204783 | 3658.41 |
| target_opposite_01 | semantic | ` cold.` | 771 | 6.683412 | 799.04 |
| target_math_01 | math | ` four.` | 1355 | 8.796672 | 6612.20 |
| target_algebra_01 | math | ` 7.` | 99 | 6.998479 | 1094.97 |
| target_language_01 | language | ` room.` | 243 | 7.386997 | 1614.85 |
| target_direction_01 | factual | ` east.` | 293 | 7.522178 | 1848.59 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 5.640048 | 6.852869 | +1.212821 | True |
| preference_astronomy_01 | factual | 7.583111 | 6.671075 | -0.912036 | False |
| preference_science_01 | factual | 7.106960 | 6.903076 | -0.203883 | False |
| preference_animal_01 | semantic | 8.204783 | 8.035453 | -0.169330 | False |
| preference_opposite_01 | semantic | 6.683412 | 6.971405 | +0.287993 | True |
| preference_math_01 | math | 8.796672 | 8.014436 | -0.782236 | False |
| preference_algebra_01 | math | 6.998479 | 5.604116 | -1.394363 | False |
| preference_grammar_01 | language | 7.388265 | 7.753302 | +0.365037 | True |
| preference_context_01 | context_tracking | 7.274483 | 6.587483 | -0.687000 | False |
| preference_language_01 | language | 7.386997 | 7.815337 | +0.428340 | True |