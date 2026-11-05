# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.1`
- Checkpoint: `mathgpt_step_008197.npz`
- Training step: `8,197`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 11.11%
- Median first-target rank: 1045.0
- Preference accuracy: 50.00%
- Mean preference margin: -0.044008

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 1334 | 6.618385 | 748.73 |
| target_astronomy_01 | factual | ` Jupiter.` | 1263 | 8.150676 | 3465.72 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 373 | 6.772397 | 873.40 |
| target_animal_01 | semantic | ` mammal.` | 1826 | 8.377012 | 4346.00 |
| target_opposite_01 | semantic | ` cold.` | 1541 | 6.923192 | 1015.56 |
| target_math_01 | math | ` four.` | 1045 | 8.514718 | 4987.64 |
| target_algebra_01 | math | ` 7.` | 42 | 6.994052 | 1090.13 |
| target_language_01 | language | ` room.` | 653 | 8.405274 | 4470.58 |
| target_direction_01 | factual | ` east.` | 255 | 7.571008 | 1941.10 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.618385 | 7.465622 | +0.847237 | True |
| preference_astronomy_01 | factual | 8.150676 | 6.921832 | -1.228845 | False |
| preference_science_01 | factual | 6.772397 | 6.885229 | +0.112833 | True |
| preference_animal_01 | semantic | 8.377012 | 8.282090 | -0.094922 | False |
| preference_opposite_01 | semantic | 6.923192 | 7.163639 | +0.240447 | True |
| preference_math_01 | math | 8.514718 | 8.073457 | -0.441261 | False |
| preference_algebra_01 | math | 6.994052 | 6.507279 | -0.486773 | False |
| preference_grammar_01 | language | 7.063570 | 8.188300 | +1.124730 | True |
| preference_context_01 | context_tracking | 7.400967 | 6.820355 | -0.580612 | False |
| preference_language_01 | language | 8.405274 | 8.472361 | +0.067087 | True |