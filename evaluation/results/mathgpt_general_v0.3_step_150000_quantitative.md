# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-step-150000`
- Checkpoint: `mathgpt_step_150000.npz`
- Training step: `150,000`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 0.00%
- Median first-target rank: 1863.0
- Preference accuracy: 70.00%
- Mean preference margin: +0.694134

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 444 | 5.830387 | 340.49 |
| target_astronomy_01 | factual | ` Jupiter.` | 1312 | 4.672376 | 106.95 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 1863 | 4.696880 | 109.60 |
| target_animal_01 | semantic | ` mammal.` | 9100 | 7.782039 | 2397.16 |
| target_opposite_01 | semantic | ` cold.` | 2203 | 6.983733 | 1078.94 |
| target_math_01 | math | ` four.` | 549 | 7.561501 | 1922.73 |
| target_algebra_01 | math | ` 7.` | 196 | 7.774100 | 2378.20 |
| target_language_01 | language | ` room.` | 3069 | 8.333095 | 4159.27 |
| target_direction_01 | factual | ` east.` | 3589 | 8.588305 | 5368.51 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 5.830387 | 7.786165 | +1.955778 | True |
| preference_astronomy_01 | factual | 4.672376 | 6.392168 | +1.719792 | True |
| preference_science_01 | factual | 4.696880 | 4.607412 | -0.089468 | False |
| preference_animal_01 | semantic | 7.782039 | 11.438865 | +3.656825 | True |
| preference_opposite_01 | semantic | 6.983733 | 7.144389 | +0.160656 | True |
| preference_math_01 | math | 7.561501 | 7.831417 | +0.269916 | True |
| preference_algebra_01 | math | 7.774100 | 5.796712 | -1.977388 | False |
| preference_grammar_01 | language | 6.684294 | 7.702274 | +1.017980 | True |
| preference_context_01 | context_tracking | 5.711652 | 6.397780 | +0.686128 | True |
| preference_language_01 | language | 8.333095 | 7.874214 | -0.458880 | False |