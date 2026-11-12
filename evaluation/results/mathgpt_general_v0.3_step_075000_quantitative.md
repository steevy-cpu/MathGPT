# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-step-075000`
- Checkpoint: `mathgpt_step_075000.npz`
- Training step: `75,000`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 11.11%
- Median first-target rank: 2640.0
- Preference accuracy: 80.00%
- Mean preference margin: +0.764900

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 918 | 6.056270 | 426.78 |
| target_astronomy_01 | factual | ` Jupiter.` | 2153 | 5.303957 | 201.13 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 3152 | 4.854870 | 128.36 |
| target_animal_01 | semantic | ` mammal.` | 11059 | 8.508677 | 4957.60 |
| target_opposite_01 | semantic | ` cold.` | 3873 | 6.736344 | 842.47 |
| target_math_01 | math | ` four.` | 404 | 7.132208 | 1251.64 |
| target_algebra_01 | math | ` 7.` | 73 | 6.898457 | 990.74 |
| target_language_01 | language | ` room.` | 2904 | 7.363087 | 1576.70 |
| target_direction_01 | factual | ` east.` | 2640 | 7.548264 | 1897.45 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.056270 | 8.507684 | +2.451414 | True |
| preference_astronomy_01 | factual | 5.303957 | 6.897260 | +1.593303 | True |
| preference_science_01 | factual | 4.854870 | 4.616761 | -0.238109 | False |
| preference_animal_01 | semantic | 8.508677 | 11.280045 | +2.771368 | True |
| preference_opposite_01 | semantic | 6.736344 | 7.253764 | +0.517420 | True |
| preference_math_01 | math | 7.132208 | 7.500964 | +0.368756 | True |
| preference_algebra_01 | math | 6.898457 | 5.140744 | -1.757713 | False |
| preference_grammar_01 | language | 6.434582 | 7.148590 | +0.714008 | True |
| preference_context_01 | context_tracking | 5.988632 | 6.723569 | +0.734937 | True |
| preference_language_01 | language | 7.363087 | 7.856705 | +0.493618 | True |