# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.2`
- Checkpoint: `mathgpt_step_050563.npz`
- Training step: `50,563`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 0.00%
- Median first-target rank: 417.0
- Preference accuracy: 50.00%
- Mean preference margin: +0.546827

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 268 | 5.549045 | 256.99 |
| target_astronomy_01 | factual | ` Jupiter.` | 819 | 5.897596 | 364.16 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 1219 | 4.637375 | 103.27 |
| target_animal_01 | semantic | ` mammal.` | 3685 | 7.172469 | 1303.06 |
| target_opposite_01 | semantic | ` cold.` | 330 | 5.924580 | 374.12 |
| target_math_01 | math | ` four.` | 192 | 7.208025 | 1350.22 |
| target_algebra_01 | math | ` 7.` | 192 | 7.384986 | 1611.60 |
| target_language_01 | language | ` room.` | 417 | 7.666732 | 2136.09 |
| target_direction_01 | factual | ` east.` | 817 | 8.221085 | 3718.53 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 5.549045 | 7.586629 | +2.037585 | True |
| preference_astronomy_01 | factual | 5.897596 | 6.474099 | +0.576503 | True |
| preference_science_01 | factual | 4.637375 | 4.149456 | -0.487920 | False |
| preference_animal_01 | semantic | 7.172469 | 10.671484 | +3.499015 | True |
| preference_opposite_01 | semantic | 5.924580 | 5.864910 | -0.059670 | False |
| preference_math_01 | math | 7.208025 | 6.969943 | -0.238082 | False |
| preference_algebra_01 | math | 7.384986 | 6.049339 | -1.335647 | False |
| preference_grammar_01 | language | 5.466809 | 6.758743 | +1.291935 | True |
| preference_context_01 | context_tracking | 6.329068 | 6.649163 | +0.320094 | True |
| preference_language_01 | language | 7.666732 | 7.531194 | -0.135538 | False |