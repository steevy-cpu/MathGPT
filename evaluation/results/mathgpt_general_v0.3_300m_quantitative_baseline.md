# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-300M`
- Checkpoint: `mathgpt_step_292969.npz`
- Training step: `292,969`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 0.00%
- Median first-target rank: 1951.0
- Preference accuracy: 70.00%
- Mean preference margin: +0.795061

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 1767 | 6.675114 | 792.44 |
| target_astronomy_01 | factual | ` Jupiter.` | 1241 | 4.815718 | 123.44 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 1971 | 4.474146 | 87.72 |
| target_animal_01 | semantic | ` mammal.` | 10877 | 7.890688 | 2672.28 |
| target_opposite_01 | semantic | ` cold.` | 6858 | 7.818651 | 2486.55 |
| target_math_01 | math | ` four.` | 1331 | 8.230839 | 3754.98 |
| target_algebra_01 | math | ` 7.` | 197 | 8.563177 | 5235.29 |
| target_language_01 | language | ` room.` | 1951 | 7.538451 | 1878.92 |
| target_direction_01 | factual | ` east.` | 3428 | 8.400220 | 4448.04 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.675114 | 9.111583 | +2.436469 | True |
| preference_astronomy_01 | factual | 4.815718 | 6.276999 | +1.461282 | True |
| preference_science_01 | factual | 4.474146 | 4.432107 | -0.042039 | False |
| preference_animal_01 | semantic | 7.890688 | 11.240138 | +3.349451 | True |
| preference_opposite_01 | semantic | 7.818651 | 8.006194 | +0.187543 | True |
| preference_math_01 | math | 8.230839 | 8.122186 | -0.108654 | False |
| preference_algebra_01 | math | 8.563177 | 6.439167 | -2.124009 | False |
| preference_grammar_01 | language | 6.592198 | 7.890383 | +1.298185 | True |
| preference_context_01 | context_tracking | 5.782942 | 6.443603 | +0.660661 | True |
| preference_language_01 | language | 7.538451 | 8.370169 | +0.831717 | True |