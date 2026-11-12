# MathGPT Quantitative Benchmark

- Model: `MathGPT-General-v0.3-step-050000`
- Checkpoint: `mathgpt_step_050000.npz`
- Training step: `50,000`
- Parameters: `8,110,848`
- Benchmark version: `0.1`

## Aggregate Results

- First-target Top-1 accuracy: 0.00%
- First-target Top-10 accuracy: 0.00%
- First-target Top-100 accuracy: 0.00%
- Median first-target rank: 1918.0
- Preference accuracy: 80.00%
- Mean preference margin: +0.629033

## Target Continuations

| Test | Category | Target | First Rank | Mean NLL | PPL |
|---|---|---|---:|---:|---:|
| target_geography_01 | factual | ` Paris.` | 1326 | 6.424398 | 616.71 |
| target_astronomy_01 | factual | ` Jupiter.` | 668 | 5.581844 | 265.56 |
| target_science_01 | factual | ` 0 degrees Celsius.` | 2048 | 4.798063 | 121.28 |
| target_animal_01 | semantic | ` mammal.` | 6343 | 7.730539 | 2276.83 |
| target_opposite_01 | semantic | ` cold.` | 1918 | 6.743805 | 848.78 |
| target_math_01 | math | ` four.` | 541 | 7.103298 | 1215.97 |
| target_algebra_01 | math | ` 7.` | 103 | 7.012738 | 1110.69 |
| target_language_01 | language | ` room.` | 4785 | 8.011463 | 3015.33 |
| target_direction_01 | factual | ` east.` | 4447 | 8.214709 | 3694.90 |

## Contrastive Preferences

| Test | Category | Preferred NLL | Rejected NLL | Margin | Correct |
|---|---|---:|---:|---:|---|
| preference_geography_01 | factual | 6.424398 | 8.731050 | +2.306652 | True |
| preference_astronomy_01 | factual | 5.581844 | 6.411607 | +0.829763 | True |
| preference_science_01 | factual | 4.798063 | 4.803977 | +0.005914 | True |
| preference_animal_01 | semantic | 7.730539 | 10.095928 | +2.365390 | True |
| preference_opposite_01 | semantic | 6.743805 | 6.603262 | -0.140543 | False |
| preference_math_01 | math | 7.103298 | 7.199206 | +0.095907 | True |
| preference_algebra_01 | math | 7.012738 | 5.052040 | -1.960698 | False |
| preference_grammar_01 | language | 5.949269 | 7.738945 | +1.789677 | True |
| preference_context_01 | context_tracking | 5.213512 | 5.689454 | +0.475942 | True |
| preference_language_01 | language | 8.011463 | 8.533788 | +0.522325 | True |