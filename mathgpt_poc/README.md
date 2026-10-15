# MathGPT POC

A decoder-only GPT language model built from scratch using NumPy.

## Goal

1. Pretrain on general knowledge.
2. Save the base model.
3. Continue pretraining on mathematical data.
4. Compare the base and math-specialized models.

## POC Target

- 5–15M parameters
- NumPy-only neural network implementation
- Custom BPE tokenizer
- ~100M general tokens
- ~25–50M math tokens

## Hardware

- Intel Core i7-12700
- 32 GB RAM
- Ubuntu 24.04
- 2 TB storage