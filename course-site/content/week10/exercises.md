# Week 10 Exercises

## Exercise 10.1: SFT on Real Data

Replace the toy dataset in `sft_demo.py` with a real instruction dataset (e.g., a subset of `yahma/alpaca-cleaned` or `databricks-dolly-15k`). Fine-tune `gpt2` and inspect outputs.

**Deliverable**: `exercises/sft_real_data.py` + before/after samples.

## Exercise 10.2: Prompt-Only Loss

Modify `sft_demo.py` to compute loss only on the response tokens, masking out the prompt tokens in the labels. Compare to training on the full sequence.

**Deliverable**: `exercises/sft_prompt_masking.py` + short comparison.

## Exercise 10.3: Expert Iteration Verifier

Design a verifier for a non-math task (e.g., JSON formatting, code syntax). Adapt `expert_iteration.py` to generate, verify, and retrain.

**Deliverable**: `exercises/ei_custom_verifier.py`.

## Exercise 10.4: Overfitting Check

Split your SFT data into train/validation. Plot train and validation loss. At what epoch does overfitting begin?

**Deliverable**: `exercises/sft_overfitting.py` + plot.

## Exercise 10.5: LoRA (optional)

Install `peft` with `uv pip install peft` and rewrite `sft_demo.py` to use LoRA. Compare trainable parameters and output quality.

**Deliverable**: `exercises/sft_lora.py`.

## Lab Files

- [`evaluate.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week10/lab/evaluate.py)
- [`expert_iteration.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week10/lab/expert_iteration.py)
- [`sft_demo.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week10/lab/sft_demo.py)
