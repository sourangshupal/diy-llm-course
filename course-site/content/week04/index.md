# Week 4: Transformer Architecture II — Full Model & Training Loop

## Learning Objectives

By the end of this week, students should be able to:

- Assemble embeddings, attention, normalization, and FFN into a decoder-only Transformer.
- Implement a training loop with cross-entropy loss, AdamW, cosine LR schedule, and gradient clipping.
- Generate text from a trained model using temperature and top-k/top-p sampling.
- Save and load checkpoints correctly.

## Pre-Reading (Optional)

- Language Model Architecture & Training chapter of the course book (optional — not yet available in English)
- Assignment 1 README: `coursework/assignment1-basics/README.md`

## Lab Files

Run these in order:

1. [`lab/model.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week04/lab/model.py) — full decoder-only Transformer.
2. [`lab/data.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week04/lab/data.py) — tiny dataset utilities.
3. [`lab/train.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week04/lab/train.py) — training loop.
4. [`lab/generate.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week04/lab/generate.py) — text generation.

## Exercises

See [`exercises/README.md`](exercises.md).

## Deliverable

A trained mini-language model plus:

- Training loss curve (from W&B or saved PNG).
- Generated text sample.
- Checkpoint file in `week04/outputs/`.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Decoder-only Transformer | Stacks of causal self-attention + FFN blocks; used by GPT, Llama, Qwen. |
| Pre-norm | Apply normalization before attention/FFN; more stable for deep models. |
| AdamW | Adam with decoupled weight decay. |
| Cosine LR schedule | Warm up then decay learning rate following half a cosine cycle. |
| Gradient clipping | Cap gradient norm to prevent spikes. |
| Teacher forcing | Use ground-truth tokens as inputs and predict the next token. |
| Top-k / top-p sampling | Constrain the output distribution during generation. |

## Common Pitfalls

- Loading a checkpoint into a model with mismatched config.
- Forgetting to call `model.eval()` during generation (affects dropout).
- Not handling the padding token correctly in loss masking.
- Saving optimizer state when you only need model weights.
