# Week 10: Alignment — SFT & Expert Iteration

## Learning Objectives

By the end of this week, students should be able to:

- Explain the difference between pre-training, supervised fine-tuning (SFT), and iterative self-improvement.
- Fine-tune a small causal LM on instruction-following data.
- Implement expert iteration: generate reasoning traces, filter incorrect ones, and retrain.
- Evaluate SFT models on a simple task.

## Pre-Reading

- `docs/en/chapter13/chapter13_第十三章大模型的基本训练流程.md`

## Lab Files

Run these in order:

1. [`lab/sft_demo.py`](./lab/sft_demo.py) — fine-tune a small model on instruction data.
2. [`lab/expert_iteration.py`](./lab/expert_iteration.py) — generate, filter, and retrain.
3. [`lab/evaluate.py`](./lab/evaluate.py) — simple evaluation harness.

## Exercises

See [`exercises/README.md`](./exercises/README.md).

## Deliverable

A fine-tuned model checkpoint plus:

- Training loss curve.
- Sample outputs before and after SFT.
- Expert-iteration improvement curve (if completed).

## Key Concepts

| Concept | Description |
|---------|-------------|
| SFT | Supervised fine-tuning on (prompt, response) pairs. |
| Instruction tuning | SFT specifically on natural-language instructions. |
| Causal LM loss for SFT | Standard next-token prediction on the concatenated prompt + response. |
| Expert iteration | Generate candidate outputs, keep only correct/high-quality ones, retrain. |
| Data quality | SFT performance is often more data-limited than compute-limited. |

## Common Pitfalls

- Overfitting the small SFT dataset.
- Catastrophic forgetting of pre-training knowledge.
- Biased data producing biased outputs.
- Not separating prompt loss from response loss (though many implementations train on both).

## Platform Note

The demos use small models (e.g., `gpt2`) so they can run on CPU/MPS. For real alignment work, larger models and GPUs are needed, and you may install `trl` / `peft` separately.
