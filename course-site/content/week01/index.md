# Week 1: Course Introduction & Experiment Tracking

## Learning Objectives

By the end of this week, students should be able to:

- Explain the high-level pipeline for building a modern LLM.
- Set up a reproducible Python environment with `uv`.
- Log metrics, hyperparameters, and artifacts to Weights & Biases (W&B).
- Interpret a simple W&B dashboard.

## Pre-Reading

- Preface chapter of the course book (English translation pending)
- W&B usage chapter of the course book (English translation pending)

## Lab Files

Run these in order:

1. [`lab/setup_check.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week01/lab/setup_check.py) — verify environment and GPU.
2. [`lab/wandb_demo.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week01/lab/wandb_demo.py) — log a toy experiment to W&B.

## Exercises

See [`exercises/README.md`](exercises.md).

## Deliverable

Submit a screenshot or link to a W&B run that logs:

- At least two hyperparameters.
- A scalar metric plotted over 100 steps.
- An artifact (e.g., a small text file or JSON config).

## Key Concepts

| Concept | Brief Description |
|---------|-------------------|
| LLM pipeline | data → tokenizer → model → training → alignment → evaluation |
| Reproducibility | pinned dependencies, seeded randomness, logged configs |
| Experiment tracking | centralized logs of metrics, hyperparams, code, and artifacts |
| W&B Run | one execution of an experiment; contains config, metrics, logs |
| W&B Project | a collection of runs that can be compared |

## Common Pitfalls

- Forgetting to call `wandb.finish()` can leave runs hanging.
- Logging too much too frequently (e.g., every token) slows training.
- Using different Python environments across machines breaks reproducibility.
