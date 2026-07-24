# Week 12: Evaluation Frameworks

## Learning Objectives

By the end of this week, students should be able to:

- Evaluate a model with `lm-evaluation-harness` on standard benchmarks.
- Build a custom evaluation task for a specific domain.
- Understand zero-shot vs. few-shot prompting in evaluation.
- Interpret evaluation results and compare models fairly.

## Pre-Reading (Optional)

- Inference chapter of the course book (optional — not yet available in English)
- Evaluation & Benchmarks chapter of the course book (optional — not yet available in English)

## Lab Files

Run these in order:

1. [`lab/lm_eval_demo.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week12/lab/lm_eval_demo.py) — run a standard benchmark.
2. [`lab/custom_eval.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week12/lab/custom_eval.py) — build a custom evaluation.
3. [`lab/evalscope_demo.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week12/lab/evalscope_demo.py) — overview of evalscope (optional).

## Exercises

See [`exercises/README.md`](exercises.md).

## Deliverable

An evaluation report containing:

- Results on at least one standard benchmark.
- Results on a custom task.
- Comparison of two models (e.g., base vs. SFT, or SFT vs. GRPO).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Benchmark | Standardized dataset + metric for comparing models. |
| Zero-shot | Evaluate without task-specific examples in the prompt. |
| Few-shot | Include K examples in the prompt before asking. |
| Perplexity | Model's average prediction uncertainty on a corpus. |
| Accuracy / exact match | Fraction of outputs exactly matching the reference. |
| lm-evaluation-harness | EleutherAI's standard evaluation framework. |
| evalscope | ModelScope evaluation framework with visualization. |

## Common Pitfalls

- Comparing models evaluated with different prompt templates or shot counts.
- Using test sets for model selection.
- Reporting only aggregate metrics without per-task breakdown.
- Ignoring generation hyperparameters (temperature, top-p).

## Platform Note

`lm-evaluation-harness` and `evalscope` are not in the core dependencies because they are large and can pin older package versions. The lab scripts detect if they are installed and provide installation instructions. For the custom evaluation, only `transformers` is needed.
