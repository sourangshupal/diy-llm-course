# Week 12 Exercises

## Exercise 12.1: Run a Standard Benchmark

Install `lm-eval` and run `lab/lm_eval_demo.py` on `gpt2` for at least one task. Report the score.

**Deliverable**: `exercises/lm_eval_report.md`.

## Exercise 12.2: Few-Shot vs. Zero-Shot

Run the same benchmark with `--num_fewshot 0` and `--num_fewshot 3`. Compare results.

**Deliverable**: `exercises/fewshot_comparison.md`.

## Exercise 12.3: Custom Domain Evaluation

Create a custom JSONL dataset of 20 questions in a domain of your choice. Use `lab/custom_eval.py` to evaluate `gpt2` and your Week 10/11 model.

**Deliverable**: `exercises/my_domain_eval.jsonl` + `exercises/custom_eval_report.md`.

## Exercise 12.4: Compare Two Models

Evaluate `gpt2` and your SFT or GRPO model on the same custom task. Create a table comparing accuracy and showing example outputs.

**Deliverable**: `exercises/model_comparison.md`.

## Exercise 12.5: Evaluation Design Principles

Write a one-page guide (`exercises/eval_design_guide.md`) on how to evaluate a new LLM fairly. Include: choice of benchmarks, prompt consistency, decoding settings, statistical significance, and reporting.
