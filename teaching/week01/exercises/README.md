# Week 1 Exercises

## Exercise 1.1: Environment Self-Check

Run `lab/setup_check.py`. If any check fails, fix it and re-run until everything passes.

**Deliverable**: Terminal output showing all checks pass.

## Exercise 1.2: First W&B Run

Run `lab/wandb_demo.py`. Open the run URL and take a screenshot of:

1. The config panel.
2. The `train/loss` chart.
3. The artifacts tab showing `demo-config`.

**Deliverable**: Screenshot or run link.

## Exercise 1.3: Compare Two Runs

Copy `lab/wandb_demo.py` to `exercises/wandb_compare.py` and modify it so that it runs two experiments back-to-back:

- Run A: `learning_rate = 1e-3`
- Run B: `learning_rate = 1e-2`

Use different run names or tags to distinguish them. After running, open W&B and compare the loss curves.

**Questions**:

1. Which learning rate converges faster in this synthetic example?
2. What other hyperparameters would you want to log in a real training run?

**Deliverable**: `exercises/wandb_compare.py` + a short written answer (2–3 sentences).

## Exercise 1.4: Log a Custom Metric

Extend the demo to log a custom metric that is not loss. For example:

- Per-step gradient norm (simulate it with `rng.exponential()`).
- Per-step validation accuracy (simulate it increasing over time).
- A histogram of random values every 10 steps using `wandb.log({"dist": wandb.Histogram(values)})`.

**Deliverable**: `exercises/wandb_custom_metric.py` + link to the run.

## Exercise 1.5: Reproducibility Checklist

Write a short reproducibility checklist (Markdown or plain text) for your own future assignments. It should include at least:

- How to recreate the environment.
- How to set random seeds.
- What to log to W&B.
- How to save and name checkpoints.

**Deliverable**: `exercises/reproducibility_checklist.md`.
