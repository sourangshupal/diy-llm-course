# Week 8 Exercises

## Exercise 8.1: Fit on Real Data

Train at least 3 models from Week 4 with different sizes (e.g., 100k, 500k, 2M parameters) for the same number of tokens. Record final loss, N, and D. Fit the scaling law with `lab/scaling_law.py`.

**Deliverable**: `exercises/real_scaling_data.json` + fitted params + plot.

## Exercise 8.2: Compare Fitting Objectives

Modify `lab/scaling_law.py` to fit in linear-space (MSE on raw loss) instead of log-space. Compare fitted parameters and prediction quality.

**Deliverable**: `exercises/scaling_linear_vs_log.md`.

## Exercise 8.3: Multi-Budget IsoFLOPs

Run `lab/isoflops.py` for three FLOPs budgets (e.g., 1e17, 1e18, 1e19). Plot optimal N and D vs. budget on log-log axes.

**Deliverable**: `exercises/multi_budget_isoflops.png` + interpretation.

## Exercise 8.4: Chinchilla Check

Using your fitted law, compute the optimal tokens-per-parameter ratio for a range of model sizes. Is it close to 20? What does that imply?

**Deliverable**: `exercises/chinchilla_check.py` + short writeup.

## Exercise 8.5: Capacity-Constrained Regime

Add a tiny model (e.g., 10k parameters) to your dataset. Does the scaling law still fit? Why or why not?

**Deliverable**: `exercises/capacity_regime.md`.
