# Week 8: Scaling Laws

## Learning Objectives

By the end of this week, students should be able to:

- Explain the Chinchilla scaling law and its parameters.
- Fit a scaling law to experimental data.
- Use scaling laws to predict the loss of a new model size or training budget.
- Design an IsoFLOPs experiment to find optimal model-data allocations.

## Pre-Reading (Optional)

- `docs/en/chapter9/chapter9_Scaling_Laws.md`
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla)

## Lab Files

Run these in order:

1. [`lab/scaling_law.py`](./lab/scaling_law.py) — fit Chinchilla law to data.
2. [`lab/isoflops.py`](./lab/isoflops.py) — simulate IsoFLOPs experiments.
3. [`lab/predict.py`](./lab/predict.py) — predict loss for a target config.

## Exercises

See [`exercises/README.md`](./exercises/README.md).

## Deliverable

A scaling-law report containing:

- Fitted Chinchilla parameters (E, A, B, α, β).
- Plot of actual vs. predicted loss.
- IsoFLOPs curve and optimal N/D for at least two FLOPs budgets.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Chinchilla law | `L(N, D) = E + A/N^α + B/D^β` |
| N | Number of non-embedding parameters. |
| D | Number of training tokens. |
| E | Irreducible entropy (bayes-optimal loss). |
| A, B, α, β | Fitted scaling coefficients. |
| IsoFLOPs curve | Models trained with the same FLOPs but different N and D. |
| Compute-optimal | The (N, D) pair that minimizes loss for a fixed FLOPs budget. |

## Common Pitfalls

- Using total parameters instead of non-embedding parameters.
- Fitting on models that are too small (capacity-constrained regime).
- Confusing training tokens with steps or epochs.
- Forgetting that scaling-law predictions assume similar architecture and data.

## Data Note

The lab scripts include a synthetic data generator for fast demonstration. For real use, replace the synthetic data with loss measurements from actual training runs (e.g., Week 4 models of different sizes).
