# Week 5 Exercises

## Exercise 5.1: Profile Your Week 4 Model

Run `lab/profile_model.py` on the checkpoint you trained in Week 4. Identify the top 3 most expensive operations.

**Deliverable**: `exercises/profile_report.md` with screenshots or tables.

## Exercise 5.2: Roofline for Different GPUs

Modify `lab/roofline.py` to plot rooflines for three GPUs on the same chart (e.g., RTX 4090, A100 80GB, H100). Use publicly available peak compute and bandwidth numbers.

**Deliverable**: `exercises/multi_gpu_roofline.png` + brief comparison.

## Exercise 5.3: Memory Scaling Experiment

Run `lab/memory_analysis.py` while varying one variable at a time (`seq_len`, `batch`, `num_layers`). Plot how total activation memory scales with each.

**Deliverable**: `exercises/memory_scaling.py` + 3 plots.

## Exercise 5.4: Estimate Attention Arithmetic Intensity

Derive the arithmetic intensity of attention (QK^T + softmax + AV) yourself, assuming:

- Inputs Q, K, V are read once from HBM.
- Attention matrix is written and read once.
- Output is written once.

Compare your formula to the one in `lab/roofline.py`.

**Deliverable**: `exercises/attention_ai_derivation.md`.

## Exercise 5.5: Mixed Precision Impact

Research and write a short note on how BF16/FP16 training affects:

- Memory usage.
- Compute throughput.
- Numerical stability.

**Deliverable**: `exercises/mixed_precision.md`.
