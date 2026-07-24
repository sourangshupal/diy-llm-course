# Week 5: GPU Architecture & Optimization

## Learning Objectives

By the end of this week, students should be able to:

- Describe the GPU execution model (SMs, warps, memory hierarchy).
- Profile a PyTorch model and identify bottlenecks.
- Estimate arithmetic intensity and use the roofline model.
- Explain why memory bandwidth often limits Transformer training.

## Pre-Reading (Optional)

- `docs/en/chapter5/chapter5_混合专家模型.md` (MoE overview)
- `docs/en/chapter6/chapter6_第六章GPU和GPU相关的优化.md`

## Lab Files

Run these in order:

1. [`lab/profile_model.py`](./lab/profile_model.py) — profile the Week 4 model with PyTorch profiler.
2. [`lab/roofline.py`](./lab/roofline.py) — plot a roofline model for your GPU.
3. [`lab/memory_analysis.py`](./lab/memory_analysis.py) — estimate activation memory.

## Exercises

See [`exercises/README.md`](./exercises/README.md).

## Deliverable

A profiling report containing:

- Top 5 operations by GPU time.
- Estimated arithmetic intensity of the attention layer.
- Roofline plot with the attention layer marked.

## Key Concepts

| Concept | Description |
|---------|-------------|
| SM (Streaming Multiprocessor) | GPU compute unit that executes warps. |
| Warp | Group of 32 threads that execute in lockstep. |
| HBM | High-bandwidth memory; large but slow compared to on-chip caches. |
| Shared memory | Fast, programmable on-chip memory per SM block. |
| Roofline model | Visualizes peak performance vs. arithmetic intensity. |
| Arithmetic intensity | FLOPs per byte of memory traffic. |
| Memory-bound vs compute-bound | Whether performance is limited by bandwidth or FLOPs. |

## Common Pitfalls

- Profiling on CPU gives no useful GPU timing.
- Not synchronizing CUDA before measuring time.
- Confusing throughput (TFLOP/s) with total time.
- Ignoring memory bandwidth when scaling model size.

## Platform Note

Profiling and roofline plots require a CUDA GPU. The code will run on CPU/MPS for syntax checks but meaningful numbers need an NVIDIA GPU.
