# Week 6: High-Performance Kernels with Triton

## Learning Objectives

By the end of this week, students should be able to:

- Explain why fused kernels beat PyTorch operator sequences for memory-bound operations.
- Write a simple Triton matmul kernel and benchmark it against `torch.matmul`.
- Understand the tiling and online-softmax ideas behind FlashAttention.
- Implement a tiled attention kernel in Triton.

## Pre-Reading

- `docs/en/chapter7/chapter7_第七章GPU高性能编程.md`
- Optional: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

## Lab Files

Run these in order (requires Linux + CUDA + Triton):

1. [`lab/01_matmul_triton.py`](./lab/01_matmul_triton.py) — fused matmul kernel.
2. [`lab/02_flash_attention.py`](./lab/02_flash_attention.py) — tiled attention kernel.

## Exercises

See [`exercises/README.md`](./exercises/README.md).

## Deliverable

A benchmark report showing:

- Triton matmul vs. `torch.matmul` across matrix sizes.
- Triton attention vs. PyTorch attention across sequence lengths.
- A short explanation of when the Triton version wins and why.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Triton | Python DSL from OpenAI for writing GPU kernels. |
| Tiling | Splitting matrices into blocks that fit in SRAM. |
| Fused kernel | A single kernel that replaces multiple PyTorch ops, reducing HBM traffic. |
| Online softmax | Computing softmax incrementally over tiles to avoid materializing the full attention matrix. |
| FlashAttention | Exact attention algorithm with O(N) memory instead of O(N²). |

## Common Pitfalls

- Triton is not available on Windows/macOS; use Linux with NVIDIA GPU.
- Block sizes must divide dimensions cleanly or require masking.
- Autotune configs can make first run slow due to compilation.
- Numerical differences between Triton and PyTorch can arise from accumulation order.

## Platform Note

These labs require `triton` and a CUDA GPU. On macOS or CPU-only machines, read the code and run the benchmark harness in "reference mode" using PyTorch to understand the logic.
