# Week 3: Transformer Architecture I — Building Blocks

## Learning Objectives

By the end of this week, students should be able to:

- Convert token IDs into embeddings and add positional information.
- Implement Rotary Positional Embeddings (RoPE) from scratch.
- Write causal (masked) multi-head self-attention.
- Understand the difference between LayerNorm and RMSNorm.
- Implement SwiGLU and compare it to standard feed-forward networks.

## Pre-Reading (Optional)

- `docs/en/chapter3/chapter3_pytorch与资源核算.md`
- `docs/en/chapter4/chapter4_第四章语言模型架构和训练的技术细节.md`

## Lab Files

Run these in order:

1. [`lab/01_embeddings.py`](./lab/01_embeddings.py) — token + positional embeddings.
2. [`lab/02_rope.py`](./lab/02_rope.py) — Rotary Positional Embeddings.
3. [`lab/03_attention.py`](./lab/03_attention.py) — causal self-attention.
4. [`lab/04_rmsnorm_swiglu.py`](./lab/04_rmsnorm_swiglu.py) — normalization and activation FFN.

## Exercises

See [`exercises/README.md`](./exercises/README.md).

## Deliverable

A short report (≤1 page) plus code showing:

- RoPE rotates query/key vectors correctly.
- Causal attention prevents attending to future tokens.
- SwiGLU has roughly 50% more parameters than a standard FFN of the same hidden dimension.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Embedding | Learned lookup table mapping token IDs to dense vectors. |
| Positional encoding | Injects sequence-position information. |
| RoPE | Encodes position by rotating query/key vectors in 2D subspaces. |
| Causal mask | Prevents positions from attending to future positions. |
| Multi-head attention | Multiple attention operations in parallel, each with its own projections. |
| RMSNorm | Normalizes by root-mean-square only; no mean-centering. |
| SwiGLU | Gated FFN activation: `Swish(xW) ⊗ (xV)`. |

## Common Pitfalls

- Confusing batch dimensions: `(batch, seq, d_model)` vs `(seq, batch, d_model)`.
- Forgetting the causal mask during inference or training.
- RoPE applied to values or outputs instead of queries/keys.
- Shape mismatches in multi-head reshaping.
