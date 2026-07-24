# Week 3 Exercises

## Exercise 3.1: Verify RoPE is Relative

Create `exercises/rope_relative.py` that:

1. Creates two identical query vectors at positions `m` and `n`.
2. Creates two identical key vectors at positions `m` and `n`.
3. Computes `dot(q_m, k_m)` and `dot(q_n, k_n)` after RoPE.
4. Computes `dot(q_m, k_n)` and `dot(q_n, k_m)` after RoPE.

**Question**: Are the cross terms equal? Why does this show RoPE encodes relative position?

**Deliverable**: `exercises/rope_relative.py` + short answer.

## Exercise 3.2: Attention Pattern Visualization

Modify `lab/03_attention.py` to save or print the attention heatmap for a small input. Use `matplotlib` to visualize one head's attention matrix.

**Deliverable**: `exercises/visualize_attention.py` + saved PNG.

## Exercise 3.3: Compare LayerNorm and RMSNorm Speed

Write `exercises/norm_speed.py` that creates a large tensor and times `LayerNorm` vs `RMSNorm` over 1,000 forward passes. Report mean and standard deviation of the times.

**Question**: Is RMSNorm measurably faster? By how much?

**Deliverable**: `exercises/norm_speed.py` + timing table.

## Exercise 3.4: Implement GELU FFN

Add a `GELUFFN` class to `exercises/ffn_variants.py` alongside `StandardFFN` and `SwiGLU`. Compare parameter counts and forward-pass outputs on the same random input.

**Deliverable**: `exercises/ffn_variants.py`.

## Exercise 3.5: Multi-Head with RoPE

Combine `lab/02_rope.py` and `lab/03_attention.py` into `exercises/attention_with_rope.py`: a multi-head causal self-attention module that applies RoPE to Q and K before computing scores.

**Deliverable**: `exercises/attention_with_rope.py` + shape-check test.
