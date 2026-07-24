# Week 6 Exercises

## Exercise 6.1: Block Size Sweep

Modify `lab/01_matmul_triton.py` to test multiple `(BLOCK_M, BLOCK_N, BLOCK_K)` combinations. Find the fastest config for a 1024×1024 matmul on your GPU.

**Deliverable**: `exercises/matmul_block_sweep.py` + table of results.

## Exercise 6.2: Add Causal Masking to Matmul (Conceptual)

Explain in `exercises/causal_masking.md` how you would modify the Triton matmul kernel to compute only the lower-triangular part of a matrix product.

**Deliverable**: `exercises/causal_masking.md`.

## Exercise 6.3: FlashAttention Numerics

Run `lab/02_flash_attention.py --test` with FP16 inputs. Then run with FP32 accumulation. How much does the max difference vs. PyTorch attention change?

**Deliverable**: `exercises/fa_numerics.md` with numbers.

## Exercise 6.4: Roofline Revisited

Return to `week05/lab/roofline.py` and add a point for the tiled attention kernel from this week. Compare its arithmetic intensity to standard attention.

**Deliverable**: Updated roofline plot + 2-sentence explanation.

## Exercise 6.5: Read Real FlashAttention

Read the official Triton FlashAttention tutorial or the FlashAttention-2 paper. Write a one-page summary (`exercises/fa2_reading.md`) of the key optimizations beyond this lab's simplified version.

**Deliverable**: `exercises/fa2_reading.md`.

## Lab Files

- [`01_matmul_triton.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week06/lab/01_matmul_triton.py)
- [`02_flash_attention.py`](https://github.com/datawhalechina/diy-llm/blob/main/teaching/week06/lab/02_flash_attention.py)
