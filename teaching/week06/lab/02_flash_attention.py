"""Week 6 Lab 2: Tiled attention kernel inspired by FlashAttention.

This kernel computes exact causal self-attention without materializing the
full N×N attention matrix. It is a simplified, teaching-oriented version.

Requires: Linux + CUDA + triton.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    N: tl.constexpr,
    d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute one row-block of causal attention output.

    Each program handles BLOCK_N query tokens. It iterates over key blocks,
    updates online softmax statistics, and writes the final output block.
    """
    pid = tl.program_id(0)

    # Offsets for this query block
    q_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, BLOCK_D)

    # Pointers
    q_ptrs = q_ptr + q_offs[:, None] * d + d_offs[None, :]
    k_ptrs = k_ptr + d_offs[:, None] * N  # will advance along N
    v_ptrs = v_ptr + d_offs[:, None] * N  # will advance along N

    # Load query block
    q = tl.load(q_ptrs, mask=(q_offs[:, None] < N) & (d_offs[None, :] < d), other=0.0)

    # Running softmax statistics and output accumulator
    m = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
    l = tl.full((BLOCK_N,), 0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, (pid + 1) * BLOCK_N, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)

        # Load key block (d, BLOCK_N)
        k = tl.load(
            k_ptrs + n_offs[None, :],
            mask=(d_offs[:, None] < d) & (n_offs[None, :] < N),
            other=0.0,
        )

        # Compute scores: (BLOCK_N, BLOCK_D) @ (BLOCK_D, BLOCK_N) -> (BLOCK_N, BLOCK_N)
        scores = tl.dot(q, k) / tl.sqrt(float(d))

        # Causal mask: queries only attend to keys <= query position
        causal_mask = n_offs[None, :] <= q_offs[:, None]
        scores = tl.where(causal_mask, scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)

        # Load value block (BLOCK_N, BLOCK_D)
        v = tl.load(
            v_ptrs + n_offs[:, None] * d,
            mask=(n_offs[:, None] < N) & (d_offs[None, :] < d),
            other=0.0,
        )

        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m = m_new

    # Normalize
    out = acc / l[:, None]

    # Store
    o_ptrs = o_ptr + q_offs[:, None] * d + d_offs[None, :]
    tl.store(o_ptrs, out, mask=(q_offs[:, None] < N) & (d_offs[None, :] < d))


def triton_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Launcher for the simplified FlashAttention kernel."""
    N, d = q.shape
    o = torch.empty_like(q)
    BLOCK_N = 32
    BLOCK_D = triton.next_power_of_2(d)
    grid = (triton.cdiv(N, BLOCK_N),)
    flash_attention_kernel[grid](
        q, k, v, o,
        N, d,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return o


def pytorch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Standard causal attention for comparison."""
    N, d = q.shape
    scores = torch.matmul(q, k.T) / math.sqrt(d)
    causal = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def benchmark() -> None:
    """Benchmark Triton attention vs PyTorch attention."""
    seq_lens = [128, 256, 512, 1024]
    d = 64
    print(f"{'Seq':>6} {'Triton (ms)':>12} {'Torch (ms)':>12} {'Max diff':>12}")
    for n in seq_lens:
        q = torch.randn(n, d, device="cuda", dtype=torch.float16)
        k = torch.randn(n, d, device="cuda", dtype=torch.float16)
        v = torch.randn(n, d, device="cuda", dtype=torch.float16)

        # Warmup
        _ = triton_flash_attention(q, k, v)
        _ = pytorch_attention(q, k, v)
        torch.cuda.synchronize()

        # Time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            o_tri = triton_flash_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        tri_ms = start.elapsed_time(end) / 10

        start.record()
        for _ in range(10):
            o_torch = pytorch_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 10

        max_diff = (o_tri - o_torch).abs().max().item()
        print(f"{n:>6} {tri_ms:>12.3f} {torch_ms:>12.3f} {max_diff:>12.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton FlashAttention-style benchmark")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This lab requires a Linux machine with an NVIDIA GPU.")
        return 0

    if args.test:
        q = torch.randn(64, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(64, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(64, 32, device="cuda", dtype=torch.float16)
        o_tri = triton_flash_attention(q, k, v)
        o_torch = pytorch_attention(q, k, v)
        print(f"Max diff: {(o_tri - o_torch).abs().max().item():.4f}")

    if args.benchmark:
        benchmark()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
