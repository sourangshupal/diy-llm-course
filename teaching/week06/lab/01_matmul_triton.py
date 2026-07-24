"""Week 6 Lab 1: Triton matrix-multiplication kernel.

This is a teaching-friendly block matmul. It is not as optimized as the
official Triton matmul tutorial, but it demonstrates the core ideas:
  - tiling,
  - shared-memory reuse,
  - grid of programs.

Requires: Linux + CUDA + triton.
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute C = A @ B where A is (M, K) and B is (K, N)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the start of blocks
    a_block_ptr = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    b_block_ptr = b_ptr + (offs_k[:, None] * N + offs_n[None, :])

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load tiles
        a = tl.load(a_block_ptr, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_block_ptr, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_block_ptr += BLOCK_K
        b_block_ptr += BLOCK_K * N

    c = accumulator.to(tl.float16)

    # Write output tile
    c_block_ptr = c_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_block_ptr, c, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Launcher for the Triton matmul kernel."""
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Choose block sizes
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def benchmark() -> None:
    """Benchmark Triton matmul against torch.matmul."""
    sizes = [256, 512, 1024, 2048]
    print(f"{'Size':>6} {'Triton (ms)':>12} {'Torch (ms)':>12} {'Speedup':>10}")
    for n in sizes:
        a = torch.randn(n, n, device="cuda", dtype=torch.float16)
        b = torch.randn(n, n, device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(3):
            _ = triton_matmul(a, b)
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Triton timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            _ = triton_matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / 10

        # Torch timing
        start.record()
        for _ in range(10):
            _ = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 10

        print(f"{n:>6} {triton_ms:>12.3f} {torch_ms:>12.3f} {torch_ms / triton_ms:>10.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton matmul benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--test", action="store_true", help="Run correctness test")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This lab requires a Linux machine with an NVIDIA GPU.")
        return 0

    if args.test:
        a = torch.randn(128, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(256, 128, device="cuda", dtype=torch.float16)
        c_triton = triton_matmul(a, b)
        c_torch = torch.matmul(a, b)
        max_diff = (c_triton - c_torch).abs().max().item()
        print(f"Max difference vs torch.matmul: {max_diff:.4f}")

    if args.benchmark:
        benchmark()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
