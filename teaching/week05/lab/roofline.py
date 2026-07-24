"""Week 5 Lab 2: Roofline model plotter.

Given a GPU's peak compute (TFLOP/s) and memory bandwidth (GB/s), plot the
roofline and mark a few representative operations.
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt


def roofline_peak_flops(arithmetic_intensity: float, peak_compute: float, bandwidth: float) -> float:
    """Return attainable TFLOP/s for a given arithmetic intensity."""
    return min(peak_compute, arithmetic_intensity * bandwidth)


def attention_intensity(
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    bytes_per_param: float = 2.0,
) -> float:
    """Estimate arithmetic intensity (FLOPs / byte) of standard attention.

    We count Q@K^T and softmax@V as the dominant terms and assume we read/write
    Q, K, V and the attention matrix once each.
    """
    d_head = d_model // num_heads
    # FLOPs: 2 * batch * num_heads * seq_len^2 * d_head (for QK^T + AV)
    flops = 2 * batch * num_heads * seq_len * seq_len * d_head * 2
    # Bytes: read Q, K, V; write attention + output
    bytes_moved = bytes_per_param * (3 * batch * seq_len * d_model + 2 * batch * num_heads * seq_len * seq_len + batch * seq_len * d_model)
    return flops / bytes_moved


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a roofline model")
    parser.add_argument("--peak_compute", type=float, default=312.0, help="Peak FP16/BF16 TFLOP/s")
    parser.add_argument("--bandwidth", type=float, default=2000.0, help="Memory bandwidth GB/s")
    parser.add_argument("--output", type=str, default="week05/roofline.png")
    args = parser.parse_args()

    # Ridge point
    ridge_intensity = args.peak_compute / args.bandwidth

    intensities = [2 ** (i / 4) for i in range(-40, 80)]
    peaks = [roofline_peak_flops(ai, args.peak_compute, args.bandwidth) for ai in intensities]

    # Example operations
    ops = {
        "matmul (large)": 100.0,
        "attention (seq=512)": attention_intensity(2, 512, 512, 8),
        "attention (seq=2048)": attention_intensity(2, 2048, 512, 8),
        "elementwise": 1.0,
    }

    plt.figure(figsize=(8, 6))
    plt.loglog(intensities, peaks, label="Roofline", linewidth=2)
    plt.axvline(ridge_intensity, color="red", linestyle="--", label=f"Ridge point = {ridge_intensity:.2f}")

    for name, ai in ops.items():
        peak = roofline_peak_flops(ai, args.peak_compute, args.bandwidth)
        plt.scatter([ai], [peak], s=100, zorder=5)
        plt.annotate(name, (ai, peak), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("Arithmetic Intensity (FLOPs / byte)")
    plt.ylabel("Attainable Performance (TFLOP/s)")
    plt.title(f"Roofline Model ({args.peak_compute} TFLOP/s, {args.bandwidth} GB/s)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Roofline plot saved to {args.output}")

    print("\nEstimated ridge point:", ridge_intensity)
    for name, ai in ops.items():
        peak = roofline_peak_flops(ai, args.peak_compute, args.bandwidth)
        bound = "compute-bound" if ai >= ridge_intensity else "memory-bound"
        print(f"  {name:25s}: AI={ai:8.2f}, {peak:7.2f} TFLOP/s ({bound})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
