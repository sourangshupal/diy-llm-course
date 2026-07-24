"""Week 7 Lab 3: Estimate distributed-training communication cost.

Computes the all-reduce communication volume and time for a given model size,
world size, and interconnect bandwidth.
"""

from __future__ import annotations

import argparse


def all_reduce_volume(params: int, bytes_per_param: float = 2.0, world_size: int = 2) -> float:
    """Return all-reduce communication volume in bytes.

    Ring all-reduce moves 2 * (n - 1) / n * total_bytes across the ring.
    """
    total_bytes = params * bytes_per_param
    return 2 * (world_size - 1) / world_size * total_bytes


def estimate_time(bytes_moved: float, bandwidth_gbps: float) -> float:
    """Return time in milliseconds given bandwidth in Gbps."""
    bandwidth_bytes_per_ms = bandwidth_gbps * 1e9 / 8 / 1000
    return bytes_moved / bandwidth_bytes_per_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate communication cost")
    parser.add_argument("--params", type=float, default=1e9, help="Model parameters")
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--bytes_per_param", type=float, default=2.0, choices=[2.0, 4.0])
    parser.add_argument("--bandwidth", type=float, default=400.0, help="Interconnect bandwidth Gbps")
    parser.add_argument("--step_time_ms", type=float, default=100.0, help="Compute time per step in ms")
    args = parser.parse_args()

    volume = all_reduce_volume(int(args.params), args.bytes_per_param, args.world_size)
    comm_time = estimate_time(volume, args.bandwidth)
    total_time = args.step_time_ms + comm_time
    overhead = comm_time / total_time * 100

    print(f"Model parameters:      {args.params:,.0f}")
    print(f"World size:            {args.world_size}")
    print(f"Bytes per param:       {args.bytes_per_param}")
    print(f"All-reduce volume:     {volume / 1e9:.2f} GB")
    print(f"Interconnect:          {args.bandwidth} Gbps")
    print(f"Communication time:    {comm_time:.2f} ms")
    print(f"Compute time:          {args.step_time_ms:.2f} ms")
    print(f"Total step time:       {total_time:.2f} ms")
    print(f"Communication overhead: {overhead:.1f}%")

    # Scaling efficiency
    ideal_speedup = args.world_size
    actual_speedup = args.step_time_ms / total_time * args.world_size
    print(f"Ideal speedup:         {ideal_speedup:.1f}x")
    print(f"Actual speedup:        {actual_speedup:.1f}x")
    print(f"Efficiency:            {actual_speedup / ideal_speedup * 100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
