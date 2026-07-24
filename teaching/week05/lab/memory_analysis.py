"""Week 5 Lab 3: Estimate activation memory for a Transformer.

This script breaks down memory usage per layer and discusses checkpointing trade-offs.
"""

from __future__ import annotations

import argparse
import math


def activation_memory_bytes(
    batch: int,
    seq_len: int,
    d_model: int,
    num_layers: int,
    vocab_size: int,
    bytes_per_param: float = 4.0,
) -> dict[str, float]:
    """Estimate activation memory for a decoder-only Transformer.

    Assumes store all activations for backward (no checkpointing).
    """
    # Per-layer activations: input to each sub-layer + attention scores + FFN intermediate
    per_layer = {
        "attn_input": batch * seq_len * d_model,
        "attn_scores": batch * num_layers * seq_len * seq_len,  # simplified
        "ffn_input": batch * seq_len * d_model,
        "ffn_intermediate": batch * seq_len * d_model * 4,  # standard FFN
    }
    per_layer_total = sum(per_layer.values())
    total = per_layer_total * num_layers + batch * seq_len * vocab_size  # logits
    return {
        "per_layer_bytes": per_layer_total * bytes_per_param,
        "total_bytes": total * bytes_per_param,
        "total_gib": total * bytes_per_param / (1024 ** 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate Transformer activation memory")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--checkpointing", action="store_true", help="Assume activation checkpointing")
    args = parser.parse_args()

    mem = activation_memory_bytes(args.batch, args.seq_len, args.d_model, args.num_layers, args.vocab_size)
    print("Activation memory estimate (no checkpointing):")
    print(f"  Per layer: {mem['per_layer_bytes'] / 1e6:.2f} MB")
    print(f"  Total:     {mem['total_bytes'] / 1e6:.2f} MB ({mem['total_gib']:.2f} GiB)")

    if args.checkpointing:
        # With checkpointing, only store sub-layer inputs; recompute attention/FFN in backward
        # Rough estimate: ~1/3 to 1/2 of full activations
        print(f"\nWith activation checkpointing (approximate):")
        print(f"  Total: {mem['total_bytes'] * 0.4 / 1e6:.2f} MB (saved ~60%)")
        print("  Trade-off: extra forward recomputation during backward pass")

    # Parameter memory
    d_ff = 4 * args.d_model
    params = (
        args.vocab_size * args.d_model  # embeddings
        + args.num_layers * (
            4 * args.d_model * args.d_model  # Q,K,V,O projections
            + 2 * args.d_model * d_ff  # FFN weights
            + 2 * args.d_model  # RMSNorm weights
        )
        + args.d_model  # final norm
    )
    param_bytes = params * 4  # float32
    print(f"\nParameter memory (float32): {param_bytes / 1e6:.2f} MB ({param_bytes / 1e9:.2f} GB)")
    print(f"Parameter memory (bf16):    {param_bytes / 2 / 1e6:.2f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
