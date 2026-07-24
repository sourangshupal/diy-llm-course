"""Week 3 Lab 2: Rotary Positional Embeddings (RoPE).

This lab shows how to apply 2D rotations to query/key vectors so that the
inner product depends on the relative position m - n.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def precompute_freqs_cis(d_model: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex rotation frequencies for RoPE.

    Returns a tensor of shape (max_seq_len, d_model // 2, 2) where the last
    dimension holds (cos, sin) for each frequency and position.
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for RoPE")

    # Frequencies: theta_i = theta^{-2i / d_model} for i in [0, d_model/2)
    freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # (max_seq_len, d_model // 2)

    # Stack cos and sin
    freqs_cis = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    return freqs_cis  # (max_seq_len, d_model // 2, 2)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a tensor x of shape (..., seq_len, d_model).

    The last dimension is interpreted as consecutive pairs (x_{2i}, x_{2i+1}).
    """
    *leading, seq_len, d_model = x.shape
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for RoPE")

    # Reshape to (..., seq_len, d_model // 2, 2)
    x_pairs = x.reshape(*leading, seq_len, d_model // 2, 2)

    # Get cos/sin for the relevant positions
    cis = freqs_cis[:seq_len]  # (seq_len, d_model // 2, 2)
    cos, sin = cis[..., 0], cis[..., 1]  # each (seq_len, d_model // 2)

    # Apply rotation: [cos, -sin; sin, cos] @ [x0; x1]
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    rotated0 = x0 * cos - x1 * sin
    rotated1 = x0 * sin + x1 * cos
    rotated = torch.stack([rotated0, rotated1], dim=-1)

    return rotated.reshape(*leading, seq_len, d_model)


def main() -> int:
    """Demonstrate RoPE on random query/key vectors."""
    d_model = 64
    num_heads = 4
    d_head = d_model // num_heads
    seq_len = 8

    # Random Q, K for one head
    q = torch.randn(seq_len, d_head)
    k = torch.randn(seq_len, d_head)

    freqs_cis = precompute_freqs_cis(d_head, max_seq_len=seq_len)
    q_rot = apply_rope(q, freqs_cis)
    k_rot = apply_rope(k, freqs_cis)

    # Dot-product attention scores
    scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(d_head)
    print("RoPE attention scores (rows = query positions, cols = key positions):")
    print(scores)

    # Demonstrate that position 0 attending to itself vs position 3 attending to itself
    # has similar self-dot-product magnitude (rotation preserves norm)
    self_0 = torch.dot(q_rot[0], k_rot[0]).item()
    self_3 = torch.dot(q_rot[3], k_rot[3]).item()
    print(f"\nSelf-attention at pos 0: {self_0:.4f}")
    print(f"Self-attention at pos 3: {self_3:.4f}")
    print("(Norms are preserved; difference comes from angle-dependent interaction.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
