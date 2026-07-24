"""Week 3 Lab 3: Causal multi-head self-attention.

Implements scaled dot-product attention with a causal mask.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Return a boolean causal mask of shape (size, size).

    True means "allow attention"; False means "mask out".
    """
    return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))


class CausalSelfAttention(nn.Module):
    """Single-head causal scaled dot-product attention."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, seq, d_model)."""
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        mask = causal_mask(scores.size(-1), x.device)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Replace NaNs from all-masked rows (should not happen with causal mask)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)
        return self.w_o(out), attn


class MultiHeadCausalAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, seq, d_model)."""
        batch, seq, _ = x.shape

        # Project and reshape to (batch, num_heads, seq, d_head)
        q = self.w_q(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = causal_mask(seq, x.device).unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)  # (batch, num_heads, seq, d_head)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.w_o(out), attn


def main() -> int:
    """Run attention demos."""
    d_model = 64
    seq_len = 6
    batch_size = 2
    x = torch.randn(batch_size, seq_len, d_model)

    print("Single-head attention:")
    single = CausalSelfAttention(d_model)
    out_single, attn_single = single(x)
    print(f"  Output shape: {out_single.shape}")
    print(f"  Upper-triangular attention max: {attn_single[0].triu(1).max().item():.6f}")

    print("\nMulti-head attention:")
    multi = MultiHeadCausalAttention(d_model, num_heads=4)
    out_multi, attn_multi = multi(x)
    print(f"  Output shape: {out_multi.shape}")
    print(f"  Attention weights shape: {attn_multi.shape}")
    print(f"  Upper-triangular attention max: {attn_multi[0, 0].triu(1).max().item():.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
