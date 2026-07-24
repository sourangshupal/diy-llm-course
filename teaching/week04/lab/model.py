"""Week 4 Lab: A minimal decoder-only Transformer language model.

This model uses modern design choices found in Llama/Qwen-style architectures:
  - RMSNorm pre-normalization,
  - causal multi-head self-attention with RoPE,
  - SwiGLU feed-forward network,
  - no bias in linear layers,
  - optional weight tying between input embeddings and output projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for the decoder-only Transformer."""

    vocab_size: int = 256
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int | None = None  # None => 8/3 * d_model rounded to 256
    max_seq_len: int = 512
    dropout: float = 0.0
    tie_weights: bool = True
    theta: float = 10000.0  # RoPE base

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.d_ff is None:
            d_ff = 8 * self.d_model // 3
            self.d_ff = 256 * ((d_ff + 255) // 256)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * norm


class RotaryEmbedding(nn.Module):
    """Precomputes and applies RoPE to query/key tensors."""

    def __init__(self, d_head: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError("d_head must be even for RoPE")
        self.d_head = d_head
        freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), freqs)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (..., seq_len, d_head)."""
        *leading, seq_len, d_head = x.shape
        x_pairs = x.reshape(*leading, seq_len, d_head // 2, 2)
        x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
        cos = self.cos[:seq_len]  # (seq_len, d_head // 2)
        sin = self.sin[:seq_len]
        rotated0 = x0 * cos - x1 * sin
        rotated1 = x0 * sin + x1 * cos
        rotated = torch.stack([rotated0, rotated1], dim=-1)
        return rotated.reshape(*leading, seq_len, d_head)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.d_head = config.d_model // config.num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.d_head, config.max_seq_len, config.theta)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape

        q = self.w_q(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = torch.tril(torch.ones(seq, seq, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.w_o(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    """One decoder block: pre-norm attention + pre-norm FFN."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    """Full decoder-only language model."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, seq, vocab_size)."""
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main() -> int:
    """Sanity check: forward pass a small model."""
    config = TransformerConfig(
        vocab_size=256,
        d_model=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=64,
    )
    model = TransformerLM(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(x)
    print(f"Config: {config}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
