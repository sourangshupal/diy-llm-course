"""Week 3 Lab 1: Token and positional embeddings.

Demonstrates:
  - token embedding lookup,
  - additive learned positional embeddings,
  - output shape checks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    """Simple token + learned positional embedding layer."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return embeddings of shape (batch, seq, d_model)."""
        batch, seq = token_ids.shape
        positions = torch.arange(seq, device=token_ids.device).unsqueeze(0).expand(batch, -1)
        tok_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)
        return tok_emb + pos_emb


def main() -> int:
    """Run a small embedding demo."""
    vocab_size = 100
    d_model = 64
    max_seq_len = 128
    batch_size = 4
    seq_len = 16

    layer = TokenAndPositionEmbedding(vocab_size, d_model, max_seq_len)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    embeddings = layer(token_ids)
    print(f"Input shape:  {token_ids.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected:     ({batch_size}, {seq_len}, {d_model})")

    # Show that different positions have different embeddings even for the same token
    same_token = torch.full((1, seq_len), 5, dtype=torch.long)
    same_token_embs = layer(same_token)
    print("\nPosition embeddings differ for the same token:")
    print(f"  pos 0 vs pos 1 L2 distance: {torch.dist(same_token_embs[0, 0], same_token_embs[0, 1]):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
