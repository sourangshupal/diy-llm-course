"""Week 3 Lab 4: RMSNorm and SwiGLU feed-forward network.

Compares:
  - LayerNorm vs RMSNorm,
  - standard ReLU FFN vs SwiGLU FFN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * norm


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network used in modern LLMs."""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        if d_ff is None:
            # Common modern choice: 8/3 * d_model, rounded to a multiple of 256
            d_ff = 8 * d_model // 3
            d_ff = 256 * ((d_ff + 255) // 256)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class StandardFFN(nn.Module):
    """Standard ReLU FFN for comparison."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> int:
    """Compare normalization and FFN variants."""
    d_model = 512
    batch, seq = 2, 8
    x = torch.randn(batch, seq, d_model)

    print("Normalization comparison:")
    layernorm = nn.LayerNorm(d_model)
    rmsnorm = RMSNorm(d_model)
    print(f"  LayerNorm output mean: {layernorm(x).mean(dim=-1).abs().mean().item():.6f}")
    print(f"  RMSNorm output mean:   {rmsnorm(x).mean(dim=-1).abs().mean().item():.6f}")
    print("  (RMSNorm does not center the mean.)")

    print("\nFFN comparison:")
    standard = StandardFFN(d_model, d_ff=4 * d_model)
    swiglu = SwiGLU(d_model)
    print(f"  Standard FFN params: {count_parameters(standard):,}")
    print(f"  SwiGLU params:       {count_parameters(swiglu):,}")
    print(f"  SwiGLU d_ff:         {swiglu.w1.out_features}")

    out_std = standard(x)
    out_swiglu = swiglu(x)
    print(f"  Standard output shape: {out_std.shape}")
    print(f"  SwiGLU output shape:   {out_swiglu.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
