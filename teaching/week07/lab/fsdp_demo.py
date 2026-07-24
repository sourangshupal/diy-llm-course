"""Week 7 Lab 2: Fully Sharded Data Parallel (FSDP) demo.

FSDP shards model parameters, gradients, and optimizer states across ranks.
This script is best run on a multi-GPU machine with the nccl backend.

CPU execution is possible for syntax checking but is not the intended use case.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, TensorDataset


class TinyModel(nn.Module):
    """Small model for the FSDP demo."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def setup_distributed(rank: int, world_size: int, backend: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def run_training(
    rank: int,
    world_size: int,
    backend: str,
    epochs: int,
) -> None:
    setup_distributed(rank, world_size, backend)

    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    model = TinyModel().to(device)

    # Wrap with FSDP using an auto-wrap policy
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy(min_num_params=1000),
        device_id=device if backend == "nccl" else None,
    )

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3)

    torch.manual_seed(42 + rank)
    x = torch.randn(128, 64)
    y = torch.randint(0, 2, (128,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = fsdp_model(batch_x)
            loss = nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Rank {rank} | Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="FSDP demo")
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    if args.backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("nccl backend requires CUDA")

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        run_training(rank, args.world_size, args.backend, args.epochs)
    else:
        mp.spawn(
            run_training,
            args=(args.world_size, args.backend, args.epochs),
            nprocs=args.world_size,
            join=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
