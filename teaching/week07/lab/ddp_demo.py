"""Week 7 Lab 1: DistributedDataParallel (DDP) demo.

This script works on both CPU (gloo backend) and GPU (nccl backend).
Use torchrun to launch multiple processes:

    CPU:
        python -m torchrun --nproc_per_node=2 week07/lab/ddp_demo.py
    GPU:
        python -m torchrun --nproc_per_node=2 week07/lab/ddp_demo.py --backend nccl
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset


class TinyModel(nn.Module):
    """A tiny MLP for the DDP demo."""

    def __init__(self, in_features: int = 10, hidden: int = 32, out_features: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def setup_distributed(rank: int, world_size: int, backend: str) -> None:
    """Initialize the process group."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup() -> None:
    """Destroy the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_training(
    rank: int,
    world_size: int,
    backend: str,
    epochs: int,
    batch_size: int,
) -> None:
    """Run training on one process."""
    setup_distributed(rank, world_size, backend)

    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    model = TinyModel().to(device)
    ddp_model = DDP(model, device_ids=[rank] if backend == "nccl" else None)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    # Each rank gets a different synthetic shard
    torch.manual_seed(42 + rank)
    x = torch.randn(256, 10)
    y = torch.randint(0, 2, (256,))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = ddp_model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

        # All ranks should have the same averaged gradients
        print(f"Rank {rank} | Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="DDP demo")
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("nccl backend requires CUDA")

    # When launched with torchrun, these env vars are set automatically.
    # For standalone testing we spawn manually.
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        run_training(rank, args.world_size, args.backend, args.epochs, args.batch_size)
    else:
        mp.spawn(
            run_training,
            args=(args.world_size, args.backend, args.epochs, args.batch_size),
            nprocs=args.world_size,
            join=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
