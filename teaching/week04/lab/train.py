"""Week 4 Lab: training loop for the mini Transformer language model.

This script is intentionally small so students can read the whole loop in one
sitting. It supports W&B logging, checkpointing, and a cosine LR schedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from data import CharDataset, build_corpus_file
from model import TransformerConfig, TransformerLM


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Return a LambdaLR that linearly warms up then cosine decays."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: TransformerLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    log_interval: int = 10,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})

        if step % log_interval == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/step": epoch * len(loader) + step,
                }
            )

    return total_loss / max(1, num_batches)


def save_checkpoint(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    path: Path,
) -> None:
    """Save model, optimizer, scheduler, and config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": model.config,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def main() -> int:
    """Parse args and train a mini language model."""
    parser = argparse.ArgumentParser(description="Train a mini Transformer LM")
    parser.add_argument("--data", type=Path, default=Path("week04/data/corpus.txt"))
    parser.add_argument("--output_dir", type=Path, default=Path("week04/outputs"))
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--wandb_project", type=str, default="diy-llm-week4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build or load corpus
    if not args.data.exists():
        build_corpus_file(args.data, repeats=100)
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    # Use the actual vocab size from the data if it's smaller
    tokenizer = CharDataset(text, seq_len=args.seq_len).tokenizer
    vocab_size = min(args.vocab_size, tokenizer.vocab_size)
    print(f"Effective vocab size: {vocab_size}")

    # Dataset / loader
    dataset = CharDataset(text, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Model
    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
    )
    model = TransformerLM(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer and schedule
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, args.min_lr_ratio
    )

    # W&B
    wandb.init(
        project=args.wandb_project,
        config={
            **vars(args),
            "vocab_size": vocab_size,
            "parameters": model.count_parameters(),
        },
    )

    # Training loop
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, loader, optimizer, scheduler, device, args.grad_clip, epoch)
        print(f"Epoch {epoch}/{args.epochs} — avg loss: {avg_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch})

        save_checkpoint(model, optimizer, scheduler, epoch, args.output_dir / "last.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, scheduler, epoch, args.output_dir / "best.pt")

    # Save config separately for easy loading during generation
    with open(args.output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(config), f, indent=2, default=lambda o: o.__dict__)

    wandb.finish()
    print("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
