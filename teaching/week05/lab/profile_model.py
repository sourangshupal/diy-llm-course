"""Week 5 Lab 1: Profile the Week 4 Transformer model with PyTorch Profiler.

Usage:
    python profile_model.py --device cuda --seq_len 128 --d_model 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler

# Allow importing the Week 4 model
# profile_model.py is at week05/lab/; model.py is at week04/lab/
model_dir = Path(__file__).resolve().parent.parent.parent / "week04" / "lab"
sys.path.insert(0, str(model_dir))
import importlib

model_module = importlib.import_module("model")
TransformerConfig = model_module.TransformerConfig
TransformerLM = model_module.TransformerLM


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile a mini Transformer")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile_steps", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    print(f"Profiling on {device}")

    config = TransformerConfig(
        vocab_size=1000,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
    )
    model = TransformerLM(config).to(device)
    x = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
    y = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)

    # Warmup
    for _ in range(args.warmup):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Profile
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        on_trace_ready=tensorboard_trace_handler("week05/profiling_logs"),
    ) as prof:
        for _ in range(args.profile_steps):
            with record_function("model_forward"):
                logits = model(x)
            with record_function("model_backward"):
                loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            prof.step()

    print("\nTop 10 operations by total time:")
    print(prof.key_averages().table(sort_by=f"{device.type}_time_total", row_limit=10))

    if device.type == "cuda":
        print("\nTop 10 CUDA kernels:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\nTrace saved to week05/profiling_logs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
