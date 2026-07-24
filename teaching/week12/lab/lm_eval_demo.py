"""Week 12 Lab 1: lm-evaluation-harness demo.

This script checks if lm-eval is installed and runs a small benchmark.
If not installed, it prints installation instructions.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def is_lm_eval_available() -> bool:
    """Check if the lm-eval package is importable."""
    try:
        import lm_eval  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="lm-evaluation-harness demo")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--tasks", type=str, default="hellaswag")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not is_lm_eval_available():
        print("lm-eval is not installed. Install it with:")
        print("    uv pip install lm-eval")
        print("\nThen run this script again.")
        return 0

    # Use the lm-eval CLI
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.model}",
        "--tasks", args.tasks,
        "--num_fewshot", str(args.num_fewshot),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--output_path", "week12/outputs/lm_eval",
        "--log_samples",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
