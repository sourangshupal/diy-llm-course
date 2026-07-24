"""Week 12 Lab 3: evalscope overview.

This script checks if evalscope is installed and prints a minimal usage example.
If installed, it runs a tiny custom evaluation via the evalscope API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="evalscope demo")
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()

    try:
        from evalscope import TaskConfig, run_task
    except ImportError:
        print("evalscope is not installed. Install it with:")
        print("    uv pip install evalscope")
        print("\nExample configuration (save as week12/evalscope_config.yaml):")
        print("""
model_args:
  pretrained: gpt2
datasets:
  - custom_math
generation_config:
  max_new_tokens: 64
  temperature: 0.7
use_cache: false
""")
        return 0

    # Minimal evalscope run using a built-in task if available
    config = TaskConfig(
        model=args.model,
        datasets=["gsm8k"],
        limit=5,
    )
    run_task(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
