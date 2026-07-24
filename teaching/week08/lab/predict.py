"""Week 8 Lab 3: Predict loss for a target model size and token count.

Usage:
    python predict.py --params week08/outputs/scaling_params.json --N 1e8 --D 2e9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scaling_law import chinchilla_law


def load_params(path: Path) -> tuple[float, float, float, float, float]:
    """Load fitted scaling-law parameters."""
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    return p["E"], p["A"], p["B"], p["alpha"], p["beta"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict loss from scaling law")
    parser.add_argument("--params", type=Path, default=Path("week08/outputs/scaling_params.json"))
    parser.add_argument("--N", type=float, required=True, help="Non-embedding parameters")
    parser.add_argument("--D", type=float, required=True, help="Training tokens")
    args = parser.parse_args()

    if not args.params.exists():
        print(f"Params file not found: {args.params}. Run scaling_law.py first.")
        return 1

    fitted_params = load_params(args.params)
    X = np.array([[args.N, args.D]])
    loss = chinchilla_law(X, *fitted_params)[0]
    flops = 6 * args.N * args.D

    print(f"Model size N:     {args.N:.2e} params")
    print(f"Training tokens D: {args.D:.2e}")
    print(f"Training FLOPs:   {flops:.2e}")
    print(f"Predicted loss:   {loss:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
