"""Week 8 Lab 2: IsoFLOPs experiment simulation.

For a fixed FLOPs budget, sweep model sizes and compute the optimal data size
via D = FLOPs / (6 * N). Then predict loss using a fitted scaling law.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scaling_law import chinchilla_law


def load_params(path: Path) -> tuple[float, float, float, float, float]:
    """Load fitted scaling-law parameters."""
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    return p["E"], p["A"], p["B"], p["alpha"], p["beta"]


def isoflops_curve(
    flops_budget: float,
    params: tuple[float, float, float, float, float],
    n_models: int = 20,
    n_min: float = 1e7,
    n_max: float = 1e10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (N, D, loss) for an IsoFLOPs sweep."""
    N = np.logspace(np.log10(n_min), np.log10(n_max), n_models)
    D = flops_budget / (6 * N)
    X = np.stack([N, D], axis=1)
    loss = chinchilla_law(X, *params)
    return N, D, loss


def main() -> int:
    parser = argparse.ArgumentParser(description="IsoFLOPs experiment")
    parser.add_argument("--params", type=Path, default=Path("week08/outputs/scaling_params.json"))
    parser.add_argument("--flops", type=float, default=1e18, help="FLOPs budget")
    parser.add_argument("--output_dir", type=Path, default=Path("week08/outputs"))
    args = parser.parse_args()

    if not args.params.exists():
        print(f"Params file not found: {args.params}. Run scaling_law.py first.")
        return 1

    fitted_params = load_params(args.params)
    N, D, loss = isoflops_curve(args.flops, fitted_params)

    # Find optimal
    idx = np.argmin(loss)
    n_opt, d_opt, l_opt = N[idx], D[idx], loss[idx]

    print(f"IsoFLOPs sweep for FLOPs budget = {args.flops:.2e}")
    print(f"Optimal N = {n_opt:.2e} params")
    print(f"Optimal D = {d_opt:.2e} tokens")
    print(f"Predicted loss = {l_opt:.4f}")
    print(f"Tokens per param = {d_opt / n_opt:.1f}")

    # Plot
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(N, loss)
    axes[0].scatter([n_opt], [l_opt], color="red", zorder=5, label="optimal")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Non-embedding parameters N")
    axes[0].set_ylabel("Predicted loss")
    axes[0].set_title(f"IsoFLOPs curve (FLOPs = {args.flops:.1e})")
    axes[0].legend()
    axes[0].grid(True, linestyle=":")

    axes[1].plot(N, D / N)
    axes[1].scatter([n_opt], [d_opt / n_opt], color="red", zorder=5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Non-embedding parameters N")
    axes[1].set_ylabel("Tokens per parameter")
    axes[1].set_title("Data-to-model ratio along IsoFLOPs curve")
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    out_path = args.output_dir / f"isoflops_{args.flops:.0e}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
