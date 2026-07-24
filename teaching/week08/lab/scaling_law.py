"""Week 8 Lab 1: Fit the Chinchilla scaling law.

L(N, D) = E + A / N^alpha + B / D^beta

This script generates synthetic data, fits the law, and plots actual vs.
predicted loss. Replace synthetic_data() with your own training results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit


def chinchilla_law(x: np.ndarray, E: float, A: float, B: float, alpha: float, beta: float) -> np.ndarray:
    """Compute L(N, D) = E + A/N^alpha + B/D^beta."""
    N, D = x[:, 0], x[:, 1]
    return E + A * np.power(N, -alpha) + B * np.power(D, -beta)


def synthetic_data(seed: int = 0, n_points: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (N, D, loss) points around a known law."""
    rng = np.random.default_rng(seed)
    # True parameters close to Chinchilla
    E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28
    N = np.exp(rng.uniform(np.log(5e7), np.log(1e10), n_points))
    D = np.exp(rng.uniform(np.log(1e9), np.log(5e11), n_points))
    noise = rng.normal(0, 0.02, n_points)
    loss = E + A * N ** (-alpha) + B * D ** (-beta) + noise
    X = np.stack([N, D], axis=1)
    return X, loss


def fit_scaling_law(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit Chinchilla parameters."""
    # Initial guess near Chinchilla
    p0 = [1.5, 400.0, 400.0, 0.3, 0.3]
    bounds = (
        [0.0, 1.0, 1.0, 0.01, 0.01],
        [5.0, 5000.0, 5000.0, 1.0, 1.0],
    )
    popt, pcov = curve_fit(chinchilla_law, X, y, p0=p0, bounds=bounds, maxfev=10000)
    return popt, pcov


def plot_fit(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot actual vs. predicted loss."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.7)
    lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
    plt.xlabel("Actual loss")
    plt.ylabel("Predicted loss")
    plt.title("Scaling law fit: actual vs. predicted")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit Chinchilla scaling law")
    parser.add_argument("--data", type=Path, default=None, help="JSON file with list of {N, D, loss}")
    parser.add_argument("--output_dir", type=Path, default=Path("week08/outputs"))
    args = parser.parse_args()

    if args.data:
        with open(args.data, "r", encoding="utf-8") as f:
            records = json.load(f)
        N = np.array([r["N"] for r in records])
        D = np.array([r["D"] for r in records])
        y = np.array([r["loss"] for r in records])
        X = np.stack([N, D], axis=1)
    else:
        print("Using synthetic data. Replace with your own training results for real use.")
        X, y = synthetic_data()

    popt, _ = fit_scaling_law(X, y)
    E, A, B, alpha, beta = popt
    print("Fitted parameters:")
    print(f"  E     = {E:.4f}")
    print(f"  A     = {A:.2f}")
    print(f"  B     = {B:.2f}")
    print(f"  alpha = {alpha:.4f}")
    print(f"  beta  = {beta:.4f}")

    y_pred = chinchilla_law(X, *popt)
    mse = np.mean((y - y_pred) ** 2)
    print(f"\nMSE: {mse:.6f}")

    # Save fitted params
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = {"E": float(E), "A": float(A), "B": float(B), "alpha": float(alpha), "beta": float(beta)}
    with open(args.output_dir / "scaling_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    plot_fit(X, y, y_pred, args.output_dir / "scaling_fit.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
