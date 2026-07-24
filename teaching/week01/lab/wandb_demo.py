"""Week 1 Lab 2: Weights & Biases experiment tracking demo.

This script demonstrates:
  - initializing a W&B run with a config,
  - logging metrics during a fake training loop,
  - creating and logging an artifact (a JSON config file),
  - finishing the run cleanly.

Before running:
    uv sync
    source .venv/bin/activate
    wandb login
"""

from __future__ import annotations

import json
import os
import random
import tempfile

import numpy as np
import wandb


def make_config() -> dict:
    """Return a small hyperparameter config for the demo run."""
    return {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 5,
        "model_dim": 128,
        "seed": 42,
    }


def train_demo(config: dict) -> list[float]:
    """Simulate a training loop and return per-step losses."""
    rng = np.random.default_rng(config["seed"])
    losses = []
    for step in range(100):
        # Fake loss that decreases with noise
        base = 2.0 * np.exp(-0.03 * step)
        noise = rng.normal(0.0, 0.05)
        loss = float(base + noise)
        losses.append(loss)

        # Log metrics to W&B every step
        wandb.log(
            {
                "train/loss": loss,
                "train/loss_smooth": float(base),
                "train/learning_rate": config["learning_rate"] * (1 - step / 120),
            },
            step=step,
        )
    return losses


def log_artifact(config: dict) -> None:
    """Create a temporary JSON config file and log it as a W&B artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        artifact = wandb.Artifact(
            name="demo-config",
            type="config",
            description="Hyperparameter configuration for the demo run",
        )
        artifact.add_file(config_path, name="config.json")
        wandb.log_artifact(artifact)


def main() -> int:
    """Run the W&B demo."""
    config = make_config()

    # Initialize W&B run with project and config
    run = wandb.init(
        project="diy-llm-week1-demo",
        config=config,
        notes="Week 1 W&B experiment tracking demo",
        tags=["week1", "demo"],
    )
    print(f"W&B run URL: {run.url}")

    # Simulate training
    print("Running fake training loop...")
    train_demo(config)

    # Log an artifact
    log_artifact(config)

    # Close the run
    wandb.finish()
    print("Demo complete. Open the run URL above to view results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
