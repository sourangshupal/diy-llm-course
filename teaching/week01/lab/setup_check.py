"""Week 1 Lab 1: Environment and GPU setup check.

Run this script to verify that the shared uv environment is correctly
installed and that PyTorch can see any available GPU.
"""

from __future__ import annotations

import platform
import sys

import numpy as np
import torch
import tqdm


def check_python() -> None:
    """Verify Python version is at least 3.12."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version < (3, 12):
        raise RuntimeError("This course requires Python >= 3.12.")
    print("✅ Python version OK")


def check_packages() -> None:
    """Verify that core packages can be imported and report versions."""
    packages = {
        "numpy": np.__version__,
        "torch": torch.__version__,
        "tqdm": tqdm.__version__,
    }
    for name, ver in packages.items():
        print(f"  {name}: {ver}")
    print("✅ Core packages imported")


def check_torch_backend() -> None:
    """Report available compute backend (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA available: {device_count} device(s)")
        print(f"   Device 0: {device_name}")
        # Quick smoke test
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x.T
        print(f"   GPU matmul smoke test OK (result shape: {y.shape})")
    elif torch.backends.mps.is_available():
        print("✅ Apple MPS backend available")
        x = torch.randn(100, 100, device="mps")
        y = x @ x.T
        print(f"   MPS matmul smoke test OK (result shape: {y.shape})")
    else:
        print("ℹ️  Running on CPU only (OK for Weeks 1–2 and Week 12 demos)")


def check_system() -> None:
    """Print platform info."""
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")


def main() -> int:
    """Run all checks and return 0 on success."""
    print("=" * 60)
    print("Diy-LLM Teaching Environment Check")
    print("=" * 60)
    check_system()
    check_python()
    check_packages()
    check_torch_backend()
    print("=" * 60)
    print("All checks passed. You are ready for Week 1!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
