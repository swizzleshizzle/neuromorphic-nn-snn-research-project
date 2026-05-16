"""Shared fixtures and matplotlib backend setup for viz smoke tests."""

import matplotlib

matplotlib.use("Agg")  # Headless backend — no GUI windows during pytest.

import pytest
import torch


@pytest.fixture(scope="session")
def small_shape():
    """Canonical small synthetic shape used by every smoke test."""
    return {"T": 20, "B": 4, "N": 50}


@pytest.fixture(scope="session")
def spk(small_shape):
    """Synthetic [T, B, N] spike tensor at ~20% firing rate."""
    torch.manual_seed(0)
    T, B, N = small_shape["T"], small_shape["B"], small_shape["N"]
    return (torch.rand(T, B, N) < 0.2).float()


@pytest.fixture(scope="session")
def mem(small_shape):
    """Synthetic [T, B, N] membrane-potential tensor."""
    torch.manual_seed(1)
    T, B, N = small_shape["T"], small_shape["B"], small_shape["N"]
    return torch.randn(T, B, N) * 0.5


@pytest.fixture(scope="session")
def W():
    """Synthetic [N_post=100, N_pre=50] weight matrix."""
    torch.manual_seed(2)
    return torch.randn(100, 50) * 0.1


@pytest.fixture(scope="session")
def history():
    """Synthetic training history dict with loss and accuracy series."""
    return {
        "train_loss": [3.0 - 0.01 * i for i in range(100)],
        "test_loss":  [3.0 - 0.02 * i for i in range(10)],
        "test_acc":   [0.1 + 0.08 * i for i in range(10)],
    }
