"""Tests for the sequential-MNIST row encoder (exp 011)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
EXP_DIR = HERE.parent / "experiments" / "011_week7_sequential_mnist"
sys.path.insert(0, str(EXP_DIR))

from models import row_encode  # noqa: E402


def test_row_encode_4d_shape():
    """[B, 1, 28, 28] -> [28, B, 28]."""
    out = row_encode(torch.zeros(8, 1, 28, 28))
    assert out.shape == (28, 8, 28), f"expected (28, 8, 28), got {tuple(out.shape)}"


def test_row_encode_3d_shape():
    """[B, 28, 28] (no channel dim) -> [28, B, 28]."""
    out = row_encode(torch.zeros(8, 28, 28))
    assert out.shape == (28, 8, 28), f"expected (28, 8, 28), got {tuple(out.shape)}"


def test_row_encode_preserves_row_content():
    """Row t of the image must end up at time index t of the sequence."""
    image = torch.arange(28 * 28).view(28, 28).float()  # row r is arange(r*28, (r+1)*28)
    batch = image.unsqueeze(0).unsqueeze(0).expand(4, 1, 28, 28).contiguous()  # [4, 1, 28, 28]
    out = row_encode(batch)
    for t in range(28):
        expected = torch.arange(t * 28, (t + 1) * 28).float()
        for b in range(4):
            assert torch.equal(out[t, b], expected), (
                f"row mismatch at t={t}, b={b}: {out[t, b]} vs {expected}"
            )


def test_row_encode_is_pure():
    """Calling row_encode twice on the same input gives equal tensors."""
    x = torch.rand(2, 1, 28, 28)
    a = row_encode(x)
    b = row_encode(x)
    assert torch.equal(a, b)
