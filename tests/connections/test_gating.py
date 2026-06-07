"""Tests for the apply_gate control primitive (Phase 2, build-order step 4)."""

from __future__ import annotations

import pytest
import torch

from neuromorphic.connections.gating import apply_gate


def test_closed_channel_blocks_signal():
    """gate_closed = 1 → channel blocked (output 0)."""
    signal = torch.ones(8, 2, 4)
    gate_closed = torch.ones(8, 2, 4)
    assert torch.count_nonzero(apply_gate(signal, gate_closed)) == 0


def test_open_channel_passes_signal():
    """gate_closed = 0 → channel open (signal unchanged)."""
    signal = torch.rand(8, 2, 4)
    gate_closed = torch.zeros(8, 2, 4)
    assert torch.equal(apply_gate(signal, gate_closed), signal)


def test_per_channel_gating():
    """Each channel gated independently."""
    signal = torch.ones(1, 1, 4)
    gate_closed = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    out = apply_gate(signal, gate_closed)
    assert out.tolist() == [[[0.0, 1.0, 0.0, 1.0]]]


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        apply_gate(torch.ones(8, 2, 4), torch.ones(8, 2, 3))
