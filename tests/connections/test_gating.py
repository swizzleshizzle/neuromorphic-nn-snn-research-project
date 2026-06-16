"""Tests for the apply_gate control primitive (Phase 2, build-order step 4)."""

from __future__ import annotations

import pytest
import torch

from neuromorphic.connections.gating import apply_gain, apply_gate


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


# --- Continuous multiplicative gain (spec §2.5, designed L11) --------------- #

def test_gain_zero_blocks_signal():
    """g = 0 → channel off (output 0), matching a fully closed gate."""
    signal = torch.rand(8, 2, 4)
    gain = torch.zeros(8, 2, 4)
    assert torch.count_nonzero(apply_gain(signal, gain)) == 0


def test_gain_one_is_identity():
    """g = 1 → pass-through (the old fully-open behaviour)."""
    signal = torch.rand(8, 2, 4)
    gain = torch.ones(8, 2, 4)
    assert torch.equal(apply_gain(signal, gain), signal)


def test_gain_amplifies():
    """g = 2 → 2× the signal (router can now amplify, not just suppress)."""
    signal = torch.rand(8, 2, 4)
    gain = torch.full((8, 2, 4), 2.0)
    assert torch.equal(apply_gain(signal, gain), 2.0 * signal)


def test_gain_suppresses():
    """0 < g < 1 → partial suppression."""
    signal = torch.ones(8, 2, 4)
    gain = torch.full((8, 2, 4), 0.5)
    assert torch.equal(apply_gain(signal, gain), 0.5 * torch.ones(8, 2, 4))


def test_gain_scalar_broadcasts():
    """A scalar gain applies uniformly to every channel."""
    signal = torch.rand(8, 2, 4)
    assert torch.equal(apply_gain(signal, 3.0), 3.0 * signal)


def test_gain_per_channel():
    """Each channel scaled independently (off / suppress / pass / amplify)."""
    signal = torch.ones(1, 1, 4)
    gain = torch.tensor([[[0.0, 0.5, 1.0, 2.0]]])
    assert apply_gain(signal, gain).tolist() == [[[0.0, 0.5, 1.0, 2.0]]]


def test_gain_equivalent_to_open_mask():
    """g = 1 − gate_closed reproduces apply_gate (binary is the special case)."""
    signal = torch.rand(8, 2, 4)
    gate_closed = (torch.rand(8, 2, 4) > 0.5).float()
    assert torch.equal(apply_gain(signal, 1.0 - gate_closed), apply_gate(signal, gate_closed))


def test_gain_tensor_shape_mismatch_raises():
    with pytest.raises(ValueError):
        apply_gain(torch.ones(8, 2, 4), torch.ones(8, 2, 3))


def test_apply_gate_unchanged_signature():
    """Back-compat: the binary gate_closed call path is untouched."""
    signal = torch.ones(1, 1, 4)
    gate_closed = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    assert apply_gate(signal, gate_closed).tolist() == [[[0.0, 1.0, 0.0, 1.0]]]
