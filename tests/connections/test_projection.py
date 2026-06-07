"""Tests for the sparse random inter-region Projection (Phase 2, Step 2.2)."""

from __future__ import annotations

import pytest
import torch

from neuromorphic.connections.projection import Projection


def test_output_shape():
    """[T, B, N_src] -> [T, B, N_tgt]."""
    proj = Projection(n_source=20, n_target=8, sparsity=0.3, delay=1, seed=0)
    out = proj(torch.rand(32, 4, 20))
    assert out.shape == (32, 4, 8)


def test_mask_density_matches_sparsity():
    """Kept-connection fraction is approximately the requested sparsity."""
    proj = Projection(n_source=200, n_target=100, sparsity=0.1, seed=0)
    density = proj.mask.float().mean().item()
    assert abs(density - 0.1) < 0.02


def test_delay_shifts_output_in_time():
    """With the same seed, a delay-Δ projection is the delay-0 output shifted by Δ."""
    T, B = 32, 3
    x = torch.rand(T, B, 20)
    p0 = Projection(n_source=20, n_target=8, sparsity=0.5, delay=0, seed=7)
    pd = Projection(n_source=20, n_target=8, sparsity=0.5, delay=3, seed=7)
    o0, od = p0(x), pd(x)
    assert torch.count_nonzero(od[:3]) == 0           # front zero-padded
    assert torch.allclose(od[3:], o0[: T - 3])        # rest is the shifted signal


def test_delay_zero_is_passthrough_in_time():
    x = torch.rand(10, 2, 12)
    proj = Projection(n_source=12, n_target=5, sparsity=0.5, delay=0, seed=1)
    out = proj(x)
    assert out.shape == (10, 2, 5)
    assert torch.count_nonzero(out) > 0


def test_same_seed_is_deterministic():
    a = Projection(n_source=30, n_target=10, sparsity=0.4, seed=42)
    b = Projection(n_source=30, n_target=10, sparsity=0.4, seed=42)
    assert torch.equal(a.weight, b.weight)
    assert torch.equal(a.mask, b.mask)
    x = torch.rand(8, 2, 30)
    assert torch.equal(a(x), b(x))


def test_different_seed_differs():
    a = Projection(n_source=30, n_target=10, sparsity=0.4, seed=1)
    b = Projection(n_source=30, n_target=10, sparsity=0.4, seed=2)
    assert not torch.equal(a.weight, b.weight)


def test_masked_synapses_are_zero():
    """Every dropped connection has exactly zero weight."""
    proj = Projection(n_source=40, n_target=20, sparsity=0.25, seed=3)
    assert torch.all(proj.weight[~proj.mask] == 0)


def test_rejects_non_3d_input():
    proj = Projection(n_source=10, n_target=5, seed=0)
    with pytest.raises(ValueError):
        proj(torch.rand(10, 10))


def test_rejects_wrong_source_dim():
    proj = Projection(n_source=10, n_target=5, seed=0)
    with pytest.raises(ValueError):
        proj(torch.rand(8, 2, 11))


def test_rejects_delay_out_of_range():
    with pytest.raises(ValueError):
        Projection(n_source=10, n_target=5, delay=-1, seed=0)
