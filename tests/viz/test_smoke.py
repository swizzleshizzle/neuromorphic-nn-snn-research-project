"""Smoke tests for neuromorphic.viz. Each public function gets one test
asserting it returns a (Figure, Axes) tuple without raising."""

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ---------- core helpers ----------

def test_ensure_ax_creates_new_when_none():
    from neuromorphic.viz.core import _ensure_ax
    fig, ax = _ensure_ax(None)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_ensure_ax_reuses_existing():
    from neuromorphic.viz.core import _ensure_ax
    fig, ax = plt.subplots()
    fig2, ax2 = _ensure_ax(ax)
    assert ax2 is ax
    assert fig2 is fig
    plt.close(fig)


def test_select_sample_indexes_3d():
    from neuromorphic.viz.core import _select_sample
    t = torch.zeros(20, 4, 50)
    t[:, 2, :] = 1.0
    out = _select_sample(t, sample_idx=2)
    assert out.shape == (20, 50)
    assert (out == 1.0).all()


def test_select_sample_passes_2d_through():
    from neuromorphic.viz.core import _select_sample
    t = torch.zeros(20, 50)
    out = _select_sample(t, sample_idx=0)
    assert out.shape == (20, 50)
    assert out is t  # no copy


def test_select_sample_rejects_1d():
    from neuromorphic.viz.core import _select_sample
    with pytest.raises(ValueError, match="2D or 3D"):
        _select_sample(torch.zeros(20), sample_idx=0)


# ---------- spike_raster ----------

def test_spike_raster_returns_fig_ax(spk):
    from neuromorphic.viz import spike_raster
    fig, ax = spike_raster(spk)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_spike_raster_respects_provided_ax(spk):
    from neuromorphic.viz import spike_raster
    fig, ax = plt.subplots()
    fig2, ax2 = spike_raster(spk, ax=ax)
    assert ax2 is ax
    assert fig2 is fig
    plt.close(fig)
