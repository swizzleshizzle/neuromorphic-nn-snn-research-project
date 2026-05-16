"""Smoke tests for neuromorphic.viz. Each public function gets one test
asserting it returns a (Figure, Axes) tuple without raising."""

import matplotlib.pyplot as plt
import numpy as np
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


# ---------- membrane_trace ----------

def test_membrane_trace_returns_fig_ax(mem):
    from neuromorphic.viz import membrane_trace
    fig, ax = membrane_trace(mem)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_membrane_trace_draws_threshold_line(mem):
    from neuromorphic.viz import membrane_trace
    fig, ax = membrane_trace(mem, threshold=1.0, show_threshold=True)
    # axhline adds a Line2D; check at least one line is at y=1.0.
    # matplotlib 3.10's axhline returns ydata as a Python list, not ndarray —
    # wrap in np.asarray so .size / element comparison work uniformly.
    threshold_lines = [
        ln for ln in ax.get_lines()
        if (yd := np.asarray(ln.get_ydata())).size == 2 and (yd == 1.0).all()
    ]
    assert len(threshold_lines) >= 1
    plt.close(fig)


def test_membrane_trace_omits_threshold_when_disabled(mem):
    from neuromorphic.viz import membrane_trace
    fig, ax = membrane_trace(mem, threshold=1.0, show_threshold=False)
    threshold_lines = [
        ln for ln in ax.get_lines()
        if (yd := np.asarray(ln.get_ydata())).size == 2 and (yd == 1.0).all()
    ]
    assert len(threshold_lines) == 0
    plt.close(fig)


# ---------- weights ----------

def test_weight_histogram_returns_fig_ax(W):
    from neuromorphic.viz import weight_histogram
    fig, ax = weight_histogram(W)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_weight_histogram_bins_count(W):
    from neuromorphic.viz import weight_histogram
    fig, ax = weight_histogram(W, bins=20)
    # ax.hist returns BarContainer; count its patches.
    assert len(ax.patches) == 20
    plt.close(fig)


def test_weight_heatmap_returns_fig_ax(W):
    from neuromorphic.viz import weight_heatmap
    fig, ax = weight_heatmap(W)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_weight_heatmap_uses_symmetric_color_limits(W):
    from neuromorphic.viz import weight_heatmap
    fig, ax = weight_heatmap(W, center=0.0)
    images = ax.get_images()
    assert len(images) == 1
    vmin, vmax = images[0].get_clim()
    assert abs(vmin + vmax) < 1e-6, f"expected symmetric clim, got ({vmin}, {vmax})"
    plt.close(fig)


# ---------- training_curve ----------

def test_training_curve_returns_fig_ax(history):
    from neuromorphic.viz import training_curve
    fig, ax = training_curve(history, log_interval=10)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_training_curve_creates_twin_axis_when_acc_present(history):
    from neuromorphic.viz import training_curve
    fig, ax = training_curve(history, log_interval=10)
    # A twin y-axis shares the same x-axis but has its own ylim.
    twins = [a for a in fig.axes if a is not ax]
    assert len(twins) >= 1, "expected a twin axis for accuracy series"
    plt.close(fig)


def test_training_curve_skips_missing_series():
    from neuromorphic.viz import training_curve
    only_loss = {"train_loss": [1.0, 0.5, 0.3]}
    fig, ax = training_curve(only_loss)
    twins = [a for a in fig.axes if a is not ax]
    assert len(twins) == 0, "no acc keys → no twin axis"
    plt.close(fig)


def test_training_curve_warns_on_unknown_key():
    import warnings
    from neuromorphic.viz import training_curve
    weird = {"train_loss": [1.0, 0.5], "mystery_metric": [0.2, 0.4]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, _ = training_curve(weird)
        assert any("mystery_metric" in str(w.message) for w in caught)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------- population_rate ----------

def test_population_rate_returns_fig_ax(spk):
    from neuromorphic.viz import population_rate
    fig, ax = population_rate(spk)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_population_rate_line_length_matches_T(spk):
    from neuromorphic.viz import population_rate
    fig, ax = population_rate(spk)
    lines = ax.get_lines()
    assert len(lines) == 1
    assert lines[0].get_xdata().size == spk.shape[0]
    plt.close(fig)


def test_population_rate_smoothing_shrinks_line(spk):
    from neuromorphic.viz import population_rate
    fig, ax = population_rate(spk, smoothing_window=5)
    lines = ax.get_lines()
    # Valid convolution: length T - window + 1.
    assert lines[0].get_xdata().size == spk.shape[0] - 5 + 1
    plt.close(fig)
