"""Spike-train plotting: rasters, PSTH, population rate."""

from __future__ import annotations

import numpy as np
import snntorch.spikeplot as splt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuromorphic.viz.core import _ensure_ax, _select_sample


def spike_raster(
    spk: torch.Tensor,
    ax: Axes | None = None,
    sample_idx: int = 0,
    **scatter_kwargs,
) -> tuple[Figure, Axes]:
    """Spike raster: one dot per (time, neuron) spike.

    Wraps ``snntorch.spikeplot.raster``, which calls ``ax.scatter`` internally.

    Args:
        spk: ``[T, B, N]`` or ``[T, N]``. If 3D, sample ``sample_idx`` is shown.
        ax: matplotlib Axes. If None, a fresh fig/ax is created.
        sample_idx: which batch element to plot (ignored if ``spk`` is 2D).
        scatter_kwargs: forwarded to ``ax.scatter`` via ``splt.raster``
            (e.g. ``s=1.5``, ``c="black"``).

    Returns:
        ``(fig, ax)``. The caller decides whether to show, save, or embed.
    """
    fig, ax = _ensure_ax(ax)
    spk_2d = _select_sample(spk, sample_idx)
    # splt.raster signature: raster(data, ax, **kwargs); data shape [T, N].
    scatter_kwargs.setdefault("s", 2)
    scatter_kwargs.setdefault("c", "black")
    splt.raster(spk_2d, ax, **scatter_kwargs)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Neuron index")
    return fig, ax


def population_rate(
    spk: torch.Tensor,
    smoothing_window: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Mean firing rate of an entire layer over time.

    For each time step, compute the fraction of (batch, neuron) pairs that
    spiked. Plot as a single line over time.

    Use this as the load-bearing diagnostic for layer-level activity (see
    parent design doc §3.4): saturation → broken inhibition; collapse to
    zero → dead layer.

    Args:
        spk: ``[T, B, N]`` spike tensor.
        smoothing_window: optional rolling-mean width (in time steps).
        ax: matplotlib Axes. If None, a fresh fig/ax is created.

    Returns:
        ``(fig, ax)``.
    """
    if spk.ndim != 3:
        raise ValueError(
            f"population_rate expects a [T, B, N] tensor, got shape {tuple(spk.shape)}"
        )
    fig, ax = _ensure_ax(ax)
    rate = spk.float().mean(dim=(1, 2)).detach().cpu().numpy()

    if smoothing_window is not None and smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        rate = np.convolve(rate, kernel, mode="valid")

    x = np.arange(len(rate))
    ax.plot(x, rate, color="C0", linewidth=1.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean firing rate (spikes / neuron / step)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    return fig, ax
