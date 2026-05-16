"""Spike-train plotting: rasters, PSTH, population rate."""

from __future__ import annotations

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
