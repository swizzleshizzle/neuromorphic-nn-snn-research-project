"""Membrane-potential trace plotting.

Implemented from scratch rather than wrapping ``snntorch.spikeplot.traces``
because the latter does not accept an ``ax`` argument and uses ``GridSpec``
internally — incompatible with the toolkit's ``(fig, ax)`` contract.
"""

from __future__ import annotations

import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuromorphic.viz.core import _ensure_ax, _select_sample


def membrane_trace(
    mem: torch.Tensor,
    spk: torch.Tensor | None = None,
    ax: Axes | None = None,
    sample_idx: int = 0,
    threshold: float = 1.0,
    show_threshold: bool = True,
    spk_height: float = 1.5,
) -> tuple[Figure, Axes]:
    """Line plot of membrane potential ``u_i(t)`` for each neuron in a layer.

    Args:
        mem: ``[T, B, N]`` or ``[T, N]`` membrane potentials.
        spk: optional ``[T, B, N]`` or ``[T, N]`` spike train. When provided,
            ``spk_height`` is added to ``mem`` wherever the neuron spiked, so
            spikes "ride on" the trace (Friedemann Zenke style).
        ax: matplotlib Axes. If None, a fresh fig/ax is created.
        sample_idx: which batch element to plot.
        threshold: y-value at which to draw the (optional) threshold line.
        show_threshold: if True, draw a dashed horizontal line at ``threshold``.
        spk_height: vertical bump applied where ``spk == 1`` (if ``spk`` given).

    Returns:
        ``(fig, ax)``.
    """
    fig, ax = _ensure_ax(ax)
    mem_2d = _select_sample(mem, sample_idx).detach().cpu()

    if spk is not None:
        spk_2d = _select_sample(spk, sample_idx).detach().cpu()
        data = mem_2d + spk_height * spk_2d
    else:
        data = mem_2d

    T, N = data.shape
    time = torch.arange(T)
    for n in range(N):
        ax.plot(time, data[:, n], linewidth=0.8, alpha=0.7)

    if show_threshold:
        ax.axhline(y=threshold, color="black", linestyle="--",
                   linewidth=1.0, alpha=0.6, label=f"threshold = {threshold}")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Time step")
    ax.set_ylabel(r"Membrane potential $U[t]$")
    ax.grid(alpha=0.3)
    return fig, ax
