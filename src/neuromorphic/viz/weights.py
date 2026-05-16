"""Weight-matrix visualization: 1D histogram of values and 2D heatmap.

The two public functions share a small ``_prep_weight`` helper for shape
validation and symmetric color-limit computation. Keep them in this module
so the helper stays private.
"""

from __future__ import annotations

import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuromorphic.viz.core import _ensure_ax


def _prep_weight(W: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Validate a weight tensor and return ``(detached_cpu, abs_max)``.

    Used by both ``weight_histogram`` (for axis-limit hints) and
    ``weight_heatmap`` (for symmetric vmin/vmax around zero).
    """
    if W.ndim != 2:
        raise ValueError(
            f"weight tensor must be 2D [N_post, N_pre], got shape {tuple(W.shape)}"
        )
    W_cpu = W.detach().cpu()
    abs_max = float(W_cpu.abs().max().item())
    return W_cpu, abs_max


def weight_histogram(
    W: torch.Tensor,
    bins: int = 50,
    ax: Axes | None = None,
    log_y: bool = False,
) -> tuple[Figure, Axes]:
    """1D histogram of synaptic weight values.

    Use this to spot dead synapses (mass at 0), saturation (mass at bounds),
    or pre/post-training distribution shifts.

    Args:
        W: ``[N_post, N_pre]`` weight matrix.
        bins: number of histogram bins.
        ax: matplotlib Axes. If None, a fresh fig/ax is created.
        log_y: log-scale the y-axis (useful when most weights cluster).

    Returns:
        ``(fig, ax)``.
    """
    fig, ax = _ensure_ax(ax)
    W_cpu, abs_max = _prep_weight(W)
    ax.hist(W_cpu.flatten().numpy(), bins=bins, color="C0",
            edgecolor="black", linewidth=0.4)
    if log_y:
        ax.set_yscale("log")
    ax.axvline(x=0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count" + (" (log)" if log_y else ""))
    ax.set_title(
        f"Weight distribution — n={W_cpu.numel()} synapses, |w|max={abs_max:.3f}"
    )
    return fig, ax


def weight_heatmap(
    W: torch.Tensor,
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    center: float = 0.0,
) -> tuple[Figure, Axes]:
    """2D heatmap of a weight matrix with diverging colormap centered at ``center``.

    Red = positive (excitatory), blue = negative (inhibitory) under the default
    ``RdBu_r`` colormap. Color limits are symmetric around ``center`` so sign
    information is preserved.

    Args:
        W: ``[N_post, N_pre]`` weight matrix.
        ax: matplotlib Axes. If None, a fresh fig/ax is created.
        cmap: matplotlib colormap name. Diverging maps recommended.
        center: value that should map to the colormap's center color.

    Returns:
        ``(fig, ax)``.
    """
    fig, ax = _ensure_ax(ax)
    W_cpu, abs_max = _prep_weight(W)
    span = max(abs_max, 1e-12)
    im = ax.imshow(
        W_cpu.numpy(),
        cmap=cmap,
        vmin=center - span,
        vmax=center + span,
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Weight value")
    ax.set_xlabel("Presynaptic neuron")
    ax.set_ylabel("Postsynaptic neuron")
    ax.set_title(
        f"Weight matrix — {W_cpu.shape[0]} × {W_cpu.shape[1]}, |w|max={abs_max:.3f}"
    )
    return fig, ax
