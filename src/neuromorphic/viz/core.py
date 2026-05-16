"""Shared helpers used by every viz function.

These exist so that every public function can:
- accept ``ax=None`` and produce a fresh fig/ax on demand
- accept either ``[T, B, N]`` or ``[T, N]`` for single-sample plots
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _ensure_ax(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``; create a new pair if ``ax`` is None."""
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.figure, ax


def _select_sample(tensor: torch.Tensor, sample_idx: int = 0) -> torch.Tensor:
    """Reduce a ``[T, B, N]`` tensor to ``[T, N]`` by picking one sample.

    A ``[T, N]`` tensor is returned unchanged. Other ranks raise ``ValueError``.
    """
    if tensor.ndim == 3:
        return tensor[:, sample_idx, :]
    if tensor.ndim == 2:
        return tensor
    raise ValueError(
        f"_select_sample expects a 2D or 3D tensor, got shape {tuple(tensor.shape)}"
    )
