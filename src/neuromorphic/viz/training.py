"""Training history plotting.

``training_curve`` takes a flexible ``{name: [values]}`` dict so the same
function works for loss-only runs and richer comparisons.
"""

from __future__ import annotations

import warnings

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuromorphic.viz.core import _ensure_ax


def _is_loss_key(name: str) -> bool:
    return "loss" in name.lower()


def _is_acc_key(name: str) -> bool:
    n = name.lower()
    return "acc" in n or "accuracy" in n


def _smooth(values: list[float], window: int) -> np.ndarray:
    """Centered rolling-mean smoothing. ``window=None`` or ``<2`` returns input."""
    if window is None or window < 2:
        return np.asarray(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def training_curve(
    history: dict[str, list[float]],
    log_interval: int = 1,
    smoothing_window: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot training-history series.

    Keys containing ``loss`` go on the left y-axis. Keys containing ``acc`` or
    ``accuracy`` go on a twin right y-axis with a fixed [0, 1] range. Keys
    matching neither prefix go on the left axis with a ``warnings.warn``.

    X-axis units are training iterations. Series whose name starts with
    ``train_`` are assumed to be sampled every iteration. All other series are
    assumed to be sampled every ``log_interval`` iterations.

    Args:
        history: ``{series_name: list_of_values}``. Empty or missing keys
            are silently skipped — only what's in ``history`` is drawn.
        log_interval: spacing (in iterations) of non-``train_`` series.
        smoothing_window: optional rolling-mean window applied to every series.
        ax: matplotlib Axes. If None, a fresh fig/ax is created.

    Returns:
        ``(fig, ax)`` where ``ax`` is the left (loss) axis. The accuracy twin
        axis, if any, is accessible via ``fig.axes``.
    """
    fig, ax = _ensure_ax(ax)
    twin = None
    color_cycle_loss = ["C0", "C1", "C4", "C5"]
    color_cycle_acc = ["C2", "C3", "C6", "C7"]
    loss_i = acc_i = 0

    for name, values in history.items():
        if not values:
            continue
        ys = _smooth(values, smoothing_window)
        n = len(ys)
        stride = 1 if name.startswith("train_") else log_interval
        xs = np.arange(n) * stride

        if _is_loss_key(name):
            ax.plot(xs, ys, label=name, color=color_cycle_loss[loss_i % len(color_cycle_loss)])
            loss_i += 1
        elif _is_acc_key(name):
            if twin is None:
                twin = ax.twinx()
                twin.set_ylabel("Accuracy")
                twin.set_ylim(0.0, 1.0)
            twin.plot(xs, ys, label=name, linestyle="--",
                      color=color_cycle_acc[acc_i % len(color_cycle_acc)])
            acc_i += 1
        else:
            warnings.warn(
                f"training_curve: key '{name}' matches neither 'loss' nor 'acc' "
                "prefixes; plotting on the loss axis."
            )
            ax.plot(xs, ys, label=name, alpha=0.6)

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)

    # Single combined legend across both axes.
    handles, labels = ax.get_legend_handles_labels()
    if twin is not None:
        h2, l2 = twin.get_legend_handles_labels()
        handles += h2
        labels += l2
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8)

    return fig, ax
