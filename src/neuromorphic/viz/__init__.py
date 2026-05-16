"""Visualization toolkit for spiking neural networks.

All functions return ``(matplotlib.figure.Figure, matplotlib.axes.Axes)`` and
accept tensors in the canonical ``[T, B, N]`` shape. See
``docs/design/stage-1-visualization-toolkit.md``.
"""

from neuromorphic.viz.spikes import spike_raster

__all__ = ["spike_raster"]
