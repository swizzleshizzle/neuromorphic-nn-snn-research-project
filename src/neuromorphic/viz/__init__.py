"""Visualization toolkit for spiking neural networks.

All functions return ``(matplotlib.figure.Figure, matplotlib.axes.Axes)`` and
accept tensors in the canonical ``[T, B, N]`` shape. See
``docs/design/stage-1-visualization-toolkit.md``.
"""

from neuromorphic.viz.membrane import membrane_trace
from neuromorphic.viz.spikes import population_rate, psth, spike_raster
from neuromorphic.viz.training import training_curve
from neuromorphic.viz.weights import weight_heatmap, weight_histogram

__all__ = [
    "spike_raster",
    "membrane_trace",
    "weight_histogram",
    "weight_heatmap",
    "training_curve",
    "population_rate",
    "psth",
]
