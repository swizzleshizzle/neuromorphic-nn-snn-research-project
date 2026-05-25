"""Models for sequential MNIST (exp 011).

Contains:
- ``row_encode``: pure helper that reshapes an MNIST batch into a
  ``[T=28, B, 28]`` sequence, one row per timestep.
- ``SequentialSNN``: added in Task 3.
"""

from __future__ import annotations

import torch


def row_encode(images: torch.Tensor) -> torch.Tensor:
    """Reshape an MNIST batch into a row-major time sequence.

    Args:
        images: ``[B, 1, 28, 28]`` or ``[B, 28, 28]`` tensor of pixel values.

    Returns:
        ``[28, B, 28]`` -- row index becomes time index. Pure function: no
        spike sampling, no normalization, no allocation beyond reshape.
    """
    if images.ndim == 4:
        # [B, 1, 28, 28] -> [B, 28, 28]
        images = images.squeeze(1)
    if images.ndim != 3 or images.shape[-2:] != (28, 28):
        raise ValueError(
            f"row_encode expects [B, 1, 28, 28] or [B, 28, 28], got {tuple(images.shape)}"
        )
    # [B, 28, 28] -> [28, B, 28]
    return images.permute(1, 0, 2).contiguous()
