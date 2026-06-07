"""``Projection`` — a sparse, delayed connection between two regions.

A projection routes weighted **current** from a source population to a target
population: input spikes ``[T, B, N_src]`` → current ``[T, B, N_tgt]``. It never
produces spikes itself — the consuming region's neurons convert current → spikes.

Connectivity is a fixed seeded sparse random mask over the ``[N_tgt, N_src]`` weight
matrix (matching the repo weight convention ``[N_post, N_pre]``). A signal at time
``t`` reaches the target at ``t + Δ``, where ``Δ`` is the integer transmission delay
(spec §5.5 — placeholder values until ring-buffers are measured).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Projection(nn.Module):
    """Sparse random delayed projection.

    Args:
        n_source: source population size ``N_src``.
        n_target: target population size ``N_tgt``.
        sparsity: fraction of connections kept (density ``p`` in ``[0, 1]``).
        delay: transmission delay ``Δ`` in ``T``-steps; requires ``0 <= Δ``.
        weight_scale: standard deviation of the ``N(0, scale)`` weight init.
        seed: RNG seed for reproducible mask + weights (``None`` = global RNG).

    Buffers:
        weight: ``[N_tgt, N_src]`` masked weight matrix (dropped synapses are 0).
        mask: ``[N_tgt, N_src]`` boolean kept-connection mask.
    """

    def __init__(
        self,
        n_source: int,
        n_target: int,
        sparsity: float = 0.1,
        delay: int = 1,
        weight_scale: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")

        self.n_source = n_source
        self.n_target = n_target
        self.delay = delay

        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)

        mask = torch.rand(n_target, n_source, generator=gen) < sparsity
        weight = torch.randn(n_target, n_source, generator=gen) * weight_scale
        weight = weight * mask  # drop masked synapses to exactly 0

        self.register_buffer("mask", mask)
        self.register_buffer("weight", weight)

    def forward(self, src_spikes: torch.Tensor) -> torch.Tensor:
        """``src_spikes`` ``[T, B, N_src]`` → current ``[T, B, N_tgt]`` (delayed)."""
        if src_spikes.ndim != 3:
            raise ValueError(
                f"Projection expects a [T, B, N_src] tensor, got shape "
                f"{tuple(src_spikes.shape)}"
            )
        if src_spikes.shape[-1] != self.n_source:
            raise ValueError(
                f"expected last dim N_src={self.n_source}, got {src_spikes.shape[-1]}"
            )

        # [T, B, N_src] @ [N_src, N_tgt] -> [T, B, N_tgt]
        current = src_spikes @ self.weight.t()

        if self.delay == 0:
            return current
        out = torch.zeros_like(current)
        out[self.delay :] = current[: current.shape[0] - self.delay]
        return out
