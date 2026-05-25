"""Spatial Poisson pattern generator for EXP-012 STDP-WTA demo.

Three fixed spatial patterns over 12 input neurons; each pattern designates
4 'active' inputs that fire at active_rate_hz, others at background_rate_hz.
"""

from __future__ import annotations

import torch

PATTERNS: dict[str, list[int]] = {
    "A": [0, 1, 2, 3],
    "B": [4, 5, 6, 7],
    "C": [8, 9, 10, 11],
}
PATTERN_NAMES = list(PATTERNS.keys())  # ["A", "B", "C"]


def poisson_pattern_spikes(
    pattern_idx: int,
    num_steps: int,
    n_inputs: int = 12,
    active_rate_hz: float = 100.0,
    background_rate_hz: float = 5.0,
    dt_s: float = 0.0025,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Bernoulli-sample a [num_steps, n_inputs] spike tensor for one pattern."""
    name = PATTERN_NAMES[pattern_idx]
    active = PATTERNS[name]

    p = torch.full((n_inputs,), background_rate_hz * dt_s)
    for i in active:
        p[i] = active_rate_hz * dt_s

    p_grid = p.unsqueeze(0).expand(num_steps, n_inputs)
    if generator is None:
        return torch.bernoulli(p_grid)
    return torch.bernoulli(p_grid, generator=generator)


def background_only_spikes(
    num_steps: int,
    n_inputs: int = 12,
    background_rate_hz: float = 5.0,
    dt_s: float = 0.0025,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Inter-trial gap: all inputs at background rate."""
    p = torch.full((num_steps, n_inputs), background_rate_hz * dt_s)
    if generator is None:
        return torch.bernoulli(p)
    return torch.bernoulli(p, generator=generator)
