"""Supervised pre-training for the sensory encoder (EXP-026).

Pre-trains ``SensoryCortex`` so its concept code linearly exposes goal-relative
displacement ``(gx-ax, gy-ay)``. A scratch ``Linear(concept -> 2)`` readout shapes the
encoder via backprop through the spiking hierarchy; the readout is discarded afterward.
The encoder is then frozen for the RL policy (ADR-0001 Amendment 2 follow-up).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neuromorphic.regions.sensory_cortex import encode_gridworld


def displacement_target(obs: torch.Tensor, grid_n: int) -> torch.Tensor:
    """``[B, 4]`` obs (ax, ay, gx, gy) -> ``[B, 2]`` normalized (gx-ax, gy-ay) / (grid_n-1)."""
    ax, ay, gx, gy = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
    disp = torch.stack([gx - ax, gy - ay], dim=1).float()
    return disp / (grid_n - 1)


def enumerate_states(grid_n: int) -> torch.Tensor:
    """All (agent, goal) cell pairs with agent != goal -> ``[M, 4]`` long tensor."""
    cells = [(x, y) for x in range(grid_n) for y in range(grid_n)]
    rows = [
        [ax, ay, gx, gy]
        for (ax, ay) in cells
        for (gx, gy) in cells
        if (ax, ay) != (gx, gy)
    ]
    return torch.tensor(rows, dtype=torch.long)


def split_states(states: torch.Tensor, frac_heldout: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic (train, heldout) row split of ``states`` by ``seed``."""
    n = states.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_held = round(n * frac_heldout)
    return states[perm[n_held:]], states[perm[:n_held]]
