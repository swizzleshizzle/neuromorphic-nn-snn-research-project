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
    """``[B, 4]`` obs (ax, ay, gx, gy) -> ``[B, 2]`` normalized (gx-ax, gy-ay) / (grid_n-1).

    Requires ``grid_n >= 2`` (normalization divides by ``grid_n - 1``).
    """
    if grid_n < 2:
        raise ValueError(f"grid_n must be >= 2 for displacement normalization, got {grid_n}")
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


def concept_rate_batch(sensory, obs, grid_n, T: int = 32, generator=None) -> torch.Tensor:
    """Encode ``[B, 4]`` obs -> mean-over-T concept rate ``[B, concept]``, differentiable.

    Calls the encoder directly (NOT via ``brain.step``, which wraps it in ``no_grad``) so
    gradients reach ``sensory.fc1`` / ``fc2`` during pre-training.
    """
    spikes = encode_gridworld(obs, grid_n, T=T, generator=generator)   # [T, B, N_obs]
    concept = sensory(spikes)                                          # [T, B, concept]
    return concept.mean(dim=0)                                         # [B, concept]


def _disp_mae(readout, sensory, obs, grid_n, T, generator) -> float:
    with torch.no_grad():
        pred = readout(concept_rate_batch(sensory, obs, grid_n, T=T, generator=generator))
        return float((pred - displacement_target(obs, grid_n)).abs().mean())


def pretrain_sensory(
    sensory, *, grid_n, epochs: int = 200, lr: float = 1e-3, frac_heldout: float = 0.2,
    seed: int = 0, T: int = 32, generator=None, freeze_encoder: bool = False,
) -> dict:
    """Pre-train ``sensory`` so its concept linearly decodes goal-relative displacement.

    Trains a scratch ``Linear(concept -> 2)`` readout (and the encoder unless
    ``freeze_encoder``) with MSE + Adam over the train state split; reports mean-absolute
    displacement error on the train and held-out splits. The readout is discarded.
    """
    torch.manual_seed(seed)
    states = enumerate_states(grid_n)
    train_states, heldout_states = split_states(states, frac_heldout, seed)

    readout = nn.Linear(sensory.concept, 2)
    params = list(readout.parameters())
    if not freeze_encoder:
        params += list(sensory.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    for _ in range(epochs):
        rate = concept_rate_batch(sensory, train_states, grid_n, T=T, generator=generator)
        pred = readout(rate)
        loss = ((pred - displacement_target(train_states, grid_n)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    return {
        "train_disp_error": _disp_mae(readout, sensory, train_states, grid_n, T, generator),
        "heldout_disp_error": _disp_mae(readout, sensory, heldout_states, grid_n, T, generator),
        "epochs": epochs,
        "freeze_encoder": freeze_encoder,
    }
