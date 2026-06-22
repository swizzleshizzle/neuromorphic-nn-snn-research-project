"""Surrogate-gradient REINFORCE training for the five-region Brain (ADR-0001).

Motor spike-counts over the inference window are surrogate-differentiable, so they
form a categorical policy: sample an action, weight its log-probability by the
(baseline-subtracted) discounted return, and backprop through the spiking layers.
The memory path is bypassed (``recall=False``) — credit flows sensory → PFC → motor.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Returns-to-go: ``G_t = r_t + gamma * G_{t+1}`` for each step."""
    out: list[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    return out


def ema(old: float, new: float, beta: float) -> float:
    """Exponential moving average: ``(1 - beta) * old + beta * new``."""
    return (1.0 - beta) * old + beta * new


def policy_logits(out: dict) -> torch.Tensor:
    """Differentiable action logits = motor spike-counts over the window (single agent)."""
    return out["action_spikes"].sum(dim=0)[0]  # [T,B,A] -> [A]


def action_distribution(
    brain, obs, *, generator: torch.Generator | None = None
) -> tuple[Categorical, torch.Tensor]:
    """One forward pass → a categorical policy over actions (memory bypassed)."""
    out = brain.step(obs, store=False, recall=False, record=False, generator=generator)
    logits = policy_logits(out)
    return Categorical(logits=logits), logits
