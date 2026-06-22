"""Surrogate-gradient REINFORCE training for the five-region Brain (ADR-0001).

Motor spike-counts over the inference window are surrogate-differentiable, so they
form a categorical policy: sample an action, weight its log-probability by the
(baseline-subtracted) discounted return, and backprop through the spiking layers.
The memory path is bypassed (``recall=False``) — credit flows sensory → PFC → motor.
"""

from __future__ import annotations

import itertools

import torch
from torch.distributions import Categorical


def policy_parameters(brain):
    """Learnable params on the differentiable policy path: sensory → PFC → motor.

    ``Brain`` is a plain orchestrator (not an ``nn.Module``), so gather the trainable
    parameters of the regions credit actually flows through. The hippocampus (memory,
    bypassed via ``recall=False``) and the structural lateral-inhibition buffers are
    deliberately excluded — see ADR-0001.
    """
    return itertools.chain(
        brain.sensory.parameters(),
        brain.pfc.parameters(),
        brain.motor.parameters(),
    )


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


def train_episode(
    brain,
    env,
    optimizer,
    *,
    gamma: float = 0.99,
    baseline: float = 0.0,
    generator: torch.Generator | None = None,
    max_steps: int | None = None,
) -> dict:
    """Run one episode, then apply one REINFORCE update. Memory bypassed (recall=False).

    Returns stats: ``steps``, ``total_reward`` (undiscounted), ``mean_return``
    (mean discounted return-to-go, for baseline tracking), ``loss``, ``reached_goal``.
    """
    obs, _ = env.reset()
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    reached_goal = False
    limit = max_steps if max_steps is not None else getattr(env, "max_steps", 100)

    steps = 0
    while steps < limit:
        dist, _ = action_distribution(brain, obs, generator=generator)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        obs, reward, terminated, truncated, _ = env.step(int(action))
        rewards.append(float(reward))
        steps += 1
        if terminated:
            reached_goal = True
            break
        if truncated:
            break

    returns = torch.tensor(discounted_returns(rewards, gamma), dtype=torch.float32)
    advantages = returns - baseline
    loss = -(torch.stack(log_probs) * advantages).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "steps": steps,
        "total_reward": float(sum(rewards)),
        "mean_return": float(returns.mean()),
        "loss": float(loss.detach()),
        "reached_goal": reached_goal,
    }
