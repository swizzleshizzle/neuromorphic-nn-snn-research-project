"""Surrogate-gradient REINFORCE training for the five-region Brain (ADR-0001, amended).

The action policy is a small **learnable linear head** reading the **sensory concept**
(a rich, state-dependent code) → action logits. Week-12 debugging showed the original
"motor spike-counts ARE the logits" readout fails two ways: (1) summed spike-counts
saturate the softmax to an exact one-hot → a zero-gradient absorbing state that freezes
learning, and (2) the motor/PFC output is a degenerate "structural favourite" (one live
neuron, barely state-dependent), so it can't express a real four-action policy. A Linear
head on the concept gives every action a gradient handle and a learnable scale. The brain
is a fixed feature extractor in v1; memory stays bypassed (``recall=False``). See ADR-0001.
"""

from __future__ import annotations

import torch
import torch.nn as nn
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


def make_policy_head(brain, head_type: str = "linear", hidden: int = 128) -> nn.Module:
    """A trainable actor head: sensory concept (``brain.content`` dims) -> action logits.

    ``head_type="linear"`` is the v1 default (a single ``nn.Linear``); ``"mlp"`` adds one
    ReLU hidden layer of width ``hidden`` to test whether a nonlinear readout extracts more
    from the frozen sensory concept (EXP-025). The brain stays frozen either way.
    """
    if head_type == "linear":
        return nn.Linear(brain.content, brain.n_actions)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(brain.content, hidden),
            nn.ReLU(),
            nn.Linear(hidden, brain.n_actions),
        )
    raise ValueError(f"unknown head_type {head_type!r} (expected 'linear' or 'mlp')")


def concept_rate(out: dict) -> torch.Tensor:
    """Mean firing rate of the sensory concept over the window, single agent. -> [concept]."""
    return out["concept"].mean(dim=0)[0]


# Backward-compatible alias (older callers import the private name).
_concept_rate = concept_rate


def action_distribution(
    brain, head: nn.Module, obs, *, generator: torch.Generator | None = None
) -> tuple[Categorical, torch.Tensor]:
    """One forward pass → a categorical policy from the head on the sensory concept.

    The brain runs under ``no_grad`` — it is a frozen feature extractor in v1, so only
    the head carries gradient (also avoids backprop through the spiking unroll).
    """
    with torch.no_grad():
        out = brain.step(obs, store=False, recall=False, record=False, generator=generator)
    logits = head(_concept_rate(out))
    return Categorical(logits=logits), logits


def greedy_action(
    brain, head: nn.Module, obs, *, generator: torch.Generator | None = None
) -> int:
    """The argmax-logit action (deterministic eval policy)."""
    _, logits = action_distribution(brain, head, obs, generator=generator)
    return int(logits.argmax())


def policy_parameters(head: nn.Module):
    """Trainable parameters of the policy. v1: the head only — the brain is frozen."""
    return head.parameters()


def train_episode(
    brain,
    head: nn.Module,
    env,
    optimizer,
    *,
    gamma: float = 0.99,
    baseline: float = 0.0,
    generator: torch.Generator | None = None,
    max_steps: int | None = None,
    entropy_beta: float = 0.0,
    normalize_advantages: bool = False,
) -> dict:
    """Run one episode, then apply one REINFORCE update to the head. Memory bypassed.

    ``entropy_beta`` > 0 adds an optional ``-beta * sum_t H`` entropy bonus to the loss
    (encourages exploration); the default 0.0 leaves the executed loss statement unchanged.
    ``normalize_advantages`` standardizes advantages per episode (zero-mean/unit-scale,
    population std to stay finite on 1-step episodes) so the entropy bonus and gradient are
    on a predictable scale; the default False leaves the executed statements unchanged.

    Returns stats: ``steps``, ``total_reward`` (undiscounted), ``mean_return``
    (mean discounted return-to-go, for baseline tracking), ``loss``, ``reached_goal``.
    """
    obs, _ = env.reset()
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    entropies: list[torch.Tensor] = []
    reached_goal = False
    limit = max_steps if max_steps is not None else getattr(env, "max_steps", 100)

    steps = 0
    while steps < limit:
        dist, _ = action_distribution(brain, head, obs, generator=generator)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
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
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    loss = -(torch.stack(log_probs) * advantages).sum()
    if entropy_beta:
        loss = loss - entropy_beta * torch.stack(entropies).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "steps": steps,
        "total_reward": float(sum(rewards)),
        "mean_return": float(returns.mean()),
        "loss": float(loss.detach()),
        "reached_goal": reached_goal,
        "mean_entropy": float(torch.stack(entropies).mean()) if entropies else 0.0,
    }
