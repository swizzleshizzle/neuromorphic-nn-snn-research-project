"""Surrogate-gradient REINFORCE training for the five-region Brain (ADR-0001, amended).

The action policy is a small **learnable linear head** reading the **sensory concept**
(a rich, state-dependent code) → action logits. Week-12 debugging showed the original
"motor spike-counts ARE the logits" readout fails two ways: (1) summed spike-counts
saturate the softmax to an exact one-hot → a zero-gradient absorbing state that freezes
learning, and (2) the motor/PFC output is a degenerate "structural favourite" (one live
neuron, barely state-dependent), so it can't express a real four-action policy. A Linear
head on the concept gives every action a gradient handle and a learnable scale. The brain
is a fixed feature extractor in v1. ``action_distribution``/``greedy_action``/``train_episode``
accept ``store``/``recall``/``feature_fn`` so callers can engage the hippocampal pathway
and read memory-derived features (EXP-030); the defaults (``store=False``, ``recall=False``,
``feature_fn=None``) still reproduce the v1 concept-only, memory-bypassed path exactly.
See ADR-0001 and its amendments.
"""

from __future__ import annotations

import contextlib

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
    brain,
    head: nn.Module,
    obs,
    *,
    generator: torch.Generator | None = None,
    store: bool = False,
    recall: bool = False,
    feature_fn=None,
    grad_brain: bool = False,
) -> tuple[Categorical, torch.Tensor]:
    """One forward pass -> a categorical policy from the head on the chosen features.

    The brain runs under ``no_grad`` by default (frozen feature extractor; also avoids
    backprop through the spiking unroll). ``feature_fn`` selects what the head reads and
    defaults to the sensory concept, so omitting it reproduces v1 exactly.
    ``store``/``recall`` engage the hippocampal pathway (both off by default, as in v1).

    ``grad_brain=True`` (EXP-047) drops the ``no_grad`` so gradients reach the sensory
    encoder through snnTorch's surrogate, making the encoder fine-tunable during RL. It
    changes **nothing** about the forward: autograd does not alter the arithmetic and does
    not touch the Poisson generator, so the concept, the action and the whole RNG stream are
    bit-identical either way. That is what lets ``encoder_lr=0.0`` reproduce a frozen run
    byte-for-byte, and it is asserted in ``tests/training/test_encoder_finetune_seam.py``.

    Note this is a no-op unless the caller ALSO puts the encoder in the optimizer, and that
    ``feature_fn`` must not re-wrap the concept in ``no_grad`` - ``MemoryReadout`` does, which
    is why EXP-047 is a ``readout="concept"`` experiment (feature_fn is None there).
    """
    with contextlib.nullcontext() if grad_brain else torch.no_grad():
        out = brain.step(obs, store=store, recall=recall, record=False, generator=generator)
    features = concept_rate(out) if feature_fn is None else feature_fn(out)
    logits = head(features)
    return Categorical(logits=logits), logits


def greedy_action(
    brain,
    head: nn.Module,
    obs,
    *,
    generator: torch.Generator | None = None,
    store: bool = False,
    recall: bool = False,
    feature_fn=None,
) -> int:
    """The argmax-logit action (deterministic eval policy).

    No ``grad_brain`` passthrough, deliberately: evaluation never needs a graph, and building
    one over every held-out state would cost memory for nothing.
    """
    _, logits = action_distribution(
        brain, head, obs, generator=generator,
        store=store, recall=recall, feature_fn=feature_fn,
    )
    return int(logits.argmax())


def policy_parameters(head: nn.Module):
    """Trainable parameters of the policy. v1: the head only — the brain is frozen.

    EXP-047 does NOT change this. A fine-tuning caller builds explicit Adam parameter groups
    instead, so the head's learning rate and the encoder's stay separate quantities and this
    function keeps meaning exactly what every prior experiment recorded it as meaning.
    """
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
    store: bool = False,
    recall: bool = False,
    feature_fn=None,
    grad_brain: bool = False,
) -> dict:
    """Run one episode, then apply one REINFORCE update to the head.

    Memory is bypassed by default (``store=False``, ``recall=False``, ``feature_fn=None``,
    which reproduces the v1 concept-only path exactly); pass ``store=True``/``recall=True``
    with a memory-aware ``feature_fn`` to engage the hippocampal pathway instead (EXP-030).

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
        dist, _ = action_distribution(
            brain, head, obs, generator=generator,
            store=store, recall=recall, feature_fn=feature_fn, grad_brain=grad_brain,
        )
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
