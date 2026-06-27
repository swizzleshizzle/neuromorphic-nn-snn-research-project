"""Random-goal generalization experiment for the v1 grid-world policy (ADR-0001).

Tests whether the frozen-brain + linear-head policy learned to navigate or merely
memorized the fixed goal: train on a subset of goal cells, evaluate on held-out cells.
The brain stays frozen; only the existing head trains.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from neuromorphic.envs.gridworld import GridWorldEnv, manhattan
from neuromorphic.training.reinforce import greedy_action


def split_goals(
    size: int, start: tuple[int, int], n_heldout: int, seed: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Partition all cells except ``start`` into (train, held_out), deterministic by seed."""
    candidates = [(x, y) for x in range(size) for y in range(size) if (x, y) != start]
    random.Random(seed).shuffle(candidates)
    held_out = candidates[:n_heldout]
    train = candidates[n_heldout:]
    return train, held_out


def optimality(start: tuple[int, int], goal: tuple[int, int], steps: int) -> float:
    """Fraction of optimal: shortest-path length / steps taken (0.0 if no steps)."""
    if steps <= 0:
        return 0.0
    return manhattan(start, goal) / steps


@dataclass
class EvalResult:
    success_rate: float
    mean_steps: float
    optimality: float
    n: int


def evaluate(
    brain,
    head,
    goals,
    *,
    size: int,
    start: tuple[int, int],
    max_steps: int,
    generator: "torch.Generator | None" = None,
) -> EvalResult:
    """Greedy rollouts from ``start`` to each goal; aggregate success, steps, optimality."""
    reached = 0
    steps_reached: list[int] = []
    opt_reached: list[float] = []
    for goal in goals:
        env = GridWorldEnv(size=size, start=start, goal=goal, max_steps=max_steps)
        obs, _ = env.reset()
        steps = 0
        done = False
        while steps < max_steps:
            with torch.no_grad():
                a = greedy_action(brain, head, obs, generator=generator)
            obs, _, term, trunc, _ = env.step(a)
            steps += 1
            if term:
                done = True
                break
            if trunc:
                break
        if done:
            reached += 1
            steps_reached.append(steps)
            opt_reached.append(optimality(start, goal, steps))
    n = len(goals)
    mean_steps = sum(steps_reached) / len(steps_reached) if steps_reached else 0.0
    mean_opt = sum(opt_reached) / len(opt_reached) if opt_reached else 0.0
    return EvalResult(success_rate=reached / n if n else 0.0, mean_steps=mean_steps, optimality=mean_opt, n=n)
