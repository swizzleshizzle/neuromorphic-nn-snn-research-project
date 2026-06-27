"""Random-goal generalization experiment for the v1 grid-world policy (ADR-0001).

Tests whether the frozen-brain + linear-head policy learned to navigate or merely
memorized the fixed goal: train on a subset of goal cells, evaluate on held-out cells.
The brain stays frozen; only the existing head trains.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from neuromorphic.envs.gridworld import manhattan


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
