"""Random-goal generalization experiment for the v1 grid-world policy (ADR-0001).

Tests whether the frozen-brain + linear-head policy learned to navigate or merely
memorized the fixed goal: train on a subset of goal cells, evaluate on held-out cells.
The brain stays frozen; only the existing head trains.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from neuromorphic.brain import Brain
from neuromorphic.envs.gridworld import GridWorldEnv, manhattan
from neuromorphic.training.pretrain import pretrain_sensory
from neuromorphic.training.reinforce import (
    ema,
    greedy_action,
    make_policy_head,
    policy_parameters,
    train_episode,
)


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


@dataclass
class GenConfig:
    seed: int = 0
    episodes: int = 600
    lr: float = 1e-2
    shaping: bool = True
    n_heldout: int = 6
    max_steps: int = 100
    gamma: float = 0.99
    baseline_beta: float = 0.1
    size: int = 5
    start: tuple[int, int] = (0, 0)
    head_type: str = "linear"
    hidden: int = 128
    entropy_beta: float = 0.0
    normalize_advantages: bool = False
    pretrain_sensory: bool = False
    pretrain_epochs: int = 200
    pretrain_lr: float = 1e-3
    tag: str = "shaped"
    out_dir: Path = field(default_factory=lambda: Path("outputs"))


def run_generalization(cfg: GenConfig) -> dict:
    """Train the head on random train-set goals; eval train vs held-out; write CSV + JSON."""
    torch.manual_seed(cfg.seed)
    train_goals, heldout_goals = split_goals(cfg.size, cfg.start, cfg.n_heldout, cfg.seed)

    env = GridWorldEnv(
        size=cfg.size, start=cfg.start, goals=train_goals, goal_seed=cfg.seed,
        reward_shaping=cfg.shaping, max_steps=cfg.max_steps,
    )
    brain = Brain(grid_n=cfg.size, seed=cfg.seed)
    pretrain_info = None
    if cfg.pretrain_sensory:
        pretrain_info = pretrain_sensory(
            brain.sensory, grid_n=cfg.size, epochs=cfg.pretrain_epochs, lr=cfg.pretrain_lr,
            seed=cfg.seed, generator=torch.Generator().manual_seed(cfg.seed),
        )
    head = make_policy_head(brain, head_type=cfg.head_type, hidden=cfg.hidden)
    gen = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.Adam(policy_parameters(head), lr=cfg.lr)

    rows = []
    baseline = 0.0
    for ep in range(cfg.episodes):
        stats = train_episode(
            brain, head, env, opt, gamma=cfg.gamma, baseline=baseline,
            generator=gen, max_steps=cfg.max_steps, entropy_beta=cfg.entropy_beta,
            normalize_advantages=cfg.normalize_advantages,
        )
        baseline = ema(baseline, stats["mean_return"], cfg.baseline_beta)
        gx, gy = int(env.goal[0]), int(env.goal[1])
        rows.append((ep + 1, gx, gy, stats["total_reward"], stats["steps"],
                     int(stats["reached_goal"]), stats["mean_entropy"]))

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.out_dir / f"024_grid_generalization_{cfg.tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "goal_x", "goal_y", "total_reward", "steps", "goal_reached", "entropy"])
        w.writerows(rows)

    eval_train = evaluate(brain, head, train_goals, size=cfg.size, start=cfg.start,
                          max_steps=cfg.max_steps, generator=gen)
    eval_held = evaluate(brain, head, heldout_goals, size=cfg.size, start=cfg.start,
                         max_steps=cfg.max_steps, generator=gen)
    gap = eval_train.success_rate - eval_held.success_rate

    summary = {
        "config": asdict(cfg) | {"out_dir": str(cfg.out_dir)},
        "train_goals": train_goals,
        "heldout_goals": heldout_goals,
        "eval": {"train": asdict(eval_train), "heldout": asdict(eval_held)},
        "pretrain": pretrain_info,
        "generalization_gap": gap,
    }
    json_path = cfg.out_dir / f"024_grid_generalization_{cfg.tag}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    return summary
