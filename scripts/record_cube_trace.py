"""Record one 2x2 cube episode to a dashboard JSONL trace.

Trains a cube policy in-process at a fixed seed, then records one greedy episode.
No checkpoint format is involved: ``checkpoints.load_trained`` hardcodes a
gridworld ``Brain`` and cannot rebuild a cube brain.

The training loop mirrors ``run_cube_baseline``'s seeding order exactly.
``tests/monitor/test_record_cube_trace.py`` guards that against drift.

Run:
    .venv\\Scripts\\python.exe scripts/record_cube_trace.py --depth 2 --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from neuromorphic.analysis.ablate import AblatedConcept
from neuromorphic.envs.cube import CubeEnv
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.monitor import CubeAdapter, FileSink, record_policy_episode
from neuromorphic.training.cube_baseline import (
    CubeConfig,
    MemoryReadout,
    ShellCubeEnv,
    feature_width,
    make_agent,
    max_steps_for,
    shell_states,
    split_shell,
)
from neuromorphic.training.reinforce import ema, policy_parameters, train_episode

DEFAULT_OUT = Path("outputs/cube_trace.jsonl")


def train_cube_policy(depth: int, seed: int, episodes: int, provider):
    """Train a concept-readout cube policy.

    Mirrors ``run_cube_baseline``'s seeding order and object construction exactly,
    including the ``AblatedConcept`` wrapper (used there even at sigma 0, with a
    ``None`` spec) and the concept-mode ``MemoryReadout``. Substituting a bare
    ``nn.Linear`` or ``feature_fn=None`` would look equivalent and silently consume
    the RNG stream differently.
    """
    cfg = CubeConfig(seed=seed, depth=depth, episodes=episodes, readout="concept")
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    states = shell_states(provider, cfg.depth)
    train_states, _, _ = split_shell(
        states, cfg.depth, seed=cfg.seed,
        heldout_cap=cfg.heldout_cap, heldout_frac=cfg.heldout_frac,
    )

    brain = make_agent(cfg)
    torch.manual_seed(cfg.seed)  # head init stream matched to run_cube_baseline
    readout = MemoryReadout(cfg.readout, random.Random(cfg.seed), brain)
    width = feature_width(cfg)
    head = AblatedConcept(nn.Linear(width, cfg.n_actions), None, width=width)
    optimizer = torch.optim.Adam(policy_parameters(head), lr=cfg.lr)
    env = ShellCubeEnv(
        train_states, random.Random(cfg.seed),
        scramble_depth=cfg.depth, max_steps=max_steps_for(cfg.depth),
    )

    baseline = 0.0
    for _ in range(cfg.episodes):
        readout.reset()
        stats = train_episode(
            brain, head, env, optimizer,
            gamma=cfg.gamma, baseline=baseline, generator=generator,
            max_steps=max_steps_for(cfg.depth),
            entropy_beta=cfg.entropy_beta,
            normalize_advantages=cfg.normalize_advantages,
            store=False, recall=False, feature_fn=readout,
        )
        baseline = ema(baseline, stats["mean_return"], cfg.baseline_beta)
    return brain, head, generator


def record(*, depth: int, seed: int, episodes: int, out_path) -> dict:
    """Train, then record one greedy episode to ``out_path``."""
    # Built once and shared: ExactBFSDistance expands the full 3,674,160-state table
    # (about 67 s). Building it separately for training and recording doubles that.
    provider = ExactBFSDistance(max_depth=max(6, depth))
    brain, head, generator = train_cube_policy(depth, seed, episodes, provider)
    env = CubeEnv(
        scramble_depth=depth,
        max_steps=max_steps_for(depth),
        scramble_seed=seed,
        distance_provider=provider,
    )
    summary = record_policy_episode(
        brain, head, env, FileSink(out_path),
        seed=seed,
        adapter=CubeAdapter(),
        max_steps=max_steps_for(depth),
        recall=False,
        policy_regions=("sensory",),
        generator=generator,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    summary = record(depth=args.depth, seed=args.seed,
                     episodes=args.episodes, out_path=args.out)
    print("cube trace written")
    print(f"  file         : {args.out}")
    print(f"  depth        : {args.depth}")
    print(f"  steps        : {summary['steps']}")
    print(f"  reached goal : {summary['reached_goal']}")


if __name__ == "__main__":
    main()
