"""EXP-029: the v1 fail-first baseline on the 2x2 cube, plus its unregionalized control.

``run_generalization`` cannot be reused: it is built on ``split_goals``, ``manhattan``
optimality and ``GridWorldEnv``. This is the cube analogue. ``reinforce.py`` IS reused
unchanged, because it is already environment-agnostic.

Difficulty is exact distance-to-solved, not move count, so the collapse curve is read off
a true axis. The distance table is an instrument only: the agent observes raw facelets and
never sees a distance. See docs/superpowers/specs/2026-07-25-cube-baseline-design.md.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from neuromorphic.analysis.ablate import AblatedConcept, AblationSpec
from neuromorphic.brain import Brain
from neuromorphic.encoders import cube_encoder
from neuromorphic.envs.cube import CubeEnv
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.monolithic import MonolithicBrain
from neuromorphic.training.reinforce import (
    ema,
    greedy_action,
    make_policy_head,
    policy_parameters,
    train_episode,
)

CUBE_N_OBS = 144      # 24 facelets x 6 colors
CUBE_OBS_WIDTH = 24   # raw facelets


@dataclass
class CubeConfig:
    """One run: one arm, one depth, one seed, one sigma."""

    seed: int = 0
    depth: int = 1
    arm: str = "regionalized"   # "regionalized" | "monolithic" | "random"
    sigma: float = 0.0          # Gaussian concept-noise dose (the EXP-028 operator)
    episodes: int = 600
    lr: float = 1e-2
    gamma: float = 0.99
    baseline_beta: float = 0.1
    entropy_beta: float = 0.0
    normalize_advantages: bool = False
    content: int = 64
    n_actions: int = 6
    max_depth: int = 6          # BFS table bound
    heldout_cap: int = 200
    heldout_frac: float = 0.25
    tag: str = "exp029"
    out_dir: Path = field(default_factory=lambda: Path("outputs"))


def max_steps_for(depth: int) -> int:
    """Step budget at a given exact distance. Optimal is ``depth``; this is generous."""
    return 2 * depth + 3


def shell_states(provider: ExactBFSDistance, depth: int) -> list[tuple[int, ...]]:
    """Every state at exact distance ``depth``, sorted so the order is deterministic."""
    return provider.states_at_distance(depth)


def split_shell(
    states: list[tuple[int, ...]],
    depth: int,
    *,
    seed: int,
    heldout_cap: int = 200,
    heldout_frac: float = 0.25,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], bool]:
    """Partition a shell into (train, eval, is_heldout).

    Depths 1 and 2 have only 6 and 27 states. Holding out 1 of 6 is not a generalization
    test, so those depths are NOT split: train and eval are both the whole shell and the
    caller must label the number training-distribution. Deeper shells are split, with the
    held-out side capped so a single evaluation stays affordable (brain.step is 90 ms).
    """
    if depth <= 2:
        return list(states), list(states), False
    shuffled = list(states)
    random.Random(seed).shuffle(shuffled)
    n_eval = min(heldout_cap, int(len(shuffled) * heldout_frac))
    return shuffled[n_eval:], shuffled[:n_eval], True


class ShellCubeEnv(CubeEnv):
    """A ``CubeEnv`` whose ``reset()`` draws its start state from a fixed pool.

    ``train_episode`` calls ``env.reset()`` with no arguments, so this is how the training
    loop is confined to the train side of the split without touching ``reinforce.py``.
    """

    def __init__(self, states, rng: random.Random, **kwargs):
        super().__init__(**kwargs)
        self._pool = list(states)
        self._pool_rng = rng

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options is None:
            options = {"state": self._pool_rng.choice(self._pool)}
        return super().reset(seed=seed, options=options)


def make_agent(cfg: CubeConfig):
    """Build the arm's feature extractor. Both arms are frozen at random init."""
    if cfg.arm == "regionalized":
        return Brain(
            encoder=cube_encoder(), n_obs=CUBE_N_OBS, obs_width=CUBE_OBS_WIDTH,
            n_actions=cfg.n_actions, content=cfg.content, seed=cfg.seed,
        )
    if cfg.arm == "monolithic":
        reference = Brain(
            encoder=cube_encoder(), n_obs=CUBE_N_OBS, obs_width=CUBE_OBS_WIDTH,
            n_actions=cfg.n_actions, content=cfg.content, seed=cfg.seed,
        )
        return MonolithicBrain(
            n_obs=CUBE_N_OBS, n_actions=cfg.n_actions, total_neurons=reference.n_neurons,
            content=cfg.content, obs_width=CUBE_OBS_WIDTH, encoder=cube_encoder(),
            seed=cfg.seed,
        )
    # "random" never reaches here: run_cube_baseline short-circuits it, since the chance
    # floor needs no feature extractor at all.
    raise ValueError(f"unknown arm {cfg.arm!r} (expected regionalized or monolithic)")


def evaluate_states(
    agent,
    head,
    states,
    *,
    depth: int,
    generator: torch.Generator | None = None,
    random_policy: bool = False,
) -> dict:
    """Greedy rollouts from each state. ``random_policy`` measures the chance floor."""
    limit = max_steps_for(depth)
    env = CubeEnv(scramble_depth=depth, max_steps=limit)
    rng = random.Random(0)
    solved = 0
    steps_solved: list[int] = []
    for state in states:
        obs, _ = env.reset(options={"state": state})
        for t in range(1, limit + 1):
            if random_policy:
                action = rng.randrange(env.action_space.n)
            else:
                with torch.no_grad():
                    action = greedy_action(agent, head, obs, generator=generator)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated:
                solved += 1
                steps_solved.append(t)
                break
            if truncated:
                break
    n = len(states)
    total_steps = sum(steps_solved)
    return {
        "success_rate": solved / n if n else 0.0,
        "mean_steps": total_steps / len(steps_solved) if steps_solved else 0.0,
        "optimality": (depth * len(steps_solved) / total_steps) if total_steps else 0.0,
        "n": n,
    }


def run_cube_baseline(cfg: CubeConfig) -> dict:
    """One (arm, depth, seed, sigma) run. Returns a JSON-safe record."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    provider = ExactBFSDistance(max_depth=max(cfg.max_depth, cfg.depth))
    states = shell_states(provider, cfg.depth)
    train_states, eval_states, is_heldout = split_shell(
        states, cfg.depth, seed=cfg.seed,
        heldout_cap=cfg.heldout_cap, heldout_frac=cfg.heldout_frac,
    )

    if cfg.arm == "random":
        result = evaluate_states(None, None, eval_states, depth=cfg.depth, random_policy=True)
        episodes_run = 0
    else:
        agent = make_agent(cfg)
        spec = AblationSpec(kind="gaussian", dose=cfg.sigma, seed=cfg.seed) if cfg.sigma else None
        head = AblatedConcept(
            make_policy_head(agent, "linear"), spec, width=cfg.content
        )
        optimizer = torch.optim.Adam(policy_parameters(head), lr=cfg.lr)
        env = ShellCubeEnv(
            train_states, random.Random(cfg.seed),
            scramble_depth=cfg.depth, max_steps=max_steps_for(cfg.depth),
        )
        baseline = 0.0
        for _ in range(cfg.episodes):
            stats = train_episode(
                agent, head, env, optimizer,
                gamma=cfg.gamma, baseline=baseline, generator=generator,
                max_steps=max_steps_for(cfg.depth),
                entropy_beta=cfg.entropy_beta,
                normalize_advantages=cfg.normalize_advantages,
            )
            baseline = ema(baseline, stats["mean_return"], cfg.baseline_beta)
        result = evaluate_states(
            agent, head, eval_states, depth=cfg.depth, generator=generator
        )
        episodes_run = cfg.episodes

    record = {
        "arm": cfg.arm,
        "depth": cfg.depth,
        "seed": cfg.seed,
        "sigma": cfg.sigma,
        "episodes": episodes_run,
        "is_heldout": is_heldout,
        "n_train": len(train_states),
        "tag": cfg.tag,
        **result,
    }

    # One file per run, never a shared append. The driver fans out over processes, and
    # concurrent appends to a single file interleave and corrupt lines on Windows.
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{cfg.tag}_{cfg.arm}_d{cfg.depth}_s{cfg.seed}_sig{cfg.sigma}.json"
    (out_dir / name).write_text(json.dumps(record), encoding="utf-8")
    return record
