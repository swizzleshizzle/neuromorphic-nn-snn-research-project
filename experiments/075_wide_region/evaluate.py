"""EXP-075 evaluation: trained judges drive EXP-074's J3V search through its state-reading critic.

J3V-X and J3V-Y load this experiment's judges; J3V-W loads EXP-074's COMMITTED judges, re-run
only for Gate 0(b). The search, held-out sets, budgets, stream discipline and no-revisit rule are
EXP-074's (its `load_cell` and `JudgeCritic` are used directly). `run_cell` and `rank_cell` follow
EXP-074's, with this experiment's arms and record names.

Spec: docs/superpowers/specs/2026-10-09-exp075-wide-sensory-region-design.md, section 8.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/075_wide_region/evaluate.py --arm J3V-X --depth 9 --seed 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import subprocess
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp075_cells", HERE / "cells.py")
e74 = _load("exp074_evaluate", REPO / "experiments" / "074_wide_judge" / "evaluate.py")

from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training.lookahead import (  # noqa: E402
    evaluate_lookahead, imag_seed_for, imagined_values, tree_levels,
)

torch.set_num_threads(1)

# arm -> trained-judge arm
_ARMS = {"J3V-X": "X", "J3V-Y": "Y", "J3V-W": "W"}
ARMS_EVAL = tuple(_ARMS)
RANK_KINDS = ("J-X", "J-Y")
_RANK_ARM = {"J-X": "J3V-X", "J-Y": "J3V-Y"}
JudgeCritic = e74.JudgeCritic
load_cell = e74.load_cell


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_judge(train_arm: str, seed: int, judge_dir=None):
    """X and Y: this experiment's checkpoint, loaded into an untrained region of the arm's width
    (the checkpoint holds the region's trained weights, so the pretrained file is not needed).
    W: EXP-074's committed judge, always from EXP-074's outputs."""
    if train_arm == "W":
        return e74.load_judge("W", seed, cells.E74_OUT)
    path = cells.ckpt_dir(train_arm, seed, judge_dir) / "judge.pt"
    if not path.exists():
        raise SystemExit(f"missing judge checkpoint {path} for arm {train_arm} seed {seed}; "
                         f"train it with train.py first")
    judge = cells.build_judge(seed, train_arm, cells.fresh_sensory(train_arm, seed))
    judge.load_state_dict(torch.load(path, map_location="cpu"))
    judge.eval()
    return judge


def _critic(arm, seed, judge_dir):
    return JudgeCritic(load_judge(_ARMS[arm], seed, judge_dir))


def run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict:
    torch.set_num_threads(1)
    if arm not in _ARMS:
        raise SystemExit(f"unknown arm {arm!r}; valid: {list(ARMS_EVAL)}")
    t0 = time.time()
    critic = _critic(arm, seed, judge_dir)
    agent, head, states, train_seed = load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=3,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    rec = {**res, "depth": depth, "seed": seed, "arm": arm,
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "limit_states": limit_states}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp075_{arm}_d{depth}_s{seed}.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def rank_cell(kind, depth, seed, out_dir, judge_dir=None, limit_states=None) -> dict:
    """Gate R: per-seed mean hit of 'the lowest-J leaf at k=3 is closer to solved than the root',
    on evaluation held-out states. BFS (max_depth = depth + 3) is the yardstick only."""
    torch.set_num_threads(1)
    if kind not in RANK_KINDS:
        raise SystemExit(f"unknown kind {kind!r}; valid: {list(RANK_KINDS)}")
    k = 3
    critic = _critic(_RANK_ARM[kind], seed, judge_dir)
    agent, head, states, train_seed = load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    provider = ExactBFSDistance(max_depth=depth + 3)
    n_actions = head.head.out_features
    hits, chances = [], []
    for i, s in enumerate(states):
        g = torch.Generator().manual_seed(imag_seed_for(train_seed, i, 0))
        levels = tree_levels(s, k, n_actions)
        scores = imagined_values(agent, critic, levels[k], generator=g)
        d = provider.distance(s)
        leaf_d = [provider.distance(x) for x in levels[k]]
        hits.append(int(leaf_d[int(scores.argmax())] < d))
        chances.append(sum(x < d for x in leaf_d) / len(leaf_d))
    rec = {"kind": kind, "depth": depth, "seed": seed, "n": len(states),
           "hit": st.mean(hits), "chance": st.mean(chances), "limit_states": limit_states}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp075_rank_{kind}_d{depth}_s{seed}.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS_EVAL)
    ap.add_argument("--rank", choices=RANK_KINDS)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--judge-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--limit-states", type=int, default=None)
    args = ap.parse_args()
    if (args.arm is None) == (args.rank is None):
        raise SystemExit("pass exactly one of --arm or --rank")
    if args.arm:
        r = run_cell(args.arm, args.depth, args.seed, args.out_dir, args.limit_states,
                     args.judge_dir)
        print(f"{r['arm']} d{r['depth']} s{r['seed']} solved {r['solved']}/{r['n']}")
    else:
        r = rank_cell(args.rank, args.depth, args.seed, args.out_dir, args.judge_dir,
                      args.limit_states)
        print(f"{r['kind']} d{r['depth']} s{r['seed']} hit {r['hit']:.4f} chance {r['chance']:.4f}")


if __name__ == "__main__":
    main()
