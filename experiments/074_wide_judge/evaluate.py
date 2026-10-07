"""EXP-074 evaluation: trained judges drive the look-ahead search through a state-reading critic.

Spec: docs/superpowers/specs/2026-10-07-exp074-wide-judge-design.md, section 7.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/074_wide_judge/evaluate.py --arm J3V-W --depth 9 --seed 0
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
import torch.nn as nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp074_cells", HERE / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)

from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training.lookahead import (  # noqa: E402
    evaluate_lookahead, imag_seed_for, imagined_logp, imagined_values, sequence_scores,
    tree_levels,
)

torch.set_num_threads(1)

# arm -> (mode, k, trained-judge arm or None)
_ARMS = {
    "J3V-W": ("C", 3, "W"),
    "J3V-A": ("C", 3, "A"),
    "J3V-B": ("C", 3, "B"),
    "P3V": ("P", 3, None),
    "R3V": ("R", 3, None),
}
ARMS_EVAL = tuple(_ARMS)
RANK_KINDS = ("J-W", "J-A", "J-B", "P")
_RANK_ARM = {"J-W": "J3V-W", "J-A": "J3V-A", "J-B": "J3V-B", "P": "P3V"}


class JudgeCritic(nn.Module):
    """-J over leaf STATES: the search's argmax then picks the lowest J. `reads_states` makes
    `imagined_values` call this on the states, so the judge's own encoder and readout run."""

    reads_states = True

    def __init__(self, judge):
        super().__init__()
        self.judge = judge

    def forward(self, states, generator):
        return -self.judge(states, generator)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_judge(train_arm: str, seed: int, judge_dir=None):
    path = cells.ckpt_dir(train_arm, seed, judge_dir) / "judge.pt"
    if not path.exists():
        raise SystemExit(f"missing judge checkpoint {path} for arm {train_arm} seed {seed}; "
                         f"train it with train.py first")
    judge = cells.make_judge(seed, train_arm)
    judge.load_state_dict(torch.load(path, map_location="cpu"))
    judge.eval()
    return judge


def _critic(arm, seed, judge_dir):
    train_arm = _ARMS[arm][2]
    return None if train_arm is None else JudgeCritic(load_judge(train_arm, seed, judge_dir))


def run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict:
    torch.set_num_threads(1)
    if arm not in _ARMS:
        raise SystemExit(f"unknown arm {arm!r}; valid: {list(ARMS_EVAL)}")
    mode, k, _ = _ARMS[arm]
    t0 = time.time()
    critic = _critic(arm, seed, judge_dir)
    agent, head, states, train_seed = cells.c70.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode=mode, k=k,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    rec = {**res, "depth": depth, "seed": seed, "arm": arm,
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "limit_states": limit_states}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp074_{arm}_d{depth}_s{seed}.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def rank_cell(kind, depth, seed, out_dir, judge_dir=None) -> dict:
    """Per-seed mean hit of 'the top leaf at k=3 is closer to solved than the root' (Gate R),
    on evaluation held-out states. BFS (max_depth = depth + 3) is the yardstick only."""
    torch.set_num_threads(1)
    if kind not in RANK_KINDS:
        raise SystemExit(f"unknown kind {kind!r}; valid: {list(RANK_KINDS)}")
    k = 3
    arm = _RANK_ARM[kind]
    critic = _critic(arm, seed, judge_dir)
    agent, head, states, train_seed = cells.c70.load_cell(depth, seed)
    provider = ExactBFSDistance(max_depth=depth + 3)
    n_actions = head.head.out_features
    hits, chances = [], []
    for i, s in enumerate(states):
        g = torch.Generator().manual_seed(imag_seed_for(train_seed, i, 0))
        levels = tree_levels(s, k, n_actions)
        if critic is not None:
            scores = imagined_values(agent, critic, levels[k], generator=g)
        else:
            level_logp = [imagined_logp(agent, head, [s], generator=g)]
            for l in range(1, k):
                level_logp.append(imagined_logp(agent, head, levels[l], generator=g))
            scores = sequence_scores(level_logp, k, n_actions)
        d = provider.distance(s)
        leaf_d = [provider.distance(x) for x in levels[k]]
        hits.append(int(leaf_d[int(scores.argmax())] < d))
        chances.append(sum(x < d for x in leaf_d) / len(leaf_d))
    rec = {"kind": kind, "depth": depth, "seed": seed, "n": len(states),
           "hit": st.mean(hits), "chance": st.mean(chances)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp074_rank_{kind}_d{depth}_s{seed}.json").write_text(
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
        r = rank_cell(args.rank, args.depth, args.seed, args.out_dir, args.judge_dir)
        print(f"{r['kind']} d{r['depth']} s{r['seed']} hit {r['hit']:.4f} chance {r['chance']:.4f}")


if __name__ == "__main__":
    main()
