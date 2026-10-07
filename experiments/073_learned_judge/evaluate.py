"""EXP-073 evaluation: the trained judge J plugged into the EXISTING look-ahead critic scorer.

Nothing in the search changes. `NegJ` flips J's sign so that `choose_move` mode "C" (which takes
the argmax of the critic's value) picks the sequence whose leaf has the LOWEST J. The BFS table
is used here only as a yardstick for the leaf-level rank checks, never inside the procedure.

Spec: docs/superpowers/specs/2026-10-07-exp073-learned-judge-design.md

Usage (from the repo root, PYTHONPATH=src set):
    .venv/bin/python -u experiments/073_learned_judge/evaluate.py --help
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp073_cells", HERE / "cells.py")
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
    "J1V-A": ("C", 1, "A"),
    "J1V-B": ("C", 1, "B"),
    "J3V-A": ("C", 3, "A"),
    "J3V-B": ("C", 3, "B"),
    "P3V": ("P", 3, None),
    "R3V": ("R", 3, None),
}
ARMS_EVAL = tuple(_ARMS)
RANK_KINDS = ("J-A", "J-B", "P")


class NegJ(nn.Module):
    """-J as a critic: `choose_move`'s argmax then selects the lowest J."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, concept):
        return -F.softplus(self.head(concept))


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _judge_path(train_arm: str, seed: int) -> Path:
    return cells.ckpt_dir(train_arm, seed) / "judge.pt"


def eval_agent(arm: str, seed: int, depth: int = 7, cell=None):
    """(agent, head, critic) for one evaluation arm.

    The agent and policy head are EXP-070's `load_cell(depth, seed)`. For a J arm the agent's
    sensory region is replaced by the trained judge's (arm A moved it; arm B left it at E1), the
    policy head is untouched (it only supplies the evaluation stream's logits), and the critic is
    NegJ over the trained judge head. P3V and R3V return critic None.
    """
    mode, k, train_arm = _ARMS[arm]
    agent, head, _states, _ts = cell if cell is not None else cells.c70.load_cell(depth, seed)
    if train_arm is None:
        return agent, head, None
    path = _judge_path(train_arm, seed)
    if not path.exists():
        raise SystemExit(f"missing judge checkpoint {path} for arm {arm} seed {seed}; "
                         f"train it with train.py first")
    sd = torch.load(path, map_location="cpu")
    judge = cells.make_judge(seed)
    judge.load_state_dict(sd)
    agent.sensory.load_state_dict(judge.sensory.state_dict())
    judge.head.eval()
    return agent, head, NegJ(judge.head)


def run_cell(arm: str, depth: int, seed: int, out_dir: Path, limit_states=None) -> dict:
    torch.set_num_threads(1)
    if arm not in _ARMS:
        raise SystemExit(f"unknown arm {arm!r}; valid: {list(ARMS_EVAL)}")
    mode, k, _ = _ARMS[arm]
    t0 = time.time()
    cell = cells.c70.load_cell(depth, seed)
    _a, _h, states, train_seed = cell
    agent, head, critic = eval_agent(arm, seed, depth, cell=cell)
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
    (out_dir / f"exp073_{arm}_d{depth}_s{seed}.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def rank_cell(kind: str, depth: int, seed: int, out_dir: Path) -> dict:
    """Per-seed mean hit of 'the top leaf at k=3 is closer to solved than the root'.

    J kinds: top leaf = lowest J. "P": top leaf = best summed policy log-probability sequence.
    BFS (max_depth = depth + 3) is the yardstick here only.
    """
    torch.set_num_threads(1)
    if kind not in RANK_KINDS:
        raise SystemExit(f"unknown kind {kind!r}; valid: {list(RANK_KINDS)}")
    k = 3
    cell = cells.c70.load_cell(depth, seed)
    _a, _h, states, train_seed = cell
    arm = {"J-A": "J3V-A", "J-B": "J3V-B", "P": "P3V"}[kind]
    agent, head, critic = eval_agent(arm, seed, depth, cell=cell)
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
    (out_dir / f"exp073_rank_{kind}_d{depth}_s{seed}.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS_EVAL)
    ap.add_argument("--rank", choices=RANK_KINDS)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--limit-states", type=int, default=None)
    args = ap.parse_args()
    if (args.arm is None) == (args.rank is None):
        raise SystemExit("pass exactly one of --arm or --rank")
    if args.arm:
        r = run_cell(args.arm, args.depth, args.seed, args.out_dir, args.limit_states)
        print(f"{r['arm']} d{r['depth']} s{r['seed']} solved {r['solved']}/{r['n']}")
    else:
        r = rank_cell(args.rank, args.depth, args.seed, args.out_dir)
        print(f"{r['kind']} d{r['depth']} s{r['seed']} hit {r['hit']:.4f} chance {r['chance']:.4f}")


if __name__ == "__main__":
    main()
