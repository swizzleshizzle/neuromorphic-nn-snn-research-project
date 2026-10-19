"""EXP-079 evaluation: EXP-076's full 4-move tree driven by judge S or by EXP-075's X.

J4V-S loads this experiment's trained S judge. J4V-X loads EXP-075's committed X judge and is
re-run here only for Gate 0(b); its claim records are EXP-076's. Held-out sets, budgets, streams
and the no-revisit rule are EXP-075's (its `load_cell` and `JudgeCritic` are used directly), and
the search is EXP-076's J4V unchanged (mode C, k 4, goal check at 4).

Spec: docs/superpowers/specs/2026-10-18-exp079-judge-trained-where-scrambles-live-design.md,
sections 3 and 7.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/079_judge_trained_deep/evaluate.py --arm J4V-S --depth 11 --seed 0
    .venv/bin/python -u experiments/079_judge_trained_deep/evaluate.py --rank --depth 11 --seed 0
    .venv/bin/python -u experiments/079_judge_trained_deep/evaluate.py --gatel-x --seed 0
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


cells = _load("exp079_cells", HERE / "cells.py")
e75 = _load("exp075_evaluate", REPO / "experiments" / "075_wide_region" / "evaluate.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training.lookahead import (  # noqa: E402
    evaluate_lookahead, imag_seed_for, imagined_values, tree_levels,
)

torch.set_num_threads(1)

ARMS = ("J4V-S", "J4V-X")
K = 4
DEPTHS = (9, 11)


def record_name(arm: str, depth: int, seed: int) -> str:
    return f"exp079_{arm}_d{depth}_s{seed}.json"


def rank_record_name(depth: int, seed: int) -> str:
    return f"exp079_rank_J-S4_d{depth}_s{seed}.json"


def gatel_x_record_name(seed: int) -> str:
    return f"exp079_gatel_X_s{seed}.json"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _check_arm(arm) -> None:
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; valid: {list(ARMS)}")


def load_judge(arm, seed, judge_dir=None):
    """J4V-S: this experiment's S checkpoint (`judge_dir`, default outputs/). J4V-X: EXP-075's
    committed X judge. Both refuse a judge short of 4000 updates."""
    _check_arm(arm)
    if arm == "J4V-S":
        return cells.load_trained(seed, judge_dir)
    return e75.load_judge("X", seed, cells.E75_OUT)


def _judge_updates(arm, seed, judge_dir=None) -> int:
    if arm == "J4V-S":
        return cells.judge_updates(seed, judge_dir)
    return e75.judge_updates("X", seed, cells.E75_OUT)


def _write(out_dir, name, rec) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")


def run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict:
    torch.set_num_threads(1)
    _check_arm(arm)
    t0 = time.time()
    critic = e75.JudgeCritic(load_judge(arm, seed, judge_dir))
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=K, k_goal=K,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    rec = {**res, "depth": depth, "seed": seed, "arm": arm, "judge": arm[-1],
           "fired4": res["first_fire_len"].get("4", 0),
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "limit_states": limit_states,
           "judge_updates": _judge_updates(arm, seed, judge_dir)}
    _write(out_dir, record_name(arm, depth, seed), rec)
    return rec


def rank_cell(depth, seed, out_dir, judge_dir=None, limit_states=None, provider=None) -> dict:
    """Gate R at k 4 for S (reported, not voiding; spec section 7): per-seed mean hit of 'the
    lowest-J 4-move leaf is closer to solved than the root', on evaluation held-out states. BFS
    is the yardstick only; unbounded by default because depth + 4 passes the 2x2's diameter."""
    t0 = time.time()
    torch.set_num_threads(1)
    critic = e75.JudgeCritic(load_judge("J4V-S", seed, judge_dir))
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    if provider is None:
        provider = ExactBFSDistance(max_depth=None)
    n_actions = head.head.out_features
    hits, chances = [], []
    for i, s in enumerate(states):
        g = torch.Generator().manual_seed(imag_seed_for(train_seed, i, 0))
        levels = tree_levels(s, K, n_actions)
        scores = imagined_values(agent, critic, levels[K], generator=g)
        d = provider.distance(s)
        leaf_d = [provider.distance(x) for x in levels[K]]
        hits.append(int(leaf_d[int(scores.argmax())] < d))
        chances.append(sum(x < d for x in leaf_d) / len(leaf_d))
    rec = {"kind": "J-S4", "k": K, "depth": depth, "seed": seed, "n": len(states),
           "hit": st.mean(hits), "chance": st.mean(chances), "limit_states": limit_states,
           "judge_updates": cells.judge_updates(seed, judge_dir),
           "wall_s": round(time.time() - t0, 1)}
    _write(out_dir, rank_record_name(depth, seed), rec)
    return rec


def gatel_x_cell(seed, out_dir, provider=None) -> dict:
    """Gate M's continuity cell: EXP-075's committed X judge for `seed`, re-measured by today's
    Gate L code on its own probe set. Gate M pairs S's margins with X's PUBLISHED ones, so this
    must reproduce the committed training record's gate_l exactly (CLAUDE.md: re-evaluate the
    baseline, never trust a published per-seed value unchecked)."""
    torch.set_num_threads(1)
    t0 = time.time()
    judge = load_judge("J4V-X", seed)
    if provider is None:
        provider = ExactBFSDistance(max_depth=None)
    probe = cells.probe_set(seed, provider)
    rec = {"kind": "gatel-X", "seed": seed,
           "gate_l": gate_l.leaf_rank_margin(judge, probe, provider.distance, seed, N_ACTIONS),
           "judge_updates": _judge_updates("J4V-X", seed),
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit()}
    _write(out_dir, gatel_x_record_name(seed), rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--gatel-x", action="store_true",
                    help="Gate M's continuity cell: re-measure X's Gate L for --seed")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--judge-dir", type=Path, default=None,
                    help="where judge_S_s<seed>/ lives; this experiment's outputs/ when omitted")
    ap.add_argument("--limit-states", type=int, default=None)
    args = ap.parse_args()
    if (args.arm is not None) + args.rank + args.gatel_x != 1:
        raise SystemExit("pass exactly one of --arm, --rank or --gatel-x")
    if args.gatel_x:
        r = gatel_x_cell(args.seed, args.out_dir)
        print(f"gatel-X s{r['seed']} margin {r['gate_l']['margin']:.6f} wall {r['wall_s']}s")
        return
    if args.depth is None:
        raise SystemExit("--depth is required with --arm or --rank")
    if args.arm:
        r = run_cell(args.arm, args.depth, args.seed, args.out_dir, args.limit_states,
                     args.judge_dir)
        print(f"{r['arm']} d{r['depth']} s{r['seed']} solved {r['solved']}/{r['n']} "
              f"fired4 {r['fired4']} wall {r['wall_s']}s")
    else:
        r = rank_cell(args.depth, args.seed, args.out_dir, args.judge_dir, args.limit_states)
        print(f"J-S4 d{r['depth']} s{r['seed']} hit {r['hit']:.4f} chance {r['chance']:.4f}")


if __name__ == "__main__":
    main()
