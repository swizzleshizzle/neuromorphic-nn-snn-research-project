"""EXP-078 evaluation: EXP-075's committed X judges drive a full five-move look-ahead tree.

J5V-X searches all 6^5 = 7776 five-move sequences with the learned judge and checks goals to 5
moves (mode C, k 5). J4V-X is EXP-076's arm, re-run here only for Gate 0(b). Cells checkpoint
within a cell (`checkpoint_every` states) so a lost wave resumes instead of restarting. Judges,
held-out sets, budgets, streams and the no-revisit rule are EXP-075's (its `load_judge`,
`JudgeCritic` and `load_cell` are used directly). Spec:
docs/superpowers/specs/2026-10-14-exp078-full-five-move-tree-design.md, sections 3, 4 and 7.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/078_full_five_move_tree/evaluate.py --arm J5V-X --depth 11 --seed 0
    .venv/bin/python -u experiments/078_full_five_move_tree/evaluate.py --rank --depth 11 --seed 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


e75 = _load("exp075_evaluate", REPO / "experiments" / "075_wide_region" / "evaluate.py")

from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training.lookahead import (  # noqa: E402
    evaluate_lookahead, imag_seed_for, imagined_values, tree_levels,
)

torch.set_num_threads(1)

E76_OUT = REPO / "experiments" / "076_deeper_tree" / "outputs"
E77_OUT = REPO / "experiments" / "077_beam_search" / "outputs"
E77_DET = REPO / "experiments" / "077_beam_search" / "outputs_det"
# arm -> (mode, judge depth k, goal check depth k_goal, beam_width); spec section 3
ARMS = {"J5V-X": ("C", 5, 5, None), "J4V-X": ("C", 4, 4, None)}
DEPTHS = (9, 11)
CKPT_EVERY = 10
FORCED_STOP = 50
RANK_K = 5


def record_name(arm: str, depth: int, seed: int) -> str:
    return f"exp078_{arm}_d{depth}_s{seed}.json"


def rank_record_name(depth: int, seed: int) -> str:
    return f"exp078_rank_J-X5_d{depth}_s{seed}.json"


def ckpt_path(out_dir, arm: str, depth: int, seed: int) -> Path:
    return Path(out_dir) / "ckpt" / (Path(record_name(arm, depth, seed)).stem + ".ckpt")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _critic(seed, judge_dir):
    return e75.JudgeCritic(e75.load_judge("X", seed, judge_dir))


def _write(out_dir, name, rec) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out_dir / name)


def run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None,
             stop_after=None):
    """Evaluate one cell with within-cell checkpoints. Returns None (and writes no record)
    when `stop_after` interrupts it; the checkpoint is deleted only after the record exists."""
    torch.set_num_threads(1)
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; valid: {list(ARMS)}")
    mode, k, k_goal, beam_width = ARMS[arm]
    target = Path(out_dir) / record_name(arm, depth, seed)
    if target.exists():
        prior = json.loads(target.read_text()).get("limit_states")
        if prior != limit_states:
            raise SystemExit(f"{target} exists with limit_states={prior!r} but this call has "
                             f"limit_states={limit_states!r}; move the old record away by hand")
    critic = _critic(seed, judge_dir)
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    ck = ckpt_path(out_dir, arm, depth, seed)
    ck.parent.mkdir(parents=True, exist_ok=True)
    res = evaluate_lookahead(agent, head, states, depth=depth, mode=mode, k=k, k_goal=k_goal,
                             beam_width=beam_width,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True,
                             checkpoint=ck, checkpoint_every=CKPT_EVERY, stop_after=stop_after)
    if res is None:
        return None
    rec = {**res, "depth": depth, "seed": seed, "arm": arm,
           "fired5": res["first_fire_len"].get("5", 0),
           "wall_s": round(res["elapsed_s"], 1), "git_commit": _git_commit(),
           "limit_states": limit_states,
           "judge_updates": e75.judge_updates("X", seed, judge_dir)}
    _write(out_dir, record_name(arm, depth, seed), rec)
    ck.unlink(missing_ok=True)
    return rec


def rank_cell(depth, seed, out_dir, judge_dir=None, limit_states=None, provider=None) -> dict:
    """Gate R at k 5 (reported, not voiding): per-seed mean hit of 'the lowest-J 5-move leaf is
    closer to solved than the root', on evaluation held-out states. BFS is the yardstick only;
    unbounded by default because depth + 5 passes the 2x2's diameter."""
    t0 = time.time()
    torch.set_num_threads(1)
    k = RANK_K
    critic = _critic(seed, judge_dir)
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    if provider is None:
        provider = ExactBFSDistance(max_depth=None)
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
    rec = {"kind": "J-X5", "k": k, "depth": depth, "seed": seed, "n": len(states),
           "hit": st.mean(hits), "chance": st.mean(chances), "limit_states": limit_states,
           "judge_updates": e75.judge_updates("X", seed, judge_dir),
           "wall_s": round(time.time() - t0, 1)}
    _write(out_dir, rank_record_name(depth, seed), rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--judge-dir", type=Path, default=None,
                    help="EXP-075's outputs (where judge_X_s<seed>/ live) when omitted")
    ap.add_argument("--limit-states", type=int, default=None)
    ap.add_argument("--stop-after", type=int, default=None,
                    help="stop after N states, checkpoint saved, no record written")
    args = ap.parse_args()
    if (args.arm is None) == (not args.rank):
        raise SystemExit("pass exactly one of --arm or --rank")
    if args.arm:
        r = run_cell(args.arm, args.depth, args.seed, args.out_dir, args.limit_states,
                     args.judge_dir, args.stop_after)
        if r is None:
            print(f"{args.arm} d{args.depth} s{args.seed} stopped, checkpoint saved")
            return
        print(f"{r['arm']} d{r['depth']} s{r['seed']} solved {r['solved']}/{r['n']} "
              f"fired5 {r['fired5']} judge_evals {r.get('judge_evals')} wall {r['wall_s']}s")
    else:
        r = rank_cell(args.depth, args.seed, args.out_dir, args.judge_dir, args.limit_states)
        print(f"J-X5 d{r['depth']} s{r['seed']} hit {r['hit']:.4f} chance {r['chance']:.4f}")


if __name__ == "__main__":
    main()
