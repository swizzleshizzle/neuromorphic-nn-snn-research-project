"""EXP-077 evaluation: EXP-075's committed X judges drive a deeper look-ahead, three arms.

J4V-X is EXP-076's arm, re-run here only for Gate 0(b) (mode C, k 4). J4VG5-X keeps the 4-move
judge tree and extends the goal check to 5 moves. B5-X is a width-64 five-move beam (mode B).
Judges, held-out sets, budgets, streams and the no-revisit rule are EXP-075's (its `load_judge`,
`JudgeCritic` and `load_cell` are used directly). Spec:
docs/superpowers/specs/2026-10-14-exp077-beam-search-design.md, sections 3 and 7.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/077_beam_search/evaluate.py --arm B5-X --depth 11 --seed 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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

from neuromorphic.training.lookahead import evaluate_lookahead  # noqa: E402

torch.set_num_threads(1)

E76_OUT = REPO / "experiments" / "076_deeper_tree" / "outputs"
# arm -> (mode, judge depth k, goal check depth k_goal, beam_width); spec section 3
ARMS = {"J4V-X": ("C", 4, 4, None), "J4VG5-X": ("C", 4, 5, None), "B5-X": ("B", 5, 5, 64)}
DEPTHS = (9, 11)


def record_name(arm: str, depth: int, seed: int) -> str:
    return f"exp077_{arm}_d{depth}_s{seed}.json"


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
    (out_dir / name).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")


def run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict:
    torch.set_num_threads(1)
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; valid: {list(ARMS)}")
    mode, k, k_goal, beam_width = ARMS[arm]
    t0 = time.time()
    critic = _critic(seed, judge_dir)
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode=mode, k=k, k_goal=k_goal,
                             beam_width=beam_width,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    rec = {**res, "depth": depth, "seed": seed, "arm": arm,
           "fired5": res["first_fire_len"].get("5", 0),
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "limit_states": limit_states,
           "judge_updates": e75.judge_updates("X", seed, judge_dir)}
    _write(out_dir, record_name(arm, depth, seed), rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--judge-dir", type=Path, default=None,
                    help="EXP-075's outputs (where judge_X_s<seed>/ live) when omitted")
    ap.add_argument("--limit-states", type=int, default=None)
    args = ap.parse_args()
    r = run_cell(args.arm, args.depth, args.seed, args.out_dir, args.limit_states,
                 args.judge_dir)
    print(f"{r['arm']} d{r['depth']} s{r['seed']} solved {r['solved']}/{r['n']} "
          f"fired5 {r['fired5']} judge_evals {r.get('judge_evals')} wall {r['wall_s']}s")


if __name__ == "__main__":
    main()
