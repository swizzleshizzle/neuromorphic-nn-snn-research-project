"""EXP-083 Gate F's chance floor: a uniform random policy with the search cell's step budget
(2d+3), over the same sets as the search, seeded by the cell's train seed. Measured, never assumed.

Spec: docs/superpowers/specs/2026-10-30-exp083-unseen-depth9-design.md, section 5 (Gate F).

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/083_unseen_depth9/floor.py --workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("exp083_evaluate", HERE / "evaluate.py")
cells = ev.cells

from neuromorphic.envs.cube import N_ACTIONS, apply_move, is_solved  # noqa: E402
from neuromorphic.training.cube_baseline import max_steps_for  # noqa: E402


def random_rollout(state, depth, rng) -> bool:
    if is_solved(state):
        return True
    for _ in range(max_steps_for(depth)):
        state = apply_move(state, rng.randrange(N_ACTIONS))
        if is_solved(state):
            return True
    return False


def train_seed(seed: int) -> int:
    return ev.e75.load_cell(ev.POLICY_DEPTH, seed)[3]


def floor_cell(depth, seed, out=cells.OUT) -> dict:
    states = ev.load_states(seed, depth, out)
    ts = train_seed(seed)
    rng = random.Random(ts)
    solved = sum(random_rollout(s, depth, rng) for s in states)
    rec = {"depth": depth, "seed": seed, "n": len(states), "solved": solved,
           "success_rate": solved / len(states), "rng_seed": ts}
    ev._write(Path(out) / cells.floor_name(depth, seed), rec)
    return rec


def pending(seeds, out=cells.OUT) -> list[tuple[int, int]]:
    return [(d, s) for d in cells.DEPTHS for s in seeds
            if not (Path(out) / cells.floor_name(d, s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = pending(args.seeds, args.out_dir)
    print(f"EXP-083 floor: {len(jobs)} cells, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(floor_cell, d, s, args.out_dir) for d, s in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  d{r['depth']} s{r['seed']} solved {r['solved']}/{r['n']}",
                  flush=True)
    print("done.")


if __name__ == "__main__":
    main()
