"""EXP-080 evaluation cells.

Gate 0: EXP-075's committed X judge for a seed, re-measured by leaf_rank_margin on that seed's
MINTED probe and leaf distances. Equality with EXP-075's committed gate_l proves both that
today's code reproduces X's side of every claim and that the mint's distances are the BFS's.

Search: EXP-076's J4V (mode C, k 4, goal check at 4, no-revisit) at depth 11 with O-Xt as the
judge, EXP-075's held-out sets, streams and policy cells unchanged. The pilot timing cell runs
seed 12's pilot O-Xt judge on seed 0's cell (seed 12 has no depth-9 policy cell), 20 states,
written under its own name with limit_states set so it can never enter a verdict.

Spec: docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md, sections 3,
5 and 6.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/080_oracle_ordering/evaluate.py --gate0 --seed 0
    .venv/bin/python -u experiments/080_oracle_ordering/evaluate.py --search --workers 6
    .venv/bin/python -u experiments/080_oracle_ordering/evaluate.py --time \
        --judge-dir experiments/080_oracle_ordering/outputs_pilot \
        --out-dir experiments/080_oracle_ordering/outputs_pilot
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp080_cells", HERE / "cells.py")
mint = _load("exp080_mint", HERE / "mint.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")
e75 = cells.e75

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.training.lookahead import evaluate_lookahead  # noqa: E402

torch.set_num_threads(1)

ARM = "J4V-O-Xt"
K = 4
DEPTH = 11
TIME_JUDGE_SEED = 12
TIME_CELL_SEED = 0
TIME_STATES = 20


def record_name(seed: int) -> str:
    return f"exp080_{ARM}_d{DEPTH}_s{seed}.json"


def time_record_name(judge_seed: int, cell_seed: int) -> str:
    return f"exp080_time_{ARM}_j{judge_seed}_c{cell_seed}.json"


def gate0_record_name(seed: int) -> str:
    return f"exp080_gate0_X_s{seed}.json"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _write(out_dir, name, rec) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out_dir / name)


def gate0_cell(seed, mint_root, out_dir, x_root=cells.E75_OUT) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    judge = e75.load_judge("X", seed, x_root)
    m = mint.read_mint(mint_root, seed)
    rec = {"kind": "gate0-X", "seed": seed, "mint_sha": m.sha,
           "gate_l": gate_l.leaf_rank_margin(judge, m.probe, m.leaf_dist.__getitem__, seed,
                                             N_ACTIONS),
           "judge_updates": e75.judge_updates("X", seed, x_root),
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit()}
    _write(out_dir, gate0_record_name(seed), rec)
    return rec


def _updates(seed, judge_dir) -> int:
    path = cells.ckpt_dir("O-Xt", seed, judge_dir) / "state.json"
    return int(json.loads(path.read_text())["update"])


def run_cell(seed, out_dir, judge_dir=None, judge_seed=None, limit_states=None, name=None) -> dict:
    """J4V-O-Xt on `seed`'s depth-11 cell. `judge_seed` defaults to `seed`; only the timing cell
    passes another."""
    torch.set_num_threads(1)
    t0 = time.time()
    judge_seed = seed if judge_seed is None else judge_seed
    critic = e75.JudgeCritic(cells.load_trained("O-Xt", judge_seed, judge_dir))
    agent, head, states, train_seed = e75.load_cell(DEPTH, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=DEPTH, mode="C", k=K, k_goal=K,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    rec = {**res, "depth": DEPTH, "seed": seed, "judge_seed": judge_seed, "arm": ARM,
           "judge": "O-Xt", "fired4": res["first_fire_len"].get("4", 0),
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "limit_states": limit_states, "judge_updates": _updates(judge_seed, judge_dir)}
    _write(out_dir, name or record_name(seed), rec)
    return rec


def pending_search(seeds, out_dir) -> list[int]:
    return [s for s in seeds if not (Path(out_dir) / record_name(s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate0", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    ap.add_argument("--mint-root", type=Path, default=None, help="--out-dir when omitted")
    ap.add_argument("--judge-dir", type=Path, default=None, help="--out-dir when omitted")
    args = ap.parse_args()
    if args.gate0 + args.search + args.time != 1:
        raise SystemExit("pass exactly one of --gate0, --search or --time")
    judge_dir = args.judge_dir if args.judge_dir is not None else args.out_dir
    if args.gate0:
        r = gate0_cell(args.seed, args.mint_root or args.out_dir, args.out_dir)
        print(f"gate0 X s{r['seed']} margin {r['gate_l']['margin']!r} wall {r['wall_s']}s")
        return
    if args.time:
        r = run_cell(TIME_CELL_SEED, args.out_dir, judge_dir, judge_seed=TIME_JUDGE_SEED,
                     limit_states=TIME_STATES,
                     name=time_record_name(TIME_JUDGE_SEED, TIME_CELL_SEED))
        print(f"time {ARM} judge s{r['judge_seed']} cell s{r['seed']} {r['n']} states "
              f"solved {r['solved']} wall {r['wall_s']}s")
        return
    cells.check_seeds(args.seeds, False)
    jobs = pending_search(args.seeds, args.out_dir)
    print(f"EXP-080 search: {len(jobs)} cells, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_cell, s, args.out_dir, judge_dir): s for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  {ARM} d{DEPTH} s{r['seed']} solved {r['solved']}/{r['n']} "
                  f"wall {r['wall_s']}s", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
