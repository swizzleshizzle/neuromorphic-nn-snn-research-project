"""EXP-072: P3V at depths 8 and 9. Re-evaluation only, nothing trains.

New arms G0V, P3V, R3V; their paired references are EXP-070's committed G0, E3, P3 and R3
records. The continuity arms G0 and P3 exist only to re-check, at seed 0, that this harness
reproduces those references exactly (Gate 0(b)).

Spec: docs/superpowers/specs/2026-10-07-exp072-p3v-frontier-design.md

Usage (from the repo root):
    .venv/bin/python -u experiments/072_p3v_frontier/run.py --arms G0V P3V R3V --workers 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location(
    "exp070_cells", REPO / "experiments" / "070_lookahead_existing" / "cells.py")
c70 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c70)

from neuromorphic.training.lookahead import evaluate_lookahead  # noqa: E402

torch.set_num_threads(1)

DEPTHS = (8, 9)
SEEDS = tuple(range(12))
ARMS = [("G", 0, True), ("P", 3, True), ("R", 3, True)]
CONTINUITY_ARMS = [("G", 0, False), ("P", 3, False)]


def arm_name(mode: str, k: int, v: bool) -> str:
    return f"{mode}{k}{'V' if v else ''}"


def parse_arm(text: str):
    for arm in ARMS + CONTINUITY_ARMS:
        if arm_name(*arm) == text:
            return arm
    raise SystemExit(f"unknown arm {text!r}; valid: {[arm_name(*a) for a in ARMS + CONTINUITY_ARMS]}")


def cell_record_name(mode: str, k: int, v: bool, depth: int, seed: int) -> str:
    return f"exp072_{arm_name(mode, k, v)}_d{depth}_s{seed}.json"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_cell(mode, k, v, depth, seed, out_dir: Path, limit_states=None) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    agent, head, states, train_seed = c70.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode=mode, k=k,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed, no_revisit=v)
    cfg = c70.published_config(depth, seed)
    rec = {**res, "depth": depth, "seed": seed, "arm": arm_name(mode, k, v),
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "limit_states": limit_states,
           "head_file": str(c70.head_path(depth, seed).relative_to(c70.REPO)),
           "encoder_file": cfg.encoder_state_path}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cell_record_name(mode, k, v, depth, seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=[arm_name(*a) for a in ARMS])
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit-states", type=int, default=None)
    args = ap.parse_args()

    arms = [parse_arm(a) for a in args.arms]
    bad = [d for d in args.depths if d not in DEPTHS]
    if bad:
        raise SystemExit(f"depths {bad} are outside the spec's {DEPTHS}")
    jobs = [(m, k, v, d, s) for (m, k, v) in arms for d in args.depths for s in args.seeds]
    if args.skip_existing:
        jobs = [j for j in jobs if not (args.out_dir / cell_record_name(*j)).exists()]
    print(f"EXP-072: {len(jobs)} cells, {args.workers} workers, arms {args.arms}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_cell, *j, args.out_dir, args.limit_states): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  {r['arm']} d{r['depth']} s{r['seed']}  "
                  f"solved {r['solved']}/{r['n']}  {r['wall_s']} s", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
