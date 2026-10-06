"""EXP-070: look-ahead on the networks we already have. Re-evaluation only, nothing trains.

Spec: docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md

Usage (from the repo root):
    .venv/bin/python -u experiments/070_lookahead_existing/run.py --arms G0 E1 E2 E3 R1 R2 R3 --workers 20
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
_spec = importlib.util.spec_from_file_location("exp070_cells", HERE / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)

from neuromorphic.training.lookahead import evaluate_lookahead  # noqa: E402

torch.set_num_threads(1)


def parse_arm(text: str) -> tuple[str, int]:
    arm = (text[0], int(text[1:]))
    if arm not in cells.ARMS:
        raise SystemExit(f"unknown arm {text!r}; valid: {[m + str(k) for m, k in cells.ARMS]}")
    return arm


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_cell(mode, k, depth, seed, out_dir: Path, limit_states=None) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    agent, head, states, train_seed = cells.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode=mode, k=k,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed)
    cfg = cells.published_config(depth, seed)
    rec = {**res, "depth": depth, "seed": seed, "wall_s": round(time.time() - t0, 1),
           "git_commit": _git_commit(), "limit_states": limit_states,
           "head_file": str(cells.head_path(depth, seed).relative_to(cells.REPO)),
           "encoder_file": cfg.encoder_state_path}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cells.cell_record_name(mode, k, depth, seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=[m + str(k) for m, k in cells.ARMS])
    ap.add_argument("--depths", type=int, nargs="+", default=list(cells.DEPTHS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit-states", type=int, default=None)
    args = ap.parse_args()

    arms = [parse_arm(a) for a in args.arms]
    jobs = [(m, k, d, s) for (m, k) in arms for d in args.depths for s in args.seeds]
    if args.skip_existing:
        jobs = [j for j in jobs if not (args.out_dir / cells.cell_record_name(*j)).exists()]
    print(f"EXP-070: {len(jobs)} cells, {args.workers} workers, arms {args.arms}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_cell, *j, args.out_dir, args.limit_states): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  {r['mode']}{r['k']} d{r['depth']} s{r['seed']}  "
                  f"solved {r['solved']}/{r['n']}  {r['wall_s']} s", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
