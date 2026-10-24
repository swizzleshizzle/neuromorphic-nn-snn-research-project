"""EXP-081 mint: EXP-080's mint_seed at 16,000 blocks of 1000. Blocks are drawn in order from
one random.Random(seed) with EXP-080's exclusion, so the first 4000 are EXP-080's; Gate C checks
that byte for byte rather than assuming it.

Spec: docs/superpowers/specs/2026-10-23-exp081-oxt-long-budget-design.md, section 3.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/081_oxt_long_budget/mint.py --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp081_cells", HERE / "cells.py")
m80 = _load("exp080_mint", cells.E80 / "mint.py")

mint_dir = m80.mint_dir
read_mint = m80.read_mint


def mint_seed(seed, out_dir) -> dict:
    return m80.mint_seed(seed, out_dir, n_blocks=cells.N_BLOCKS, block=cells.BATCH,
                         max_len=cells.MAX_LEN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = m80.pending_seeds(args.seeds, args.out_dir)
    print(f"EXP-081 mint: {len(jobs)} seeds, {args.workers} workers, {cells.N_BLOCKS} blocks, "
          f"into {mint_dir(args.out_dir)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(mint_seed, s, args.out_dir): s for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            m = f.result()
            print(f"  {i}/{len(jobs)}  seed {m['seed']}  n_blocks {m['n_blocks']}  "
                  f"wall_s {m['wall_s']}  sha {m['sha256'][:12]}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
