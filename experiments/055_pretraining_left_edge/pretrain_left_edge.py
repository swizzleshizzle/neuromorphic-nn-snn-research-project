"""EXP-055 phase 1: pretrain encoders at 1, 2, 3 and 5 epochs, from scratch.

Two independent instruments say everything happens before 10 epochs and nothing after - the
policy goes 0.0000 to 0.2012 then flat, and EXP-054's `S` goes 0.0100 to 0.0242 then flat.
Nobody has measured a point inside that window. These are those points.

REUSES EXP-052's `pretrain_one` UNCHANGED. That function applies EXP-040's `rl_heldout_union`
exclusions and asserts no RL held-out state leaked in as either endpoint of a training pair.
Without them an arm could win by leakage rather than by epochs. Do not reimplement it.

Run (repo root):
    .venv/bin/python -u experiments/055_pretraining_left_edge/pretrain_left_edge.py --workers 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

# EXP-052's sweep is a script, not a package, so it is loaded by path exactly as it loads
# EXP-040's driver.
_SWEEP_PATH = HERE.parent / "052_pretraining_optimum" / "pretrain_sweep.py"
_spec = importlib.util.spec_from_file_location("exp052_sweep", _SWEEP_PATH)
exp052 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp052)

EPOCH_ARMS = (1, 2, 3, 5)
SEEDS = tuple(range(12))


def encoder_path(out_dir: Path, epochs: int, seed: int) -> Path:
    """EXP-055's own name. EXP-052's encoders share the epoch and seed fields, so reusing its
    naming would let an arm silently load the wrong encoder."""
    return Path(out_dir) / f"exp055_encoder_e{epochs}_s{seed}.pt"


def _pretrain(epochs: int, seed: int, out_dir: Path) -> dict:
    torch.set_num_threads(1)
    result = exp052.pretrain_one(epochs, seed, out_dir)
    # `pretrain_one` writes EXP-052's filename; rename to EXP-055's so the two sets cannot be
    # confused, and so `run.py` can find them by this experiment's convention.
    src = exp052.encoder_path(out_dir, epochs, seed)
    dst = encoder_path(out_dir, epochs, seed)
    src.replace(dst)
    return {**result, "encoder": dst.name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=list(EPOCH_ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = [(e, s) for e in args.epochs for s in args.seeds]
    if args.skip_existing:
        cells = [(e, s) for e, s in cells if not encoder_path(args.out_dir, e, s).exists()]

    print(f"EXP-055 phase 1: epochs {tuple(args.epochs)}, {len(cells)} encoders, "
          f"{args.workers} workers")
    print("  FROM SCRATCH, not warm-started. EXP-040's rl_heldout_union exclusions apply.")
    print("  Pretraining is memory-bandwidth-bound: about 2.86 effective cores from 10 workers.\n",
          flush=True)
    if args.dry_run or not cells:
        print(f"  {len(cells)} cell(s) NOT started.")
        return

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_pretrain, e, s, args.out_dir): (e, s) for e, s in cells}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            print(f"  {i}/{len(cells)}", flush=True)

    (args.out_dir / "pretrain_left_edge.json").write_text(
        json.dumps(records), encoding="utf-8")
    print(f"\ndone. encoders in {args.out_dir}.")


if __name__ == "__main__":
    main()
