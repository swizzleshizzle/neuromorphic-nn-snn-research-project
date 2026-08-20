# experiments/047_encoder_finetuning/pretrain_seeds.py
"""Generate EXP-040 pretrained encoders for seeds that do not have one yet.

Only seeds 0-11 were ever pretrained (`exp040_encoder_s0..11.pt`). Two separate pieces of work
need more:

  - EXP-047's pilot runs on seeds 12-13, deliberately disjoint from the confirmatory set so that
    selecting `encoder_lr` cannot tune a claim's metric (spec section 5.1).
  - The depth-5-at-24-seeds follow-up needs seeds 12-23 to settle EXP-043's Claim 1.

This runs EXP-040's `pretrain_one` **unmodified**, imported by file path, and writes into
EXP-040's own outputs directory under its own naming. That is deliberate: these encoders must be
indistinguishable from seeds 0-11's, produced by the same procedure with the same held-out
exclusions (`rl_heldout_union` over depths 4, 5, 6), or the new seeds are not poolable with the
old ones and neither follow-up is valid.

Phase 2 of EXP-040 is NOT run. Calling `run.py --seeds 12 ...` directly would also launch 36 RL
cells for depths 4/5/6, which is not what either follow-up wants.

Idempotent: seeds whose encoder already exists are skipped, so an interrupted run resumes rather
than recomputing ~15 minutes per banked seed.

Run (repo root):
    .venv/bin/python -u experiments/047_encoder_finetuning/pretrain_seeds.py --seeds 12 13
    .venv/bin/python -u experiments/047_encoder_finetuning/pretrain_seeds.py \
        --seeds 12 13 14 15 16 17 18 19 20 21 22 23 --workers 12
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

# EXP-040's driver, imported UNMODIFIED and by file path. `experiments` is not a package and its
# directory names start with digits, so a normal import is impossible - the same pattern EXP-039
# uses to reach EXP-033's probe. Reimplementing `pretrain_one` would silently produce encoders
# that are not comparable with seeds 0-11's, which is the one thing this script must not do.
_EXP040_PATH = HERE.parent / "040_pretrained_encoder_policy" / "run.py"
_spec = importlib.util.spec_from_file_location("exp040_run", _EXP040_PATH)
exp040 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp040)

OUT_DIR = Path("experiments/040_pretrained_encoder_policy/outputs")


def _pretrain(seed: int, out_dir: Path) -> dict:
    torch.set_num_threads(1)
    return exp040.pretrain_one(seed, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in args.seeds if not exp040.encoder_path(args.out_dir, s).exists()]
    have = len(args.seeds) - len(todo)

    print(f"EXP-040 pretraining via {_EXP040_PATH}")
    print(f"  {len(todo)} encoder(s) to build, {have} already present, {args.workers} workers")
    print(f"  config {exp040.PRETRAIN}, held-out exclusions over depths {exp040.DEPTHS}")
    print(f"  -> {args.out_dir}\n", flush=True)

    if not todo:
        print("nothing to do.")
        return
    if args.dry_run:
        print(f"  --dry-run: {len(todo)} encoder(s) NOT started.")
        return

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_pretrain, s, args.out_dir): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            print(f"  {i}/{len(todo)} seed {row['seed']}: {row['n_pairs']} pairs, "
                  f"{row['n_forbidden']} excluded, move-acc {row['move_accuracy']:.3f}, "
                  f"{row['seconds']}s", flush=True)

    for s in args.seeds:
        path = exp040.encoder_path(args.out_dir, s)
        if not path.exists():
            raise SystemExit(f"encoder for seed {s} missing after pretraining: {path}")

    print(f"\ndone in {time.time() - t0:.0f}s. {len(args.seeds)} encoder(s) available.")


if __name__ == "__main__":
    main()
