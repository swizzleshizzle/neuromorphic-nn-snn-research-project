"""Is the E0 encoder chain actually reproducible? Regenerate and compare bytes.

`.gitignore` excludes `exp040_encoder_s*.pt` on a documented and reasonable ground: they are
"reproducible from EXP-040's phase 1". **FIFTEEN experiment `run.py` files load them**, including
EXP-059, 061, 063, 064 and 066, so every one of those results rests on that claim - and the claim
has never been tested. If regeneration is not byte-identical, a fresh checkout reproduces none of
them, and a public release would be shipping a reproducibility guarantee it does not have.

This regenerates one or more encoders into a scratch directory and compares them against the
copies already on disk, byte for byte.

> RUN IT ON BOTH MACHINES, because they answer different questions. On the laptop, where the
> originals were made, it tests DETERMINISM: does the same code on the same machine produce the
> same weights twice. On any other machine it tests PORTABILITY, which is the claim a public
> release actually makes. Determinism passing does not imply portability passing: different
> BLAS kernels, thread counts and CPU instruction sets all reorder float accumulation.

Nothing is overwritten: the regenerated copies go to `--scratch`, and the originals are only
ever read.

Usage:
    .venv/bin/python -u experiments/040_pretrained_encoder_policy/verify_regeneration.py --seeds 0
    .venv/bin/python -u experiments/040_pretrained_encoder_policy/verify_regeneration.py \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp040_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _one(args):
    seed, scratch = args
    torch.set_num_threads(1)
    t0 = time.time()
    row = _run.pretrain_one(seed, scratch)
    row["wall"] = round(time.time() - t0, 1)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--original-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--scratch", type=Path, default=HERE / "outputs_regen_check")
    args = ap.parse_args()

    args.scratch.mkdir(parents=True, exist_ok=True)

    originals = {s: _run.encoder_path(args.original_dir, s) for s in args.seeds}
    absent = [s for s, p in originals.items() if not p.exists()]
    if absent:
        raise SystemExit(
            f"no original encoder for seeds {absent} in {args.original_dir}. There is nothing to "
            "compare against; fetch them before running this."
        )

    print(f"E0 REGENERATION CHECK - {len(args.seeds)} seed(s), {args.workers} worker(s)")
    print(f"  originals : {args.original_dir}")
    print(f"  scratch   : {args.scratch}")
    print( "  ON THE LAPTOP this tests DETERMINISM; anywhere else it tests PORTABILITY, which")
    print( "  is the claim a public release makes. 15 experiments depend on the answer.\n",
          flush=True)

    jobs = [(s, args.scratch) for s in args.seeds]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_one, jobs))
    else:
        rows = [_one(j) for j in jobs]

    print(f"{'seed':>5s} {'identical':>10s} {'original':>18s} {'regenerated':>18s} "
          f"{'move_acc':>9s} {'wall_s':>8s}")
    identical = 0
    for row in sorted(rows, key=lambda r: r["seed"]):
        s = row["seed"]
        a, b = originals[s], _run.encoder_path(args.scratch, s)
        same = a.read_bytes() == b.read_bytes()
        identical += int(same)
        print(f"{s:>5d} {'YES' if same else 'NO':>10s} {sha(a):>18s} {sha(b):>18s} "
              f"{row['move_accuracy']:>9.4f} {row['wall']:>8.1f}")

    print(f"\n{identical} of {len(rows)} byte-identical.")
    if identical == len(rows):
        print("REPRODUCIBLE on this machine. The .gitignore's reasoning holds here.")
    else:
        print("NOT REPRODUCIBLE. Every experiment that loads these encoders - EXP-041, 042, 043,")
        print("044, 045, 046, 047, 053, 054, 059, 061, 063, 064, 066 - would differ from its")
        print("recorded result on a fresh checkout. The encoders must be tracked, or the")
        print("regeneration made deterministic, before any public release claims otherwise.")


if __name__ == "__main__":
    main()
