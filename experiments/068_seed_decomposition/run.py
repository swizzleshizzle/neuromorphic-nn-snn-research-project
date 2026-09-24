"""EXP-068: is the seed effect the TASK DRAW or the TRAINING TRAJECTORY?

`docs/seed-effect.md` measured the seed effect (sd 0.0906, +0.419 across 8 arms, 28 of 28 arm
pairs positive) and banked it as a standing confound. It did not say WHERE it lives. A seed fixes
two things at once:

    split_seed   the train/held-out partition, i.e. the TASK DRAW
    train_seed   head init, sampling, scramble stream, readout rng, i.e. the TRAJECTORY

EXP-067 put an accidental number on the split. A machine change preserves the task draw exactly
(its floor arm is byte-equal on 12 of 12 seeds) and redraws the trajectory entirely (0 of 390
parameters), so its cross-machine per-seed correlation estimates the task draw's share of
variance: r = +0.428, r-squared 0.183.

    PREDICTION, out of sample: the task draw is about 18% of per-seed variance, a MINORITY.

> THE CELL IS EXP-036's DEPTH 3, obtained by `dataclasses.replace` on the config EXP-036's own
> `sweep_configs` builds. Never retyped: retyping is how a decomposition silently tests a
> different cell.

> THE FILENAME COLLISION TRAP IS REAL AND DOCUMENTED IN `record_filename` ITSELF, which names "a
> seed-decomposition sweep" as the case that lands every cell in one file. The tag encodes BOTH
> seeds and the guard below uses the real `record_filename`, as that docstring asks.

Spec: docs/superpowers/specs/2026-09-24-exp068-seed-decomposition-design.md

Usage:
    .venv/bin/python -u experiments/068_seed_decomposition/run.py --phase calibrate --workers 10
    .venv/bin/python -u experiments/068_seed_decomposition/run.py --phase full --workers 10 --skip-existing
"""

from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import record_filename, run_cube_baseline

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
EXP036 = HERE.parent / "036_generalisation_gap"

_spec = importlib.util.spec_from_file_location("exp036_run", EXP036 / "run.py")
_exp036 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp036)

DEPTH = 3
GRID = tuple(range(10))          # 10 x 10 = 100 cells, and 10 workers divides it into 10 waves
TASK_SHARE_BAR = 0.35            # Claim 1: CONFIRMED below this. EXP-067 predicts ~0.18
EXP067_PREDICTED_SHARE = 0.183   # r-squared of the cross-machine per-seed correlation
MIN_TOTAL_SD = 0.05              # Claim 3a. EXP-036 measured 0.1158 on this exact cell
EXP036_D3_MEAN = 0.3972
EXP036_D3_SD = 0.1158


def base_config(out_dir: Path):
    """EXP-036's depth-3 trained cell, built by EXP-036's own driver."""
    cells = [
        c for c in _exp036.sweep_configs([0], out_dir)
        if c.depth == DEPTH and c.arm == "regionalized"
    ]
    if len(cells) != 1:
        raise SystemExit(f"expected exactly one EXP-036 depth-{DEPTH} trained cell, got {len(cells)}")
    return cells[0]


def cell_tag(split: int, train: int) -> str:
    """Both seeds in the tag, because `record_filename` encodes neither."""
    return f"exp068_sp{split}tr{train}"


def sweep_configs(out_dir: Path, diagonal_only: bool = False):
    base = base_config(out_dir)
    return [
        replace(
            base,
            tag=cell_tag(split, train),
            seed=split,
            split_seed=split,
            train_seed=train,
            out_dir=out_dir,
        )
        for split in GRID
        for train in GRID
        if not diagonal_only or split == train
    ]


def _run(cfg):
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("calibrate", "full"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = sweep_configs(args.out_dir, diagonal_only=args.phase == "calibrate")

    # Each of these is a way this stops being "EXP-036 depth 3 with two seeds crossed".
    expected = len(GRID) if args.phase == "calibrate" else len(GRID) ** 2
    if len(configs) != expected:
        raise SystemExit(f"expected {expected} cells, built {len(configs)}")
    if any(c.depth != DEPTH for c in configs):
        raise SystemExit(f"every cell must be depth {DEPTH}")
    if any(c.readout != "concept" for c in configs):
        raise SystemExit("EXP-036 uses the concept readout; that must not change")
    if any(c.encoder_state_path is not None for c in configs):
        raise SystemExit("EXP-036 takes NO pretrained encoder. That is what isolates the question.")
    if any(c.critic_lr is not None for c in configs):
        raise SystemExit("EXP-036 has no critic; critic_lr must stay None")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit(
            "record filename collision. `record_filename` encodes neither split_seed nor "
            "train_seed, so the tag must carry both or every cell lands in one file."
        )

    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-068 phase {args.phase!r}: depth {DEPTH}, {len(configs)} cells, "
          f"{args.workers} workers")
    print(f"  split_seed x train_seed over {GRID[0]}-{GRID[-1]}. The DIAGONAL is the ordinary config.")
    print(f"  Cell is EXP-036 depth 3 via dataclasses.replace, never retyped "
          f"(published mean {EXP036_D3_MEAN}, per-seed sd {EXP036_D3_SD}).")
    print(f"  CLAIM 1 (primary): task-draw share of total variance < {TASK_SHARE_BAR}. "
          f"EXP-067 predicts ~{EXP067_PREDICTED_SHARE}.")
    print(f"  CLAIM 2: the TRAJECTORY main effect resolves at p < 0.05.")
    print(f"  GATE: total sd >= {MIN_TOTAL_SD} AND at least one main effect resolves.")
    if args.phase == "calibrate":
        print(f"  This wave is the 10 DIAGONAL cells. It is a COST measurement and its cells are "
              f"real grid cells, so nothing is wasted.\n", flush=True)
    else:
        print(flush=True)

    if not configs:
        print("nothing to do.")
        return
    if args.dry_run:
        print(f"  --dry-run: {len(configs)} cell(s) NOT started.")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            print(f"  {i}/{len(configs)}  {r['tag']}  success {r['success_rate']:.3f}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
