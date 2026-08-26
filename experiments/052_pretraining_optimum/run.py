# experiments/052_pretraining_optimum/run.py
"""EXP-052 phase 2: each swept encoder FROZEN with a fresh head, depth 6.

All arms are `frozen encoder + fresh head, 10,000 episodes` with IDENTICAL RL compute, so the
single variable is HOW MANY EPOCHS the encoder was pretrained for.

    epochs   depth-6 policy   source
         0           0.0000   EXP-036 (random frozen encoder)
        10                ?   THIS EXPERIMENT
        20                ?   THIS EXPERIMENT - THE PRIMARY
        40           0.1800   EXP-043 / EXP-050 arm A
        80           0.0887   EXP-050 arm F (warm-started)

The curve rises from 0 and falls by 80, so an interior optimum exists. This locates it.

PRE-REGISTERED CONTRACT, committed at 3fcf21e before any number existed. Full version:
docs/superpowers/specs/2026-08-26-exp052-pretraining-optimum-design.md

  1. PRIMARY, PAIRED, SINGLE COMPARISON: 20 epochs minus 40 epochs. CONFIRMED at >= +0.05,
     p <= 0.05. The 10-epoch arm is EXPLORATORY and carries NO BAR - testing both against 40 and
     reporting the winner would inflate the false-positive rate.

     A confirmed Claim 1 is a CORRECTION TO THE WHOLE SERIES, not a new capability: every
     frozen-encoder result since EXP-040 would have run from a needlessly over-trained encoder.

  2. THE CURVE, descriptive, four points. All three shapes have their reading fixed in the spec,
     including the one most worth being ready for: if 10 > 20 > 40 the peak is BELOW 10 and the
     EXP-039/040 premise is substantially weaker than believed.

  3. Does the pretraining metric predict policy? PREDICTED NO - move-accuracy should rise
     monotonically with epochs while policy peaks and falls, as EXP-050 already saw once. If so,
     the only signal available without running the RL arm is useless for choosing epochs.

  4. Trajectory metrics across arms. THE PROBE IS DELIBERATELY NOT RUN.

  5. A refuted Claim 1 with the peak at 40 means the inherited configuration was accidentally
     correct - worth establishing, because it converts an unexamined assumption into a measured
     one instead of leaving a standing doubt under every prior result.

Run (repo root), after pretrain_sweep.py:
    .venv/bin/python -u experiments/052_pretraining_optimum/run.py --workers 6
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import (
    CubeConfig,
    curriculum_schedule,
    max_steps_for,
    record_filename,
    run_cube_baseline,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
ARM_40_DIR = Path("experiments/043_cap_at_depth_5_6/outputs")

DEPTH = 6
EPISODES = 10_000
CAP = ((1, 2),)
SEEDS = tuple(range(12))
EPOCH_ARMS = (10, 20)

ARM_40_TAG = "exp043_capped_d6"
ARM_40_MEAN = 0.1800
ARM_80_MEAN = 0.0887
BAR = 0.05


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def tag_for(epochs: int) -> str:
    """`record_filename` does not encode the encoder, so the epoch count must live in the tag or
    the arms would silently overwrite each other."""
    return f"exp052_e{epochs}_d{DEPTH}"


def encoder_path(out_dir: Path, epochs: int, seed: int) -> Path:
    return out_dir / f"exp052_encoder_e{epochs}_s{seed}.pt"


def env_steps() -> int:
    override = dict(CAP)
    return sum(n * override.get(d, max_steps_for(d))
               for d, n in curriculum_schedule(curriculum_for(DEPTH), EPISODES, None))


def sweep_configs(seeds, out_dir: Path, epoch_arms) -> list[CubeConfig]:
    """Copied from EXP-043's depth-6 cell in EVERY field except the encoder and the tag."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=tag_for(e),
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(DEPTH), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(encoder_path(out_dir, e, seed)),
            max_depth=DEPTH, out_dir=out_dir,
        )
        for e in epoch_arms
        for seed in seeds
    ]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=list(EPOCH_ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing = [(e, s) for e in args.epochs for s in args.seeds
               if not encoder_path(args.out_dir, e, s).exists()]
    if missing:
        raise SystemExit(f"missing swept encoders {missing[:5]}... Run pretrain_sweep.py first.")
    bad = [s for s in args.seeds
           if not list(ARM_40_DIR.glob(f"{ARM_40_TAG}_regionalized_d{DEPTH}_s{s}_*.json"))]
    if bad:
        raise SystemExit(f"EXP-043 depth-{DEPTH} records missing for seeds {bad} (the 40-epoch arm).")

    configs = sweep_configs(args.seeds, args.out_dir, args.epochs)
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("every arm here is FROZEN: encoder_lr stays None.")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    print(f"EXP-052 phase 2: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} runs, "
          f"{args.workers} workers")
    print(f"  epoch arms {args.epochs}, each FROZEN + fresh head. Trainable 390 throughout.")
    print(f"  {env_steps():,} env steps per run - IDENTICAL RL compute across every arm.")
    print(f"  known: 0 -> 0.0000 (EXP-036), 40 -> {ARM_40_MEAN}, 80 -> {ARM_80_MEAN} (EXP-050)")
    print(f"  PRIMARY is 20 vs 40 at >= +{BAR}, p <= 0.05. The 10-epoch arm carries NO BAR.")
    print(f"  A confirmed Claim 1 CORRECTS THE WHOLE SERIES: every frozen-encoder result since")
    print(f"  EXP-040 would have started from a needlessly over-trained encoder.\n", flush=True)

    if not configs:
        print("nothing to do.")
        return
    if args.dry_run:
        print(f"  --dry-run: {len(configs)} cell(s) NOT started.")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
