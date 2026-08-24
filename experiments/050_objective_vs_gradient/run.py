# experiments/050_objective_vs_gradient/run.py
"""EXP-050 phase 2 (arm F): E0+ FROZEN with a fresh head. Was it the objective, or just gradient?

Three arms, two of which already exist. All are `frozen encoder + fresh head, 10,000 episodes`,
so RL compute is IDENTICAL at 1.0 unit and the single variable is how the encoder's extra ~10,000
gradient steps were spent:

    A  E0   40 epochs inverse-model                              0.1800  (EXP-043, NOT re-run)
    F  E0+  80 epochs inverse-model (40 + 40 more)               ?       (THIS ARM)
    B  E1   40 epochs inverse-model + 10,000 RL updates          0.3112  (EXP-048, NOT re-run)

PRE-REGISTERED CONTRACT, committed at f569540 before any number existed. Full version:
docs/superpowers/specs/2026-08-24-exp050-objective-vs-gradient-design.md

  1. PRIMARY. F - A: does MORE PRETRAINING improve the encoder at all? >= +0.05 at p <= 0.05.
     EXP-040 recorded that the objective had NOT saturated, so this is live, not a formality.

  2. DECISIVE. F - B: does it MATCH RL fine-tuning? A null is NOT equivalence at n=12 and is
     pre-committed not to be called one.

  3. THE GRID IS FIXED IN THE SPEC, including the outcome that would supersede this whole line:
     if more pretraining BEATS RL fine-tuning, the cheaper offline route wins.

  4. Does the probe's anti-correlation belong to the RL objective specifically? EXP-039 showed
     pretraining RAISES the probe (+0.3396, 12-0); EXP-049 showed RL fine-tuning LOWERS it
     (0-12, p 0.0005) while success nearly doubles. Predicted: E0+ probes HIGHER than E0 while
     arm F gains LESS policy than arm B.

  5. A refuted Claim 1 CLOSES the 'more gradient' alternative and lets three specs lift their
     scope caveats rather than restate them.

> THE STEP MATCH FAVOURS THE CONTROL, ON PURPOSE. 10,360 pretraining updates against RL's 10,000
> is 1.036x, but each pretraining update sees 256 pairs where an RL update sees ~15 env steps.
> The control gets ~17x the data per step and a clean supervised gradient. If it still loses,
> that is a stronger conclusion than a hedged one.

Run (repo root), after extend_pretrain.py:
    .venv/bin/python -u experiments/050_objective_vs_gradient/run.py --workers 6
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
ARM_A_DIR = Path("experiments/043_cap_at_depth_5_6/outputs")

DEPTH = 6
EPISODES = 10_000
CAP = ((1, 2),)
SEEDS = tuple(range(12))
TAG = f"exp050_pre2_d{DEPTH}"

ARM_A_MEAN = 0.1800
ARM_B_MEAN = 0.3112
BAR = 0.05


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def e0plus(out_dir: Path, seed: int) -> Path:
    return out_dir / f"exp050_encoder_plus_s{seed}.pt"


def env_steps() -> int:
    override = dict(CAP)
    return sum(n * override.get(d, max_steps_for(d))
               for d, n in curriculum_schedule(curriculum_for(DEPTH), EPISODES, None))


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    """Identical to EXP-043's depth-6 cell and EXP-048's arm B in EVERY field except which
    frozen encoder is loaded. `encoder_lr` stays None - arm F is a FROZEN arm."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=TAG,
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(DEPTH), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e0plus(out_dir, seed)),
            max_depth=DEPTH, out_dir=out_dir,
        )
        for seed in seeds
    ]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing = [s for s in args.seeds if not e0plus(args.out_dir, s).exists()]
    if missing:
        raise SystemExit(
            f"missing E0+ encoders for seeds {missing}. Run extend_pretrain.py first - arm F "
            "freezes ITS output."
        )
    bad = [s for s in args.seeds
           if not list(ARM_A_DIR.glob(f"exp043_capped_d{DEPTH}_regionalized_d{DEPTH}_s{s}_*.json"))]
    if bad:
        raise SystemExit(f"EXP-043 depth-{DEPTH} records missing for seeds {bad} (arm A).")

    configs = sweep_configs(args.seeds, args.out_dir)
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("arm F must be FROZEN: encoder_lr stays None.")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    print(f"EXP-050 arm F: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} runs, "
          f"{args.workers} workers")
    print(f"  E0+ (80 epochs inverse-model), FROZEN, with a FRESH head. Trainable back to 390.")
    print(f"  {env_steps():,} env steps per run, 1.0x - IDENTICAL RL compute to arms A and B.")
    print(f"    A  E0  (40 epochs)                       {ARM_A_MEAN}")
    print(f"    B  E1  (40 epochs + 10,000 RL updates)   {ARM_B_MEAN}")
    print(f"  ONE VARIABLE: how the encoder's extra ~10,000 gradient steps were spent.")
    print(f"  Claim 1 CONFIRMED at >= +{BAR} over arm A, p <= 0.05.")
    print(f"  Claim 2 (F - B) is the decisive one: does the objective matter, or just gradient?\n",
          flush=True)

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

    print(f"\ndone. records in {args.out_dir}. Run aggregate.py for the verdicts.")


if __name__ == "__main__":
    main()
