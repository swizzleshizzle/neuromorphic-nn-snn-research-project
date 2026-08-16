# experiments/044_depth7_frontier/run.py
"""EXP-044 driver: where is the break point now? Depth 7.

Depths 3 through 6 all clear the working bar as of EXP-043, and **the location of the break point
has been unknown since 2026-08-13** - the first time in the project it has been. Depth 7 has never
been attempted.

This is NOT a comparison. There is no prior depth-7 arm, so the primary claim is ABSOLUTE - does
depth 7 clear the pre-registered working rule - and **no p-value is attached to it**. The
uncertainty rides on the margin and seed-count conditions instead.

PRE-REGISTERED CONTRACT, committed before any number exists. Full version:
docs/superpowers/specs/2026-08-14-exp044-depth7-frontier-design.md

  1. PRIMARY. WORKING if mean >= BAR, margin >= 1.0 SE above BAR, and >= 8 of 12 seeds above
     BAR, where BAR = max(0.10, 2 x the MEASURED floor). EXP-036's rule with the two conditions
     EXP-043 added after 0.1037 cleared the bare rule on noise.

  2. THE ESCALATION IS PRE-REGISTERED. Per-state coverage falls from 0.190 episodes/state at
     depth 6 to 0.044 at depth 7, so a FAILURE would confound "harder" with "seen a quarter as
     often". Matching depth 6's coverage needs 44,000 episodes, ~52 h against this arm's ~12.
     That is not paid up front, because THE CONFOUND ONLY BITES ON A NEGATIVE: a win at low
     coverage is still a win, and a stronger generalisation result than depth 6's. Arm B runs
     only if Claim 1 refutes, and its reading is fixed in the spec before it exists.

  3. FAILURE COUNTS, descriptive, NO p-value. n=12 cannot show a count went to zero.

  4. VARIANCE, descriptive, against EXP-043's 0.1277 (d5) and 0.0985 (d6).

  5. THE NULL IS PRE-COMMITTED. A refuted Claim 1 with a refuted arm B is the first LOCATED
     break point since EXP-036, not a failed experiment.

> THE FLOOR IS MEASURED, NOT ASSUMED. Standing rule: at depth 1 the chance floor is 21%, not
> 1/6, because a 2d+3 budget lets a random walk cycle home. `arm="random"` short-circuits
> training and only evaluates, so 12 floor seeds cost minutes rather than hours.

Run (repo root):
    .venv/bin/python -u experiments/044_depth7_frontier/run.py \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10 --skip-existing
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.cube_baseline import (
    CubeConfig,
    curriculum_schedule,
    max_steps_for,
    record_filename,
    run_cube_baseline,
    shell_states,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
ENCODERS = Path("experiments/040_pretrained_encoder_policy/outputs")

DEPTH = 7
EPISODES_A = 10_000          # matched to every other depth in the series
EPISODES_B = 44_000          # arm B only: matches depth 6's episodes-per-train-state (0.190)
CAP = ((1, 2),)              # EXP-042's confirmed arm, unchanged
HELDOUT_CAP, HELDOUT_FRAC = 200, 0.25        # CubeConfig defaults; only used for the banner

# EXP-043, for the pre-flight print. The comparison the run is placed against, NOT a paired test.
EXP043_D6 = 0.1800
COVERAGE_D5, COVERAGE_D6 = 0.973, 0.190      # episodes per train state, both at 10,000 episodes


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def cell_tag(episodes: int) -> str:
    """`record_filename` covers tag/arm/depth/seed/sigma and NOT `episodes`, so arm B would
    silently overwrite arm A's records if the budget were not in the tag."""
    return f"exp044_d{DEPTH}_e{episodes}"


def sweep_configs(seeds, out_dir: Path, episodes: int, floor: bool) -> list[CubeConfig]:
    """Arm A (or B) plus the measured floor. Everything except `depth` and `curriculum` is copied
    from EXP-043's depth-6 cell, so the only thing that differs from a working depth is depth."""
    configs = [
        CubeConfig(
            arm="regionalized", readout="concept", tag=cell_tag(episodes),
            depth=DEPTH, seed=seed, sigma=0.0, episodes=episodes,
            curriculum=curriculum_for(DEPTH), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(ENCODERS / f"exp040_encoder_s{seed}.pt"),
            max_depth=DEPTH, out_dir=out_dir,
        )
        for seed in seeds
    ]
    if floor:
        # The floor never trains, so it takes no encoder and no budget.
        configs += [
            CubeConfig(
                arm="random", readout="concept", tag=f"exp044_floor_d{DEPTH}",
                depth=DEPTH, seed=seed, sigma=0.0, episodes=episodes,
                curriculum=curriculum_for(DEPTH), max_steps_by_depth=CAP,
                max_depth=DEPTH, out_dir=out_dir,
            )
            for seed in seeds
        ]
    return configs


def coverage(episodes: int) -> float:
    """Episodes per TRAINING state at the deepest curriculum stage.

    Computed, not a constant. It was a constant until arm B was dispatched, and the banner then
    announced arm A's 0.044 for a run whose entire purpose is to raise it to 0.190 - a log a
    later reader would have taken at face value.
    """
    stage = dict(curriculum_schedule(curriculum_for(DEPTH), episodes, None))[DEPTH]
    shell = len(shell_states(ExactBFSDistance(max_depth=DEPTH), DEPTH))
    train_side = shell - min(HELDOUT_CAP, int(shell * HELDOUT_FRAC))
    return stage / train_side


def env_steps(episodes: int) -> int:
    override = dict(CAP)
    sched = curriculum_schedule(curriculum_for(DEPTH), episodes, None)
    return sum(n * override.get(d, max_steps_for(d)) for d, n in sched)


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--episodes", type=int, default=EPISODES_A,
                    help=f"{EPISODES_A} = arm A. {EPISODES_B} = arm B, WHICH RUNS ONLY IF "
                         "CLAIM 1 REFUTES - see the spec before dispatching it.")
    ap.add_argument("--no-floor", action="store_true",
                    help="skip the measured chance floor. Only sensible once it is recorded.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the pre-flight banner and STOP. Checking the banner by running "
                         "the driver and killing the pipe nearly started a 44,000-episode run on "
                         "the wrong machine, against a seed the laptop was already computing.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.seeds:
        p = ENCODERS / f"exp040_encoder_s{s}.pt"
        if not p.exists():
            raise SystemExit(f"missing pretrained encoder {p}. EXP-044 runs on EXP-040's "
                             "encoders, the same ones EXP-042 and EXP-043 used.")

    configs = sweep_configs(args.seeds, args.out_dir, args.episodes, not args.no_floor)
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})[:5]
        raise SystemExit(f"record filename collision: {dupes}")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    arm = "A" if args.episodes == EPISODES_A else ("B" if args.episodes == EPISODES_B else "?")
    n_train = sum(1 for c in configs if c.arm != "random")
    print(f"EXP-044 arm {arm}: depth {DEPTH}, {args.episodes:,} episodes, "
          f"{len(configs)} runs ({n_train} training + {len(configs) - n_train} floor), "
          f"{args.workers} workers")
    print(f"  {env_steps(args.episodes):,} env steps per training run "
          f"({env_steps(args.episodes) / env_steps(EPISODES_A):.2f}x arm A)")
    print(f"  coverage at the deepest stage: {coverage(args.episodes):.3f} episodes/train state "
          f"against depth 6's {COVERAGE_D6:.3f} and depth 5's {COVERAGE_D5:.3f}")
    print(f"  depth 6 scored {EXP043_D6} here. THIS IS NOT A PAIRED TEST: there is no prior")
    print("  depth-7 arm, so Claim 1 is absolute and carries NO p-value.")
    print("  WORKING needs mean >= BAR, >= 1.0 SE of margin, and >= 8/12 seeds above BAR,")
    print("  where BAR = max(0.10, 2 x the floor this run measures.)\n", flush=True)

    if arm == "B":
        print("  !! ARM B. It runs ONLY if Claim 1 refuted. Check the spec.\n", flush=True)

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
