# experiments/045_budget_vs_coverage/run.py
"""EXP-045 driver: was EXP-044 arm B's gain the total budget, or the deepest shell?

Arm B took depth 7 from 0.0621 to 0.1971 by going from 10,000 episodes to 44,000. That raised
BOTH the total budget and the episodes at the deepest shell (1,432 -> 6,290), and arm B cannot
separate them. This arm gives depth 7 arm B's deepest-shell exposure on arm A's BUDGET, by
weighting the curriculum instead of enlarging it.

    EXP-044 arm A   10,000 episodes, uniform          -> stage 7 gets 1,432   (coverage 0.044)
    EXP-044 arm B   44,000 episodes, uniform          -> stage 7 gets 6,290   (coverage 0.191)
    EXP-045         10,000 episodes, weights ..,10    -> stage 7 gets 6,250   (coverage 0.190)

H-coverage predicts this reproduces arm B's gain. H-budget predicts it does not.
**EXP-037 predicts it HURTS**: at depth 4 with a fixed budget, shifting share toward the evaluated
depth dropped success monotonically, 0.1591 -> 0.1078 -> 0.0921. That prediction is what makes
this a risky test rather than a confirmation.

PRE-REGISTERED CONTRACT, committed before any number exists. Full version:
docs/superpowers/specs/2026-08-17-exp045-budget-vs-coverage-design.md

  1. PRIMARY, and PAIRED - unlike EXP-044 there IS a baseline. Delta against arm A per seed,
     exact permutation over 2**12. H-coverage CONFIRMED at >= +0.05 with p <= 0.05. Arm B's gain
     was +0.1350, so the bar is a third of it. A delta between 0 and +0.05, or positive at
     p > 0.05, refutes H-coverage as THE explanation without showing the effect is zero.

  2. WORKING BAR, same rule as EXP-044 for comparability: mean >= 0.10, >= 1.0 SE, >= 8/12 seeds.
     The floor is already measured at exactly 0.0000, so 0.10 binds and no floor arm is re-run.

  3. MECHANISM, descriptive: greedy_modal_action_frac against arm A's, and against EXP-037's
     back-loading trend (0.685 -> 0.757). If back-loading hurts, this says whether it hurts by
     driving collapse.

  4. FAILURE COUNTS, descriptive, NO p-value. n=12 cannot show a count went to zero.

  5. THE NULL IS PRE-COMMITTED AND IS THE INTERESTING OUTCOME. A refuted Claim 1 means the
     operative variable is TOTAL BUDGET, that EXP-037's decline survives a pretrained encoder and
     three more depths, and that every "depth N stopped working" result here is confounded with
     episodes. The next experiment would then be depth 6 at raised total budget.

> THIS DOES NOT REOPEN EXP-037'S REFUTATION. That answered "is weighting a lever for performance"
> - it is not, and it stays closed. This asks which quantity explains EXP-044 arm B, at depth 7,
> with a pretrained encoder and the depth-1 cap.

Run (repo root):
    .venv/bin/python -u experiments/045_budget_vs_coverage/run.py \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 12 --skip-existing
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
BASELINE = Path("experiments/044_depth7_frontier/outputs")

DEPTH = 7
EPISODES = 10_000                    # arm A's budget, unchanged. That is the whole point.
WEIGHTS = (1, 1, 1, 1, 1, 1, 10)     # -> stage 7 gets 6,250 of the 10,000
CAP = ((1, 2),)                      # EXP-042's confirmed arm, unchanged
HELDOUT_CAP, HELDOUT_FRAC = 200, 0.25

# EXP-044, for the pre-flight print. Arm A is the PAIRED baseline.
ARM_A_MEAN, ARM_B_MEAN = 0.0621, 0.1971
ARM_A_STAGE7, ARM_B_STAGE7 = 1_432, 6_290


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def cell_tag() -> str:
    """`record_filename` covers tag/arm/depth/seed/sigma and NOT `curriculum_weights`, so this
    would overwrite EXP-044 arm A's records if the tag did not differ."""
    return f"exp045_backloaded_d{DEPTH}"


def schedule(weights):
    return dict(curriculum_schedule(curriculum_for(DEPTH), EPISODES, weights))


def coverage(weights) -> float:
    shell = len(shell_states(ExactBFSDistance(max_depth=DEPTH), DEPTH))
    train_side = shell - min(HELDOUT_CAP, int(shell * HELDOUT_FRAC))
    return schedule(weights)[DEPTH] / train_side


def env_steps(weights) -> int:
    override = dict(CAP)
    return sum(n * override.get(d, max_steps_for(d))
               for d, n in curriculum_schedule(curriculum_for(DEPTH), EPISODES, weights))


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    """Copied from EXP-044 arm A in every field except `curriculum_weights` and the tag."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=cell_tag(),
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(DEPTH), curriculum_weights=WEIGHTS,
            max_steps_by_depth=CAP, entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(ENCODERS / f"exp040_encoder_s{seed}.pt"),
            max_depth=DEPTH, out_dir=out_dir,
        )
        for seed in seeds
    ]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the pre-flight banner and STOP, without starting anything.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.seeds:
        p = ENCODERS / f"exp040_encoder_s{s}.pt"
        if not p.exists():
            raise SystemExit(f"missing pretrained encoder {p}. EXP-045 runs on EXP-040's "
                             "encoders, the same ones EXP-042/043/044 used.")
    # The paired baseline must exist, or Claim 1 cannot be computed at all.
    missing = [s for s in args.seeds
               if not list(BASELINE.glob(f"exp044_d{DEPTH}_e10000_regionalized_d{DEPTH}_s{s}_*.json"))]
    if missing:
        raise SystemExit(f"EXP-044 arm A records missing for seeds {missing}. They are the PAIRED "
                         f"baseline for Claim 1; fetch them into {BASELINE} first.")

    configs = sweep_configs(args.seeds, args.out_dir)
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit(f"record filename collision: {sorted({n for n in names if names.count(n) > 1})[:5]}")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    sched = schedule(WEIGHTS)
    print(f"EXP-045: depth {DEPTH}, {EPISODES:,} episodes, weights {WEIGHTS}, "
          f"{len(configs)} runs, {args.workers} workers")
    print(f"  stage {DEPTH} gets {sched[DEPTH]:,} episodes (arm A {ARM_A_STAGE7:,}, "
          f"arm B {ARM_B_STAGE7:,}); shallow stages {sched[1]:,} each")
    print(f"  coverage at stage {DEPTH}: {coverage(WEIGHTS):.3f} "
          f"(arm A 0.044, arm B 0.191, depth 6 at 10k 0.190)")
    print(f"  {env_steps(WEIGHTS):,} env steps per run, {env_steps(WEIGHTS) / 105_740:.2f}x arm A")
    print(f"  ONE VARIABLE vs arm A: the stage weighting. Same 10,000 episodes.")
    print(f"  H-coverage CONFIRMED at delta >= +0.05, p <= 0.05 vs arm A's {ARM_A_MEAN:.4f}.")
    print(f"  EXP-037 PREDICTS THIS HURTS (0.1591 -> 0.1078 -> 0.0921 as share rose at depth 4).")
    print(f"  A refuted Claim 1 means the operative variable is TOTAL BUDGET.\n", flush=True)

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
