# experiments/046_depth6_budget/run.py
"""EXP-046 driver: is the depth series a BUDGET series? Depth 6 at 4.4x.

EXP-044 took depth 7 from 0.0621 to 0.1971 with 4.4x the episodes. EXP-045 showed the operative
variable is TOTAL BUDGET, not exposure at the deepest shell - moving budget toward the deep end at
a fixed total scored 0.0142, a paired -0.0479 at p 0.0010, with the policy collapsing outright.

So: was depth 7 specifically starved, or is every depth here budget-limited? Every published depth
in this project was measured at a fixed 10,000 episodes. This gives depth 6 the SAME 4.4x
multiplier that worked at depth 7, paired against EXP-043's depth-6 cell.

    EXP-043 depth 6   10,000 episodes  ->  0.1800   (the paired baseline, NOT re-run)
    EXP-046 depth 6   44,000 episodes  ->  ?

PRE-REGISTERED CONTRACT, committed before any number exists. Full version:
docs/superpowers/specs/2026-08-18-exp046-depth6-budget-design.md

  1. PRIMARY, PAIRED. Delta against EXP-043 depth 6, exact permutation over 2**12. CONFIRMED at
     >= +0.05 with p <= 0.05. Depth 7's gain from the same multiplier was +0.1350, and +0.05 is
     the same bar EXP-043 and EXP-045 used, so the three are comparable. A delta between 0 and
     +0.05, or positive at p > 0.05, refutes the strong reading WITHOUT showing the effect is zero.

  2. ESCALATION, conditional. If Claim 1 CONFIRMS, run a 25,000-episode midpoint (~13 h) to see
     whether returns are still climbing or flattening. If it REFUTES, the midpoint is NOT run.

  3. MECHANISM, descriptive: the deepest stage's entropy trace and training solve rate against
     EXP-045's collapse signature (0.5914 -> 0.0979, min 2.7e-06, solve rate 0.0218).

  4. FAILURE COUNTS, descriptive, NO p-value. n=12 cannot show a count went to zero.

  5. THE NULL IS PRE-COMMITTED AND IS A REAL RESULT. A refuted Claim 1 means depth 7 was
     SPECIFICALLY starved - most plausibly because its shell is 3.7x depth 6's - and the series is
     not simply a budget series. Earlier numbers would then stand as published, with "at 10,000
     episodes" as a caveat rather than a correction.

> WHY 44,000 AND NOT 51,000: an earlier note proposed 51,000 to match a COVERAGE figure, and
> EXP-045 refuted coverage as the operative variable. 44,000 is the multiplier that worked at
> depth 7, which makes this a direct analogue rather than a new quantity.

> NO FLOOR ARM: EXP-036 measured depth 6's floor at 0.0008, so the 0.10 working bar binds.

Run (repo root):
    .venv/bin/python -u experiments/046_depth6_budget/run.py \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 12 --skip-existing
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
ENCODERS = Path("experiments/040_pretrained_encoder_policy/outputs")
BASELINE = Path("experiments/043_cap_at_depth_5_6/outputs")

DEPTH = 6
EPISODES = 44_000            # 4.4x, the multiplier that worked at depth 7
EPISODES_MID = 25_000        # Claim 2 only, and only if Claim 1 confirms
BASELINE_EPISODES = 10_000
CAP = ((1, 2),)              # EXP-042's confirmed arm, unchanged

BASELINE_MEAN = 0.1800       # EXP-043 depth 6
D7_GAIN = 0.1350             # EXP-044 arm A -> arm B, same multiplier


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def cell_tag(episodes: int) -> str:
    """`record_filename` covers tag/arm/depth/seed/sigma and NOT `episodes`, so the budget lives
    in the tag or this would overwrite EXP-043's depth-6 records."""
    return f"exp046_d{DEPTH}_e{episodes}"


def env_steps(episodes: int) -> int:
    override = dict(CAP)
    return sum(n * override.get(d, max_steps_for(d))
               for d, n in curriculum_schedule(curriculum_for(DEPTH), episodes, None))


def sweep_configs(seeds, out_dir: Path, episodes: int) -> list[CubeConfig]:
    """Copied from EXP-043's depth-6 cell in every field except `episodes` and the tag."""
    return [
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


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--episodes", type=int, default=EPISODES,
                    help=f"{EPISODES} = the primary arm. {EPISODES_MID} = the Claim 2 midpoint, "
                         "WHICH RUNS ONLY IF CLAIM 1 CONFIRMS - see the spec first.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the pre-flight banner and STOP, without starting anything.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.seeds:
        p = ENCODERS / f"exp040_encoder_s{s}.pt"
        if not p.exists():
            raise SystemExit(f"missing pretrained encoder {p}. EXP-046 runs on EXP-040's "
                             "encoders, the same ones EXP-042/043/044/045 used.")
    missing = [s for s in args.seeds
               if not list(BASELINE.glob(f"exp043_capped_d{DEPTH}_regionalized_d{DEPTH}_s{s}_*.json"))]
    if missing:
        raise SystemExit(f"EXP-043 depth-6 records missing for seeds {missing}. They are the "
                         f"PAIRED baseline for Claim 1; fetch them into {BASELINE} first.")

    configs = sweep_configs(args.seeds, args.out_dir, args.episodes)
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit(f"record filename collision: {sorted({n for n in names if names.count(n) > 1})[:5]}")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    mult = args.episodes / BASELINE_EPISODES
    print(f"EXP-046: depth {DEPTH}, {args.episodes:,} episodes ({mult:.1f}x the baseline), "
          f"{len(configs)} runs, {args.workers} workers")
    print(f"  {env_steps(args.episodes):,} env steps per run, "
          f"{env_steps(args.episodes) / env_steps(BASELINE_EPISODES):.2f}x EXP-043's depth-6 cell")
    print(f"  ONE VARIABLE vs EXP-043 depth 6 ({BASELINE_MEAN}): the episode budget.")
    print(f"  Depth 7 gained {D7_GAIN:+.4f} from this same multiplier. CONFIRMED at >= +0.05, "
          f"p <= 0.05.")
    print(f"  A refuted Claim 1 means depth 7 was SPECIFICALLY starved and the series is not")
    print(f"  simply a budget series - a real result, not a null.\n", flush=True)

    if args.episodes == EPISODES_MID:
        print("  !! MIDPOINT ARM. It runs ONLY if Claim 1 confirmed. Check the spec.\n", flush=True)
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
