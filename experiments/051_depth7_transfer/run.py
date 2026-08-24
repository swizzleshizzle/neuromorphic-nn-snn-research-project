# experiments/051_depth7_transfer/run.py
"""EXP-051: does the encoder gain transfer to depth 7, which it was never trained on?

Every result in weeks 19-20 is at DEPTH 6. The goal is the full 2x2, where a random scramble sits
at depth 11. And E1 was fine-tuned on the curriculum (1..6), so ==depth 7 was never in its
training distribution==. This is an out-of-distribution generalisation test, not a formality.

    EXP-044 arm A   E0 frozen, fresh head, depth 7, 10,000 episodes  ->  0.0621  (NOT re-run)
    EXP-051         E1 frozen, fresh head, depth 7, 10,000 episodes  ->  ?

ONE VARIABLE: which frozen encoder. E1 and not the better E2, deliberately - E1 is the encoder
EXP-048 measured at depth 6, so this is the same comparison moved to a new depth and reads
directly against EXP-048's +0.1312.

PRE-REGISTERED CONTRACT, committed before any number existed. Full version:
docs/superpowers/specs/2026-08-24-exp051-depth7-transfer-design.md

  1. PRIMARY. Delta vs EXP-044 arm A. CONFIRMED at >= +0.05, p <= 0.05.

  2. POINT PREDICTION: complete transfer of EXP-048's +0.1312 predicts 0.0621 + 0.1312 = 0.1933.
     Report the transfer fraction (observed - 0.0621) / 0.1312.

  3. THE HEADLINE COMPARISON: EXP-044 arm B reached 0.1971 at depth 7 and it cost 4.4x THE
     EPISODES. Complete transfer predicts 0.1933 at 1x. If those land together, the encoder buys
     at the frontier what budget buys, for a quarter of the episodes.

  4. CONSISTENCY: depth 7's budget rate back-solves to 0.210/log10 from EXP-044's own two arms,
     against depth 6's 0.22. This arm's budget-equivalent is 0.1392, so complete transfer would
     show an excess of +0.0541 - against +0.0628 / +0.0504 / +0.0540 measured at depth 6. If it
     lands near +0.05 again, constant returns hold ACROSS DEPTH too.

  5. MECHANISM via the instrument that replaced the probe: revisits lower, optimality higher.
     THE PROBE IS DELIBERATELY NOT RUN - EXP-049 showed it moves opposite to policy quality over
     this encoder sequence, so it would add a number nobody should use.

  6. A refuted Claim 1 means the gain is DEPTH-SPECIFIC, bounds EXP-047/048/049 to depth 6, and
     redirects week 21 toward why - most likely shell-specific fitting from the (1..6) cap.

Run (repo root), AFTER EXP-050 finishes:
    .venv/bin/python -u experiments/051_depth7_transfer/run.py --workers 6
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
E1_DIR = Path("experiments/047_encoder_finetuning/outputs")
BASELINE_DIR = Path("experiments/044_depth7_frontier/outputs")

DEPTH = 7
EPISODES = 10_000
CAP = ((1, 2),)
SEEDS = tuple(range(12))
TAG = f"exp051_transfer_d{DEPTH}"

BASELINE_TAG = "exp044_d7_e10000"
BASELINE_MEAN = 0.0621        # EXP-044 arm A
D6_GAIN = 0.1312              # EXP-048, the gain being tested for transfer
COMPLETE_TRANSFER = 0.1933    # 0.0621 + 0.1312
EXP044_ARM_B = 0.1971         # what 4.4x the episodes bought at this depth
BUDGET_EQUIV = 0.1392         # 0.0621 + 0.210*log10(2.33)
BAR = 0.05


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def e1_encoder(seed: int) -> Path:
    return E1_DIR / (f"exp047_ft_d6_lr0.0001_regionalized_d6_s{seed}_sig0.0_encoder.pt")


def env_steps() -> int:
    override = dict(CAP)
    return sum(n * override.get(d, max_steps_for(d))
               for d, n in curriculum_schedule(curriculum_for(DEPTH), EPISODES, None))


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    """Copied from EXP-044 arm A in EVERY field except `encoder_state_path` and the tag."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=TAG,
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(DEPTH), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e1_encoder(seed)),
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
    for s in args.seeds:
        if not e1_encoder(s).exists():
            raise SystemExit(f"missing EXP-047 encoder {e1_encoder(s)} (tracked in git)")
    missing = [s for s in args.seeds
               if not list(BASELINE_DIR.glob(f"{BASELINE_TAG}_regionalized_d{DEPTH}_s{s}_*.json"))]
    if missing:
        raise SystemExit(
            f"EXP-044 arm A records missing for seeds {missing}; they are the PAIRED baseline."
        )

    configs = sweep_configs(args.seeds, args.out_dir)
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("EXP-051 is a FROZEN arm: encoder_lr stays None.")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    print(f"EXP-051: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} runs, "
          f"{args.workers} workers")
    print(f"  E1 (fine-tuned at depth 6) FROZEN + fresh head, at a depth it NEVER TRAINED ON")
    print(f"  {env_steps():,} env steps per run. ONE VARIABLE vs EXP-044 arm A "
          f"({BASELINE_MEAN}): the encoder.")
    print(f"  CONFIRMED at >= +{BAR}, p <= 0.05.")
    print(f"  Complete transfer of EXP-048's +{D6_GAIN} predicts {COMPLETE_TRANSFER}.")
    print(f"  EXP-044 arm B reached {EXP044_ARM_B} at this depth and it cost 4.4x THE EPISODES.")
    print(f"  Budget-equivalent {BUDGET_EQUIV}; complete transfer would be an excess of +0.0541,")
    print(f"  against +0.0628 / +0.0504 / +0.0540 measured at depth 6.")
    print(f"  A REFUTED Claim 1 means the gain is DEPTH-SPECIFIC and bounds EXP-047/048/049.\n",
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

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
