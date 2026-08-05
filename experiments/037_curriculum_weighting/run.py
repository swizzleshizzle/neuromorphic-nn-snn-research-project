# experiments/037_curriculum_weighting/run.py
"""EXP-037 driver: does the curriculum's stage weighting matter?

`curriculum_schedule` has split the episode budget EQUALLY across stages since EXP-034
introduced it. Nobody chose that; it was the obvious default. EXP-036 made the consequence
explicit: at depth 6 the equal split leaves only 1,666 episodes at depth 6 ITSELF, and
EXP-036's own limitations section flagged that as possibly the whole story of its collapse.

One axis: the SHARE of the budget spent at the evaluated depth, remainder split equally
among the bootstrap stages.

    share   weights        depth-4 schedule (10,000 episodes)      env steps
    12.5%   (7,7,7,3)      2916 / 2916 / 2916 / 1252                  75,008
    25%     (1,1,1,1)      2500 / 2500 / 2500 / 2500                  80,000   <- EXP-036, 0.1591
    50%     (1,1,1,3)      1666 / 1666 / 1666 / 5002                  90,008
    75%     (1,1,1,9)       833 /  833 /  833 / 7501                 100,004
    100%    -              direct training                                     <- EXP-034 refuted

The 25% point is NOT re-run: it is EXP-036's depth-4 cell, same seeds, same machine. The
100% endpoint is direct training, already refuted by EXP-034. Those two known points are
what make this a dose-response with a PREDICTED INTERIOR OPTIMUM rather than a fishing trip.

PRE-REGISTERED CONTRACT, committed before any number exists. Full version and the reasoning
behind each threshold: docs/superpowers/specs/2026-08-05-exp037-curriculum-weighting-design.md

  All comparisons are PAIRED per seed against EXP-036's 25% arm, same twelve seeds, same
  machine, exact permutation over all 2**12 = 4096 sign flips.

  1. IS WEIGHTING A LEVER?
       50% beats 25% by >= 0.03 with p <= 0.05  -> established as a lever.
       otherwise                                -> REFUTED at this budget; the equal default
                                                   stands. Do NOT then hunt for another share.
     The 0.03 bar is set against measured spread: EXP-036's depth-4 arm had sd 0.0739, so
     0.03 is about 0.4 sd and a 19% relative gain on a base of 0.1591.

  2. IS THERE AN INTERIOR OPTIMUM?
       50% > 75%   -> confirmed, consistent with EXP-034's refuted 100% endpoint.
       75% >= 50%  -> no turnover in range; the next experiment is 85/95%, not more seeds.

  3. THE CONTROL. 12.5% must be WORSE than 25%. If it is not, performance is not tracking
     share at the evaluated depth, and any 50% win needs a different explanation than
     starvation. Report it as such rather than keeping the headline.

  4. DEPTH 6, one arm at 50% share (5,000 episodes there instead of 1,666) against EXP-036's
     0.0000 on all twelve seeds. Any seed above zero is worth reporting. Still 0.0000 on all
     twelve means depth 6's failure is NOT starvation but the collapse the instruments
     already showed (modal fraction 0.975), and the lever there is EXP-031/032 territory.

  5. INSTRUMENTS. Report greedy_modal_action_frac and mean_train_entropy per arm. If
     back-loading helps, does it help by REDUCING COLLAPSE? A gain arriving with modal
     fraction falling is a different and more trustworthy mechanism than one that does not.

  6. THE CONFOUND, disclosed rather than hidden. Holding EPISODES fixed does not hold
     COMPUTE fixed: an episode at depth d runs up to 2d+3 steps, so the 75% arm spends 25%
     more environment steps than the 25% arm at an identical episode budget. Episodes stay
     the fixed currency for continuity with EXP-034/035/036. A 75% win by a margin under 25%
     is NOT separable from its extra compute on this design alone; say so. A matched-steps
     arm is the clean resolution and is deliberately deferred.

No random arm: EXP-036 measured the floors at depths 4 and 6 (0.0031 and 0.0008) on this
machine with these seeds. Re-measuring an unchanged quantity would cost 24 runs for nothing.

Run (repo root):
    .venv/bin/python -u experiments/037_curriculum_weighting/run.py \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10
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

EPISODES = 10000

# (depth, share_label, weights). 25% at depth 4 and the equal arm at depth 6 come from
# EXP-036 and are deliberately absent.
CELLS = [
    (4, "0125", (7, 7, 7, 3)),
    (4, "0500", (1, 1, 1, 3)),
    (4, "0750", (1, 1, 1, 9)),
    (6, "0500", (1, 1, 1, 1, 1, 5)),
]

EXP036_BASELINE = {4: 0.1591, 6: 0.0000}   # the arms this is measured against


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def cell_tag(depth: int, share: str) -> str:
    """Unique per cell. record_filename encodes tag/arm/depth/seed/sigma and NOT curriculum
    or curriculum_weights, so the share must live in the tag or records overwrite silently."""
    return f"exp037_s{share}_d{depth}"


def sweep_configs(seeds, out_dir) -> list[CubeConfig]:
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=cell_tag(depth, share),
            depth=depth, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(depth), curriculum_weights=weights,
            max_depth=max(6, depth), out_dir=out_dir,
        )
        for depth, share, weights in CELLS
        for seed in seeds
    ]


def env_steps(depth: int, weights: tuple[int, ...]) -> int:
    """Upper-bound environment steps for one run, for the confound disclosure."""
    sched = curriculum_schedule(curriculum_for(depth), EPISODES, weights)
    return sum(n * max_steps_for(d) for d, n in sched)


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    # 10, not 16. EXP-036 ran 16 workers at 920 MB private each, drove system commit to
    # 48.6 of 50.4 GB, and held utilisation at a measured 43.1% because the workers spent
    # most of their time paging. This is a testable prediction, not a style preference.
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = sweep_configs(args.seeds, args.out_dir)

    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        dupes = {n for n in names if names.count(n) > 1}
        raise SystemExit(f"record filename collision, cells would overwrite: {sorted(dupes)[:5]}")

    print(f"EXP-037: {len(configs)} runs, {args.workers} workers")
    total = 0
    for depth, share, weights in CELLS:
        sched = curriculum_schedule(curriculum_for(depth), EPISODES, weights)
        steps = env_steps(depth, weights)
        total += steps * len(args.seeds)
        share_frac = sched[-1][1] / EPISODES
        print(f"  d{depth} share {int(share)/10000:.3f} (actual {share_frac:.3f}) "
              f"{[n for _, n in sched]}  {steps:,} steps/run")
    print(f"  total {total:,} env steps")
    print(f"  measured against EXP-036: depth 4 = {EXP036_BASELINE[4]}, "
          f"depth 6 = {EXP036_BASELINE[6]}")
    print("  prediction (pre-registered): 50% beats 25%, and 50% beats 75%\n", flush=True)

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. one record + one head checkpoint per run in {args.out_dir}")
    print("Run aggregate.py for the pre-registered verdicts.")


if __name__ == "__main__":
    main()
