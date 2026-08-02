# experiments/035_budget_scaling/run.py
"""EXP-035 driver: does the curriculum gain saturate, or keep climbing?

EXP-034 established the curriculum as the active ingredient and left one clear
limitation: only two budget points, and the trend had not saturated.

    curriculum @  600 episodes -> 0.0972
    curriculum @ 3000 episodes -> 0.2556      (oracle ceiling 0.481, EXP-033)

Two points cannot distinguish "still climbing toward the ceiling" from "about to level off
well below it", and those imply completely different next experiments. This adds 10,000 and
30,000 so the curve has four points spanning 50x.

`direct` is NOT re-run here. EXP-034 refuted volume-alone at 3000 episodes decisively
(-0.003, W-L-T 2-1-9, p = 1.000) with a mechanism: modal fraction rose and entropy fell, so
the extra training drove the policy further into determinism rather than nowhere. Spending
another twelve 30,000-episode runs to re-refute it would cost more compute than the rest of
this experiment and is not where the uncertainty lives.

PRE-REGISTERED CONTRACT, committed before any number exists:

  TARGET remains the MEASURED oracle ceiling of 0.481 from EXP-033, on this exact frozen
  concept@64. It is what this representation supports, not an aspiration.

  1. SATURATION is the primary question. Fit is judged by eye on four points, but the
     decision rule is fixed here: if 30,000 episodes does NOT beat 10,000 by at least 0.02,
     the curve has levelled and more episodes are refuted as a lever.
  2. If any cell reaches >= 0.35 (EXP-034's unmet "ceiling neighbourhood" bar), the learning
     signal is established as the binding constraint and the representation becomes the next
     target rather than a competing hypothesis.
  3. If the curve levels below 0.35, more episodes are exhausted and the remaining levers are
     credit assignment, curriculum design, and the representation itself.
  4. Report the collapse instruments. EXP-034's gain came with entropy FALLING, which is what
     distinguished competence from the randomness EXP-032 bought. If a larger budget raises
     success while entropy climbs, that is a different and less trustworthy mechanism.
  5. Seed variance is reported per cell. EXP-034 had sd 0.162 with one seed at 0.000 and the
     best at 0.467, so the mean alone hides most of what is happening.

Run (repo root):
    .venv/bin/python -u experiments/035_budget_scaling/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
"""

from __future__ import annotations

import argparse
import statistics as st
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import CubeConfig, run_cube_baseline

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

DEPTH = 3
CURRICULUM = (1, 2, 3)
BUDGETS = [10000, 30000]

ORACLE_CEILING = 0.481
KNOWN = {600: 0.0972, 3000: 0.2556}   # EXP-034, same schedule and seeds
SATURATION_DELTA = 0.02
TARGET_CEILING = 0.35


def cell_tag(budget: int) -> str:
    """Unique per cell: record_filename encodes tag/arm/depth/seed/sigma and NOT episodes."""
    return f"exp035_curriculum_e{budget}"


def sweep_configs(seeds, out_dir) -> list[CubeConfig]:
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=cell_tag(budget),
            depth=DEPTH, seed=seed, sigma=0.0, episodes=budget,
            curriculum=CURRICULUM, out_dir=out_dir,
        )
        for budget in BUDGETS
        for seed in seeds
    ]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = sweep_configs(args.seeds, args.out_dir)
    if len({(c.tag, c.seed) for c in configs}) != len(configs):
        raise SystemExit("tag collision: cells would overwrite each other")

    total_eps = sum(c.episodes for c in configs)
    print(f"EXP-035: {len(configs)} runs, budgets {BUDGETS}, {total_eps:,} episodes total")
    print(f"known curve: {KNOWN}   oracle ceiling {ORACLE_CEILING}")
    print(f"saturation rule (pre-registered): 30000 must beat 10000 by >= {SATURATION_DELTA}\n")

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. one record per run in {args.out_dir}\n")
    print(f"{'episodes':>10}{'success':>18}{'modal':>9}{'entropy':>9}{'best seed':>11}")
    means = {}
    for budget, val in KNOWN.items():
        print(f"{budget:>10}{val:>18.4f}{'':>9}{'':>9}{'(EXP-034)':>11}")
    for budget in BUDGETS:
        sub = [r for r in records if r["tag"] == cell_tag(budget)]
        if not sub:
            continue
        s = [r["success_rate"] for r in sub]
        means[budget] = st.mean(s)
        print(f"{budget:>10}{st.mean(s):>11.4f}+-{st.stdev(s):<5.3f}"
              f"{st.mean(r['greedy_modal_action_frac'] for r in sub):>9.3f}"
              f"{st.mean(r['mean_train_entropy'] for r in sub):>9.3f}{max(s):>11.3f}")

    if len(means) == 2:
        gain = means[30000] - means[10000]
        print(f"\n30000 minus 10000: {gain:+.4f}")
        if gain < SATURATION_DELTA:
            print("SATURATED. More episodes are refuted as a lever; the remaining levers are")
            print("credit assignment, curriculum design, and the representation itself.")
        else:
            print("STILL CLIMBING. More episodes remain a live lever.")
        if max(means.values()) >= TARGET_CEILING:
            print(f"Reached the {TARGET_CEILING} ceiling neighbourhood: the learning signal is")
            print("established as the binding constraint. Representation is the next target.")


if __name__ == "__main__":
    main()
