# experiments/029_cube_baseline/run.py
"""EXP-029 driver: the v1 fail-first cube baseline and its unregionalized control.

Phase 1 sweeps sigma at depth 1 on BOTH trained arms and picks each arm's winner. Phase 2
runs depths 2 to 6 at each arm's winning sigma. The random arm is evaluation only and
measures the chance floor at every depth.

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/029_cube_baseline/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import CubeConfig, run_cube_baseline

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
SIGMAS = [0.0, 0.2, 0.4]
TRAINED_ARMS = ["regionalized", "monolithic"]
DEPTHS = [1, 2, 3, 4, 5, 6]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(episodes=args.episodes, out_dir=args.out_dir)

    # Phase 1: sigma sweep at depth 1, both trained arms. Supplies the depth-1 cell.
    sweep = [
        CubeConfig(arm=arm, depth=1, seed=s, sigma=sig, **common)
        for arm in TRAINED_ARMS
        for sig in SIGMAS
        for s in args.seeds
    ]
    print(f"phase 1: sigma sweep, {len(sweep)} runs")
    sweep_records = _fan_out(sweep, args.workers)

    # Each arm keeps its own winning sigma. Tuning on one arm and applying it to the
    # other would hand the control an untuned hyperparameter and bias the comparison.
    by_arm_sigma = defaultdict(list)
    for r in sweep_records:
        by_arm_sigma[(r["arm"], r["sigma"])].append(r["success_rate"])
    best = {}
    for arm in TRAINED_ARMS:
        means = {
            sig: sum(by_arm_sigma[(arm, sig)]) / len(by_arm_sigma[(arm, sig)])
            for sig in SIGMAS
            if by_arm_sigma.get((arm, sig))
        }
        # Deterministic: fixed SIGMAS order, and on an exact tie prefer the smaller
        # sigma rather than whichever worker finished first.
        best[arm] = max(SIGMAS, key=lambda s: (means.get(s, float("-inf")), -s))
        print(f"  {arm}: sigma means {means} -> winner {best[arm]}")

    manifest = args.out_dir / "029_winners.json"
    manifest.write_text(json.dumps(best), encoding="utf-8")
    print(f"  winners written to {manifest}")

    # Phase 2: depths 2-6 at each arm's winning sigma, plus the random floor everywhere.
    rest = [
        CubeConfig(arm=arm, depth=d, seed=s, sigma=best[arm], **common)
        for arm in TRAINED_ARMS
        for d in DEPTHS[1:]
        for s in args.seeds
    ] + [
        CubeConfig(arm="random", depth=d, seed=s, sigma=0.0, **common)
        for d in DEPTHS
        for s in args.seeds
    ]
    print(f"phase 2: {len(rest)} runs")
    _fan_out(rest, args.workers)
    print(f"done. one record per run in {args.out_dir}")


def _fan_out(configs, workers: int) -> list[dict]:
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if i % 10 == 0 or i == len(configs):
                print(f"  {i}/{len(configs)}")
    return records


if __name__ == "__main__":
    main()
