# experiments/030_memory_engagement/run.py
"""EXP-030 driver: does episodic memory content improve cube solving?

Phase 1 runs the concept arm alone and reports its revisit rate. That is a GATE: with a
2d+3 step budget, if the policy rarely revisits a state then cycle-avoidance memory has
nothing to do, and a null would say nothing about memory. Phase 2 runs the two memory
arms only after that number has been read.

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/030_memory_engagement/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import CubeConfig, run_cube_baseline

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
DEPTHS = [1, 2, 3]
MEMORY_ARMS = ["memory", "memory_shuffled"]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def _fan_out(configs, workers: int) -> list[dict]:
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if i % 10 == 0 or i == len(configs):
                print(f"  {i}/{len(configs)}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-gate", action="store_true",
                    help="run the memory arms without pausing on the revisit gate")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(arm="regionalized", episodes=args.episodes, sigma=0.0,
                  out_dir=args.out_dir, tag="exp030")

    print(f"phase 1: concept arm, {len(args.depths) * len(args.seeds)} runs")
    concept_records = _fan_out(
        [CubeConfig(readout="concept", depth=d, seed=s, **common)
         for d in args.depths for s in args.seeds],
        args.workers,
    )

    print("\nREVISIT GATE (the number that decides whether this experiment can work):")
    for d in args.depths:
        rates = [r["revisit_rate"] for r in concept_records if r["depth"] == d]
        mean = sum(rates) / len(rates) if rates else 0.0
        print(f"  depth {d}: mean revisit rate {mean:.3f}")
    print("  If these are near zero there are no cycles to avoid, and a null in the")
    print("  memory arms would be a statement about the task, not about memory.\n")

    if not args.skip_gate:
        reply = input("continue to the memory arms? [y/N] ").strip().lower()
        if reply != "y":
            print("stopped at the gate. Records for the concept arm are already written.")
            return

    rest = [CubeConfig(readout=r, depth=d, seed=s, **common)
            for r in MEMORY_ARMS for d in args.depths for s in args.seeds]
    print(f"phase 2: memory arms, {len(rest)} runs")
    _fan_out(rest, args.workers)
    print(f"done. one record per run in {args.out_dir}")


if __name__ == "__main__":
    main()
