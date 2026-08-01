# experiments/034_learning_signal/run.py
"""EXP-034 driver: is the learning signal the bottleneck, not the representation?

EXP-033 measured an oracle ceiling. Fitting the SAME frozen concept@64 and the SAME
`Linear(64 -> 6)` head with supervised labels solves **48.1%** of depth-3 cubes; training
those identical 390 weights with REINFORCE solves **2.2%**. Same representation, same head
shape, same environment. Only the method of choosing the weights differs.

So the information is present and linearly extractable by the head the policy already has,
and reinforcement learning does not find it. The likely reason is signal starvation: at
depth 3 the reward is almost never obtained, so REINFORCE has almost nothing to reinforce
across 600 episodes.

This changes ONLY how the head is trained. The brain stays frozen at random init, the
concept stays 64-wide, the head stays linear, evaluation stays at depth 3.

    schedule   direct        train all episodes at depth 3 (shipped behaviour)
               curriculum    split the SAME budget across depths 1 -> 2 -> 3
    episodes   600           the budget every previous cube experiment used
               3000          5x, to separate "needs a better signal" from "needs more of it"

Crossing the two matters. A curriculum win at a fixed budget is about the signal; a win that
only appears at 3000 episodes is about volume; a win at both is about neither alone. The
budget is CONSERVED inside a curriculum run (`curriculum_schedule` splits rather than
multiplies), so the curriculum arm never buys extra compute.

PRE-REGISTERED CONTRACT, committed before any number exists:

  TARGET is the measured oracle ceiling, 0.481 depth-3 success, NOT an assumed value. The
  claim under test is that the gap between 0.022 and 0.481 is closable by changing the
  learning signal alone.

  1. PRIMARY: does any arm move depth-3 success materially above the 0.022 baseline?
     "Materially" is fixed here as reaching 0.10, roughly a 4.5x improvement and still only
     a fifth of the ceiling.
  2. If an arm reaches the ceiling's neighbourhood (>= 0.35), the learning signal was the
     binding constraint and the representation question from EXP-033 Finding 1 is
     downstream, not upstream.
  3. If NO arm clears 0.10 even at 3000 episodes, signal starvation is REFUTED as the sole
     explanation and the next suspect is credit assignment itself (REINFORCE with a scalar
     baseline over a 9-step horizon), not the amount of experience.
  4. Report `greedy_modal_action_frac` and `mean_train_entropy` alongside, so a success
     change can be attributed to the policy actually reading its input rather than to a
     different flavour of degeneracy. EXP-032 showed those can move in opposite directions.

Success rate IS the headline here, unlike EXP-032. The oracle ceiling makes it a fair bar
because it is a measured statement about what this exact representation can support.

Run (repo root):
    .venv/bin/python -u experiments/034_learning_signal/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
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
SCHEDULES = {"direct": (), "curriculum": CURRICULUM}
BUDGETS = [600, 3000]

ORACLE_CEILING = 0.481   # EXP-033, measured on this exact frozen concept@64
RL_BASELINE = 0.022      # EXP-030/031/032, depth-3 concept arm
TARGET_MATERIAL = 0.10   # pre-registered "materially above baseline"
TARGET_CEILING = 0.35    # pre-registered "reached the ceiling's neighbourhood"


def cell_tag(schedule: str, budget: int) -> str:
    """Unique per cell.

    REQUIRED: `record_filename` encodes tag/arm/depth/seed/sigma and NOT the schedule or the
    episode budget, so without this every cell would overwrite the same file. Same trap as
    EXP-032, where a beta-blind tag would have turned 192 runs into 24 files.
    """
    return f"exp034_{schedule}_e{budget}"


def sweep_configs(seeds, out_dir) -> list[CubeConfig]:
    return [
        CubeConfig(
            arm="regionalized",
            readout="concept",
            tag=cell_tag(name, budget),
            depth=DEPTH,
            seed=seed,
            sigma=0.0,
            episodes=budget,
            curriculum=stages,
            out_dir=out_dir,
        )
        for name, stages in SCHEDULES.items()
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

    names = {(c.tag, c.seed) for c in configs}
    if len(names) != len(configs):
        raise SystemExit("tag collision: cells would overwrite each other")

    print(f"EXP-034: {len(configs)} runs "
          f"({len(SCHEDULES)} schedules x {len(BUDGETS)} budgets x {len(args.seeds)} seeds), "
          f"evaluated at depth {DEPTH}")
    print(f"baseline {RL_BASELINE}  ->  target {TARGET_MATERIAL}  ->  "
          f"oracle ceiling {ORACLE_CEILING} (measured, EXP-033)\n")

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if i % 5 == 0 or i == len(configs):
                print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. one record per run in {args.out_dir}\n")
    print(f"{'schedule':>12}{'episodes':>10}{'success':>18}{'modal_frac':>13}"
          f"{'entropy':>10}   verdict")
    for name in SCHEDULES:
        for budget in BUDGETS:
            sub = [r for r in records
                   if r["tag"] == cell_tag(name, budget)]
            if not sub:
                continue
            s = st.mean(r["success_rate"] for r in sub)
            sd = st.stdev(r["success_rate"] for r in sub) if len(sub) > 1 else 0.0
            m = st.mean(r["greedy_modal_action_frac"] for r in sub)
            e = st.mean(r["mean_train_entropy"] for r in sub)
            if s >= TARGET_CEILING:
                verdict = "REACHED CEILING"
            elif s >= TARGET_MATERIAL:
                verdict = "material gain"
            else:
                verdict = "no gain"
            print(f"{name:>12}{budget:>10}{s:>11.3f}+-{sd:<5.3f}{m:>13.3f}{e:>10.3f}"
                  f"   {verdict}")

    best = max(records, key=lambda r: r["success_rate"], default=None)
    if best is not None:
        pooled = {name: st.mean(r["success_rate"] for r in records
                                if r["tag"].startswith(f"exp034_{name}"))
                  for name in SCHEDULES}
        print(f"\nbest single cell mean is reported above; per-schedule pooled: {pooled}")
        if max(pooled.values()) < TARGET_MATERIAL:
            print("\nNo arm cleared the pre-registered 0.10 bar. Signal starvation is REFUTED")
            print("as the sole explanation; the next suspect is credit assignment, not volume.")
            print("Do not relax the bar.")


if __name__ == "__main__":
    main()
