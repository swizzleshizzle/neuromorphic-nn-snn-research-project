# experiments/031_policy_collapse/aggregate.py
"""Aggregate EXP-031 records: is the trained policy collapsed to a constant action?

Two instruments, reported together because they fail independently. The `F'` x9
observation that motivated this was made under the GREEDY policy, while `entropy_beta=0.0`
is a TRAINING setting:

- ``greedy_modal_action_frac``: fraction of a rollout spent on its single most-common
  action, averaged per episode. Collapse scores 1.0. A uniform policy over the 6 cube
  moves scores 0.354 over a 9-step budget and 0.429 over 5 (measured over 20,000 simulated
  rollouts).
- ``mean_train_entropy``: per-episode policy entropy during training, against a log(6) =
  1.792 ceiling.

**The modal fraction is only meaningful where episodes actually run the budget.** An
episode solved in one move has a modal fraction of 1.0 by construction, so a depth with a
high success rate reports collapse it has not measured. The confound table is printed
first, and depth 1 is excluded from the verdict for exactly this reason.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

ARMS = ["concept", "memory", "memory_shuffled", "memory_amnesic"]
DEPTHS = [1, 2, 3]

# Measured, not assumed: 20,000 simulated uniform rollouts on 6 actions, 2026-07-31.
UNIFORM_FLOOR = {1: 0.429, 2: 0.380, 3: 0.354}
MAX_ENTROPY = math.log(6)

# A depth whose episodes mostly end early cannot support a modal-fraction claim.
FULL_BUDGET_MIN = 0.60
COLLAPSE_BAR = 0.95


def load(out_dir: Path) -> list[dict]:
    records = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not records:
        raise SystemExit(f"no records in {out_dir}")
    return records


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def cell(values: list[float]) -> str:
    if not values:
        return f"{'n/a':>16}"
    if len(values) == 1:
        return f"{values[0]:>16.3f}"
    return f"{st.mean(values):>9.3f}+-{st.stdev(values):<6.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()
    recs = load(args.out_dir)

    print(f"EXP-031: {len(recs)} records from {args.out_dir}\n")

    print("CONFOUND CHECK (read this before the tables below)")
    print("An episode solved in one move scores modal_frac 1.0 by construction, so the")
    print("metric is only interpretable where episodes run the full budget.")
    print(f"{'depth':>6}{'mean success':>15}{'full-budget frac':>19}{'corr(modal,succ)':>19}   verdict")
    usable = []
    for d in DEPTHS:
        sub = [r for r in recs if r["depth"] == d]
        if not sub:
            continue
        succ = [r["success_rate"] for r in sub]
        modal = [r["greedy_modal_action_frac"] for r in sub]
        full = 1.0 - st.mean(succ)
        ok = full >= FULL_BUDGET_MIN
        usable.append((d, ok))
        print(f"{d:>6}{st.mean(succ):>15.3f}{full:>19.3f}{pearson(modal, succ):>19.3f}"
              f"   {'usable' if ok else 'DISCARD (episodes end early)'}")

    print(f"\ngreedy_modal_action_frac   (collapse = 1.000; uniform floor {UNIFORM_FLOOR})")
    header = f"{'arm':<18}" + "".join(f"{'depth ' + str(d):>16}" for d in DEPTHS)
    print(header)
    for arm in ARMS:
        row = f"  {arm:<16}"
        for d in DEPTHS:
            row += cell([r["greedy_modal_action_frac"] for r in recs
                         if r["readout"] == arm and r["depth"] == d])
        print(row)

    print(f"\nmean_train_entropy   (collapse -> 0; uniform ceiling log6 = {MAX_ENTROPY:.3f})")
    print(header)
    for arm in ARMS:
        row = f"  {arm:<16}"
        for d in DEPTHS:
            row += cell([r["mean_train_entropy"] for r in recs
                         if r["readout"] == arm and r["depth"] == d])
        print(row)

    print(f"\nseeds at or above {COLLAPSE_BAR} modal fraction (effectively constant-action)")
    print(header)
    for arm in ARMS:
        row = f"  {arm:<16}"
        for d in DEPTHS:
            v = [r["greedy_modal_action_frac"] for r in recs
                 if r["readout"] == arm and r["depth"] == d]
            row += f"{str(sum(1 for x in v if x >= COLLAPSE_BAR)) + '/' + str(len(v)):>16}"
        print(row)

    for d, ok in usable:
        if not ok:
            continue
        v = sorted(r["greedy_modal_action_frac"] for r in recs
                   if r["readout"] == "concept" and r["depth"] == d)
        print(f"\ndepth {d}, concept arm, per seed (sorted):")
        print("  " + " ".join(f"{x:.3f}" for x in v))

    deep = [d for d, ok in usable if ok]
    if deep:
        d = max(deep)
        v = [r["greedy_modal_action_frac"] for r in recs
             if r["readout"] == "concept" and r["depth"] == d]
        n_collapsed = sum(1 for x in v if x >= COLLAPSE_BAR)
        print(f"\nVERDICT at depth {d} (the deepest usable depth): {n_collapsed}/{len(v)} concept "
              f"seeds are effectively constant-action, against a {UNIFORM_FLOOR[d]} uniform floor.")


if __name__ == "__main__":
    main()
