"""EXP-036 aggregator: read the records, apply the pre-registered rules, print the verdicts.

Separate from run.py deliberately. run.py prints a summary at the end of a 12-hour job, and
if that job is interrupted, or the summary needs re-reading, or a threshold needs checking
against a different subset, re-running the sweep is not an option. This reads the records off
disk and can be run any number of times.

The thresholds here are NOT re-derived. They are the ones committed in
docs/superpowers/specs/2026-08-03-exp036-generalisation-gap-design.md and in run.py's
docstring, before any number existed. If a number in this file disagrees with the spec, the
spec wins and this file is the bug.

Usage:
    .venv/bin/python experiments/036_generalisation_gap/aggregate.py
    .venv/bin/python experiments/036_generalisation_gap/aggregate.py --out-dir some/other/dir
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEPTHS = [3, 4, 5, 6]

# Pre-registered. Do not tune these to make a result land.
GAP_REFUTE_BELOW = 0.05
GAP_CONFIRM_AT = 0.15
ALPHA = 0.05
BREAK_MULTIPLE = 2.0
# The floor COLLAPSES with depth (a random walk almost never solves a depth-6 cube), so
# "2x the measured floor" is 0.029 at depth 3 but only 0.009 at depth 6: the bar vanishes
# exactly where it has to bite. Caught on synthetic records 2026-08-03, BEFORE any real
# number existed, when a synthetic depth-6 policy at 0.017 (plainly failing) was reported
# "working". A policy must therefore ALSO clear an absolute bar.
#
# 0.10 is one quarter of the depth-3 result at this same budget (EXP-035's 0.397). Below a
# quarter of what the approach achieves where it works, it is not working. It is also well
# clear of resolution: the held-out sets are 30/133/200/200 states, so 0.10 is 3 to 20 solves.
BREAK_ABSOLUTE = 0.10
EXP035_DEPTH3_AT_10K = 0.397
REPLICATION_TOLERANCE = 0.02

PREDICTION = "the curriculum breaks at depth 5"

# EXP-033's raw-facelet linear probe, the basis for that prediction. Chance is about 0.19.
PROBE = {3: 0.956, 4: 0.766, 5: 0.598}


def permutation_p(diffs: list[float]) -> float:
    """Exact paired permutation over all 2**n sign flips, two-sided.

    n = 12 is 4096 flips. No scipy in the venv and a normal approximation at n = 12 is not
    trustworthy, so this is exhaustive rather than sampled or approximated.
    """
    n = len(diffs)
    if n == 0 or n > 20:
        raise ValueError(f"exact permutation needs 1 <= n <= 20, got {n}")
    observed = abs(sum(diffs))
    hits = sum(
        1
        for signs in itertools.product((1, -1), repeat=n)
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12
    )
    return hits / (2 ** n)


def load(out_dir: Path) -> list[dict]:
    records = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not records:
        raise SystemExit(f"no records in {out_dir}. Has the run finished, or been fetched?")
    return records


def sd(xs: list[float]) -> float:
    return st.stdev(xs) if len(xs) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    records = load(args.out_dir)
    trained = [r for r in records if r["arm"] != "random"]
    floors = [r for r in records if r["arm"] == "random"]

    print(f"EXP-036: {len(records)} records ({len(trained)} trained, {len(floors)} floors)")
    expected = 96
    if len(records) != expected:
        print(f"  INCOMPLETE: expected {expected}. Numbers below are over what is present,")
        print("  which is not the pre-registered sample. Treat every verdict as provisional.")

    def cells(depth, arm):
        return [r for r in records if r["depth"] == depth and
                (r["arm"] == "random") == (arm == "random")]

    print()
    print(f"{'depth':>6}{'n':>4}{'heldout':>17}{'train':>10}{'gap':>9}{'p':>9}"
          f"{'floor':>8}{'bar':>9}{'modal':>8}{'entropy':>9}{'verdict':>10}")

    summary = {}
    for depth in DEPTHS:
        sub, flo = cells(depth, "trained"), cells(depth, "random")
        if not sub:
            continue
        held = [r["success_rate"] for r in sub]
        train = [r["train_success_rate"] for r in sub]
        gaps = [r["generalisation_gap"] for r in sub]
        floor = st.mean(r["success_rate"] for r in flo) if flo else float("nan")
        p = permutation_p(gaps) if len(gaps) > 1 else float("nan")
        # Broken unless it clears BOTH bars.
        broken = not (st.mean(held) >= BREAK_MULTIPLE * floor
                      and st.mean(held) >= BREAK_ABSOLUTE)
        summary[depth] = {"held": st.mean(held), "gap": st.mean(gaps), "p": p,
                          "floor": floor, "broken": broken, "n": len(sub)}
        print(f"{depth:>6}{len(sub):>4}{st.mean(held):>11.4f}+-{sd(held):<5.3f}"
              f"{st.mean(train):>10.4f}{st.mean(gaps):>+9.4f}{p:>9.4f}{floor:>8.4f}"
              f"{max(BREAK_MULTIPLE * floor, BREAK_ABSOLUTE):>9.4f}"
              f"{st.mean(r['greedy_modal_action_frac'] for r in sub):>8.3f}"
              f"{st.mean(r['mean_train_entropy'] for r in sub):>9.3f}"
              f"{'BROKEN' if broken else 'working':>10}")

    # The empirical null. A random policy cannot overfit, so whatever gap it shows is what
    # zero looks like at this sample size. Any trained gap must clear THIS, not clear zero.
    print("\nnull gap (random arm, cannot overfit):")
    for depth in DEPTHS:
        flo = cells(depth, "random")
        if len(flo) > 1:
            fg = [r["generalisation_gap"] for r in flo]
            print(f"  depth {depth}: {st.mean(fg):+.4f}  sd {sd(fg):.4f}  "
                  f"range [{min(fg):+.3f}, {max(fg):+.3f}]  p {permutation_p(fg):.4f}")

    # --- replication gate. Nothing above is trustworthy until this passes. ---
    print("\n" + "=" * 78)
    d3 = cells(3, "trained")
    if d3:
        got = st.mean(r["success_rate"] for r in d3)
        delta = got - EXP035_DEPTH3_AT_10K
        print(f"REPLICATION: depth 3 = {got:.4f} vs EXP-035's {EXP035_DEPTH3_AT_10K} "
              f"({delta:+.4f})")
        if abs(delta) > REPLICATION_TOLERANCE:
            print("  MISMATCH. Either the EXP-036 code changes moved something, or this did")
            print("  not run on the same machine as EXP-035 (seeded runs are NOT reproducible")
            print("  across platforms). RESOLVE THIS BEFORE READING ANY ROW ABOVE.")
        else:
            print("  PASS. The train-side eval and head checkpointing were neutral.")

    # --- claim 1 ---
    print("\nCLAIM 1, the gap at depth 3:")
    if d3:
        gaps = [r["generalisation_gap"] for r in d3]
        mg, p = st.mean(gaps), permutation_p(gaps)
        print(f"  mean {mg:+.4f}, sd {sd(gaps):.4f}, exact p {p:.4f}, n = {len(gaps)}")
        print(f"  per seed: {[round(g, 3) for g in gaps]}")
        if mg < GAP_REFUTE_BELOW and p > ALPHA:
            print("  -> COVERAGE REFUTED as a lever. The Stage-1a train-fraction sweep from")
            print("     road-to-a-solved-cube.md is CANCELLED, not run.")
        elif mg >= GAP_CONFIRM_AT and p <= ALPHA:
            print("  -> OVERFITTING ESTABLISHED. The train-fraction sweep is justified.")
        else:
            print("  -> INCONCLUSIVE. Report it, act on neither. Do not move the threshold.")

    # --- claims 2 and 3 ---
    print(f"\nCLAIM 2, the break point. Pre-registered prediction: {PREDICTION}")
    broken_depths = [d for d, s in summary.items() if s["broken"]]
    first = min(broken_depths) if broken_depths else None
    for depth in DEPTHS:
        if depth in summary:
            s = summary[depth]
            probe = f", EXP-033 probe {PROBE[depth]}" if depth in PROBE else ""
            bar = max(BREAK_MULTIPLE * s['floor'], BREAK_ABSOLUTE)
            print(f"  depth {depth}: {s['held']:.4f} vs bar {bar:.4f}"
                  f" (2x floor {2 * s['floor']:.4f}, absolute {BREAK_ABSOLUTE})"
                  f" -> {'BROKEN' if s['broken'] else 'working'}{probe}")
    if first == 5:
        print("  -> PREDICTION CONFIRMED. It breaks at depth 5, as the probe trend implied.")
    elif first is None:
        print("  -> PREDICTION REFUTED (claim 3). Nothing broke, not even depth 6. Wall 1 is")
        print("     further out than EXP-033's probe trend implied and the linear head has")
        print("     more room than that experiment suggested. Log this as a refutation.")
    else:
        print(f"  -> PREDICTION REFUTED. It breaks at depth {first}, not 5.")

    print("\nREMEMBER (claim 5): this is the break point AT 10,000 EPISODES. EXP-035 showed")
    print("depth 3 climbing 0.397 -> 0.500 between 10k and 30k without saturating, so a depth")
    print("that failed here may only be under-trained. Do NOT report the break point as a")
    print("property of the architecture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
