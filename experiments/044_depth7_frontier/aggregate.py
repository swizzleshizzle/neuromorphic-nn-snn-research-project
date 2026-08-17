"""EXP-044 aggregator: apply the pre-registered rules to the records on disk.

WRITTEN BEFORE ANY VALUE WAS READ - in fact written while arm A was still running, so no record
existed to peek at. Rules on disk before numbers is the standing habit.

Thresholds are the ones committed in
docs/superpowers/specs/2026-08-14-exp044-depth7-frontier-design.md at b992854.

> THERE IS NO PAIRED BASELINE. Depth 7 has never been attempted, so Claim 1 is ABSOLUTE and this
> file computes NO p-value for it. The uncertainty is carried by the margin and seed-count
> conditions. A paired p appearing here would mean a baseline was invented after the fact.

> AND AS IN EXP-042/043: n=12 cannot show a failure count went to zero. Claim 3 is printed
> WITHOUT a p-value, deliberately.

Usage:
    .venv/bin/python experiments/044_depth7_frontier/aggregate.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- pre-registered. Do not edit after data exists. ---
ABSOLUTE_BAR = 0.10          # EXP-036's bar
FLOOR_MULTIPLE = 2.0         # EXP-036's other half: >= 2x the MEASURED floor
WORKING_MIN_SE = 1.0         # margin, because EXP-040's 0.1037 cleared the bare rule on noise
WORKING_MIN_SEEDS = 8        # of 12, each individually above the bar
DEPTH = 7
EPISODES_A = 10_000
EPISODES_B = 44_000

# Earlier depths under the same recipe, for the descriptive series only.
PRIOR = {4: {"mean": 0.5351, "sd": 0.1012}, 5: {"mean": 0.3412, "sd": 0.1277},
         6: {"mean": 0.1800, "sd": 0.0985}}
COVERAGE = {5: 0.973, 6: 0.190, 7: 0.044}   # all at 10,000 episodes
COVERAGE_B = 0.191                         # depth 7 at 44,000, which is the point of arm B


def load(d: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def cells(recs, *, arm: str, tag_has: str) -> dict:
    out = {}
    for r in recs:
        if r.get("depth") != DEPTH or r.get("arm") != arm:
            continue
        if tag_has not in r.get("tag", ""):
            continue
        out[r["seed"]] = r
    return out


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def se(xs):
    return sd(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0


def verdict(ok: bool) -> str:
    return "CONFIRMED" if ok else "REFUTED"


def report(recs: list[dict], episodes: int) -> dict:
    arm_name = "A" if episodes == EPISODES_A else ("B" if episodes == EPISODES_B else "?")
    train = cells(recs, arm="regionalized", tag_has=f"exp044_d{DEPTH}_e{episodes}")
    floor = cells(recs, arm="random", tag_has=f"exp044_floor_d{DEPTH}")

    print(f"\n{'=' * 78}\nEXP-044 arm {arm_name} - depth {DEPTH}, {episodes:,} episodes\n{'=' * 78}")
    if not train:
        print(f"  no records for arm {arm_name}. Nothing to aggregate.")
        return {}

    vals = [train[s]["success_rate"] for s in sorted(train)]
    n = len(vals)
    mean = st.mean(vals)

    # --- the floor, MEASURED. The bar cannot be computed without it. ---
    if floor:
        floor_vals = [floor[s]["success_rate"] for s in sorted(floor)]
        floor_mean = st.mean(floor_vals)
        print(f"\nfloor (arm=random, {len(floor_vals)} seeds): {floor_mean:.4f}  "
              f"max {max(floor_vals):.4f}")
    else:
        floor_mean = None
        print("\nfloor: NOT MEASURED. Re-run without --no-floor; the bar is undefined without it,")
        print("       and assuming it is the mistake the depth-1 trap was made of.")

    bar = ABSOLUTE_BAR if floor_mean is None else max(ABSOLUTE_BAR, FLOOR_MULTIPLE * floor_mean)
    which = "0.10 binds" if bar == ABSOLUTE_BAR else f"2x floor binds ({FLOOR_MULTIPLE} x {floor_mean:.4f})"
    print(f"BAR = {bar:.4f}   ({which})")

    # --- Claim 1 (PRIMARY), absolute, NO p-value ---
    margin_se = (mean - bar) / se(vals) if se(vals) else 0.0
    above = sum(1 for v in vals if v > bar)
    conds = [mean >= bar, margin_se >= WORKING_MIN_SE, above >= WORKING_MIN_SEEDS]
    print(f"\nCLAIM 1 (PRIMARY) - is depth {DEPTH} working?   {verdict(all(conds))}")
    print(f"  mean {mean:.4f} over {n} seeds (sd {sd(vals):.4f}, se {se(vals):.4f})")
    print(f"    mean >= {bar:.4f}          {'PASS' if conds[0] else 'FAIL'}")
    print(f"    margin {margin_se:+.2f} SE >= {WORKING_MIN_SE:.1f}   {'PASS' if conds[1] else 'FAIL'}")
    print(f"    {above}/{n} seeds above bar >= {WORKING_MIN_SEEDS}  {'PASS' if conds[2] else 'FAIL'}")
    print("  no p-value: there is no prior depth-7 arm to pair against, by design.")

    # --- Claim 2, the pre-registered escalation ---
    print(f"\nCLAIM 2 - escalation")
    cov = COVERAGE[DEPTH] if episodes == EPISODES_A else COVERAGE_B
    if all(conds) and arm_name == "B":
        # The reading fixed in the spec BEFORE arm B existed. Printed here rather than left to a
        # reader, because "depth 7 works" and "depth 7 works only at 4.4x the budget" are
        # different findings and the second one is the true one.
        print("  Arm B CONFIRMED and arm A did not. By the pre-registered reading, arm A's")
        print("  failure was STARVATION, NOT DIFFICULTY: depth 7 works once each training state")
        print(f"  is seen as often as depth 6's were ({cov:.3f} vs {COVERAGE[6]:.3f} episodes each).")
        print("  THE BREAK POINT IS NOT FOUND. What is found is a budget scaling law, and every")
        print("  depth's number becomes a function of coverage rather than of depth alone.")
        print("  Do NOT write that the cube is solved: depths 1-7 are 0.9% of the state space")
        print("  and a random scramble sits at depth 11.")
    elif all(conds):
        print("  Claim 1 CONFIRMED, so ARM B DOES NOT RUN. The frontier is past depth 7 and")
        print("  still unmeasured; the next experiment is depth 8.")
        print(f"  Coverage was {cov:.3f} episodes/train state against depth 6's "
              f"{COVERAGE[6]:.3f}, so this is a")
        print("  STRONGER generalisation result than depth 6's, not a weaker one.")
        print("  Do NOT write that the cube is solved: depths 1-7 are 0.9% of the state space")
        print("  and a random scramble sits at depth 11.")
    elif arm_name == "A":
        print("  Claim 1 REFUTED, so ARM B IS TRIGGERED. Dispatch:")
        print(f"    run.py --episodes {EPISODES_B} --seeds 0 1 2 3 4 5 6 7 8 9 10 11 "
              "--workers 12 --skip-existing --no-floor")
        print(f"  It matches depth 6's {COVERAGE[6]:.3f} episodes/train state and costs about 52 h.")
        print("  Reading, fixed in advance: B works -> the failure was STARVATION and the break")
        print("  point is not found. B also fails -> the break point IS depth 7 for this recipe.")
    else:
        print("  Arm B refuted as well: THE BREAK POINT IS DEPTH 7 for this recipe. That is the")
        print("  frontier result, not a failed experiment (Claim 5).")

    # --- Claim 3, descriptive, NO p-value ---
    zeros = sum(1 for v in vals if v == 0.0)
    print(f"\nCLAIM 3 - failure counts, DESCRIPTIVE\n  {zeros}/{n} seeds at exactly 0.000")
    print("  no p-value: n=12 cannot show a count went to zero (Fisher's ~0.48 on 2/12 vs 0/12).")

    # --- Claim 4, descriptive ---
    print("\nCLAIM 4 - variance across depths, DESCRIPTIVE")
    for d in sorted(PRIOR):
        print(f"  depth {d}: mean {PRIOR[d]['mean']:.4f}  sd {PRIOR[d]['sd']:.4f}")
    print(f"  depth {DEPTH}: mean {mean:.4f}  sd {sd(vals):.4f}")

    print("\nper-seed:")
    for s in sorted(train):
        v = train[s]["success_rate"]
        print(f"  s{s:<3} {v:.4f}{'  *' if v > bar else ''}")

    return {"mean": mean, "sd": sd(vals), "bar": bar, "above": above, "n": n,
            "margin_se": margin_se, "working": all(conds), "zeros": zeros,
            "floor": floor_mean}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    recs = load(args.out_dir)
    if not recs:
        raise SystemExit(f"no records in {args.out_dir}")
    print(f"{len(recs)} records in {args.out_dir}")

    for episodes in (EPISODES_A, EPISODES_B):
        report(recs, episodes)


if __name__ == "__main__":
    main()
