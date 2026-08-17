"""EXP-045 aggregator: apply the pre-registered rules to the records on disk.

WRITTEN BEFORE THE RUN WAS DISPATCHED. No record existed. Rules on disk before numbers.

Thresholds are the ones committed in
docs/superpowers/specs/2026-08-17-exp045-budget-vs-coverage-design.md.

> UNLIKE EXP-044, CLAIM 1 IS PAIRED AND CARRIES A P-VALUE. EXP-044 arm A is the baseline: same
> seeds, same encoders, same 10,000 episodes, same cap, differing only in `curriculum_weights`.
> That is what makes a paired test legitimate here and illegitimate there.

> AND AS ALWAYS: n=12 cannot show a failure count went to zero. Claim 4 is printed WITHOUT a
> p-value, deliberately.

Usage:
    .venv/bin/python experiments/045_budget_vs_coverage/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "044_depth7_frontier" / "outputs"

# --- pre-registered. Do not edit after data exists. ---
CLAIM1_DELTA = 0.05          # a third of arm B's +0.1350
ALPHA = 0.05
WORKING_BAR = 0.10           # the floor is measured at exactly 0.0000, so this binds
WORKING_MIN_SE = 1.0
WORKING_MIN_SEEDS = 8
DEPTH = 7

# EXP-044, measured. Arm A is the paired baseline; arm B is the effect being explained.
ARM_A_MEAN, ARM_B_MEAN = 0.0621, 0.1971
ARM_B_GAIN = ARM_B_MEAN - ARM_A_MEAN
# EXP-037 at depth 4, fixed budget, share at the evaluated depth rising. The prediction.
EXP037 = {"25%": (0.1591, 0.685), "50%": (0.1078, 0.735), "75%": (0.0921, 0.757)}


def permutation_p(diffs: list[float]) -> float:
    """Exact paired permutation over all 2**n sign flips, two-sided."""
    n = len(diffs)
    if not 1 <= n <= 20:
        raise ValueError(f"exact permutation needs 1 <= n <= 20, got {n}")
    observed = abs(sum(diffs))
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n)
               if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12)
    return hits / 2 ** n


def load(d: Path, tag: str) -> dict:
    out = {}
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text())
        if tag in r.get("tag", "") and r.get("depth") == DEPTH and r.get("arm") == "regionalized":
            out[r["seed"]] = r
    return out


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def se(xs):
    return sd(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    new = load(args.out_dir, "exp045_backloaded")
    base = load(BASELINE, "exp044_d7_e10000")
    if not new:
        raise SystemExit(f"no EXP-045 records in {args.out_dir}")
    if not base:
        raise SystemExit(f"no EXP-044 arm A records in {BASELINE} - they are the paired baseline")

    seeds = sorted(set(new) & set(base))
    vals = [new[s]["success_rate"] for s in seeds]
    diffs = [new[s]["success_rate"] - base[s]["success_rate"] for s in seeds]
    n = len(seeds)

    print(f"{'=' * 78}\nEXP-045 - back-loaded depth 7, {n} seeds paired against EXP-044 arm A\n{'=' * 78}")
    if n < len(new):
        print(f"  NOTE: {len(new) - n} EXP-045 seed(s) have no baseline and are excluded.")

    # --- Claim 1 (PRIMARY), paired ---
    delta = st.mean(diffs)
    p = permutation_p(diffs)
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    confirmed = delta >= CLAIM1_DELTA and p <= ALPHA
    print(f"\nCLAIM 1 (PRIMARY) - does deepest-shell coverage reproduce arm B's gain?")
    print(f"  H-coverage: {'CONFIRMED' if confirmed else 'REFUTED'}")
    print(f"  mean {st.mean(vals):.4f} (sd {sd(vals):.4f}) against arm A's {ARM_A_MEAN:.4f}")
    print(f"  paired delta {delta:+.4f}, W-L-T {wins}-{losses}-{n - wins - losses}, exact p {p:.4f}")
    print(f"    delta >= +{CLAIM1_DELTA}   {'PASS' if delta >= CLAIM1_DELTA else 'FAIL'}")
    print(f"    p <= {ALPHA}          {'PASS' if p <= ALPHA else 'FAIL'}")
    print(f"  arm B's gain over arm A was {ARM_B_GAIN:+.4f}; this arm recovers "
          f"{delta / ARM_B_GAIN * 100:.0f}% of it.")
    if not confirmed and 0 < delta < CLAIM1_DELTA:
        print("  NOTE: delta is positive but under the bar. That refutes H-coverage as THE")
        print("  explanation WITHOUT showing the effect is zero. Report it that way.")

    # --- Claim 2, the working bar ---
    margin = (st.mean(vals) - WORKING_BAR) / se(vals) if se(vals) else 0.0
    above = sum(1 for v in vals if v > WORKING_BAR)
    conds = [st.mean(vals) >= WORKING_BAR, margin >= WORKING_MIN_SE, above >= WORKING_MIN_SEEDS]
    print(f"\nCLAIM 2 - does it clear the working bar?   {'YES' if all(conds) else 'NO'}")
    print(f"  mean >= {WORKING_BAR}  {'PASS' if conds[0] else 'FAIL'};  "
          f"margin {margin:+.2f} SE  {'PASS' if conds[1] else 'FAIL'};  "
          f"{above}/{n} above  {'PASS' if conds[2] else 'FAIL'}")

    # --- Claim 3, MECHANISM, descriptive ---
    modal_new = [new[s].get("greedy_modal_action_frac") for s in seeds]
    modal_base = [base[s].get("greedy_modal_action_frac") for s in seeds]
    modal_new = [m for m in modal_new if m is not None]
    modal_base = [m for m in modal_base if m is not None]
    print("\nCLAIM 3 - MECHANISM, descriptive")
    if modal_new and modal_base:
        print(f"  modal action fraction: {st.mean(modal_new):.3f} here, "
              f"{st.mean(modal_base):.3f} in arm A")
    print("  EXP-037 at depth 4, share at eval depth rising (mean, modal):")
    for k, (m, mo) in EXP037.items():
        print(f"    {k:>4}: {m:.4f}  modal {mo:.3f}")
    print("  If back-loading hurt here, a RISE in modal says it hurt by driving collapse -")
    print("  which is the mechanism EXP-037 saw. Descriptive: no threshold was pre-registered.")

    # --- Claim 4, descriptive, NO p-value ---
    print(f"\nCLAIM 4 - failure counts, DESCRIPTIVE\n  {sum(1 for v in vals if v == 0.0)}/{n} "
          f"seeds at exactly 0.000, against arm A's 1/12")
    print("  no p-value: n=12 cannot show a count went to zero.")

    # --- Claim 5, the pre-committed reading ---
    print("\nCLAIM 5 - what this means")
    if confirmed:
        print("  H-coverage CONFIRMED. Episodes at the deepest shell are the operative variable,")
        print("  and EXP-044's coverage framing can be restored. Note this CONTRADICTS EXP-037's")
        print("  depth-4 decline, so the difference in regime (depth, pretrained encoder, cap)")
        print("  becomes the thing to explain.")
    else:
        print("  H-coverage REFUTED, so THE OPERATIVE VARIABLE IS TOTAL BUDGET. EXP-037's")
        print("  monotone decline survives a pretrained encoder, the depth-1 cap and three extra")
        print("  depths. Consequences, pre-committed:")
        print("   - the whole depth series is a BUDGET series; every 'depth N stopped working'")
        print("     result in this project is confounded with episodes;")
        print("   - retire the word 'coverage' from EXP-044's write-up;")
        print("   - the next experiment is depth 6 at raised TOTAL budget, not anything weighted.")

    print("\nper-seed (EXP-045, arm A, delta):")
    for s in seeds:
        a, b = new[s]["success_rate"], base[s]["success_rate"]
        print(f"  s{s:<3} {a:.4f}  {b:.4f}  {a - b:+.4f}")


if __name__ == "__main__":
    main()
