"""EXP-043 aggregator: apply the pre-registered rules to the records on disk.

WRITTEN BEFORE ANY VALUE WAS READ. The records were fetched and counted; no number in them was
inspected before this file existed. Rules on disk before numbers is the standing habit.

Thresholds are the ones committed in
docs/superpowers/specs/2026-08-13-exp043-cap-at-depth-5-6-design.md.

The baseline is EXP-040's depth-5 and depth-6 cells: same pretrained encoders, same seeds, same
machine, same budget, same curriculum, differing by the depth-1 training cap and nothing else.
EXP-040 is NOT re-run.

> THE BASELINE HAS NO `stage_trace`. EXP-040 predates that telemetry, so - unlike EXP-042 - the
> primary claim is on SUCCESS, not entropy. This file does not attempt to pair the mechanism.

> AND AS IN EXP-042: n=12 cannot show a failure count went to zero. Fisher's exact on 2/12
> against 0/12 gives about 0.48. Claim 3 is printed WITHOUT a p-value, deliberately.

Usage:
    .venv/bin/python experiments/043_cap_at_depth_5_6/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- pre-registered. Do not edit after data exists. ---
CLAIM1_DELTA = 0.05
ALPHA = 0.05
WORKING_BAR = 0.10           # EXP-036's absolute bar; binds above 2x floor at both depths
WORKING_MIN_SE = 1.0         # margin required, because EXP-040 cleared the bare rule by 0.11 SE
WORKING_MIN_SEEDS = 8        # of 12, individually above WORKING_BAR
FLOOR = {5: 0.0000, 6: 0.0008}
DEPTHS = [5, 6]

# EXP-040, measured with the depth-1 trap still in place.
EXP040_MEAN = {5: 0.2304, 6: 0.1037}
EXP040_ZEROS = {5: 2, 6: 3}
EXP040_SD = {5: 0.1590, 6: 0.1206}
# EXP-042 at depth 4, for the variance comparison.
EXP042_D4 = {"baseline_sd": 0.2242, "capped_sd": 0.1012,
             "baseline_mean": 0.3471, "capped_mean": 0.5351}


def permutation_p(diffs: list[float]) -> float:
    """Exact paired permutation over all 2**n sign flips, two-sided."""
    n = len(diffs)
    if n == 0 or n > 20:
        raise ValueError(f"exact permutation needs 1 <= n <= 20, got {n}")
    observed = abs(sum(diffs))
    hits = sum(
        1 for signs in itertools.product((1, -1), repeat=n)
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12
    )
    return hits / (2 ** n)


def load(d: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def cells(recs, depth: int, tag_has: str | None = None) -> dict:
    out = {}
    for r in recs:
        if r.get("depth") != depth or r.get("arm") != "regionalized":
            continue
        if tag_has and tag_has not in r.get("tag", ""):
            continue
        out[r["seed"]] = r
    return out


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--exp040-dir", type=Path,
                    default=HERE.parent / "040_pretrained_encoder_policy" / "outputs")
    args = ap.parse_args()

    recs = load(args.out_dir)
    if not recs:
        raise SystemExit(f"no records in {args.out_dir}. Has the run finished, or been fetched?")
    base_recs = load(args.exp040_dir)
    if not base_recs:
        raise SystemExit(
            f"no EXP-040 records in {args.exp040_dir}. They are the paired baseline and every "
            "claim needs them. They are gitignored - re-fetch from the laptop."
        )

    print(f"EXP-043: {len(recs)} records")
    if len(recs) != 24:
        print("  INCOMPLETE: expected 24. Verdicts below are provisional.")
    print("\nONE VARIABLE vs EXP-040: the depth-1 TRAINING budget, 5 -> 2.")
    print("Evaluation is untouched, so both arms are scored on the same yardstick.\n")

    print(f"{'depth':>6}{'EXP-040':>10}{'capped':>9}{'delta':>9}{'exact p':>10}"
          f"{'sd 040':>9}{'sd cap':>9}{'zeros':>12}")
    stats = {}
    for d in DEPTHS:
        cap = cells(recs, d, "exp043_capped")
        base = cells(base_recs, d)
        if not cap or not base:
            print(f"{d:>6}   (missing records)")
            continue
        seeds = sorted(set(cap) & set(base))
        diffs = [cap[s]["success_rate"] - base[s]["success_rate"] for s in seeds]
        cv = [cap[s]["success_rate"] for s in seeds]
        bv = [base[s]["success_rate"] for s in seeds]
        m, p = st.mean(diffs), permutation_p(diffs)
        zc = sum(1 for x in cv if x == 0)
        stats[d] = {"m": m, "p": p, "cap": cv, "base": bv, "seeds": seeds,
                    "zeros": zc, "se": sd(cv) / len(cv) ** 0.5}
        print(f"{d:>6}{st.mean(bv):>10.4f}{st.mean(cv):>9.4f}{m:>+9.4f}{p:>10.4f}"
              f"{sd(bv):>9.4f}{sd(cv):>9.4f}{f'{EXP040_ZEROS[d]} -> {zc}':>12}")

    print("\n" + "=" * 78)
    print(f"CLAIM 1 (PRIMARY), success vs EXP-040. Need >= {CLAIM1_DELTA} at p <= {ALPHA}.")
    confirmed = []
    for d in DEPTHS:
        if d not in stats:
            continue
        s = stats[d]
        wins = sum(1 for a, b in zip(s["cap"], s["base"]) if a > b)
        ties = sum(1 for a, b in zip(s["cap"], s["base"]) if a == b)
        print(f"  depth {d}: {s['m']:+.4f}  W-L-T {wins}-{len(s['cap'])-wins-ties}-{ties}"
              f"  exact p {s['p']:.4f}")
        if s["m"] >= CLAIM1_DELTA and s["p"] <= ALPHA:
            confirmed.append(d)
            print("            CONFIRMED. The depth-1 cap transfers to this depth.")
        else:
            print(f"            REFUTED at depth {d} ({s['m']:+.4f}, p {s['p']:.4f}).")

    print(f"\nCLAIM 2, does depth 6 become 'working'? Needs mean >= {WORKING_BAR}, margin "
          f">= {WORKING_MIN_SE} SE, and >= {WORKING_MIN_SEEDS}/12 seeds above the bar.")
    if 6 in stats:
        s = stats[6]
        mean6 = st.mean(s["cap"])
        above = sum(1 for x in s["cap"] if x >= WORKING_BAR)
        margin_se = (mean6 - WORKING_BAR) / s["se"] if s["se"] else float("nan")
        print(f"  mean {mean6:.4f} +-{s['se']:.4f} SE   margin {margin_se:+.2f} SE   "
              f"{above}/12 seeds above {WORKING_BAR}")
        print(f"  (EXP-040 reached {EXP040_MEAN[6]}, clearing the bare rule by 0.11 SE on "
              f"5/12 seeds - noise)")
        if mean6 >= WORKING_BAR and margin_se >= WORKING_MIN_SE and above >= WORKING_MIN_SEEDS:
            print("  WORKING. Depth 6 clears the bar with a real margin, not on noise.")
            print("  The break point has moved past depth 5.")
        elif mean6 >= WORKING_BAR:
            print("  AT THE BAR, NOT ABOVE IT. The bare rule fires but the margin or the seed")
            print("  count does not support it. Report as 'off the floor', not 'working' -")
            print("  the same call EXP-040 got.")
        else:
            print("  still BROKEN at depth 6.")

    print("\nCLAIM 3, failure count. DESCRIPTIVE - no p-value, by design.")
    for d in DEPTHS:
        if d in stats:
            print(f"  depth {d}: {EXP040_ZEROS[d]}/12 -> {stats[d]['zeros']}/12 seeds at 0.000")
    print("  Fisher's exact cannot distinguish these counts at n=12 (~0.48). Suggestive only.")

    print("\nCLAIM 4, does the variance collapse repeat?")
    print(f"  depth 4 (EXP-042): sd {EXP042_D4['baseline_sd']:.4f} -> "
          f"{EXP042_D4['capped_sd']:.4f}")
    for d in DEPTHS:
        if d not in stats:
            continue
        s = stats[d]
        ratio = sd(s["cap"]) / sd(s["base"]) if sd(s["base"]) else float("nan")
        print(f"  depth {d}: sd {sd(s['base']):.4f} -> {sd(s['cap']):.4f}  "
              f"(ratio {ratio:.2f})")
    print("  A repeat is evidence EXP-040's 'powerful but unreliable' caveat was largely THE")
    print("  TRAP, not the encoder.")

    print("\nCLAIM 5, the pre-committed null.")
    if not confirmed:
        print("  REFUTED AT BOTH DEPTHS while EXP-042 stands. The finding is that the depth-1")
        print("  trap is a DEPTH-4 phenomenon: deeper runs spend proportionally less of the")
        print("  budget in stage 1 and have more stages to recover in. Report as a SCOPING")
        print("  result, not a failure.")
    else:
        print(f"  Not triggered: confirmed at depth(s) {confirmed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
