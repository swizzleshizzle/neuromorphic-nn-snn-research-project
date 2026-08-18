"""EXP-046 aggregator: apply the pre-registered rules to the records on disk.

WRITTEN BEFORE THE RUN WAS DISPATCHED. No record existed. Rules on disk before numbers.

Thresholds are the ones committed in
docs/superpowers/specs/2026-08-18-exp046-depth6-budget-design.md.

> CLAIM 1 IS PAIRED AND CARRIES A P-VALUE, because a real baseline exists: EXP-043's depth-6
> cell, same seeds, same encoders, same cap, same curriculum, differing only in `episodes`.
> EXP-044's Claim 1 had no baseline and deliberately carried none.

> AND AS ALWAYS: n=12 cannot show a failure count went to zero. Claim 4 is printed WITHOUT a
> p-value, deliberately.

Usage:
    .venv/bin/python experiments/046_depth6_budget/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "043_cap_at_depth_5_6" / "outputs"

# --- pre-registered. Do not edit after data exists. ---
CLAIM1_DELTA = 0.05
ALPHA = 0.05
DEPTH = 6
EPISODES = 44_000
EPISODES_MID = 25_000
D7_GAIN = 0.1350             # EXP-044 arm A -> arm B, same 4.4x multiplier
# EXP-045's collapse signature, for the mechanism comparison.
EXP045 = {"entropy_first": 0.5914, "entropy_last": 0.0979, "entropy_min": 2.7e-06,
          "solve_frac": 0.0218, "modal": 0.847}


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


def deepest_stage(rec):
    trace = rec.get("stage_trace") or []
    return next((s for s in reversed(trace) if s.get("depth") == DEPTH), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    args = ap.parse_args()

    new = load(args.out_dir, f"exp046_d{DEPTH}_e{args.episodes}")
    base = load(BASELINE, f"exp043_capped_d{DEPTH}")
    if not new:
        raise SystemExit(f"no EXP-046 records for {args.episodes:,} episodes in {args.out_dir}")
    if not base:
        raise SystemExit(f"no EXP-043 depth-6 records in {BASELINE} - they are the paired baseline")

    seeds = sorted(set(new) & set(base))
    vals = [new[s]["success_rate"] for s in seeds]
    diffs = [new[s]["success_rate"] - base[s]["success_rate"] for s in seeds]
    n = len(seeds)
    arm = "PRIMARY" if args.episodes == EPISODES else (
        "MIDPOINT" if args.episodes == EPISODES_MID else "?")

    print(f"{'=' * 78}\nEXP-046 {arm} - depth {DEPTH} at {args.episodes:,} episodes, "
          f"{n} seeds paired against EXP-043\n{'=' * 78}")

    # --- Claim 1 (PRIMARY), paired ---
    delta = st.mean(diffs)
    p = permutation_p(diffs)
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    confirmed = delta >= CLAIM1_DELTA and p <= ALPHA
    print(f"\nCLAIM 1 (PRIMARY) - does 4.4x help depth {DEPTH}?   "
          f"{'CONFIRMED' if confirmed else 'REFUTED'}")
    print(f"  mean {st.mean(vals):.4f} (sd {sd(vals):.4f}) against EXP-043's "
          f"{st.mean([base[s]['success_rate'] for s in seeds]):.4f}")
    print(f"  paired delta {delta:+.4f}, W-L-T {wins}-{losses}-{n - wins - losses}, exact p {p:.4f}")
    print(f"    delta >= +{CLAIM1_DELTA}   {'PASS' if delta >= CLAIM1_DELTA else 'FAIL'}")
    print(f"    p <= {ALPHA}          {'PASS' if p <= ALPHA else 'FAIL'}")
    print(f"  depth 7 gained {D7_GAIN:+.4f} from the same multiplier; depth {DEPTH} recovers "
          f"{delta / D7_GAIN * 100:.0f}% of that.")
    if not confirmed and 0 < delta < CLAIM1_DELTA:
        print("  NOTE: positive but under the bar. That refutes the strong reading WITHOUT")
        print("  showing the effect is zero. Report it that way.")

    # --- Claim 2, the escalation ---
    print("\nCLAIM 2 - escalation")
    if arm == "MIDPOINT":
        print("  This IS the midpoint. Compare its delta with the primary arm's to see whether")
        print("  returns were still climbing at 44,000 or already flattening.")
    elif confirmed:
        print(f"  Claim 1 CONFIRMED, so the midpoint IS triggered. Dispatch:")
        print(f"    run.py --episodes {EPISODES_MID} --seeds 0 1 2 3 4 5 6 7 8 9 10 11 "
              "--workers 12 --skip-existing")
        print("  ~13 h. It answers whether returns are still climbing at 44,000.")
    else:
        print("  Claim 1 REFUTED, so the midpoint is NOT run (Claim 2, pre-registered).")

    # --- Claim 3, MECHANISM, descriptive ---
    print("\nCLAIM 3 - MECHANISM, descriptive")
    st_new = [deepest_stage(new[s]) for s in seeds]
    st_new = [s for s in st_new if s]
    if st_new:
        f = st.mean([s["entropy_first_10pct"] for s in st_new])
        l = st.mean([s["entropy_last_10pct"] for s in st_new])
        m = st.mean([s["entropy_min"] for s in st_new])
        sf = st.mean([s["train_solved_frac"] for s in st_new])
        print(f"  deepest stage: entropy {f:.4f} -> {l:.4f}, min {m:.2e}, solve rate {sf:.4f}")
    print(f"  EXP-045 (collapsed): entropy {EXP045['entropy_first']:.4f} -> "
          f"{EXP045['entropy_last']:.4f}, min {EXP045['entropy_min']:.1e}, "
          f"solve rate {EXP045['solve_frac']:.4f}")
    modal_new = [new[s].get("greedy_modal_action_frac") for s in seeds]
    modal_base = [base[s].get("greedy_modal_action_frac") for s in seeds]
    if all(m is not None for m in modal_new + modal_base):
        print(f"  modal action fraction: {st.mean(modal_new):.3f} here, "
              f"{st.mean(modal_base):.3f} in EXP-043 (EXP-045 collapsed at {EXP045['modal']:.3f})")

    # --- Claim 4, descriptive, NO p-value ---
    print(f"\nCLAIM 4 - failure counts, DESCRIPTIVE\n  {sum(1 for v in vals if v == 0.0)}/{n} "
          f"seeds at exactly 0.000, against EXP-043's "
          f"{sum(1 for s in seeds if base[s]['success_rate'] == 0.0)}/{n}")
    print("  no p-value: n=12 cannot show a count went to zero.")

    # --- Claim 5 ---
    print("\nCLAIM 5 - what this means")
    if confirmed:
        print("  Depth 6 responds to budget as depth 7 did. THE DEPTH SERIES IS A BUDGET SERIES:")
        print("  every 'depth N stopped working' result here measured depth AT 10,000 EPISODES,")
        print("  and the published numbers need restating with that attached.")
    else:
        print("  Depth 7 was SPECIFICALLY starved; the series is NOT simply a budget series.")
        print("  Most plausible reason: depth 7's shell is 3.7x depth 6's (33,058 vs 8,969), so")
        print("  the budget requirement tracks how much there is to learn, not depth as such.")
        print("  Earlier numbers stand as published, with 'at 10,000 episodes' as a caveat.")

    print("\nper-seed (EXP-046, EXP-043, delta):")
    for s in seeds:
        a, b = new[s]["success_rate"], base[s]["success_rate"]
        print(f"  s{s:<3} {a:.4f}  {b:.4f}  {a - b:+.4f}")


if __name__ == "__main__":
    main()
