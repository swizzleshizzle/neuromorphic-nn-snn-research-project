"""EXP-049 aggregator: apply the pre-registered rules to the records on disk.

WRITTEN BEFORE THE RUN WAS DISPATCHED. No record existed. Rules on disk before numbers.

Thresholds committed at aaca17c in
docs/superpowers/specs/2026-08-24-exp049-second-round-design.md.

> CLAIM 1 IS PREDICTED TO REFUTE, and that prediction is in the spec and the driver, not added
> here. Refutation with Claim 2 near +0.05 is the EXPECTED result and means constant returns.

> THE UNINTERPRETABLE BAND IS PRE-REGISTERED. The extra round's own compute is worth +0.0432 on
> EXP-046's curve against a +0.05 bar, so a delta in between is printed as UNINTERPRETABLE rather
> than as a near-miss.

Usage:
    .venv/bin/python experiments/049_second_round/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM_A_DIR = HERE.parent / "043_cap_at_depth_5_6" / "outputs"
ARM_B_DIR = HERE.parent / "048_fresh_head" / "outputs"
ARM_C_DIR = HERE.parent / "047_encoder_finetuning" / "outputs"

# --- pre-registered. Do not edit after data exists. ---
BAR = 0.05
ALPHA = 0.05
DEPTH = 6
RATE = 0.22                  # EXP-046: success per log10 of spend at depth 6
ARM_A_MEAN = 0.1800
COMPUTE = {"A": 1.00, "C": 1.33, "B": 2.33, "D": 2.66, "E": 3.66}
EXCESS_PRIOR = {"C": 0.0628, "B": 0.0504}
E_MINUS_B_BUDGET_EQUIV = 0.0432
FROZEN_TRAINABLE, FINETUNE_TRAINABLE = 390, 27_206

TAGS = {"A": "exp043_capped_d6", "B": "exp048_freshhead_d6",
        "C": "exp047_ft_d6_lr0.0001", "D": "exp049_ft2_d6", "E": "exp049_fresh2_d6"}


def permutation_p(diffs):
    n = len(diffs)
    obs = abs(sum(diffs))
    hits = sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12)
    return hits / 2 ** n


def load(d: Path, tag: str) -> dict:
    out = {}
    for p in Path(d).glob("*.json"):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and r.get("tag") == tag and r.get("depth") == DEPTH:
            out[r["seed"]] = r
    return out


def budget_equiv(arm: str) -> float:
    return ARM_A_MEAN + RATE * math.log10(COMPUTE[arm])


def paired(label, X, Y, seeds):
    d = [X[s]["success_rate"] - Y[s]["success_rate"] for s in seeds]
    delta, p = sum(d) / len(d), permutation_p(d)
    w = sum(1 for v in d if v > 0)
    l = sum(1 for v in d if v < 0)
    print(f"  {label}: {delta:+.4f}   W-L-T {w}-{l}-{len(d)-w-l}   exact p {p:.4f}")
    return delta, p, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    arms = {"A": load(ARM_A_DIR, TAGS["A"]), "B": load(ARM_B_DIR, TAGS["B"]),
            "C": {s: r for s, r in load(ARM_C_DIR, TAGS["C"]).items() if s < 12},
            "D": load(args.out_dir, TAGS["D"]), "E": load(args.out_dir, TAGS["E"])}
    if not arms["E"]:
        raise SystemExit(f"no arm E records tagged {TAGS['E']} in {args.out_dir}")
    seeds = sorted(set.intersection(*(set(a) for a in arms.values())))
    if not seeds:
        raise SystemExit("no seed present in all five arms")

    print("=" * 78)
    print(f"EXP-049: does the two-stage recipe COMPOUND? depth {DEPTH}, n={len(seeds)}")
    print("=" * 78)

    print("\nSANITY - arm D must be fine-tuned, arm E must be frozen")
    for arm, want in (("D", FINETUNE_TRAINABLE), ("E", FROZEN_TRAINABLE)):
        got = {r["trainable_params"] for r in arms[arm].values()}
        print(f"  arm {arm} trainable_params {sorted(got)}  (expected [{want}])")
        if got != {want}:
            raise SystemExit(f"arm {arm} is not what the design says it is.")

    print(f"\nTHE SERIES  (budget-equivalent from EXP-046's {RATE}/log10)")
    print(f"  {'arm':>3}  {'compute':>7}  {'budget-eq':>9}  {'actual':>7}  {'excess':>8}")
    for arm in ("A", "C", "B", "D", "E"):
        m = st.mean(arms[arm][s]["success_rate"] for s in seeds)
        be = budget_equiv(arm)
        tail = "" if arm == "A" else f"  {m - be:+8.4f}"
        print(f"  {arm:>3}  {COMPUTE[arm]:7.2f}  {be:9.4f}  {m:7.4f}{tail}")

    print(f"\nCLAIM 1 (PRIMARY) - does a second round beat what its compute buys?")
    delta, p, diffs = paired("E - B", arms["E"], arms["B"], seeds)
    if delta >= BAR and p <= ALPHA:
        verdict = "CONFIRMED - the recipe COMPOUNDS. This was the surprise outcome."
    elif delta > E_MINUS_B_BUDGET_EQUIV:
        verdict = (f"UNINTERPRETABLE - between the +{E_MINUS_B_BUDGET_EQUIV} the extra round's "
                   f"compute alone buys and the +{BAR} bar.")
    else:
        verdict = "REFUTED - as predicted. See Claim 2 for what that means."
    print(f"  bar: >= +{BAR} at p <= {ALPHA}")
    print(f"  VERDICT: {verdict}")
    print(f"  per-seed: " + "  ".join(f"{d:+.3f}" for d in diffs))

    print(f"\nCLAIM 2 (THE REAL QUANTITY) - excess over budget-equivalent, in series")
    e_excess = st.mean(arms["E"][s]["success_rate"] for s in seeds) - budget_equiv("E")
    print(f"  arm C {EXCESS_PRIOR['C']:+.4f}   arm B {EXCESS_PRIOR['B']:+.4f}   "
          f"arm E {e_excess:+.4f}")
    if e_excess > 0.08:
        print("  >> COMPOUNDING. Iterating is the route past EXP-046's budget wall.")
    elif e_excess < 0.03:
        print("  >> DIMINISHING. The first round was special - most likely because E0 was")
        print("     pretrained on a different objective and had the most to gain.")
    else:
        print("  >> CONSTANT RETURNS, as predicted. Each round buys a fixed increment over its")
        print("     own cost. Iterating is a better way to spend compute, NOT an escape from the")
        print("     budget wall. Next move is a different second-stage objective, not a round 3.")

    print(f"\nCLAIM 3 - does round 2 help the co-adapted arm too?")
    paired("D - C", arms["D"], arms["C"], seeds)

    print(f"\nCLAIM 4 (MECHANISM) - falsifiable prediction from EXP-048")
    print(f"  EXP-048 found revisits FELL and optimality ROSE while probe accuracy went down.")
    for k, want in (("eval_revisit_rate", "lower"), ("optimality", "higher")):
        a = [arms["B"][s][k] for s in seeds]
        b = [arms["E"][s][k] for s in seeds]
        d = [y - x for x, y in zip(a, b)]
        print(f"  {k:20s} B {st.mean(a):.4f} -> E {st.mean(b):.4f}  "
              f"{st.mean(d):+.4f}  p {permutation_p(d):.4f}   (predicted {want})")
    print("  If success rose while BOTH stayed flat, EXP-048's mechanism does not generalise")
    print("  to round 2 and the explanation is incomplete.")

    print(f"\nCLAIM 5 - the probe should drift DOWN again (E2 below E1)")
    print(f"  run: .venv/bin/python experiments/048_fresh_head/diagnose_probe_tension.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
