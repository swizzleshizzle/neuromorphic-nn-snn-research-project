"""EXP-038 aggregator: apply the pre-registered rules to the records on disk.

Separate from run.py so the verdicts do not depend on catching a one-shot summary at the end
of a 21-hour job, and so they can be re-read any number of times.

Thresholds are NOT re-derived here. They are the ones committed in
docs/superpowers/specs/2026-08-07-exp038-depth6-collapse-design.md before any number existed.
If a number here disagrees with the spec, the spec wins and this file is the bug.

Needs EXP-036's records, which carry ALL FOUR comparators and were not re-run:
    depth 6 random (0.0008)   <- Claim 1's baseline. NOT the trained 0.0000; see below.
    depth 6 trained (0.0000)
    depth 5 trained (0.0396)  <- Claim 3's baseline
    depth 5 random  (0.0000)

Usage:
    .venv/bin/python experiments/038_depth6_collapse/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- pre-registered, from the spec. Do not edit after data exists. ---
D6_MIN_MEAN = 0.02      # 25x the measured floor, 4x the 1/200 resolution
D6_ALPHA = 0.017        # Bonferroni over the three depth-6 cells
D5_MIN_DELTA = 0.02     # ~0.75 sd of EXP-036's measured 0.0272
D5_ALPHA = 0.05
MODAL_LEARNING_MIN = 0.45   # Claim 2: below this the policy is sampling, not selecting
# Uniform modal fraction is BUDGET-DEPENDENT and 0.354 is the 9-step figure. Depth 6 runs
# 2d+3 = 15 steps and depth 5 runs 13, so neither is 0.354. These are measured directly from
# EXP-036's random arms on this machine with these seeds.
UNIFORM_MODAL = {5: 0.321, 6: 0.309}
# The instrument check is ENTROPY SATURATION, not "modal reaches uniform". An entropy bonus
# flattens the SAMPLED training policy; it does not make the GREEDY argmax vary, and pushing
# beta higher flattens the logits toward a deterministic tie-break, i.e. back toward a constant
# action. See spec section 5a. The pilot measured 95% of ceiling at beta=0.8.
ENTROPY_CEILING = math.log(6)              # 1.792
ENTROPY_SATURATED = 0.90 * ENTROPY_CEILING  # 1.613
FLOOR = {5: 0.0000, 6: 0.0008}   # EXP-036, same machine, same seeds


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


def by_seed(recs, depth, arm="regionalized", beta=None):
    out = {}
    for r in recs:
        if r["depth"] != depth or r["arm"] != arm:
            continue
        if beta is not None and r["config"]["entropy_beta"] != beta:
            continue
        out[r["seed"]] = r
    return out


def betas_at(recs, depth) -> list[float]:
    """Discover the swept betas from the records rather than hardcoding them, so the pilot's
    chosen values cannot drift out of sync with this file."""
    return sorted({r["config"]["entropy_beta"] for r in recs if r["depth"] == depth})


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def mean_of(arm, field):
    return st.mean(r[field] for r in arm.values()) if arm else float("nan")


def describe(label, arm):
    if not arm:
        return f"  {label:14s} (no records)"
    s = [r["success_rate"] for r in arm.values()]
    return (f"  {label:14s} n={len(s):2d}  mean {st.mean(s):.4f} +-{sd(s):.3f}  "
            f"best {max(s):.4f}  zeros {sum(1 for x in s if x == 0)}  "
            f"modal {mean_of(arm, 'greedy_modal_action_frac'):.3f}  "
            f"entropy {mean_of(arm, 'mean_train_entropy'):.3f}")


def paired(a: dict, b: dict, field="success_rate"):
    seeds = sorted(set(a) & set(b))
    return seeds, [a[s][field] - b[s][field] for s in seeds]


def compare(label, arm, base, base_label):
    if not arm or not base:
        print(f"  {label} vs {base_label}: missing records")
        return None, None
    _, diffs = paired(arm, base)
    if not diffs:
        print(f"  {label} vs {base_label}: no shared seeds")
        return None, None
    m, p = st.mean(diffs), permutation_p(diffs)
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    print(f"  {label} vs {base_label}: {m:+.4f}  W-L-T {wins}-{len(diffs)-wins-ties}-{ties}"
          f"  exact p {p:.4f}  (n={len(diffs)})")
    return m, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--exp036-dir", type=Path,
                    default=HERE.parent / "036_generalisation_gap" / "outputs")
    args = ap.parse_args()

    recs = load(args.out_dir)
    if not recs:
        raise SystemExit(f"no records in {args.out_dir}. Has the run finished, or been fetched?")
    base_recs = load(args.exp036_dir)
    if not base_recs:
        raise SystemExit(
            f"no EXP-036 records in {args.exp036_dir}. They carry ALL FOUR comparators; "
            "every claim here is paired against them and none can be evaluated without them."
        )

    print(f"EXP-038: {len(recs)} records")
    if len(recs) != 48:
        print("  INCOMPLETE: expected 48. Verdicts below are provisional.")

    d6_random = by_seed(base_recs, 6, arm="random")
    d6_trained = by_seed(base_recs, 6)
    d5_trained = by_seed(base_recs, 5)
    d5_random = by_seed(base_recs, 5, arm="random")

    d6_betas = betas_at(recs, 6)
    d5_betas = betas_at(recs, 5)
    d6_cells = [(b, by_seed(recs, 6, beta=b)) for b in d6_betas]

    print("\nDEPTH 6, dose axis (normalize_advantages=True throughout):")
    print(describe("EXP-036 rand", d6_random))
    print(describe("EXP-036 train", d6_trained))
    for b, arm in d6_cells:
        print(describe(f"beta {b}", arm))

    print("\npaired against the EXP-036 RANDOM arm (the floor, not the 0.0000 baseline):")
    stats = {}
    for b, arm in d6_cells:
        m, p = compare(f"beta {b:<5}", arm, d6_random, "random")
        stats[b] = (m, p, mean_of(arm, "success_rate"), mean_of(arm, "greedy_modal_action_frac"))

    print("\n" + "=" * 78)
    print("CLAIM 1 (PRIMARY), is it a lever at depth 6?")
    print(f"  bar: mean >= {D6_MIN_MEAN} AND p <= {D6_ALPHA} (Bonferroni over "
          f"{len(d6_cells)} cells) AND modal >= {MODAL_LEARNING_MIN}")
    # Print the whole ordering next to the verdict. EXP-037's aggregator printed a verdict
    # derived from ONE comparison the rule named, on a result the ordering did not support.
    print("  ordering (beta, mean, p, modal):")
    for b in d6_betas:
        m, p, mean_s, modal = stats[b]
        print(f"    {b:<6} {mean_s:.4f}  p {p if p is not None else float('nan'):.4f}  "
              f"modal {modal:.3f}")

    passing = [
        b for b in d6_betas
        if stats[b][1] is not None
        and stats[b][2] >= D6_MIN_MEAN
        and stats[b][1] <= D6_ALPHA
        and stats[b][3] >= MODAL_LEARNING_MIN
    ]
    arithmetic_only = [
        b for b in d6_betas
        if stats[b][1] is not None
        and stats[b][2] >= D6_MIN_MEAN
        and stats[b][1] <= D6_ALPHA
        and stats[b][3] < MODAL_LEARNING_MIN
    ]

    if passing:
        print(f"  LEVER ESTABLISHED at beta {passing}. Success cleared the random floor with the")
        print("  policy still selecting rather than sampling.")
    elif arithmetic_only:
        print(f"  REFUTED BY CLAIM 2. beta {arithmetic_only} cleared the arithmetic but its modal")
        print(f"  fraction is below {MODAL_LEARNING_MIN}, i.e. at the {UNIFORM_MODAL[6]} uniform")
        print("  anchor. That is RANDOMIZATION, NOT LEARNING (EXP-032 Finding 3). Not a lever.")
    else:
        print("  REFUTED at this budget. No cell cleared the random floor by the pre-registered")
        print("  margin. Depth 6's collapse is a SYMPTOM, not the binding constraint.")

    print("\nCLAIM 2 (DISCRIMINATOR), instrument check = ENTROPY SATURATION.")
    print("  (Not 'modal reaches uniform': an entropy bonus flattens the SAMPLED training")
    print("   policy, not the GREEDY argmax used at eval. Spec section 5a.)")
    if d6_cells:
        top_b, top_arm = d6_cells[-1]
        top_modal = mean_of(top_arm, "greedy_modal_action_frac")
        top_ent = mean_of(top_arm, "mean_train_entropy")
        print(f"  beta {top_b}: entropy {top_ent:.3f} = {100 * top_ent / ENTROPY_CEILING:.0f}% "
              f"of the {ENTROPY_CEILING:.3f} ceiling  (need >= {ENTROPY_SATURATED:.3f} = 90%)")
        print(f"           modal {top_modal:.3f} (uniform anchor {UNIFORM_MODAL[6]}), "
              f"mean {mean_of(top_arm, 'success_rate'):.4f}")
        if top_ent >= ENTROPY_SATURATED:
            print("  DOSE AXIS SATURATED. The sweep reached the limit of what the entropy bonus")
            print("  can do, which is what EXP-032's 'bounded too low' limitation required.")
        else:
            print("  SPAN STILL TOO LOW. The top beta did not saturate entropy, so this sweep")
            print("  repeats EXP-032's limitation and cannot say what happens past its boundary.")
            print("  Report that rather than a clean null.")

        # Any cell at/below the uniform anchor that still scores above the floor means the
        # measurement, not the policy, is the thing that moved.
        for depth, cells in ((6, d6_cells), (5, [(b, by_seed(recs, 5, beta=b)) for b in d5_betas])):
            for b, arm in cells:
                if not arm:
                    continue
                if (mean_of(arm, "greedy_modal_action_frac") <= UNIFORM_MODAL[depth] + 0.01
                        and mean_of(arm, "success_rate") > max(3 * FLOOR[depth], 0.005)):
                    print(f"  INSTRUMENT BROKEN. d{depth} beta {b} is at the uniform anchor yet")
                    print("  scores materially ABOVE the random floor. NO OTHER CLAIM IN THIS")
                    print("  EXPERIMENT MAY BE READ until this is explained.")

    print("\nCLAIM 3, depth 5 coherence (the powered arm).")
    print(describe("EXP-036 train", d5_trained))
    print(describe("EXP-036 rand", d5_random))
    for b in d5_betas:
        print(describe(f"beta {b}", by_seed(recs, 5, beta=b)))
    if not d5_betas:
        print("  cannot evaluate: no depth-5 records")
    for b in d5_betas:
        arm = by_seed(recs, 5, beta=b)
        m, p = compare(f"beta {b:<5}", arm, d5_trained, "036 trained")
        if m is None:
            continue
        if m >= D5_MIN_DELTA and p <= D5_ALPHA:
            print(f"  CONFIRMED. The stabilizer improved depth 5 by {m:+.4f} at p {p:.4f}.")
        else:
            print(f"  REFUTED ({m:+.4f}, p {p:.4f}; needs >= {D5_MIN_DELTA} and p <= {D5_ALPHA}).")

    print("\nCLAIM 4, mechanism. Modal fraction must be read WITH entropy, never entropy alone")
    print("(EXP-037 Claim 5: the two rose together in the back-loaded arms).")
    print(f"  {'depth':>6}{'beta':>7}{'modal':>9}{'entropy':>10}{'success':>10}")
    print(f"  {6:>6}{'036':>7}{mean_of(d6_trained, 'greedy_modal_action_frac'):>9.3f}"
          f"{mean_of(d6_trained, 'mean_train_entropy'):>10.3f}"
          f"{mean_of(d6_trained, 'success_rate'):>10.4f}")
    for b, arm in d6_cells:
        print(f"  {6:>6}{b:>7}{mean_of(arm, 'greedy_modal_action_frac'):>9.3f}"
              f"{mean_of(arm, 'mean_train_entropy'):>10.3f}"
              f"{mean_of(arm, 'success_rate'):>10.4f}")
    modals = [mean_of(a, "greedy_modal_action_frac") for _, a in d6_cells]
    if len(modals) > 1:
        monotone = all(x > y for x, y in zip(modals, modals[1:]))
        print(f"  modal falls monotonically with beta: {monotone}")

    print("\nCLAIM 5, the closing verdict.")
    d5_confirmed = any(
        (lambda mp: mp[0] is not None and mp[0] >= D5_MIN_DELTA and mp[1] <= D5_ALPHA)(
            (lambda d: (st.mean(d[1]), permutation_p(d[1])) if d[1] else (None, None))(
                paired(by_seed(recs, 5, beta=b), d5_trained)))
        for b in d5_betas
    )
    if passing or d5_confirmed:
        print("  The stabilizers remain live. Do NOT close them.")
    else:
        print("  BOTH REFUTED. Depth 6 was the strongest remaining case for the trainer")
        print("  stabilizers: the one regime where the failure they target is the failure the")
        print("  instruments diagnose. They are now CLOSED as a lever and join width (EXP-033),")
        print("  volume alone (EXP-034), curriculum weighting and starvation (EXP-037).")
        print("  The next move is the encoder (vault Stage 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
