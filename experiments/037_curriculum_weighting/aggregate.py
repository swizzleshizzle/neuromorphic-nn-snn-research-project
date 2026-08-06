"""EXP-037 aggregator: apply the pre-registered rules to the records on disk.

Separate from run.py so the verdicts do not depend on catching a one-shot summary at the end
of a 24-hour job, and so they can be re-read any number of times.

Thresholds are NOT re-derived here. They are the ones committed in
docs/superpowers/specs/2026-08-05-exp037-curriculum-weighting-design.md before any number
existed. If a number here disagrees with the spec, the spec wins and this file is the bug.

Needs EXP-036's records for the 25% comparator, which are the paired baseline:
    --exp036-dir experiments/036_generalisation_gap/outputs

Usage:
    .venv/bin/python experiments/037_curriculum_weighting/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

LEVER_DELTA = 0.03      # 50% must beat 25% by at least this
ALPHA = 0.05
BREAK_ABSOLUTE = 0.10   # EXP-036's absolute bar, reused for the depth-6 claim
BREAK_MULTIPLE = 2.0
FLOOR = {4: 0.0031, 6: 0.0008}   # measured in EXP-036, same machine, same seeds


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


def by_seed(recs, depth, tag_contains=None, arm="regionalized"):
    out = {}
    for r in recs:
        if r["depth"] != depth or r["arm"] != arm:
            continue
        if tag_contains and tag_contains not in r["tag"]:
            continue
        out[r["seed"]] = r
    return out


def paired(a: dict, b: dict, field="success_rate"):
    """Per-seed differences a - b over the seeds both arms share."""
    seeds = sorted(set(a) & set(b))
    return seeds, [a[s][field] - b[s][field] for s in seeds]


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def describe(label, arm):
    if not arm:
        return f"  {label:12s} (no records)"
    s = [r["success_rate"] for r in arm.values()]
    return (f"  {label:12s} n={len(s):2d}  mean {st.mean(s):.4f} +-{sd(s):.3f}  "
            f"best {max(s):.4f}  zeros {sum(1 for x in s if x == 0)}  "
            f"modal {st.mean(r['greedy_modal_action_frac'] for r in arm.values()):.3f}  "
            f"entropy {st.mean(r['mean_train_entropy'] for r in arm.values()):.3f}")


def compare(label, arm, base, base_label):
    """Paired comparison against the baseline, with the exact permutation p."""
    if not arm or not base:
        print(f"  {label} vs {base_label}: missing records")
        return None, None
    seeds, diffs = paired(arm, base)
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
            f"no EXP-036 records in {args.exp036_dir}. They ARE the 25% comparator; "
            "every claim here is paired against them and none can be evaluated without them."
        )

    print(f"EXP-037: {len(recs)} records ({len(recs)//12 if recs else 0} cells x 12 seeds)")
    if len(recs) != 48:
        print(f"  INCOMPLETE: expected 48. Verdicts below are provisional.")

    d4_25 = by_seed(base_recs, 4)              # EXP-036 equal arm = 25% share
    d6_eq = by_seed(base_recs, 6)              # EXP-036 equal arm at depth 6 = 16.7% share
    d4_125 = by_seed(recs, 4, "s0125")
    d4_50 = by_seed(recs, 4, "s0500")
    d4_75 = by_seed(recs, 4, "s0750")
    d6_50 = by_seed(recs, 6, "s0500")

    print("\nDEPTH 4, share of the budget at the evaluated depth:")
    print(describe("12.5%", d4_125))
    print(describe("25% (036)", d4_25))
    print(describe("50%", d4_50))
    print(describe("75%", d4_75))

    print("\npaired against the 25% arm:")
    m50, p50 = compare("50%  ", d4_50, d4_25, "25%")
    m75, p75 = compare("75%  ", d4_75, d4_25, "25%")
    m125, p125 = compare("12.5%", d4_125, d4_25, "25%")

    print("\n" + "=" * 78)
    print("CLAIM 1, is weighting a lever?")
    if m50 is None:
        print("  cannot evaluate: the 50% arm or its comparator is missing")
    elif m50 >= LEVER_DELTA and p50 <= ALPHA:
        print(f"  LEVER ESTABLISHED. 50% beats 25% by {m50:+.4f} at p {p50:.4f}.")
        print("  The equal split was leaving performance on the table.")
    else:
        print(f"  REFUTED at this budget ({m50:+.4f}, p {p50:.4f}; needs >= {LEVER_DELTA} "
              f"and p <= {ALPHA}).")
        print("  The equal default stands. Do NOT hunt for another share that works.")

    print("\nCLAIM 2, is there an interior optimum?")
    if m50 is None or m75 is None:
        print("  cannot evaluate")
    else:
        seeds, diffs = paired(d4_50, d4_75)
        m, p = st.mean(diffs), permutation_p(diffs)
        print(f"  50% vs 75%: {m:+.4f}, exact p {p:.4f}")
        # The pre-registered rule is 50% > 75%. But that rule was written expecting 50% to
        # BEAT 25%; if it does not, "interior optimum between 50 and 100" is the wrong
        # sentence, because the peak is then at or below 25%. Read the ordering, not just
        # the one comparison the rule names.
        if m50 is not None and m50 <= 0:
            print("  The peak is at or BELOW 25%, not between 50% and 100%. Both back-loaded")
            print("  arms score worse than the equal split, so the pre-registered wording")
            print("  ('interior optimum') does not apply: the curve is DECLINING across the")
            print("  whole tested range from 25% upward.")
        elif m > 0:
            print("  INTERIOR OPTIMUM. Consistent with EXP-034's refuted 100% endpoint;")
            print("  the curve turns over between 50% and 100%.")
        else:
            print("  NO TURNOVER in range. More back-loading is still live;")
            print("  the next experiment is 85/95%, not more seeds here.")

    print("\nCLAIM 3, the control. 12.5% must be WORSE than 25%.")
    if m125 is None:
        print("  cannot evaluate")
    elif m125 < 0 and p125 is not None and p125 <= ALPHA:
        print(f"  CONTROL HOLDS ({m125:+.4f}, p {p125:.4f}). Starving the evaluated depth hurt,")
        print("  so success tracks share at the evaluated depth.")
    elif m125 < 0:
        # A slightly negative point estimate at p ~ 0.9 is indistinguishable from zero.
        # Calling that "the control holds" is exactly the weak inference this repo keeps
        # catching: the sign of a noisy estimate is not evidence.
        print(f"  INDISTINGUISHABLE FROM 25% ({m125:+.4f}, p {p125:.4f}). HALVING the share at")
        print("  the evaluated depth changed nothing measurable. Combined with both")
        print("  back-loaded arms scoring worse, success does NOT track share in the way the")
        print("  starvation hypothesis predicted. Do not report this as the control holding.")
    else:
        print(f"  CONTROL FAILED ({m125:+.4f}). Starving the evaluated depth did NOT hurt,")
        print("  so performance is not tracking share and any 50% win needs a different")
        print("  explanation. Report this rather than keeping the headline.")

    print("\nCLAIM 4, depth 6 at 50% share (vs EXP-036's 0.0000 on all twelve):")
    if not d6_50:
        print("  cannot evaluate: no depth-6 records")
    else:
        s = [r["success_rate"] for r in d6_50.values()]
        nz = sum(1 for x in s if x > 0)
        bar = max(BREAK_MULTIPLE * FLOOR[6], BREAK_ABSOLUTE)
        print(f"  mean {st.mean(s):.4f}, best {max(s):.4f}, {nz}/{len(s)} seeds above zero")
        if st.mean(s) >= bar:
            print(f"  CLEARS the EXP-036 break bar ({bar:.4f}). Depth 6 is reachable and the")
            print("  break point moves.")
        elif nz > 0:
            print("  MOVED OFF THE FLOOR but does not clear the break bar. Depth 6 was partly")
            print("  starved; starvation is not the whole story.")
        else:
            print("  STILL ZERO. Depth 6's failure is NOT starvation. It is the collapse the")
            print("  instruments showed (modal fraction 0.975); the lever is EXP-031/032, not")
            print("  the curriculum.")
        print(f"  modal {st.mean(r['greedy_modal_action_frac'] for r in d6_50.values()):.3f} "
              f"(EXP-036 equal arm: "
              f"{st.mean(r['greedy_modal_action_frac'] for r in d6_eq.values()):.3f})"
              if d6_eq else "")

    print("\nCLAIM 5, mechanism. Did a gain arrive with modal fraction FALLING?")
    for label, arm in (("12.5%", d4_125), ("25%", d4_25), ("50%", d4_50), ("75%", d4_75)):
        if arm:
            print(f"  {label:6s} modal {st.mean(r['greedy_modal_action_frac'] for r in arm.values()):.3f}"
                  f"  entropy {st.mean(r['mean_train_entropy'] for r in arm.values()):.3f}")

    print("\nCLAIM 6, the confound. Env steps per run at an identical 10,000-episode budget:")
    print("  12.5%  75,008   (-6.2% vs the 25% arm)")
    print("  25%    80,000")
    print("  50%    90,008   (+12.5%)")
    print("  75%   100,004   (+25.0%)")
    print("  A 75% win by a margin under 25% is NOT separable from its extra compute here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
