"""EXP-040 aggregator: apply the pre-registered rules to the records on disk.

Separate from run.py so the verdicts do not depend on catching a one-shot summary at the end
of a 20-hour job, and so they can be re-read any number of times.

Thresholds are NOT re-derived here. They are the ones committed in
docs/superpowers/specs/2026-08-09-exp040-pretrained-encoder-policy-design.md before any number
existed. If a number here disagrees with the spec, the spec wins and this file is the bug.

Needs EXP-036's records, which carry all three comparators and were not re-run:
    depth 4 trained 0.1591, depth 5 trained 0.0396, depth 6 trained 0.0000

Usage:
    .venv/bin/python experiments/040_pretrained_encoder_policy/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- pre-registered, from the spec. Do not edit after data exists. ---
BAR1_DELTA = 0.05          # depth-4 pretrained - EXP-036, = 0.68 sd of the measured spread
ALPHA = 0.05
PRIMARY_DEPTH = 4
BREAK_ABSOLUTE = 0.10      # EXP-036's "working" bar
BREAK_MULTIPLE = 2.0
FLOOR = {4: 0.0031, 5: 0.0000, 6: 0.0008}      # EXP-036, same machine, same seeds
EXP036_MODAL = {4: 0.685, 5: 0.779, 6: 0.975}
DEPTHS = [4, 5, 6]
# EXP-039's probe, for the "did it convert?" framing. Not a bar.
EXP039_PROBE = {4: (0.447, 0.786), 5: (0.406, 0.660), 6: (0.344, 0.575)}
MOVE_LEARNED_MIN = 0.30    # pretraining sanity; EXP-039 measured 0.454


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


def by_seed(recs, depth, arm="regionalized", tag_contains=None):
    out = {}
    for r in recs:
        if r["depth"] != depth or r["arm"] != arm:
            continue
        if tag_contains and tag_contains not in r["tag"]:
            continue
        out[r["seed"]] = r
    return out


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def mean_of(arm, field):
    return st.mean(r[field] for r in arm.values()) if arm else float("nan")


def describe(label, arm):
    if not arm:
        return f"  {label:16s} (no records)"
    s = [r["success_rate"] for r in arm.values()]
    return (f"  {label:16s} n={len(s):2d}  mean {st.mean(s):.4f} +-{sd(s):.3f}  "
            f"best {max(s):.4f}  zeros {sum(1 for x in s if x == 0)}  "
            f"modal {mean_of(arm, 'greedy_modal_action_frac'):.3f}  "
            f"entropy {mean_of(arm, 'mean_train_entropy'):.3f}")


def paired(a: dict, b: dict, field="success_rate"):
    seeds = sorted(set(a) & set(b))
    return seeds, [a[s][field] - b[s][field] for s in seeds]


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
            f"no EXP-036 records in {args.exp036_dir}. They carry all three comparators and "
            "every claim here is paired against them. They are gitignored, so re-fetch from "
            "the laptop if outputs/ is empty."
        )

    print(f"EXP-040: {len(recs)} records")
    if len(recs) != 36:
        print("  INCOMPLETE: expected 36. Verdicts below are provisional.")

    print("\nEXACTLY ONE VARIABLE vs EXP-036: which weights the FROZEN encoder holds.")
    print("The head is still Linear(64 -> 6), 390 trainable parameters.\n")

    print(f"{'depth':>6}{'EXP-036':>10}{'pretrained':>12}{'delta':>9}{'exact p':>9}"
          f"{'floor':>9}{'probe was':>18}")
    stats = {}
    for d in DEPTHS:
        pre = by_seed(recs, d)
        base = by_seed(base_recs, d)
        if not pre or not base:
            print(f"{d:>6}   (missing records)")
            continue
        _, diffs = paired(pre, base)
        m, p = st.mean(diffs), permutation_p(diffs)
        stats[d] = (m, p, mean_of(pre, "success_rate"), pre, base)
        fr, to = EXP039_PROBE[d]
        print(f"{d:>6}{mean_of(base, 'success_rate'):>10.4f}"
              f"{mean_of(pre, 'success_rate'):>12.4f}{m:>+9.4f}{p:>9.4f}"
              f"{FLOOR[d]:>9.4f}{f'{fr:.3f}->{to:.3f}':>18}")

    print("\nper-depth detail:")
    for d in DEPTHS:
        if d in stats:
            print(describe(f"d{d} EXP-036", stats[d][4]))
            print(describe(f"d{d} pretrained", stats[d][3]))

    # ---- pretraining sanity (reported, not gated) ----
    accs = [r["config"].get("_move_accuracy") for r in recs]
    accs = [a for a in accs if a is not None]
    if accs:
        print(f"\npretraining move-naming accuracy: {st.mean(accs):.3f} "
              f"(EXP-039 measured 0.454; bar {MOVE_LEARNED_MIN})")

    print("\n" + "=" * 78)
    print(f"CLAIM 1 (PRIMARY), depth {PRIMARY_DEPTH}. "
          f"Need >= {BAR1_DELTA} at p <= {ALPHA}.")
    print("  (the POWERED arm: the only depth where the policy currently works, so the only")
    print("   one where an improvement is measurable. Depths 5-6 are resolution-bound at 1/200)")
    if PRIMARY_DEPTH not in stats:
        print("  cannot evaluate: no depth-4 records")
    else:
        m, p, mean_pre, pre, base = stats[PRIMARY_DEPTH]
        _, diffs = paired(pre, base)
        wins = sum(1 for x in diffs if x > 0)
        ties = sum(1 for x in diffs if x == 0)
        print(f"  {m:+.4f}  W-L-T {wins}-{len(diffs)-wins-ties}-{ties}  exact p {p:.4f}")
        if m >= BAR1_DELTA and p <= ALPHA:
            print("  CONFIRMED. The raised representational ceiling CONVERTS into policy")
            print("  success. The encoder was a binding constraint and pretraining relieved it.")
        else:
            print("  REFUTED. A raised ceiling does NOT convert into policy success at the")
            print("  frontier depth. See Claim 4 - this is informative, not a disappointment.")

    print("\nCLAIM 2, does the break point move?")
    print(f"  EXP-036's 'working' rule: >= {BREAK_MULTIPLE}x floor AND >= {BREAK_ABSOLUTE}")
    # The margin in STANDARD ERRORS is printed alongside the verdict. This adds precision to the
    # report; it does NOT change the pre-registered rule, which remains "mean >= bar". A rule
    # that fires on a margin inside its own noise is true-but-misleading, and EXP-037 logged
    # exactly that failure. Read the margin, not only the word.
    for d in (5, 6):
        if d not in stats:
            continue
        mean_pre = stats[d][2]
        arm = stats[d][3]
        vals = [r["success_rate"] for r in arm.values()]
        se = (sd(vals) / len(vals) ** 0.5) if len(vals) > 1 else float("nan")
        bar = max(BREAK_MULTIPLE * FLOOR[d], BREAK_ABSOLUTE)
        verdict = "WORKING" if mean_pre >= bar else "still BROKEN"
        margin_se = (mean_pre - bar) / se if se and se == se else float("nan")
        above = sum(1 for v in vals if v >= bar)
        print(f"  depth {d}: {mean_pre:.4f} +-{se:.4f} vs bar {bar:.4f} -> {verdict}"
              f"   margin {margin_se:+.2f} SE, {above}/{len(vals)} seeds above bar")
        if verdict == "WORKING" and margin_se < 1.0:
            print(f"    CAUTION: clears by {margin_se:+.2f} SE. The rule fires, but the margin")
            print(f"    is inside the noise. Do NOT report depth {d} as solidly working.")
    if 5 in stats and stats[5][2] >= max(BREAK_MULTIPLE * FLOOR[5], BREAK_ABSOLUTE):
        print("  THE BREAK POINT MOVES. Depth 5 has been broken since EXP-036 and nothing has")
        print("  shifted it until now. This is the most consequential cube result to date.")
    else:
        print("  Break point unchanged at depth 5.")

    print("\nCLAIM 3, mechanism. Did a gain arrive WITH modal fraction falling?")
    print("  (read modal WITH entropy - EXP-035, EXP-037 and EXP-038 each produced a different")
    print("   relationship between the two, so neither alone is the general rule)")
    print(f"  {'depth':>6}{'modal 036':>11}{'modal pre':>11}{'entropy pre':>13}{'delta succ':>12}")
    for d in DEPTHS:
        if d not in stats:
            continue
        m, _, _, pre, base = stats[d]
        print(f"  {d:>6}{EXP036_MODAL[d]:>11.3f}"
              f"{mean_of(pre, 'greedy_modal_action_frac'):>11.3f}"
              f"{mean_of(pre, 'mean_train_entropy'):>13.3f}{m:>+12.4f}")

    print("\nCLAIM 4, the null is pre-committed and informative.")
    m4 = stats.get(PRIMARY_DEPTH, (None,))[0]
    if m4 is not None and not (m4 >= BAR1_DELTA and stats[PRIMARY_DEPTH][1] <= ALPHA):
        print("  EXP-039 raised the depth-4 probe 0.447 -> 0.786, past the 0.742 facelet")
        print("  ceiling, at p 0.0005. If that stands and the policy did not move, then the")
        print("  REPRESENTATION WAS NEVER THE BINDING CONSTRAINT on the policy - the readout or")
        print("  the learning signal is. That is EXP-033 Finding 2 writ large (oracle probe 0.48")
        print("  vs RL 0.22 at depth 3) and the strongest evidence yet for Stage 3: a value")
        print("  function carried on the currently idle `neuromod` pathway.")
        print("  Report this as a positive redirection, NOT as 'inconclusive, needs a bigger")
        print("  encoder'.")
    else:
        print("  Not triggered: Claim 1 confirmed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
