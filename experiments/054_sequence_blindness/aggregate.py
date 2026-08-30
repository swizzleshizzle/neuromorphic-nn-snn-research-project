"""EXP-054 aggregator: apply the pre-registered rules to the records on disk.

Thresholds committed in the spec before any number existed.

> A RANDOM ENCODER MAY SCORE HIGHEST and that is the RESULT, not a failure of it. Random
> projections preserve geometry, so E0 separating shells best while scoring 0.0000 policy would
> say the 10-epoch optimum is a tradeoff: pretraining buys move-structure and spends
> sequence-structure from the first epoch. This is written down in the spec, in advance.

Usage:
    .venv/bin/python experiments/054_sequence_blindness/aggregate.py
"""

from __future__ import annotations

import itertools
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from neuromorphic.analysis.sequence_sensitivity import sensitivity_from_similarity, sim_from_record

HERE = Path(__file__).resolve().parent

ALPHA = 0.05
ADJACENT = (("E10", "E20"), ("E20", "E40"), ("E40", "E80"))
MOVE_ACCURACY = {"E0": None, "E10": 0.383, "E20": 0.414, "E40": 0.437, "E80": 0.452}
ORDER = ("E0", "E10", "E20", "E40", "E80")


def permutation_p(diffs) -> float:
    """Exact two-sided paired permutation over all 2**n sign flips. No scipy in the venv."""
    n, obs = len(diffs), abs(sum(diffs))
    return sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12) / 2 ** n


def spearman(x, y) -> float:
    """Rank correlation. n is 4 or 12 here, so an exact library is not needed."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for pos, i in enumerate(order):
            out[i] = pos + 1
        return out

    rx, ry = rank(list(x)), rank(list(y))
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def claim4_verdict(within: dict, between: float) -> str:
    """The pre-registered disqualifier, as a function rather than a paragraph.

    Encoding it removes the step where a human re-derives the rule while looking at the
    numbers. EXP-050's Claim 4 was satisfied and its inference was still wrong; EXP-052's
    aggregator named a shape from indistinguishable means. This is the response to both.
    """
    signs = {k: (1 if v > 0 else -1 if v < 0 else 0) for k, v in within.items()}
    nonzero = [s for s in signs.values() if s != 0]
    if not nonzero or len(set(nonzero)) > 1:
        return ("CLAIM 4 INCONCLUSIVE. The within-arm correlations disagree with each other "
                f"({within}), so there is no coherent within-arm sign to compare against the "
                f"between-arm {between:+.3f}. S is neither cleared nor retired; report the "
                "correlations and do not use S as a diagnostic until this resolves.")
    within_sign = nonzero[0]
    between_sign = 1 if between > 0 else -1 if between < 0 else 0
    if between_sign != 0 and within_sign != between_sign:
        return ("CLAIM 4 TRIPPED. S IS RETIRED. The within-arm correlations "
                f"({within}) and the between-arm correlation ({between:+.3f}) carry OPPOSITE "
                "SIGNS. S is a fifth inverted instrument, alongside the EXP-033 probe, "
                "pretraining move-accuracy and the entropy trace. It may not be used as a "
                "diagnostic and may not appear in a later spec. Report this in the headline, "
                "not in a caveat.")
    return ("CLAIM 4 PASSED. Within-arm and between-arm correlations agree in sign "
            f"({within}, between {between:+.3f}). S is not disqualified. This is NOT the same "
            "as S being a good predictor of policy - see Claim 2 for the tradeoff reading.")


def arm_policy_mean(records: dict) -> float:
    """Mean `policy_success` across an arm's seeds - the summary table's `policy` column.

    Must be a mean over seeds, never an arbitrary one: this used to be
    `next(iter(records.values()))["policy_success"]`, which reports whichever seed's JSON file
    the filesystem glob happened to yield first. E10 seed 0 is 0.285 against the arm's actual
    mean of 0.2012 - and this column is Claim 2's deliverable, the single most-read number in
    the output.
    """
    return st.mean([r["policy_success"] for r in records.values()])


def s_cross(record: dict) -> float:
    """`S_cross`: the same negated-slope fit as `S`, restricted to pairs with `|d1-d2| >= 1`.

    `S` is computed over ALL pairs including `|dd| = 0`, the within-shell term, which measures
    clustering rather than distance ordering. Measured on this repo's own `_shell_structured`
    test fixture - shell centres that are independent random directions with NO distance
    ordering at all - `S` comes out to 0.26708 while `S_cross` on the same fixture is 0.02318.
    So `S` alone cannot tell "shells cluster" from "shells are graded by distance"; `S_cross` is
    reported beside it as an equally prominent quantity for exactly that reason. See the spec
    amendment. Computed from the record's stored `sim` dict, so no re-run is needed.
    """
    return sensitivity_from_similarity(sim_from_record(record["sim"]), min_separation=1)


def level(record: dict) -> float:
    """The collapse control: the grand mean over ALL stored similarities in this record,
    within-shell and cross-shell alike, unfiltered by separation.

    If pretraining drives every cosine toward a constant, `S` (and `S_cross`) fall to zero with
    zero sequence-blindness in the mechanism sense - the code has lost distance structure only
    because it has lost ALL structure. `level` is what would fall alongside `S` in that case,
    and its printed note exists so that reading is available at a glance rather than requiring
    the raw `sim` dict to be re-derived by hand.
    """
    return st.mean(record["sim"].values())


def arm_trips_alone(corr: float, between: float) -> bool:
    """Would THIS ARM'S within-arm correlation, alone, trip the Claim 4 disqualifier against
    the between-arm sign?

    `claim4_verdict` requires all four within-arm signs to agree with each other before it
    compares them against the between-arm sign - four independent n=12 Spearman correlations
    agree in sign only about 12.5% of the time under noise, so INCONCLUSIVE is the modal
    outcome. The precedent the spec cites - one arm at +0.881 against an opposite between-arm
    sign - would be swallowed by that requirement rather than caught. This function is
    descriptive only: it changes no verdict, but lets a reader see an individual opposing arm
    even when the aggregate verdict is INCONCLUSIVE.
    """
    corr_sign = 1 if corr > 0 else -1 if corr < 0 else 0
    between_sign = 1 if between > 0 else -1 if between < 0 else 0
    return corr_sign != 0 and between_sign != 0 and corr_sign != between_sign


def load(out_dir: Path) -> dict:
    by_arm = defaultdict(dict)
    for p in Path(out_dir).glob("exp054_*.json"):
        r = json.loads(p.read_text())
        by_arm[r["arm"]][r["seed"]] = r
    return by_arm


def separation_table(by_arm: dict) -> dict:
    """Descriptive only: mean similarity at each shell separation |dd|, per arm, over seeds.

    S is a slope fitted over separations INCLUDING |d1 - d2| = 0, so it is partly driven by
    within-shell tightness (clustering) rather than purely by graded ordering across distances.
    That is the pre-registered formula and it stands unmodified. This table exists so a reader
    can see whether the decay behind a given S is a STEP (shells cluster but are not ordered
    among themselves) or a GRADIENT (similarity falls steadily with distance). It decides
    nothing and no claim rests on it.

    Returns {arm: {separation: mean_similarity}}, averaged first within a seed's own sim dict
    (a seed may report several shell pairs at the same separation) and then across seeds.
    """
    table = {}
    for arm, by_seed in by_arm.items():
        by_sep = defaultdict(list)
        for record in by_seed.values():
            per_seed_by_sep = defaultdict(list)
            for key, value in record["sim"].items():
                d1, d2 = key.split("_")
                sep = abs(int(d2) - int(d1))
                per_seed_by_sep[sep].append(value)
            for sep, values in per_seed_by_sep.items():
                by_sep[sep].append(st.mean(values))
        table[arm] = {sep: st.mean(values) for sep, values in sorted(by_sep.items())}
    return table


def main() -> None:
    by_arm = load(HERE / "outputs")
    if not by_arm:
        print("no records in outputs/; run run.py first")
        return

    print("EXP-054: is the concept sequence-blind?")
    print("The statistic trains nothing. A random encoder may score highest - see spec 3.\n")

    print(f"{'arm':>5} {'epochs':>7} {'S mean':>9} {'S sd':>8} {'Scr mean':>9} {'Scr sd':>8} "
          f"{'level':>8} {'move-acc':>9} {'policy':>8}")
    for arm in ORDER:
        if arm not in by_arm:
            continue
        records = by_arm[arm]
        vals = [records[s]["S"] for s in sorted(records)]
        cross_vals = [s_cross(records[s]) for s in sorted(records)]
        level_vals = [level(records[s]) for s in sorted(records)]
        acc = MOVE_ACCURACY[arm]
        pol = arm_policy_mean(records)
        acc_s = "-" if acc is None else f"{acc:.3f}"
        print(f"{arm:>5} {next(iter(records.values()))['epochs']:>7} "
              f"{st.mean(vals):>9.4f} {st.stdev(vals) if len(vals) > 1 else 0.0:>8.4f} "
              f"{st.mean(cross_vals):>9.4f} {st.stdev(cross_vals) if len(cross_vals) > 1 else 0.0:>8.4f} "
              f"{st.mean(level_vals):>8.4f} "
              f"{acc_s:>9} {pol:>8.4f}")
    print("  Scr = S_cross, the same fit restricted to |d1-d2| >= 1 (excludes within-shell")
    print("  clustering). level = collapse control, the grand mean of all stored similarities.")
    print("  A fall in S accompanied by a fall in level is consistent with representational")
    print("  collapse (everything converging toward a constant), not with selective loss of")
    print("  distance structure. See the spec amendment.")

    print("\nDESCRIPTIVE - mean similarity by shell separation |dd| (averaged over seeds).")
    print("Decides nothing: shows whether S's decay is a step (clustering) or a gradient")
    print("(genuine distance ordering). |dd|=0 is within-shell tightness, not cross-shell decay.")
    sep_table = separation_table(by_arm)
    all_seps = sorted({sep for row in sep_table.values() for sep in row})
    header = "  " + f"{'arm':>5} " + " ".join(f"|dd|={s:>2}" for s in all_seps)
    print(header)
    for arm in ORDER:
        if arm not in sep_table:
            continue
        cells = " ".join(
            f"{sep_table[arm][s]:>7.4f}" if s in sep_table[arm] else f"{'-':>7}"
            for s in all_seps
        )
        print(f"  {arm:>5} {cells}")

    print("\nCLAIM 1 PRIMARY - does S fall with pretraining? "
          "CONFIRMED if it decreases in >= 2 of 3 at p <= 0.05.")
    decreases = 0
    for a, b in ADJACENT:
        if a not in by_arm or b not in by_arm:
            continue
        seeds = sorted(set(by_arm[a]) & set(by_arm[b]))
        diffs = [by_arm[b][s]["S"] - by_arm[a][s]["S"] for s in seeds]
        p = permutation_p(diffs)
        fell = st.mean(diffs) < 0 and p <= ALPHA
        decreases += int(fell)
        print(f"  {a} -> {b}   delta {st.mean(diffs):+.4f}   p {p:.4f}   "
              f"{'DECREASE' if fell else 'not significant'}")
    print(f"  => {decreases}/3 significant decreases -> "
          f"{'CONFIRMED' if decreases >= 2 else 'NOT CONFIRMED'}")

    print("\nCLAIM 3 FLOOR - E0 against E10, paired.")
    if "E0" in by_arm and "E10" in by_arm:
        seeds = sorted(set(by_arm["E0"]) & set(by_arm["E10"]))
        diffs = [by_arm["E0"][s]["S"] - by_arm["E10"][s]["S"] for s in seeds]
        print(f"  E0 minus E10   delta {st.mean(diffs):+.4f}   p {permutation_p(diffs):.4f}")
        if st.mean(diffs) > 0:
            print("  E0 is HIGHER: pretraining degrades sequence structure from the first "
                  "epoch, and no amount of it is protective.")

    print("\nCLAIM 4 DISQUALIFIER - does S invert between within-arm and between-arm?")
    within = {}
    for arm in ORDER:
        if arm == "E0" or arm not in by_arm:
            continue   # E0 has zero policy variance across seeds, so no correlation exists
        seeds = sorted(by_arm[arm])
        s_vals = [by_arm[arm][s]["S"] for s in seeds]
        pol = [by_arm[arm][s]["policy_success"] for s in seeds]
        if len(set(pol)) < 2:
            continue   # arm-level policy is a constant here; see the note below
        within[arm] = spearman(s_vals, pol)
    arms_present = [a for a in ORDER if a in by_arm and a != "E0"]
    between = spearman(
        [st.mean([by_arm[a][s]["S"] for s in by_arm[a]]) for a in arms_present],
        [st.mean([by_arm[a][s]["policy_success"] for s in by_arm[a]]) for a in arms_present],
    )
    if not within:
        print("  NO within-arm correlation could be computed: every arm's policy is constant "
              "across seeds. That should not happen now that records carry per-seed policy, so "
              "treat it as a bug in the policy lookup rather than as a result.")
    else:
        print("  DESCRIPTIVE, per arm - does THIS arm alone oppose the between-arm sign? "
              "(does not change the verdict below)")
        for arm in arms_present:
            if arm not in within:
                continue
            corr = within[arm]
            flag = ("WOULD TRIP the disqualifier alone" if arm_trips_alone(corr, between)
                    else "agrees with (or is neutral to) the between-arm sign")
            print(f"    {arm}: within-arm {corr:+.3f} vs between-arm {between:+.3f} -> {flag}")
        print("  " + claim4_verdict(within, between))


if __name__ == "__main__":
    main()
