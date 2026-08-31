"""EXP-055 aggregator: apply the pre-registered rules to the records on disk.

Thresholds committed in the spec before any number existed.

> TWO RULES HERE ARE CONDITIONS, NOT CONVENTIONS, and both exist because this project broke
> them. EXP-052's aggregator named a monotone shape from four means, three of which were
> indistinguishable at p 0.49 to 0.84. EXP-054's aggregator then printed a verdict derived from
> a rank correlation over four means whose spread was 0.08x their own within-arm sd, three days
> after the rule against exactly that was adopted. Prose did not stop either one.

Usage:
    .venv/bin/python experiments/055_pretraining_left_edge/aggregate.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import statistics as st
from pathlib import Path

from neuromorphic.analysis.sequence_sensitivity import (
    sensitivity_from_similarity,
    sim_from_record,
)

HERE = Path(__file__).resolve().parent

_RUN_PATH = HERE / "run.py"
_spec = importlib.util.spec_from_file_location("exp055_run", _RUN_PATH)
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

ALPHA = 0.05
BAR = 0.05
# Two families, counted separately (see spec Multiplicity). The e1 floor (Claim 2) is
# descriptive - EXP-036's zero-epoch arm has no variance, so there is no p-value - and belongs
# to neither count.
BONFERRONI_POLICY = 0.01     # 0.05 / 5: Claim 1 (primary) + the four Claim 3 adjacent contrasts
BONFERRONI_S = 0.0125        # 0.05 / 4: the four Claim 4 adjacent contrasts on S, a separate
                             # family because they test a different statistic
T95_DF11 = 2.201             # two-sided 95% t multiplier at df=11 (n=12 paired)
EPOCH_ARMS = _run.EPOCH_ARMS
ANCHORS = _run.ANCHORS
ZERO_EPOCH_MEAN = _run.ZERO_EPOCH_MEAN
ADJACENT = ((1, 2), (2, 3), (3, 5), (5, 10))


def permutation_p(diffs) -> float:
    """Exact two-sided paired permutation over all 2**n sign flips. No scipy in the venv."""
    n, obs = len(diffs), abs(sum(diffs))
    return sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12) / 2 ** n


def describe_contrast(diffs, p: float, bar: float = BAR, alpha: float = ALPHA) -> str:
    """The Claim 1 wording rule, as a function.

    Takes the per-seed differences, not just their mean, because a non-significant result
    needs an interval reported, not a bound claimed. The paired-difference sd measured on this
    project's real arms is 0.10-0.14 (e10 vs e20 sd 0.137, se 0.040; e10 vs e40 sd 0.102, se
    0.029), which at n=12 makes a non-significant contrast consistent with true effects up to
    roughly +0.09 - nearly twice the +0.05 bar. Printing "bounds the effect below +0.05" in
    that regime is false, not conservative, so this function reports the approximate 95%
    interval instead and says plainly that n=12 does not resolve the question.

    Claim 1 is directional (`e10 - e1 >= +0.05`), so confirmation requires `delta >= bar`, not
    `abs(delta) >= bar`. A significant delta at or beyond the bar in the OPPOSITE direction is a
    real finding but is not a confirmation of this claim, and must never be reported as one.

    The interval sentence is COMPUTED from `lo`/`hi`, not asserted. It used to say "the interval
    includes effects larger than the bar" unconditionally, which happens to be true for every
    paired-difference sd this project has measured and is therefore invisible until it is not.
    That is the same defect as the bound wording it replaced: a sentence about the data that
    never consults the data. If the interval really does stay inside the bar, n=12 HAS resolved
    the question at that size, and saying otherwise understates the result.
    """
    n = len(diffs)
    delta = st.mean(diffs)
    se = st.stdev(diffs) / (n ** 0.5) if n > 1 else 0.0
    lo, hi = delta - T95_DF11 * se, delta + T95_DF11 * se

    if p <= alpha:
        if delta >= bar:
            return f"CONFIRMED: delta {delta:+.4f} at p {p:.4f}, clearing the +{bar} bar."
        if delta <= -bar:
            return (f"SIGNIFICANT, OPPOSITE DIRECTION: delta {delta:+.4f} at p {p:.4f}. This "
                    f"is a real, significant result, but it runs the opposite way to the "
                    f"claim's +{bar} bar - it does not confirm the claim.")
        return (f"significant but sub-bar: delta {delta:+.4f} at p {p:.4f}. Real, and smaller "
                f"than the +{bar} the claim required.")

    if abs(delta) < bar:
        head = (f"indistinguishable at n={n}: delta {delta:+.4f}, p {p:.4f}, approx 95% "
                f"interval [{lo:+.4f}, {hi:+.4f}]. ")
        if hi >= bar or lo <= -bar:
            return head + (f"n={n} does not resolve this: the interval reaches the +{bar} bar. "
                           f"This is NOT evidence that the two arms behave identically.")
        return head + (f"the interval stays inside the +{bar} bar in both directions, so this "
                       f"contrast IS resolved against an effect that large. It is still NOT "
                       f"evidence that the two arms behave identically: a real effect smaller "
                       f"than the bar remains consistent with it.")
    return (f"unresolved: delta {delta:+.4f} exceeds the +{bar} bar but p {p:.4f} misses "
            f"significance. Nothing is confirmed and nothing is settled; n={n} cannot "
            f"resolve it.")


def shape_word(delta: float, p: float, alpha: float = ALPHA) -> str:
    """The Claim 3 gate. Returns a direction word ONLY for a significant contrast.

    This is the condition that EXP-052 and EXP-054 each needed and neither had.
    """
    if p > alpha:
        return "indistinguishable"
    return "rises" if delta > 0 else "falls"


def load(directory: Path, tag: str) -> dict:
    out = {}
    for p in Path(directory).glob("*.json"):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and r.get("tag") == tag and r.get("depth") == 6:
            out[int(r["seed"])] = r
    return out


def load_s(directory: Path) -> dict:
    """S records written by measure_s.py, keyed by epochs then seed.

    Epoch 10 is populated from EXP-054's own records instead of EXP-055's outputs: EXP-055
    never pretrains a 10-epoch encoder (10 is an anchor, reused from EXP-052), so there is no
    `exp055_S_e10_*.json` and Claim 4's e5 -> e10 contrast - the one comparison that actually
    answers whether S and policy saturate together - would otherwise hit a silent gap. EXP-054
    already measured S, with the same statistic and the same twelve seeds, on the identical
    `exp052_encoder_e10_s*.pt` encoders (`exp054_E10_s{0..11}.json`), so loading those keeps
    the e5 -> e10 contrast genuinely paired by seed without re-measuring anything.
    """
    out = {}
    for p in Path(directory).glob("exp055_S_e*.json"):
        r = json.loads(p.read_text())
        out.setdefault(int(r["epochs"]), {})[int(r["seed"])] = r

    exp054_dir = HERE.parent / "054_sequence_blindness" / "outputs"
    e10 = {}
    for p in Path(exp054_dir).glob("exp054_E10_s*.json"):
        r = json.loads(p.read_text())
        if int(r.get("epochs", -1)) != 10:
            continue
        sim = sim_from_record(r["sim"])
        e10[int(r["seed"])] = {
            "epochs": 10,
            "seed": int(r["seed"]),
            "S": r["S"],
            # EXP-054's own record does not store S_cross or level; derive them exactly as
            # EXP-054's aggregator does, from the stored `sim` dict, rather than re-measuring.
            "S_cross": sensitivity_from_similarity(sim, min_separation=1),
            "level": st.mean(r["sim"].values()),
        }
    if e10:
        out[10] = e10
    return out


def arm_records() -> tuple[dict, list[int]]:
    """Every epoch level that has records, keyed by epoch. Includes the anchors.

    Also returns the list of anchor epochs whose directory/tag produced no records at all, so a
    missing anchor can be reported explicitly rather than just vanishing from the table. run.py's
    pre-flight only checks that EXP-055's own encoders exist - it never checks the anchors, since
    they are printed there, not read. This function DOES read them, so a missing one must degrade
    to a clear message rather than a silent gap or a contrast computed against nothing.
    """
    by_epoch = {}
    for e in EPOCH_ARMS:
        recs = load(HERE / "outputs", _run.tag_for(e))
        if recs:
            by_epoch[e] = recs
    missing_anchors = []
    for e, (directory, tag, _mean) in ANCHORS.items():
        recs = load(directory, tag)
        if recs:
            by_epoch[e] = {s: r for s, r in recs.items() if s < 12}
        else:
            missing_anchors.append(e)
    return by_epoch, sorted(missing_anchors)


def paired(a: dict, b: dict, field: str = "success_rate"):
    seeds = sorted(set(a) & set(b))
    return seeds, [b[s][field] - a[s][field] for s in seeds]


def main() -> None:
    by_epoch, missing_anchors = arm_records()
    if missing_anchors:
        print("WARNING: anchor record(s) not found on disk, skipped rather than treated as")
        print("missing-equals-zero. Any claim that needs one of these will report itself as")
        print("unavailable below, not silently compare against nothing:")
        for e in missing_anchors:
            directory, tag, mean = ANCHORS[e]
            print(f"  epoch {e}: expected tag '{tag}' in {directory} (published mean {mean}),"
                  " found 0 matching records")
        print()

    if not by_epoch:
        print("no records; run pretrain_left_edge.py then run.py first")
        return

    print("EXP-055: the 0-to-10 pretraining window")
    print("Two rules here are conditions, not conventions: a non-significant contrast is")
    print("reported as an interval and never as an equivalence, and no shape word is emitted")
    print("without significance.\n")

    print(f"{'epochs':>7} {'n':>3} {'policy':>9} {'sd':>8}")
    print(f"{0:>7} {12:>3} {ZERO_EPOCH_MEAN:>9.4f} {0.0:>8.4f}   (EXP-036, no variance)")
    for e in sorted(by_epoch):
        v = [by_epoch[e][s]["success_rate"] for s in sorted(by_epoch[e])]
        print(f"{e:>7} {len(v):>3} {st.mean(v):>9.4f} "
              f"{st.stdev(v) if len(v) > 1 else 0.0:>8.4f}")

    print("\nCLAIM 1 PRIMARY - is there a real ramp between 1 and 10 epochs?")
    if 1 in by_epoch and 10 in by_epoch:
        seeds, diffs = paired(by_epoch[1], by_epoch[10])
        p = permutation_p(diffs)
        print(f"  e10 minus e1, n {len(seeds)}")
        print(f"  {describe_contrast(diffs, p)}")
    elif 10 in missing_anchors:
        print("  requires the e10 anchor, which is missing on disk (see WARNING above).")
    else:
        print("  requires both the e1 arm and the e10 anchor.")

    print("\nCLAIM 2 THE FLOOR - e1 against 0 epochs (EXP-036, 0.0000 on all twelve seeds).")
    if 1 in by_epoch:
        v = [by_epoch[1][s]["success_rate"] for s in sorted(by_epoch[1])]
        print(f"  e1 mean {st.mean(v):.4f} against {ZERO_EPOCH_MEAN:.4f}. Descriptive: the "
              "zero-epoch arm has no variance, so there is no paired test to run.")
        print("  If this lands near 0.20, pretraining's contribution is almost entirely "
              "escaping random init.")

    print("\nCLAIM 3 SHAPE - adjacent contrasts. A shape word appears ONLY where significant.")
    print("  POWER, stated before the numbers: at a paired-difference sd of about 0.11, n=12")
    print("  gives roughly 28% power for Claim 1's own +0.05 effect, and roughly 10% power at")
    print("  Bonferroni for an adjacent step of that size. Four 'indistinguishable' verdicts")
    print("  below are likely whatever the true shape is, and must NOT be read as flatness.")
    print("  The experiment is well powered for the case it cares about most instead: if e1 is")
    print("  near zero, e10 - e1 is about +0.20, where power is near 1.00.")
    for a, b in ADJACENT:
        if a not in by_epoch or b not in by_epoch:
            missing = [x for x in (a, b) if x not in by_epoch]
            note = " (missing anchor)" if any(x in missing_anchors for x in missing) else ""
            print(f"  e{a} -> e{b}: skipped, missing epoch(s) {missing}{note}")
            continue
        seeds, diffs = paired(by_epoch[a], by_epoch[b])
        p = permutation_p(diffs)
        word = shape_word(st.mean(diffs), p, alpha=BONFERRONI_POLICY)
        flag = "" if p <= BONFERRONI_POLICY else f"   (above Bonferroni {BONFERRONI_POLICY})"
        print(f"  e{a} -> e{b}: {word}   delta {st.mean(diffs):+.4f}   p {p:.4f}{flag}")

    print("\nCLAIM 4 - does S saturate at the same point as policy?")
    s_by_epoch = load_s(HERE / "outputs")
    if not s_by_epoch:
        print("  no S records; run measure_s.py. Claim 4 cannot be evaluated without them.")
    else:
        print("  S is near-deterministic per seed: a tiny paired delta can still be")
        print("  significant. 'sd' below is the within-arm spread across seeds, so a")
        print("  trivially small but significant delta in the contrasts reads as trivial.")
        print(f"  {'epochs':>7} {'S':>9} {'sd':>8} {'S_cross':>9} {'level':>9}")
        for e in sorted(s_by_epoch):
            rs = s_by_epoch[e]
            s_vals = [r["S"] for r in rs.values()]
            print(f"  {e:>7} {st.mean(s_vals):>9.4f} "
                  f"{st.stdev(s_vals) if len(s_vals) > 1 else 0.0:>8.4f} "
                  f"{st.mean([r['S_cross'] for r in rs.values()]):>9.4f} "
                  f"{st.mean([r['level'] for r in rs.values()]):>9.4f}")
        print(f"  adjacent contrasts on S, own family, Bonferroni {BONFERRONI_S} (0.05 / 4):")
        for a, b in ADJACENT:
            if a not in s_by_epoch or b not in s_by_epoch:
                continue
            seeds = sorted(set(s_by_epoch[a]) & set(s_by_epoch[b]))
            diffs = [s_by_epoch[b][x]["S"] - s_by_epoch[a][x]["S"] for x in seeds]
            p = permutation_p(diffs)
            word = shape_word(st.mean(diffs), p, alpha=BONFERRONI_S)
            print(f"    e{a} -> e{b}: {word}   delta {st.mean(diffs):+.4f}   p {p:.4f}")
        print("  A DISSOCIATION - S turning over at a different epoch than policy - separates")
        print("  'the encoder has the structure' from 'the policy can use it'.")

    print(f"\nMULTIPLICITY: two families, counted separately. The e1 floor (Claim 2) is")
    print(f"descriptive and has no p-value, so it belongs to neither. POLICY family: five")
    print(f"comparisons (Claim 1 + four Claim 3 adjacent contrasts), Bonferroni")
    print(f"{BONFERRONI_POLICY} (0.05 / 5). Claim 1 keeps its own {ALPHA} as the single")
    print(f"primary; the four Claim 3 contrasts are read against Bonferroni whenever one is")
    print(f"used to name a shape. S family: four comparisons (Claim 4 adjacent contrasts),")
    print(f"its own Bonferroni {BONFERRONI_S} (0.05 / 4).")


if __name__ == "__main__":
    main()
