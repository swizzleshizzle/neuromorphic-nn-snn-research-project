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

HERE = Path(__file__).resolve().parent

_RUN_PATH = HERE / "run.py"
_spec = importlib.util.spec_from_file_location("exp055_run", _RUN_PATH)
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

ALPHA = 0.05
BAR = 0.05
BONFERRONI = 0.0083          # six pre-registered comparisons
EPOCH_ARMS = _run.EPOCH_ARMS
ANCHORS = _run.ANCHORS
ZERO_EPOCH_MEAN = _run.ZERO_EPOCH_MEAN
ADJACENT = ((1, 2), (2, 3), (3, 5), (5, 10))


def permutation_p(diffs) -> float:
    """Exact two-sided paired permutation over all 2**n sign flips. No scipy in the venv."""
    n, obs = len(diffs), abs(sum(diffs))
    return sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12) / 2 ** n


def describe_contrast(delta: float, p: float, bar: float = BAR, alpha: float = ALPHA) -> str:
    """The Claim 1 wording rule, as a function.

    A non-significant difference is NOT evidence of equality. When the contrast misses
    significance AND the delta is under the bar, the honest output is a BOUND on the effect at
    the power available. When it misses significance with a delta OVER the bar, nothing is
    bounded - it is simply unresolved, and calling that a bound would invert the finding.
    """
    if abs(delta) >= bar and p <= alpha:
        return (f"CONFIRMED: delta {delta:+.4f} at p {p:.4f}, clearing the +{bar} bar.")
    if p > alpha and abs(delta) < bar:
        return (f"indistinguishable at n=12: delta {delta:+.4f}, p {p:.4f}. This BOUNDS the "
                f"effect below +{bar} at this power. It is NOT evidence that the two arms "
                f"behave identically.")
    if p > alpha:
        return (f"unresolved: delta {delta:+.4f} exceeds the +{bar} bar but p {p:.4f} misses "
                f"significance. Nothing is confirmed and nothing is settled; n=12 cannot "
                f"resolve it.")
    return (f"significant but sub-bar: delta {delta:+.4f} at p {p:.4f}. Real, and smaller than "
            f"the +{bar} the claim required.")


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

    Only EXP-055's own encoders have these. The anchors' S values live in EXP-054's outputs and
    are reported in that experiment's RESULTS rather than recomputed here.
    """
    out = {}
    for p in Path(directory).glob("exp055_S_e*.json"):
        r = json.loads(p.read_text())
        out.setdefault(int(r["epochs"]), {})[int(r["seed"])] = r
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
    print("Two rules here are conditions, not conventions: a non-significant contrast is a")
    print("BOUND and never an equivalence, and no shape word is emitted without significance.\n")

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
        print(f"  {describe_contrast(st.mean(diffs), p)}")
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
    for a, b in ADJACENT:
        if a not in by_epoch or b not in by_epoch:
            missing = [x for x in (a, b) if x not in by_epoch]
            note = " (missing anchor)" if any(x in missing_anchors for x in missing) else ""
            print(f"  e{a} -> e{b}: skipped, missing epoch(s) {missing}{note}")
            continue
        seeds, diffs = paired(by_epoch[a], by_epoch[b])
        p = permutation_p(diffs)
        word = shape_word(st.mean(diffs), p)
        flag = "" if p <= BONFERRONI else f"   (above Bonferroni {BONFERRONI})"
        print(f"  e{a} -> e{b}: {word}   delta {st.mean(diffs):+.4f}   p {p:.4f}{flag}")

    print("\nCLAIM 4 - does S saturate at the same point as policy?")
    s_by_epoch = load_s(HERE / "outputs")
    if not s_by_epoch:
        print("  no S records; run measure_s.py. Claim 4 cannot be evaluated without them.")
    else:
        print(f"  {'epochs':>7} {'S':>9} {'S_cross':>9} {'level':>9}")
        for e in sorted(s_by_epoch):
            rs = s_by_epoch[e]
            print(f"  {e:>7} {st.mean([r['S'] for r in rs.values()]):>9.4f} "
                  f"{st.mean([r['S_cross'] for r in rs.values()]):>9.4f} "
                  f"{st.mean([r['level'] for r in rs.values()]):>9.4f}")
        print("  adjacent contrasts on S, under the SAME gate as Claim 3:")
        for a, b in ADJACENT:
            if a not in s_by_epoch or b not in s_by_epoch:
                continue
            seeds = sorted(set(s_by_epoch[a]) & set(s_by_epoch[b]))
            diffs = [s_by_epoch[b][x]["S"] - s_by_epoch[a][x]["S"] for x in seeds]
            p = permutation_p(diffs)
            print(f"    e{a} -> e{b}: {shape_word(st.mean(diffs), p)}   "
                  f"delta {st.mean(diffs):+.4f}   p {p:.4f}")
        print("  A DISSOCIATION - S turning over at a different epoch than policy - separates")
        print("  'the encoder has the structure' from 'the policy can use it'.")

    print(f"\nMULTIPLICITY: six pre-registered comparisons, Bonferroni {BONFERRONI}. Claim 1")
    print(f"keeps its {ALPHA} as the single primary; the rest are read against Bonferroni")
    print("whenever one is used to name a shape.")


if __name__ == "__main__":
    main()
