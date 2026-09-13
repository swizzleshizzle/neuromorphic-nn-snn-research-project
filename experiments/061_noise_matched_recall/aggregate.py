"""EXP-061 aggregator: the pre-registered rules from the spec, applied to the records on disk.

Spec: `docs/superpowers/specs/2026-09-13-exp061-noise-matched-recall-design.md`.

> WRITTEN BEFORE ANY EXP-061 NUMBER WAS SEEN. Completion was confirmed from counts alone -
> 24 records, 24 head checkpoints, zero python processes - WITHOUT reading the run log, which
> prints one success rate per line. EXP-060's aggregator could not claim this: verifying that
> run meant reading its log tail, and roughly twelve cell values had been seen beforehand. The
> counts-only check is the cheap fix and it is the practice to keep.

> CLAIM 3 IS A CONDITION AND IS CHECKED FIRST. The manipulation is a substitution INSIDE the
> recall block. If that block is negligible beside the concept block, the policy head barely
> sees it and any result here is vacuous rather than informative.
>
> A MISSING RATIO IS A FAILURE, NEVER A PASS. `None` coerced to 0.0 would sail under a
> "must be at least 0.05" floor while measuring nothing - the mirror of EXP-058's gate that
> could not pass, and the same trap EXP-059's aggregator had to defend against.

> THE PRIMARY'S INTERESTING OUTCOME IS A NULL, AND THIS FILE MUST SAY SO WHEN IT HAPPENS.
> H1 ("the recall is noise on the policy path") is supported by M ~= N. A null at n=24 is a
> BOUND, not evidence, and the spec fixed that wording before dispatch precisely so it could
> not be upgraded afterwards into "H1 confirmed".

NOT IMPORTED FROM EXP-055/056/057: their `describe_contrast` hardcodes `T95_DF11 = 2.201`, the
multiplier at df=11 for n=12. This is n=24, where it is 2.069, so reusing that code would report
every interval about 6% too wide.

Usage:
    .venv/bin/python -u experiments/061_noise_matched_recall/aggregate.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

_spec = importlib.util.spec_from_file_location("exp061_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

SEEDS = _run.SEEDS
TAG_N = _run.TAG
BAR = _run.BAR                                   # 0.05
GATE_MIN = _run.GATE_MIN_NORM_RATIO              # 0.05
ALPHA = 0.05
N_CONTRASTS = 2                                  # Claims 1 and 2. Claim 3 is a condition.
BONFERRONI = ALPHA / N_CONTRASTS                 # 0.025

EXP059 = REPO / "experiments/059_memory_depth5/outputs"
TAG_A, TAG_M = _run.EXP059_A, _run.EXP059_M
EXP059_M_MINUS_A, EXP059_M_MINUS_A_P = -0.0954, 0.0056

EXACT_MAX_N = 20
SAMPLED_DRAWS = 200_000
SAMPLED_SEED = 20260913

T95 = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
       20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069}


def t95_for(n: int) -> float:
    df = n - 1
    if df in T95:
        return T95[df]
    if df > 23:
        return 1.96
    raise ValueError(f"n={n} is too small to report an interval for")


def permutation_p(diffs) -> tuple[float, str]:
    """Two-sided paired permutation p, plus the NAME of the method - the spec requires the
    method be printed, because an exact proportion and a Monte Carlo estimate are not
    interchangeable."""
    n, obs = len(diffs), abs(sum(diffs))
    if n <= EXACT_MAX_N:
        hits = sum(1 for s in itertools.product((1, -1), repeat=n)
                   if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12)
        return hits / 2 ** n, f"exact over all 2**{n} = {2 ** n:,} sign flips"
    rng = random.Random(SAMPLED_SEED)
    hits = 0
    for _ in range(SAMPLED_DRAWS):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total) >= obs - 1e-12:
            hits += 1
    # Add-one: plain hits/draws can print 0.0000 and claim an exactness sampling cannot support.
    return (hits + 1) / (SAMPLED_DRAWS + 1), (
        f"sampled, {SAMPLED_DRAWS:,} draws at seed {SAMPLED_SEED} (exhaustive would be "
        f"2**{n} = {2 ** n:,}); add-one estimator, floor {1 / (SAMPLED_DRAWS + 1):.1e}")


def ci95(diffs) -> tuple[float, float]:
    n = len(diffs)
    d = st.mean(diffs)
    se = st.stdev(diffs) / (n ** 0.5) if n > 1 else 0.0
    t = t95_for(n)
    return d - t * se, d + t * se


def load(directory: Path, tag: str, expect_readout: str | None = None) -> dict:
    """Records by seed. `expect_readout` is checked when given: all three arms share
    `arm="regionalized"` and differ ONLY in the readout, so a tag pointing at the wrong readout
    would contrast an arm against itself with nothing in the numbers looking wrong."""
    out: dict[int, dict] = {}
    for p in sorted(Path(directory).glob("*.json")):
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or r.get("tag") != tag:
            continue
        if expect_readout is not None and r.get("readout") != expect_readout:
            raise SystemExit(f"{p.name}: tag {tag} carries readout {r.get('readout')!r}, "
                             f"expected {expect_readout!r}; refusing.")
        out[int(r["seed"])] = r
    return out


def paired(arm: dict, ctrl: dict, field="success_rate"):
    use = sorted(set(arm) & set(ctrl) & set(SEEDS))
    missing = [s for s in use if arm[s].get(field) is None or ctrl[s].get(field) is None]
    if missing:
        raise SystemExit(f"field {field!r} is None for seeds {missing}; cannot pair on it")
    return use, [arm[s][field] - ctrl[s][field] for s in use]


def gate_reading(records: dict, field: str) -> tuple[float | None, list[int]]:
    """Mean over seeds, plus the seeds where the field is MISSING. Missing is RETURNED, not
    skipped: a `None` dropped from a mean or coerced to 0.0 turns a floor gate into one that
    cannot fail."""
    seeds = sorted(set(records) & set(SEEDS))
    vals = [records[s].get(field) for s in seeds]
    missing = [s for s, v in zip(seeds, vals) if v is None]
    present = [v for v in vals if v is not None]
    return (st.mean(present) if present else None), missing


def check_gate(N: dict) -> tuple[bool, list[str]]:
    """Claim 3: arm N's recall block must be non-negligible beside its concept block."""
    lines: list[str] = []
    ratio, missing = gate_reading(N, "recall_concept_norm_ratio")
    if missing or ratio is None:
        lines.append(f"  arm N recall/concept norm ratio: MISSING for seeds "
                     f"{missing or 'all'}. Treated as a FAILURE - an absent instrument "
                     "measures nothing, and a None coerced to 0.0 would pass no floor anyway.")
        return False, lines
    passed = ratio >= GATE_MIN
    lines.append(f"  arm N recall/concept norm ratio: {ratio:.4f}  must be >= {GATE_MIN}  "
                 f"{'PASS' if passed else 'FAIL'}")
    per = sorted(N[s]["recall_concept_norm_ratio"] for s in sorted(N))
    lines.append(f"  across seeds: min {per[0]:.4f}  median {st.median(per):.4f}  "
                 f"max {per[-1]:.4f}")
    below = [s for s in sorted(N) if N[s]["recall_concept_norm_ratio"] < GATE_MIN]
    lines.append(f"  seeds individually below the floor: {below or 'none'}")
    return passed, lines


def _sig(p: float) -> str:
    if p <= BONFERRONI:
        return f"clears Bonferroni {BONFERRONI:.4f}"
    if p <= ALPHA:
        return f"clears alpha {ALPHA} but NOT Bonferroni {BONFERRONI:.4f}"
    return f"above alpha {ALPHA}"


def claim1_verdict(diffs, p: float) -> tuple[str, str]:
    """Claim 1, PRIMARY: `M` minus `N`. NOT directional - H2 predicts N above M, H1 a null."""
    n = len(diffs)
    d = st.mean(diffs)
    lo, hi = ci95(diffs)
    iv = f"approx 95% interval [{lo:+.4f}, {hi:+.4f}]"
    if p <= ALPHA and d <= -BAR:
        return "H2", (f"H2 SUPPORTED, and it belongs in the headline: delta {d:+.4f} at "
                      f"p {p:.4f}, clearing the {-BAR} bar downward. NOISE BEATS REAL MEMORY, so "
                      f"the stored content is ACTIVELY MISLEADING rather than merely useless. "
                      f"{_sig(p)}.")
    if p <= ALPHA and d >= BAR:
        return "THIRD", (f"A THIRD OUTCOME NEITHER HYPOTHESIS PREDICTED: delta {d:+.4f} at "
                         f"p {p:.4f}. Real memory BEATS matched noise while still losing to the "
                         f"amnesic transform, so the stored content is worth something after "
                         f"all. This needs its own explanation, not a fit to H1 or H2. "
                         f"{_sig(p)}.")
    if p <= ALPHA:
        way = "toward H2" if d < 0 else "against H2"
        return "SUB_BAR", (f"significant but sub-bar, leaning {way}: delta {d:+.4f} at "
                           f"p {p:.4f}, {iv}. Real, smaller than the {BAR} the claim required, "
                           f"and so neither a confirmation nor a null. {_sig(p)}.")
    head = (f"INDISTINGUISHABLE at n={n}: delta {d:+.4f}, p {p:.4f}, {iv}. **A BOUND, "
            f"CONSISTENT WITH H1 AND NOT PROOF OF IT.** The spec pre-registered this reading "
            f"before dispatch: H1 predicts exactly this, which is why a null here cannot "
            f"confirm it. ")
    if hi >= BAR or lo <= -BAR:
        head += f"The interval still reaches the {BAR} bar, so n={n} does not resolve it."
    else:
        head += f"The interval stays inside the {BAR} bar, so n={n} DOES bound the effect."
    return "NULL", head


def claim2_verdict(diffs, p: float) -> tuple[str, str]:
    """Claim 2: `N` minus `A`. Directional - under BOTH hypotheses, destroying the recall
    block's information should cost roughly what EXP-059 measured for memory."""
    n = len(diffs)
    d = st.mean(diffs)
    lo, hi = ci95(diffs)
    iv = f"approx 95% interval [{lo:+.4f}, {hi:+.4f}]"
    if p <= ALPHA and d <= -BAR:
        return "CONFIRMED", (f"CONFIRMED: delta {d:+.4f} at p {p:.4f}, clearing the {-BAR} bar "
                             f"downward. Replacing the recall block's information costs about "
                             f"what real memory cost in EXP-059 ({EXP059_M_MINUS_A:+.4f}), so "
                             f"the block's value is what it says about the CURRENT state. "
                             f"{_sig(p)}.")
    if p <= ALPHA and d >= BAR:
        return "REVERSED", (f"SIGNIFICANT REVERSAL: delta {d:+.4f} at p {p:.4f}. Noise in the "
                            f"recall block BEAT the amnesic transform, which no hypothesis here "
                            f"predicted. {_sig(p)}.")
    if p <= ALPHA:
        return "SUB_BAR", (f"significant but sub-bar: delta {d:+.4f} at p {p:.4f}, {iv}. "
                           f"{_sig(p)}.")
    return "NULL", (f"NOT CONFIRMED at n={n}: delta {d:+.4f}, p {p:.4f}, {iv}. A bound. If noise "
                    f"costs much less than memory did ({EXP059_M_MINUS_A:+.4f}), then `M` is not "
                    f"simply information-destroying and H2 gains support from a second "
                    f"direction.")


def joint_reading(c1: str, c2: str) -> str:
    """Pre-registered before any number existed."""
    if c1 == "NULL" and c2 == "CONFIRMED":
        return ("THE CLEANEST OUTCOME THIS DESIGN CAN PRODUCE: noise is indistinguishable from "
                "real memory, and both cost about 0.1 against the amnesic transform. That is "
                "consistent with H1 - the recall block's value is its memory-free transform of "
                "the CURRENT state, and stored content destroys that as thoroughly as noise "
                "does. Still a bound on the primary, so H1 is supported, not confirmed.")
    if c1 == "H2" and c2 == "CONFIRMED":
        return ("Noise beats real memory AND both lose to amnesic. The stored content is worse "
                "than nothing: actively misleading, not merely uninformative.")
    if c1 == "NULL" and c2 == "NULL":
        return ("Neither contrast resolved. Nothing here distinguishes the hypotheses, and the "
                "honest summary is that the experiment did not discriminate.")
    return (f"Claim 1 {c1}, Claim 2 {c2}. Read each verdict on its own terms above; this "
            "combination was not one of the anticipated shapes.")


def report(label: str, arm: dict, ctrl: dict, field="success_rate"):
    seeds, diffs = paired(arm, ctrl, field)
    p, method = permutation_p(diffs)
    w = sum(1 for x in diffs if x > 0)
    l = sum(1 for x in diffs if x < 0)
    print(f"\n{label}  on {field}")
    print(f"  n {len(seeds)}   arm {st.mean([arm[s][field] for s in seeds]):.4f}   "
          f"control {st.mean([ctrl[s][field] for s in seeds]):.4f}")
    print(f"  paired delta {st.mean(diffs):+.4f}   W-L-T {w}-{l}-{len(diffs)-w-l}   p {p:.4f}")
    print(f"  method: {method}")
    return diffs, p


def main() -> None:
    N = load(HERE / "outputs", TAG_N, expect_readout=_run.READOUT)
    A = load(EXP059, TAG_A, expect_readout="memory_amnesic")
    M = load(EXP059, TAG_M, expect_readout="memory")
    if not N:
        raise SystemExit(f"no EXP-061 records with tag {TAG_N} in {HERE / 'outputs'}")
    if not A or not M:
        raise SystemExit(f"EXP-059's arms are missing from {EXP059}; they are REUSED, not re-run")

    print("EXP-061: WHY does the memory read hurt? Matched-magnitude noise in the recall block.")
    print("  H1  the recall is NOISE on the policy path     -> N ~= M")
    print("  H2  the stored content is ACTIVELY MISLEADING  -> N BETTER than M\n")
    print(f"  arm A  {TAG_A:<22} {len(A)}/{len(SEEDS)}  REUSED from EXP-059")
    print(f"  arm M  {TAG_M:<22} {len(M)}/{len(SEEDS)}  REUSED from EXP-059")
    print(f"  arm N  {TAG_N:<22} {len(N)}/{len(SEEDS)}  NEW")
    print("\n  Reuse is legitimate because the instruments added for this experiment were")
    print("  verified numerically INERT for `memory` and `memory_amnesic` - identical to full")
    print("  float repr before and after - and the noise generator is constructed only for")
    print("  `memory_noise`. See tests/training/test_memory_noise_readout.py.")
    print("\n  This aggregator was written BEFORE any EXP-061 number was seen: completion was")
    print("  confirmed from counts alone, without reading the run log.")

    print("\n" + "=" * 78)
    print("CLAIM 3, THE VALIDITY GATE - a CONDITION, checked FIRST.")
    print(f"Is the recall block big enough for a substitution to matter? Floor {GATE_MIN}.")
    print("Calibrated at depth 5 on the real frozen encoder BEFORE the spec was written:")
    print("arm M measured 0.1653 / 0.2138 / 0.3258 / 0.2983 across seeds 0-3.")
    print("=" * 78)
    ok, lines = check_gate(N)
    for line in lines:
        print(line)

    nc, nc_missing = gate_reading(N, "noise_real_cos")
    if nc is not None:
        print(f"\n  sanity, NOT a gate: mean cosine between the real recall and the substituted")
        print(f"  block = {nc:+.5f}. ~0 by construction, so gating on it could not fail. It is")
        print(f"  reported because a LEAK would show up here - mutation testing found the first")
        print(f"  version of this instrument blind to a 50/50 blend.")
        if abs(nc) > 0.05:
            print(f"  WARNING: |{nc:.5f}| > 0.05 suggests the real recall leaked into the "
                  "substitute.")
    elif nc_missing:
        print(f"\n  sanity: noise_real_cos missing for seeds {nc_missing}.")

    if not ok:
        print("\n" + "=" * 78)
        print("EXPERIMENT VOID by its own pre-registered condition. The recall block is too")
        print("small beside the concept block for its substitution to be informative, so no")
        print("claim below may be read. The floor is NOT to be lowered now that numbers exist -")
        print("that is the EXP-058 mistake, and rewriting a gate afterwards costs the method.")
        print("=" * 78)
        return
    print("\n  gate: PASSED. The claims below may be read.")

    print("\n" + "=" * 78)
    print("CLAIM 1, PRIMARY - M minus N. NOT directional. H1 predicts a NULL.")
    print("=" * 78)
    d1, p1 = report("M - N (primary)", M, N)
    c1, w1 = claim1_verdict(d1, p1)
    print(f"  {w1}")

    print("\n" + "=" * 78)
    print("CLAIM 2 - N minus A. Directional, and the contrast with a positive prediction.")
    print("=" * 78)
    d2, p2 = report("N - A", N, A)
    c2, w2 = claim2_verdict(d2, p2)
    print(f"  {w2}")
    print(f"\n  For comparison, EXP-059 measured M - A = {EXP059_M_MINUS_A:+.4f} at "
          f"p {EXP059_M_MINUS_A_P} on the SAME seeds, config and encoder.")
    print("  That is context for the magnitude, not a contrast computed here.")

    print("\n" + "=" * 78)
    print("JOINT READING, pre-registered before any number existed:")
    print("=" * 78)
    print(f"  {joint_reading(c1, c2)}")

    print("\n" + "=" * 78)
    print(f"MULTIPLICITY: {N_CONTRASTS} inferential contrasts, Bonferroni {BONFERRONI:.4f}.")
    print("Claim 3 is a condition with no p-value and belongs to no family.")
    print("POWER, from the spec before dispatch: at the paired sd this project measures,")
    print("se is about 0.020-0.031, so power is ~50-60% at a 0.05 effect and much less below.")
    print("THE INTERESTING HYPOTHESIS PREDICTS A NULL ON THE PRIMARY, which is this design's")
    print("central weakness and was stated in the spec rather than discovered afterwards.")
    print("=" * 78)


if __name__ == "__main__":
    main()
