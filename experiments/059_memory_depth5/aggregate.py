"""EXP-059 aggregator: apply the pre-registered rules to the records on disk.

Thresholds committed in `docs/superpowers/specs/2026-09-09-exp059-memory-depth5-design.md`.
**This file was written while the run was still in its first wave, with ZERO records on disk**,
which is the only condition under which an aggregator can be honestly authored: written
afterwards, every choice of test, threshold and wording is made by someone who already knows
which way it comes out.

> CLAIM 3 IS A CONDITION AND IT IS CHECKED FIRST. The three arms differ at the READ site, so the
> gate measures the read site: `recall_content_cos`, the cosine between the real recall and a
> `W_rec`-zeroed one. Near 1.0 means the attractor's stored content changed nothing and an
> M-versus-A contrast has nothing to measure. If the gate fails, this script prints the diagnosis
> and REFUSES to compute the claims - it does not print them with a warning attached.
>
> WHY THAT MATTERS HERE SPECIFICALLY. EXP-058 died of a gate that could not pass
> (`mean_n_stored > 10`, bounded above by a 7.76-step mean episode). The mirror-image failure is a
> gate that cannot FAIL, and there is a live path to it in this file: `recall_content_cos` is
> `None` in a record whose readout had no real recall to probe, and `None` coerced to 0.0 would
> sail under a "must be below 0.95" ceiling while measuring nothing at all. Missing probe data is
> therefore treated as a gate FAILURE, never as a pass. See `gate_reading`.

NOT IMPORTED FROM EXP-055/056/057: their `describe_contrast` and interval code hardcode
`T95_DF11 = 2.201`, the t multiplier at df=11 for n=12. This experiment is n=24, where the
multiplier is 2.069, so reusing that code would silently report every interval about 6% too wide.
The wording rules are reimplemented here against a df-indexed table for that reason.

Usage:
    .venv/bin/python experiments/059_memory_depth5/aggregate.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

_RUN_PATH = HERE / "run.py"
_spec = importlib.util.spec_from_file_location("exp059_run", _RUN_PATH)
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

# Every threshold comes from the driver, which the spec fixed. Nothing is restated as a literal
# here: a second copy is a second thing to get out of step with the pre-registration.
ARMS = _run.ARMS
SEEDS = _run.SEEDS
BAR = _run.BAR                                    # +/-0.05 on success, Claim 1
REVISIT_BAR = _run.REVISIT_BAR                    # -0.02 on revisit_rate, Claim 2
GATE_MAX_RECALL_COS = _run.GATE_MAX_RECALL_COS    # 0.95, Claim 3
GATE_MAX_UNSHUFFLED = _run.GATE_MAX_UNSHUFFLED    # 0.20, Claim 3
CAPPED_D5_CONTEXT = _run.CAPPED_D5_CONTEXT        # 0.3229 - CONTEXT, not a control

ALPHA = 0.05
N_CONTRASTS = 3                                   # Claims 1, 2, 4. Claim 3 is a condition.
BONFERRONI = ALPHA / N_CONTRASTS                  # 0.0167

EXACT_MAX_N = 20            # spec: exact below this, sampled above. 2**24 is 16.8M flips.
SAMPLED_DRAWS = 200_000     # spec: fixed-seed 200,000-sample permutation
SAMPLED_SEED = 20260909     # fixed so the p-value is reproducible, per the seeding discipline

# Two-sided 95% t multipliers by degrees of freedom. Spelled out because there is no scipy in
# the venv and because a single hardcoded value is exactly the defect described in the header.
T95 = {
    5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069,
}


def t95_for(n: int) -> float:
    """Multiplier at df = n-1, falling back to the normal approximation only for large n."""
    df = n - 1
    if df in T95:
        return T95[df]
    if df > 23:
        return 1.96
    raise ValueError(f"n={n} is too small to report an interval for")


def permutation_p(diffs) -> tuple[float, str]:
    """Two-sided paired permutation p, plus the NAME of the method used.

    The spec requires the method be printed rather than left implicit, because the two are not
    interchangeable: the exact test is a proportion of a complete enumeration, the sampled test
    is an estimate with its own Monte Carlo error.
    """
    n = len(diffs)
    obs = abs(sum(diffs))
    if n <= EXACT_MAX_N:
        hits = sum(1 for signs in itertools.product((1, -1), repeat=n)
                   if abs(sum(x * y for x, y in zip(signs, diffs))) >= obs - 1e-12)
        return hits / 2 ** n, f"exact over all 2**{n} = {2 ** n:,} sign flips"

    rng = random.Random(SAMPLED_SEED)
    hits = 0
    for _ in range(SAMPLED_DRAWS):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total) >= obs - 1e-12:
            hits += 1
    # Add-one estimator. Without it a sampled test can report p = 0.0000, which claims an
    # exactness the sampling cannot support; the floor here is 1/(B+1) = 5.0e-06.
    p = (hits + 1) / (SAMPLED_DRAWS + 1)
    return p, (f"sampled, {SAMPLED_DRAWS:,} draws at seed {SAMPLED_SEED} "
               f"(exhaustive would be 2**{n} = {2 ** n:,}); add-one estimator, floor "
               f"{1 / (SAMPLED_DRAWS + 1):.1e}")


def ci95(diffs) -> tuple[float, float]:
    n = len(diffs)
    delta = st.mean(diffs)
    se = st.stdev(diffs) / (n ** 0.5) if n > 1 else 0.0
    t = t95_for(n)
    return delta - t * se, delta + t * se


def load(directory: Path, tag: str, expected_readout: str | None = None) -> dict:
    """Records for one arm, keyed by seed.

    ``expected_readout`` is checked when given. All three arms share `arm="regionalized"` and
    differ ONLY in `readout`, so a driver edit that pointed two tags at the same readout would
    produce a contrast of an arm against itself and nothing in the numbers would look wrong.
    """
    out: dict[int, dict] = {}
    for p in sorted(Path(directory).glob("*.json")):
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or r.get("tag") != tag:
            continue
        if expected_readout is not None and r.get("readout") != expected_readout:
            raise SystemExit(
                f"{p.name}: tag {tag} carries readout {r.get('readout')!r}, expected "
                f"{expected_readout!r}. The arms are not what the spec describes; refusing."
            )
        out[int(r["seed"])] = r
    return out


def paired(arm: dict, control: dict, field: str) -> tuple[list[int], list[float]]:
    """Per-seed differences, arm minus control, over seeds present in BOTH and in SEEDS."""
    seeds = sorted(set(arm) & set(control) & set(SEEDS))
    missing = [s for s in seeds if arm[s].get(field) is None or control[s].get(field) is None]
    if missing:
        raise SystemExit(f"field {field!r} is None for seeds {missing}; cannot pair on it")
    return seeds, [arm[s][field] - control[s][field] for s in seeds]


def gate_reading(records: dict, field: str) -> tuple[float | None, list[int]]:
    """Mean of ``field`` over seeds, and the seeds where it is MISSING.

    Missing is returned rather than skipped. A `None` silently dropped from a mean, or coerced
    to 0.0, turns a ceiling gate into one that cannot fail - the mirror of the EXP-058 gate that
    could not pass. The caller must treat a non-empty missing list as a gate failure.
    """
    seeds = sorted(set(records) & set(SEEDS))
    values = [records[s].get(field) for s in seeds]
    missing = [s for s, v in zip(seeds, values) if v is None]
    present = [v for v in values if v is not None]
    return (st.mean(present) if present else None), missing


def check_gate(m: dict, s: dict) -> tuple[bool, list[str]]:
    """Claim 3. Returns (passed, lines to print). Both conditions must hold."""
    lines: list[str] = []
    ok = True

    cos, cos_missing = gate_reading(m, "recall_content_cos")
    if cos_missing or cos is None:
        ok = False
        lines.append(f"  arm M recall_content_cos: MISSING for seeds {cos_missing or 'all'}. "
                     "Treated as a FAILURE, never a pass - an absent probe measures nothing.")
    else:
        passed = cos < GATE_MAX_RECALL_COS
        ok = ok and passed
        lines.append(f"  arm M recall_content_cos: {cos:.4f}  must be < "
                     f"{GATE_MAX_RECALL_COS}  {'PASS' if passed else 'FAIL'}")

    frac, frac_missing = gate_reading(s, "unshuffled_frac")
    if frac_missing or frac is None:
        ok = False
        lines.append(f"  arm S unshuffled_frac: MISSING for seeds {frac_missing or 'all'}. "
                     "Treated as a FAILURE.")
    else:
        passed = frac < GATE_MAX_UNSHUFFLED
        ok = ok and passed
        lines.append(f"  arm S unshuffled_frac:   {frac:.4f}  must be < "
                     f"{GATE_MAX_UNSHUFFLED}  {'PASS' if passed else 'FAIL'}")

    return ok, lines


def _sig_note(p: float) -> str:
    if p <= BONFERRONI:
        return f"clears Bonferroni {BONFERRONI:.4f}"
    if p <= ALPHA:
        return (f"clears the claim's alpha {ALPHA} but NOT Bonferroni {BONFERRONI:.4f} "
                f"across {N_CONTRASTS} contrasts")
    return f"above alpha {ALPHA}"


def claim1_verdict(diffs, p: float) -> tuple[str, str]:
    """Claim 1, PRIMARY: `M` minus `A` on held-out success. Returns (status, wording).

    NOT DIRECTIONAL, and that is pre-registered. EXP-058's unlicensed ordering put M BELOW A, so
    collapsing the signs would discard the more likely outcome. A significant -0.05 is reported as
    a real finding that memory HURTS, in the headline, not as a failed confirmation.
    """
    n = len(diffs)
    delta = st.mean(diffs)
    lo, hi = ci95(diffs)
    interval = f"approx 95% interval [{lo:+.4f}, {hi:+.4f}]"

    if p <= ALPHA and delta >= BAR:
        return "HELPS", (f"CONFIRMED: delta {delta:+.4f} at p {p:.4f}, clearing the +{BAR} bar. "
                         f"Correct memory helps a working policy. {_sig_note(p)}.")
    if p <= ALPHA and delta <= -BAR:
        return "HURTS", (f"MEMORY HURTS, and this is a REAL FINDING, not a failed confirmation: "
                         f"delta {delta:+.4f} at p {p:.4f}, clearing the {-BAR} bar downward. "
                         f"The spec pre-registered this reading and requires it in the headline. "
                         f"{_sig_note(p)}.")
    if p <= ALPHA:
        # The likeliest outcome, and the one most easily under-reported. EXP-058's unlicensed
        # ordering put M vs A near -0.03, which lands here: significant, and just under a bar
        # set at 0.05. EXP-057's Claim 2 was the same shape at -0.0446 and its RESULTS.md had
        # to say "do NOT call this a null" in prose, because the wording did not. So the
        # DIRECTION is named here rather than left to the reader.
        way = "AGAINST memory" if delta < 0 else "in memory's favour"
        return "SUB_BAR", (f"significant but sub-bar, and it runs {way}: delta {delta:+.4f} at "
                           f"p {p:.4f}, {interval}. Real, and smaller than the {BAR} the claim "
                           f"required, so it is NOT a confirmation and it is NOT a null. "
                           f"{_sig_note(p)}.")

    head = (f"INDISTINGUISHABLE at n={n}: delta {delta:+.4f}, p {p:.4f}, {interval}. "
            "A BOUND, NEVER AN EQUIVALENCE. ")
    if hi >= BAR or lo <= -BAR:
        head += (f"The interval still reaches the {BAR} bar, so n={n} does NOT resolve the "
                 "question in either direction.")
    else:
        head += (f"The interval stays inside the {BAR} bar, so n={n} DOES bound the effect "
                 "below it at this size.")
    return "NULL", head


def claim2_verdict(diffs, p: float) -> tuple[str, str]:
    """Claim 2, THE MECHANISM: `M` minus `A` on `revisit_rate`.

    MIND THE SIGN. Memory is supposed to REDUCE cycling, so confirmation is delta <= -0.02.
    A significant INCREASE is a real finding and must not be dressed up as a confirmation.

    This is the better-powered instrument: a per-episode rate has a smaller within-arm spread
    than a success count, which is why the spec makes it a claim and not a footnote.
    """
    n = len(diffs)
    delta = st.mean(diffs)
    lo, hi = ci95(diffs)
    interval = f"approx 95% interval [{lo:+.4f}, {hi:+.4f}]"

    if p <= ALPHA and delta <= REVISIT_BAR:
        return "CONFIRMED", (f"CONFIRMED: delta {delta:+.4f} at p {p:.4f}, clearing the "
                             f"{REVISIT_BAR} bar. Memory REDUCED cycling. {_sig_note(p)}.")
    if p <= ALPHA and delta >= -REVISIT_BAR:
        return "OPPOSITE", (f"SIGNIFICANT, OPPOSITE DIRECTION: delta {delta:+.4f} at p {p:.4f}. "
                            f"Memory INCREASED cycling. A real result, and NOT a confirmation "
                            f"of this claim. {_sig_note(p)}.")
    if p <= ALPHA:
        return "SUB_BAR", (f"significant but sub-bar: delta {delta:+.4f} at p {p:.4f}, "
                           f"{interval}. {_sig_note(p)}.")
    return "NULL", (f"indistinguishable at n={n}: delta {delta:+.4f}, p {p:.4f}, {interval}. "
                    "A bound, not an equivalence.")


def joint_reading(c1: str, c2: str) -> str:
    """The four Claim1 x Claim2 combinations, pre-registered in the spec before any number."""
    perf = c1 in ("HELPS", "HURTS", "SUB_BAR")
    mech = c2 == "CONFIRMED"
    if mech and perf:
        return ("Mechanism AND performance both moved. The readout is doing what it was "
                "designed to do and it shows up in the score.")
    if mech and not perf:
        return ("MECHANISM CONFIRMS, PERFORMANCE DOES NOT. The spec names this the interesting "
                "outcome: it localises the failure to the READOUT rather than the hippocampus. "
                "Memory changed the trajectory without changing the score.")
    if perf and not mech:
        return ("Performance moved while the cycling mechanism did not. Whatever memory did, it "
                "was NOT the anti-cycling story this design was built to test, so do not "
                "narrate it as one.")
    return ("Neither moved. A bound on both, and with n=24 only ~30-40% powered at a 0.03 "
            "effect this is a plausible outcome even if memory does something small.")


def report(label: str, arm: dict, control: dict, field: str) -> tuple[list[float], float]:
    seeds, diffs = paired(arm, control, field)
    delta = st.mean(diffs)
    p, method = permutation_p(diffs)
    w = sum(1 for d in diffs if d > 0)
    l = sum(1 for d in diffs if d < 0)
    print(f"\n{label}  on {field}")
    print(f"  n {len(seeds)}   arm {st.mean([arm[s][field] for s in seeds]):.4f}   "
          f"control {st.mean([control[s][field] for s in seeds]):.4f}")
    print(f"  paired delta {delta:+.4f}   W-L-T {w}-{l}-{len(diffs)-w-l}   p {p:.4f}")
    print(f"  method: {method}")
    return diffs, p


def main() -> None:
    out = HERE / "outputs"
    arms = {k: load(out, tag, expected_readout=readout)
            for k, (readout, tag) in ARMS.items()}
    A, M, S = arms["A"], arms["M"], arms["S"]

    print("EXP-059: does episodic memory help a policy that WORKS? Depth 5, n=24.")
    print("Three arms, ONE variable: the readout. Encoder frozen (exp040 E0).\n")
    for k, (readout, tag) in ARMS.items():
        print(f"  arm {k}  {tag:<22} readout {readout:<16} records {len(arms[k])}/{len(SEEDS)}")
    print(f"\n  context, NOT a control: exp043_capped_d5 = {CAPPED_D5_CONTEXT}. No claim is "
          "paired against it,")
    print("  because a readout change alters the feature WIDTH and the amnesic arm is what "
          "controls for that.")

    missing = {k: sorted(set(SEEDS) - set(v)) for k, v in arms.items()}
    if any(missing.values()):
        print("\n  INCOMPLETE:")
        for k, ms in missing.items():
            if ms:
                print(f"    arm {k} missing seeds {ms}")
        print("  Claims below are computed on the seeds present, and n is printed with each.")

    if not (A and M and S):
        raise SystemExit("\nat least one arm has no records; nothing to aggregate")

    print("\n" + "=" * 78)
    print("CLAIM 3, THE VALIDITY GATE - a CONDITION, checked FIRST. Calibrated at `bb1efa5`")
    print("before the spec existed: empty attractor 1.000000, random loaded 0.9437, real")
    print("depth-5 run 0.8128. The threshold sits between a real run and total failure.")
    print("=" * 78)
    ok, lines = check_gate(M, S)
    for line in lines:
        print(line)

    # Descriptive corroboration, deciding nothing. Arm A zeroes `W_rec` at the read site, so its
    # real and zeroed recalls are the same vector and its cosine should sit at ~1.0. That is the
    # cheap check that the probe measures what the gate assumes it measures, rather than the gate
    # merely returning a number. Read it; do not gate on it.
    a_cos, a_missing = gate_reading(A, "recall_content_cos")
    if a_cos is not None:
        print(f"\n  descriptive: arm A recall_content_cos = {a_cos:.6f} "
              f"(expected ~1.0 by construction; arm A zeroes W_rec at the read site).")
        if a_cos < 0.99:
            print("  WARNING: that is not ~1.0. The probe may not be measuring the read site, "
                  "which would undermine the gate above even though the gate passed.")
    elif a_missing:
        print(f"\n  descriptive: arm A recall_content_cos missing for seeds {a_missing}.")

    if not ok:
        print("\n" + "=" * 78)
        print("EXPERIMENT VOID. The read site does not discriminate the arms, so an")
        print("M-versus-A contrast has nothing to measure and NO claim below may be reported.")
        print("Per the gate-calibration rule, this threshold is NOT to be edited now that")
        print("numbers exist. That is what cost EXP-058, and rewriting it afterwards costs")
        print("the method rather than one experiment.")
        print("=" * 78)
        return
    print("\n  gate: PASSED. The claims below may be read.")

    print("\n" + "=" * 78)
    print("CLAIM 1, PRIMARY - M minus A on held-out success. NOT DIRECTIONAL.")
    print("=" * 78)
    d1, p1 = report("M - A (primary)", M, A, "success_rate")
    c1, words1 = claim1_verdict(d1, p1)
    print(f"  {words1}")

    print("\n" + "=" * 78)
    print("CLAIM 2, THE MECHANISM - M minus A on revisit_rate. Confirmation is NEGATIVE.")
    print("=" * 78)
    d2, p2 = report("M - A (mechanism)", M, A, "revisit_rate")
    c2, words2 = claim2_verdict(d2, p2)
    print(f"  {words2}")

    print("\n  JOINT READING, pre-registered before any number existed:")
    print(f"  {joint_reading(c1, c2)}")

    print("\n" + "=" * 78)
    print("CLAIM 4, SECONDARY - M minus S.")
    print("=" * 78)
    d4, p4 = report("M - S (secondary)", M, S, "success_rate")
    _, words4 = claim1_verdict(d4, p4)
    print(f"  {words4}")
    print("\n  THIS MEASURES THE HARM OF *INCORRECT* MEMORY, AND IS NOT EVIDENCE ABOUT THE")
    print("  BENEFIT OF CORRECT MEMORY. The spec requires that sentence attached, because")
    print("  EXP-030's headline came from this contrast and was misread for months.")

    print("\n" + "=" * 78)
    print(f"MULTIPLICITY: {N_CONTRASTS} inferential contrasts (Claims 1, 2, 4), Bonferroni "
          f"{BONFERRONI:.4f}.")
    print("Claim 3 is a condition with no p-value and belongs to no family.")
    print(f"POWER, stated in the spec beforehand: at n={len(d1)} and this project's measured")
    print("paired sd of 0.10-0.14, power is ~60-70% at a 0.05 effect and ~30-40% at 0.03.")
    print("EXP-058's unlicensed ordering suggested about -0.03 for M vs A, which sits in the")
    print("poorly-powered row, so an indistinguishable Claim 1 was always a plausible outcome.")
    print("=" * 78)


if __name__ == "__main__":
    main()
