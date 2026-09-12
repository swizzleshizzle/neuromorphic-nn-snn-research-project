"""EXP-060 aggregator: the pre-registered rules from the spec, applied to the records on disk.

Spec: `docs/superpowers/specs/2026-09-10-exp060-flattened-critic-replication-design.md`.

> DISCLOSURE, because the standard here is to say so. This file was written AFTER the run
> finished, not before, unlike EXP-059's. Verifying phase 2 had completed meant reading the
> launcher log's tail, which prints one line per cell, so the author had seen roughly twelve
> INDIVIDUAL cell success rates before writing this. No contrast, arm mean, or gate value had
> been computed. Every decision rule below - the bar, the direction, the three readings, the
> multiplicity, the test, the gate threshold - was fixed in the spec before dispatch, so there
> was little discretion left to contaminate; but the exposure is real and is recorded rather
> than glossed.

> CLAIM 3 IS A CONDITION AND IS CHECKED FIRST. If `V` barely varied within an episode there was
> nothing for flattening to remove, and Claim 1's null would be vacuous rather than informative.
>
> NOTE A DELIBERATE DIVERGENCE FROM EXP-056. That experiment's own aggregator passed its gate on
> `any(ratio >= 0.05)`. **This spec pre-registered `>= 0.05` at EVERY stage**, which is strictly
> stricter. The spec's wording binds, because weakening a gate once numbers exist is precisely
> what the gate-calibration rule forbids. Both verdicts are printed so the difference is visible,
> and EXP-056's smallest observed ratio was 0.415 - eight times the threshold - so the stricter
> form was expected to pass comfortably.

Usage:
    .venv/bin/python -u experiments/060_flattened_critic_replication/aggregate.py
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

_spec = importlib.util.spec_from_file_location("exp060_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

SEEDS = _run.SEEDS                       # 14-23, the FRESH seeds
ARMS = _run.ARMS
BAR = _run.BAR                           # 0.05
ALPHA = 0.05
N_CONTRASTS = 2                          # Claims 1 and 2. Claim 3 is a condition.
BONFERRONI = ALPHA / N_CONTRASTS         # 0.025, the same bar EXP-056 cleared by 6.4%
GATE_FRACTION = 0.05                     # spec Claim 3

# The pooled secondary needs EXP-056's flattened arm and EXP-053's arm B, seeds 0-11.
POOL_SEEDS = tuple(range(12))
POOL_F = (REPO / "experiments/056_flattened_critic/outputs", "exp056_flat_d7")
POOL_B = (REPO / "experiments/053_neuromod_stage3/outputs", "exp053_critic_d7")
EXP056_DELTA, EXP056_P = -0.0646, 0.0234

EXACT_MAX_N = 20
SAMPLED_DRAWS = 200_000
SAMPLED_SEED = 20260910

T95 = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
       20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069}


def t95_for(n: int) -> float:
    """df = n-1. Spelled out because there is no scipy, and because a single hardcoded value is
    the defect EXP-055/056/057 all carry: they use df=11 regardless of n."""
    df = n - 1
    if df in T95:
        return T95[df]
    if df > 23:
        return 1.96
    raise ValueError(f"n={n} too small for an interval")


def permutation_p(diffs) -> tuple[float, str]:
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
    # Add-one: a plain hits/draws can print 0.0000 and claim an exactness sampling cannot support.
    return (hits + 1) / (SAMPLED_DRAWS + 1), (
        f"sampled, {SAMPLED_DRAWS:,} draws at seed {SAMPLED_SEED} (exhaustive would be "
        f"2**{n} = {2 ** n:,}); add-one estimator")


def ci95(diffs) -> tuple[float, float]:
    n = len(diffs)
    d = st.mean(diffs)
    se = st.stdev(diffs) / (n ** 0.5) if n > 1 else 0.0
    t = t95_for(n)
    return d - t * se, d + t * se


def load(directory: Path, tag: str, expect_flatten: bool | None = None) -> dict:
    """Records by seed. `expect_flatten` is checked when given: both EXP-060 arms share
    `arm="regionalized"` and `readout="concept"` and differ ONLY in `flatten_critic`, so a driver
    slip pointing two tags at one setting would contrast an arm against itself invisibly."""
    out: dict[int, dict] = {}
    for p in sorted(Path(directory).glob("*.json")):
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or r.get("tag") != tag:
            continue
        if expect_flatten is not None:
            got = (r.get("config") or {}).get("flatten_critic")
            if bool(got) is not expect_flatten:
                raise SystemExit(f"{p.name}: tag {tag} has flatten_critic={got}, expected "
                                 f"{expect_flatten}. The arms are not what the spec describes.")
        out[int(r["seed"])] = r
    return out


def paired(arm: dict, ctrl: dict, seeds, field="success_rate"):
    use = sorted(set(arm) & set(ctrl) & set(seeds))
    return use, [arm[s][field] - ctrl[s][field] for s in use]


def gate_by_stage(arm: dict) -> list[dict]:
    """Claim 3, measured on the FLATTENED arm - that is where the removed variation lived."""
    seeds = sorted(arm)
    rows = []
    for i in range(len(arm[seeds[0]]["stage_trace"])):
        cw = st.mean([arm[s]["stage_trace"][i].get("critic_within_rms", 0.0) for s in seeds])
        rw = st.mean([arm[s]["stage_trace"][i].get("return_within_rms", 0.0) for s in seeds])
        rows.append({"depth": arm[seeds[0]]["stage_trace"][i]["depth"],
                     "critic_within_rms": cw, "return_within_rms": rw,
                     "ratio": (cw / rw) if rw else 0.0})
    return rows


def gate_every_stage(rows) -> bool:
    """THE PRE-REGISTERED CONDITION for this experiment: every stage."""
    return all(r["ratio"] >= GATE_FRACTION for r in rows)


def gate_any_stage(rows) -> bool:
    """EXP-056's own, looser form. Printed for comparison only; it does NOT decide anything."""
    return any(r["ratio"] >= GATE_FRACTION for r in rows)


def _sig(p: float) -> str:
    if p <= BONFERRONI:
        return f"clears Bonferroni {BONFERRONI:.4f}"
    if p <= ALPHA:
        return f"clears alpha {ALPHA} but NOT Bonferroni {BONFERRONI:.4f}"
    return f"above alpha {ALPHA}"


def claim1_verdict(diffs, p: float) -> tuple[str, str]:
    """Claim 1, PRIMARY: `F` minus `B`, seeds 14-23 ONLY. Directional - EXP-056 pre-registered a
    direction and found it, so a replication tests that direction rather than re-opening it."""
    n = len(diffs)
    d = st.mean(diffs)
    lo, hi = ci95(diffs)
    iv = f"approx 95% interval [{lo:+.4f}, {hi:+.4f}]"
    if p <= ALPHA and d <= -BAR:
        return "REPLICATED", (f"REPLICATED: delta {d:+.4f} at p {p:.4f}, clearing the {-BAR} bar "
                              f"downward. The EXP-056/057 picture stands. {_sig(p)}.")
    if p <= ALPHA and d >= BAR:
        return "REVERSED", (f"SIGNIFICANT REVERSAL: delta {d:+.4f} at p {p:.4f}. This belongs in "
                            f"the headline and puts the critic conclusion in serious doubt. "
                            f"{_sig(p)}.")
    if p <= ALPHA:
        way = "in EXP-056's direction" if d < 0 else "AGAINST EXP-056's direction"
        return "SUB_BAR", (f"significant but sub-bar, running {way}: delta {d:+.4f} at p {p:.4f}, "
                           f"{iv}. Not a replication at the pre-registered bar, and not a null. "
                           f"{_sig(p)}.")
    head = (f"NOT REPLICATED AT n={n}, and this is a BOUND - never a refutation of EXP-056 and "
            f"never an equivalence: delta {d:+.4f}, p {p:.4f}, {iv}. ")
    if lo <= -BAR:
        head += (f"The interval still reaches {-BAR}, so n={n} does not exclude EXP-056's effect.")
    else:
        head += (f"The interval excludes {-BAR}, so n={n} DOES bound the effect short of the bar.")
    return "NULL", head


def main() -> None:
    out = HERE / "outputs"
    F = load(out, ARMS["F"][1], expect_flatten=True)
    B = load(out, ARMS["B"][1], expect_flatten=False)
    if not F or not B:
        raise SystemExit(f"missing EXP-060 records in {out}")

    print("EXP-060: an INDEPENDENT replication of EXP-056's flattened critic, depth 7.")
    print(f"  arm F {ARMS['F'][1]:<18} records {len(F)}/{len(SEEDS)}")
    print(f"  arm B {ARMS['B'][1]:<18} records {len(B)}/{len(SEEDS)}")
    print(f"  seeds {tuple(sorted(set(F) & set(B)))} - FRESH, never used for this question.\n")
    print("  DISCLOSURE: this aggregator was written after the run finished, and the author had")
    print("  seen individual cell success rates from the log while verifying completion. Every")
    print("  rule applied below was fixed in the spec before dispatch. See the module docstring.")

    rows = gate_by_stage(F)
    ok_every, ok_any = gate_every_stage(rows), gate_any_stage(rows)
    print("\n" + "=" * 78)
    print("CLAIM 3, THE VALIDITY GATE - a CONDITION, checked FIRST, on the FLATTENED arm.")
    print(f"Did flattening remove anything? Ratio must be >= {GATE_FRACTION} at EVERY stage.")
    print("=" * 78)
    print(f"{'depth':>6} {'V within RMS':>14} {'return within RMS':>18} {'ratio':>9}")
    for r in rows:
        print(f"{r['depth']:>6} {r['critic_within_rms']:>14.4f} "
              f"{r['return_within_rms']:>18.4f} {r['ratio']:>9.4f}")
    print(f"  EVERY-stage (pre-registered here): {'PASS' if ok_every else 'FAIL'}")
    print(f"  ANY-stage (EXP-056's own, looser, decides NOTHING here): "
          f"{'pass' if ok_any else 'fail'}")

    if not ok_every:
        print("\n" + "=" * 78)
        print("EXPERIMENT VOID by its own pre-registered condition. `V` did not vary enough at")
        print("every stage, so there was nothing for flattening to remove and Claim 1 cannot be")
        print("read. The threshold is NOT to be relaxed now that numbers exist - that is the")
        print("EXP-058 mistake, and rewriting a gate afterwards costs the method, not one run.")
        if ok_any:
            print("NOTE: EXP-056's looser ANY-stage form would have passed. That is a reason to")
            print("write the next spec more carefully, NOT a reason to switch forms now.")
        print("=" * 78)
        return
    print("\n  gate: PASSED. Claim 1 may be read.")

    print("\n" + "=" * 78)
    print("CLAIM 1, PRIMARY - F minus B on seeds 14-23 ALONE. Directional.")
    print("=" * 78)
    seeds1, d1 = paired(F, B, SEEDS)
    p1, method1 = permutation_p(d1)
    w = sum(1 for x in d1 if x > 0); l = sum(1 for x in d1 if x < 0)
    print(f"  n {len(seeds1)}   arm F {st.mean([F[s]['success_rate'] for s in seeds1]):.4f}   "
          f"arm B {st.mean([B[s]['success_rate'] for s in seeds1]):.4f}")
    print(f"  paired delta {st.mean(d1):+.4f}   W-L-T {w}-{l}-{len(d1)-w-l}   p {p1:.4f}")
    print(f"  method: {method1}")
    status1, words1 = claim1_verdict(d1, p1)
    print(f"  {words1}")
    print(f"\n  EXP-056 measured {EXP056_DELTA:+.4f} at p {EXP056_P}, clearing Bonferroni 0.025 by")
    print("  6.4% of its margin. That is the claim under test, not a control.")

    print("\n" + "=" * 78)
    print("CLAIM 2, SECONDARY - the POOLED contrast. Read the caveat, every time.")
    print("=" * 78)
    pf = load(*POOL_F)
    pb = load(*POOL_B)
    if not pf or not pb:
        print(f"  POOLED SKIPPED: missing records for {POOL_F[1]} or {POOL_B[1]}.")
    else:
        seeds0, d0 = paired(pf, pb, POOL_SEEDS)
        pooled = d0 + d1
        pp, methodp = permutation_p(pooled)
        lo, hi = ci95(pooled)
        print(f"  n {len(pooled)} = {len(d0)} from EXP-056/053 (seeds 0-11) + {len(d1)} fresh")
        print(f"  paired delta {st.mean(pooled):+.4f}   p {pp:.4f}   "
              f"approx 95% interval [{lo:+.4f}, {hi:+.4f}]")
        print(f"  method: {methodp}")
        print("\n  **THIS FIGURE INHERITS OPTIONAL STOPPING AND MUST CARRY THAT SENTENCE WHEREVER")
        print("  IT IS QUOTED.** The decision to add seeds 14-23 was made BECAUSE EXP-056's")
        print("  p-value was marginal, so a pooled p-value is selected-on and is not the number")
        print("  it appears to be. The PRIMARY is Claim 1 above, on the fresh seeds alone.")

    print("\n" + "=" * 78)
    print(f"MULTIPLICITY: {N_CONTRASTS} inferential contrasts, Bonferroni {BONFERRONI:.4f} - the")
    print("same bar EXP-056 cleared by 6.4% of its margin. Claim 3 is a condition, no p-value.")
    print("POWER, from the spec before dispatch: ~50-60% at EXP-056's -0.0646, and that estimate")
    print("is UPWARD-BIASED because it was selected for significance, so the realistic case is")
    print("lower. A null primary was always a plausible outcome and is a BOUND, not a refutation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
