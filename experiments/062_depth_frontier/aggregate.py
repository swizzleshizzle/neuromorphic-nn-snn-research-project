"""EXP-062 aggregator: the pre-registered rules from the spec, applied to the records on disk.

Spec: `docs/superpowers/specs/2026-09-15-exp062-depth-frontier-design.md`.

> WRITTEN BEFORE ANY EXP-062 NUMBER WAS SEEN. Completion was confirmed from COUNTS alone - 24
> records, 24 head checkpoints, zero python processes - without reading the run log, which prints
> one success rate per line. EXP-060's aggregator could not claim this; EXP-061's and this one can.

> CLAIM 3 IS A CONDITION AND IS CHECKED FIRST. These numbers are only interpretable if the critic
> - the recipe component that makes depth 7 work at all - is present and state-dependent. A
> missing or `None` ratio is a FAILURE, never a pass: coercing it to 0.0 would fail a floor gate
> anyway, but silently dropping it from a mean would not, and that is the trap.

> CLAIM 1 IS A ONE-SAMPLE CONTAINMENT TEST, NOT A PAIRED CONTRAST. There is nothing to pair depth
> 8 against: the question is whether a law fitted at depths 3-7 predicts it. So the statistic is a
> two-sided 95% t interval on the depth-8 mean, and the test is whether the law's 0.0588 falls
> inside it.

> CLAIM 2 IS A FRONTIER MEASUREMENT AND CANNOT BE A TEST OF THE LAW. Success is bounded below at
> zero, so the law's -0.0828 prediction at depth 9 can only manifest as "about zero" - this design
> cannot discriminate -0.08 from -0.5. Reading a zero there as CONFIRMATION would be reading a
> bound as evidence, which is the error the spec exists to prevent.

Usage:
    .venv/bin/python -u experiments/062_depth_frontier/aggregate.py
"""

from __future__ import annotations

import importlib.util
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("exp062_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

SEEDS = _run.SEEDS
DEPTHS = _run.DEPTHS
CHANCE_FLOOR = _run.CHANCE_FLOOR
FLOOR_BAR = _run.FLOOR_BAR              # 0.02
GATE_MIN = _run.GATE_MIN_RATIO          # 0.05
ANCHOR_DEPTH = _run.ANCHOR_DEPTH        # 7
ANCHOR_SUCCESS = _run.ANCHOR_SUCCESS    # 0.2004
law_prediction = _run.law_prediction
tag_for = _run.tag_for

T95 = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
       20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069}


def t95_for(n: int) -> float:
    """df = n-1. n IS 12 here, so df=11 and the multiplier is 2.201 - the same value
    EXP-055/056/057 hardcode. That is a COINCIDENCE of this experiment's sample size, not a
    reason to import their code, which would be wrong the moment n changed."""
    df = n - 1
    if df in T95:
        return T95[df]
    if df > 23:
        return 1.96
    raise ValueError(f"n={n} is too small to report an interval for")


def ci95(vals) -> tuple[float, float]:
    n = len(vals)
    m = st.mean(vals)
    se = st.stdev(vals) / (n ** 0.5) if n > 1 else 0.0
    t = t95_for(n)
    return m - t * se, m + t * se


def load(directory: Path, tag: str, expect_depth: int) -> dict:
    """Records by seed. `expect_depth` is checked: the two arms differ ONLY in depth, and a tag
    pointing at the wrong depth would silently compare an arm against itself."""
    out: dict[int, dict] = {}
    for p in sorted(Path(directory).glob("*.json")):
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or r.get("tag") != tag:
            continue
        got = (r.get("config") or {}).get("depth")
        if got != expect_depth:
            raise SystemExit(f"{p.name}: tag {tag} has depth {got}, expected {expect_depth}.")
        out[int(r["seed"])] = r
    return out


def gate_rows(arm: dict) -> list[dict]:
    """Claim 3: the critic's within-episode RMS against the returns' own, per stage."""
    seeds = sorted(arm)
    n_stages = len(arm[seeds[0]]["stage_trace"])
    rows = []
    for i in range(n_stages):
        cw = st.mean([arm[s]["stage_trace"][i].get("critic_within_rms", 0.0) for s in seeds])
        rw = st.mean([arm[s]["stage_trace"][i].get("return_within_rms", 0.0) for s in seeds])
        rows.append({"depth": arm[seeds[0]]["stage_trace"][i]["depth"],
                     "critic": cw, "ret": rw, "ratio": (cw / rw) if rw else None})
    return rows


def gate_passed(rows) -> bool:
    """EVERY stage, as pre-registered. A None ratio is a FAILURE, never a pass."""
    return all(r["ratio"] is not None and r["ratio"] >= GATE_MIN for r in rows)


def claim1_verdict(vals) -> tuple[str, str]:
    """PRIMARY: does the 95% interval on depth-8 mean success contain the law's prediction?"""
    n = len(vals)
    m = st.mean(vals)
    lo, hi = ci95(vals)
    pred = law_prediction(8)
    iv = f"mean {m:.4f}, approx 95% interval [{lo:.4f}, {hi:.4f}], n={n}"
    if lo <= pred <= hi:
        return "HOLDS", (f"LAW SUPPORTED OUT OF SAMPLE: the interval CONTAINS the predicted "
                         f"{pred:.4f}. {iv}. **The Stage-4 pricing is evidence-backed rather "
                         f"than extrapolated** - depth 11 really is priced out.")
    if m > hi or pred < lo:
        return "PESSIMISTIC", (f"LAW IS PESSIMISTIC: the interval EXCLUDES {pred:.4f} and the "
                               f"measured mean is ABOVE it. {iv}. **Stage 4 is cheaper than "
                               f"priced and the frontier question reopens** - the extrapolation "
                               f"that declared it unreachable was too harsh.")
    return "OPTIMISTIC", (f"LAW IS OPTIMISTIC: the interval EXCLUDES {pred:.4f} and the measured "
                          f"mean is BELOW it. {iv}. **Stage 4 is even further out of reach than "
                          f"priced, and earlier depth projections were too kind.**")


def claim2_verdict(vals) -> tuple[str, str]:
    """The FRONTIER at depth 9. NOT a test of the law - see the module docstring."""
    n = len(vals)
    m = st.mean(vals)
    lo, hi = ci95(vals)
    floor = CHANCE_FLOOR.get(9, 0.0)
    nz = sum(1 for v in vals if v > 0)
    if m < FLOOR_BAR:
        return "AT_FLOOR", (f"AT THE FLOOR: mean {m:.4f} < {FLOOR_BAR}, against a MEASURED chance "
                            f"floor of exactly {floor:.4f}. {nz} of {n} seeds scored above zero. "
                            f"**The frontier is depth 8.**")
    return "ABOVE_FLOOR", (f"ABOVE THE FLOOR: mean {m:.4f} >= {FLOOR_BAR}, approx 95% interval "
                           f"[{lo:.4f}, {hi:.4f}], {nz} of {n} seeds above zero. **Depth 9 is not "
                           f"dead**, which the law did not predict and which pushes the frontier "
                           f"out at least one depth.")


def main() -> None:
    out = HERE / "outputs"
    arms = {d: load(out, tag_for(d), d) for d in DEPTHS}
    for d, a in arms.items():
        if not a:
            raise SystemExit(f"no records for depth {d} in {out}")

    print("EXP-062: the honest depth frontier, and an out-of-sample test of the budget law.\n")
    print("  WHY: road-to-a-solved-cube calls Stage 4 - depth-11 random scrambles - 'the actual")
    print("  deliverable' and 'genuinely achievable... nothing about it requires new science'.")
    print("  That was written BEFORE week 20's budget law, which prices depth 11 at depth-7")
    print("  parity at ~1,796 h per cell, ~75 days per seed. The pricing extrapolates a law")
    print("  fitted at depths 3-7 out to 11, and THAT is what this experiment tests.\n")
    for d in DEPTHS:
        print(f"  depth {d}: {len(arms[d])}/{len(SEEDS)} cells, law predicts "
              f"{law_prediction(d):+.4f}, measured chance floor {CHANCE_FLOOR.get(d):.4f}")
    print("\n  This aggregator was written BEFORE any number was seen: completion was confirmed")
    print("  from counts alone, without reading the run log.")

    print("\n" + "=" * 78)
    print(f"CLAIM 3, THE VALIDITY GATE - a CONDITION, checked FIRST, BOTH arms.")
    print(f"Is the critic present and state-dependent? Ratio must be >= {GATE_MIN} at EVERY stage.")
    print("=" * 78)
    ok = True
    for d in DEPTHS:
        rows = gate_rows(arms[d])
        passed = gate_passed(rows)
        ok = ok and passed
        ratios = " ".join(f"{r['ratio']:.3f}" if r["ratio"] is not None else "None" for r in rows)
        worst = min((r["ratio"] for r in rows if r["ratio"] is not None), default=None)
        print(f"  depth {d}: {ratios}")
        print(f"           worst {worst:.4f}" if worst is not None else "           worst None")
        print(f"           {'PASS' if passed else 'FAIL'}")
    if not ok:
        print("\n" + "=" * 78)
        print("EXPERIMENT VOID by its own pre-registered condition. The critic did not vary")
        print("within episodes, so these arms are not the recipe the spec describes and no claim")
        print("below may be read. The floor is NOT to be lowered now that numbers exist - that is")
        print("the EXP-058 mistake, and rewriting a gate afterwards costs the method.")
        print("=" * 78)
        return
    print("\n  gate: PASSED in both arms. The claims below may be read.")

    d8 = [arms[8][s]["success_rate"] for s in sorted(arms[8])]
    d9 = [arms[9][s]["success_rate"] for s in sorted(arms[9])]

    print("\n" + "=" * 78)
    print("CLAIM 1, PRIMARY - does the budget law hold OUT OF SAMPLE at depth 8?")
    print("A ONE-SAMPLE CONTAINMENT TEST, not a paired contrast.")
    print("=" * 78)
    print(f"  depth 8 per-seed: {' '.join(f'{v:.3f}' for v in d8)}")
    print(f"  law fitted at depths 3-7, anchored at depth {ANCHOR_DEPTH} = {ANCHOR_SUCCESS}")
    c1, w1 = claim1_verdict(d8)
    print(f"\n  {w1}")

    print("\n" + "=" * 78)
    print("CLAIM 2 - the FRONTIER at depth 9. NOT a test of the law.")
    print("=" * 78)
    print(f"  depth 9 per-seed: {' '.join(f'{v:.3f}' for v in d9)}")
    c2, w2 = claim2_verdict(d9)
    print(f"\n  {w2}")
    print("\n  WHY THIS IS NOT A TEST OF THE LAW: success is bounded below at zero, so the law's")
    print(f"  {law_prediction(9):+.4f} prediction can only manifest as 'about zero'. This design")
    print("  CANNOT discriminate -0.08 from -0.5. A zero here establishes where the frontier is;")
    print("  it is not evidence for the law, and must never be reported as confirmation.")

    print("\n" + "=" * 78)
    print("THE FRONTIER, stated plainly")
    print("=" * 78)
    print(f"  depth 7 (EXP-053 arm B)  {ANCHOR_SUCCESS:.4f}")
    print(f"  depth 8                  {st.mean(d8):.4f}")
    print(f"  depth 9                  {st.mean(d9):.4f}")
    print(f"  measured chance floor at depths 7, 8, 9: exactly 0.0000 (12 seeds, random policy)")
    print("\n  STAGE 4 (depth 11) REMAINS PRICED OUT under any of these readings. Even the most")
    print("  favourable leaves depth 11 at tens of days per seed. This experiment prices the")
    print("  deliverable; it does not deliver it, and Phase 3 should report that plainly.")
    print("=" * 78)


if __name__ == "__main__":
    main()
