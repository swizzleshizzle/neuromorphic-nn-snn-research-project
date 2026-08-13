"""EXP-042 aggregator: apply the pre-registered rules to the records on disk.

WRITTEN BEFORE THE RECORDS WERE READ, which is the standing habit - the rules go on disk before
the numbers do. Thresholds are the ones committed in
docs/superpowers/specs/2026-08-12-exp042-depth1-trap-design.md. If a number here disagrees with
the spec, the spec wins and this file is the bug.

> THE STATISTICAL LIMIT, WHICH DECIDES HOW EVERYTHING BELOW IS REPORTED
> The effect under test is 2 of 12 seeds. In a paired permutation test where two seeds carry the
> whole difference and ten are ~0, only 2 of the 4 sign assignments on those two exceed the
> observed sum, so p is about 0.5 BY CONSTRUCTION. Fisher's exact on 2/12 against 0/12 gives
> about 0.48. NO ARRANGEMENT OF 12 SEEDS CAN PROVE THE FAILURES WERE ELIMINATED.
>
> So the primary claim sits on entropy entering the final stage - a per-seed measure that varies
> across all twelve - and the failure count is printed WITHOUT a p-value. This file refuses to
> attach one.

Usage:
    .venv/bin/python experiments/042_depth1_trap/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- pre-registered. Do not edit after data exists. ---
CLAIM1_DELTA = 0.05        # entropy entering the final stage, vs baseline
ALPHA = 0.05
COST_DELTA = 0.05          # Claim 3: a fall this large among the never-failed seeds
COLLAPSE_ENTROPY = 0.05    # below this entering the final stage = collapsed
EXP040_ZEROS = [2, 4]      # the seeds that failed in EXP-040
ARMS = ["baseline", "capped", "skipped"]
# Deterministic, enumerated, already true (test_depth1_cap_removes_the_constant_action_exploit).
CONST_REWARD = {"baseline": 0.3333, "capped": 0.1667, "skipped": None}
RANDOM_REWARD_D1 = 0.2208


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


def arm_cells(recs, arm: str) -> dict:
    return {r["seed"]: r for r in recs if f"exp042_{arm}_" in r.get("tag", "")}


def final_stage_entropy(rec: dict):
    """Entropy ENTERING the final curriculum stage - the primary quantity.

    `stage_trace` is ordered by stage, so the last entry is the evaluated depth. Its
    `entropy_first_10pct` is the exploration the policy still had when it arrived there.
    """
    trace = rec.get("stage_trace") or []
    return trace[-1]["entropy_first_10pct"] if trace else None


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def paired(a: dict, b: dict, fn):
    seeds = sorted(set(a) & set(b))
    pairs = [(fn(a[s]), fn(b[s])) for s in seeds]
    return seeds, [x - y for x, y in pairs if x is not None and y is not None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    recs = load(args.out_dir)
    if not recs:
        raise SystemExit(f"no records in {args.out_dir}. Has the run finished, or been fetched?")

    cells = {a: arm_cells(recs, a) for a in ARMS}
    print(f"EXP-042: {len(recs)} records")
    for a in ARMS:
        print(f"  {a:<9} {len(cells[a])} seeds")
    if len(recs) != 36:
        print("  INCOMPLETE: expected 36. Verdicts below are provisional.")

    missing = [a for a in ARMS if not cells[a]]
    if missing:
        raise SystemExit(f"missing arm(s): {missing}")

    # ---- instrument check: is stage_trace actually present? ----
    if any(final_stage_entropy(r) is None for r in recs):
        n = sum(1 for r in recs if final_stage_entropy(r) is None)
        print(f"\n  WARNING: {n} record(s) have no stage_trace. Those predate the telemetry and")
        print("  cannot answer Claim 1. Re-run them rather than comparing across instruments.")

    print("\n" + "=" * 78)
    print("CLAIM 4 (mechanism check, deterministic - enumerated, not measured here)")
    for a in ARMS:
        r = CONST_REWARD[a]
        note = "n/a - no depth-1 stage" if r is None else (
            f"{r:.4f} vs random {RANDOM_REWARD_D1}  "
            f"({'degeneracy WINS' if r > RANDOM_REWARD_D1 else 'degeneracy loses'})")
        print(f"  {a:<9} depth-1 constant-action reward: {note}")

    print("\nPER-ARM SUMMARY")
    print(f"  {'arm':<9}{'success':>10}{'sd':>8}{'zeros':>7}{'final-stage entropy':>22}"
          f"{'collapsed':>11}")
    summary = {}
    for a in ARMS:
        succ = [r["success_rate"] for r in cells[a].values()]
        ents = [e for e in (final_stage_entropy(r) for r in cells[a].values()) if e is not None]
        zeros = sum(1 for x in succ if x == 0)
        collapsed = sum(1 for e in ents if e < COLLAPSE_ENTROPY)
        summary[a] = {"succ": succ, "ents": ents, "zeros": zeros}
        ent_txt = f"{st.mean(ents):.4f} +-{sd(ents):.4f}" if ents else "n/a"
        print(f"  {a:<9}{st.mean(succ):>10.4f}{sd(succ):>8.4f}{zeros:>7}{ent_txt:>22}"
              f"{collapsed:>11}")

    base = cells["baseline"]
    print("\n" + "=" * 78)
    print(f"CLAIM 1 (PRIMARY), entropy entering the final stage. "
          f"Need >= {CLAIM1_DELTA} at p <= {ALPHA}.")
    for a in ("capped", "skipped"):
        seeds, diffs = paired(cells[a], base, final_stage_entropy)
        if not diffs:
            print(f"  {a:<9} cannot evaluate (no stage_trace on both sides)")
            continue
        m, p = st.mean(diffs), permutation_p(diffs)
        # How much of the effect is carried by the two seeds that failed in EXP-040? If it is
        # nearly all of it, the test is underpowered BY CONSTRUCTION and must be reported so.
        #
        # FIXED 2026-08-13, after the first run printed "354%". Dividing by `abs(sum(diffs))`
        # is unbounded: when the arm's net effect is near zero because seeds cancel, a modest
        # contribution divides by a tiny denominator and exceeds 100%, which is meaningless.
        # A share must be a share. Denominator is now the total absolute movement, so it is
        # bounded by 1. NO THRESHOLD CHANGED - this is a broken diagnostic, not a moved bar.
        idx = {s: i for i, s in enumerate(seeds)}
        from_failed = sum(abs(diffs[idx[s]]) for s in EXP040_ZEROS if s in idx)
        total_abs = sum(abs(d) for d in diffs)
        share = (from_failed / total_abs) if total_abs else 0.0
        # The directly interpretable quantity: does the effect survive dropping those seeds?
        rest = [diffs[idx[s]] for s in seeds if s not in EXP040_ZEROS]
        mean_rest = st.mean(rest) if rest else 0.0
        nonzero = sum(1 for d in diffs if abs(d) > 0.01)
        print(f"  {a:<9} {m:+.4f}  exact p {p:.4f}  (n={len(diffs)}, "
              f"{nonzero} seeds moved >0.01, {100 * share:.0f}% of total movement from "
              f"seeds {EXP040_ZEROS}, mean excluding them {mean_rest:+.4f})")
        if m >= CLAIM1_DELTA and p <= ALPHA:
            print("            CONFIRMED. The fix keeps exploration alive into the final stage.")
        elif share > 0.8:
            print(f"            UNDERPOWERED BY CONSTRUCTION. {100 * share:.0f}% of the movement")
            print(f"            comes from seeds {EXP040_ZEROS}, and a 2-of-12 effect cannot")
            print("            reach significance in a paired test. This is NOT 'the fix does")
            print("            not work' - read Claim 2's descriptive result instead.")
        else:
            print(f"            REFUTED ({m:+.4f}, p {p:.4f}). The effect is SPREAD across seeds")
            print(f"            ({nonzero}/12 moved, {mean_rest:+.4f} excluding the failed two)")
            print("            and still did not clear the bar, so this IS a real refutation.")

    print("\nCLAIM 2 (SECONDARY, DESCRIPTIVE - no p-value, by design)")
    print(f"  EXP-040 baseline for reference: 2/12 seeds at exactly 0.000 (seeds {EXP040_ZEROS})")
    for a in ARMS:
        s = summary[a]
        print(f"  {a:<9} {s['zeros']}/12 seeds at 0.000   mean {st.mean(s['succ']):.4f}")
    print("  A drop to 0/12 is SUGGESTIVE AND NO MORE. Confirming it needs ~40+ seeds per arm.")

    print(f"\nCLAIM 3 (COST), among the ten seeds that did NOT fail in EXP-040. "
          f"Flag at -{COST_DELTA}.")
    keep = [s for s in base if s not in EXP040_ZEROS]
    for a in ("capped", "skipped"):
        pairs = [(cells[a][s]["success_rate"], base[s]["success_rate"])
                 for s in keep if s in cells[a]]
        if not pairs:
            continue
        d = st.mean(x - y for x, y in pairs)
        print(f"  {a:<9} {d:+.4f} on {len(pairs)} previously-working seeds"
              f"{'   COST FLAGGED' if d <= -COST_DELTA else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
