"""EXP-039 aggregator: apply the pre-registered rules to the records on disk.

Separate from run.py so the verdicts do not depend on catching a one-shot summary at the end
of a long job, and so they can be re-read any number of times.

Thresholds are NOT re-derived here. They are the ones committed in
docs/superpowers/specs/2026-08-08-exp039-encoder-pretraining-design.md before any number
existed. If a number here disagrees with the spec, the spec wins and this file is the bug.

Usage:
    .venv/bin/python experiments/039_encoder_pretraining/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- pre-registered, from the spec. Do not edit after data exists. ---
BAR1_DELTA = 0.05             # depth-4 trained - frozen
ALPHA = 0.05
BAR2_CEILING = 0.766          # EXP-033's raw-facelet probe at depth 4
EXP033_FROZEN_D4 = 0.459      # EXTERNAL sanity check only
SANITY_TOLERANCE = 0.10
PRIMARY_DEPTH = "4"
REPORT_DEPTHS = ["3", "4", "5", "6"]
# Naming one of six moves. If pretraining accuracy sits here, the objective was never learned
# and every probe number below describes a random encoder under another name.
MOVE_CHANCE = 1.0 / 6.0
MOVE_LEARNED_MIN = 0.30


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
    return sorted((json.loads(p.read_text()) for p in d.glob("exp039_s*.json")),
                  key=lambda r: r["seed"])


def at(rec: dict, arm: str, depth: str, field: str = "top1"):
    node = rec.get(arm, {}).get("by_depth", {}).get(depth)
    return None if node is None else node[field]


def series(recs, arm: str, depth: str, field: str = "top1") -> list[float]:
    return [v for r in recs if (v := at(r, arm, depth, field)) is not None]


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    recs = load(args.out_dir)
    if not recs:
        raise SystemExit(f"no records in {args.out_dir}. Has the run finished?")

    print(f"EXP-039: {len(recs)} seeds")
    if len(recs) != 12:
        print("  INCOMPLETE: expected 12. Verdicts below are provisional.")

    # ---- instrument check 0: was the pretraining objective learned at all? ----
    move_acc = [r["pretrain_final_move_accuracy"] for r in recs]
    print(f"\npretraining move-naming accuracy: mean {st.mean(move_acc):.3f} "
          f"(chance {MOVE_CHANCE:.3f}, need >= {MOVE_LEARNED_MIN})")
    if st.mean(move_acc) < MOVE_LEARNED_MIN:
        print("  OBJECTIVE NOT LEARNED. The encoder never solved the inverse task, so the")
        print("  probe numbers below describe an essentially random encoder. Fix the")
        print("  pretraining (epochs, lr, capacity) before reading any bar.")

    # ---- the table ----
    print(f"\n{'depth':>6}{'chance':>9}{'facelets':>10}{'frozen':>9}{'trained':>9}"
          f"{'delta':>9}{'exact p':>9}")
    deltas_by_depth = {}
    for d in REPORT_DEPTHS:
        fro, tra = series(recs, "frozen", d), series(recs, "trained", d)
        fac = series(recs, "facelets", d)
        cha = series(recs, "frozen", d, "chance")
        if not fro or not tra:
            print(f"{d:>6}   (no records)")
            continue
        diffs = [t - f for t, f in zip(tra, fro)]
        p = permutation_p(diffs)
        deltas_by_depth[d] = (st.mean(diffs), p)
        print(f"{d:>6}{st.mean(cha):>9.3f}{st.mean(fac):>10.3f}{st.mean(fro):>9.3f}"
              f"{st.mean(tra):>9.3f}{st.mean(diffs):>+9.3f}{p:>9.4f}")

    # ---- Control B sanity check, before any verdict is read ----
    fro4 = series(recs, "frozen", PRIMARY_DEPTH)
    print(f"\nCONTROL B, external sanity check on the frozen arm:")
    print(f"  frozen depth-4 {st.mean(fro4):.3f} +-{sd(fro4):.3f} "
          f"vs EXP-033's {EXP033_FROZEN_D4} (tolerance +-{SANITY_TOLERANCE})")
    sane = abs(st.mean(fro4) - EXP033_FROZEN_D4) <= SANITY_TOLERANCE
    if sane:
        print("  WITHIN TOLERANCE. The batched pipeline reproduces EXP-033's frozen arm.")
    else:
        print("  OUTSIDE TOLERANCE. STOP. The batched pipeline is not measuring what EXP-033")
        print("  measured, and every number here inherits the discrepancy. Explain this first.")

    # ---- Bar 1 ----
    tra4 = series(recs, "trained", PRIMARY_DEPTH)
    m4, p4 = deltas_by_depth.get(PRIMARY_DEPTH, (None, None))
    print(f"\n{'=' * 78}")
    print(f"BAR 1 (PRIMARY), depth-4 trained vs frozen. Need >= {BAR1_DELTA} at p <= {ALPHA}.")
    if m4 is None:
        print("  cannot evaluate: no depth-4 records")
    else:
        wins = sum(1 for t, f in zip(tra4, fro4) if t > f)
        print(f"  {m4:+.4f}  W-L {wins}-{len(tra4) - wins}  exact p {p4:.4f}")
        if m4 >= BAR1_DELTA and p4 <= ALPHA:
            print("  CONFIRMED. Inverse-model pretraining improves optimal-move decodability")
            print("  at the frontier depth, by at least what a width doubling buys.")
        else:
            print("  REFUTED. Inverse dynamics is NOT sufficient to make optimality linearly")
            print("  readable. That is a finding about what 'learning how the cube works'")
            print("  buys; redirect Stage 2 to a value/heuristic objective rather than")
            print("  leaving the stage open-ended.")

    # ---- Bar 2 ----
    print(f"\nBAR 2 (THESIS), depth-4 trained > {BAR2_CEILING}, the raw-facelet linear ceiling.")
    if tra4:
        print(f"  trained {st.mean(tra4):.3f} vs ceiling {BAR2_CEILING}")
        if st.mean(tra4) > BAR2_CEILING:
            print("  CLEARED. A linear probe on the TRAINED concept beats any linear map on")
            print("  the observation, so the encoder is supplying genuine nonlinear structure.")
            print("  This is the first moment the SNN earns its place, and width provably")
            print("  cannot get here (concept@512 reaches 0.638).")
        else:
            print("  NOT CLEARED. The encoder has not yet overtaken what the raw observation")
            print("  supports linearly. Bar 1 passing does NOT imply this bar.")

    # ---- Bar 3 ----
    print("\nBAR 3, depth profile. Does the gain GROW with depth?")
    print("  (gains at depth 3 are least interesting: frozen already reaches ~0.631 there)")
    ordered = [(d, deltas_by_depth[d][0]) for d in REPORT_DEPTHS if d in deltas_by_depth]
    for d, m in ordered:
        print(f"    depth {d}: {m:+.4f}")
    if len(ordered) >= 2:
        growing = all(b >= a for (_, a), (_, b) in zip(ordered, ordered[1:]))
        deep = [m for d, m in ordered if d in ("5", "6")]
        if growing and deep and max(deep) > 0:
            print("  GROWS WITH DEPTH. Materially stronger than a uniform shift: the encoder")
            print("  helps most exactly where Wall 1 bites hardest.")
        elif deep and max(deep) <= 0:
            print("  NO GAIN AT DEPTH 5-6, which is where the policy actually fails")
            print("  (0.0396 and 0.0000). Any shallow gain is the least useful kind.")
        else:
            print("  NOT MONOTONE. Report the profile as measured; do not summarise it as a")
            print("  single number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
