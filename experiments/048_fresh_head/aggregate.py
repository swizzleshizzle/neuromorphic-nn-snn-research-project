"""EXP-048 aggregator: apply the pre-registered rules to the records on disk.

WRITTEN BEFORE THE RUN WAS DISPATCHED. No record existed. Rules on disk before numbers.

Thresholds are the ones committed at 652b517 in
docs/superpowers/specs/2026-08-23-exp048-fresh-head-design.md.

> THE INTERPRETATION GRID IS APPLIED MECHANICALLY. All four cells are in the spec, including
> the one that says "do not interpret this". That cell exists because a refuted Claim 1 paired
> with a null Claim 2 means arm B is indistinguishable from BOTH a 0.1800 arm and a 0.2700 arm,
> which is a statement about power, not about mechanism.

> A NULL ON CLAIM 2 IS NOT EVIDENCE OF EQUIVALENCE at n=12, and this file refuses to print it
> as one.

Usage:
    .venv/bin/python experiments/048_fresh_head/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM_A_DIR = HERE.parent / "043_cap_at_depth_5_6" / "outputs"
ARM_C_DIR = HERE.parent / "047_encoder_finetuning" / "outputs"

# --- pre-registered. Do not edit after data exists. ---
CLAIM1_DELTA = 0.05
ALPHA = 0.05
DEPTH = 6
SELECTED_LR = 1e-4
ARM_A_TAG = f"exp043_capped_d{DEPTH}"
ARM_B_TAG = f"exp048_freshhead_d{DEPTH}"
ARM_C_TAG = f"exp047_ft_d{DEPTH}_lr{SELECTED_LR:g}"
FROZEN_TRAINABLE = 390


def permutation_p(diffs: list[float]) -> float:
    """Exact paired permutation over all 2**n sign flips, two-sided."""
    n = len(diffs)
    if not 1 <= n <= 24:
        raise ValueError(f"exact permutation needs 1 <= n <= 24, got {n}")
    observed = abs(sum(diffs))
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n)
               if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12)
    return hits / 2 ** n


def load(d: Path, tag: str) -> dict:
    """Records for one cell, keyed by seed. Exact tag match, and non-record JSON skipped."""
    out = {}
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text())
        if not isinstance(r, dict):
            continue
        if r.get("tag") == tag and r.get("depth") == DEPTH and r.get("arm") == "regionalized":
            out[r["seed"]] = r
    return out


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def paired(name: str, x: dict, y: dict, seeds: list[int]) -> dict:
    a = [x[s]["success_rate"] for s in seeds]
    b = [y[s]["success_rate"] for s in seeds]
    diffs = [p - q for p, q in zip(a, b)]
    delta = sum(diffs) / len(diffs)
    p = permutation_p(diffs)
    w = sum(1 for v in diffs if v > 0)
    lose = sum(1 for v in diffs if v < 0)
    print(f"  {name}: {delta:+.4f}   W-L-T {w}-{lose}-{len(diffs)-w-lose}   exact p {p:.4f}")
    return {"delta": delta, "p": p, "diffs": diffs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    B = load(args.out_dir, ARM_B_TAG)
    A = load(ARM_A_DIR, ARM_A_TAG)
    C = {s: r for s, r in load(ARM_C_DIR, ARM_C_TAG).items() if s < 12}
    if not B:
        raise SystemExit(f"no EXP-048 records tagged {ARM_B_TAG} in {args.out_dir}")
    seeds = sorted(set(A) & set(B) & set(C))
    if not seeds:
        raise SystemExit("no seeds present in all three arms")

    a = [A[s]["success_rate"] for s in seeds]
    b = [B[s]["success_rate"] for s in seeds]
    c = [C[s]["success_rate"] for s in seeds]

    print("=" * 78)
    print(f"EXP-048: does EXP-047's gain survive a FRESH head? depth {DEPTH}, n={len(seeds)}")
    print("=" * 78)

    counts = {r["trainable_params"] for r in B.values()}
    print(f"\nSANITY - arm B must be FROZEN")
    print(f"  trainable_params {sorted(counts)}  (expected [{FROZEN_TRAINABLE}])")
    lrs = {r["config"].get("encoder_lr") for r in B.values()}
    print(f"  encoder_lr {sorted(lrs, key=lambda v: (v is not None, v))}  (expected [None])")
    if counts != {FROZEN_TRAINABLE} or lrs != {None}:
        raise SystemExit("arm B is NOT frozen. It has silently become EXP-047 and answers nothing.")

    print(f"\nTHE THREE ARMS")
    print(f"  A  EXP-040 encoder, frozen, fresh head   {st.mean(a):.4f}  sd {sd(a):.4f}")
    print(f"  B  EXP-047 encoder, FROZEN, fresh head   {st.mean(b):.4f}  sd {sd(b):.4f}")
    print(f"  C  EXP-047 encoder, trained, own head    {st.mean(c):.4f}  sd {sd(c):.4f}")

    print(f"\nCLAIM 1 (PRIMARY) - did the ENCODER itself get better?")
    r1 = paired("B - A", B, A, seeds)
    c1 = "CONFIRMED" if (r1["delta"] >= CLAIM1_DELTA and r1["p"] <= ALPHA) else "REFUTED"
    print(f"  bar: >= +{CLAIM1_DELTA} at p <= {ALPHA}   VERDICT: {c1}")
    print(f"  per-seed: " + "  ".join(f"{d:+.3f}" for d in r1["diffs"]))

    print(f"\nCLAIM 2 - is B below C?")
    r2 = paired("B - C", B, C, seeds)
    c2_sig = r2["p"] <= ALPHA and r2["delta"] < 0
    print(f"  significantly below C: {c2_sig}")
    if not c2_sig:
        print(f"  NOT EVIDENCE OF EQUIVALENCE. n={len(seeds)} cannot show B and C are the same.")
        print(f"  The strongest honest sentence is 'B is not detectably worse than C'.")

    print(f"\nCLAIM 3 (RETENTION) - descriptive, no p-value")
    gain_c = st.mean(c) - st.mean(a)
    gain_b = st.mean(b) - st.mean(a)
    if abs(gain_c) < 1e-9:
        print("  arm C shows no gain over A on these seeds; retention is undefined.")
    else:
        print(f"  (B - A) / (C - A) = {gain_b:+.4f} / {gain_c:+.4f} = {gain_b / gain_c:.2f}")
        print(f"  1.00 = the encoder carries all of EXP-047's gain; 0.00 = it was all co-adaptation")

    print(f"\nCLAIM 4 - THE PRE-REGISTERED INTERPRETATION GRID")
    if c1 == "CONFIRMED" and not c2_sig:
        print("  >> THE ENCODER GENUINELY IMPROVED. EXP-047's leak-free probe was the wrong")
        print("     instrument, or too insensitive at n=12. The probe's negative depth-6 delta")
        print("     still needs explaining.")
    elif c1 == "CONFIRMED" and c2_sig:
        print("  >> BOTH EFFECTS ARE REAL AND PARTIAL. The encoder improved AND the head")
        print("     co-adapted. Retention above is the split.")
    elif c1 == "REFUTED" and c2_sig:
        print("  >> CO-ADAPTATION. EXP-047's gain was in the pairing, not the encoder. This")
        print("     corroborates the leak-free probe (+0.0050, p 0.5732) and the memorisation")
        print("     reading of its depth profile. EXP-047's headline needs restating as")
        print("     'joint training beats training the head alone'.")
    else:
        print("  >> INCOHERENT - DO NOT INTERPRET. B is indistinguishable from both a 0.1800")
        print("     arm and a 0.2700 arm, which is a statement about POWER, not mechanism.")
        print("     Report as underpowered and say so.")

    print(f"\nSCOPE - what a confirmed Claim 1 does NOT license")
    print(f"  Arm B's encoder had 10,000 episodes of extra shaping that arm A's did not. This")
    print(f"  experiment cannot say whether RL's OBJECTIVE caused the improvement or merely more")
    print(f"  gradient steps of any kind. See spec section 4.")
    print("=" * 78)


if __name__ == "__main__":
    main()
