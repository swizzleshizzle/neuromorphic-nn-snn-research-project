"""EXP-051 aggregator: apply the pre-registered rules to the records on disk.

Thresholds committed at 520d7ab, before any number existed.

> THE PROBE IS DELIBERATELY ABSENT. EXP-049 and EXP-050 between them showed it is anti-correlated
> with policy quality at depth 6 - down 0-12 while success doubled, up 12-0 while success halved,
> both at p 0.0005. Adding a probe number here would add a number nobody should use.

Usage:
    .venv/bin/python experiments/051_depth7_transfer/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE.parent / "044_depth7_frontier" / "outputs"

DEPTH = 7
BAR, ALPHA = 0.05, 0.05
TAG = "exp051_transfer_d7"
BASELINE_TAG = "exp044_d7_e10000"
D6_GAIN = 0.1312              # EXP-048, the gain being tested for transfer
COMPLETE_TRANSFER = 0.1933    # 0.0621 + 0.1312
EXP044_ARM_B = 0.1971         # what 4.4x the episodes bought at this depth
BUDGET_EQUIV = 0.1392         # 0.0621 + 0.210*log10(2.33)
D6_EXCESSES = (0.0628, 0.0504, 0.0540)   # arms C, B, E at depth 6
FROZEN_TRAINABLE = 390


def permutation_p(d):
    n, obs = len(d), abs(sum(d))
    return sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, d))) >= obs - 1e-12) / 2 ** n


def load(d: Path, tag: str) -> dict:
    out = {}
    for p in Path(d).glob("*.json"):
        r = json.loads(p.read_text())
        if (isinstance(r, dict) and r.get("tag") == tag and r.get("depth") == DEPTH
                and r.get("arm") == "regionalized"):
            out[r["seed"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    new = load(args.out_dir, TAG)
    base = load(BASELINE_DIR, BASELINE_TAG)
    if not new:
        raise SystemExit(f"no EXP-051 records tagged {TAG} in {args.out_dir}")
    seeds = sorted(set(new) & set(base))

    counts = {r["trainable_params"] for r in new.values()}
    print("=" * 78)
    print(f"EXP-051: does the encoder gain transfer to depth {DEPTH}? n={len(seeds)}")
    print("=" * 78)
    print(f"\nSANITY - frozen arm: trainable_params {sorted(counts)} (expected [{FROZEN_TRAINABLE}])")
    if counts != {FROZEN_TRAINABLE}:
        raise SystemExit("arm is not frozen")

    a = [base[s]["success_rate"] for s in seeds]
    b = [new[s]["success_rate"] for s in seeds]
    diffs = [y - x for x, y in zip(a, b)]
    delta, p = sum(diffs) / len(diffs), permutation_p(diffs)
    w = sum(1 for v in diffs if v > 0)
    l = sum(1 for v in diffs if v < 0)

    print(f"\n  EXP-044 arm A  E0 frozen   {st.mean(a):.4f}  sd {st.pstdev(a):.4f}")
    print(f"  EXP-051        E1 frozen   {st.mean(b):.4f}  sd {st.pstdev(b):.4f}")

    print(f"\nCLAIM 1 (PRIMARY) - does it transfer to a depth it never trained on?")
    print(f"  delta {delta:+.4f}   W-L-T {w}-{l}-{len(diffs)-w-l}   exact p {p:.4f}")
    verdict = "CONFIRMED" if (delta >= BAR and p <= ALPHA) else "REFUTED"
    print(f"  bar: >= +{BAR} at p <= {ALPHA}   VERDICT: {verdict}")
    print(f"  per-seed: " + "  ".join(f"{v:+.3f}" for v in diffs))

    print(f"\nCLAIM 2 - the point prediction")
    frac = delta / D6_GAIN
    print(f"  depth-6 gain was +{D6_GAIN}; complete transfer predicts {COMPLETE_TRANSFER}")
    print(f"  observed {st.mean(b):.4f}   TRANSFER FRACTION {frac:.2f}  (1.00 = complete)")

    print(f"\nCLAIM 3 - does the encoder buy at the frontier what BUDGET buys?")
    print(f"  EXP-044 arm B reached {EXP044_ARM_B} at this depth, at 4.4x THE EPISODES.")
    print(f"  EXP-051 reached {st.mean(b):.4f} at 1x the episodes.")
    if abs(st.mean(b) - EXP044_ARM_B) < 0.03:
        print(f"  >> THEY LAND TOGETHER. The encoder buys at the frontier what 4.4x budget buys,")
        print(f"     for a quarter of the episodes.")

    print(f"\nCLAIM 4 - does the constant-return model hold across DEPTH?")
    excess = st.mean(b) - BUDGET_EQUIV
    print(f"  budget-equivalent at 2.33x: {BUDGET_EQUIV}")
    print(f"  excess {excess:+.4f}   against depth 6's " +
          "  ".join(f"{v:+.4f}" for v in D6_EXCESSES))
    if 0.03 <= excess <= 0.08:
        print(f"  >> CONSISTENT. Constant returns hold across depth as well as across rounds.")

    print(f"\nCLAIM 5 - mechanism, via the instrument that replaced the probe")
    for k, want in (("eval_revisit_rate", "lower"), ("optimality", "higher")):
        x = [base[s][k] for s in seeds]
        y = [new[s][k] for s in seeds]
        d = [q - r for r, q in zip(x, y)]
        print(f"  {k:20s} {st.mean(x):.4f} -> {st.mean(y):.4f}  {st.mean(d):+.4f}  "
              f"p {permutation_p(d):.4f}   (predicted {want})")

    if verdict == "REFUTED":
        print(f"\nCLAIM 6 - the null: the gain would be DEPTH-SPECIFIC, bounding EXP-047/048/049")
        print(f"  to depth 6 and redirecting toward why - most likely shell-specific fitting from")
        print(f"  the (1..6) curriculum cap.")
    print("=" * 78)


if __name__ == "__main__":
    main()
