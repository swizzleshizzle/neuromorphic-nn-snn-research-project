"""EXP-050 aggregator: apply the pre-registered rules to the records on disk.

WRITTEN BEFORE THE RUN WAS DISPATCHED. No record existed. Rules on disk before numbers.
Thresholds committed at f569540.

> THE GRID IS APPLIED MECHANICALLY, including the cell that would supersede the whole
> EXP-047/048/049 line (more pretraining beating RL fine-tuning).

> A NULL ON CLAIM 2 IS NOT EQUIVALENCE at n=12, and this file refuses to print it as one.

Usage:
    .venv/bin/python experiments/050_objective_vs_gradient/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM_A_DIR = HERE.parent / "043_cap_at_depth_5_6" / "outputs"
ARM_B_DIR = HERE.parent / "048_fresh_head" / "outputs"

BAR = 0.05
ALPHA = 0.05
DEPTH = 6
FROZEN_TRAINABLE = 390
TAGS = {"A": "exp043_capped_d6", "B": "exp048_freshhead_d6", "F": "exp050_pre2_d6"}


def permutation_p(diffs):
    n = len(diffs)
    obs = abs(sum(diffs))
    return sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12) / 2 ** n


def load(d: Path, tag: str) -> dict:
    out = {}
    for p in Path(d).glob("*.json"):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and r.get("tag") == tag and r.get("depth") == DEPTH:
            out[r["seed"]] = r
    return out


def paired(label, X, Y, seeds):
    d = [X[s]["success_rate"] - Y[s]["success_rate"] for s in seeds]
    delta, p = sum(d) / len(d), permutation_p(d)
    w = sum(1 for v in d if v > 0)
    l = sum(1 for v in d if v < 0)
    print(f"  {label}: {delta:+.4f}   W-L-T {w}-{l}-{len(d)-w-l}   exact p {p:.4f}")
    return delta, p, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    F = load(args.out_dir, TAGS["F"])
    A = load(ARM_A_DIR, TAGS["A"])
    B = load(ARM_B_DIR, TAGS["B"])
    if not F:
        raise SystemExit(f"no arm F records tagged {TAGS['F']} in {args.out_dir}")
    seeds = sorted(set(A) & set(B) & set(F))
    if not seeds:
        raise SystemExit("no seed present in all three arms")

    print("=" * 78)
    print(f"EXP-050: was it RL's OBJECTIVE, or just more gradient? depth {DEPTH}, n={len(seeds)}")
    print("=" * 78)

    counts = {r["trainable_params"] for r in F.values()}
    print(f"\nSANITY - arm F must be FROZEN")
    print(f"  trainable_params {sorted(counts)}  (expected [{FROZEN_TRAINABLE}])")
    if counts != {FROZEN_TRAINABLE}:
        raise SystemExit("arm F is not frozen; it answers a different question.")

    a = [A[s]["success_rate"] for s in seeds]
    b = [B[s]["success_rate"] for s in seeds]
    f = [F[s]["success_rate"] for s in seeds]
    print(f"\nTHE THREE ARMS - identical RL compute, differing only in the encoder")
    print(f"  A  E0  40 epochs                        {st.mean(a):.4f}  sd {st.pstdev(a):.4f}")
    print(f"  F  E0+ 80 epochs                        {st.mean(f):.4f}  sd {st.pstdev(f):.4f}")
    print(f"  B  E1  40 epochs + 10,000 RL updates    {st.mean(b):.4f}  sd {st.pstdev(b):.4f}")

    print(f"\nCLAIM 1 (PRIMARY) - does MORE PRETRAINING help at all?")
    d1, p1, diffs1 = paired("F - A", F, A, seeds)
    c1 = "CONFIRMED" if (d1 >= BAR and p1 <= ALPHA) else "REFUTED"
    print(f"  bar: >= +{BAR} at p <= {ALPHA}   VERDICT: {c1}")
    print(f"  per-seed: " + "  ".join(f"{v:+.3f}" for v in diffs1))

    print(f"\nCLAIM 2 (DECISIVE) - does it MATCH RL fine-tuning?")
    d2, p2, _ = paired("F - B", F, B, seeds)
    fb_sig = p2 <= ALPHA
    if not fb_sig:
        print(f"  NOT EVIDENCE OF EQUIVALENCE. n={len(seeds)} cannot show F and B are the same.")
        print(f"  The strongest honest sentence is 'F is not detectably different from B'.")

    print(f"\nCLAIM 3 - THE PRE-REGISTERED GRID")
    if c1 == "REFUTED" and d2 < 0 and fb_sig:
        print("  >> RL'S OBJECTIVE IS WHAT MATTERS. A step-matched control that was handed ~17x")
        print("     more data per update gets nothing, while RL fine-tuning gets +0.13. The")
        print("     'more gradient' alternative is CLOSED, and EXP-047/048/049 can lift their")
        print("     scope caveats rather than restate them.")
    elif c1 == "CONFIRMED" and d2 > 0 and fb_sig:
        print("  >> MORE PRETRAINING BEATS RL FINE-TUNING. The EXP-047/048/049 line is")
        print("     superseded by something OFFLINE and far cheaper. EXP-040 did record that")
        print("     the objective had not saturated.")
    elif c1 == "CONFIRMED" and not fb_sig:
        print("  >> IT WAS JUST MORE GRADIENT. This substantially deflates EXP-047/048/049, and")
        print("     the practical consequence is to stop RL fine-tuning and pretrain longer,")
        print("     which is offline and cheaper. (Claim 2 null is not equivalence - but a")
        print("     confirmed Claim 1 with no detectable gap is the deflationary reading.)")
    elif c1 == "CONFIRMED":
        print("  >> BOTH CONTRIBUTE. Generic gradient is worth (F - A); the objective is worth")
        print("     (B - F). Report the split.")
    else:
        print("  >> More pretraining did not clear the bar and F is not detectably below B.")
        print("     UNDERPOWERED for the decisive comparison - report as such, do not pick a cell.")

    print(f"\nCLAIM 4 - does the probe's anti-correlation belong to the RL objective?")
    print(f"  EXP-039: pretraining RAISES the probe (+0.3396, 12-0).")
    print(f"  EXP-049: RL fine-tuning LOWERS it (0-12, p 0.0005) while success nearly doubles.")
    print(f"  Predicted: E0+ probes HIGHER than E0 while arm F gains LESS policy than arm B.")
    print(f"  run: .venv/bin/python experiments/050_objective_vs_gradient/probe_e0plus.py")

    print(f"\nSCOPE - what this still does not control for")
    print(f"  The pretraining objective is INVERSE DYNAMICS specifically. A refuted Claim 1 says")
    print(f"  THIS objective adds nothing further, not that no supervised objective could.")
    print("=" * 78)


if __name__ == "__main__":
    main()
