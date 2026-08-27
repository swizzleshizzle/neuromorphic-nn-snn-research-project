"""EXP-052 aggregator: apply the pre-registered rules to the records on disk.

Thresholds committed at 3fcf21e, before any number existed.

> THE PRIMARY IS 20-vs-40 ONLY. The 10-epoch arm is EXPLORATORY and carries no bar. Testing both
> against 40 and reporting the winner would inflate the false-positive rate.

> THE PROBE IS DELIBERATELY ABSENT (EXP-049/050: it moves opposite to policy quality here).

Usage:
    .venv/bin/python experiments/052_pretraining_optimum/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM_40_DIR = HERE.parent / "043_cap_at_depth_5_6" / "outputs"
ARM_80_DIR = HERE.parent / "050_objective_vs_gradient" / "outputs"

BAR, ALPHA = 0.05, 0.05
DEPTH = 6
PRIMARY_EPOCHS = 20
FROZEN_TRAINABLE = 390
ZERO_EPOCH_POLICY = 0.0000      # EXP-036, random frozen encoder at depth 6
MOVE_ACC = {10: 0.383, 20: 0.414, 40: 0.437, 80: 0.452}   # phase-1 logs / EXP-040 / EXP-050


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

    arms = {
        10: load(args.out_dir, "exp052_e10_d6"),
        20: load(args.out_dir, "exp052_e20_d6"),
        40: load(ARM_40_DIR, "exp043_capped_d6"),
        80: {s: r for s, r in load(ARM_80_DIR, "exp050_pre2_d6").items()},
    }
    for e in (10, 20):
        if not arms[e]:
            raise SystemExit(f"no records for the {e}-epoch arm in {args.out_dir}")
    seeds = sorted(set.intersection(*(set(a) for a in arms.values())))

    print("=" * 78)
    print(f"EXP-052: where is the pretraining optimum? depth {DEPTH}, n={len(seeds)}")
    print("=" * 78)

    for e in (10, 20):
        got = {r["trainable_params"] for r in arms[e].values()}
        if got != {FROZEN_TRAINABLE}:
            raise SystemExit(f"{e}-epoch arm is not frozen: {got}")
    print(f"\nSANITY - every arm frozen at {FROZEN_TRAINABLE} trainable parameters. OK.")

    print(f"\nTHE CURVE  (identical RL compute across every arm)")
    print(f"  {'epochs':>7}  {'move-acc':>9}  {'policy':>8}  {'sd':>7}")
    print(f"  {0:>7}  {'-':>9}  {ZERO_EPOCH_POLICY:8.4f}  {'-':>7}    EXP-036, random encoder")
    best_e, best_v = None, -1.0
    for e in (10, 20, 40, 80):
        v = [arms[e][s]["success_rate"] for s in seeds]
        m = st.mean(v)
        note = ""
        if e == 40:
            note = "    EXP-043"
        elif e == 80:
            note = "    EXP-050, warm-started"
        print(f"  {e:>7}  {MOVE_ACC[e]:9.3f}  {m:8.4f}  {st.pstdev(v):7.4f}{note}")
        if m > best_v:
            best_e, best_v = e, m

    print(f"\nCLAIM 1 (PRIMARY, single comparison) - is 40 epochs past the optimum?")
    d = [arms[PRIMARY_EPOCHS][s]["success_rate"] - arms[40][s]["success_rate"] for s in seeds]
    delta, p = sum(d) / len(d), permutation_p(d)
    w = sum(1 for v in d if v > 0)
    l = sum(1 for v in d if v < 0)
    print(f"  20 - 40: {delta:+.4f}   W-L-T {w}-{l}-{len(d)-w-l}   exact p {p:.4f}")
    verdict = "CONFIRMED" if (delta >= BAR and p <= ALPHA) else "REFUTED"
    print(f"  bar: >= +{BAR} at p <= {ALPHA}   VERDICT: {verdict}")
    print(f"  per-seed: " + "  ".join(f"{v:+.3f}" for v in d))
    if verdict == "CONFIRMED":
        print(f"  >> A CORRECTION TO THE WHOLE SERIES: every frozen-encoder result since EXP-040")
        print(f"     started from a needlessly over-trained encoder.")

    print(f"\n  (exploratory, NO BAR) 10 - 40: ", end="")
    d10 = [arms[10][s]["success_rate"] - arms[40][s]["success_rate"] for s in seeds]
    print(f"{sum(d10)/len(d10):+.4f}   p {permutation_p(d10):.4f}")

    # CLAIM 2 MUST BE READ THROUGH SIGNIFICANCE, NOT MEAN ORDERING.
    # Corrected 2026-08-27, after the first version declared "monotone decreasing, peak below 10"
    # purely from the ordering of four means - while 10, 20 and 40 are pairwise indistinguishable
    # (p 0.69, 0.49, 0.84). Ranking noise is exactly the over-read this project refuses. No
    # threshold changed; this only stops the interpreter asserting a shape the data lacks.
    print(f"\nCLAIM 2 - THE CURVE SHAPE, read through significance")
    pairs = [(a, b) for a, b in ((10, 20), (10, 40), (20, 40), (40, 80))]
    sig = {}
    for a, b in pairs:
        dd = [arms[a][s]["success_rate"] - arms[b][s]["success_rate"] for s in seeds]
        sig[(a, b)] = (sum(dd) / len(dd), permutation_p(dd))
        print(f"  {a:>2} vs {b:>2}: {sig[(a, b)][0]:+.4f}  p {sig[(a, b)][1]:.4f}"
              f"{'  *' if sig[(a, b)][1] <= ALPHA else ''}")
    plateau = [pr for pr in ((10, 20), (10, 40), (20, 40)) if sig[pr][1] > ALPHA]
    if len(plateau) == 3 and sig[(40, 80)][1] <= ALPHA:
        print("  >> A PLATEAU FROM 10 TO 40, THEN A COLLAPSE BY 80. None of 10/20/40 differ")
        print("     from each other; only 40-vs-80 does. So pretraining SATURATES EARLY for")
        print("     policy purposes: 10 epochs buys everything 40 does, at a quarter of the")
        print("     cost - but there is NO free performance sitting there, and no baseline moves.")
    elif best_e == 40:
        print("  >> The inherited value is at the top of the measured range.")
    else:
        print(f"  >> Highest measured mean is {best_e} epochs; check the pairwise tests above")
        print("     before describing any shape.")

    print(f"\nCLAIM 3 - does the pretraining metric predict policy? PREDICTED NO.")
    accs = [MOVE_ACC[e] for e in (10, 20, 40, 80)]
    pols = [st.mean([arms[e][s]["success_rate"] for s in seeds]) for e in (10, 20, 40, 80)]
    mono_acc = all(x < y for x, y in zip(accs, accs[1:]))
    mono_pol = all(x < y for x, y in zip(pols, pols[1:]))
    print(f"  move-accuracy monotone increasing: {mono_acc}   ({' -> '.join(f'{a:.3f}' for a in accs)})")
    print(f"  policy monotone increasing:        {mono_pol}   ({' -> '.join(f'{v:.4f}' for v in pols)})")
    if mono_acc and not mono_pol:
        print("  >> CONFIRMED. The pretraining objective's own metric says 80 epochs is best and")
        print("     is monotone throughout, while policy peaks and falls. ==You cannot tune")
        print("     pretraining length by watching pretraining.==")

    print(f"\nCLAIM 4 - trajectory metrics (the probe is deliberately not run)")
    for k in ("eval_revisit_rate", "optimality"):
        vals = "  ".join(f"e{e}={st.mean([arms[e][s][k] for s in seeds]):.4f}"
                         for e in (10, 20, 40, 80))
        print(f"  {k:20s} {vals}")
    print("=" * 78)


if __name__ == "__main__":
    main()
