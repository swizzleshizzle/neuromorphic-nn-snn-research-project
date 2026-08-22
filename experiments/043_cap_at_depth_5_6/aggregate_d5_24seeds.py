"""EXP-043 Claim 1 at depth 5, re-asked with 24 seeds instead of 12.

WHY THIS EXISTS. EXP-043's Claim 1 at depth 5 measured **+0.1108 at p 0.0815** and was recorded
**REFUTED**, because the pre-registered rule was ">= +0.05 at p <= 0.05" and 0.0815 misses it.
Its RESULTS.md said so plainly:

    "Depth 5 has the LARGER effect and still fails the rule. Report it as refuted. +0.1108 is
     more than double the 0.05 bar. It misses on p, not on effect size... It is also not a null.
     A follow-up with more seeds is well motivated, and is the one place in this experiment
     where more data would clearly change the answer."

This is that follow-up. Seeds 12-23 were run for BOTH arms on 2026-08-21/22, because Claim 1 is
a PAIRED delta and EXP-040 phase 2 only ever ran seeds 0-11 - so the new seeds had no baseline
to pair against until now.

> [!important] THE BAR IS NOT RE-NEGOTIATED. `>= +0.05 at p <= 0.05`, exactly as EXP-043
> pre-registered it. n changes; nothing else does. Lowering a bar after seeing 0.0815 would be
> precisely the move EXP-043's own RESULTS.md refused to make.

> [!warning] THIS IS NOT A CLEAN PRE-REGISTRATION AND MUST NOT BE PRESENTED AS ONE.
> The 12 original seeds were seen before the extra 12 were commissioned, and the decision to
> extend was made BECAUSE the p-value was near the bar. That is a real form of optional
> stopping. What protects it: the bar is unchanged, n was fixed in advance at 24 (not "keep
> adding seeds until it passes"), and this file reports the NEW 12 seeds separately so the
> replication can be judged on its own. Report all three numbers, always.

Usage:
    .venv/bin/python experiments/043_cap_at_depth_5_6/aggregate_d5_24seeds.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPPED = HERE / "outputs"
BASELINE = HERE.parent / "040_pretrained_encoder_policy" / "outputs"

# --- EXP-043's pre-registered rule, unchanged. ---
CLAIM1_DELTA = 0.05
ALPHA = 0.05
DEPTH = 5
ORIGINAL_SEEDS = set(range(12))
EXP043_ORIGINAL = {"delta": 0.1108, "p": 0.0815, "wlt": "8-4-0"}


def permutation_p(diffs: list[float]) -> float:
    """Exact paired permutation over all 2**n sign flips, two-sided.

    n=24 is 16,777,216 flips. That is ~30 s of pure Python and still exact, so it stays exact
    rather than sampling: this project has no scipy and an approximation here would be a third
    thing to defend in a result that already needs care.
    """
    n = len(diffs)
    if not 1 <= n <= 24:
        raise ValueError(f"exact permutation needs 1 <= n <= 24, got {n}")
    observed = abs(sum(diffs))
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n)
               if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12)
    return hits / 2 ** n


def load(d: Path, prefix: str) -> dict:
    out = {}
    for p in sorted(d.glob(f"{prefix}*.json")):
        r = json.loads(p.read_text())
        if not isinstance(r, dict):
            continue
        if r.get("depth") == DEPTH and r.get("arm") == "regionalized":
            out[r["seed"]] = r
    return out


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def report(name: str, seeds: list[int], capped: dict, base: dict) -> dict:
    cap = [capped[s]["success_rate"] for s in seeds]
    bas = [base[s]["success_rate"] for s in seeds]
    diffs = [a - b for a, b in zip(cap, bas)]
    delta = sum(diffs) / len(diffs)
    p = permutation_p(diffs)
    w = sum(1 for x in diffs if x > 0)
    lose = sum(1 for x in diffs if x < 0)
    verdict = "CONFIRMED" if (delta >= CLAIM1_DELTA and p <= ALPHA) else "REFUTED"
    print(f"\n{name}  (n = {len(seeds)})")
    print(f"  capped   mean {st.mean(cap):.4f}  sd {sd(cap):.4f}")
    print(f"  baseline mean {st.mean(bas):.4f}  sd {sd(bas):.4f}")
    print(f"  paired delta {delta:+.4f}   W-L-T {w}-{lose}-{len(diffs)-w-lose}   exact p {p:.4f}")
    print(f"  VERDICT (>= +{CLAIM1_DELTA} at p <= {ALPHA}): {verdict}")
    return {"n": len(seeds), "delta": delta, "p": p, "verdict": verdict, "diffs": diffs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capped-dir", type=Path, default=CAPPED)
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE)
    args = ap.parse_args()

    capped = load(args.capped_dir, "exp043_capped_d5_")
    base = load(args.baseline_dir, "exp040_pre_d5_")
    seeds = sorted(set(capped) & set(base))
    if not seeds:
        raise SystemExit("no paired depth-5 seeds found in both arms")

    new_seeds = [s for s in seeds if s not in ORIGINAL_SEEDS]
    old_seeds = [s for s in seeds if s in ORIGINAL_SEEDS]

    print("=" * 78)
    print(f"EXP-043 Claim 1 at depth {DEPTH}, re-asked at n={len(seeds)}")
    print(f"EXP-043 published: {EXP043_ORIGINAL['delta']:+.4f} at p {EXP043_ORIGINAL['p']:.4f} "
          f"({EXP043_ORIGINAL['wlt']}) -> REFUTED on p, not on effect size")
    print(f"Bar UNCHANGED: >= +{CLAIM1_DELTA} at p <= {ALPHA}")
    print("=" * 78)

    r_old = report("ORIGINAL 12 SEEDS (EXP-043's own, recomputed here)", old_seeds, capped, base)
    r_new = report("REPLICATION - THE NEW 12 SEEDS, judged on their own", new_seeds, capped, base)
    r_all = report("ALL 24 SEEDS - the headline", seeds, capped, base)

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print(f"  The replication alone ({r_new['delta']:+.4f}, p {r_new['p']:.4f}) is the honest test of")
    print( "  whether the original effect was real, because those seeds were commissioned before")
    print( "  any of their numbers existed and none of them influenced the decision to run.")
    print(f"  The 24-seed figure ({r_all['delta']:+.4f}, p {r_all['p']:.4f}) is the best estimate, but it")
    print( "  INCLUDES the seeds whose near-miss motivated the extension. Report all three.")
    if r_new["verdict"] != r_all["verdict"]:
        print( "  !! The replication and the pooled result DISAGREE. The replication is the one that")
        print( "     speaks to whether the effect is real. Lead with it.")
    print("=" * 78)


if __name__ == "__main__":
    main()
