"""EXP-070 aggregator. Gates and the unresolved band are VERDICTS (the EXP-068 rule).

Spec: docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md, section 2.5.

Usage: .venv/bin/python experiments/070_lookahead_existing/aggregate.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Set ONLY in the dated spec-amendment commit after pre-flight step 1. None blocks every verdict.
GATE0_FORM: str | None = "wilson"  # set 2026-10-06 by the pre-launch spec amendment
ALPHA = 0.05
FLOOR = 0.02
CEILING = 0.98
CLAIM2_BAR = 0.10
PRIMARY = (8, 3)
SECONDARY = ((8, 2), (9, 2), (9, 3))


def one_sided_p(diffs) -> float:
    """Exact one-sided paired sign-flip test: P(flipped sum >= observed sum)."""
    n, obs = len(diffs), sum(diffs)
    hits = sum(1 for s in itertools.product((1, -1), repeat=n)
               if sum(x * y for x, y in zip(s, diffs)) >= obs - 1e-12)
    return hits / 2 ** n


def gate1_verdict(e_mean: float, p_mean: float) -> str:
    if (e_mean < FLOOR and p_mean < FLOOR) or (e_mean > CEILING and p_mean > CEILING):
        return "UNRESOLVED"
    return "RESOLVED"


def claim1_verdict(diffs, e_mean: float, p_mean: float) -> tuple[str, float]:
    p = one_sided_p(diffs)
    if gate1_verdict(e_mean, p_mean) == "UNRESOLVED":
        return "UNRESOLVED", p
    if st.mean(diffs) <= 0:
        return "REFUTED", p
    if p < ALPHA:
        return "CONFIRMED", p
    return "NOT SIGNIFICANT", p


def wilson95(successes: int, n: int) -> tuple[float, float]:
    z = 1.959964
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def gate0_verdict(form, g_counts: dict, published: dict, determinism_ok: bool) -> str:
    if form is None:
        raise SystemExit(
            "GATE0_FORM is unset. Run the laptop pre-flight (step 1) and amend the spec "
            "before any verdict can be read."
        )
    if not determinism_ok:
        return "FAIL"
    if form == "exact":
        return "PASS" if all(g_counts[c] == published[c]["solved"] for c in published) else "FAIL"
    if form == "wilson":
        by_depth = defaultdict(lambda: [0, 0, 0])
        for cell, pub in published.items():
            d = cell.split("_")[0]
            by_depth[d][0] += pub["solved"]
            by_depth[d][1] += pub["n"]
            by_depth[d][2] += g_counts[cell]
        for pub_solved, n, g_solved in by_depth.values():
            lo, hi = wilson95(pub_solved, n)
            if not lo <= g_solved / n <= hi:
                return "FAIL"
        return "PASS"
    raise SystemExit(f"unknown GATE0_FORM {form!r}")


def load(out_dir: Path) -> dict:
    recs = {}
    for p in sorted(out_dir.glob("exp070_*.json")):
        r = json.loads(p.read_text())
        if r.get("limit_states") is not None:
            continue  # smoke records never enter a verdict
        recs[(r["mode"], r["k"], r["depth"], r["seed"])] = r
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="set only after re-running a sample of cells and diffing records")
    args = ap.parse_args()
    recs = load(args.out_dir)
    published = json.loads((HERE / "published_counts.json").read_text())
    seeds = range(12)

    def rate(mode, k, d, s):
        return recs[(mode, k, d, s)]["success_rate"]

    print("EXP-070 mean held-out success")
    arms = [("G", 0), ("E", 1), ("E", 2), ("E", 3), ("P", 2), ("P", 3), ("R", 1), ("R", 2), ("R", 3)]
    for d in (7, 8, 9):
        row = []
        for m, k in arms:
            vals = [rate(m, k, d, s) for s in seeds if (m, k, d, s) in recs]
            row.append(f"{m}{k} {st.mean(vals):.4f} (n={len(vals)})" if vals else f"{m}{k} -")
        print(f"  d{d}: " + "  ".join(row))

    g_counts = {f"d{d}_s{s}": recs[("G", 0, d, s)]["solved"] for d in (7, 8, 9) for s in seeds}
    g0 = gate0_verdict(GATE0_FORM, g_counts, published, args.determinism_ok)
    print(f"\nGATE 0 ({GATE0_FORM}): {g0}")
    if g0 != "PASS":
        print("Gate 0 did not pass. No claim may be read.")
        return

    for label, cellset in (("PRIMARY", (PRIMARY,)), ("secondary", SECONDARY)):
        for d, k in cellset:
            diffs = [rate("P", k, d, s) - rate("E", k, d, s) for s in seeds]
            e_mean = st.mean(rate("E", k, d, s) for s in seeds)
            p_mean = st.mean(rate("P", k, d, s) for s in seeds)
            v, p = claim1_verdict(diffs, e_mean, p_mean)
            print(f"CLAIM 1 {label} d{d} k{k}: P {p_mean:.4f} vs E {e_mean:.4f}, "
                  f"diff {st.mean(diffs):+.4f}, p {p:.4f} -> {v}")

    p3_d9 = st.mean(rate("P", 3, 9, s) for s in seeds)
    print(f"CLAIM 2 d9 P3 {p3_d9:.4f} vs bar {CLAIM2_BAR}: "
          f"{'MET' if p3_d9 >= CLAIM2_BAR else 'NOT MET'} (practical only; read with Claim 1)")
    print("CLAIM 3 revisit rate (no threshold):")
    for d in (7, 8, 9):
        print(f"  d{d}: " + "  ".join(
            f"{m}{k} {st.mean(recs[(m, k, d, s)]['eval_revisit_rate'] for s in seeds):.3f}"
            for m, k in arms if all((m, k, d, s) in recs for s in seeds)))


if __name__ == "__main__":
    main()
