"""EXP-071 aggregator. Four gates and two primaries at alpha 0.025, split family-wise with 0.05.

Spec: docs/superpowers/specs/2026-10-06-exp071-critic-and-no-revisit-design.md, sections 6-7.

Usage: .venv/bin/python experiments/071_critic_and_no_revisit/aggregate.py [--out-dir DIR] [--determinism-ok]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

_spec = importlib.util.spec_from_file_location(
    "exp071_cells", HERE / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)

_a70_spec = importlib.util.spec_from_file_location(
    "exp070_aggregate", REPO / "experiments" / "070_lookahead_existing" / "aggregate.py")
a70 = importlib.util.module_from_spec(_a70_spec)
_a70_spec.loader.exec_module(a70)

one_sided_p = a70.one_sided_p
gate1_verdict = a70.gate1_verdict

# Set ONLY in the dated spec-amendment commit after step 0 (laptop pre-flight). None blocks
# every verdict. The controller sets these, never code.
GATE_R1_PASSED: bool | None = None
GATE_R3_PASSED: bool | None = None

ALPHA = 0.025
GATE_R_ALPHA = 0.05
V_RATIO = 0.5
OUTCOME_FIELDS = ("solved", "n", "success_rate", "mean_steps", "optimality",
                  "eval_revisit_rate", "greedy_modal_action_frac", "goal_fired_frac")
SEEDS = range(12)


def gate0b_verdict(recs071: dict, recs070: dict) -> str:
    """Every (arm, seed) key in recs071 must exist in recs070 with equal OUTCOME_FIELDS."""
    for key, rec071 in recs071.items():
        rec070 = recs070.get(key)
        if rec070 is None:
            return "FAIL"
        for field in OUTCOME_FIELDS:
            if rec071[field] != rec070[field]:
                return "FAIL"
    return "PASS"


def gate_r_verdict(rank_rows: list[dict], hit_key: str = "critic_hit",
                    chance_key: str = "chance") -> tuple[bool, float]:
    diffs = [row[hit_key] - row[chance_key] for row in rank_rows]
    p = one_sided_p(diffs)
    return (p < GATE_R_ALPHA and st.mean(diffs) > 0), p


def gate_v_verdict(g0v_revisit: float, g0_revisit: float) -> bool:
    return g0v_revisit <= V_RATIO * g0_revisit


def claim_verdict(diffs, a_mean: float, b_mean: float, gate_ok: bool) -> tuple[str, float]:
    p = one_sided_p(diffs)
    if not gate_ok:
        return "VOID", p
    if gate1_verdict(a_mean, b_mean) == "UNRESOLVED":
        return "UNRESOLVED", p
    if st.mean(diffs) <= 0:
        return "REFUTED", p
    if p < ALPHA:
        return "CONFIRMED", p
    return "NOT SIGNIFICANT", p


def load_071(out_dir: Path) -> dict:
    recs = {}
    for p in sorted(out_dir.glob("exp071_*.json")):
        if p.name.startswith("exp071_rank_"):
            continue  # rank.py writes its per-seed records into this same directory
        r = json.loads(p.read_text())
        if r.get("limit_states") is not None:
            continue
        recs[(r["arm"], r["seed"])] = r
    return recs


def load_070(out_dir: Path) -> dict:
    recs = {}
    for p in sorted(out_dir.glob("exp070_*.json")):
        r = json.loads(p.read_text())
        if r.get("limit_states") is not None:
            continue
        if r.get("depth") != cells.DEPTH:
            continue
        recs[(f"{r['mode']}{r['k']}", r["seed"])] = r
    return recs


def load_rank(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(out_dir.glob(f"exp071_rank_d{cells.DEPTH}_s*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="set only after re-running Gate 0(a)'s determinism cells and diffing")
    args = ap.parse_args()

    recs071 = load_071(args.out_dir)
    recs070 = load_070(REPO / "experiments" / "070_lookahead_existing" / "outputs")
    rank_rows = load_rank(args.out_dir)

    def rate(arm, seed, field="success_rate"):
        return recs071[(arm, seed)][field]

    print("EXP-071 mean held-out success")
    arms = [cells.arm_name(*a) for a in cells.ARMS]
    row = []
    for arm in arms:
        vals = [rate(arm, s) for s in SEEDS if (arm, s) in recs071]
        row.append(f"{arm} {st.mean(vals):.4f} (n={len(vals)})" if vals else f"{arm} -")
    print("  " + "  ".join(row))

    print("EXP-071 mean eval_revisit_rate")
    row = []
    for arm in arms:
        vals = [rate(arm, s, "eval_revisit_rate") for s in SEEDS if (arm, s) in recs071]
        row.append(f"{arm} {st.mean(vals):.4f}" if vals else f"{arm} -")
    print("  " + "  ".join(row))

    # Gate 0(b): continuity with EXP-070 for the plain arms, all 4 arms x 12 seeds.
    plain071 = {k: v for k, v in recs071.items() if k[0] in ("G0", "E3", "P3", "R3")}
    expected = 4 * len(SEEDS)
    if len(plain071) != expected:
        print(f"\nGATE 0(b): only {len(plain071)}/{expected} plain-arm cells on disk. "
              "No claim may be read until all are present.")
        return
    g0b = gate0b_verdict(plain071, recs070)
    print(f"\nGATE 0(b) (continuity with EXP-070): {g0b}")
    if g0b != "PASS":
        print("Gate 0(b) did not pass. No claim may be read.")
        return

    if not args.determinism_ok:
        print("\nGate 0(a) determinism not confirmed (pass --determinism-ok). No claim may be read.")
        return

    if GATE_R1_PASSED is None or GATE_R3_PASSED is None:
        raise SystemExit(
            "GATE_R1_PASSED / GATE_R3_PASSED is unset. Run the laptop pre-flight (step 0, rank "
            "phase) and amend the spec with the dated gate commitment before any verdict can "
            "be read."
        )

    if not rank_rows:
        raise SystemExit("no rank records found; run the rank phase before aggregating.")

    r1_ok, r1_p = gate_r_verdict(rank_rows)
    r3_ok, r3_p = gate_r_verdict(rank_rows, "leaf_closer_hit", "leaf_closer_chance")
    print(f"\nGATE R1 (child-level critic above chance): {r1_ok} (p={r1_p:.4f})")
    print(f"GATE R3 (leaf-level critic above chance): {r3_ok} (p={r3_p:.4f})")
    if r1_ok != GATE_R1_PASSED or r3_ok != GATE_R3_PASSED:
        raise SystemExit(
            f"recomputed R1/R3 ({r1_ok}/{r3_ok}) disagree with the committed constants "
            f"(GATE_R1_PASSED={GATE_R1_PASSED}, GATE_R3_PASSED={GATE_R3_PASSED}). "
            "Refusing to aggregate."
        )

    g0v_revisit = st.mean(rate("G0V", s, "eval_revisit_rate") for s in SEEDS)
    g0_revisit = st.mean(rate("G0", s, "eval_revisit_rate") for s in SEEDS)
    gate_v = gate_v_verdict(g0v_revisit, g0_revisit)
    print(f"\nGATE V (no-revisit engaged): {gate_v} (G0V {g0v_revisit:.4f} vs G0 {g0_revisit:.4f})")

    def diffs_for(arm_a, arm_b, field="success_rate"):
        return [rate(arm_a, s, field) - rate(arm_b, s, field) for s in SEEDS]

    c3_mean = st.mean(rate("C3", s) for s in SEEDS)
    e3_mean = st.mean(rate("E3", s) for s in SEEDS)
    v1, p1 = claim_verdict(diffs_for("C3", "E3"), c3_mean, e3_mean, gate_ok=GATE_R3_PASSED)
    print(f"\nCLAIM 1 (primary, critic track) C3 - E3: C3 {c3_mean:.4f} vs E3 {e3_mean:.4f}, "
          f"diff {c3_mean - e3_mean:+.4f}, p {p1:.4f} -> {v1}")

    g0v_mean = st.mean(rate("G0V", s) for s in SEEDS)
    g0_mean = st.mean(rate("G0", s) for s in SEEDS)
    v2, p2 = claim_verdict(diffs_for("G0V", "G0"), g0v_mean, g0_mean, gate_ok=gate_v)
    print(f"CLAIM 2 (primary, no-revisit track) G0V - G0: G0V {g0v_mean:.4f} vs G0 {g0_mean:.4f}, "
          f"diff {g0v_mean - g0_mean:+.4f}, p {p2:.4f} -> {v2}")

    print("\nSecondary (reported with p-values, read as a pattern, never as confirmations):")
    for a, b, gate_ok in (("C3", "P3", GATE_R3_PASSED), ("C3V", "C3", GATE_R3_PASSED),
                          ("C1", "G0", GATE_R1_PASSED), ("E3V", "E3", True), ("P3V", "P3", True)):
        a_mean = st.mean(rate(a, s) for s in SEEDS)
        b_mean = st.mean(rate(b, s) for s in SEEDS)
        v, p = claim_verdict(diffs_for(a, b), a_mean, b_mean, gate_ok=gate_ok)
        print(f"  {a} - {b}: {a} {a_mean:.4f} vs {b} {b_mean:.4f}, diff {a_mean - b_mean:+.4f}, "
              f"p {p:.4f} -> {v}")

    print("\nFallback fractions (V arms):")
    v_arms = [a for a in arms if a.endswith("V")]
    row = []
    for arm in v_arms:
        vals = [rate(arm, s, "fallback_frac") for s in SEEDS if (arm, s) in recs071]
        row.append(f"{arm} {st.mean(vals):.4f}" if vals else f"{arm} -")
    print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
