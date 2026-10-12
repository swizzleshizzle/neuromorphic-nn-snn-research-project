"""EXP-076 aggregator. Gates 0, D and 1 are VERDICTS: a claim whose gate failed prints VOID, a
contrast on the floor or ceiling prints UNRESOLVED, and a confirmed gain under 0.01 prints
CONFIRMED (SMALL). Gate G returns UNRESOLVED for the goal-check decomposition when the 4-move
goal check never fired. Gate R is reported, never voiding.

Spec: docs/superpowers/specs/2026-10-11-exp076-deeper-tree-design.md, sections 6 to 8.

The sign-flip test and Gate 1 are EXP-070's, Gate 0(b)'s field comparison and Gate R are
EXP-071's, seed pairing is EXP-072's, all reused through importlib. Alpha is 0.05 / 3.

Usage:
    .venv/bin/python experiments/076_deeper_tree/aggregate.py --pilot-report
    .venv/bin/python experiments/076_deeper_tree/aggregate.py --stage cont
    .venv/bin/python experiments/076_deeper_tree/aggregate.py --determinism-ok
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a70 = _load("exp070_aggregate", REPO / "experiments" / "070_lookahead_existing" / "aggregate.py")
a71 = _load("exp071_aggregate", REPO / "experiments" / "071_critic_and_no_revisit" / "aggregate.py")
a72 = _load("exp072_aggregate", REPO / "experiments" / "072_p3v_frontier" / "aggregate.py")
cells75 = _load("exp075_cells", REPO / "experiments" / "075_wide_region" / "cells.py")
ev = _load("exp076_evaluate", HERE / "evaluate.py")
one_sided_p = a70.one_sided_p

SEEDS = range(12)
DEPTHS = ev.DEPTHS
ALPHA = 0.05 / 3
SMALL = 0.01
N_UPDATES = 4000
PILOT_LIMIT = 20
NEW_ARMS = ("J3VG4-X", "J4V-X")
ALL_ARMS = ("J3V-X",) + NEW_ARMS

# Set by the dated post-pilot amendment (spec section 6): worker count and schedule. The
# launcher refuses every phase but `check` and `pilot` while this is None.
AMENDMENT: dict | None = None


def claim_verdict(diffs, a_mean: float, b_mean: float, gate_ok: bool) -> tuple[str, float]:
    p = one_sided_p(diffs)
    if not gate_ok:
        return "VOID", p
    if a70.gate1_verdict(a_mean, b_mean) == "UNRESOLVED":
        return "UNRESOLVED", p
    m = st.mean(diffs)
    if m <= 0:
        return "REFUTED", p
    if p < ALPHA:
        return ("CONFIRMED (SMALL)" if m < SMALL else "CONFIRMED"), p
    return "NOT SIGNIFICANT", p


def contrast(rates, a, b, gate_ok):
    diffs = a72.pair_diffs(rates[a], rates[b])
    a_mean, b_mean = st.mean(rates[a].values()), st.mean(rates[b].values())
    verdict, p = claim_verdict(diffs, a_mean, b_mean, gate_ok)
    return verdict, p, st.mean(diffs), a_mean, b_mean


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path.name}: the run is incomplete")
    r = json.loads(path.read_text())
    if r.get("limit_states") is not None:
        raise SystemExit(f"{path.name} is a smoke record and may not enter a verdict")
    return r


def check_eval_record(rec, path) -> None:
    name = Path(path).name
    arm = rec.get("arm")
    if arm not in ev.ARMS:
        raise SystemExit(f"{name}: arm is {arm!r}, expected one of {list(ev.ARMS)}")
    k, kg = ev.ARMS[arm]
    kg = k if kg is None else kg
    if rec.get("k") != k:
        raise SystemExit(f"{name}: k is {rec.get('k')!r}, spec says {k} for {arm}")
    # EXP-075's J3V-X records predate k_goal; their goal check was k by construction.
    if rec.get("k_goal", k) != kg:
        raise SystemExit(f"{name}: k_goal is {rec.get('k_goal')!r}, spec says {kg} for {arm}")
    if rec.get("mode") != "C" or rec.get("no_revisit") is not True:
        raise SystemExit(f"{name}: mode {rec.get('mode')!r}, no_revisit "
                         f"{rec.get('no_revisit')!r}; spec says C with no-revisit")
    if rec.get("judge_updates") != N_UPDATES:
        raise SystemExit(f"{name}: judge_updates is {rec.get('judge_updates')!r}, spec says "
                         f"{N_UPDATES}: it was evaluated from a partly trained judge")


def _read_checked(path: Path) -> dict:
    r = _read(path)
    check_eval_record(r, path)
    return r


def load_records(out_dir) -> dict:
    """(arm, depth) -> {seed: record}. J3V-X from EXP-075's committed records (licensed by
    Gate 0(b)); the two new arms from this experiment's outputs."""
    out_dir = Path(out_dir)
    recs = {}
    for d in DEPTHS:
        recs[("J3V-X", d)] = {s: _read_checked(ev.E75_OUT / f"exp075_J3V-X_d{d}_s{s}.json")
                              for s in SEEDS}
        for arm in NEW_ARMS:
            recs[(arm, d)] = {s: _read_checked(out_dir / ev.record_name(arm, d, s))
                              for s in SEEDS}
    return recs


def rates_of(recs) -> dict:
    return {key: {s: r["success_rate"] for s, r in by_seed.items()}
            for key, by_seed in recs.items()}


def gate_d(recs) -> list:
    """(depth, seed) cells where the goal-check arm solved fewer than J3V-X. Empty is PASS."""
    return [(d, s) for d in DEPTHS for s in SEEDS
            if recs[("J3VG4-X", d)][s]["solved"] < recs[("J3V-X", d)][s]["solved"]]


def gate_g(recs) -> dict:
    return {d: "RESOLVED" if sum(recs[("J3VG4-X", d)][s]["fired4"] for s in SEEDS) > 0
            else "UNRESOLVED" for d in DEPTHS}


def stage_cont(out) -> str:
    """Gate 0(b): J3V-X at seed 0, depths 9 and 11, re-run here, against EXP-075's records."""
    out = Path(out)
    return a71.gate0b_verdict(
        {("J3V-X", d): _read(out / ev.record_name("J3V-X", d, 0)) for d in DEPTHS},
        {("J3V-X", d): _read(ev.E75_OUT / f"exp075_J3V-X_d{d}_s0.json") for d in DEPTHS})


def primary_verdicts(rates: dict, gate_ok: bool, seeds) -> dict:
    """Claim 1: d11 J4V-X vs J3V-X. Claim 2: d9 J4V-X vs J3V-X. Claim 3: d11 J4V-X vs J3VG4-X."""
    r = {k: {s: v[s] for s in seeds} for k, v in rates.items()}
    return {"claim1": contrast(r, ("J4V-X", 11), ("J3V-X", 11), gate_ok),
            "claim2": contrast(r, ("J4V-X", 9), ("J3V-X", 9), gate_ok),
            "claim3": contrast(r, ("J4V-X", 11), ("J3VG4-X", 11), gate_ok)}


def sensitivity_rates(rates: dict) -> dict:
    return {k: {s: v for s, v in d.items() if s not in cells75.SENSITIVITY_DROP}
            for k, d in rates.items()}


def pilot_report(pilot_dir) -> dict:
    """Seconds per state and projected depth-11 cell hours per new arm, from the pilot's
    20-state records (spec section 6). Depth 9 is projected by EXP-075's own d9/d11 wall
    ratio for J3V-X. Memory is read on the laptop, not here."""
    pilot_dir = Path(pilot_dir)

    def pilot(path):
        if not path.exists():
            raise SystemExit(f"missing pilot record {path.name}")
        r = json.loads(path.read_text())
        if r.get("limit_states") != PILOT_LIMIT:
            raise SystemExit(f"{path.name}: limit_states is {r.get('limit_states')!r}, the "
                             f"pilot is {PILOT_LIMIT} states")
        return r

    walls = {d: st.mean(json.loads((ev.E75_OUT / f"exp075_J3V-X_d{d}_s{s}.json").read_text())
                        ["wall_s"] for s in SEEDS) for d in DEPTHS}
    ratio = walls[9] / walls[11]
    sps = {arm: pilot(pilot_dir / ev.record_name(arm, 11, 0))["wall_s"] / PILOT_LIMIT
           for arm in NEW_ARMS}
    rank = pilot(pilot_dir / ev.rank_record_name(11, 0))
    d11 = {arm: v * 200 / 3600 for arm, v in sps.items()}
    total = sum(12 * (h + h * ratio) for h in d11.values())
    return {"s_per_state": sps, "d11_cell_hours": d11, "d9_over_d11_wall_ratio": ratio,
            "rank_s_per_state": rank["wall_s"] / PILOT_LIMIT,
            "eval_cell_hours_total": total}


def _mean(recs, key, field):
    return st.mean(r[field] for r in recs[key].values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--pilot-dir", type=Path, default=HERE / "outputs_pilot")
    ap.add_argument("--pilot-report", action="store_true",
                    help="print the amendment's numbers from the pilot records and stop")
    ap.add_argument("--stage", choices=("cont",), help="read only Gate 0(b) and stop")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="set only after the det re-run matches its full-run copy")
    args = ap.parse_args()
    if args.pilot_report:
        for key, val in pilot_report(args.pilot_dir).items():
            print(f"{key}: {val}")
        return
    out = args.out_dir
    g0b = stage_cont(out)
    if args.stage == "cont":
        print(f"GATE 0(b) J3V-X continuity with EXP-075 (d9, d11, seed 0): {g0b}")
        return
    recs = load_records(out)
    rates = rates_of(recs)
    viol = gate_d(recs)
    gg = gate_g(recs)
    rank = {d: [_read(out / ev.rank_record_name(d, s)) for s in SEEDS] for d in DEPTHS}
    rank75 = {d: [json.loads((ev.E75_OUT / f"exp075_rank_J-X_d{d}_s{s}.json").read_text())
                  for s in SEEDS] for d in DEPTHS}

    print(f"GATE 0(a) determinism: {'PASS' if args.determinism_ok else 'NOT CHECKED'}")
    print(f"GATE 0(b) J3V-X continuity with EXP-075 (d9, d11, seed 0): {g0b}")
    print(f"GATE D dominance (J3VG4-X >= J3V-X on every cell): "
          f"{'PASS' if not viol else 'FAIL at (depth, seed) ' + str(viol)}")
    print(f"GATE G (4-move goal check fired): {gg}")
    print("GATE R at k 4 (reported, not voiding): lowest-J 4-move leaf closer than root")
    for d in DEPTHS:
        ok, p = a71.gate_r_verdict(rank[d], "hit", "chance")
        print(f"  d{d}: hit {st.mean(r['hit'] for r in rank[d]):.4f} chance "
              f"{st.mean(r['chance'] for r in rank[d]):.4f} p {p:.4f} -> {ok}   "
              f"(EXP-075 k 3: hit {st.mean(r['hit'] for r in rank75[d]):.4f} chance "
              f"{st.mean(r['chance'] for r in rank75[d]):.4f})")

    print("\nMean held-out success (mean steps to solve, optimality)")
    for d in DEPTHS:
        print(f"  d{d}: " + "  ".join(
            f"{a} {_mean(recs, (a, d), 'success_rate'):.4f} "
            f"({_mean(recs, (a, d), 'mean_steps'):.2f}, {_mean(recs, (a, d), 'optimality'):.3f})"
            for a in ALL_ARMS))

    gate_ok = g0b == "PASS" and args.determinism_ok and not viol
    if not gate_ok:
        print("Gate 0 or Gate D did not pass. Every claim is VOID.")
    full = primary_verdicts(rates, gate_ok, SEEDS)
    sens_rates = sensitivity_rates(rates)
    sens = primary_verdicts(sens_rates, gate_ok, sorted(sens_rates[("J4V-X", 9)]))
    for name, label in (("claim1", "CLAIM 1 (primary) d11 J4V-X - J3V-X"),
                        ("claim2", "CLAIM 2 (primary) d9 J4V-X - J3V-X"),
                        ("claim3", "CLAIM 3 (primary) d11 J4V-X - J3VG4-X")):
        v, p, diff, am, bm = full[name]
        sv, sp, sdiff, _, _ = sens[name]
        print(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} -> {v}")
        print(f"    sensitivity, seeds 0 and 3 dropped: diff {sdiff:+.4f}, p {sp:.4f} -> {sv}")
        if sv != v:
            print("    SENSITIVITY DISAGREES WITH THE VERDICT: RESULTS.md must lead with this.")
    if gg[11] == "UNRESOLVED":
        print("    Claim 3 reads against a 4-move goal check that never fired at depth 11.")

    print("\nSecondary (patterns, never confirmations):")
    v, p, diff, am, bm = contrast(rates, ("J4V-X", 9), ("J3VG4-X", 9), gate_ok)
    print(f"  d9 J4V-X - J3VG4-X: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f}")
    for d in DEPTHS:
        _, p, diff, am, bm = contrast(rates, ("J3VG4-X", d), ("J3V-X", d), gate_ok)
        f4 = sum(recs[("J3VG4-X", d)][s]["fired4"] for s in SEEDS)
        print(f"  d{d} J3VG4-X - J3V-X (goal reach alone, >= 0 by construction): "
              f"diff {diff:+.4f}, summed fired4 {f4} -> Gate G {gg[d]}")


if __name__ == "__main__":
    main()
