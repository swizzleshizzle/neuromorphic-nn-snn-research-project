"""EXP-077 aggregator. Gates 0, D, B and 1 are VERDICTS: a claim whose gate failed prints VOID,
a contrast on the floor or ceiling prints UNRESOLVED, and a confirmed gain under 0.01 prints
CONFIRMED (SMALL).

Spec: docs/superpowers/specs/2026-10-14-exp077-beam-search-design.md, sections 6 to 8.

The sign-flip test and Gate 1 are EXP-070's, Gate 0(b)'s field comparison is EXP-071's, seed
pairing is EXP-072's, all reused through importlib. Alpha is 0.05 / 3.

Usage:
    .venv/bin/python experiments/077_beam_search/aggregate.py --pilot-report
    .venv/bin/python experiments/077_beam_search/aggregate.py --stage cont
    .venv/bin/python experiments/077_beam_search/aggregate.py   # full report; Gate 0(a) is
    #   computed from outputs_det/ (--det-dir) against outputs/
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
ev = _load("exp077_evaluate", HERE / "evaluate.py")
one_sided_p = a70.one_sided_p

SEEDS = range(12)
DEPTHS = ev.DEPTHS
ALPHA = 0.05 / 3
SMALL = 0.01
N_UPDATES = 4000
PILOT_LIMIT = 20
NEW_ARMS = ("J4VG5-X", "B5-X")
ALL_ARMS = ("J4V-X",) + NEW_ARMS
MAX_EVALS = 1026

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
    mode, k, kg, w = ev.ARMS[arm]
    if rec.get("k") != k:
        raise SystemExit(f"{name}: k is {rec.get('k')!r}, spec says {k} for {arm}")
    if rec.get("k_goal") != kg:
        raise SystemExit(f"{name}: k_goal is {rec.get('k_goal')!r}, spec says {kg} for {arm}")
    if rec.get("mode") != mode or rec.get("no_revisit") is not True:
        raise SystemExit(f"{name}: mode {rec.get('mode')!r}, no_revisit "
                         f"{rec.get('no_revisit')!r}; spec says {mode} with no-revisit")
    if rec.get("beam_width") != w:
        raise SystemExit(f"{name}: beam_width is {rec.get('beam_width')!r}, spec says {w} "
                         f"for {arm}")
    if rec.get("judge_updates") != N_UPDATES:
        raise SystemExit(f"{name}: judge_updates is {rec.get('judge_updates')!r}, spec says "
                         f"{N_UPDATES}: it was evaluated from a partly trained judge")


def _read_checked(path: Path) -> dict:
    r = _read(path)
    check_eval_record(r, path)
    return r


def _read_cell(path: Path, arm: str, depth: int, seed: int) -> dict:
    """Read a record, check its spec fields, and check it is the cell its filename claims."""
    r = _read_checked(path)
    for field, want in (("arm", arm), ("depth", depth), ("seed", seed)):
        if r.get(field) != want:
            raise SystemExit(f"{Path(path).name}: {field} is {r.get(field)!r}, the file name "
                             f"says {want!r}")
    return r


def load_records(out_dir) -> dict:
    """(arm, depth) -> {seed: record}. J4V-X from EXP-076's committed records (licensed by
    Gate 0(b)); the two new arms from this experiment's outputs."""
    out_dir = Path(out_dir)
    recs = {}
    for d in DEPTHS:
        recs[("J4V-X", d)] = {s: _read_cell(ev.E76_OUT / f"exp076_J4V-X_d{d}_s{s}.json",
                                              "J4V-X", d, s) for s in SEEDS}
        for arm in NEW_ARMS:
            recs[(arm, d)] = {s: _read_cell(out_dir / ev.record_name(arm, d, s), arm, d, s)
                              for s in SEEDS}
    return recs


DET_NAME = "exp077_B5-X_d9_s0.json"
DET_IGNORED = ("wall_s", "git_commit")


def gate0a(out_dir, det_dir) -> str:
    """Gate 0(a): the det re-run of B5-X d9 s0 equals the full-run record in every field except
    wall_s and git_commit. Computed, never asserted by hand."""
    recs = []
    for d in (Path(out_dir), Path(det_dir)):
        p = d / DET_NAME
        if not p.exists():
            raise SystemExit(f"missing {p}: Gate 0(a) cannot be computed")
        recs.append({k: v for k, v in json.loads(p.read_text()).items()
                     if k not in DET_IGNORED})
    return "PASS" if recs[0] == recs[1] else "FAIL"


def rates_of(recs) -> dict:
    return {key: {s: r["success_rate"] for s, r in by_seed.items()}
            for key, by_seed in recs.items()}


def gate_d(recs) -> list:
    """Cells (depth, seed), sorted and deduplicated, that violate Gate D. Empty is PASS.

    Three clauses (spec section 7):
      (i)   dominance: J4VG5-X solved fewer than J4V-X;
      (ii)  first-fire instrument: for either new arm, first_fire_len has a key other than "5";
      (iii) J4VG5-X fired5 < J4V-X solved."""
    bad = set()
    for d in DEPTHS:
        for s in SEEDS:
            j4, g5, b5 = recs[("J4V-X", d)][s], recs[("J4VG5-X", d)][s], recs[("B5-X", d)][s]
            if g5["solved"] < j4["solved"]:
                bad.add((d, s))
            if any(set(r["first_fire_len"]) - {"5"} for r in (g5, b5)):
                bad.add((d, s))
            if g5["fired5"] < j4["solved"]:
                bad.add((d, s))
    return sorted(bad)


def gate_b(recs) -> dict:
    """Beam instrument check: B5-X may not exceed MAX_EVALS judge evaluations on any cell."""
    cells = {d: list(recs[("B5-X", d)].values()) for d in DEPTHS}
    return {"ok": all(r["judge_evals"] <= MAX_EVALS for rs in cells.values() for r in rs),
            "judge_evals": {d: st.mean(r["judge_evals"] for r in rs) for d, rs in cells.items()},
            "fallback": {d: st.mean(r["fallback_frac"] for r in rs) for d, rs in cells.items()}}


def stage_cont(out) -> str:
    """Gate 0(b): J4V-X at seed 0, depths 9 and 11, re-run here, against EXP-076's records."""
    out = Path(out)
    return a71.gate0b_verdict(
        {("J4V-X", d): _read_cell(out / ev.record_name("J4V-X", d, 0), "J4V-X", d, 0)
         for d in DEPTHS},
        {("J4V-X", d): _read(ev.E76_OUT / f"exp076_J4V-X_d{d}_s0.json") for d in DEPTHS})


def primary_verdicts(rates: dict, gate_ok: bool, seeds) -> dict:
    """Claim 1: d11 B5-X vs J4V-X. Claim 2: d9 B5-X vs J4V-X. Claim 3: d11 B5-X vs J4VG5-X."""
    r = {k: {s: v[s] for s in seeds} for k, v in rates.items()}
    return {"claim1": contrast(r, ("B5-X", 11), ("J4V-X", 11), gate_ok),
            "claim2": contrast(r, ("B5-X", 9), ("J4V-X", 9), gate_ok),
            "claim3": contrast(r, ("B5-X", 11), ("J4VG5-X", 11), gate_ok)}


def sensitivity_rates(rates: dict) -> dict:
    return {k: {s: v for s, v in d.items() if s not in cells75.SENSITIVITY_DROP}
            for k, d in rates.items()}


def pilot_report(pilot_dir) -> dict:
    """Seconds per state and projected depth-11 cell hours per new arm, from the pilot's
    20-state records (spec section 6). Depth 9 is projected by EXP-076's own d9/d11 wall
    ratio for J4V-X. Memory is read on the laptop, not here."""
    pilot_dir = Path(pilot_dir)

    def pilot(path):
        if not path.exists():
            raise SystemExit(f"missing pilot record {path.name}")
        r = json.loads(path.read_text())
        if r.get("limit_states") != PILOT_LIMIT:
            raise SystemExit(f"{path.name}: limit_states is {r.get('limit_states')!r}, the "
                             f"pilot is {PILOT_LIMIT} states")
        return r

    walls = {d: st.mean(json.loads((ev.E76_OUT / f"exp076_J4V-X_d{d}_s{s}.json").read_text())
                        ["wall_s"] for s in SEEDS) for d in DEPTHS}
    ratio = walls[9] / walls[11]
    sps = {arm: pilot(pilot_dir / ev.record_name(arm, 11, 0))["wall_s"] / PILOT_LIMIT
           for arm in NEW_ARMS}
    d11 = {arm: v * 200 / 3600 for arm, v in sps.items()}
    total = sum(12 * (h + h * ratio) for h in d11.values())
    return {"s_per_state": sps, "d11_cell_hours": d11, "d9_over_d11_wall_ratio": ratio,
            "eval_cell_hours_total": total}


def _mean(recs, key, field):
    return st.mean(r[field] for r in recs[key].values())


def _evals(recs, arm, d) -> str:
    """Mean judge_evals; EXP-076's committed J4V-X records predate the field."""
    if arm not in NEW_ARMS:
        return "n/a"
    return f"{_mean(recs, (arm, d), 'judge_evals'):.0f}"


def report_lines(recs, g0b, determinism_ok) -> list:
    """Everything main prints after loading, as lines, so it can be tested on real-shaped
    records."""
    out = []
    rates = rates_of(recs)
    viol = gate_d(recs)
    gb = gate_b(recs)

    out.append(f"GATE 0(a) determinism: {'PASS' if determinism_ok else 'NOT CHECKED'}")
    out.append(f"GATE 0(b) J4V-X continuity with EXP-076 (d9, d11, seed 0): {g0b}")
    out.append(f"GATE D dominance and first-fire instrument (J4VG5-X >= J4V-X on every cell): "
          f"{'PASS' if not viol else 'FAIL at (depth, seed) ' + str(viol)}")
    out.append(f"GATE B beam judge evaluations <= {MAX_EVALS} on every cell: "
          f"{'PASS' if gb['ok'] else 'FAIL'}; mean judge_evals {gb['judge_evals']}, "
          f"mean fallback_frac {gb['fallback']}")

    out.append("\nMean held-out success (mean steps to solve, optimality, judge_evals)")
    for d in DEPTHS:
        out.append(f"  d{d}: " + "  ".join(
            f"{a} {_mean(recs, (a, d), 'success_rate'):.4f} "
            f"({_mean(recs, (a, d), 'mean_steps'):.2f}, {_mean(recs, (a, d), 'optimality'):.3f}, "
            f"{_evals(recs, a, d)})"
            for a in ALL_ARMS))

    out.append("\nSeconds per state (mean wall_s / n)")
    for d in DEPTHS:
        out.append(f"  d{d}: " + "  ".join(
            f"{a} {st.mean(r['wall_s'] / r['n'] for r in recs[(a, d)].values()):.3f}"
            for a in ALL_ARMS))

    gate_ok = g0b == "PASS" and determinism_ok and not viol and gb["ok"]
    if not gate_ok:
        out.append("Gate 0, D or B did not pass. Every claim is VOID.")
    full = primary_verdicts(rates, gate_ok, SEEDS)
    sens_rates = sensitivity_rates(rates)
    sens = primary_verdicts(sens_rates, gate_ok, sorted(sens_rates[("B5-X", 9)]))
    for name, label in (("claim1", "CLAIM 1 (primary) d11 B5-X - J4V-X"),
                        ("claim2", "CLAIM 2 (primary) d9 B5-X - J4V-X"),
                        ("claim3", "CLAIM 3 (primary) d11 B5-X - J4VG5-X")):
        v, p, diff, am, bm = full[name]
        sv, sp, sdiff, _, _ = sens[name]
        out.append(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} -> {v}")
        out.append(f"    10-seed line, seeds 0 and 3 dropped: diff {sdiff:+.4f}, p {sp:.4f} -> {sv}")
        if sv != v:
            out.append("    10-SEED LINE DISAGREES WITH THE VERDICT: RESULTS.md must lead with this.")

    out.append("\nSecondary (patterns, never confirmations):")
    v, p, diff, am, bm = contrast(rates, ("B5-X", 9), ("J4VG5-X", 9), gate_ok)
    out.append(f"  d9 B5-X - J4VG5-X: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f}")
    for d in DEPTHS:
        _, p, diff, am, bm = contrast(rates, ("J4VG5-X", d), ("J4V-X", d), gate_ok)
        f5 = sum(recs[("J4VG5-X", d)][s]["fired5"] for s in SEEDS)
        out.append(f"  d{d} J4VG5-X - J4V-X (goal reach alone, >= 0 by construction): "
              f"diff {diff:+.4f}, summed fired5 {f5}")
    return out



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--pilot-dir", type=Path, default=HERE / "outputs_pilot")
    ap.add_argument("--pilot-report", action="store_true",
                    help="print the amendment's numbers from the pilot records and stop")
    ap.add_argument("--stage", choices=("cont",), help="read only Gate 0(b) and stop")
    ap.add_argument("--det-dir", type=Path, default=HERE / "outputs_det",
                    help="the det re-run's output directory, compared for Gate 0(a)")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="ignored: Gate 0(a) is now computed from --det-dir")
    args = ap.parse_args()
    if args.pilot_report:
        for key, val in pilot_report(args.pilot_dir).items():
            print(f"{key}: {val}")
        return
    if args.determinism_ok:
        print("note: --determinism-ok is ignored; Gate 0(a) is computed from --det-dir")
    out = args.out_dir
    g0b = stage_cont(out)
    if args.stage == "cont":
        print(f"GATE 0(b) J4V-X continuity with EXP-076 (d9, d11, seed 0): {g0b}")
        return
    print("\n".join(report_lines(load_records(out), g0b,
                                   gate0a(out, args.det_dir) == "PASS")))


if __name__ == "__main__":
    main()
