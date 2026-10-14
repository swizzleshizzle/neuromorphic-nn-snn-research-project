"""EXP-078 aggregator. Gates 0, D and EXP-077's validity are VERDICTS: a claim whose gate failed
prints VOID, a contrast on the floor or ceiling prints UNRESOLVED, and a confirmed gain under
0.01 prints CONFIRMED (SMALL). Claims 3 and 4 read EXP-077's arms, so they are also VOID unless
EXP-077's own gates are computed to pass here (exp077_valid).

Spec: docs/superpowers/specs/2026-10-14-exp078-full-five-move-tree-design.md, sections 3, 7, 8.

The sign-flip test and Gate 1 are EXP-070's, Gate 0(b)'s field comparison is EXP-071's, seed
pairing is EXP-072's, EXP-077's gates are its own, all reused through importlib. Alpha is
0.05 / 4.

Usage:
    .venv/bin/python experiments/078_full_five_move_tree/aggregate.py --pilot-report
    .venv/bin/python experiments/078_full_five_move_tree/aggregate.py --stage cont
    .venv/bin/python experiments/078_full_five_move_tree/aggregate.py   # full report; Gate 0(a)
    #   is computed from outputs_det/ (--det-dir) against outputs/
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
a77 = _load("exp077_aggregate", REPO / "experiments" / "077_beam_search" / "aggregate.py")
ev = _load("exp078_evaluate", HERE / "evaluate.py")
one_sided_p = a70.one_sided_p

SEEDS = range(12)
DEPTHS = ev.DEPTHS
ALPHA = 0.05 / 4
SMALL = 0.01
N_UPDATES = 4000
PILOT_LIMIT = 20
E77_ARMS = ("J4VG5-X", "B5-X")
ALL_ARMS = ("J5V-X", "J4V-X") + E77_ARMS
E76_RANK = REPO / "experiments" / "076_deeper_tree" / "outputs"

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
    arm = rec.get("arm")
    if arm in ev.ARMS:
        name = Path(path).name
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
    elif arm in a77.ev.ARMS:
        a77.check_eval_record(rec, path)
    else:
        raise SystemExit(f"{Path(path).name}: arm is {arm!r}, expected one of "
                         f"{list(ev.ARMS) + list(a77.ev.ARMS)}")


def _read_cell(path: Path, arm: str, depth: int, seed: int) -> dict:
    """Read a record, check its spec fields, and check it is the cell its filename claims."""
    r = _read(path)
    check_eval_record(r, path)
    for field, want in (("arm", arm), ("depth", depth), ("seed", seed)):
        if r.get(field) != want:
            raise SystemExit(f"{Path(path).name}: {field} is {r.get(field)!r}, the file name "
                             f"says {want!r}")
    return r


def load_records(out_dir, include_e77: bool = True) -> dict:
    """(arm, depth) -> {seed: record}. J5V-X from this experiment's outputs; J4V-X from
    EXP-076's committed records (licensed by Gate 0(b)); J4VG5-X and B5-X from EXP-077's."""
    out_dir = Path(out_dir)
    recs = {}
    for d in DEPTHS:
        recs[("J5V-X", d)] = {s: _read_cell(out_dir / ev.record_name("J5V-X", d, s),
                                              "J5V-X", d, s) for s in SEEDS}
        recs[("J4V-X", d)] = {s: _read_cell(ev.E76_OUT / f"exp076_J4V-X_d{d}_s{s}.json",
                                              "J4V-X", d, s) for s in SEEDS}
        for arm in (E77_ARMS if include_e77 else ()):
            recs[(arm, d)] = {s: _read_cell(ev.E77_OUT / f"exp077_{arm}_d{d}_s{s}.json",
                                              arm, d, s) for s in SEEDS}
    return recs


DET_IGNORED = ("wall_s", "git_commit", "resumes", "elapsed_s")


def gate0a(out_dir, det_dir) -> str:
    """Gate 0(a): the det re-run of J5V-X d9 s0 equals the full-run record in every field except
    the session fields (a resumed cell differs in resumes and elapsed_s by design)."""
    recs = []
    for d in (Path(out_dir), Path(det_dir)):
        p = d / ev.record_name("J5V-X", 9, 0)
        if not p.exists():
            raise SystemExit(f"missing {p}: Gate 0(a) cannot be computed")
        recs.append({k: v for k, v in json.loads(p.read_text()).items()
                     if k not in DET_IGNORED})
    return "PASS" if recs[0] == recs[1] else "FAIL"


def exp077_valid() -> tuple[bool, str]:
    """Whether EXP-077's own gates pass, computed from its committed outputs. Claims 3 and 4
    read its arms, so they are VOID when this is False."""
    try:
        if a77.stage_cont(ev.E77_OUT) != "PASS":
            return False, "EXP-077 Gate 0(b) did not pass"
        if a77.gate0a(ev.E77_OUT, ev.E77_DET) != "PASS":
            return False, "EXP-077 Gate 0(a) did not pass"
        recs77 = a77.load_records(ev.E77_OUT)
        bad = a77.gate_d(recs77)
        if bad:
            return False, f"EXP-077 Gate D failed at (depth, seed) {bad}"
        if not a77.gate_b(recs77)["ok"]:
            return False, "EXP-077 Gate B did not pass"
    except SystemExit as e:
        return False, f"EXP-077 records unavailable: {e}"
    return True, ""


def rates_of(recs) -> dict:
    return {key: {s: r["success_rate"] for s, r in by_seed.items()}
            for key, by_seed in recs.items()}


def gate_d(recs) -> list:
    """Cells (depth, seed), sorted, where J5V-X's first_fire_len has a key other than "5"
    (the first-fire instrument check). Empty is PASS."""
    bad = set()
    for d in DEPTHS:
        for s in SEEDS:
            if set(recs[("J5V-X", d)][s]["first_fire_len"]) - {"5"}:
                bad.add((d, s))
    return sorted(bad)


def stage_cont(out) -> str:
    """Gate 0(b): J4V-X at seed 0, depths 9 and 11, re-run here, against EXP-076's records."""
    out = Path(out)
    return a71.gate0b_verdict(
        {("J4V-X", d): _read_cell(out / ev.record_name("J4V-X", d, 0), "J4V-X", d, 0)
         for d in DEPTHS},
        {("J4V-X", d): _read(ev.E76_OUT / f"exp076_J4V-X_d{d}_s0.json") for d in DEPTHS})


def primary_verdicts(rates: dict, gate_ok: bool, e77_ok: bool, seeds) -> dict:
    """Claim 1: d11 J5V-X vs J4V-X. Claim 2: d9 J5V-X vs J4V-X. Claim 3: d11 J5V-X vs J4VG5-X.
    Claim 4: d11 J5V-X vs B5-X. Claims 3 and 4 also need EXP-077 to be valid."""
    r = {k: {s: v[s] for s in seeds} for k, v in rates.items()}
    g34 = gate_ok and e77_ok
    nan = float("nan")
    absent = ("VOID", nan, nan, nan, nan)  # EXP-077's arms were not loaded: it is invalid
    return {"claim1": contrast(r, ("J5V-X", 11), ("J4V-X", 11), gate_ok),
            "claim2": contrast(r, ("J5V-X", 9), ("J4V-X", 9), gate_ok),
            "claim3": (contrast(r, ("J5V-X", 11), ("J4VG5-X", 11), g34)
                       if ("J4VG5-X", 11) in r else absent),
            "claim4": (contrast(r, ("J5V-X", 11), ("B5-X", 11), g34)
                       if ("B5-X", 11) in r else absent)}


def sensitivity_rates(rates: dict) -> dict:
    return {k: {s: v for s, v in d.items() if s not in cells75.SENSITIVITY_DROP}
            for k, d in rates.items()}


def gate_r(out_dir) -> list:
    """Reported only: per depth, mean k 5 hit and chance against EXP-076's k 4."""
    lines = []
    for d in DEPTHS:
        row = []
        for label, base, name in (
                ("k5", Path(out_dir), ev.rank_record_name),
                ("k4", E76_RANK, lambda dd, s: f"exp076_rank_J-X4_d{dd}_s{s}.json")):
            paths = [base / name(d, s) for s in SEEDS]
            if not all(p.exists() for p in paths):
                row.append(f"{label} not run")
                continue
            rs = [json.loads(p.read_text()) for p in paths]
            row.append(f"{label} hit {st.mean(r['hit'] for r in rs):.4f} "
                       f"chance {st.mean(r['chance'] for r in rs):.4f}")
        lines.append(f"  d{d}: " + "; ".join(row))
    return lines


def pilot_report(pilot_dir) -> dict:
    """Seconds per state of the J5V-X pilot, projected depth-11 cell hours and total eval cell
    hours. Depth 9 is projected by EXP-076's own d9/d11 wall ratio for J4V-X."""
    path = Path(pilot_dir) / ev.record_name("J5V-X", 11, 0)
    if not path.exists():
        raise SystemExit(f"missing pilot record {path.name}")
    r = json.loads(path.read_text())
    if r.get("limit_states") != PILOT_LIMIT:
        raise SystemExit(f"{path.name}: limit_states is {r.get('limit_states')!r}, the pilot "
                         f"is {PILOT_LIMIT} states")
    walls = {d: st.mean(json.loads((ev.E76_OUT / f"exp076_J4V-X_d{d}_s{s}.json").read_text())
                        ["wall_s"] for s in SEEDS) for d in DEPTHS}
    ratio = walls[9] / walls[11]
    sps = r["wall_s"] / PILOT_LIMIT
    d11 = sps * 200 / 3600
    return {"s_per_state": sps, "d11_cell_hours": d11, "d9_over_d11_wall_ratio": ratio,
            "eval_cell_hours_total": 12 * (d11 + d11 * ratio)}


def _mean(recs, key, field):
    return st.mean(r[field] for r in recs[key].values())


def _evals(recs, arm, d) -> str:
    """Mean judge_evals; only J5V-X's records are required to carry the field here."""
    if arm != "J5V-X":
        return "n/a"
    return f"{_mean(recs, (arm, d), 'judge_evals'):.0f}"


CLAIM_LABELS = (("claim1", "CLAIM 1 (primary) d11 J5V-X - J4V-X"),
                ("claim2", "CLAIM 2 (primary) d9 J5V-X - J4V-X"),
                ("claim3", "CLAIM 3 (primary) d11 J5V-X - J4VG5-X"),
                ("claim4", "CLAIM 4 (primary) d11 J5V-X - B5-X"))


def report_lines(recs, g0b, g0a, e77=(True, ""), out_dir=None) -> list:
    """Everything main prints after loading, as lines, so it can be tested on real-shaped
    records."""
    out = []
    determinism_ok = g0a == "PASS"
    arms = [a for a in ALL_ARMS if (a, 11) in recs]
    rates = rates_of(recs)
    viol = gate_d(recs)
    e77_ok, e77_why = e77

    out.append(f"GATE 0(a) determinism: {g0a}")
    out.append(f"GATE 0(b) J4V-X continuity with EXP-076 (d9, d11, seed 0): {g0b}")
    out.append("GATE D first-fire instrument (J5V-X first_fire_len only \"5\"): "
               f"{'PASS' if not viol else 'FAIL at (depth, seed) ' + str(viol)}")
    out.append(f"EXP-077 validity (Claims 3 and 4): "
               f"{'PASS' if e77_ok else 'FAIL, ' + e77_why}")

    out.append("\nMean held-out success (mean steps to solve, optimality, judge_evals)")
    for d in DEPTHS:
        out.append(f"  d{d}: " + "  ".join(
            f"{a} {_mean(recs, (a, d), 'success_rate'):.4f} "
            f"({_mean(recs, (a, d), 'mean_steps'):.2f}, {_mean(recs, (a, d), 'optimality'):.3f}, "
            f"{_evals(recs, a, d)})"
            for a in arms))

    out.append("\nSeconds per state (mean wall_s / n)")
    for d in DEPTHS:
        out.append(f"  d{d}: " + "  ".join(
            f"{a} {st.mean(r['wall_s'] / r['n'] for r in recs[(a, d)].values()):.3f}"
            for a in arms))

    gate_ok = g0b == "PASS" and determinism_ok and not viol
    if not gate_ok:
        out.append("Gate 0 or D did not pass. Every claim is VOID.")
    elif not e77_ok:
        out.append("EXP-077 is not valid. Claims 3 and 4 are VOID.")
    full = primary_verdicts(rates, gate_ok, e77_ok, SEEDS)
    sens_rates = sensitivity_rates(rates)
    sens = primary_verdicts(sens_rates, gate_ok, e77_ok, sorted(sens_rates[("J5V-X", 9)]))
    for name, label in CLAIM_LABELS:
        v, p, diff, am, bm = full[name]
        if am != am:  # nan: EXP-077's arms were not loaded
            out.append(f"{label}: VOID (EXP-077 invalid: {e77_why})")
            continue
        sv, sp, sdiff, _, _ = sens[name]
        out.append(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} -> {v}")
        out.append(f"    10-seed line, seeds 0 and 3 dropped: diff {sdiff:+.4f}, p {sp:.4f} -> {sv}")
        if sv != v:
            out.append("    10-SEED LINE DISAGREES WITH THE VERDICT: RESULTS.md must lead with this.")

    out.append("\nSecondary (patterns, never confirmations):")
    for arm in ("J4VG5-X", "B5-X"):
        if (arm, 9) not in rates:
            out.append(f"  d9 J5V-X - {arm}: EXP-077 invalid ({e77_why})")
            continue
        _, p, diff, am, bm = contrast(rates, ("J5V-X", 9), (arm, 9), gate_ok and e77_ok)
        out.append(f"  d9 J5V-X - {arm}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f}"
                   + ("" if e77_ok else "  [EXP-077 invalid]"))
    out.append("Gate R (reported only), lowest-J leaf closer than root, k 5 vs EXP-076 k 4:")
    out.extend(gate_r(out_dir if out_dir is not None else HERE / "outputs"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--pilot-dir", type=Path, default=HERE / "outputs_pilot")
    ap.add_argument("--pilot-report", action="store_true",
                    help="print the amendment's numbers from the pilot record and stop")
    ap.add_argument("--stage", choices=("cont",), help="read only Gate 0(b) and stop")
    ap.add_argument("--det-dir", type=Path, default=HERE / "outputs_det",
                    help="the det re-run's output directory, compared for Gate 0(a)")
    args = ap.parse_args()
    if args.pilot_report:
        for key, val in pilot_report(args.pilot_dir).items():
            print(f"{key}: {val}")
        return
    out = args.out_dir
    g0b = stage_cont(out)
    if args.stage == "cont":
        print(f"GATE 0(b) J4V-X continuity with EXP-076 (d9, d11, seed 0): {g0b}")
        return
    e77 = exp077_valid()
    print("\n".join(report_lines(load_records(out, include_e77=e77[0]), g0b,
                                 gate0a(out, args.det_dir), e77, out)))


if __name__ == "__main__":
    main()
