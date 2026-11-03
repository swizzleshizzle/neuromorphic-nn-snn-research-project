"""EXP-085 aggregator: Gates C, H and T, Claims 1 and 2, the depth-9 guard, and the one-line
verdict of spec section 8. One-sided exact sign-flip over all 4096 flips, paired by seed.

Spec: docs/superpowers/specs/2026-11-03-exp085-outward-walks-design.md, sections 6 to 9.

Usage (repo root, PYTHONPATH=src):
    python experiments/085_outward_walks/aggregate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp085_cells", HERE / "cells.py")
a80 = _load("exp080_aggregate", cells.REPO / "experiments" / "080_oracle_ordering" / "aggregate.py")
gateh = _load("exp085_gateh", HERE / "gateh.py")
one_sided_p, far, overall = a80.one_sided_p, a80.far, a80.overall

SEEDS = list(cells.SEEDS)
CLAIM_ALPHA = a80.CLAIM_ALPHA
C1_MIN = 0.03
GUARD_DROP = -0.05
GATE_T_THRESHOLD = a80.GATE_T_THRESHOLD


def verdict(c_ok: bool, h_ok: bool, t_ok: bool, c1: str, c2: str, guard: bool) -> str:
    """Spec section 8, one line. CONFIRMED (SMALL) falls through to the next row."""
    if not c_ok:
        return "VOID"
    if not h_ok:
        return "VOID (DESKTOP NOT REPRODUCIBLE)"
    if not t_ok:
        return "UNRESOLVED (ARM UNTRAINED)"
    if c1 == "CONFIRMED" and guard:
        return "OUTWARD WALKS PAY AT DEPTH 11, COST DEPTH 9"
    if c1 == "CONFIRMED":
        return "OUTWARD WALKS PAY AT DEPTH 11"
    if c2 == "CONFIRMED":
        return "ORDERING IMPROVES FAR OUT, SEARCH DOES NOT PAY"
    return "OUTWARD WALKS DO NOT MOVE DEPTH 11"


def claim1_verdict(diffs) -> tuple[str, float, float]:
    p, m = one_sided_p(diffs), st.mean(diffs)
    if p < CLAIM_ALPHA and m > 0:
        return ("CONFIRMED" if m >= C1_MIN else "CONFIRMED (SMALL)"), p, m
    return "REFUTED", p, m


def guard_trips(d9_diffs) -> bool:
    """The depth-9 guard: a mean drop of at least 0.05 that a one-sided test for a drop finds."""
    return st.mean(d9_diffs) <= GUARD_DROP and one_sided_p([-d for d in d9_diffs]) < CLAIM_ALPHA


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing record {path.name}")
    return json.loads(path.read_text())


def _expect(rec, name, checks) -> dict:
    for k, want in checks:
        if rec.get(k) != want:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, spec says {want!r}")
    return rec


def read_train(out, arm, update, seed) -> dict:
    name = cells.record_name(arm, update, seed)
    return _expect(_read(out / name), name, (
        ("arm", arm), ("seed", seed), ("n_updates", update), ("device", "cuda"),
        ("max_len_schedule", cells.schedule_label(arm))))


def read_search(out, arm, depth, seed, name=None) -> dict:
    name = name or cells.search_name(arm, depth, seed)
    return _expect(_read(out / name), name, (
        ("arm", f"J4V-{arm}"), ("judge", arm), ("depth", depth), ("seed", seed),
        ("judge_seed", seed), ("k", 4), ("k_goal", 4), ("mode", "C"), ("no_revisit", True),
        ("judge_updates", cells.END), ("device", "cuda"), ("limit_states", None), ("n", 200)))


def check_one_checkout(recs) -> str:
    commits = {r["git_commit"] for r in recs}
    if len(commits) != 1:
        raise SystemExit(f"records come from more than one checkout: {sorted(commits)}")
    return commits.pop()


def _diffs(a: dict, b: dict) -> list:
    return [a[s] - b[s] for s in SEEDS]


def gate_h(out) -> dict:
    a_bad = [(arm, d, s) for arm, d, s in gateh.search_picks()
             if not gateh.same(read_search(out, arm, d, s),
                               read_search(out, arm, d, s, cells.h_search_name(arm, d, s)))]
    b = [_read(out / cells.h_train_name(arm, s)) for arm, s in gateh.train_picks()]
    b_available = all(r.get("available") is True for r in b)
    b_bad = [r for r in b if r.get("available") is True and r.get("equal") is not True]
    return {"ok": not a_bad and not b_bad, "a_bad": a_bad, "b_available": b_available,
            "b_bad": len(b_bad)}


def _h_line(h) -> str:
    a = "PASS" if not h["a_bad"] else f"FAIL on {h['a_bad']}"
    if h["b_available"]:
        b = "PASS" if not h["b_bad"] else f"FAIL on {h['b_bad']} of 4"
    else:
        b = "unavailable (pre-flight found GPU training not deterministic)"
    return f"GATE H (hardware): (a) search replay {a}; (b) {b}"


def run_all(out=cells.OUT, e82_root=cells.E82_OUT) -> dict:
    out, e82_root = Path(out), Path(e82_root)
    gc_fail = [s for s in SEEDS if _read(out / cells.gatec_name(s)).get("pass") is not True]
    if gc_fail:
        return {"verdict": verdict(False, False, False, "VOID", "VOID", False),
                "lines": [f"GATE C (continuity): FAIL on seeds {gc_fail}"], "reported": []}
    tr = {a: {u: {s: read_train(out, a, u, s) for s in SEEDS} for u in cells.RECORD_LEGS}
          for a in cells.ARMS}
    sr = {a: {d: {s: read_search(out, a, d, s) for s in SEEDS} for d in cells.DEPTHS}
          for a in cells.ARMS}
    check_one_checkout([r for a in tr for u in tr[a] for r in tr[a][u].values()])
    h = gate_h(out)
    end = {a: tr[a][cells.END] for a in cells.ARMS}
    t_ok = {a: a80.gate_t([overall(end[a][s]) for s in SEEDS]) for a in cells.ARMS}
    rate = {a: {d: {s: sr[a][d][s]["success_rate"] for s in SEEDS} for d in cells.DEPTHS}
            for a in cells.ARMS}
    d11, d9 = _diffs(rate["O"][11], rate["C"][11]), _diffs(rate["O"][9], rate["C"][9])
    c1, p1, m1 = claim1_verdict(d11)
    d2 = _diffs({s: far(end["O"][s]) for s in SEEDS}, {s: far(end["C"][s]) for s in SEEDS})
    c2, p2, m2 = a80.claim_verdict(d2, True)
    guard = guard_trips(d9)
    v = verdict(True, h["ok"], all(t_ok.values()), c1, c2, guard)
    ahead = lambda d: sum(x > 0 for x in d)  # noqa: E731
    lines = [
        f"GATE C (continuity): PASS on {len(SEEDS)} of {len(SEEDS)}",
        _h_line(h),
        "GATE T (trained): " + "; ".join(
            f"{a} {'PASS' if t_ok[a] else 'FAIL'} mean overall "
            f"{st.mean(overall(end[a][s]) for s in SEEDS):.4f}" for a in cells.ARMS)
        + f" vs {GATE_T_THRESHOLD:.5f}",
        f"CLAIM 1 J4V-O - J4V-C d11: {st.mean(rate['O'][11].values()):.4f} vs "
        f"{st.mean(rate['C'][11].values()):.4f}, diff {m1:+.4f}, ahead on {ahead(d11)} of 12, "
        f"p {p1:.4f} -> {c1}",
        f"CLAIM 2 O - C far band: diff {m2:+.4f}, ahead on {ahead(d2)} of 12, p {p2:.4f} -> {c2}",
        f"GUARD J4V-O - J4V-C d9: diff {st.mean(d9):+.4f}, drop p "
        f"{one_sided_p([-d for d in d9]):.4f} -> {'TRIPPED' if guard else 'not tripped'}",
        f"VERDICT: {v}"]
    return {"verdict": v, "lines": lines, "reported": reported(tr, sr, rate, e82_root, out)}


def _band(rec, d) -> float:
    b = rec["gate_l"]["by_distance"][str(d)]
    return b["hit"] - b["chance"]


def reported(tr, sr, rate, e82_root, out) -> list[str]:
    lines = ["REPORTED (never confirmations):"]
    x16 = {s: _read(e82_root / f"exp082_X16k_u{cells.START}_s{s}.json") for s in SEEDS}
    for d in cells.DEPTHS:
        base = {s: _read(e82_root / f"exp082_J4V-X16k_d{d}_s{s}.json")["success_rate"]
                for s in SEEDS}
        lines.append(f"  d{d} vs EXP-082 J4V-X16k {st.mean(base.values()):.4f}: " + "  ".join(
            f"{a} {st.mean(rate[a][d].values()):.4f} ({st.mean(_diffs(rate[a][d], base)):+.4f})"
            for a in cells.ARMS))
    for d in (7, 8, 9, 10, 11):
        cols = [f"u{cells.START} {st.mean(_band(x16[s], d) for s in SEEDS):.4f}"]
        for a in cells.ARMS:
            cols += [f"{a} u{u} {st.mean(_band(tr[a][u][s], d) for s in SEEDS):.4f}"
                     for u in cells.RECORD_LEGS]
        lines.append(f"  margin at distance {d}: " + "  ".join(cols))
    for a in cells.ARMS:
        e = tr[a][cells.END]
        loss = [e[s]["loss_last"] for s in SEEDS if e[s].get("loss_last") is not None]
        sp = [tr[a][u][s]["s_per_update"] for u in cells.RECORD_LEGS for s in SEEDS]
        lines.append(f"  {a}: loss_last {st.mean(loss) if loss else None}  encoder drift "
                     f"{st.mean(e[s]['encoder_drift'] for s in SEEDS):.3f}  s/update "
                     f"{st.mean(sp):.3f}")
        for k in ("optimality", "mean_steps", "goal_fired_frac"):
            lines.append(f"  {a} d11 {k}: {st.mean(sr[a][11][s][k] for s in SEEDS):.4f}")
        w = [sr[a][d][s]["wall_s"] for d in cells.DEPTHS for s in SEEDS]
        lines.append(f"  {a} search wall per cell: {min(w):.0f} to {max(w):.0f} s")
    walks = out / cells.PREFLIGHT_WALKS_NAME
    if walks.exists():
        for row in json.loads(walks.read_text())["rows"]:
            lines.append(f"  walk shares {row['arm']} u{row['update']} (max {row['max_len']}), "
                         f"distances 0-8/9/10/11/12+: {row['shares']}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    res = run_all(args.out_dir)
    for line in res["lines"] + res["reported"]:
        print(line)


if __name__ == "__main__":
    main()
