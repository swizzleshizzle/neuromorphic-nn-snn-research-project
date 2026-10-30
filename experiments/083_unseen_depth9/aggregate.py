"""EXP-083 aggregator: Gates V, R, S and F, Claims 1 and 2, and the one-line verdict of spec
section 7, with the reported block of section 9. `--stage sets` reads only the sets files, for
the chain to stop on a failed Gate V or a short set before any search runs.

Spec: docs/superpowers/specs/2026-10-30-exp083-unseen-depth9-design.md, sections 5 to 7 and 9.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python experiments/083_unseen_depth9/aggregate.py [--stage sets]
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


cells = _load("exp083_cells", HERE / "cells.py")
a80 = _load("exp080_aggregate", cells.REPO / "experiments" / "080_oracle_ordering" / "aggregate.py")
one_sided_p = a80.one_sided_p
pair_diffs = a80.a72.pair_diffs

SEEDS = list(cells.EVAL_SEEDS)
CLAIM_ALPHA = a80.CLAIM_ALPHA
C1_MIN = 0.10
C2_MIN = 0.05
FLOOR_MARGIN = 0.05
JUDGES = {"J4V-X16k": ("X16k", cells.X16K_END), "J4V-X": ("X", cells.X_END)}


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing record {path}")
    return json.loads(path.read_text())


def claim_verdict(diffs, minimum, floor_ok) -> tuple[str, float, float]:
    p, m = one_sided_p(diffs), st.mean(diffs)
    if not floor_ok:
        return "UNRESOLVED", p, m
    if p < CLAIM_ALPHA and m > 0:
        return ("CONFIRMED" if m >= minimum else "CONFIRMED (SMALL)"), p, m
    return "NOT CONFIRMED", p, m


def verdict(v_ok: bool, r_ok: bool, s_ok: bool, f9_ok: bool, c1: str, c2: str) -> str:
    """Spec section 7, one line."""
    if not (v_ok and r_ok):
        return "VOID"
    if not (s_ok and f9_ok):
        return "UNRESOLVED"
    if c1 == "CONFIRMED":
        if c2 == "CONFIRMED":
            return "DEPTH-9 GAIN CONFIRMED, REACHES DEPTH 10"
        return "DEPTH-9 GAIN CONFIRMED, STOPS SHORT OF DEPTH 10"
    if c1 == "CONFIRMED (SMALL)":
        return "DEPTH-9 GAIN SMALLER THAN REPORTED"
    return "DEPTH-9 GAIN DOES NOT HOLD ON UNSEEN STATES"


def stage_sets(out=cells.OUT) -> dict:
    recs = {s: _read(Path(out) / cells.sets_name(s)) for s in SEEDS}
    v_fail = [s for s, r in recs.items() if r.get("pass") is not True]
    short = [(s, d, len(r["sets"][str(d)])) for s, r in recs.items() for d in cells.DEPTHS
             if r.get("pass") is True and len(r["sets"][str(d)]) != cells.CAP]
    return {"v_ok": not v_fail, "v_fail": v_fail, "s_ok": not short, "short": short,
            "recs": recs}


def _sets_lines(res) -> list[str]:
    return [f"GATE V (replay is the training stream): "
            f"{'PASS' if res['v_ok'] else 'FAIL on seeds ' + str(res['v_fail'])}",
            f"GATE S (sets full at {cells.CAP}): "
            f"{'PASS' if res['s_ok'] else 'FAIL ' + str(res['short'])}"]


def stage_sets_lines(out=cells.OUT) -> list[str]:
    res = stage_sets(out)
    last = ("VERDICT: VOID" if not res["v_ok"] else
            "VERDICT: UNRESOLVED" if not res["s_ok"] else "SETS: OK")
    return _sets_lines(res) + [last]


def read_search(out, arm, depth, seed) -> dict:
    name = cells.search_name(arm, depth, seed)
    r = _read(Path(out) / name)
    judge, updates = JUDGES[arm]
    for k, want in (("arm", arm), ("judge", judge), ("depth", depth), ("seed", seed),
                    ("judge_seed", seed), ("k", 4), ("k_goal", 4), ("mode", "C"),
                    ("no_revisit", True), ("n", cells.CAP), ("limit_states", None),
                    ("judge_updates", updates)):
        if r.get(k) != want:
            raise SystemExit(f"{name}: {k} is {r.get(k)!r}, spec says {want!r}")
    return r


def _mean(recs, key) -> float:
    return st.mean(r[key] for r in recs.values())


def run_all(out=cells.OUT, e76_root=cells.E76_OUT, e82_root=cells.E82_OUT) -> dict:
    out = Path(out)
    ss = stage_sets(out)
    lines = _sets_lines(ss)
    if not ss["v_ok"]:
        lines.append("VERDICT: VOID")
        return {"verdict": "VOID", "lines": lines}
    if not ss["s_ok"]:
        # A short set's search records hold n < 200, which read_search refuses.
        lines.append("VERDICT: UNRESOLVED")
        return {"verdict": "UNRESOLVED", "lines": lines}
    gr = _read(out / cells.GATE_R_NAME)
    r_ok = gr.get("pass") is True
    lines.append(f"GATE R (EXP-082 cell reproduces): {'PASS' if r_ok else 'FAIL'}, solved "
                 f"{gr.get('solved')} want {gr.get('expected_solved')}")
    srch = {(a, d): {s: read_search(out, a, d, s) for s in SEEDS}
            for a in cells.ARMS for d in cells.DEPTHS}
    commits = {r["git_commit"] for recs in srch.values() for r in recs.values()}
    commits.add(gr.get("git_commit"))
    if len(commits) != 1:
        raise SystemExit(f"records come from more than one checkout: {sorted(map(str, commits))}")
    floors = {d: {s: _read(out / cells.floor_name(d, s)) for s in SEEDS} for d in cells.DEPTHS}
    claims, f_ok = {}, {}
    for d, minimum in ((9, C1_MIN), (10, C2_MIN)):
        a = {s: r["success_rate"] for s, r in srch[("J4V-X16k", d)].items()}
        b = {s: r["success_rate"] for s, r in srch[("J4V-X", d)].items()}
        fl = _mean(floors[d], "success_rate")
        best = max(st.mean(a.values()), st.mean(b.values()))
        f_ok[d] = best >= fl + FLOOR_MARGIN
        diffs = pair_diffs(a, b)
        claims[d] = claim_verdict(diffs, minimum, f_ok[d]) + (sum(x > 0 for x in diffs),)
        lines.append(f"GATE F d{d}: {'PASS' if f_ok[d] else 'FAIL'}, best arm {best:.4f} vs "
                     f"floor {fl:.4f} + {FLOOR_MARGIN}")
    for n, d in ((1, 9), (2, 10)):
        v, p, m, wins = claims[d]
        a = st.mean(r["success_rate"] for r in srch[("J4V-X16k", d)].values())
        b = st.mean(r["success_rate"] for r in srch[("J4V-X", d)].values())
        lines.append(f"CLAIM {n} J4V-X16k - J4V-X d{d}: {a:.4f} vs {b:.4f}, diff {m:+.4f}, "
                     f"ahead on {wins} of {len(SEEDS)}, p {p:.4f} -> {v}")
    v = verdict(ss["v_ok"], r_ok, ss["s_ok"], f_ok[9], claims[9][0], claims[10][0])
    lines.append(f"VERDICT: {v}")
    lines += reported(srch, floors, ss["recs"], e76_root, e82_root)
    return {"verdict": v, "lines": lines, "c1": claims[9], "c2": claims[10]}


def reported(srch, floors, sets_recs, e76_root, e82_root) -> list[str]:
    old_x = st.mean(_read(Path(e76_root) / f"exp076_J4V-X_d9_s{s}.json")["success_rate"]
                    for s in SEEDS)
    old_k = st.mean(_read(Path(e82_root) / f"exp082_J4V-X16k_d9_s{s}.json")["success_rate"]
                    for s in SEEDS)
    new_x = _mean(srch[("J4V-X", 9)], "success_rate")
    new_k = _mean(srch[("J4V-X16k", 9)], "success_rate")
    out = ["REPORTED (never confirmations):",
           f"  d9 new sets vs old held-out: J4V-X16k {new_k:.4f} vs {old_k:.4f}; "
           f"J4V-X {new_x:.4f} vs {old_x:.4f}"]
    for (a, d), recs in srch.items():
        out.append(f"  {a} d{d}: optimality {_mean(recs, 'optimality'):.4f}  mean steps "
                   f"{_mean(recs, 'mean_steps'):.2f}  goal-fired "
                   f"{_mean(recs, 'goal_fired_frac'):.4f}  fallback "
                   f"{_mean(recs, 'fallback_frac'):.4f}")
    for d in cells.DEPTHS:
        out.append(f"  random-policy floor d{d}: {_mean(floors[d], 'success_rate'):.4f}")
    for s in SEEDS:
        r = sets_recs[s]
        out.append(f"  seed {s}: n_seen {r['n_seen']}  candidates d9 {r['candidates']['9']} "
                   f"d10 {r['candidates']['10']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sets", "all"], default="all")
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    lines = (stage_sets_lines(args.out_dir) if args.stage == "sets"
             else run_all(args.out_dir)["lines"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
