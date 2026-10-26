"""EXP-082 aggregator. Every gate outcome is a verdict of its own: a failed Gate C prints VOID, a
failed Gate T prints UNRESOLVED (X16k UNTRAINED), and the VERDICT line is spec section 8's table.
Claim 1 is the depth-11 search (p < 0.025 and mean >= 0.03); Claim 2 is the far band at EXP-080's
Claim 1 bar (claim_verdict). CONFIRMED (SMALL) on either falls through to the next row.

Spec: docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md, sections 6 to 9.

Usage:
    .venv/bin/python experiments/082_x_long_budget/aggregate.py --stage gates
    .venv/bin/python experiments/082_x_long_budget/aggregate.py
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


ev = _load("exp082_evaluate", HERE / "evaluate.py")
cells = ev.cells
a80 = _load("exp080_aggregate", cells.REPO / "experiments" / "080_oracle_ordering" / "aggregate.py")
one_sided_p = a80.one_sided_p
far = a80.far
overall = a80.overall
pair_diffs = a80.a72.pair_diffs

SEEDS = cells.EVAL_SEEDS
CLAIM_ALPHA = a80.CLAIM_ALPHA
C1_MIN = 0.03
GATE_T_THRESHOLD = a80.GATE_T_THRESHOLD
U8, U16 = cells.LEGS
SETTINGS = {"arm": cells.ARM, "batch": cells.BATCH, "sync_every": cells.SYNC_EVERY,
            "probe_every": cells.PROBE_EVERY, "draws": cells.DRAWS, "max_len": 14,
            "readout": "wide", "hidden": 512}


def read_gatec(out, seed) -> dict:
    p = Path(out) / cells.gatec_name(seed)
    if not p.exists():
        raise SystemExit(f"missing {p.name}: run the gatec phase first")
    return json.loads(p.read_text())


def x_record(seed, e75_root=cells.E75_OUT) -> dict:
    return a80._read(Path(e75_root) / cells.c75.record_name(cells.SRC_ARM, seed))


def check_leg_record(rec, name, update, seed, x_rec) -> None:
    for k, want in {**SETTINGS, "seed": seed, "n_updates": update}.items():
        if rec.get(k) != want:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, spec says {want!r}")
    r = rec.get("resumed_from")
    if not (isinstance(r, int) and r >= cells.START):
        raise SystemExit(f"{name}: resumed_from {r!r} is under {cells.START}; that is a fresh "
                         f"run, not EXP-075's X continued")
    if (rec.get("continued_from") or {}).get("experiment") != "EXP-075":
        raise SystemExit(f"{name}: continued_from is {rec.get('continued_from')!r}")
    drift = rec.get("encoder_drift")
    if not (isinstance(drift, float) and drift > 0.0):
        raise SystemExit(f"{name}: the region did not move (encoder_drift {drift!r})")
    if rec.get("n_probe") != x_rec.get("n_probe"):
        raise SystemExit(f"{name}: n_probe {rec.get('n_probe')!r} differs from EXP-075's "
                         f"{x_rec.get('n_probe')!r}")
    got_n = {k: v["n"] for k, v in rec["gate_l"]["by_distance"].items()}
    want_n = {k: v["n"] for k, v in x_rec["gate_l"]["by_distance"].items()}
    if got_n != want_n:
        raise SystemExit(f"{name}: gate_l n per distance {got_n} differs from X's {want_n}")


def read_leg(out, update, seed, x_rec) -> dict:
    name = cells.record_name(update, seed)
    rec = a80._read(Path(out) / name)
    check_leg_record(rec, name, update, seed, x_rec)
    return rec


def check_one_checkout(recs: dict) -> str:
    commits = {r.get("git_commit") for r in recs.values()}
    if len(commits) != 1:
        raise SystemExit(f"the {U16}-update records come from {len(commits)} checkouts "
                         f"{sorted(map(str, commits))}; spec section 6 needs one")
    return commits.pop()


def claim1_verdict(diffs) -> tuple[str, float, float]:
    p, m = one_sided_p(diffs), st.mean(diffs)
    if p < CLAIM_ALPHA and m > 0:
        return ("CONFIRMED" if m >= C1_MIN else "CONFIRMED (SMALL)"), p, m
    return "REFUTED", p, m


def verdict(gate_c_ok: bool, t_ok: bool, c1: str, c2: str) -> str:
    """Spec section 8, one line."""
    if not gate_c_ok:
        return "VOID"
    if not t_ok:
        return "UNRESOLVED (X16k UNTRAINED)"
    if c1 == "CONFIRMED":
        return "BUDGET-LIMITED, PAYS"
    if c2 == "CONFIRMED":
        return "ORDERING IMPROVES, SEARCH DOES NOT PAY"
    return "VALUE ITERATION SATURATED AT 4X BUDGET"


def stage_gates(out=cells.OUT, e75_root=cells.E75_OUT) -> dict:
    out = Path(out)
    gc = {s: read_gatec(out, s) for s in SEEDS}
    gc_fail = [s for s, r in gc.items() if r.get("pass") is not True]
    if gc_fail:
        return {"gate_c": False, "gate_c_fail": gc_fail}
    x = {s: x_record(s, e75_root) for s in SEEDS}
    r8 = {s: read_leg(out, U8, s, x[s]) for s in SEEDS}
    r16 = {s: read_leg(out, U16, s, x[s]) for s in SEEDS}
    commit = check_one_checkout(r16)
    t_ok = a80.gate_t([overall(r16[s]) for s in SEEDS])
    d2 = [far(r16[s]) - far(x[s]) for s in SEEDS]
    c2 = a80.claim_verdict(d2, t_ok) + (sum(d > 0 for d in d2),)
    return {"gate_c": True, "t": t_ok, "c2": c2, "commit": commit,
            "recs": {cells.START: x, U8: r8, U16: r16}}


def format_stage_gates(res) -> list[str]:
    if not res["gate_c"]:
        return [f"GATE C (continuity): FAIL on seeds {res['gate_c_fail']}",
                "SEARCH: NOT RUN (Gate C failed; the experiment is VOID)"]
    recs = res["recs"]
    pts = "  ".join(f"u{u} far {st.mean(far(recs[u][s]) for s in SEEDS):.4f} overall "
                    f"{st.mean(overall(recs[u][s]) for s in SEEDS):.4f}"
                    for u in (cells.START, U8, U16))
    v, p, m, wins = res["c2"]
    return [f"GATE C (continuity): PASS on {len(SEEDS)} of {len(SEEDS)}",
            f"GATE T (trained) X16k: {'PASS' if res['t'] else 'FAIL'}, mean overall margin vs "
            f"{GATE_T_THRESHOLD:.5f}; {pts}",
            f"CLAIM 2 X16k - X far-band: diff {m:+.4f}, ahead on {wins} of {len(SEEDS)}, "
            f"p {p:.4f} -> {v}",
            "SEARCH: RUN (spec section 8: the search runs whatever the probe says)"]


def read_search(out, depth, seed) -> dict:
    name = cells.search_name(depth, seed)
    r = a80._read(Path(out) / name)
    for k, want in (("arm", ev.ARM), ("judge", cells.ARM), ("depth", depth), ("seed", seed),
                    ("judge_seed", seed), ("k", ev.K), ("k_goal", ev.K), ("mode", "C"),
                    ("no_revisit", True), ("judge_updates", U16)):
        if r.get(k) != want:
            raise SystemExit(f"{name}: {k} is {r.get(k)!r}, spec says {want!r}")
    return r


def _rates(recs) -> dict:
    return {s: r["success_rate"] for s, r in recs.items()}


def _contrast(a: dict, b: dict) -> tuple[list, float, float, int]:
    d = pair_diffs(a, b)
    return d, one_sided_p(d), st.mean(d), sum(x > 0 for x in d)


def search_stage(out, e76_root, e81_root) -> dict:
    s = {d: {x: read_search(out, d, x) for x in SEEDS} for d in cells.DEPTHS}
    b = {d: {x: a80._read(Path(e76_root) / f"exp076_J4V-X_d{d}_s{x}.json") for x in SEEDS}
         for d in cells.DEPTHS}
    o81 = {x: a80._read(Path(e81_root) / f"exp081_J4V-O-Xt16k_d11_s{x}.json") for x in SEEDS}
    d11, _, _, wins = _contrast(_rates(s[11]), _rates(b[11]))
    return {"c1": claim1_verdict(d11) + (wins,), "search": s, "base": b, "o81": o81}


def _band(rec, d) -> float:
    v = rec["gate_l"]["by_distance"][d]
    return v["hit"] - v["chance"]


def reported(res, srch) -> list[str]:
    recs, pts = res["recs"], (cells.START, U8, U16)
    lines = ["far band per seed (X u4000 / u8000 / u16000):"]
    for s in SEEDS:
        lines.append(f"  s{s}: " + " / ".join(f"{far(recs[u][s]):.4f}" for u in pts))
    for d in ("7", "8", "9", "10", "11"):
        lines.append(f"margin at distance {d}: " + "  ".join(
            f"u{u} {st.mean(_band(recs[u][s], d) for s in SEEDS):.4f}" for u in pts))
    for u in pts:
        r = recs[u]
        sp = st.mean(r[s]["final_probe"]["spearman_7_11"] for s in SEEDS)
        mj = "  ".join(
            f"J@{d} {st.mean(r[s]['final_probe']['mean_j_by_distance'][d] for s in SEEDS):.3f}"
            for d in ("9", "10", "11"))
        loss = [r[s]["loss_last"] for s in SEEDS if r[s]["loss_last"] is not None]
        lines.append(f"u{u}: overall {st.mean(overall(r[s]) for s in SEEDS):.4f}  "
                     f"spearman_7_11 {sp:.4f}  {mj}  loss_last "
                     f"{st.mean(loss) if loss else None}")
    lines.append(f"encoder drift at u{U16}: "
                 f"{st.mean(recs[U16][s]['encoder_drift'] for s in SEEDS):.3f}")
    a9, b9 = _rates(srch["search"][9]), _rates(srch["base"][9])
    _, p9, m9, w9 = _contrast(a9, b9)
    lines.append(f"depth 9 J4V-X16k - J4V-X: {st.mean(a9.values()):.4f} vs "
                 f"{st.mean(b9.values()):.4f}, diff {m9:+.4f}, ahead on {w9} of 12, "
                 f"p {p9:.4f} (reported, not a claim)")
    for k in ("optimality", "mean_steps", "goal_fired_frac"):
        a = st.mean(srch["search"][11][s][k] for s in SEEDS)
        b = st.mean(srch["base"][11][s][k] for s in SEEDS)
        lines.append(f"d11 {k}: X16k {a:.4f}  J4V-X {b:.4f}")
    a11, o11 = _rates(srch["search"][11]), _rates(srch["o81"])
    _, _, mo, wo = _contrast(a11, o11)
    lines.append(f"d11 X16k - O-Xt16k (EXP-081, oracle ceiling, descriptive): "
                 f"{st.mean(a11.values()):.4f} vs {st.mean(o11.values()):.4f}, diff {mo:+.4f}, "
                 f"ahead on {wo} of 12")
    w = [recs[U8][s]["wall_s"] + recs[U16][s]["wall_s"] for s in SEEDS]
    lines.append(f"training wall per seed, both legs: {min(w):.0f} to {max(w):.0f} s")
    return lines


def run_all(out=cells.OUT, e75_root=cells.E75_OUT, e76_root=cells.E76_OUT,
            e81_root=cells.E81_OUT) -> dict:
    res = stage_gates(out, e75_root)
    lines = format_stage_gates(res)
    if not res["gate_c"]:
        return {"verdict": verdict(False, False, "VOID", "VOID"), "lines": lines, "reported": []}
    srch = search_stage(out, e76_root, e81_root)
    v, p, m, wins = srch["c1"]
    a, b = _rates(srch["search"][11]), _rates(srch["base"][11])
    lines.append(f"CLAIM 1 J4V-X16k - J4V-X d11: {st.mean(a.values()):.4f} vs "
                 f"{st.mean(b.values()):.4f}, diff {m:+.4f}, ahead on {wins} of 12, "
                 f"p {p:.4f} -> {v}")
    return {"verdict": verdict(True, res["t"], v, res["c2"][0]), "lines": lines,
            "reported": reported(res, srch)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    ap.add_argument("--stage", choices=("gates",))
    args = ap.parse_args()
    if args.stage == "gates":
        for line in format_stage_gates(stage_gates(args.out_dir)):
            print(line)
        return
    res = run_all(args.out_dir)
    for line in res["lines"]:
        print(line)
    print("VERDICT: " + res["verdict"])
    if res["reported"]:
        print("\nReported (spec section 9, never confirmations):")
        for line in res["reported"]:
            print("  " + line)


if __name__ == "__main__":
    main()
