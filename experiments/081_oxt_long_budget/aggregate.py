"""EXP-081 aggregator. Every gate outcome is a verdict of its own: a failed Gate C prints VOID,
a failed Gate T prints UNRESOLVED (O-Xt16k UNTRAINED), and the VERDICT line is spec section 8's
table. Claim 1 is EXP-080's Claim 1 bar unchanged (claim_verdict); Claim 2 needs p < 0.025 and a
positive mean only.

Spec: docs/superpowers/specs/2026-10-23-exp081-oxt-long-budget-design.md, sections 6 to 11.

Usage:
    .venv/bin/python experiments/081_oxt_long_budget/aggregate.py --stage gates
    .venv/bin/python experiments/081_oxt_long_budget/aggregate.py
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


ev = _load("exp081_evaluate", HERE / "evaluate.py")
cells = ev.cells
a80 = _load("exp080_aggregate", cells.E80 / "aggregate.py")
one_sided_p = a80.one_sided_p
far = a80.far
overall = a80.overall

SEEDS = cells.EVAL_SEEDS
CLAIM_ALPHA = a80.CLAIM_ALPHA
GATE_T_THRESHOLD = a80.GATE_T_THRESHOLD
MINT_SETTINGS = {"n_blocks": cells.N_BLOCKS, "block": cells.BATCH, "max_len": cells.MAX_LEN}
U8, U16 = cells.LEGS


def read_gatec(out, seed) -> dict:
    p = Path(out) / cells.gatec_name(seed)
    if not p.exists():
        raise SystemExit(f"missing {p.name}: run the gatec phase first")
    return json.loads(p.read_text())


def check_leg_record(rec, name, update, seed, gatec_rec) -> None:
    for k, want in (("arm", cells.ARM), ("seed", seed), ("n_updates", update),
                    ("batch", cells.BATCH), ("probe_every", cells.PROBE_EVERY)):
        if rec.get(k) != want:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, spec says {want!r}")
    meta = rec.get("mint_meta", {})
    for k, v in MINT_SETTINGS.items():
        if meta.get(k) != v:
            raise SystemExit(f"{name}: mint {k} is {meta.get(k)!r}, spec says {v!r}")
    if rec.get("mint_sha") != gatec_rec.get("new_mint_sha"):
        raise SystemExit(f"{name}: trained on mint {rec.get('mint_sha')!r}, Gate C certified "
                         f"{gatec_rec.get('new_mint_sha')!r}")
    r = rec.get("resumed_from")
    if not (isinstance(r, int) and r >= cells.START):
        raise SystemExit(f"{name}: resumed_from {r!r} is under {cells.START}; that is a fresh run, "
                         f"not EXP-080's continued")
    if (rec.get("continued_from") or {}).get("experiment") != "EXP-080":
        raise SystemExit(f"{name}: continued_from is {rec.get('continued_from')!r}")
    drift = rec.get("encoder_drift")
    if not (isinstance(drift, float) and drift > 0.0):
        raise SystemExit(f"{name}: the region did not move (encoder_drift {drift!r})")
    got_n = {k: v["n"] for k, v in rec["gate_l"]["by_distance"].items()}
    want_n = {k: v["n"] for k, v in a80._x_train(seed)["gate_l"]["by_distance"].items()}
    if got_n != want_n:
        raise SystemExit(f"{name}: gate_l n per distance {got_n} differs from X's {want_n}")


def read_leg(out, update, seed, gatec_rec) -> dict:
    name = cells.record_name(update, seed)
    rec = a80._read(Path(out) / name)
    check_leg_record(rec, name, update, seed, gatec_rec)
    return rec


def claim2_verdict(diffs, gate_ok) -> tuple[str, float, float]:
    p, m = one_sided_p(diffs), st.mean(diffs)
    if not gate_ok:
        return "VOID", p, m
    return ("CONFIRMED" if p < CLAIM_ALPHA and m > 0 else "REFUTED"), p, m


def verdict(gate_c_ok: bool, t_ok: bool, c1: str, c2: str, search) -> str:
    """Spec section 8, one line."""
    if not gate_c_ok:
        return "VOID"
    if not t_ok:
        return "UNRESOLVED (O-Xt16k UNTRAINED)"
    if c1 == "CONFIRMED":
        if search is None:
            raise SystemExit("Claim 1 is CONFIRMED: run the search cells before reading the verdict")
        return {"PAYS": "BUDGET-LIMITED, PAYS",
                "DOES NOT PAY": "BUDGET-LIMITED, DOES NOT PAY AT DEPTH 11",
                "UNRESOLVED": "BUDGET-LIMITED, SEARCH UNRESOLVED"}[search]
    if c2 == "CONFIRMED":
        return "BUDGET HELPS BUT FALLS SHORT"
    return "ENCODER-LIMITED AT 4X BUDGET"


def stage_gates(out=cells.OUT, e80_root=cells.E80_OUT) -> dict:
    out = Path(out)
    gc = {s: read_gatec(out, s) for s in SEEDS}
    gc_fail = [s for s, r in gc.items() if r.get("pass") is not True]
    if gc_fail:
        return {"gate_c": False, "gate_c_fail": gc_fail}
    r4 = {s: a80.read_train(e80_root, cells.ARM, s) for s in SEEDS}
    r8 = {s: read_leg(out, U8, s, gc[s]) for s in SEEDS}
    r16 = {s: read_leg(out, U16, s, gc[s]) for s in SEEDS}
    x_far = [far(a80._x_train(s)) for s in SEEDS]
    t_ok = a80.gate_t([overall(r16[s]) for s in SEEDS])
    d1 = [far(r16[s]) - x for s, x in zip(SEEDS, x_far)]
    d2 = [far(r16[s]) - far(r4[s]) for s in SEEDS]
    c1 = a80.claim_verdict(d1, t_ok) + (sum(d > 0 for d in d1),)
    c2 = claim2_verdict(d2, t_ok) + (sum(d > 0 for d in d2),)
    return {"gate_c": True, "t": t_ok, "c1": c1, "c2": c2, "run_search": c1[0] == "CONFIRMED",
            "recs": {cells.START: r4, U8: r8, U16: r16}, "x_far": x_far}


def format_stage_gates(res) -> list[str]:
    if not res["gate_c"]:
        return [f"GATE C (continuity): FAIL on seeds {res['gate_c_fail']}",
                "SEARCH: NOT RUN (Gate C failed; the experiment is VOID)"]
    recs = res["recs"]
    pts = "  ".join(f"u{u} far {st.mean(far(recs[u][s]) for s in SEEDS):.4f} overall "
                    f"{st.mean(overall(recs[u][s]) for s in SEEDS):.4f}"
                    for u in (cells.START, U16))
    lines = [f"GATE C (continuity): PASS on {len(SEEDS)} of {len(SEEDS)}",
             f"GATE T (trained) O-Xt16k: {'PASS' if res['t'] else 'FAIL'}, mean overall margin vs "
             f"{GATE_T_THRESHOLD:.5f}; {pts}"]
    for key, label in (("c1", "CLAIM 1 O-Xt16k - X far-band"),
                       ("c2", "CLAIM 2 O-Xt16k - O-Xt4k far-band")):
        v, p, m, wins = res[key]
        lines.append(f"{label}: diff {m:+.4f}, ahead on {wins} of {len(SEEDS)}, p {p:.4f} -> {v}")
    lines.append("SEARCH: RUN (Claim 1 CONFIRMED)" if res["run_search"]
                 else f"SEARCH: NOT RUN (Claim 1 {res['c1'][0]}; the search is not read)")
    return lines


def _read_search(out, seed) -> dict:
    name = ev.record_name(seed)
    r = a80._read(Path(out) / name)
    for k, want in (("arm", ev.ARM), ("judge", cells.ARM), ("depth", ev.DEPTH), ("seed", seed),
                    ("judge_seed", seed), ("k", ev.K), ("k_goal", ev.K), ("mode", "C"),
                    ("no_revisit", True), ("judge_updates", U16)):
        if r.get(k) != want:
            raise SystemExit(f"{name}: {k} is {r.get(k)!r}, spec says {want!r}")
    return r


def _band(rec, d) -> float:
    v = rec["gate_l"]["by_distance"][d]
    return v["hit"] - v["chance"]


def reported(res) -> list[str]:
    recs = res["recs"]
    pts = (cells.START, U8, U16)
    x = {s: a80._x_train(s) for s in SEEDS}
    lines = ["far band per seed (u4000 / u8000 / u16000, X):"]
    for s, xf in zip(SEEDS, res["x_far"]):
        lines.append(f"  s{s}: " + " / ".join(f"{far(recs[u][s]):.4f}" for u in pts)
                     + f"  X {xf:.4f}")
    for d in ("7", "8", "9", "10", "11"):
        lines.append(f"margin at distance {d}: " + "  ".join(
            f"u{u} {st.mean(_band(recs[u][s], d) for s in SEEDS):.4f}" for u in pts)
            + f"  X {st.mean(_band(x[s], d) for s in SEEDS):.4f}")
    for u in pts:
        r = recs[u]
        sp = st.mean(r[s]["final_probe"]["spearman_7_11"] for s in SEEDS)
        mj = "  ".join(f"J@{d} {st.mean(r[s]['final_probe']['mean_j_by_distance'][d] for s in SEEDS):.3f}"
                       for d in ("9", "10", "11"))
        loss = [r[s]["loss_last"] for s in SEEDS if r[s]["loss_last"] is not None]
        lines.append(f"u{u}: overall {st.mean(overall(r[s]) for s in SEEDS):.4f}  spearman_7_11 "
                     f"{sp:.4f}  {mj}  loss_last {st.mean(loss) if loss else None}")
    w = [recs[U16][s]["wall_s"] + recs[U8][s]["wall_s"] for s in SEEDS]
    lines.append(f"training wall per seed, both legs: {min(w):.0f} to {max(w):.0f} s")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    ap.add_argument("--stage", choices=("gates",))
    args = ap.parse_args()
    res = stage_gates(args.out_dir)
    for line in format_stage_gates(res):
        print(line)
    if args.stage == "gates":
        return
    if not res["gate_c"]:
        print("VERDICT: " + verdict(False, False, "VOID", "VOID", None))
        return
    search = None
    if res["run_search"]:
        a = {s: _read_search(args.out_dir, s)["success_rate"] for s in SEEDS}
        b = {s: a80._read(cells.c80.E76_OUT / f"exp076_J4V-X_d11_s{s}.json")["success_rate"]
             for s in SEEDS}
        diffs = a80.a72.pair_diffs(a, b)
        search, p, m = a80.search_reading(diffs, st.mean(a.values()), st.mean(b.values()))
        print(f"SEARCH d11 J4V-O-Xt16k - J4V-X: {st.mean(a.values()):.4f} vs "
              f"{st.mean(b.values()):.4f}, diff {m:+.4f}, ahead on {sum(d > 0 for d in diffs)} "
              f"of 12, p {p:.4f} -> {search}")
    print("VERDICT: " + verdict(True, res["t"], res["c1"][0], res["c2"][0], search))
    print("\nReported (spec section 11, never confirmations):")
    for line in reported(res):
        print("  " + line)


if __name__ == "__main__":
    main()
