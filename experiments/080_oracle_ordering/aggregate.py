"""EXP-080 aggregator. Every gate outcome is a verdict of its own: Gate 0 or Gate V failing, or
O-raw failing Gate T, prints UNRESOLVED; O-Xt failing Gate T prints UNRESOLVED (O-Xt UNTRAINED);
a confirmed gain under 0.05 prints CONFIRMED (SMALL) and its verdict line says so; a search
contrast on the floor or ceiling prints SEARCH UNRESOLVED. The one-line VERDICT is spec section
8's table.

Spec: docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md, sections 5
to 9.

The sign-flip test and Gate 1 are EXP-070's, seed pairing is EXP-072's, Gate T is EXP-074's
Gate L verdict, s/update is EXP-075's, and the far band is EXP-079's margin_band, all reused
through importlib.

Usage:
    .venv/bin/python experiments/080_oracle_ordering/aggregate.py --pilot-report
    .venv/bin/python experiments/080_oracle_ordering/aggregate.py --stage gate0
    .venv/bin/python experiments/080_oracle_ordering/aggregate.py --stage gates
    .venv/bin/python experiments/080_oracle_ordering/aggregate.py
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
a72 = _load("exp072_aggregate", REPO / "experiments" / "072_p3v_frontier" / "aggregate.py")
a74 = _load("exp074_aggregate", REPO / "experiments" / "074_wide_judge" / "aggregate.py")
a75 = _load("exp075_aggregate", REPO / "experiments" / "075_wide_region" / "aggregate.py")
a79 = _load("exp079_aggregate", REPO / "experiments" / "079_judge_trained_deep" / "aggregate.py")
cells = _load("exp080_cells", HERE / "cells.py")
ev = _load("exp080_evaluate", HERE / "evaluate.py")
one_sided_p = a70.one_sided_p
margin_band = a79.margin_band

SEEDS = cells.EVAL_SEEDS
ARMS = cells.ARMS
CLAIM_ALPHA = 0.05 / 2
CLAIM_MIN = 0.05
GATE_V_ALPHA = 0.05
GATE_V_MIN = 0.05
SEARCH_ALPHA = 0.05
SEARCH_MIN = 0.03
# Spec section 5: half X's 12-seed mean overall margin (EXP-075's committed records, 0.27127...).
GATE_T_THRESHOLD = 0.1356358024691358
# Spec section 6: X's mean pilot far-band margin on seeds 12 and 13 (0.09532...) plus 0.05.
STOP_BAR = 0.14532407407407405
MINT_SETTINGS = {"n_blocks": cells.N_UPDATES, "block": cells.BATCH, "max_len": cells.MAX_LEN}
X_SETTINGS = {"n_updates": cells.N_UPDATES, "batch": cells.BATCH,
              "probe_every": cells.PROBE_EVERY}
RAW_SETTINGS = {"raw_states": cells.RAW_STATES, "raw_epochs": cells.RAW_EPOCHS,
                "raw_batch": cells.RAW_BATCH, "raw_hidden": cells.RAW_HIDDEN}

# Set by the dated post-pilot amendment (spec section 6), before any seed 0-11 is minted: the
# worker counts. The launcher refuses every phase but check and the pilot phases while it is None.
AMENDMENT: dict | None = None


def require_amendment() -> None:
    if AMENDMENT is None:
        raise SystemExit("AMENDMENT is unset: the dated post-pilot amendment sets it; no gate or "
                         "claim may be read before then")


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path.name}: the run is incomplete")
    r = json.loads(path.read_text())
    if r.get("limit_states") is not None:
        raise SystemExit(f"{path.name} is a smoke record and may not enter a verdict")
    return r


def _x_train(seed, pilot=False) -> dict:
    d = cells.E75_PILOT if pilot else cells.E75_OUT
    return _read(d / cells.c75.record_name("X", seed))


def check_train_record(rec, name, arm, seed) -> None:
    for k, want in (("arm", arm), ("seed", seed)):
        if rec.get(k) != want:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, the file name says {want!r}")
    meta = rec.get("mint_meta", {})
    for k, v in MINT_SETTINGS.items():
        if meta.get(k) != v:
            raise SystemExit(f"{name}: mint {k} is {meta.get(k)!r}, spec says {v!r}")
    for k, v in (RAW_SETTINGS if arm == "O-raw" else X_SETTINGS).items():
        if rec.get(k) != v:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, spec says {v!r}")
    drift = rec.get("encoder_drift")
    if arm == "O-Xf" and drift != 0.0:
        raise SystemExit(f"{name}: O-Xf's region moved (encoder_drift {drift!r}); the freeze "
                         f"did not hold")
    if arm == "O-Xt" and not (isinstance(drift, float) and drift > 0.0):
        raise SystemExit(f"{name}: O-Xt's region did not move (encoder_drift {drift!r}); the "
                         f"gradient never reached it")


def read_train(out, arm, seed) -> dict:
    name = cells.record_name(arm, seed)
    rec = _read(Path(out) / name)
    check_train_record(rec, name, arm, seed)
    return rec


def check_mint_pairing(recs) -> None:
    """Every arm of a seed trained on the same mint (spec section 3)."""
    for seed in sorted(next(iter(recs.values()))):
        shas = {a: recs[a][seed]["mint_sha"] for a in recs}
        if len(set(shas.values())) != 1:
            raise SystemExit(f"seed {seed}: arms trained on different mints {shas}")


def far(rec) -> float:
    return margin_band(rec["gate_l"])


def overall(rec) -> float:
    return rec["gate_l"]["margin"]


def claim_verdict(diffs, gate_ok) -> tuple[str, float, float]:
    p, m = one_sided_p(diffs), st.mean(diffs)
    if not gate_ok:
        return "VOID", p, m
    if p < CLAIM_ALPHA and m > 0:
        return ("CONFIRMED" if m >= CLAIM_MIN else "CONFIRMED (SMALL)"), p, m
    return "REFUTED", p, m


def gate_v(oraw_far, x_far) -> dict:
    diffs = [a - b for a, b in zip(oraw_far, x_far)]
    p, m = one_sided_p(diffs), st.mean(diffs)
    return {"diffs": diffs, "p": p, "mean": m, "wins": sum(d > 0 for d in diffs),
            "pass": bool(p < GATE_V_ALPHA and m >= GATE_V_MIN)}


def gate_t(margins) -> bool:
    return a74.gate_l_verdict(margins, GATE_T_THRESHOLD)


def search_reading(diffs, a_mean, b_mean) -> tuple[str, float, float]:
    p, m = one_sided_p(diffs), st.mean(diffs)
    if a70.gate1_verdict(a_mean, b_mean) == "UNRESOLVED":
        return "UNRESOLVED", p, m
    return ("PAYS" if p < SEARCH_ALPHA and m >= SEARCH_MIN else "DOES NOT PAY"), p, m


def verdict(g0: bool, gv: bool, t: dict, claim1: str, search) -> str:
    """Spec section 8, one line."""
    if not (g0 and gv and t["O-raw"]):
        return "UNRESOLVED"
    if not t["O-Xt"]:
        return "UNRESOLVED (O-Xt UNTRAINED)"
    if claim1 == "REFUTED":
        return "ENCODER-LIMITED"
    if claim1 == "CONFIRMED (SMALL)":
        return "ORDERING LEARNABLE BUT SMALL"
    if claim1 != "CONFIRMED":
        raise SystemExit(f"Claim 1 is {claim1!r} with every gate passed; no verdict row fits")
    if search is None:
        raise SystemExit("Claim 1 is CONFIRMED: run the search cells before reading the verdict")
    return {"PAYS": "ORDERING LEARNABLE, PAYS",
            "DOES NOT PAY": "ORDERING LEARNABLE, DOES NOT PAY AT DEPTH 11",
            "UNRESOLVED": "ORDERING LEARNABLE, SEARCH UNRESOLVED"}[search]


def gate0(out) -> str:
    p = Path(out) / ev.gate0_record_name(0)
    if not p.exists():
        raise SystemExit(f"missing {p.name}: run the gate0 phase first")
    got = json.loads(p.read_text())["gate_l"]
    return "PASS" if got == _x_train(0)["gate_l"] else "FAIL"


def format_gate0(v: str) -> str:
    return f"GATE 0 (X seed 0 re-measured through the mint): {v}"


def stage_gates(out) -> dict:
    require_amendment()
    out = Path(out)
    g0 = gate0(out)
    recs = {a: {s: read_train(out, a, s) for s in SEEDS} for a in ARMS}
    check_mint_pairing(recs)
    x_far = [far(_x_train(s)) for s in SEEDS]
    v = gate_v([far(recs["O-raw"][s]) for s in SEEDS], x_far)
    t = {a: gate_t([overall(recs[a][s]) for s in SEEDS]) for a in ARMS}
    t_mean = {a: st.mean(overall(recs[a][s]) for s in SEEDS) for a in ARMS}
    base = g0 == "PASS" and v["pass"]

    def claim(arm):
        diffs = [far(recs[arm][s]) - x for s, x in zip(SEEDS, x_far)]
        return claim_verdict(diffs, base and t[arm]) + (sum(d > 0 for d in diffs),)

    c1, c2 = claim("O-Xt"), claim("O-Xf")
    return {"g0": g0, "v": v, "t": t, "t_mean": t_mean, "c1": c1, "c2": c2,
            "run_search": c1[0] == "CONFIRMED", "recs": recs, "x_far": x_far}


def format_stage_gates(res) -> list[str]:
    v = res["v"]
    lines = [format_gate0(res["g0"]),
             f"GATE V (protocol resolves): {'PASS' if v['pass'] else 'FAIL'}, O-raw - X far-band "
             f"mean {v['mean']:+.4f}, ahead on {v['wins']} of {len(v['diffs'])}, p {v['p']:.4f}"]
    for a in ARMS:
        lines.append(f"GATE T (trained) {a}: {'PASS' if res['t'][a] else 'FAIL'}, mean overall "
                     f"margin {res['t_mean'][a]:.4f} vs {GATE_T_THRESHOLD:.5f}")
    for key, label in (("c1", "CLAIM 1 O-Xt - X far-band"), ("c2", "CLAIM 2 O-Xf - X far-band")):
        verdict_, p, m, wins = res[key]
        lines.append(f"{label}: diff {m:+.4f}, ahead on {wins} of 12, p {p:.4f} -> {verdict_}")
    lines.append("SEARCH: RUN (Claim 1 CONFIRMED)" if res["run_search"]
                 else f"SEARCH: NOT RUN (Claim 1 {res['c1'][0]}; the search is not read)")
    return lines


def _read_search(out, seed) -> dict:
    name = ev.record_name(seed)
    r = _read(Path(out) / name)
    for k, want in (("arm", ev.ARM), ("judge", "O-Xt"), ("depth", ev.DEPTH), ("seed", seed),
                    ("judge_seed", seed), ("k", ev.K), ("k_goal", ev.K), ("mode", "C"),
                    ("no_revisit", True), ("judge_updates", cells.N_UPDATES)):
        if r.get(k) != want:
            raise SystemExit(f"{name}: {k} is {r.get(k)!r}, spec says {want!r}")
    return r


def pilot_report(pilot_dir=cells.PILOT_OUT) -> dict:
    pilot_dir = Path(pilot_dir)
    seeds = cells.PILOT_SEEDS
    recs = {a: {s: read_train(pilot_dir, a, s) for s in seeds} for a in ARMS}
    x = {s: _x_train(s, pilot=True) for s in seeds}
    t = pilot_dir / ev.time_record_name(ev.TIME_JUDGE_SEED, ev.TIME_CELL_SEED)
    timing = json.loads(t.read_text()) if t.exists() else None
    fars = {a: st.mean(far(recs[a][s]) for s in seeds) for a in ARMS}
    return {
        "far": fars, "overall": {a: st.mean(overall(recs[a][s]) for s in seeds) for a in ARMS},
        "x_far": st.mean(far(x[s]) for s in seeds),
        "x_overall": st.mean(overall(x[s]) for s in seeds),
        "far_by_seed": {a: {s: far(recs[a][s]) for s in seeds} for a in ARMS},
        "stop": fars["O-raw"] < STOP_BAR, "stop_bar": STOP_BAR,
        "seconds_per_update": {a: {s: a75.seconds_per_update(recs[a][s]) for s in seeds}
                               for a in ("O-Xf", "O-Xt")},
        "oraw_wall_s": {s: recs["O-raw"][s]["wall_s"] for s in seeds},
        "encoder_drift": {a: {s: recs[a][s]["encoder_drift"] for s in seeds}
                          for a in ("O-Xf", "O-Xt")},
        "timing": None if timing is None else {"n": timing["n"], "wall_s": timing["wall_s"],
                                               "solved": timing["solved"]},
    }


def _mean(recs, field):
    return st.mean(r[field] for r in recs.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    ap.add_argument("--pilot-dir", type=Path, default=cells.PILOT_OUT)
    ap.add_argument("--pilot-report", action="store_true")
    ap.add_argument("--stage", choices=("gate0", "gates"))
    args = ap.parse_args()
    if args.pilot_report:
        rep = pilot_report(args.pilot_dir)
        for k, v in rep.items():
            print(f"{k}: {v}")
        if rep["stop"]:
            print(f"STOP: O-raw's mean pilot far-band margin {rep['far']['O-raw']:.4f} is below "
                  f"{STOP_BAR:.5f}. Report; there is no amendment (spec section 6).")
        return
    require_amendment()
    out = args.out_dir
    if args.stage == "gate0":
        print(format_gate0(gate0(out)))
        return
    res = stage_gates(out)
    for line in format_stage_gates(res):
        print(line)
    if args.stage == "gates":
        return
    search = None
    srecs = x76 = None
    if res["run_search"]:
        srecs = {s: _read_search(out, s) for s in SEEDS}
        x76 = {s: _read(cells.E76_OUT / f"exp076_J4V-X_d11_s{s}.json") for s in SEEDS}
        a = {s: r["success_rate"] for s, r in srecs.items()}
        b = {s: r["success_rate"] for s, r in x76.items()}
        diffs = a72.pair_diffs(a, b)
        search, p, m = search_reading(diffs, st.mean(a.values()), st.mean(b.values()))
        print(f"SEARCH d11 J4V-O-Xt - J4V-X: {st.mean(a.values()):.4f} vs "
              f"{st.mean(b.values()):.4f}, diff {m:+.4f}, ahead on {sum(d > 0 for d in diffs)} "
              f"of 12, p {p:.4f} -> {search}")
    print("VERDICT: " + verdict(res["g0"] == "PASS", res["v"]["pass"], res["t"], res["c1"][0],
                                search))

    print("\nReported (spec section 9, never confirmations):")
    s79 = {s: json.loads((cells.E79_OUT / f"exp079_train_S_s{s}.json").read_text())
           for s in SEEDS}
    d = [far(res["recs"]["O-Xt"][s]) - far(s79[s]) for s in SEEDS]
    print(f"  O-Xt - S far-band: {st.mean(d):+.4f}, ahead on {sum(x > 0 for x in d)} of 12, "
          f"p {one_sided_p(d):.4f}")
    x = {s: _x_train(s) for s in SEEDS}
    for dist in ("9", "10", "11"):
        row = {a: st.mean(res["recs"][a][s]["gate_l"]["by_distance"][dist]["hit"]
                          - res["recs"][a][s]["gate_l"]["by_distance"][dist]["chance"]
                          for s in SEEDS) for a in ARMS}
        xm = st.mean(x[s]["gate_l"]["by_distance"][dist]["hit"]
                     - x[s]["gate_l"]["by_distance"][dist]["chance"] for s in SEEDS)
        print(f"  margin at distance {dist}: " + "  ".join(f"{a} {v:.4f}" for a, v in row.items())
              + f"  X {xm:.4f}")
    for a in ARMS:
        sp = _mean({s: res["recs"][a][s]["final_probe"] for s in SEEDS}, "spearman_7_11")
        mj = "  ".join(
            f"J@{dd} {st.mean(res['recs'][a][s]['final_probe']['mean_j_by_distance'][dd] for s in SEEDS):.3f}"
            for dd in ("9", "10", "11"))
        print(f"  {a}: spearman_7_11 {sp:.4f}  {mj}  loss_last "
              f"{_mean(res['recs'][a], 'loss_last'):.4f}")
    if srecs is not None:
        s79e = {s: json.loads((cells.E79_OUT / f"exp079_J4V-S_d11_s{s}.json").read_text())
                for s in SEEDS}
        for label, r in (("J4V-O-Xt", srecs), ("J4V-X", x76), ("J4V-S", s79e)):
            print(f"  d11 {label}: success {_mean(r, 'success_rate'):.4f}  mean steps "
                  f"{_mean(r, 'mean_steps'):.2f}  optimality {_mean(r, 'optimality'):.3f}")


if __name__ == "__main__":
    main()
