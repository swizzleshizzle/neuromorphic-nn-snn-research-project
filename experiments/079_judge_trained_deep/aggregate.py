"""EXP-079 aggregator. Gates 0, L and E are VERDICTS: a claim whose gate failed prints VOID, a
contrast on the floor or ceiling prints UNRESOLVED, and a confirmed gain under 0.01 prints
CONFIRMED (SMALL). Gate M and Gate R are reported; Gate M binds the reading, and the fixed
readings of spec section 8 are printed as lines of their own.

Spec: docs/superpowers/specs/2026-10-18-exp079-judge-trained-where-scrambles-live-design.md,
sections 6 to 8.

The sign-flip test and Gate 1 are EXP-070's, Gate 0(b)'s field comparison and Gate R are
EXP-071's, seed pairing is EXP-072's, Gate L's verdict is EXP-074's, and the pilot stop bar is
EXP-075's X Gate L threshold, all reused through importlib. Alpha is 0.05 / 2.

Usage:
    .venv/bin/python experiments/079_judge_trained_deep/aggregate.py --pilot-report
    .venv/bin/python experiments/079_judge_trained_deep/aggregate.py --stage cont
    .venv/bin/python experiments/079_judge_trained_deep/aggregate.py --stage train
    .venv/bin/python experiments/079_judge_trained_deep/aggregate.py
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
a74 = _load("exp074_aggregate", REPO / "experiments" / "074_wide_judge" / "aggregate.py")
a75 = _load("exp075_aggregate", REPO / "experiments" / "075_wide_region" / "aggregate.py")
cells = _load("exp079_cells", HERE / "cells.py")
ev = _load("exp079_evaluate", HERE / "evaluate.py")
one_sided_p = a70.one_sided_p

SEEDS = cells.EVAL_SEEDS
DEPTHS = ev.DEPTHS
ALPHA = 0.05 / 2
SMALL = 0.01
GATE_M_ALPHA = 0.05
M_BAND = (9, 11)
# Spec section 6: S must clear the Gate L threshold EXP-075's X cleared, or the pilot stops.
X_PILOT_GATE_L = a75.GATE_L_THRESHOLD["X"]
TRAIN_SETTINGS = {"n_updates": 4000, "batch": 1000, "sync_every": 100, "probe_every": 250,
                  "draws": 1, "max_len": cells.MAX_LEN}
# Evaluation runs without a checkpoint, so these are the only session fields a record has.
DET_IGNORED = ("wall_s", "git_commit")

# Set together by the dated post-pilot amendment (spec section 6), before any seed 0-11 trains:
# AMENDMENT = {"date", "train_workers", "eval_workers", "spec_section"}, and Gate L's threshold,
# half S's mean pilot margin. The launcher refuses every phase but check and pilot while either
# is None.
AMENDMENT: dict | None = None
GATE_L_THRESHOLD: float | None = None


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


def check_train_record(rec, name) -> None:
    for k, v in TRAIN_SETTINGS.items():
        if rec.get(k) != v:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, spec says {v!r}")
    for k, v in (("arm", cells.ARM), ("readout", "wide"),
                 ("hidden", cells.c75.HIDDEN[cells.BASE_ARM])):
        if rec.get(k) != v:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, spec says {v!r}")
    if rec.get("walk_max_lens") != [cells.MAX_LEN]:
        raise SystemExit(
            f"{name}: walk_max_lens is {rec.get('walk_max_lens')!r}; the walks must be observed "
            f"at exactly [{cells.MAX_LEN}]. An empty list means the session that wrote this record "
            f"trained no update (its record write was lost after the final checkpoint): delete "
            f"the record and judge_S_s<seed>/ and retrain that seed.")


def margin_band(gate_l, lo=M_BAND[0], hi=M_BAND[1]) -> float:
    """Gate L's margin restricted to probe distances lo..hi, pooled by n."""
    rows = [v for d, v in gate_l["by_distance"].items() if lo <= int(d) <= hi]
    n = sum(v["n"] for v in rows)
    return (sum(v["hit"] * v["n"] for v in rows) - sum(v["chance"] * v["n"] for v in rows)) / n


def gate_m(s_recs: dict, x_recs: dict) -> dict:
    """Gate M (spec section 7): per seed, S's margin at probe distances 9 to 11 minus X's on the
    same probe states, by exact one-sided sign-flip. The probes must agree in n per distance."""
    for s in sorted(s_recs):
        bs, bx = s_recs[s]["gate_l"]["by_distance"], x_recs[s]["gate_l"]["by_distance"]
        if {d: v["n"] for d, v in bs.items()} != {d: v["n"] for d, v in bx.items()}:
            raise SystemExit(f"seed {s}: S's Gate L probe differs from X's in n per distance; "
                             f"Gate M needs the same probe states")
    diffs = [margin_band(s_recs[s]["gate_l"]) - margin_band(x_recs[s]["gate_l"])
             for s in sorted(s_recs)]
    p = one_sided_p(diffs)
    m = st.mean(diffs)
    return {"diffs": diffs, "mean": m, "p": p, "wins": sum(d > 0 for d in diffs),
            "above": bool(p < GATE_M_ALPHA and m > 0)}


def _x_train(seed, pilot=False) -> dict:
    d = cells.E75_PILOT if pilot else cells.E75_OUT
    return _read(d / cells.c75.record_name(cells.BASE_ARM, seed))


def gate_m_continuity(out) -> str:
    """X seed 0's Gate L, re-measured here by evaluate.gatel_x_cell, must equal EXP-075's
    committed training record exactly, or Gate M's X side is a number today's code does not
    produce."""
    p = Path(out) / ev.gatel_x_record_name(0)
    if not p.exists():
        raise SystemExit(f"missing {p.name}: run the cont phase (Gate M continuity) first")
    got = json.loads(p.read_text())["gate_l"]
    return "PASS" if got == _x_train(0)["gate_l"] else "FAIL"


def pilot_amendment(pilot_dir=cells.PILOT_OUT) -> dict:
    """The numbers the dated amendment states (spec section 6), from the two S pilot records,
    with EXP-075's X pilot judges on the same seeds alongside."""
    pilot_dir = Path(pilot_dir)
    s, x = {}, {}
    for seed in cells.PILOT_SEEDS:
        name = cells.record_name(seed)
        s[seed] = _read(pilot_dir / name)
        check_train_record(s[seed], name)
        x[seed] = _x_train(seed, pilot=True)
    margins = {seed: s[seed]["gate_l"]["margin"] for seed in cells.PILOT_SEEDS}
    mean = st.mean(margins.values())

    def by_d(rec):
        return {d: v["hit"] - v["chance"] for d, v in sorted(rec["gate_l"]["by_distance"].items(),
                                                             key=lambda kv: int(kv[0]))}

    return {"margins": margins, "mean_margin": mean, "x_pilot_gate_l": X_PILOT_GATE_L,
            "stop": mean < X_PILOT_GATE_L, "gate_l_threshold": mean / 2,
            "by_distance_margin": {seed: {"S": by_d(s[seed]), "X": by_d(x[seed])}
                                   for seed in cells.PILOT_SEEDS},
            "seconds_per_update": {seed: a75.seconds_per_update(s[seed])
                                   for seed in cells.PILOT_SEEDS},
            "train_hist": {seed: s[seed]["train_hist"] for seed in cells.PILOT_SEEDS},
            "train_hist_updates": {seed: s[seed]["train_hist_updates"]
                                   for seed in cells.PILOT_SEEDS}}


def require_amendment() -> None:
    if AMENDMENT is None or GATE_L_THRESHOLD is None:
        raise SystemExit("AMENDMENT or GATE_L_THRESHOLD is unset: the dated post-pilot "
                         "amendment sets both; no gate or claim may be read before then")


def stage_train(out) -> dict:
    """Gates L and E, and Gate M (reported), from the 12 S training records alone."""
    require_amendment()
    out = Path(out)
    recs = {}
    for seed in SEEDS:
        name = cells.record_name(seed)
        recs[seed] = _read(out / name)
        check_train_record(recs[seed], name)
    margins = [recs[seed]["gate_l"]["margin"] for seed in SEEDS]
    return {"l": {"mean": st.mean(margins), "threshold": GATE_L_THRESHOLD,
                  "p": one_sided_p(margins),
                  "pass": a74.gate_l_verdict(margins, GATE_L_THRESHOLD)},
            "e": all(recs[seed]["encoder_drift"] > 0 for seed in SEEDS),
            "m": gate_m(recs, {seed: _x_train(seed) for seed in SEEDS}),
            "m_cont": gate_m_continuity(out),
            "recs": recs}


def format_stage_train(res) -> list[str]:
    l, m = res["l"], res["m"]
    return [f"GATE L (VI worked): {'PASS' if l['pass'] else 'FAIL'}, mean margin "
            f"{l['mean']:.4f} vs threshold {l['threshold']:.4f}, p {l['p']:.4f}",
            f"GATE E (encoders trained): {'PASS' if res['e'] else 'FAIL'}",
            f"GATE M continuity (X seed 0 Gate L re-measured here): {res['m_cont']}",
            f"GATE M (reported, binds the reading): S - X margin at probe distances 9-11, mean "
            f"{m['mean']:+.4f}, S ahead on {m['wins']} of {len(m['diffs'])}, p {m['p']:.4f} -> "
            f"{'ABOVE 0' if m['above'] else 'NOT ABOVE 0'}"]


def check_eval_record(rec, path) -> None:
    name = Path(path).name
    arm = rec.get("arm")
    if arm not in ev.ARMS:
        raise SystemExit(f"{name}: arm is {arm!r}, expected one of {list(ev.ARMS)}")
    if rec.get("k") != ev.K or rec.get("k_goal") != ev.K:
        raise SystemExit(f"{name}: k is {rec.get('k')!r} and k_goal {rec.get('k_goal')!r}; "
                         f"spec says {ev.K} and {ev.K}")
    if rec.get("mode") != "C" or rec.get("no_revisit") is not True:
        raise SystemExit(f"{name}: mode {rec.get('mode')!r}, no_revisit "
                         f"{rec.get('no_revisit')!r}; spec says C with no-revisit")
    if rec.get("judge_updates") != cells.N_UPDATES:
        raise SystemExit(f"{name}: judge_updates is {rec.get('judge_updates')!r}, spec says "
                         f"{cells.N_UPDATES}: it was evaluated from a partly trained judge")
    # EXP-076's J4V-X records predate the judge field; this experiment's S records must carry it.
    if arm == "J4V-S" and rec.get("judge") != "S":
        raise SystemExit(f"{name}: judge is {rec.get('judge')!r}, a J4V-S record must say 'S'")


def _read_cell(path: Path, arm: str, depth: int, seed: int) -> dict:
    """Read a record, check its spec fields, and check it is the cell its filename claims."""
    r = _read(path)
    check_eval_record(r, path)
    for field, want in (("arm", arm), ("depth", depth), ("seed", seed)):
        if r.get(field) != want:
            raise SystemExit(f"{Path(path).name}: {field} is {r.get(field)!r}, the file name "
                             f"says {want!r}")
    return r


def load_records(out_dir) -> dict:
    """(arm, depth) -> {seed: record}. J4V-X from EXP-076's committed records (licensed by
    Gate 0(b)); J4V-S from this experiment's outputs."""
    out_dir = Path(out_dir)
    recs = {}
    for d in DEPTHS:
        recs[("J4V-X", d)] = {s: _read_cell(cells.E76_OUT / f"exp076_J4V-X_d{d}_s{s}.json",
                                              "J4V-X", d, s) for s in SEEDS}
        recs[("J4V-S", d)] = {s: _read_cell(out_dir / ev.record_name("J4V-S", d, s),
                                              "J4V-S", d, s) for s in SEEDS}
    return recs


def rates_of(recs) -> dict:
    return {key: {s: r["success_rate"] for s, r in by_seed.items()}
            for key, by_seed in recs.items()}


def stage_cont(out) -> str:
    """Gate 0(b): J4V-X at seed 0, depths 9 and 11, re-run here, against EXP-076's records."""
    out = Path(out)
    return a71.gate0b_verdict(
        {("J4V-X", d): _read_cell(out / ev.record_name("J4V-X", d, 0), "J4V-X", d, 0)
         for d in DEPTHS},
        {("J4V-X", d): _read(cells.E76_OUT / f"exp076_J4V-X_d{d}_s0.json") for d in DEPTHS})


def format_stage_cont(verdict) -> str:
    return f"GATE 0(b) J4V-X continuity with EXP-076 (d9, d11, seed 0): {verdict}"


def gate0a(out_dir, det_dir) -> str:
    """Gate 0(a): the det re-run of J4V-S d9 s0 equals the full-run record in every field but
    the session fields."""
    recs = []
    for d in (Path(out_dir), Path(det_dir)):
        p = d / ev.record_name("J4V-S", 9, 0)
        if not p.exists():
            raise SystemExit(f"missing {p}: Gate 0(a) cannot be computed")
        recs.append(json.loads(p.read_text()))
    keep = lambda r: {k: v for k, v in r.items() if k not in DET_IGNORED}  # noqa: E731
    return "PASS" if keep(recs[0]) == keep(recs[1]) else "FAIL"


def primary_verdicts(rates: dict, gate_ok: bool) -> dict:
    """Claim 1: d11 J4V-S vs J4V-X. Claim 2: d9 J4V-S vs J4V-X."""
    return {"claim1": contrast(rates, ("J4V-S", 11), ("J4V-X", 11), gate_ok),
            "claim2": contrast(rates, ("J4V-S", 9), ("J4V-X", 9), gate_ok)}


def readings(full: dict, gate_m_above: bool) -> list[str]:
    """Spec sections 7 and 8, fixed before the numbers: what the output must say beside the
    verdicts."""
    c1, c2 = full["claim1"][0], full["claim2"][0]
    out = []
    if c1.startswith("CONFIRMED") and not gate_m_above:
        out.append("GATE M IS NOT ABOVE 0 WHILE CLAIM 1 IS CONFIRMED: RESULTS.md must lead with "
                   "this. The gain is not attributable to better ranking far from the goal.")
    if c1.startswith("CONFIRMED") and c2 == "REFUTED":
        out.append("CLAIM 1 CONFIRMED WITH CLAIM 2 REFUTED: report a trade, not a win. S gave up "
                   "near-goal training share and depth 9 paid for it.")
    if c1 in ("NOT SIGNIFICANT", "REFUTED"):
        out.append("CLAIM 1 READS: the judge is not the limit BY THIS LEVER. That is not "
                   "evidence that search is the limit.")
    return out


def _mean(recs, key, field):
    return st.mean(r[field] for r in recs[key].values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--pilot-dir", type=Path, default=cells.PILOT_OUT)
    ap.add_argument("--det-dir", type=Path, default=HERE / "outputs_det")
    ap.add_argument("--pilot-report", action="store_true",
                    help="print the amendment's numbers from the pilot records and stop")
    ap.add_argument("--stage", choices=("cont", "train"),
                    help="read only Gate 0(b), or only Gates L, E and M, and stop")
    args = ap.parse_args()
    if args.pilot_report:
        a = pilot_amendment(args.pilot_dir)
        for key, val in a.items():
            print(f"{key}: {val}")
        if a["stop"]:
            print(f"STOP: S's mean pilot Gate L margin {a['mean_margin']:.4f} is below X's "
                  f"threshold {X_PILOT_GATE_L:.4f}. Report; there is no amendment "
                  f"(spec section 6).")
        return
    require_amendment()
    out = args.out_dir
    if args.stage == "cont":
        print(format_stage_cont(stage_cont(out)))
        return
    tr = stage_train(out)
    if args.stage == "train":
        for line in format_stage_train(tr):
            print(line)
        return
    g0b = stage_cont(out)
    g0a = gate0a(out, args.det_dir)
    recs = load_records(out)
    rates = rates_of(recs)
    rank = {d: [_read(out / ev.rank_record_name(d, s)) for s in SEEDS] for d in DEPTHS}
    rank76 = {d: [json.loads((cells.E76_OUT / f"exp076_rank_J-X4_d{d}_s{s}.json").read_text())
                  for s in SEEDS] for d in DEPTHS}

    print(f"GATE 0(a) determinism (J4V-S d9 s0 re-run): {g0a}")
    print(format_stage_cont(g0b))
    for line in format_stage_train(tr):
        print(line)
    print("GATE R at k 4 (reported, not voiding): lowest-J 4-move leaf closer than root")
    for d in DEPTHS:
        ok, p = a71.gate_r_verdict(rank[d], "hit", "chance")
        print(f"  d{d}: S hit {st.mean(r['hit'] for r in rank[d]):.4f} chance "
              f"{st.mean(r['chance'] for r in rank[d]):.4f} p {p:.4f} -> {ok}   "
              f"(EXP-076 X: hit {st.mean(r['hit'] for r in rank76[d]):.4f} chance "
              f"{st.mean(r['chance'] for r in rank76[d]):.4f})")

    print("\nMean held-out success (mean steps to solve, optimality)")
    for d in DEPTHS:
        print(f"  d{d}: " + "  ".join(
            f"{a} {_mean(recs, (a, d), 'success_rate'):.4f} "
            f"({_mean(recs, (a, d), 'mean_steps'):.2f}, {_mean(recs, (a, d), 'optimality'):.3f})"
            for a in ev.ARMS))

    gate_ok = g0a == "PASS" and g0b == "PASS" and tr["l"]["pass"] and tr["e"]
    if not gate_ok:
        print("Gate 0, Gate L or Gate E did not pass. Every claim is VOID.")
    full = primary_verdicts(rates, gate_ok)
    for name, label in (("claim1", "CLAIM 1 (primary) d11 J4V-S - J4V-X"),
                        ("claim2", "CLAIM 2 (primary) d9 J4V-S - J4V-X")):
        v, p, diff, am, bm = full[name]
        wins = sum(x > 0 for x in a72.pair_diffs(rates[("J4V-S", 11 if name == "claim1" else 9)],
                                                  rates[("J4V-X", 11 if name == "claim1" else 9)]))
        print(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, S ahead on {wins} of 12, "
              f"p {p:.4f} -> {v}")
    for line in readings(full, tr["m"]["above"]):
        print("    " + line)

    print("\nSecondary (patterns, never confirmations):")
    j5 = st.mean(json.loads((cells.E78_OUT / f"exp078_J5V-X_d11_s{s}.json").read_text())
                 ["success_rate"] for s in SEEDS)
    print(f"  d11 EXP-078 full 5-move tree with X alongside: {j5:.4f}")
    s_sp = st.mean(r["final_probe"]["spearman_7_11"] for r in tr["recs"].values())
    x_sp = st.mean(_x_train(s)["final_probe"]["spearman_7_11"] for s in SEEDS)
    print(f"  probe spearman_7_11, mean over seeds: S {s_sp:.4f}  X {x_sp:.4f}")
    for d in ("9", "10", "11"):
        sj = st.mean(r["final_probe"]["mean_j_by_distance"][d] for r in tr["recs"].values())
        xj = st.mean(_x_train(s)["final_probe"]["mean_j_by_distance"][d] for s in SEEDS)
        print(f"  probe mean J at distance {d}: S {sj:.3f}  X {xj:.3f}")
    dists = sorted({d for r in tr["recs"].values() for d in r["train_hist"]}, key=int)
    hist = {d: st.mean(r["train_hist"].get(d, 0.0) for r in tr["recs"].values()) for d in dists}
    print("  realised training histogram, mean over seeds (last probe interval): "
          + "  ".join(f"{d}:{v:.3f}" for d, v in hist.items()))


if __name__ == "__main__":
    main()
