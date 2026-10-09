"""EXP-075 aggregator. Gates 0, P, L, E, R and 1 are VERDICTS: a claim whose gate failed prints
VOID, and a contrast on the floor or ceiling prints UNRESOLVED.

Spec: docs/superpowers/specs/2026-10-09-exp075-wide-sensory-region-design.md, sections 6 to 9.

The sign-flip test and Gate 1 are EXP-070's, Gate 0(b)'s field comparison and Gate R are EXP-071's,
seed pairing is EXP-072's, and Gate L and EXP-074's rates are EXP-074's, all reused through
importlib so the experiments cannot drift apart. Alpha is this spec's own: 0.05 / 3.

Usage:
    .venv/bin/python experiments/075_wide_region/aggregate.py --pilot-report
    .venv/bin/python experiments/075_wide_region/aggregate.py --determinism-ok
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
cells = _load("exp075_cells", HERE / "cells.py")
one_sided_p = a70.one_sided_p
SEEDS = range(12)

ALPHA = 0.05 / 3

# Set by the dated amendment (spec section 12, 2026-10-09) from the committed pilot records,
# before any seed 0-11 pretrains. Gate P: 0.9 x that arm's mean pilot move accuracy.
# Gate L: half that arm's mean pilot margin (spec section 7). None blocks every verdict.
GATE_P_THRESHOLD: dict | None = {"X": 0.5023642954536297, "Y": 0.40965700094402796}
GATE_L_THRESHOLD: dict | None = {"X": 0.12983333333333333, "Y": 0.10483333333333333}
PILOT_FAIL_ACCURACY = 0.30
# Spec section 6: Y must clear W's own Gate L threshold to count as a working control.
W_GATE_L_THRESHOLD = a74.GATE_L_THRESHOLD["W"]

J_ARMS = ("J3V-X", "J3V-Y")
DEPTHS = (7, 8, 9, 11)
# W's Gate R is measured here, at the two depths its claims use (EXP-074 ranked W at 7 to 9 only).
W_RANK_DEPTHS = (9, 11)
_KIND = {"J3V-X": "J-X", "J3V-Y": "J-Y"}
_TRAIN_ARM = {"J3V-X": "X", "J3V-Y": "Y"}
TRAIN_SETTINGS = {"n_updates": 4000, "batch": 1000, "sync_every": 100, "probe_every": 250,
                  "draws": 1}
PRETRAIN_SETTINGS = {"epochs": 40, "batch_size": 256, "lr": 3e-3}


def claim_verdict(diffs, a_mean: float, b_mean: float, gate_ok: bool) -> tuple[str, float]:
    p = one_sided_p(diffs)
    if not gate_ok:
        return "VOID", p
    if a70.gate1_verdict(a_mean, b_mean) == "UNRESOLVED":
        return "UNRESOLVED", p
    if st.mean(diffs) <= 0:
        return "REFUTED", p
    if p < ALPHA:
        return "CONFIRMED", p
    return "NOT SIGNIFICANT", p


def contrast(rates, a, b, gate_ok):
    diffs = a72.pair_diffs(rates[a], rates[b])
    a_mean, b_mean = st.mean(rates[a].values()), st.mean(rates[b].values())
    verdict, p = claim_verdict(diffs, a_mean, b_mean, gate_ok)
    return verdict, p, st.mean(diffs), a_mean, b_mean


def _check(rec, settings, kind) -> None:
    who = f"{kind} record for arm {rec.get('arm')} seed {rec.get('seed')}"
    if "arm" not in rec:
        raise SystemExit(f"{who}: the record has no 'arm' field")
    for k, v in settings.items():
        if rec.get(k) != v:
            raise SystemExit(f"{who}: {k} is {rec.get(k)!r}, spec says {v!r}")
    if rec.get("hidden") != cells.HIDDEN[rec["arm"]]:
        raise SystemExit(f"{who}: hidden is {rec.get('hidden')!r}, "
                         f"spec says {cells.HIDDEN[rec['arm']]!r}")


def check_train_record(rec) -> None:
    _check(rec, TRAIN_SETTINGS, "training")
    if rec.get("readout") != "wide":
        raise SystemExit(f"training record for arm {rec.get('arm')} seed {rec.get('seed')}: "
                         f"readout is {rec.get('readout')!r}, spec says 'wide'")


def check_pretrain_record(rec) -> None:
    _check(rec, PRETRAIN_SETTINGS, "pretraining")


def gate_p_verdict(accuracies, threshold: float) -> bool:
    return all(a >= threshold for a in accuracies)


def gate_e_verdict(drifts) -> bool:
    return all(d > 0 for d in drifts)


def require_thresholds() -> None:
    for name in ("GATE_P_THRESHOLD", "GATE_L_THRESHOLD"):
        if globals()[name] is None:
            raise SystemExit(f"{name} is unset. The controller sets it in the dated post-pilot "
                             f"amendment; no verdict may be read before then.")


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path.name}: the run is incomplete")
    r = json.loads(path.read_text())
    if r.get("limit_states") is not None:
        raise SystemExit(f"{path.name} is a smoke record and may not enter a verdict")
    return r


def seconds_per_update(rec):
    """wall_s covers only the session that wrote the record, so divide by the updates that session
    trained: n_updates - resumed_from. None when it trained nothing, or the fields are absent."""
    wall, n, res = rec.get("wall_s"), rec.get("n_updates"), rec.get("resumed_from")
    if wall is None or n is None or res is None or n - res <= 0:
        return None
    return wall / (n - res)


def _mean_or_none(values):
    """Mean of the non-None values; None when there are none."""
    vals = [v for v in values if v is not None]
    return st.mean(vals) if vals else None


def pilot_amendment(pilot_dir) -> dict:
    """The numbers the dated amendment states (spec sections 6 and 7), from the pilot records."""
    pilot_dir = Path(pilot_dir)
    pre, trn = {}, {}
    for a in cells.ARMS_TRAIN:
        for s in cells.PILOT_SEEDS:
            pre[(a, s)] = _read(pilot_dir / cells.pretrain_record_name(a, s))
            check_pretrain_record(pre[(a, s)])
            trn[(a, s)] = _read(pilot_dir / cells.record_name(a, s))
            check_train_record(trn[(a, s)])
    acc = {a: [pre[(a, s)]["final_move_accuracy"] for s in cells.PILOT_SEEDS]
           for a in cells.ARMS_TRAIN}
    mar = {a: [trn[(a, s)]["gate_l"]["margin"] for s in cells.PILOT_SEEDS]
           for a in cells.ARMS_TRAIN}
    drift = {a: [trn[(a, s)].get("encoder_drift") for s in cells.PILOT_SEEDS]
             for a in cells.ARMS_TRAIN}
    spu = {a: [seconds_per_update(trn[(a, s)]) for s in cells.PILOT_SEEDS]
           for a in cells.ARMS_TRAIN}
    return {
        "mean_encoder_drift": {a: _mean_or_none(v) for a, v in drift.items()},
        "mean_seconds_per_update": {a: _mean_or_none(v) for a, v in spu.items()},
        "seconds_per_update_by_seed": spu,
        "mean_accuracy": {a: st.mean(v) for a, v in acc.items()},
        "mean_margin": {a: st.mean(v) for a, v in mar.items()},
        "gate_p": {a: 0.9 * st.mean(v) for a, v in acc.items()},
        "gate_l": {a: st.mean(v) / 2 for a, v in mar.items()},
        "pilot_failed": any(st.mean(v) < PILOT_FAIL_ACCURACY for v in acc.values()),
        "y_working_control": st.mean(mar["Y"]) >= W_GATE_L_THRESHOLD,
    }


def check_judge_updates(rec, path) -> None:
    """An X or Y eval or rank record must come from a judge trained to the full 4000 updates."""
    if rec.get("judge_updates") != cells.N_UPDATES:
        raise SystemExit(f"{path.name}: judge_updates is {rec.get('judge_updates')!r}, spec says "
                         f"{cells.N_UPDATES}: it was evaluated from a partly trained judge")


def _read_judged(path: Path) -> dict:
    r = _read(path)
    check_judge_updates(r, path)
    return r


def stage_pretrain(out) -> dict:
    """Gate P from the 24 pretraining records alone. Readable right after pretraining."""
    require_thresholds()
    out = Path(out)
    res = {}
    for a in cells.ARMS_TRAIN:
        v = []
        for s in SEEDS:
            r = _read(out / cells.pretrain_record_name(a, s))
            check_pretrain_record(r)
            v.append(r["final_move_accuracy"])
        res[a] = {"min": min(v), "mean": st.mean(v), "threshold": GATE_P_THRESHOLD[a],
                  "pass": gate_p_verdict(v, GATE_P_THRESHOLD[a])}
    return res


def stage_train(out) -> dict:
    """Gates L and E from the 24 training records alone. Readable right after training."""
    require_thresholds()
    out = Path(out)
    res = {"l": {}, "e": {}}
    for a in cells.ARMS_TRAIN:
        recs = []
        for s in SEEDS:
            r = _read(out / cells.record_name(a, s))
            check_train_record(r)
            recs.append(r)
        m = [r["gate_l"]["margin"] for r in recs]
        res["l"][a] = {"mean": st.mean(m), "threshold": GATE_L_THRESHOLD[a],
                       "p": one_sided_p(m), "pass": a74.gate_l_verdict(m, GATE_L_THRESHOLD[a])}
        res["e"][a] = gate_e_verdict([r["encoder_drift"] for r in recs])
    return res


def stage_cont(out) -> str:
    """Gate 0(b): J3V-W at depths 9 and 11, seed 0, against EXP-074's committed records."""
    require_thresholds()
    return a71.gate0b_verdict(
        {("J3V-W", d): _read(Path(out) / f"exp075_J3V-W_d{d}_s0.json") for d in (9, 11)},
        {("J3V-W", d): _read(cells.E74_OUT / f"exp074_J3V-W_d{d}_s0.json") for d in (9, 11)})


def print_stage_pretrain(res) -> None:
    print("GATE P (pretraining move accuracy, every seed)")
    for a, g in res.items():
        print(f"  arm {a}: min {g['min']:.4f} mean {g['mean']:.4f} vs threshold "
              f"{g['threshold']:.4f} -> {g['pass']}")


def print_stage_train(res) -> None:
    print("GATE L (3-move leaf ranking on probe distances 7-11, margin over chance)")
    for a, g in res["l"].items():
        print(f"  arm {a}: mean margin {g['mean']:.4f} vs threshold {g['threshold']:.4f}, "
              f"p {g['p']:.4f} -> {g['pass']}")
    print(f"GATE E: {res['e']}")


def _arm_ok(gates, j_arm, depth) -> bool:
    a = _TRAIN_ARM[j_arm]
    return gates["p"][a] and gates["l"][a] and gates["e"][a] and gates["r"][(_KIND[j_arm], depth)]


def primary_verdicts(rates: dict, gates: dict, seeds) -> dict:
    """Claim 1: d9 J3V-X vs J3V-W. Claim 2: d9 J3V-X vs J3V-Y. Claim 3: d11 J3V-X vs J3V-W.
    W's Gates P, L and E were EXP-074's; its Gate R at depths 9 and 11 is measured here
    (gates["r"][("J-W", d)]). X and Y are gated here. Claim 2 does not involve W."""
    r = {k: {s: v[s] for s in seeds} for k, v in rates.items()}
    c1_ok = gates["gate0"] and _arm_ok(gates, "J3V-X", 9) and gates["r"][("J-W", 9)]
    c2_ok = gates["gate0"] and _arm_ok(gates, "J3V-X", 9) and _arm_ok(gates, "J3V-Y", 9)
    c3_ok = gates["gate0"] and _arm_ok(gates, "J3V-X", 11) and gates["r"][("J-W", 11)]
    return {"claim1": contrast(r, ("J3V-X", 9), ("J3V-W", 9), c1_ok),
            "claim2": contrast(r, ("J3V-X", 9), ("J3V-Y", 9), c2_ok),
            "claim3": contrast(r, ("J3V-X", 11), ("J3V-W", 11), c3_ok)}


def sensitivity_rates(rates: dict) -> dict:
    return {k: {s: v for s, v in d.items() if s not in cells.SENSITIVITY_DROP}
            for k, d in rates.items()}


def load_rates(out_dir: Path) -> dict:
    """(arm, depth) -> {seed: success_rate}. X and Y from this experiment; W, P3V and R3V from
    EXP-074's aggregator (which reads EXP-071 and EXP-072 for P3V and R3V at depths 7 to 9)."""
    rates = {k: v for k, v in a74.load_rates(cells.E74_OUT).items()
             if k[0] in ("J3V-W", "P3V", "R3V")}
    for d in DEPTHS:
        for arm in J_ARMS:
            rates[(arm, d)] = {
                s: _read_judged(out_dir / f"exp075_{arm}_d{d}_s{s}.json")["success_rate"]
                for s in SEEDS}
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--pilot-dir", type=Path, default=HERE / "outputs_pilot")
    ap.add_argument("--pilot-report", action="store_true",
                    help="print the amendment's numbers from the pilot records and stop")
    ap.add_argument("--stage", choices=("pretrain", "train", "cont"),
                    help="read only this stage's gate (spec section 10) and stop")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="set only after the det re-run matches its full-run copy")
    args = ap.parse_args()
    if args.pilot_report:
        a = pilot_amendment(args.pilot_dir)
        for key, val in a.items():
            print(f"{key}: {val}")
        if a["pilot_failed"]:
            print(f"PILOT FAILED: an arm's mean pilot move accuracy is below "
                  f"{PILOT_FAIL_ACCURACY}. Stop and report; no amendment (spec section 7).")
        if not a["y_working_control"]:
            print(f"Y IS NOT A WORKING CONTROL: mean pilot margin below W's Gate L threshold "
                  f"{W_GATE_L_THRESHOLD:.4f}. Stop and report; no amendment (spec section 6).")
        return
    require_thresholds()
    out = args.out_dir
    if args.stage == "pretrain":
        print_stage_pretrain(stage_pretrain(out))
        return
    if args.stage == "train":
        print_stage_train(stage_train(out))
        return
    if args.stage == "cont":
        print(f"GATE 0(b) J3V-W continuity with EXP-074 (d9, d11, seed 0): {stage_cont(out)}")
        return
    p_res, t_res, g0b = stage_pretrain(out), stage_train(out), stage_cont(out)
    rank_keys = [(k, d) for k in ("J-X", "J-Y") for d in DEPTHS] \
        + [("J-W", d) for d in W_RANK_DEPTHS]
    rank = {(k, d): [(_read if k == "J-W" else _read_judged)(
        out / f"exp075_rank_{k}_d{d}_s{s}.json") for s in SEEDS] for k, d in rank_keys}
    rates = load_rates(out)

    gate_p = {a: p_res[a]["pass"] for a in cells.ARMS_TRAIN}
    gate_l = {a: t_res["l"][a]["pass"] for a in cells.ARMS_TRAIN}
    gate_e = t_res["e"]
    gate_r = {(k, d): bool(a71.gate_r_verdict(rank[(k, d)], "hit", "chance")[0])
              for k, d in rank_keys}

    print_stage_pretrain(p_res)
    print_stage_train(t_res)
    print("GATE R (lowest-J leaf closer than root on held-out states)")
    for d in DEPTHS:
        print(f"  d{d}: " + "  |  ".join(
            f"{k} {st.mean(r['hit'] for r in rank[(k, d)]):.4f}/"
            f"{st.mean(r['chance'] for r in rank[(k, d)]):.4f} -> {gate_r[(k, d)]}"
            for k in ("J-X", "J-Y")))
    for d in W_RANK_DEPTHS:
        print(f"  d{d}: J-W {st.mean(r['hit'] for r in rank[('J-W', d)]):.4f}/"
              f"{st.mean(r['chance'] for r in rank[('J-W', d)]):.4f} -> {gate_r[('J-W', d)]}")
    print(f"GATE 0(a) determinism: {'PASS' if args.determinism_ok else 'NOT CHECKED'}")
    print(f"GATE 0(b) J3V-W continuity with EXP-074 (d9, d11, seed 0): {g0b}")

    print("\nMean held-out success")
    for d in DEPTHS:
        print(f"  d{d}: " + "  ".join(f"{a} {st.mean(rates[(a, d)].values()):.4f}"
                                      for a in ("J3V-X", "J3V-Y", "J3V-W", "P3V", "R3V")))

    gate0_ok = g0b == "PASS" and args.determinism_ok
    if not gate0_ok:
        print("Gate 0 did not pass. No claim may be read.")
        return
    gates = {"gate0": gate0_ok, "p": gate_p, "l": gate_l, "e": gate_e, "r": gate_r}
    full = primary_verdicts(rates, gates, SEEDS)
    sens_rates = sensitivity_rates(rates)
    sens = primary_verdicts(sens_rates, gates, sorted(sens_rates[("J3V-X", 9)]))
    for name, label in (("claim1", "CLAIM 1 (primary) d9 J3V-X - J3V-W"),
                        ("claim2", "CLAIM 2 (primary) d9 J3V-X - J3V-Y"),
                        ("claim3", "CLAIM 3 (primary) d11 J3V-X - J3V-W")):
        v, p, diff, am, bm = full[name]
        sv, sp, sdiff, _, _ = sens[name]
        print(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} -> {v}")
        print(f"    sensitivity, seeds 0 and 3 dropped: diff {sdiff:+.4f}, p {sp:.4f} -> {sv}")
        if sv != v:
            print("    SENSITIVITY DISAGREES WITH THE VERDICT: RESULTS.md must lead with this.")

    print("\nSecondary (patterns, never confirmations):")
    sec = [(d, "J3V-X", "J3V-W") for d in (7, 8)] + [(d, "J3V-X", "J3V-Y") for d in (7, 8, 11)] \
        + [(9, "J3V-Y", "J3V-W")]
    for d, a, b in sec:
        # W has a Gate R cell only at depths 9 and 11, so the d7/d8 X - W secondaries keep the
        # arm gates of X alone; d9 Y - W also needs ("J-W", 9).
        ok = gate0_ok and _arm_ok(gates, a, d) and (b not in _TRAIN_ARM or _arm_ok(gates, b, d))
        if b == "J3V-W" and d in W_RANK_DEPTHS:
            ok = ok and gates["r"][("J-W", d)]
        v, p, diff, am, bm = contrast(rates, (a, d), (b, d), ok)
        tag = "VOID (gate)" if v == "VOID" else (
            "positive pattern" if diff > 0 else "non-positive pattern")
        print(f"  d{d} {a} - {b}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} ({tag})")
    print("  mechanism line, leaf hit X - Y: " + "  ".join(
        f"d{d} {st.mean(r['hit'] for r in rank[('J-X', d)]) - st.mean(r['hit'] for r in rank[('J-Y', d)]):+.4f}"
        for d in DEPTHS))


if __name__ == "__main__":
    main()
