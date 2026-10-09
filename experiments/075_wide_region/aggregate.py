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

# Set ONLY by the controller's dated amendment, after the pilot and before any seed 0-11
# pretrains: {"X": float, "Y": float}. Gate P: 0.9 x that arm's mean pilot move accuracy.
# Gate L: half that arm's mean pilot margin (spec section 7). None blocks every verdict.
GATE_P_THRESHOLD: dict | None = None
GATE_L_THRESHOLD: dict | None = None
PILOT_FAIL_ACCURACY = 0.30
# Spec section 6: Y must clear W's own Gate L threshold to count as a working control.
W_GATE_L_THRESHOLD = a74.GATE_L_THRESHOLD["W"]

J_ARMS = ("J3V-X", "J3V-Y")
DEPTHS = (7, 8, 9, 11)
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


def pilot_amendment(pilot_dir) -> dict:
    """The numbers the dated amendment states (spec sections 6 and 7), from the pilot records."""
    pilot_dir = Path(pilot_dir)
    acc = {a: [_read(pilot_dir / cells.pretrain_record_name(a, s))["final_move_accuracy"]
               for s in cells.PILOT_SEEDS] for a in cells.ARMS_TRAIN}
    mar = {a: [_read(pilot_dir / cells.record_name(a, s))["gate_l"]["margin"]
               for s in cells.PILOT_SEEDS] for a in cells.ARMS_TRAIN}
    return {
        "mean_accuracy": {a: st.mean(v) for a, v in acc.items()},
        "mean_margin": {a: st.mean(v) for a, v in mar.items()},
        "gate_p": {a: 0.9 * st.mean(v) for a, v in acc.items()},
        "gate_l": {a: st.mean(v) / 2 for a, v in mar.items()},
        "pilot_failed": any(st.mean(v) < PILOT_FAIL_ACCURACY for v in acc.values()),
        "y_working_control": st.mean(mar["Y"]) >= W_GATE_L_THRESHOLD,
    }


def _arm_ok(gates, j_arm, depth) -> bool:
    a = _TRAIN_ARM[j_arm]
    return gates["p"][a] and gates["l"][a] and gates["e"][a] and gates["r"][(_KIND[j_arm], depth)]


def primary_verdicts(rates: dict, gates: dict, seeds) -> dict:
    """Claim 1: d9 J3V-X vs J3V-W. Claim 2: d9 J3V-X vs J3V-Y. Claim 3: d11 J3V-X vs J3V-W.
    W is EXP-074's reused reference, whose gates passed there; X and Y are gated here."""
    r = {k: {s: v[s] for s in seeds} for k, v in rates.items()}
    c1_ok = gates["gate0"] and _arm_ok(gates, "J3V-X", 9)
    c2_ok = c1_ok and _arm_ok(gates, "J3V-Y", 9)
    c3_ok = gates["gate0"] and _arm_ok(gates, "J3V-X", 11)
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
            rates[(arm, d)] = {s: _read(out_dir / f"exp075_{arm}_d{d}_s{s}.json")["success_rate"]
                               for s in SEEDS}
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--pilot-dir", type=Path, default=HERE / "outputs_pilot")
    ap.add_argument("--pilot-report", action="store_true",
                    help="print the amendment's numbers from the pilot records and stop")
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
    pre = {(a, s): _read(out / cells.pretrain_record_name(a, s))
           for a in cells.ARMS_TRAIN for s in SEEDS}
    for r in pre.values():
        check_pretrain_record(r)
    train = {(a, s): _read(out / cells.record_name(a, s)) for a in cells.ARMS_TRAIN for s in SEEDS}
    for r in train.values():
        check_train_record(r)
    rank = {(k, d): [_read(out / f"exp075_rank_{k}_d{d}_s{s}.json") for s in SEEDS]
            for k in ("J-X", "J-Y") for d in DEPTHS}
    rates = load_rates(out)

    gate_p = {a: gate_p_verdict([pre[(a, s)]["final_move_accuracy"] for s in SEEDS],
                                GATE_P_THRESHOLD[a]) for a in cells.ARMS_TRAIN}
    gate_l = {a: a74.gate_l_verdict([train[(a, s)]["gate_l"]["margin"] for s in SEEDS],
                                    GATE_L_THRESHOLD[a]) for a in cells.ARMS_TRAIN}
    gate_e = {a: gate_e_verdict([train[(a, s)]["encoder_drift"] for s in SEEDS])
              for a in cells.ARMS_TRAIN}
    gate_r = {(k, d): bool(a71.gate_r_verdict(rank[(k, d)], "hit", "chance")[0])
              for k in ("J-X", "J-Y") for d in DEPTHS}
    g0b = a71.gate0b_verdict(
        {("J3V-W", d): _read(out / f"exp075_J3V-W_d{d}_s0.json") for d in (9, 11)},
        {("J3V-W", d): _read(cells.E74_OUT / f"exp074_J3V-W_d{d}_s0.json") for d in (9, 11)})

    print("GATE P (pretraining move accuracy, every seed)")
    for a in cells.ARMS_TRAIN:
        v = [pre[(a, s)]["final_move_accuracy"] for s in SEEDS]
        print(f"  arm {a}: min {min(v):.4f} mean {st.mean(v):.4f} vs threshold "
              f"{GATE_P_THRESHOLD[a]:.4f} -> {gate_p[a]}")
    print("GATE L (3-move leaf ranking on probe distances 7-11, margin over chance)")
    for a in cells.ARMS_TRAIN:
        m = [train[(a, s)]["gate_l"]["margin"] for s in SEEDS]
        print(f"  arm {a}: mean margin {st.mean(m):.4f} vs threshold {GATE_L_THRESHOLD[a]:.4f}, "
              f"p {one_sided_p(m):.4f} -> {gate_l[a]}")
    print(f"GATE E: {gate_e}")
    print("GATE R (lowest-J leaf closer than root on held-out states)")
    for d in DEPTHS:
        print(f"  d{d}: " + "  |  ".join(
            f"{k} {st.mean(r['hit'] for r in rank[(k, d)]):.4f}/"
            f"{st.mean(r['chance'] for r in rank[(k, d)]):.4f} -> {gate_r[(k, d)]}"
            for k in ("J-X", "J-Y")))
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
        ok = gate0_ok and _arm_ok(gates, a, d) and (b not in _TRAIN_ARM or _arm_ok(gates, b, d))
        v, p, diff, am, bm = contrast(rates, (a, d), (b, d), ok)
        tag = "VOID (gate)" if v == "VOID" else (
            "positive pattern" if diff > 0 else "non-positive pattern")
        print(f"  d{d} {a} - {b}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} ({tag})")
    print("  mechanism line, leaf hit X - Y: " + "  ".join(
        f"d{d} {st.mean(r['hit'] for r in rank[('J-X', d)]) - st.mean(r['hit'] for r in rank[('J-Y', d)]):+.4f}"
        for d in DEPTHS))


if __name__ == "__main__":
    main()
