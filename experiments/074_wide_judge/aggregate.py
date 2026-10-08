"""EXP-074 aggregator. Gates 0, L, E, R and 1 are VERDICTS: a claim whose gate failed prints VOID.

Spec: docs/superpowers/specs/2026-10-07-exp074-wide-judge-design.md, sections 6 to 8.

The ladder (alpha 0.025, Gate 1, one-sided exact sign-flip, continuity comparison) is EXP-071's
and EXP-072's, reused through importlib so the experiments cannot drift apart.

Usage: .venv/bin/python experiments/074_wide_judge/aggregate.py --determinism-ok
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


a71 = _load("exp071_aggregate", REPO / "experiments" / "071_critic_and_no_revisit" / "aggregate.py")
a72 = _load("exp072_aggregate", REPO / "experiments" / "072_p3v_frontier" / "aggregate.py")
cells = _load("exp074_cells", HERE / "cells.py")
one_sided_p = a71.one_sided_p
contrast = a72.contrast
SEEDS = range(12)
EXP071_OUT = REPO / "experiments" / "071_critic_and_no_revisit" / "outputs"
EXP072_OUT = REPO / "experiments" / "072_p3v_frontier" / "outputs"
EXP073_OUT = REPO / "experiments" / "073_learned_judge" / "outputs"

# Set ONLY by the controller's dated amendment, after the pilot and before any seed 0-11 trains:
# {"W": float, "A": float, "B": float}, each half that arm's mean pilot Gate L margin (spec
# section 6). None blocks every verdict. Code never chooses it.
GATE_L_THRESHOLD: dict | None = None
GATE_L_ALPHA = 0.05

J_ARMS = ("J3V-W", "J3V-A", "J3V-B")
RANK_DEPTHS = (7, 8, 9)
_KIND = {"J3V-W": "J-W", "J3V-A": "J-A", "J3V-B": "J-B"}
_TRAIN_ARM = {"J3V-W": "W", "J3V-A": "A", "J3V-B": "B"}


TRAIN_SETTINGS = {"n_updates": 4000, "batch": 1000, "sync_every": 100, "probe_every": 250,
                  "draws": 1}
READOUT = {"W": "wide", "A": "concept", "B": "concept"}


def check_train_record(rec) -> None:
    """A training record made at other than the spec's settings (section 4) or with the wrong
    readout may not enter a verdict."""
    who = f"arm {rec.get('arm')} seed {rec.get('seed')}"
    for k, v in TRAIN_SETTINGS.items():
        if rec.get(k) != v:
            raise SystemExit(f"training record for {who}: {k} is {rec.get(k)!r}, spec says {v!r}")
    if rec.get("readout") != READOUT[rec["arm"]]:
        raise SystemExit(f"training record for {who}: readout is {rec.get('readout')!r}, "
                         f"spec says {READOUT[rec['arm']]!r}")


def _read_train(path: Path) -> dict:
    r = _read(path)
    check_train_record(r)
    return r


def gate_l_verdict(margins: list[float], threshold: float) -> bool:
    """Per-seed margin > 0 by exact one-sided sign-flip (p < 0.05) AND mean margin >= threshold."""
    return bool(one_sided_p(margins) < GATE_L_ALPHA and st.mean(margins) > 0
                and st.mean(margins) >= threshold)


def gate_e_verdicts(drift: dict) -> dict:
    """W and A: every drift > 0 (the encoder trained). B: every drift exactly 0.0."""
    return {"W": all(d > 0 for d in drift["W"]), "A": all(d > 0 for d in drift["A"]),
            "B": all(d == 0.0 for d in drift["B"])}


def gate0c_verdict(mine: dict, ref: dict) -> str:
    """Every key in `ref` (EXP-073 pilot 1 records) must exist in `mine` with an identical
    probe history (every probe, not only the last)."""
    for key, r in ref.items():
        m = mine.get(key)
        if m is None or m["probes"] != r["probes"]:
            return "FAIL"
    return "PASS"


def _arm_ok(gates, j_arm, depth) -> bool:
    a = _TRAIN_ARM[j_arm]
    return gates["l"][a] and gates["e"][a] and gates["r"][(_KIND[j_arm], depth)]


def primary_verdicts(rates: dict, gates: dict, seeds) -> dict:
    """Claim 1: J3V-W vs P3V at depth 9. Claim 2: J3V-W vs J3V-A at depth 9. VOID on Gate 0, or
    Gate L, E or R for any J arm in the contrast. `seeds` restricts both arms' seed sets."""
    r = {k: {s: v[s] for s in seeds} for k, v in rates.items()}
    c1_ok = gates["gate0"] and _arm_ok(gates, "J3V-W", 9)
    c2_ok = c1_ok and _arm_ok(gates, "J3V-A", 9)
    return {"claim1": contrast(r, ("J3V-W", 9), ("P3V", 9), c1_ok),
            "claim2": contrast(r, ("J3V-W", 9), ("J3V-A", 9), c2_ok)}


def sensitivity_rates(rates: dict) -> dict:
    return {k: {s: v for s, v in d.items() if s not in cells.SENSITIVITY_DROP}
            for k, d in rates.items()}


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path.name}: the run is incomplete")
    r = json.loads(path.read_text())
    if r.get("limit_states") is not None:
        raise SystemExit(f"{path.name} is a smoke record and may not enter a verdict")
    return r


def load_rates(out_dir: Path) -> dict:
    """(arm, depth) -> {seed: success_rate}. J arms from this experiment at depths 7, 8, 9, 11;
    P3V and R3V at depth 7 from EXP-071, at 8 and 9 from EXP-072, at 11 from this experiment."""
    rates = {}
    for d in (*RANK_DEPTHS, 11):
        for arm in J_ARMS:
            rates[(arm, d)] = {s: _read(out_dir / f"exp074_{arm}_d{d}_s{s}.json")["success_rate"]
                               for s in SEEDS}
        for arm in ("P3V", "R3V"):
            if d == 11:
                path = lambda s: out_dir / f"exp074_{arm}_d11_s{s}.json"  # noqa: E731
            elif d == 7:
                path = lambda s: EXP071_OUT / f"exp071_{arm}_d7_s{s}.json"  # noqa: E731
            else:
                path = lambda s: EXP072_OUT / f"exp072_{arm}_d{d}_s{s}.json"  # noqa: E731
            rates[(arm, d)] = {s: _read(path(s))["success_rate"] for s in SEEDS}
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--pilot-dir", type=Path, default=HERE / "outputs_pilot")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="set only after the det re-run matches its full-run copies")
    args = ap.parse_args()
    if GATE_L_THRESHOLD is None:
        raise SystemExit("GATE_L_THRESHOLD is unset. The controller sets it in the dated "
                         "post-pilot amendment; no verdict may be read before then.")

    train = {(a, s): _read_train(args.out_dir / cells.record_name(a, s))
             for a in cells.ARMS_TRAIN for s in SEEDS}
    rank = {(k, d): [_read(args.out_dir / f"exp074_rank_{k}_d{d}_s{s}.json") for s in SEEDS]
            for k in ("J-W", "J-A", "J-B", "P") for d in RANK_DEPTHS}
    rates = load_rates(args.out_dir)

    gate_l = {a: gate_l_verdict([train[(a, s)]["gate_l"]["margin"] for s in SEEDS],
                                GATE_L_THRESHOLD[a]) for a in cells.ARMS_TRAIN}
    gate_e = gate_e_verdicts({a: [train[(a, s)]["encoder_drift"] for s in SEEDS]
                              for a in cells.ARMS_TRAIN})
    gate_r = {(k, d): bool(a71.gate_r_verdict(rank[(k, d)], "hit", "chance")[0])
              for k in ("J-W", "J-A", "J-B") for d in RANK_DEPTHS}
    mine0c = {(a, s): _read_train(args.pilot_dir / cells.record_name(a, s))
              for a in ("A", "B") for s in cells.PILOT_SEEDS}
    ref0c = {(a, s): _read(EXP073_OUT / f"exp073_train_{a}_s{s}.json")
             for a in ("A", "B") for s in cells.PILOT_SEEDS}
    g0c = gate0c_verdict(mine0c, ref0c)
    g0b = a71.gate0b_verdict(
        {("P3V", d): _read(args.out_dir / f"exp074_P3V_d{d}_s0.json") for d in (8, 9)},
        {("P3V", d): _read(EXP072_OUT / f"exp072_P3V_d{d}_s0.json") for d in (8, 9)})

    print("GATE L (3-move leaf ranking on probe distances 7-11, margin over chance)")
    for a in cells.ARMS_TRAIN:
        m = [train[(a, s)]["gate_l"]["margin"] for s in SEEDS]
        print(f"  arm {a}: mean margin {st.mean(m):.4f} vs threshold {GATE_L_THRESHOLD[a]:.4f}, "
              f"p {one_sided_p(m):.4f} -> {gate_l[a]}")
    print(f"GATE E: {gate_e}")
    print("GATE R (top 3-move leaf closer than root on held-out states; policy alongside)")
    for d in RANK_DEPTHS:
        row = []
        for k in ("J-W", "J-A", "J-B", "P"):
            rows = rank[(k, d)]
            row.append(f"{k} {st.mean(r['hit'] for r in rows):.4f}/"
                       f"{st.mean(r['chance'] for r in rows):.4f}"
                       + (f" -> {gate_r[(k, d)]}" if k != "P" else " (reference)"))
        print(f"  d{d}: " + "  |  ".join(row))
    print(f"GATE 0(a) determinism: {'PASS' if args.determinism_ok else 'NOT CHECKED'}")
    print(f"GATE 0(b) continuity with EXP-072: {g0b}")
    print(f"GATE 0(c) concept path equals EXP-073 pilot 1: {g0c}")

    print("\nMean held-out success")
    for d in (*RANK_DEPTHS, 11):
        print(f"  d{d}: " + "  ".join(f"{a} {st.mean(rates[(a, d)].values()):.4f}"
                                      for a in (*J_ARMS, "P3V", "R3V")))

    gate0_ok = g0b == "PASS" and g0c == "PASS" and args.determinism_ok
    if not gate0_ok:
        print("Gate 0 did not pass. No claim may be read.")
        return
    gates = {"gate0": gate0_ok, "l": gate_l, "e": gate_e, "r": gate_r}
    full = primary_verdicts(rates, gates, SEEDS)
    sens_rates = sensitivity_rates(rates)
    sens = primary_verdicts(sens_rates, gates, sorted(sens_rates[("P3V", 9)]))
    for name, label in (("claim1", "CLAIM 1 (primary) d9 J3V-W - P3V"),
                        ("claim2", "CLAIM 2 (primary) d9 J3V-W - J3V-A")):
        v, p, diff, am, bm = full[name]
        sv, sp, sdiff, _, _ = sens[name]
        print(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} -> {v}")
        print(f"    sensitivity, seeds 0 and 3 dropped: diff {sdiff:+.4f}, p {sp:.4f} -> {sv}")
        if sv != v:
            print("    SENSITIVITY DISAGREES WITH THE VERDICT: RESULTS.md must lead with this.")

    print("\nSecondary (a pattern, never confirmations):")
    for d in RANK_DEPTHS:
        pairs = [("J3V-A", "P3V"), ("J3V-A", "J3V-B")] if d == 9 else \
                [("J3V-W", "P3V"), ("J3V-W", "J3V-A")]
        for a, b in pairs:
            ok = gate0_ok and _arm_ok(gates, a, d) and (b not in _TRAIN_ARM or _arm_ok(gates, b, d))
            v, p, diff, am, bm = contrast(rates, (a, d), (b, d), ok)
            tag = "VOID (gate)" if v == "VOID" else (
                "positive pattern" if diff > 0 else "non-positive pattern")
            print(f"  d{d} {a} - {b}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} ({tag})")

    print("\nExploratory, depth 11 (no claims):")
    print("  " + "  ".join(f"{a} {st.mean(rates[(a, 11)].values()):.4f}"
                           for a in (*J_ARMS, "P3V", "R3V")))


if __name__ == "__main__":
    main()
