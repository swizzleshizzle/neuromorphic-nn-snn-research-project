"""EXP-073 aggregator. Gates 0, T, E, R and 1 are VERDICTS: a claim whose gate failed prints VOID.

Spec: docs/superpowers/specs/2026-10-07-exp073-learned-judge-design.md, sections 5 to 7.

The ladder (alpha 0.025, Gate 1, one-sided exact sign-flip, the continuity comparison) is
EXP-071's, reused through importlib so the experiments cannot drift apart.

Usage: .venv/bin/python experiments/073_learned_judge/aggregate.py --determinism-ok
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
one_sided_p = a71.one_sided_p
contrast = a72.contrast
SEEDS = range(12)
EXP071_OUT = REPO / "experiments" / "071_critic_and_no_revisit" / "outputs"
EXP072_OUT = REPO / "experiments" / "072_p3v_frontier" / "outputs"

# Set ONLY by the controller's dated amendment, after the pilot and before any seed 0-11 trains:
# {"A": float, "B": float}, each half of that arm's mean pilot end-of-training spearman_7_11 and
# never below 0.30 (spec section 6). None blocks every verdict. Code never chooses it.
GATE_T_THRESHOLD: dict | None = None

J_ARMS = ("J1V-A", "J1V-B", "J3V-A", "J3V-B")
RANK_DEPTHS = (7, 8, 9)


def gate_t_verdict(final_probes: list[dict], threshold: float) -> bool:
    """Mean spearman_7_11 over seeds >= threshold AND the seed-mean of mean_j_by_distance
    strictly increasing over distances 7..11. A NaN spearman fails (the comparison is False)."""
    mean_sp = st.mean(p["spearman_7_11"] for p in final_probes)
    means = [st.mean(p["mean_j_by_distance"][str(d)] for p in final_probes) for d in range(7, 12)]
    increasing = all(b > a for a, b in zip(means, means[1:]))
    return bool(mean_sp >= threshold and increasing)


def gate_e_verdict(drift_a: list[float], drift_b: list[float]) -> bool:
    """Every arm-A drift > 0 and every arm-B drift exactly 0.0 (bit-identical to E1)."""
    return all(d > 0 for d in drift_a) and all(d == 0.0 for d in drift_b)


def gate_r_verdict(rank_rows: list[dict]) -> bool:
    """Exact one-sided p < 0.05 and mean (hit - chance) > 0, over the per-seed rows."""
    return bool(a71.gate_r_verdict(rank_rows, "hit", "chance")[0])


def primary_verdicts(rates: dict, gates: dict) -> dict:
    """Claim 1: ("J3V-A", 9) vs ("P3V", 9). Claim 2: ("J3V-A", 9) vs ("J3V-B", 9).

    `gates` = {"gate0": bool, "t": {"A": bool, "B": bool}, "e": bool,
               "r": {("J-A", 9): bool, ("J-B", 9): bool}}.
    Claim 1 is VOID on Gate 0, A's Gate T or A's Gate R. Claim 2 additionally needs B's Gate T and
    R, and Gate E (A trained, B frozen).
    """
    t, r = gates["t"], gates["r"]
    c1_ok = gates["gate0"] and t["A"] and r[("J-A", 9)]
    c2_ok = (gates["gate0"] and t["A"] and t["B"] and gates["e"]
             and r[("J-A", 9)] and r[("J-B", 9)])
    return {
        "claim1": contrast(rates, ("J3V-A", 9), ("P3V", 9), c1_ok),
        "claim2": contrast(rates, ("J3V-A", 9), ("J3V-B", 9), c2_ok),
    }


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path.name}: the run is incomplete")
    r = json.loads(path.read_text())
    if r.get("limit_states") is not None:
        raise SystemExit(f"{path.name} is a smoke record and may not enter a verdict")
    return r


def _ref_dir(depth: int) -> tuple[Path, str]:
    return (EXP071_OUT, "exp071") if depth == 7 else (EXP072_OUT, "exp072")


def load_rates(out_dir: Path) -> dict:
    """(arm, depth) -> {seed: success_rate}. J arms from this experiment; P3V, R3V and G0V at
    depths 7 to 9 from EXP-071 (depth 7) and EXP-072 (depths 8, 9); depth 11 from this one."""
    rates = {}
    for d in RANK_DEPTHS:
        for arm in J_ARMS:
            rates[(arm, d)] = {s: _read(out_dir / f"exp073_{arm}_d{d}_s{s}.json")["success_rate"]
                               for s in SEEDS}
        ref, tag = _ref_dir(d)
        for arm in ("P3V", "R3V", "G0V"):
            rates[(arm, d)] = {s: _read(ref / f"{tag}_{arm}_d{d}_s{s}.json")["success_rate"]
                               for s in SEEDS}
    for arm in ("J3V-A", "J3V-B", "P3V", "R3V"):
        rates[(arm, 11)] = {s: _read(out_dir / f"exp073_{arm}_d11_s{s}.json")["success_rate"]
                            for s in SEEDS}
    return rates


def gate0b(out_dir: Path) -> str:
    """P3V at seed 0, depths 8 and 9, re-run here, against EXP-072's committed records."""
    mine, ref = {}, {}
    for d in (8, 9):
        mine[("P3V", d)] = _read(out_dir / f"exp073_P3V_d{d}_s0.json")
        ref[("P3V", d)] = _read(EXP072_OUT / f"exp072_P3V_d{d}_s0.json")
    return a71.gate0b_verdict(mine, ref)


def load_train(out_dir: Path) -> dict:
    return {(arm, s): _read(out_dir / f"exp073_train_{arm}_s{s}.json")
            for arm in ("A", "B") for s in SEEDS}


def load_rank(out_dir: Path) -> dict:
    return {(kind, d): [_read(out_dir / f"exp073_rank_{kind}_d{d}_s{s}.json") for s in SEEDS]
            for kind in ("J-A", "J-B", "P") for d in RANK_DEPTHS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="set only after the det re-run matches its full-run copies")
    args = ap.parse_args()

    if GATE_T_THRESHOLD is None:
        raise SystemExit("GATE_T_THRESHOLD is unset. The controller sets it in the dated "
                         "post-pilot amendment; no verdict may be read before then.")

    train = load_train(args.out_dir)
    rank = load_rank(args.out_dir)
    rates = load_rates(args.out_dir)
    g0b = gate0b(args.out_dir)

    gate_t = {arm: gate_t_verdict([train[(arm, s)]["final_probe"] for s in SEEDS],
                                  GATE_T_THRESHOLD[arm]) for arm in ("A", "B")}
    gate_e = gate_e_verdict([train[("A", s)]["encoder_drift"] for s in SEEDS],
                            [train[("B", s)]["encoder_drift"] for s in SEEDS])
    gate_r = {(kind, d): gate_r_verdict(rank[(kind, d)])
              for kind in ("J-A", "J-B") for d in RANK_DEPTHS}

    print("GATE T (probe distances 7-11, mean over seeds)")
    for arm in ("A", "B"):
        probes = [train[(arm, s)]["final_probe"] for s in SEEDS]
        means = [st.mean(p["mean_j_by_distance"][str(d)] for p in probes) for d in range(7, 12)]
        print(f"  arm {arm}: spearman {st.mean(p['spearman_7_11'] for p in probes):.4f} "
              f"vs threshold {GATE_T_THRESHOLD[arm]:.4f}; mean J 7..11 "
              + " ".join(f"{m:.2f}" for m in means) + f" -> {gate_t[arm]}")
    print(f"GATE E (A drift > 0, B drift == 0): {gate_e}")
    print("GATE R (top 3-move leaf closer than root, J vs chance; policy reference alongside)")
    for d in RANK_DEPTHS:
        row = []
        for kind in ("J-A", "J-B", "P"):
            rows = rank[(kind, d)]
            row.append(f"{kind} hit {st.mean(r['hit'] for r in rows):.4f} "
                       f"chance {st.mean(r['chance'] for r in rows):.4f}"
                       + (f" -> {gate_r[(kind, d)]}" if kind != "P" else " (reference)"))
        print(f"  d{d}: " + "  |  ".join(row))
    print(f"GATE 0(a) determinism: {'PASS' if args.determinism_ok else 'NOT CHECKED'}")
    print(f"GATE 0(b) continuity with EXP-072: {g0b}")

    print("\nMean held-out success")
    for d in (*RANK_DEPTHS, 11):
        arms = [a for a in (*J_ARMS, "P3V", "R3V", "G0V") if (a, d) in rates]
        print(f"  d{d}: " + "  ".join(f"{a} {st.mean(rates[(a, d)].values()):.4f}" for a in arms))

    gate0_ok = g0b == "PASS" and args.determinism_ok
    if not gate0_ok:
        print("Gate 0 did not pass. No claim may be read.")
        return

    gates = {"gate0": gate0_ok, "t": gate_t, "e": gate_e,
             "r": {k: v for k, v in gate_r.items()}}
    out = primary_verdicts(rates, gates)
    for name, label in (("claim1", "CLAIM 1 (primary) d9 J3V-A - P3V"),
                        ("claim2", "CLAIM 2 (primary) d9 J3V-A - J3V-B")):
        v, p, diff, am, bm = out[name]
        print(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} -> {v}")

    print("\nSecondary (a pattern, never confirmations; labels follow the ladder for reference only):")
    for d in (7, 8):
        for a, b in (("J3V-A", "P3V"), ("J3V-A", "J3V-B")):
            ok = (gates["gate0"] and gate_t["A"] and gate_r[("J-A", d)]
                  and (b != "J3V-B" or (gate_t["B"] and gate_e and gate_r[("J-B", d)])))
            v, p, diff, am, bm = contrast(rates, (a, d), (b, d), ok)
            print(f"  d{d} {a} - {b}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} ({v})")
    for d in RANK_DEPTHS:
        ok = gates["gate0"] and gate_t["A"] and gate_r[("J-A", d)]
        v, p, diff, am, bm = contrast(rates, ("J1V-A", d), ("G0V", d), ok)
        print(f"  d{d} J1V-A - G0V: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} ({v})")
    ok = gates["gate0"] and gate_t["B"] and gate_r[("J-B", 9)]
    v, p, diff, am, bm = contrast(rates, ("J3V-B", 9), ("P3V", 9), ok)
    print(f"  d9 J3V-B - P3V: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} ({v})")

    print("\nExploratory, depth 11 (200 held-out states, no claims):")
    print("  " + "  ".join(f"{a} {st.mean(rates[(a, 11)].values()):.4f}"
                           for a in ("J3V-A", "J3V-B", "P3V", "R3V")))


if __name__ == "__main__":
    main()
