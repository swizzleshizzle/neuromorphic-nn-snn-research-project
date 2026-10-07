"""EXP-072 aggregator. Gates and the unresolved band are VERDICTS (the EXP-068 rule).

Spec: docs/superpowers/specs/2026-10-07-exp072-p3v-frontier-design.md, sections 3 and 4.

Usage: .venv/bin/python experiments/072_p3v_frontier/aggregate.py --determinism-ok
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


# The verdict ladder, alpha 0.025, Gate 1 and the continuity comparison are EXP-071's, reused
# rather than copied so the two experiments cannot drift apart.
a71 = _load("exp071_aggregate", REPO / "experiments" / "071_critic_and_no_revisit" / "aggregate.py")
one_sided_p = a71.one_sided_p
OUTCOME_FIELDS = a71.OUTCOME_FIELDS
ALPHA = a71.ALPHA
DEPTHS = (8, 9)
SEEDS = range(12)
EXP070_OUT = REPO / "experiments" / "070_lookahead_existing" / "outputs"


def pair_diffs(a: dict, b: dict) -> list[float]:
    """a - b per seed. Refuses unequal seed sets: never compare arms across seed sets."""
    if set(a) != set(b):
        raise ValueError(f"seed sets differ: {sorted(set(a) ^ set(b))}")
    return [a[s] - b[s] for s in sorted(a)]


def contrast(rates, a, b, gate_ok):
    diffs = pair_diffs(rates[a], rates[b])
    a_mean, b_mean = st.mean(rates[a].values()), st.mean(rates[b].values())
    verdict, p = a71.claim_verdict(diffs, a_mean, b_mean, gate_ok=gate_ok)
    return verdict, p, st.mean(diffs), a_mean, b_mean


def primary_verdicts(rates: dict, gate_v: dict, gate0_ok: bool) -> dict:
    """Claim 1: depth 8 P3V - P3. Claim 2: depth 9 P3V - G0. VOID on Gate 0 or that depth's V."""
    return {
        "claim1": contrast(rates, ("P3V", 8), ("P3", 8), gate0_ok and gate_v[8]),
        "claim2": contrast(rates, ("P3V", 9), ("G0", 9), gate0_ok and gate_v[9]),
    }


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def load_rates(out_dir: Path) -> tuple[dict, dict]:
    """(success rates, revisit rates) keyed (arm, depth) -> {seed: value}."""
    rates, revisits = {}, {}
    for depth in DEPTHS:
        for arm in ("G0", "E3", "P3", "R3"):
            recs = {s: _read(EXP070_OUT / f"exp070_{arm}_d{depth}_s{s}.json") for s in SEEDS}
            rates[(arm, depth)] = {s: r["success_rate"] for s, r in recs.items()}
            revisits[(arm, depth)] = {s: r["eval_revisit_rate"] for s, r in recs.items()}
        for arm in ("G0V", "P3V", "R3V"):
            recs = {}
            for s in SEEDS:
                path = out_dir / f"exp072_{arm}_d{depth}_s{s}.json"
                if not path.exists():
                    raise SystemExit(f"missing {path.name}: the run is incomplete")
                r = _read(path)
                if r.get("limit_states") is not None:
                    raise SystemExit(f"{path.name} is a smoke record and may not enter a verdict")
                recs[s] = r
            rates[(arm, depth)] = {s: r["success_rate"] for s, r in recs.items()}
            revisits[(arm, depth)] = {s: r["eval_revisit_rate"] for s, r in recs.items()}
    return rates, revisits


def gate0b(out_dir: Path) -> str:
    recs072, recs070 = {}, {}
    for depth in DEPTHS:
        for arm in ("G0", "P3"):
            path = out_dir / f"exp072_{arm}_d{depth}_s0.json"
            if not path.exists():
                raise SystemExit(f"missing continuity record {path.name}")
            recs072[(arm, depth)] = _read(path)
            recs070[(arm, depth)] = _read(EXP070_OUT / f"exp070_{arm}_d{depth}_s0.json")
    return a71.gate0b_verdict(recs072, recs070)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="set only after the det re-run matches its full-run copies")
    args = ap.parse_args()

    g0b = gate0b(args.out_dir)
    rates, revisits = load_rates(args.out_dir)
    gate0_ok = g0b == "PASS" and args.determinism_ok
    gate_v = {d: a71.gate_v_verdict(st.mean(revisits[("G0V", d)].values()),
                                    st.mean(revisits[("G0", d)].values())) for d in DEPTHS}

    print("EXP-072 mean held-out success (G0/E3/P3/R3 from EXP-070's records)")
    for d in DEPTHS:
        print(f"  d{d}: " + "  ".join(f"{a} {st.mean(rates[(a, d)].values()):.4f}"
                                      for a in ("G0", "G0V", "E3", "P3", "P3V", "R3", "R3V")))
    print("EXP-072 mean eval_revisit_rate")
    for d in DEPTHS:
        print(f"  d{d}: " + "  ".join(f"{a} {st.mean(revisits[(a, d)].values()):.3f}"
                                      for a in ("G0", "G0V", "P3", "P3V", "R3", "R3V")))
    print(f"\nGATE 0(a) determinism: {'PASS' if args.determinism_ok else 'NOT CHECKED'}")
    print(f"GATE 0(b) continuity with EXP-070: {g0b}")
    for d in DEPTHS:
        print(f"GATE V d{d}: {gate_v[d]}")
    if not gate0_ok:
        print("Gate 0 did not pass. No claim may be read.")
        return

    out = primary_verdicts(rates, gate_v, gate0_ok)
    for name, label in (("claim1", "CLAIM 1 (primary) d8 P3V - P3"),
                        ("claim2", "CLAIM 2 (primary) d9 P3V - G0")):
        v, p, diff, am, bm = out[name]
        print(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} -> {v}")

    print("\nSecondary (a pattern, never confirmations; labels follow the ladder for reference only):")
    for a, b, d in ((("P3V", 8), ("G0", 8), 8), (("P3V", 9), ("P3", 9), 9),
                    (("G0V", 8), ("G0", 8), 8), (("G0V", 9), ("G0", 9), 9),
                    (("P3V", 8), ("G0V", 8), 8), (("P3V", 9), ("G0V", 9), 9)):
        v, p, diff, am, bm = contrast(rates, a, b, gate0_ok and gate_v[d])
        print(f"  d{d} {a[0]} - {b[0]}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} ({v})")


if __name__ == "__main__":
    main()
