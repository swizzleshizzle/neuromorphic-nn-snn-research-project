"""EXP-064 aggregator: the pre-registered rules from the spec, applied to the records on disk.

Spec: `docs/superpowers/specs/2026-09-19-exp064-motor-policy-path-design.md`.

> WRITTEN BEFORE ANY EXP-064 NUMBER WAS SEEN. The run finished while the dashboard work was
> going on; completion was confirmed from COUNTS alone - 24 records, zero python processes -
> and neither the run log (which prints one success rate per line) nor any record was opened
> until this file existed. EXP-060's aggregator could not claim that; 061, 062, 063 could.

> THE GATES APPLY TO ARM P ONLY, and that asymmetry is deliberate. Arm C never trains the
> regions and never reads the motor pathway, so `region_drift` and `motor_rate_mean` are None
> for it BY CONSTRUCTION. Gating arm C on them would be a gate that cannot pass. Instead this
> file asserts they ARE None for C, which is a real check: if the control somehow recorded a
> drift, the arms are not what the spec says they are.

> A MISSING READING IS A FAILURE, NEVER A PASS. `None` coerced to 0.0 would sail under a
> "must be at least X" floor while measuring nothing.

> THE PRIMARY IS NOT DIRECTIONAL. Both directions are interesting and the spec refuses to
> privilege one. A NULL means the brain's own pathway is neither better nor measurably worse
> than a conventional MLP of the same size - a notable outcome for this project, and still a
> BOUND at n=12 rather than proof of equivalence. That wording was fixed before dispatch.

Usage:
    .venv/bin/python -u experiments/064_motor_policy_path/aggregate.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp064_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

SEEDS = _run.SEEDS
TAG_P, TAG_C = _run.ARMS[0]["tag"], _run.ARMS[1]["tag"]
READOUT_P, READOUT_C = _run.ARMS[0]["readout"], _run.ARMS[1]["readout"]
BAR = _run.BAR
GATE_DRIFT = _run.GATE_MIN_REGION_DRIFT
GATE_RATE = _run.GATE_MIN_MOTOR_RATE
MOTOR_TRAINABLE = _run.MOTOR_TRAINABLE
CONTROL_TRAINABLE = _run.CONTROL_TRAINABLE
ALPHA = 0.05

T95_DF11 = 2.201   # n=12. Correct HERE and wrong for EXP-059/061/063, which are n=24 (2.069).


def permutation_p(diffs) -> tuple[float, str]:
    """Exact two-sided paired sign-flip permutation. n=12 is 4,096 flips, so no sampling."""
    n, obs = len(diffs), abs(sum(diffs))
    hits = sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12)
    return hits / 2 ** n, f"exact over all 2**{n} = {2 ** n:,} sign flips"


def ci95(diffs) -> tuple[float, float]:
    n = len(diffs)
    d = st.mean(diffs)
    se = st.stdev(diffs) / (n ** 0.5) if n > 1 else 0.0
    return d - T95_DF11 * se, d + T95_DF11 * se


def load(directory: Path, tag: str, expect_readout: str) -> dict:
    """Records by seed. The readout is CHECKED: both arms share `arm="regionalized"` and differ
    only in the readout, so a tag pointing at the wrong one would contrast an arm against
    itself with nothing in the numbers looking wrong."""
    out: dict[int, dict] = {}
    for p in sorted(Path(directory).glob("*.json")):
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or r.get("tag") != tag:
            continue
        if r.get("readout") != expect_readout:
            raise SystemExit(f"{p.name}: tag {tag} carries readout {r.get('readout')!r}, "
                             f"expected {expect_readout!r}; refusing.")
        out[int(r["seed"])] = r
    return out


def paired(arm: dict, ctrl: dict, field="success_rate"):
    use = sorted(set(arm) & set(ctrl) & set(SEEDS))
    missing = [s for s in use if arm[s].get(field) is None or ctrl[s].get(field) is None]
    if missing:
        raise SystemExit(f"field {field!r} is None for seeds {missing}; cannot pair on it")
    return use, [arm[s][field] - ctrl[s][field] for s in use]


def reading(records: dict, field: str):
    """Mean, missing seeds, min. Missing is RETURNED, never skipped."""
    vals, missing = [], []
    for s in sorted(SEEDS):
        r = records.get(s)
        v = None if r is None else r.get(field)
        if v is None:
            missing.append(s)
        else:
            vals.append(v)
    return (st.mean(vals) if vals else None), missing, (min(vals) if vals else None)


def check_gates(P: dict, C: dict) -> tuple[bool, list[str]]:
    ok, lines = True, []
    for n, field, floor in ((1, "region_drift", GATE_DRIFT), (2, "motor_rate_mean", GATE_RATE)):
        mean, missing, lo = reading(P, field)
        if missing:
            ok = False
            lines.append(f"  GATE {n} arm P ({field}): MISSING for seeds {missing} - FAIL. "
                         "A missing reading is never a pass.")
            continue
        passed = mean >= floor
        ok = ok and passed
        lines.append(f"  GATE {n} arm P ({field}): mean {mean:.4f}, min {lo:.4f}, "
                     f"floor {floor} -> {'PASS' if passed else 'FAIL'}")
        # The control must NOT have this reading at all. If it does, the arms are not what the
        # spec says they are, and the contrast is measuring something unregistered.
        ctrl_mean, _, _ = reading(C, field)
        if ctrl_mean is not None:
            ok = False
            lines.append(f"  GATE {n} arm C: recorded {field} = {ctrl_mean:.4f}, expected None. "
                         "The control is not the frozen-brain arm the spec describes - FAIL.")
    return ok, lines


def check_capacity(P: dict, C: dict) -> tuple[bool, list[str]]:
    """The contrast rests on the arms being capacity-matched. Verified from the RECORDS, not
    from the spec's arithmetic, because the record reports what an optimizer actually held."""
    p_vals = {r["trainable_params"] for r in P.values()}
    c_vals = {r["trainable_params"] for r in C.values()}
    lines = [f"  arm P trainable {sorted(p_vals)}   arm C trainable {sorted(c_vals)}"]
    ok = p_vals == {MOTOR_TRAINABLE} and c_vals == {CONTROL_TRAINABLE}
    if not ok:
        lines.append(f"  MISMATCH: expected {MOTOR_TRAINABLE} and {CONTROL_TRAINABLE} - FAIL")
    else:
        gap = abs(MOTOR_TRAINABLE - CONTROL_TRAINABLE) / MOTOR_TRAINABLE
        lines.append(f"  capacity gap {gap:.2%} -> PASS")
    return ok, lines


def verdict(diffs, p: float) -> tuple[str, str]:
    d = st.mean(diffs)
    if d >= BAR and p <= ALPHA:
        return "CONFIRMED", (f"+{d:.4f} clears +{BAR} at p {p:.4f}. The brain's own pathway "
                             "BEATS a conventional MLP of the same size.")
    if d <= -BAR and p <= ALPHA:
        return "REFUTED", (f"{d:.4f} clears -{BAR} at p {p:.4f}. The topology COSTS performance "
                           "at matched capacity.")
    return "NULL", (f"{d:+.4f} against a +/-{BAR} bar, p {p:.4f}. Neither better nor measurably "
                    "worse. A BOUND at n=12, NOT proof of equivalence.")


def main() -> None:
    P = load(HERE / "outputs", TAG_P, READOUT_P)
    C = load(HERE / "outputs", TAG_C, READOUT_C)
    print("EXP-064: does the brain's OWN pathway learn a policy?")
    print(f"  cells on disk: P {len(P)}, C {len(C)} of {len(SEEDS)} seeds each")
    for name, recs in (("P", P), ("C", C)):
        short = sorted(set(SEEDS) - set(recs))
        if short:
            print(f"  INCOMPLETE: arm {name} is missing seeds {short}")

    print("\n--- capacity match (the contrast rests on this) ---")
    cap_ok, cap_lines = check_capacity(P, C)
    print("\n".join(cap_lines))

    print("\n--- validity gates (conditions, checked FIRST) ---")
    gate_ok, gate_lines = check_gates(P, C)
    print("\n".join(gate_lines))

    if not (cap_ok and gate_ok):
        print("\nGATE FAILED. The claim below is VOID and must not be read. Reporting it anyway "
              "is the outcome-dependent reading the pre-registration exists to prevent.")
        return

    seeds, diffs = paired(P, C)
    p, method = permutation_p(diffs)
    lo, hi = ci95(diffs)
    v, why = verdict(diffs, p)
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    print(f"\nCLAIM 1, PRIMARY - P minus C (capacity-matched)   n={len(seeds)}")
    print(f"  arm P mean {st.mean([P[s]['success_rate'] for s in seeds]):.4f}   "
          f"arm C mean {st.mean([C[s]['success_rate'] for s in seeds]):.4f}")
    print(f"  mean diff {st.mean(diffs):+.4f}   approx 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  W-L-T {wins}-{losses}-{len(diffs) - wins - losses}")
    print(f"  {method}")
    print(f"  VERDICT: {v} - {why}")

    print("\n--- mechanism readings (NEVER gates) ---")
    for field, label in (("region_drift", "how far prefrontal+motor travelled"),
                         ("motor_rate_mean", "mean motor firing rate"),
                         ("motor_silent_frac", "fraction of steps with NO motor spike"),
                         ("mean_train_entropy", "policy entropy"),
                         ("revisit_rate", "revisit rate"),
                         ("optimality", "optimality")):
        pm, _, _ = reading(P, field)
        cm, _, _ = reading(C, field)
        fmt = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
        print(f"  {field:20s} P {fmt(pm):>8s}   C {fmt(cm):>8s}   ({label})")

    # ------------------------------------------------------------------ resolution, POST HOC
    #
    # ADDED 2026-09-19 AFTER THE NUMBERS EXISTED, and labelled as such. This is NOT a gate and
    # it does NOT change the verdict above, which stands exactly as pre-registered. It is a
    # correction to a REPORTING bug: the interpretation text below assumed a null meant the
    # arms performed comparably, and had no branch for "both arms scored zero", which is what
    # happened. Printing "COMPETITIVE" for 0.0000 against 0.0000 would be plainly false.
    #
    # THE GATE THIS SHOULD HAVE BEEN. A contrast between two arms that both sit on the floor
    # has no resolution: the difference is 0 by construction and measures nothing about the
    # question. EXP-043 ran this exact config with a LINEAR head at depth 5 and scored 0.3229
    # across 24 seeds, so a floor for the control was calibratable BEFORE dispatch from a
    # number already in the repo. It was not gated on. That is the miss.
    p_mean = st.mean([P[s]["success_rate"] for s in seeds])
    c_mean = st.mean([C[s]["success_rate"] for s in seeds])
    FLOOR = 0.02
    if max(p_mean, c_mean) < FLOOR:
        print("\n--- RESOLUTION CHECK (post-hoc, added after the numbers existed) ---")
        print(f"  BOTH ARMS ARE ON THE FLOOR: P {p_mean:.4f}, C {c_mean:.4f}, bar {FLOOR}.")
        print("  The primary's NULL is VACUOUS, not 'competitive'. A difference between two")
        print("  arms that both scored zero is 0 by construction and says nothing about the")
        print("  topology. EXP-043 ran this same config with a LINEAR head at depth 5 and got")
        print("  0.3229 over 24 seeds, so the control here is BROKEN rather than informative:")
        print("  matching capacity with an MLP head cost 0.32 and bought a dead policy.")
        print("  READ NOTHING ABOUT THE TOPOLOGY FROM THIS RUN.")
        return

    print("\n--- what this means ---")
    if v == "CONFIRMED":
        print("  The five-region topology is not decoration: routing the policy through the")
        print("  brain's own prefrontal-to-motor pathway BEATS a conventional head of the same")
        print("  size. The capstone's central claim has support for the first time.")
    elif v == "REFUTED":
        print("  The topology COSTS performance at matched capacity. The honest headline for the")
        print("  write-up is that the brain's own pathway is worse than a plain MLP on the same")
        print("  features, and the five-region design is not carrying the result.")
    else:
        print("  The brain's own pathway is COMPETITIVE with a conventional MLP of the same size.")
        print("  That is the first evidence the topology can carry a policy at all - and at n=12")
        print("  it is a BOUND, not proof of equivalence. It does not show the topology HELPS.")


if __name__ == "__main__":
    main()
