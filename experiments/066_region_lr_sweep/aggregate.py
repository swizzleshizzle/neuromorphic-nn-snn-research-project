"""EXP-066 aggregator: the pre-registered rules, applied to the records on disk.

Spec: `docs/superpowers/specs/2026-09-20-exp066-region-lr-sweep-design.md`.

> WRITTEN BEFORE ANY EXP-066 NUMBER EXISTS. Confirm completion from COUNTS - 24 records, zero
> python processes - and NOT from the run log, which prints one success rate per line.

> THIS IS A ONE-SAMPLE READING, NOT A CONTRAST, and that is the design point. EXP-064's
> capacity-matched control collapsed, both arms landed on the floor, and the difference was 0
> by construction. Here each arm is read against a FIXED bar, so no other arm's behaviour can
> collapse the measurement. The resolution argument is structural: the measured chance floor at
> this depth is 0.000 and a working arm on this exact config scores 0.3229, so the dynamic
> range is the full span and nothing can flatten it.

> ALL THREE ARMS AT THE FLOOR IS A REAL RESULT and this file says so in those words. The spec
> fixed that wording before any number existed, precisely so a uniformly negative sweep could
> not be written up afterwards as inconclusive.

> EXP-043 IS CONTEXT, NEVER A CONTROL. 390 trainable parameters against 15,540, a 40x gap. It
> is reported as a ratio with NO p-value and NO verdict.

Usage:
    .venv/bin/python -u experiments/066_region_lr_sweep/aggregate.py
"""

from __future__ import annotations

import importlib.util
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
_spec = importlib.util.spec_from_file_location("exp066_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

SEEDS = _run.SEEDS
ARMS = _run.ARMS
BAR = _run.BAR
GATE_DRIFT = _run.GATE_MIN_REGION_DRIFT
GATE_RATE = _run.GATE_MIN_MOTOR_RATE
MOTOR_TRAINABLE = _run.MOTOR_TRAINABLE
EXP043_REFERENCE = _run.EXP043_REFERENCE
EXP064_DIR = REPO / "experiments/064_motor_policy_path/outputs"
EXP064_TAG, EXP064_LR = _run.EXP064_TAG, _run.EXP064_LR

T95_DF11 = 2.201   # n=12. NOT 2.069, which is df=23 and belongs to the n=24 experiments.


def ci95(vals) -> tuple[float, float]:
    n = len(vals)
    m = st.mean(vals)
    se = st.stdev(vals) / (n ** 0.5) if n > 1 else 0.0
    return m - T95_DF11 * se, m + T95_DF11 * se


def load(directory: Path, tag: str) -> dict:
    out: dict[int, dict] = {}
    for p in sorted(Path(directory).glob("*.json")):
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or r.get("tag") != tag:
            continue
        if r.get("readout") != "motor":
            raise SystemExit(f"{p.name}: tag {tag} carries readout {r.get('readout')!r}, "
                             "expected 'motor'; refusing.")
        out[int(r["seed"])] = r
    return out


def reading(records: dict, field: str):
    """Mean, the seeds where the field is MISSING, and the min. Missing is RETURNED, never
    skipped: a `None` dropped from a mean turns a floor gate into one that cannot fail."""
    vals, missing = [], []
    for s in sorted(SEEDS):
        r = records.get(s)
        v = None if r is None else r.get(field)
        if v is None:
            missing.append(s)
        else:
            vals.append(v)
    return (st.mean(vals) if vals else None), missing, (min(vals) if vals else None)


def check_gates(name: str, records: dict) -> tuple[bool, list[str]]:
    ok, lines = True, []
    for n, field, floor in ((1, "region_drift", GATE_DRIFT), (2, "motor_rate_mean", GATE_RATE)):
        mean, missing, lo = reading(records, field)
        if missing:
            ok = False
            lines.append(f"  GATE {n} arm {name} ({field}): MISSING for seeds {missing} - FAIL. "
                         "A missing reading is never a pass.")
            continue
        passed = mean >= floor
        ok = ok and passed
        lines.append(f"  GATE {n} arm {name} ({field}): mean {mean:.4f}, min {lo:.4f}, "
                     f"floor {floor} -> {'PASS' if passed else 'FAIL'}")
    caps = {r["trainable_params"] for r in records.values()}
    if caps != {MOTOR_TRAINABLE}:
        ok = False
        lines.append(f"  arm {name}: trainable_params {sorted(caps)}, expected "
                     f"{MOTOR_TRAINABLE} - FAIL. This is not the motor arm the spec describes.")
    return ok, lines


def verdict(vals) -> tuple[str, str, float, tuple[float, float]]:
    """The pre-registered one-sample threshold. BOTH conditions, deliberately: a mean over the
    bar with an interval straddling zero is not evidence the pathway learned."""
    m = st.mean(vals)
    lo, hi = ci95(vals)
    if m >= BAR and lo > 0.0:
        return "CONFIRMED", (f"mean {m:.4f} clears {BAR} and the 95% CI lower bound {lo:+.4f} "
                             "is above zero. THE PATHWAY LEARNS."), m, (lo, hi)
    why = f"mean {m:.4f} against a {BAR} bar, 95% CI [{lo:+.4f}, {hi:+.4f}]"
    if m >= BAR:
        why += " - the mean clears the bar but the interval includes zero"
    return "REFUTED", why + ". AT THE FLOOR.", m, (lo, hi)


def main() -> None:
    arms = [(a["key"], a["region_lr"], load(HERE / "outputs", a["tag"])) for a in ARMS]
    reused = load(EXP064_DIR, EXP064_TAG)
    arms.append(("L2", EXP064_LR, reused))
    arms.sort(key=lambda t: t[1])

    print("EXP-066: can the brain's OWN pathway learn at all?")
    for name, lr, recs in arms:
        src = " (REUSED from EXP-064)" if name == "L2" else ""
        print(f"  arm {name}  region_lr {lr:<8g} cells {len(recs)} of {len(SEEDS)}{src}")
        short = sorted(set(SEEDS) - set(recs))
        if short:
            print(f"    INCOMPLETE: missing seeds {short}")

    print("\n--- validity gates (conditions, checked FIRST) ---")
    all_ok = True
    for name, _lr, recs in arms:
        ok, lines = check_gates(name, recs)
        all_ok = all_ok and ok
        print("\n".join(lines))
    if not all_ok:
        print("\nGATE FAILED. The claims below are VOID and must not be read.")
        return

    print(f"\n--- CLAIM 1, PRIMARY: one-sample against a {BAR} bar, per arm ---")
    results = []
    for name, lr, recs in arms:
        vals = [recs[s]["success_rate"] for s in sorted(recs)]
        v, why, m, (lo, hi) = verdict(vals)
        nz = sum(1 for x in vals if x > 0)
        results.append((name, lr, v, m))
        print(f"\n  arm {name}  region_lr {lr:g}   n={len(vals)}")
        print(f"    {why}")
        print(f"    seeds above zero: {nz}/{len(vals)}   "
              f"drift {reading(recs, 'region_drift')[0]:.4f}   "
              f"entropy {reading(recs, 'mean_train_entropy')[0]:.4f}")
        print(f"    VERDICT: {v}")

    print("\n--- CLAIM 2, SECONDARY and DESCRIPTIVE (no p-value, no verdict) ---")
    best = max(results, key=lambda t: t[3])
    print(f"  best arm {best[0]} at region_lr {best[1]:g}: {best[3]:.4f}")
    print(f"  EXP-043 linear head on this config: {EXP043_REFERENCE:.4f} "
          f"({best[3] / EXP043_REFERENCE:.1%} of it)")
    print(f"  Its trainable surface is 390 against this arm's {MOTOR_TRAINABLE:,}, a 40x gap.")
    print( "  CONTEXT, NOT A CONTROL. No test is reported and none was registered.")

    print("\n--- what this means ---")
    if all(v == "REFUTED" for _n, _l, v, _m in results):
        print("  ALL THREE ARMS ARE AT THE FLOOR, across three orders of magnitude of")
        print("  region_lr, on a configuration where a 390-parameter linear head scores 0.3229.")
        print("  ==The brain's own prefrontal-to-motor pathway does not learn a policy here.==")
        print("  This is a REAL RESULT and the spec said so before any number existed. It is")
        print("  the strongest negative this project can state about the five-region topology,")
        print("  and it closes the question EXP-064 opened and could not answer.")
        print("  It does NOT show the topology is worthless in principle: the recipe, head and")
        print("  curriculum were all tuned around a concept-reading linear head.")
    else:
        win = [r for r in results if r[2] == "CONFIRMED"]
        print(f"  THE PATHWAY LEARNS at region_lr {', '.join(f'{r[1]:g}' for r in win)}.")
        print("  EXP-064's null was an OPTIMISATION artifact, not a statement about the")
        print("  topology: at 1e-2 the regions moved 2.71x their own magnitude and learned")
        print("  nothing. The topology question REOPENS, and the next step is a capacity-matched")
        print("  control that is demonstrated to clear the floor BEFORE it is used as one.")


if __name__ == "__main__":
    main()
