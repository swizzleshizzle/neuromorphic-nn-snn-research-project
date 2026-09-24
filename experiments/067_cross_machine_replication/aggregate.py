"""EXP-067 aggregator: EXP-036's own verdicts, re-evaluated on a different machine's retraining.

Spec: `docs/superpowers/specs/2026-09-23-exp067-cross-machine-replication-design.md`.

> WRITTEN BEFORE ANY EXP-067 NUMBER EXISTS. Confirm completion from COUNTS - 24 records, zero
> python processes - and NOT from the run log, which prints one success rate per line.

> THE UNIT OF REPLICATION IS THE VERDICT, NOT THE NUMBER. The week-25 audit already proved the
> numbers differ: seed 0 retrained here shares 0 of 390 parameters with its published head.
> A replication that moves a number without moving a verdict is a SUCCESSFUL replication, and
> that was fixed in the spec so a small numeric shift could not later be presented as a failure
> nor a large one waved away.

> EVERY THRESHOLD IS EXP-036'S, IMPORTED FROM ITS run.py RATHER THAN COPIED. A replication that
> invents its own bar is not a replication.

> THE GATE IS INVERTED: the retrained heads must DIFFER from the published ones. If they
> matched, this machine would not be meaningfully different and the experiment would be testing
> nothing. A same-machine re-run fails this gate, which is what makes it a real one.

Usage:
    .venv/bin/python -u experiments/067_cross_machine_replication/aggregate.py
"""

from __future__ import annotations

import importlib.util
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP036_OUT = REPO / "experiments" / "036_generalisation_gap" / "outputs"

_spec = importlib.util.spec_from_file_location("exp067_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

from neuromorphic.training.cube_baseline import record_filename  # noqa: E402

SEEDS = _run.SEEDS
DEPTH = _run.DEPTH
PUBLISHED_MEAN = _run.EXP036_DEPTH3_MEAN
TOL = _run.REPLICATION_TOLERANCE
BREAK_MULTIPLE = _run.BREAK_MULTIPLE
BREAK_ABSOLUTE = _run.BREAK_ABSOLUTE
GAP_REFUTE_BELOW = _run.GAP_REFUTE_BELOW
GAP_CONFIRM_AT = _run.GAP_CONFIRM_AT
PUBLISHED_GAP = _run.EXP036_DEPTH3_GAP


def load(arm: str) -> dict:
    out = {}
    for p in sorted((HERE / "outputs").glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("arm") == arm and r.get("depth") == DEPTH:
            out[int(r["seed"])] = r
    return out


def mean_of(records: dict, field: str):
    """Mean, plus the seeds where the field is MISSING. Missing is RETURNED, never skipped."""
    vals, missing = [], []
    for s in sorted(SEEDS):
        r = records.get(s)
        v = None if r is None else r.get(field)
        if v is None:
            missing.append(s)
        else:
            vals.append(v)
    return (st.mean(vals) if vals else None), missing


def gap_zone(gap: float) -> str:
    if gap < GAP_REFUTE_BELOW:
        return "REFUTED"
    if gap >= GAP_CONFIRM_AT:
        return "CONFIRMED"
    return "INCONCLUSIVE"


def check_gate(trained: dict) -> tuple[bool, list[str]]:
    """INVERTED. At least one retrained head must DIFFER from its published counterpart."""
    lines, differ, compared, absent = [], 0, 0, []
    for s in sorted(trained):
        cfg_name = Path(trained[s]["config"]["out_dir"])
        name = record_filename_for(trained[s])
        pub = EXP036_OUT / name
        mine = HERE / "outputs" / name
        if not pub.exists() or not mine.exists():
            absent.append(s)
            continue
        compared += 1
        if pub.read_bytes() != mine.read_bytes():
            differ += 1
    if absent:
        lines.append(f"  GATE: checkpoints absent for seeds {absent} - FAIL. The gate cannot be "
                     "evaluated, and an unevaluable gate is never a pass.")
        return False, lines
    ok = differ >= 1
    lines.append(f"  GATE (inverted): {differ} of {compared} retrained heads DIFFER from their "
                 f"published counterparts -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        lines.append("  Every head came back byte-identical, so this machine is not meaningfully "
                     "different and the replication tests nothing.")
    return ok, lines


def record_filename_for(record: dict) -> str:
    """The head filename for a finished record, rebuilt from its own stored config."""
    from neuromorphic.training.cube_baseline import CubeConfig
    cfg = record["config"]
    fields = {k: v for k, v in cfg.items() if k != "out_dir"}
    fields["out_dir"] = Path(cfg["out_dir"])
    for k, v in list(fields.items()):
        if isinstance(v, list):
            fields[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v)
    return record_filename(CubeConfig(**fields)).replace(".json", "_head.pt")


def main() -> None:
    trained, floors = load("regionalized"), load("random")
    print(f"EXP-067: EXP-036 depth {DEPTH} retrained on a DIFFERENT machine")
    print(f"  trained cells {len(trained)} of {len(SEEDS)}   floor cells {len(floors)} of {len(SEEDS)}")
    for name, recs in (("trained", trained), ("floor", floors)):
        short = sorted(set(SEEDS) - set(recs))
        if short:
            print(f"  INCOMPLETE: {name} is missing seeds {short}")

    print("\n--- validity gate (inverted: the runs must DIFFER) ---")
    ok, lines = check_gate(trained)
    print("\n".join(lines))
    if not ok:
        print("\nGATE FAILED. The claims below are VOID and must not be read.")
        return

    mean, miss = mean_of(trained, "success_rate")
    floor, fmiss = mean_of(floors, "success_rate")
    gap, gmiss = mean_of(trained, "generalisation_gap")
    if miss or fmiss or gmiss:
        print(f"\nMISSING success/gap readings: trained {miss}, floor {fmiss}, gap {gmiss}. VOID.")
        return

    delta = mean - PUBLISHED_MEAN
    print(f"\n--- CLAIM 1, PRIMARY: the replication bar (EXP-036's own, tolerance {TOL}) ---")
    print(f"  published (laptop)  {PUBLISHED_MEAN:.4f}")
    print(f"  retrained (this VPS){mean:>8.4f}")
    print(f"  delta               {delta:+.4f}   tolerance +/-{TOL}")
    c1 = abs(delta) <= TOL
    print(f"  VERDICT: {'REPLICATED' if c1 else 'NOT REPLICATED'}")
    print(f"  (same-machine, EXP-036 against EXP-035, this delta was +0.0002)")

    print(f"\n--- CLAIM 2: the 'working' verdict ---")
    need = max(BREAK_MULTIPLE * floor, BREAK_ABSOLUTE)
    c2 = mean >= BREAK_MULTIPLE * floor and mean >= BREAK_ABSOLUTE
    print(f"  measured floor {floor:.4f}   needs >= {BREAK_MULTIPLE}x floor ({BREAK_MULTIPLE*floor:.4f}) "
          f"AND >= {BREAK_ABSOLUTE}")
    print(f"  trained mean {mean:.4f} -> {'WORKING' if c2 else 'NOT WORKING'}  (EXP-036: WORKING)")

    print(f"\n--- CLAIM 3: the gap verdict ---")
    zone = gap_zone(gap)
    pub_zone = gap_zone(PUBLISHED_GAP)
    print(f"  published gap {PUBLISHED_GAP:+.4f} -> {pub_zone}")
    print(f"  retrained gap {gap:+.4f} -> {zone}")
    print(f"  VERDICT: {'SAME ZONE' if zone == pub_zone else 'ZONE MOVED'}")

    print("\n--- what this means ---")
    verdicts_hold = c1 and c2 and zone == pub_zone
    if verdicts_hold:
        print("  ==EVERY PRE-REGISTERED VERDICT SURVIVES RETRAINING ON A DIFFERENT MACHINE,==")
        print("  while the weights do not: seed 0 shares 0 of 390 parameters with its published")
        print("  head. So the write-up may say the NUMBERS are machine-specific and the")
        print("  CONCLUSIONS are not - which is the stronger claim, and it is now measured")
        print("  rather than argued.")
    else:
        print("  AT LEAST ONE PRE-REGISTERED VERDICT MOVED. Retraining on a different machine")
        print("  changes not just the numbers but a conclusion, so the write-up must say that")
        print("  reproduction requires the tracked checkpoints and that retraining is not a")
        print("  substitute even for the findings. That is a weaker position and an honest one.")


if __name__ == "__main__":
    main()
