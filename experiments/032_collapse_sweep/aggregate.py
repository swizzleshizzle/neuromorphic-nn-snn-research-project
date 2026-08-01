# experiments/032_collapse_sweep/aggregate.py
"""Aggregate EXP-032 records: did either stabilizer fix the cube policy collapse?

Reports the full `entropy_beta` x `normalize_advantages` grid at each depth against the
gate pre-registered in `run.py` before any number existed:

    greedy_modal_action_frac < 0.60  AND  mean_train_entropy > 1.20

**The gate is read from run.py, never redefined here.** A copy would let the bar drift to
meet the data, which is the failure the repo's "never weaken a passing threshold" rule names.

Three things this prints that the driver's inline summary does not:

1. **The baseline audit.** The `(beta=0, normalize=False)` cells are EXP-030/031's exact
   configuration. If they do not reproduce the reference numbers, something about this run
   is wrong and nothing else in the table can be trusted.
2. **Depth 2 as well as depth 3.** EXP-030's +10.8 point primary effect lives at depth 2.
3. **Per-cell collapsed-seed counts**, because a mean of 0.7 could be a uniformly mediocre
   policy or half the seeds pinned at 1.0, and those call for different next steps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("exp032_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

BETAS = _run.BETAS
NORMALIZE = _run.NORMALIZE
DEPTHS = _run.DEPTHS
GATE_MODAL_MAX = _run.GATE_MODAL_MAX
GATE_ENTROPY_MIN = _run.GATE_ENTROPY_MIN

MAX_ENTROPY = math.log(6)
COLLAPSE_BAR = 0.95

# Measured over 20,000 simulated uniform rollouts on 6 actions (2026-07-31).
UNIFORM_FLOOR = {2: 0.380, 3: 0.354}

# EXP-031 depth-3 concept baseline, for the audit below.
REFERENCE = {
    3: {"modal": 0.932, "entropy": 0.541, "success": 0.022},
    2: {"modal": 0.825, "entropy": 0.703, "success": 0.380},
}


def cell(recs, beta, normalize, depth):
    return [r for r in recs
            if r["depth"] == depth
            and r["config"]["entropy_beta"] == beta
            and r["config"]["normalize_advantages"] == normalize]


def fmt(values):
    if not values:
        return f"{'--':>15}"
    if len(values) == 1:
        return f"{values[0]:>15.3f}"
    return f"{st.mean(values):>8.3f}+-{st.stdev(values):<6.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    recs = [json.loads(p.read_text()) for p in sorted(args.out_dir.glob("*.json"))]
    if not recs:
        raise SystemExit(f"no records in {args.out_dir}")

    expected = len(BETAS) * len(NORMALIZE) * len(DEPTHS) * 12
    print(f"EXP-032: {len(recs)}/{expected} records from {args.out_dir}")
    if len(recs) < expected:
        print(f"  PARTIAL RUN: {expected - len(recs)} records missing, treat as provisional")
    print(f"gate (pre-registered in run.py): modal_frac < {GATE_MODAL_MAX} "
          f"AND train_entropy > {GATE_ENTROPY_MIN}\n")

    # 1. Baseline audit before anything else.
    print("BASELINE AUDIT: (beta=0, normalize=False) must reproduce EXP-030/031")
    ok = True
    for depth in DEPTHS:
        sub = cell(recs, 0.0, False, depth)
        if not sub:
            print(f"  depth {depth}: no records yet")
            continue
        modal = st.mean(r["greedy_modal_action_frac"] for r in sub)
        ref = REFERENCE.get(depth, {}).get("modal")
        match = ref is not None and abs(modal - ref) < 0.005
        ok &= bool(match)
        print(f"  depth {depth}: modal_frac {modal:.3f} vs reference {ref}  "
              f"{'MATCH' if match else 'MISMATCH - investigate before trusting the sweep'}")
    print()

    # 2. The grid.
    for depth in DEPTHS:
        floor = UNIFORM_FLOOR[depth]
        print(f"DEPTH {depth}   (collapse = 1.000, uniform floor {floor}, "
              f"entropy ceiling {MAX_ENTROPY:.3f})")
        print(f"{'beta':>7}{'normalize':>11}{'modal_frac':>16}{'train_entropy':>16}"
              f"{'success':>16}{'collapsed':>11}   gate")
        for beta in BETAS:
            for normalize in NORMALIZE:
                sub = cell(recs, beta, normalize, depth)
                if not sub:
                    continue
                modal = [r["greedy_modal_action_frac"] for r in sub]
                ent = [r["mean_train_entropy"] for r in sub]
                succ = [r["success_rate"] for r in sub]
                n_col = sum(1 for x in modal if x >= COLLAPSE_BAR)
                passed = st.mean(modal) < GATE_MODAL_MAX and st.mean(ent) > GATE_ENTROPY_MIN
                print(f"{beta:>7}{str(normalize):>11}{fmt(modal)}{fmt(ent)}{fmt(succ)}"
                      f"{f'{n_col}/{len(sub)}':>11}   {'PASS' if passed else 'fail'}")
        print()

    # 3. Verdict, stated against the gate rather than against whatever looks best.
    passing = []
    for beta in BETAS:
        for normalize in NORMALIZE:
            sub = cell(recs, beta, normalize, 3)
            if not sub:
                continue
            if (st.mean(r["greedy_modal_action_frac"] for r in sub) < GATE_MODAL_MAX
                    and st.mean(r["mean_train_entropy"] for r in sub) > GATE_ENTROPY_MIN):
                passing.append((beta, normalize, st.mean(r["success_rate"] for r in sub)))

    if not passing:
        print("VERDICT: NO cell clears the pre-registered gate at depth 3.")
        print("  The collapse is not fixed by either stabilizer at the values swept.")
        print("  Do NOT re-run the memory arms. Widen the sweep or change the readout instead.")
        print("  Do not relax the gate to manufacture a pass.")
    else:
        print(f"VERDICT: {len(passing)} cell(s) clear the gate at depth 3:")
        for beta, normalize, succ in sorted(passing, key=lambda t: -t[2]):
            print(f"  entropy_beta={beta}, normalize_advantages={normalize}, "
                  f"success={succ:.3f}")
        print("\n  Success rate is reported but is NOT the selection criterion: the claim")
        print("  under test is that the policy reads its input at all.")


if __name__ == "__main__":
    main()
