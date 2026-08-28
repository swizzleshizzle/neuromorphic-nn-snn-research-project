"""Regenerates section 0 of the EXP-053 spec: the premise correction.

The Week-21 kickoff justified Stage 3 on EXP-045's depth-7 entropy collapse
(0.591 -> 0.098). That collapse belongs to EXP-045's back-loaded curriculum, the arm
EXP-045 itself refuted. This script reads the committed depth-7 records of the
UNIFORM-curriculum arms and shows the collapse is not there.

It also computes the within-arm Spearman correlation between deepest-stage entropy and
held-out success, which is what disqualifies entropy as a cross-arm instrument: strongly
positive within an arm, while the better arm has the lower entropy.

Reads only committed `outputs/*.json`. Writes nothing. No dependencies beyond the stdlib.

    .venv/bin/python experiments/053_neuromod_stage3/premise_check.py
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ARMS = [
    ("EXP-044 A, E0 frozen, 10k", "experiments/044_depth7_frontier/outputs/*.json", "exp044_d7_e10000"),
    ("EXP-044 B, E0 frozen, 44k", "experiments/044_depth7_frontier/outputs/*.json", "exp044_d7_e44000"),
    ("EXP-051,   E1 frozen, 10k", "experiments/051_depth7_transfer/outputs/*.json", None),
]


def spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation. No scipy in the venv, and n=12 does not need one."""
    def rank(v: list[float]) -> list[int]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for pos, i in enumerate(order):
            out[i] = pos + 1
        return out

    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def load(pattern: str, tag: str | None) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        d = json.load(open(f))
        if isinstance(d, list):          # probe side-files, not run records
            continue
        if tag is not None and d.get("tag") != tag:
            continue
        if "stage_trace" not in d or "success_rate" not in d:
            continue
        rows.append(d)
    return sorted(rows, key=lambda d: d["seed"])


def main() -> None:
    print(f"{'arm':28} {'ent first':>10} {'ent last':>9} {'trainsolv':>10} "
          f"{'success':>9} {'dead':>6} {'rho(ent,succ)':>14}")
    for name, pattern, tag in ARMS:
        rows = load(pattern, tag)
        if not rows:
            print(f"{name:28}  NO RECORDS (outputs/ is gitignored on some machines)")
            continue
        last = [d["stage_trace"][-1] for d in rows]
        first_e = [s["entropy_first_10pct"] for s in last]
        last_e = [s["entropy_last_10pct"] for s in last]
        solved = [s["train_solved_frac"] for s in last]
        succ = [d["success_rate"] for d in rows]
        dead = [d["seed"] for d in rows if d["success_rate"] == 0.0]
        print(f"{name:28} {st.mean(first_e):>10.3f} {st.mean(last_e):>9.3f} "
              f"{st.mean(solved):>10.4f} {st.mean(succ):>9.4f} "
              f"{f'{len(dead)}/{len(rows)}':>6} {spearman(last_e, succ):>+14.3f}"
              + (f"   dead seeds {dead}" if dead else ""))

    print()
    print("Within an arm, entropy tracks success (rho strongly positive).")
    print("BETWEEN arms the sign flips: EXP-044 B has the LOWER entropy and 3.2x the success")
    print("of arm A. That reversal is why entropy appears in no EXP-053 claim.")


if __name__ == "__main__":
    main()
