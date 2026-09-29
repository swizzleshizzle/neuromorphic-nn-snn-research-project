"""EXP-069 aggregator. Written and committed BEFORE any EXP-069 number exists.

Rows are the ENCODER (e), columns the REST (r: split_seed = train_seed). The decomposition and the
randomization test are EXP-068's, imported rather than copied, so both experiments are read by the
same audited code (16 tests, 8 of 8 mutations caught).

THE EXP-068 LESSON IS APPLIED: every unresolved band lives IN a verdict function below and is
returned as a verdict of its own. EXP-068 put its band only in the spec's prose, so its aggregator
printed CONFIRMED for a result the spec had declared unresolvable.

Spec: docs/superpowers/specs/2026-09-25-exp069-encoder-decomposition-design.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "exp068_aggregate", HERE.parent / "068_seed_decomposition" / "aggregate.py"
)
_a068 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a068)
anova = _a068.anova
randomization_p = _a068.randomization_p

GRID = tuple(range(8))

# Claim 1 bands, calibrated by simulation at 8x8 BEFORE any number existed. A true share of 0.05
# lands in [0.000, 0.175] 90% of the time; a true 0.30 in [0.046, 0.505]. Misclassification into
# the WRONG extreme is about 5% at worst. Everything between is honestly unresolvable here.
ENCODER_MINOR_BELOW = 0.10
ENCODER_MAJOR_AT = 0.35
# Claim 2 bands. At a true interaction share of 0.60 the estimate spans about [0.39, 0.87].
INTERACTION_REPLICATED_AT = 0.50
INTERACTION_NOT_BELOW = 0.30
# Gate.
MIN_GRAND_MEAN = 0.10     # EXP-036's BREAK_ABSOLUTE: the grid must be a WORKING policy
MIN_TOTAL_SD = 0.05


def encoder_verdict(share_e: float) -> str:
    if share_e < ENCODER_MINOR_BELOW:
        return "MINOR"
    if share_e >= ENCODER_MAJOR_AT:
        return "MAJOR"
    return "UNRESOLVED"


def interaction_verdict(share_i: float) -> str:
    if share_i >= INTERACTION_REPLICATED_AT:
        return "REPLICATED"
    if share_i < INTERACTION_NOT_BELOW:
        return "NOT REPLICATED"
    return "UNRESOLVED"


def check_gate(stats, p_e: float, p_r: float):
    """Three conditions. The first is competence, which EXP-064 taught: a grid of dead policies
    has nothing to decompose however its variance happens to split."""
    working = stats["grand"] >= MIN_GRAND_MEAN
    sd_ok = stats["total_sd"] >= MIN_TOTAL_SD
    resolves = min(p_e, p_r) < 0.05
    return working and sd_ok and resolves, working, sd_ok, resolves


def load_grid(out_dir: Path):
    grid = [[None] * len(GRID) for _ in GRID]
    for path in out_dir.glob("exp069_en*rs*.json"):
        rec = json.load(open(path))
        if not isinstance(rec, dict) or "success_rate" not in rec:
            continue
        body = path.name.split("_")[1]            # en<e>rs<r>
        e_s, r_s = body[2:].split("rs")
        grid[int(e_s)][int(r_s)] = float(rec["success_rate"])
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    grid = load_grid(args.out_dir)
    filled = sum(1 for row in grid for x in row if x is not None)
    print(f"EXP-069: encoder x rest at depth 5, cells {filled} of {len(GRID) ** 2}")
    if filled != len(GRID) ** 2:
        raise SystemExit("incomplete grid; refusing to decompose a partial design.")

    s = anova(grid)
    p_e, p_r = randomization_p(grid)
    print(f"\n  grand mean {s['grand']:.4f}   total sd {s['total_sd']:.4f}")
    print(f"  {'component':<24}{'F':>9}{'share':>9}{'p':>9}")
    print(f"  {'E  encoder':<24}{s['f_a']:>9.3f}{s['share_a']:>9.3f}{p_e:>9.4f}")
    print(f"  {'R  rest (split+train)':<24}{s['f_b']:>9.3f}{s['share_b']:>9.3f}{p_r:>9.4f}")
    print(f"  {'I  interaction':<24}{'-':>9}{s['share_i']:>9.3f}{'-':>9}")

    ok, working, sd_ok, resolves = check_gate(s, p_e, p_r)
    print(f"\n--- GATE ---  working (mean >= {MIN_GRAND_MEAN}): {working}   "
          f"sd >= {MIN_TOTAL_SD}: {sd_ok}   a main effect resolves: {resolves}")
    if not ok:
        raise SystemExit("GATE FAILED - the experiment is VOID and no claim may be read.")
    print("  GATE: PASS")

    print(f"\n--- CLAIM 1, PRIMARY: is the ENCODER a carrier of the seed effect? ---")
    print(f"  encoder share {s['share_a']:.3f}   MINOR < {ENCODER_MINOR_BELOW}, "
          f"MAJOR >= {ENCODER_MAJOR_AT}, UNRESOLVED between")
    print(f"  VERDICT: {encoder_verdict(s['share_a'])}")

    print(f"\n--- CLAIM 2: does EXP-068's headline (interaction dominates) hold here? ---")
    print(f"  interaction share {s['share_i']:.3f}   REPLICATED >= {INTERACTION_REPLICATED_AT}, "
          f"NOT < {INTERACTION_NOT_BELOW}, UNRESOLVED between")
    print(f"  VERDICT: {interaction_verdict(s['share_i'])}")


if __name__ == "__main__":
    main()
