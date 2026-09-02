"""EXP-056 aggregator: apply the pre-registered rules to the records on disk.

Thresholds committed in the spec at `48e1e36`, before any number existed.

> THE VALIDITY GATE IS A CONDITION, NOT A CAVEAT. Flattening removes the within-episode
> variation of `V`. If there was none to remove, Claim 1's null says nothing about
> state-dependence and says only that the intervention was inert. That is checked BEFORE the
> Claim 1 verdict is formed, in the same style as EXP-055's shape gate and EXP-054's (fixed)
> Claim 4 gate, because prose did not stop this class of error in either of those.

Usage:
    .venv/bin/python experiments/056_flattened_critic/aggregate.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

_RUN_PATH = HERE / "run.py"
_spec = importlib.util.spec_from_file_location("exp056_run", _RUN_PATH)
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

# EXP-055's interval wording, reused rather than reimplemented. Its gating was fixed at
# `439f6bc` after it asserted the interval reached the bar without consulting the interval.
_A55 = HERE.parent / "055_pretraining_left_edge" / "aggregate.py"
_s55 = importlib.util.spec_from_file_location("exp055_agg", _A55)
_agg55 = importlib.util.module_from_spec(_s55)
_s55.loader.exec_module(_agg55)
describe_contrast = _agg55.describe_contrast

ALPHA = 0.05
BAR = 0.05
BONFERRONI = 0.025          # 0.025 = 0.05 / 2: Claim 1 and Claim 2 are the whole policy family
T95_DF11 = 2.201            # two-sided 95% t multiplier at df=11 (n=12 paired)
GATE_FRACTION = 0.05        # spec Claim 3: V's within-episode RMS against the returns' own
TAG = _run.TAG
SEEDS = _run.SEEDS

POWER_NOTE = """  POWER, stated in the spec before the numbers existed: the measured paired sd for
  arm B minus its control is 0.0826, and at that spread n=12 gives roughly 28%
  power for a +0.05 effect. TWO OF THE THREE Claim 1 outcomes are therefore
  unlikely to resolve even if true, and 'indistinguishable' is the modal result
  whatever the truth is. It is a BOUND with an interval, never evidence that the
  arms behave identically."""


def permutation_p(diffs) -> float:
    """Exact two-sided paired permutation over all 2**n sign flips. No scipy in the venv."""
    n, obs = len(diffs), abs(sum(diffs))
    return sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12) / 2 ** n


def load(directory: Path, tag: str) -> dict:
    out = {}
    for p in Path(directory).glob("*.json"):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and r.get("tag") == tag:
            out[int(r["seed"])] = r
    return out


def paired(arm: dict, control: dict, field: str = "success_rate"):
    seeds = sorted(set(arm) & set(control) & set(SEEDS))
    return seeds, [arm[s][field] - control[s][field] for s in seeds]


def gate_by_stage(arm: dict) -> list[dict]:
    """Claim 3. How much within-episode variation did `V` actually have, per stage?

    Averaged over seeds. `critic_within_rms` and `return_within_rms` are pooled within-episode
    RMS values: `critic_fit_terms` centres every episode on its OWN mean before summing, so
    between-episode variation cannot leak in and inflate this.
    """
    seeds = sorted(arm)
    n_stages = len(arm[seeds[0]]["stage_trace"])
    rows = []
    for i in range(n_stages):
        cw = [arm[s]["stage_trace"][i].get("critic_within_rms", 0.0) for s in seeds]
        rw = [arm[s]["stage_trace"][i].get("return_within_rms", 0.0) for s in seeds]
        c, r = st.mean(cw), st.mean(rw)
        rows.append({
            "depth": arm[seeds[0]]["stage_trace"][i]["depth"],
            "critic_within_rms": c,
            "return_within_rms": r,
            "ratio": (c / r) if r else 0.0,
        })
    return rows


def gate_passed(rows) -> bool:
    """The intervention removed something measurable at SOME stage. The spec's condition is
    'below 5% at EVERY stage' for the null to be uninterpretable, so passing needs only one."""
    return any(row["ratio"] >= GATE_FRACTION for row in rows)


def claim1_verdict(diffs, p: float, gate_ok: bool, bar: float = BAR,
                   alpha: float = ALPHA) -> str:
    """The three pre-registered readings, with the validity gate checked FIRST.

    Claim 1 is not directional: the spec pre-registers `F` above `B` as an informative outcome
    meaning `V`'s within-episode variation was harmful noise in the baseline, which is
    consistent with an explained variance near zero at depth 7. So this must not collapse the
    two signs, and must not treat a positive delta as the only success.
    """
    n = len(diffs)
    delta = st.mean(diffs)
    se = st.stdev(diffs) / (n ** 0.5) if n > 1 else 0.0
    lo, hi = delta - T95_DF11 * se, delta + T95_DF11 * se
    interval = f"approx 95% interval [{lo:+.4f}, {hi:+.4f}]"

    if not gate_ok:
        return (f"CLAIM 1 UNINTERPRETABLE. delta {delta:+.4f}, p {p:.4f}. The validity gate "
                f"FAILED: `V` varied within an episode by less than {GATE_FRACTION:.0%} of the "
                "returns' own within-episode spread at every stage, so flattening removed "
                "nothing measurable. This says the intervention was inert. It says NOTHING "
                "about whether within-episode state-dependence matters.")

    if p <= alpha and delta >= bar:
        return (f"CLAIM 1: F ABOVE B. delta {delta:+.4f} at p {p:.4f}, clearing the {bar} bar. "
                "`V`'s within-episode variation was HARMFUL NOISE in the baseline: removing it "
                "improved the arm. Consistent with explained variance near zero at depth 7, "
                "where variation uncorrelated with returns adds gradient variance without "
                "cancelling any.")
    if p <= alpha and delta <= -bar:
        return (f"CLAIM 1: F BELOW B. delta {delta:+.4f} at p {p:.4f}, clearing the {bar} bar "
                "downward. Within-episode state-dependence is doing real work DESPITE "
                "explaining almost none of final-stage return variance. That is the most "
                "surprising of the three pre-registered outcomes and deserves the headline.")
    if p <= alpha:
        return (f"CLAIM 1: significant but sub-bar. delta {delta:+.4f} at p {p:.4f}, {interval}. "
                f"Real and smaller than the {bar} the spec set. Report the delta and the bar.")
    return (f"CLAIM 1: indistinguishable at n={n}. delta {delta:+.4f}, p {p:.4f}, {interval}. "
            "Within-episode state-dependence contributes nothing this design can resolve. "
            "THIS IS A BOUND, NOT AN EQUIVALENCE, and at ~28% power it is the modal outcome "
            "whatever the truth is.")


def report(name: str, arm: dict, control: dict, control_mean: float):
    seeds, diffs = paired(arm, control)
    delta, p = summarise(diffs)
    w = sum(1 for d in diffs if d > 0)
    l = sum(1 for d in diffs if d < 0)
    print(f"\n{name}")
    print(f"  n {len(seeds)}   arm {st.mean([arm[s]['success_rate'] for s in seeds]):.4f}   "
          f"control {st.mean([control[s]['success_rate'] for s in seeds]):.4f}   "
          f"(published {control_mean})")
    print(f"  paired delta {delta:+.4f}   W-L-T {w}-{l}-{len(diffs) - w - l}   "
          f"exact p {p:.4f}   Bonferroni {BONFERRONI}")
    return diffs, p


def summarise(diffs) -> tuple[float, float]:
    return (st.mean(diffs) if diffs else 0.0), (permutation_p(diffs) if diffs else 1.0)


def main() -> None:
    out_dir = HERE / "outputs"
    arm = load(out_dir, TAG)
    if not arm:
        raise SystemExit(f"no EXP-056 records with tag {TAG} in {out_dir}")

    print("EXP-056: does the critic's WITHIN-EPISODE state-dependence do anything?")
    print("Arm B copied field for field, with advantages formed against v.mean() not v.\n")
    print(POWER_NOTE)

    # ---- Claim 3 first. It decides whether Claim 1's null may be interpreted at all. ----
    rows = gate_by_stage(arm)
    ok = gate_passed(rows)
    print("\nCLAIM 3 VALIDITY GATE - did flattening remove anything? Checked BEFORE Claim 1.")
    print(f"{'depth':>6} {'V within RMS':>13} {'return within RMS':>18} {'ratio':>8}")
    for row in rows:
        print(f"{row['depth']:>6} {row['critic_within_rms']:>13.4f} "
              f"{row['return_within_rms']:>18.4f} {row['ratio']:>8.3f}")
    print(f"  gate: {'PASSED' if ok else 'FAILED'} - "
          f"{'at least one stage' if ok else 'NO stage'} has V varying by at least "
          f"{GATE_FRACTION:.0%} of the returns' own within-episode spread.")

    for label, (directory, tag, mean) in _run.CONTROLS.items():
        control = load(directory, tag)
        if not control:
            print(f"\nMISSING control records for {label} ({tag}) in {directory}; skipped.")
            continue
        diffs, p = report(f"vs {label}", arm, control, mean)
        if "arm B" in label:
            print("  " + claim1_verdict(diffs, p, ok))
        else:
            print("  CLAIM 2 (directional, >= +%.2f): " % BAR + describe_contrast(diffs, p))

    print(f"\nMULTIPLICITY: two policy comparisons, Claim 1 and Claim 2, Bonferroni "
          f"{BONFERRONI}. Claim 3 is a condition with no p-value and belongs to neither count.")


if __name__ == "__main__":
    main()
