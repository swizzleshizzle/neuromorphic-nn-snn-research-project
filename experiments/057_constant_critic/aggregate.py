"""EXP-057 aggregator: apply the pre-registered rules to the records on disk.

Thresholds committed in the spec at `e93baf6`, with Claim 4's gate amended to 1e-6 at `3b270b4`,
both before any number existed.

> CLAIM 4 IS CHECKED FIRST AND IT CAN VOID THE EXPERIMENT. A constant critic reads the same scalar
> at every timestep, so its pooled within-episode RMS must sit at the float-reassociation floor. If
> it does not, the arm is not state-blind, it is not the control the spec describes, and no other
> number here means anything.
>
> NOTE THE DIRECTION. EXP-056's gate passes when `V` DOES vary, because that experiment removed
> within-episode variation and a null is vacuous if there was none to remove. This gate passes when
> `V` does NOT vary. The two conditions are opposites and share instrumentation, which is precisely
> why this file was written from scratch rather than copied from that one.

Usage:
    .venv/bin/python experiments/057_constant_critic/aggregate.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

_RUN_PATH = HERE / "run.py"
_spec = importlib.util.spec_from_file_location("exp057_run", _RUN_PATH)
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

# EXP-055's interval wording, reused rather than reimplemented. Its gating was fixed at `439f6bc`
# after it asserted the interval reached the bar without consulting the interval.
_A55 = HERE.parent / "055_pretraining_left_edge" / "aggregate.py"
_s55 = importlib.util.spec_from_file_location("exp055_agg", _A55)
_agg55 = importlib.util.module_from_spec(_s55)
_s55.loader.exec_module(_agg55)
describe_contrast = _agg55.describe_contrast

ALPHA = 0.05
BAR = 0.05
BONFERRONI = 0.05 / 3        # three policy comparisons: the EMA control, arm B, arm F
GATE_CEILING = 1e-6          # spec Claim 4 as amended. Float floor is ~1e-9; arm B's real
                             # within-episode RMS is 0.65 to 2.20, so this is six orders below
                             # any genuine variation and three above the noise.
T95_DF11 = 2.201
TAG = _run.TAG
SEEDS = _run.SEEDS

POWER_NOTE = """  POWER, stated in the spec before the numbers existed: the measured paired sd on
  this project's depth-7 comparison arms is 0.0826, and at that spread n=12 gives
  roughly 28% power for a +0.05 effect. THIS EXPERIMENT IS WELL POWERED FOR
  NOTHING. 'Indistinguishable' is the modal outcome on Claims 2 and 3 whatever is
  true, and every null below is a BOUND with an interval, never an equivalence.
  It was run anyway because the question is load-bearing and the arm is cheap;
  that trade is recorded in the spec rather than discovered here."""


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
    """Claim 4. The pooled within-episode RMS of `V`, per stage, averaged over seeds."""
    seeds = sorted(arm)
    rows = []
    for i in range(len(arm[seeds[0]]["stage_trace"])):
        cw = [arm[s]["stage_trace"][i].get("critic_within_rms", 0.0) for s in seeds]
        rw = [arm[s]["stage_trace"][i].get("return_within_rms", 0.0) for s in seeds]
        ev = [arm[s]["stage_trace"][i].get("critic_ev", 0.0) for s in seeds]
        rows.append({
            "depth": arm[seeds[0]]["stage_trace"][i]["depth"],
            "critic_within_rms": max(cw),      # the WORST seed, not the mean: one varying
                                               # seed is enough to break the claim
            "return_within_rms": st.mean(rw),
            "critic_ev": st.mean(ev),
        })
    return rows


def gate_passed(rows) -> bool:
    """EVERY stage must be below the ceiling. This is the opposite direction to EXP-056's gate."""
    return all(row["critic_within_rms"] < GATE_CEILING for row in rows)


def claim2_verdict(diffs, p: float, bar: float = BAR, alpha: float = ALPHA) -> str:
    """`C` minus arm `B`: the total cost of removing all state-dependence. NOT directional.

    The spec pre-registers `C` ABOVE `B` as an informative outcome meaning the state input was
    net harmful, so this must not collapse the two signs.
    """
    n = len(diffs)
    delta = st.mean(diffs)
    se = st.stdev(diffs) / (n ** 0.5) if n > 1 else 0.0
    lo, hi = delta - T95_DF11 * se, delta + T95_DF11 * se
    interval = f"approx 95% interval [{lo:+.4f}, {hi:+.4f}]"

    if p <= alpha and delta <= -bar:
        return (f"CLAIM 2: C BELOW B. delta {delta:+.4f} at p {p:.4f}, clearing the {bar} bar "
                "downward. That is the total cost of removing ALL state-dependence. Read it "
                "beside EXP-056's -0.0646, which removed only the within-episode part.")
    if p <= alpha and delta >= bar:
        return (f"CLAIM 2: C ABOVE B. delta {delta:+.4f} at p {p:.4f}. The state input was NET "
                "HARMFUL. Consistent with EXP-056's finding that `V`'s variation can be noise, "
                "and a genuine surprise that deserves the headline.")
    if p <= alpha:
        return (f"CLAIM 2: significant but sub-bar. delta {delta:+.4f} at p {p:.4f}, {interval}.")
    return (f"CLAIM 2: indistinguishable at n={n}. delta {delta:+.4f}, p {p:.4f}, {interval}. "
            "A BOUND, NOT AN EQUIVALENCE, and at ~28% power it is the modal outcome whatever "
            "the truth is.")


def report(name: str, arm: dict, control: dict, published: float):
    seeds, diffs = paired(arm, control)
    delta = st.mean(diffs)
    p = permutation_p(diffs)
    w = sum(1 for d in diffs if d > 0)
    l = sum(1 for d in diffs if d < 0)
    print(f"\nvs {name}")
    print(f"  n {len(seeds)}   arm {st.mean([arm[s]['success_rate'] for s in seeds]):.4f}   "
          f"control {st.mean([control[s]['success_rate'] for s in seeds]):.4f}   "
          f"(published {published})")
    print(f"  paired delta {delta:+.4f}   W-L-T {w}-{l}-{len(diffs)-w-l}   "
          f"exact p {p:.4f}   Bonferroni {BONFERRONI:.4f}")
    return diffs, p


def main() -> None:
    arm = load(HERE / "outputs", TAG)
    if not arm:
        raise SystemExit(f"no EXP-057 records with tag {TAG} in {HERE / 'outputs'}")

    print("EXP-057: is the critic's benefit CALIBRATION, or state-dependence?")
    print("Arm B copied field for field, with a critic that CANNOT SEE THE STATE.\n")
    print(POWER_NOTE)

    rows = gate_by_stage(arm)
    ok = gate_passed(rows)
    print("\nCLAIM 4 VALIDITY GATE - is the critic actually state-blind? Checked FIRST.")
    print(f"  Every stage must show a within-episode RMS below {GATE_CEILING:g}.")
    print(f"{'depth':>6} {'worst-seed V within RMS':>24} {'return within RMS':>18} {'critic_ev':>10}")
    for row in rows:
        print(f"{row['depth']:>6} {row['critic_within_rms']:>24.3e} "
              f"{row['return_within_rms']:>18.4f} {row['critic_ev']:>+10.4f}")
    print(f"  gate: {'PASSED' if ok else 'FAILED'}")

    if not ok:
        print("\n" + "=" * 74)
        print("EXPERIMENT VOID. The critic varied within an episode, so it is not state-blind")
        print("and it is not the control the spec describes. No claim below may be reported.")
        print("=" * 74)
        return

    print("\n  `critic_ev` above is DESCRIPTIVE and decides nothing. A constant predictor")
    print("  explains approximately none of the pooled stage-level return variance, which is")
    print("  expected; EXP-056 established that critic_ev does not track whether a baseline")
    print("  helps, so it appears here for the record only.")

    for label, (directory, tag, published) in _run.CONTROLS.items():
        control = load(directory, tag)
        if not control:
            print(f"\nMISSING control records for {label} ({tag}); skipped.")
            continue
        diffs, p = report(label, arm, control, published)
        if "EMA" in label:
            print("  CLAIM 1 (PRIMARY, directional, >= +%.2f): " % BAR + describe_contrast(diffs, p))
        elif "arm B" in label:
            print("  " + claim2_verdict(diffs, p))
        else:
            print("  CLAIM 3 (between-episode contribution): " + describe_contrast(diffs, p))
            print("    A null here must NOT be read as 'between-episode state-dependence")
            print("    contributes nothing'. At ~28% power it is the modal outcome, and the")
            print("    spec pre-registered that warning because this is the contrast most")
            print("    tempting to over-read.")

    print(f"\nMULTIPLICITY: three policy comparisons, Bonferroni {BONFERRONI:.4f}. Claim 4 is a "
          "condition with no p-value and belongs to no family.")


if __name__ == "__main__":
    main()
