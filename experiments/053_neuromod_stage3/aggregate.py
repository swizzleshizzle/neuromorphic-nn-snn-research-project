"""EXP-053 aggregator: apply the pre-registered rules to the records on disk.

Thresholds committed in the spec before any number existed.

> THE PROBE IS DELIBERATELY ABSENT, and so is any claim on the ENTROPY TRACE. Entropy is
> the fourth instrument in this project to move against policy quality: within EXP-044 arm A
> it correlates with success at Spearman +0.881, while between arms the better arm has the
> LOWER entropy. It is printed for the record and it decides nothing.

Usage:
    .venv/bin/python experiments/053_neuromod_stage3/aggregate.py
"""

from __future__ import annotations

import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP047_DIR = HERE.parent / "047_encoder_finetuning" / "outputs"
EXP051_DIR = HERE.parent / "051_depth7_transfer" / "outputs"

BAR, ATTRIBUTION_BAR, ALPHA = 0.05, 0.03, 0.05
SEEDS = tuple(range(12))

ARM_TAGS = {"B": "exp053_critic_d7", "G": "exp053_gate_d6", "R": "exp053_rgate_d6"}
CONTROL = {
    "B": (EXP051_DIR, "exp051_transfer_d7", 0.1471),
    "G": (EXP047_DIR, "exp047_ft_d6_lr0.0001", 0.2700),
}
# EXP-046's budget curve, used to price the compute confound rather than wave at it.
BUDGET_RATE = {6: 0.22, 7: 0.210}


def permutation_p(diffs) -> float:
    """Exact two-sided paired permutation over all 2**n sign flips. No scipy in the venv,
    and at n=12 that is 4096 flips, which is cheap and assumption-free."""
    n, obs = len(diffs), abs(sum(diffs))
    return sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12) / 2 ** n


def load(directory: Path, tag: str) -> dict:
    out = {}
    for p in Path(directory).glob("*.json"):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and r.get("tag") == tag:
            out[r["seed"]] = r
    return out


def paired(arm: dict, control: dict, field: str = "success_rate"):
    """Per-seed differences, arm minus control, over the seeds BOTH have."""
    seeds = sorted(set(arm) & set(control) & set(SEEDS))
    diffs = [arm[s][field] - control[s][field] for s in seeds]
    return seeds, diffs


def summarise(seeds, diffs) -> tuple[float, float]:
    return (st.mean(diffs) if diffs else 0.0), permutation_p(diffs) if diffs else 1.0


def wlt(diffs) -> str:
    w = sum(1 for d in diffs if d > 0)
    l = sum(1 for d in diffs if d < 0)
    return f"{w}-{l}-{len(diffs) - w - l}"


def claim3_verdict(g_vs_control: tuple[float, float],
                   g_vs_r: tuple[float, float]) -> str:
    """The decision table fixed in the spec section 3, Claim 3.

    Takes (delta, p) for each contrast and returns what may be claimed. This is a FUNCTION
    and not a paragraph in a write-up deliberately: EXP-050's Claim 4 was satisfied and
    still reasoned wrongly, and EXP-052's aggregator named a shape from indistinguishable
    means. Encoding the table removes the step where a human re-derives it under the
    influence of the numbers.
    """
    gd, gp = g_vs_control
    rd, rp = g_vs_r
    g_wins = gd >= BAR and gp <= ALPHA
    g_loses = gd <= -BAR and gp <= ALPHA
    beats_r = rd >= ATTRIBUTION_BAR and rp <= ALPHA

    if g_loses:
        return ("CLAIM 2 LOSS. The gate costs. Report the loss and its size. No "
                "neuromorphic claim.")
    if g_wins and beats_r:
        return ("CLAIM 2 and CLAIM 3 CONFIRMED. The neuromodulatory gate works: the bus is "
                "load-bearing and the SIGNAL is doing the work, not merely the rate.")
    if g_wins and not beats_r:
        return ("CLAIM 2 CONFIRMED, CLAIM 3 NOT. Fewer encoder updates is the whole effect. "
                "Report as an efficiency finding. The neuromorphic claim is NOT made.")
    # A delta that clears the bar but misses significance is NOT a null result. Calling it
    # one would repeat EXP-050's Claim 4 error: the pre-registered condition was satisfied
    # and the inference drawn from it was still wrong. Underpowered and measured-null are
    # different states of the world and this table must not collapse them.
    if abs(gd) >= BAR and gp > ALPHA:
        return ("CLAIM 2 AMBIGUOUS. The delta clears the bar at " + f"{gd:+.4f}" + " but does "
                "not reach significance (p " + f"{gp:.4f}" + "). This is UNDERPOWERED, not a "
                "null result, and it must not be reported as a refutation. Report the delta, "
                "the p-value and the seed count, and state that n=12 could not resolve it.")
    # A delta that is SIGNIFICANT but does not clear the bar is a different state of the
    # world again, and the fifth row's fix did not cover it: `(gd=+0.03, gp=0.02)` used to
    # fall all the way to the catch-all below and print "REFUTED", which contradicts the
    # p-value describing it. A significant nonzero delta is a real effect, just a small one.
    # The pre-registered +0.05 bar was not met, so Claim 2 is still not confirmed - but that
    # is "bar not met", not "no effect", and the word REFUTED belongs only to a genuine null.
    if gp <= ALPHA and abs(gd) < BAR:
        if gd > 0:
            return ("CLAIM 2 NOT CONFIRMED (bar not met). The delta is significant and "
                    f"POSITIVE at {gd:+.4f} (p {gp:.4f}), but below the pre-registered +{BAR} "
                    "bar. This is a real but small effect, not a null: the encoder updates "
                    "did something, just not enough to confirm Claim 2. Report the delta, "
                    "the p-value and the bar it missed.")
        return ("CLAIM 2 NOT CONFIRMED (bar not met). The delta is significant and NEGATIVE "
                f"at {gd:+.4f} (p {gp:.4f}), with magnitude below the pre-registered {BAR} "
                "bar. This is a real but small cost, not a null: the encoder updates hurt, "
                "just not by enough to call it a loss. Report the delta, the p-value and the "
                "bar it missed.")
    return ("CLAIM 2 NOT CONFIRMED and CLAIM 3 NOT CONFIRMED. Encoder updates are redundant "
            "at this rate. The neuromorphic claim is REFUTED, not deferred. "
            "'We need a better gate' is NOT an available conclusion from this experiment.")


def report_contrast(name: str, arm: dict, control: dict, bar: float) -> tuple[float, float]:
    seeds, diffs = paired(arm, control)
    delta, p = summarise(seeds, diffs)
    verdict = "CONFIRMED" if delta >= bar and p <= ALPHA else "not confirmed"
    print(f"\n{name}")
    print(f"  n {len(seeds)}   arm {st.mean([arm[s]['success_rate'] for s in seeds]):.4f}"
          f"   control {st.mean([control[s]['success_rate'] for s in seeds]):.4f}")
    print(f"  paired delta {delta:+.4f}   W-L-T {wlt(diffs)}   exact p {p:.4f}"
          f"   bar +{bar}  ->  {verdict}")
    for field in ("revisit_rate", "optimality"):
        _, md = paired(arm, control, field)
        print(f"  CLAIM 4 mechanism  {field:<13} {st.mean(md):+.4f}  p {permutation_p(md):.4f}")
    dead_arm = sum(1 for s in seeds if arm[s]["success_rate"] == 0.0)
    dead_ctl = sum(1 for s in seeds if control[s]["success_rate"] == 0.0)
    print(f"  CLAIM 5 dead seeds  arm {dead_arm}/{len(seeds)}   control {dead_ctl}/{len(seeds)}")
    ents = [arm[s]["stage_trace"][-1]["entropy_last_10pct"] for s in seeds]
    print(f"  (entropy last-10% {st.mean(ents):.4f} - RECORDED, DECIDES NOTHING, see spec 0)")
    return delta, p


def main() -> None:
    arms = {a: load(HERE / "outputs", t) for a, t in ARM_TAGS.items()}

    print("EXP-053: a learned critic, and a neuromodulated plasticity gate")
    print("Thresholds pre-registered. The probe and the entropy trace decide nothing.")

    b_dir, b_tag, b_mean = CONTROL["B"]
    if arms["B"]:
        report_contrast(f"CLAIM 1 PRIMARY, the critic. Arm B vs EXP-051 ({b_mean})",
                        arms["B"], load(b_dir, b_tag), BAR)
        evs = [r["critic_ev"] for r in arms["B"].values()]
        print(f"  critic explained variance, final stage: mean {st.mean(evs):.4f}")

    g_vs_control = g_vs_r = None
    g_dir, g_tag, g_mean = CONTROL["G"]
    if arms["G"]:
        g_control = {s: r for s, r in load(g_dir, g_tag).items() if s in SEEDS}
        g_vs_control = report_contrast(
            f"CLAIM 2 PRIMARY, the gate. Arm G vs EXP-047 ({g_mean}). "
            "DIRECTION NOT PRE-COMMITTED", arms["G"], g_control, BAR)
        rates = [r["gate_rate"] for r in arms["G"].values()]
        print(f"  realized gate_rate: mean {st.mean(rates):.4f}  "
              f"min {min(rates):.4f}  max {max(rates):.4f}")

    if arms["G"] and arms["R"]:
        g_vs_r = report_contrast(
            "CLAIM 3 ATTRIBUTION. Arm G vs arm R, the rate-matched control",
            arms["G"], arms["R"], ATTRIBUTION_BAR)

    if g_vs_control and g_vs_r:
        print("\n" + "=" * 78)
        print(claim3_verdict(g_vs_control, g_vs_r))
        print("=" * 78)

    print("\nCLAIM 6 COMPUTE. Arms G and R take strictly FEWER optimizer steps than their")
    print("control, so a WIN there cannot be a compute artifact; a LOSS might be. Arm B adds")
    print(f"65 parameters and one forward per step. Budget curve: {BUDGET_RATE} per log10.")
    print("Price any per-step cost difference against it before reading a delta as real.")


if __name__ == "__main__":
    main()
