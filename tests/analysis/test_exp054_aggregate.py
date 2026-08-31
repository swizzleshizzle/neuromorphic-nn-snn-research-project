"""EXP-054: the aggregator's decision rules, tested before any number exists.

Claim 4 is a HARD DISQUALIFIER, not a caveat, and the test below is what keeps it hard. This
project has retired three instruments that each moved opposite to policy quality, and each was
reported with caveats that did not stop later experiments from building inferences on them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

AGG_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "054_sequence_blindness" / "aggregate.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp054_agg", AGG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_permutation_p_on_a_known_case():
    """Twelve all-positive differences is the most extreme two-sided outcome: 2/4096."""
    assert _module().permutation_p([0.1] * 12) == pytest.approx(2 / 4096)


def test_spearman_on_a_monotone_case():
    m = _module()
    assert m.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert m.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_opposite_signs_retire_the_metric():
    """THE DISQUALIFIER. Within-arm positive, between-arm negative - exactly how the entropy
    trace behaved - must retire S on the spot, in the headline and not in a caveat."""
    verdict = _module().claim4_verdict(
        within={"E10": 0.7, "E20": 0.6, "E40": 0.8, "E80": 0.5}, between=-0.9,
        between_resolvable=True)
    assert "RETIRED" in verdict
    assert "fifth inverted instrument" in verdict


def test_agreeing_signs_keep_the_metric():
    verdict = _module().claim4_verdict(
        within={"E10": 0.7, "E20": 0.6, "E40": 0.8, "E80": 0.5}, between=0.9,
        between_resolvable=True)
    assert "RETIRED" not in verdict


def test_a_mixed_within_arm_picture_does_not_silently_pass():
    """If the within-arm correlations disagree with EACH OTHER, there is no coherent within
    sign to compare against, and the aggregator must say so rather than pick one."""
    verdict = _module().claim4_verdict(
        within={"E10": 0.7, "E20": -0.6, "E40": 0.8, "E80": -0.5}, between=0.9,
        between_resolvable=True)
    assert "INCONCLUSIVE" in verdict
    assert "RETIRED" not in verdict


def _synthetic_arm(s_vals, pol_vals):
    return {i: {"S": s_vals[i], "policy_success": pol_vals[i]} for i in range(len(s_vals))}


def test_an_unresolvable_between_arm_axis_cannot_pass():
    """THE CLAIM 4 GATE. On the 2026-08-29 run this printed CLAIM 4 PASSED from a Spearman over
    four S means whose spread was 0.08x the within-arm sd. Agreeing signs are not a pass when
    one of the two signs came from noise. Against the pre-fix code this case returned
    "CLAIM 4 PASSED ... S is not disqualified"."""
    verdict = _module().claim4_verdict(
        within={"E10": -0.119, "E20": -0.245, "E40": -0.329, "E80": -0.147}, between=-0.600,
        between_resolvable=False, why_unresolvable=" (widest S contrast E40 vs E20: p 0.8394)")
    assert "UNEVALUATED" in verdict
    assert "CLAIM 4 PASSED" not in verdict
    assert "CLAIM 4 TRIPPED" not in verdict and "S IS RETIRED" not in verdict
    assert "neither cleared nor retired" in verdict.lower()
    assert "checked and cleared" in verdict.lower(), "must name the misquote it is preventing"


def test_an_unresolvable_between_arm_axis_cannot_retire_either():
    """The gate must fail SAFE in both directions. A sign drawn from noise can no more retire
    the metric than clear it, so opposite signs over indistinguishable means are not a trip."""
    verdict = _module().claim4_verdict(
        within={"E10": 0.7, "E20": 0.6, "E40": 0.8, "E80": 0.5}, between=-0.9,
        between_resolvable=False)
    assert "UNEVALUATED" in verdict
    assert "CLAIM 4 TRIPPED" not in verdict and "S IS RETIRED" not in verdict


def test_axis_is_resolvable_separates_a_real_gap_from_noise():
    """The gate is only worth having if it can tell the two apart. Identical arms up to an
    alternating 1e-4 wobble must be unresolvable; a flat +0.05 shift on every seed must not."""
    m = _module()
    base = [0.20 + 0.01 * (i % 5) for i in range(12)]
    noise = [b + (0.0001 if i % 2 == 0 else -0.0001) for i, b in enumerate(base)]
    shifted = [b + 0.05 for b in base]

    ok, hi, lo, pval = m.axis_is_resolvable(
        {"A": _synthetic_arm(base, base), "B": _synthetic_arm(noise, noise)}, ["A", "B"], "S")
    assert not ok and pval > 0.05, f"noise was called separable at p {pval}"

    ok, hi, lo, pval = m.axis_is_resolvable(
        {"A": _synthetic_arm(base, base), "B": _synthetic_arm(shifted, shifted)}, ["A", "B"], "S")
    assert ok and pval <= 0.05, f"a flat +0.05 shift was called noise at p {pval}"
    assert (hi, lo) == ("B", "A"), "hi/lo must name the widest contrast, not an arbitrary pair"


def test_arm_summary_reports_mean_policy_not_an_arbitrary_seed():
    """THE HEADLINE BUG. The printed `policy` column used to be
    `next(iter(by_arm[arm].values()))["policy_success"]` - whichever seed's JSON file the
    filesystem glob yielded first, not the arm's actual mean. E10 seed 0 is 0.285 against an
    arm mean of 0.2012; that arbitrary-seed number is what used to print in Claim 2's
    deliverable column.

    Two synthetic records with clearly different `policy_success` values: the reported figure
    must be their mean, and must not equal either individual value (which is what an
    arbitrary-element bug would produce for dict iteration order in either direction).
    """
    m = _module()
    records = {0: {"policy_success": 0.1}, 1: {"policy_success": 0.9}}
    result = m.arm_policy_mean(records)
    assert result == pytest.approx(0.5)
    assert result not in (pytest.approx(0.1), pytest.approx(0.9))


def test_arm_trips_alone_flags_an_opposing_arm_even_when_others_agree():
    """Descriptive, does not change the verdict: one arm at +0.881 against an opposite
    between-arm sign is exactly the precedent the spec cites (the entropy trace within
    EXP-044 arm A). It must be visible per-arm even though claim4_verdict's aggregate
    disqualifier requires ALL FOUR within-arm signs to agree before it fires at all.
    """
    m = _module()
    assert m.arm_trips_alone(0.881, -0.5) is True
    assert m.arm_trips_alone(0.881, 0.5) is False
    assert m.arm_trips_alone(-0.2, -0.5) is False


def test_s_cross_reads_a_stored_record_the_same_as_the_tuple_keyed_form():
    """`s_cross(record)` parses the on-disk `{"d1_d2": value}` sim dict and must land on the
    same number as calling `sensitivity_from_similarity` directly on the tuple-keyed form -
    this pins the string-key parsing (`sim_from_record`) against a hand-computed answer, the
    same 0.88 / 1.6 pair used in test_sequence_sensitivity.py.
    """
    m = _module()
    record = {"sim": {"1_1": 1.0, "2_2": 1.0, "3_3": 1.0, "1_2": 0.8, "2_3": 0.4, "1_3": -1.0}}
    assert m.s_cross(record) == pytest.approx(1.6)


def test_separation_table_averages_within_seed_then_across_seeds():
    """Pins the two-level averaging and the "{d1}_{d2}" key parsing against hand
    arithmetic. A future edit to either would otherwise have nothing to catch it.

    separation_table() takes an arm-keyed dict of {seed: record}, so the seed-keyed
    records below are wrapped under one arm, "E10", rather than passed bare.
    """
    m = _module()
    by_arm = {
        "E10": {
            0: {"sim": {"1_1": 1.0, "2_2": 0.8, "1_2": 0.4, "1_3": 0.2,
                        "3_3": 0.6, "2_3": 0.0}},
            1: {"sim": {"1_1": 0.6, "2_2": 0.4, "1_2": 0.2, "1_3": 0.0,
                        "3_3": 0.2, "2_3": 0.4}},
        }
    }
    # seed 0: |dd|=0 -> mean(1.0, 0.8, 0.6) = 0.8 ; |dd|=1 -> mean(0.4, 0.0) = 0.2 ;
    #         |dd|=2 -> 0.2
    # seed 1: |dd|=0 -> mean(0.6, 0.4, 0.2) = 0.4 ; |dd|=1 -> mean(0.2, 0.4) = 0.3 ;
    #         |dd|=2 -> 0.0
    # across seeds: 0 -> 0.6, 1 -> 0.25, 2 -> 0.1
    table = m.separation_table(by_arm)
    assert table["E10"][0] == pytest.approx(0.6)
    assert table["E10"][1] == pytest.approx(0.25)
    assert table["E10"][2] == pytest.approx(0.1)
