"""EXP-053: the aggregator's decision rules, tested before any number exists.

Two process failures last week happened with EVERY pre-registered threshold obeyed:
EXP-050's Claim 4 was satisfied and still wrong, and EXP-052's aggregator named a shape
from the ordering of four means that were indistinguishable at p 0.49 to 0.84. So the
INTERPRETIVE layer gets tests, not just the arithmetic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

AGG_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "053_neuromod_stage3" / "aggregate.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp053_agg", AGG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_permutation_p_on_a_known_case():
    """All twelve differences positive is the most extreme two-sided outcome: 2/4096."""
    m = _module()
    assert m.permutation_p([0.1] * 12) == pytest.approx(2 / 4096)


def test_permutation_p_on_a_symmetric_case():
    m = _module()
    assert m.permutation_p([0.1, -0.1]) == pytest.approx(1.0)


def test_claim3_blocks_the_neuromorphic_claim_when_g_matches_r():
    """The row this table exists for. G beat its control, but not its rate-matched control,
    so the gate's RATE is the whole effect."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.08, 0.01), g_vs_r=(0.005, 0.80))
    assert "efficiency" in verdict.lower()
    assert "neuromorphic claim is NOT made" in verdict


def test_claim3_licenses_the_claim_only_on_both():
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.08, 0.01), g_vs_r=(0.06, 0.01))
    assert "load-bearing" in verdict


def test_claim3_calls_a_flat_result_refuted_not_deferred():
    """'We need a better gate' is not an available conclusion. If this test is ever edited
    to allow it, the escape hatch the spec closed has been reopened.

    Also covers the genuinely-flat case reading REFUTED and not AMBIGUOUS, so the
    significant-but-sub-bar branch below cannot be widened to swallow a real null."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.005, 0.9), g_vs_r=(0.002, 0.95))
    assert "REFUTED" in verdict
    assert "deferred" not in verdict.lower().replace("not deferred", "")
    assert "AMBIGUOUS" not in verdict


def test_a_large_but_nonsignificant_result_is_ambiguous_not_refuted():
    """Underpowered is not the same as null. Reporting an unresolved +0.10 as a
    refutation is the EXP-050 Claim 4 error, which this project has made once already."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.10, 0.06), g_vs_r=(0.02, 0.5))
    assert "AMBIGUOUS" in verdict
    assert "REFUTED" not in verdict


def test_a_genuinely_flat_result_is_still_refuted():
    """The escape hatch stays closed. A small delta with no significance is a real null
    and must still read REFUTED, or the significant-but-sub-bar branch below has swallowed
    the whole table. Distinct inputs from the sibling test above: non-significant here
    (p 0.85/0.5) rather than p 0.9/0.95, so this is not a second call on the same pair."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.004, 0.85), g_vs_r=(0.001, 0.5))
    assert "REFUTED" in verdict
    assert "AMBIGUOUS" not in verdict


def test_significant_sub_bar_positive_delta_is_a_small_effect_not_refuted():
    """gd=+0.03, gp=0.02: significant, but under the +0.05 bar. The pre-fix code fell
    through every branch to the REFUTED catch-all, which contradicts a significant
    p-value describing the same delta. Claim 2 still is not confirmed (bar not met), but
    the word REFUTED must not appear - a significant effect is not a null."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.03, 0.02), g_vs_r=(0.02, 0.5))
    assert "REFUTED" not in verdict
    assert "NOT CONFIRMED" in verdict
    assert "+0.05" in verdict or "0.05" in verdict


def test_significant_sub_bar_negative_delta_is_a_small_cost_not_refuted():
    """Mirror of the positive case: a significant but sub-bar-magnitude NEGATIVE delta is a
    real small cost, not a null, and must not read REFUTED."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(-0.03, 0.02), g_vs_r=(0.0, 1.0))
    assert "REFUTED" not in verdict
    assert "cost" in verdict.lower()
    assert "NOT CONFIRMED" in verdict


def test_the_added_row_fires_when_g_loses_to_its_control_but_beats_r():
    """THE ROW THE PRE-REGISTERED TABLE LACKED. "G does not beat its control, yet G beats R"
    had no row: every Claim-2-did-not-confirm branch returned immediately and threw the R
    contrast away, so this case printed the flat REFUTED catch-all and the attribution result
    vanished. Against the pre-fix code this returned "...REFUTED, not deferred..." with no
    mention of R at all."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.01, 0.60), g_vs_r=(0.06, 0.01))
    assert "ATTRIBUTION WITHOUT A PERFORMANCE RESULT" in verdict
    assert "REFUTED" not in verdict, "an attribution win is not a refutation of the gate"
    assert "NOT THE NEUROMORPHIC CLAIM CONFIRMED" in verdict
    assert "RATE and not SPACING" in verdict, "must carry the limitation that bounds the reading"


def test_the_added_row_also_fires_from_the_ambiguous_claim_2_state():
    """The row must not sit in only ONE of the four Claim-2-did-not-confirm states. A delta
    that clears the bar without reaching significance is still "G does not beat its control",
    so an R win must survive that branch too."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.10, 0.06), g_vs_r=(0.06, 0.01))
    assert "CLAIM 2 AMBIGUOUS" in verdict
    assert "ATTRIBUTION WITHOUT A PERFORMANCE RESULT" in verdict


def test_adding_the_row_did_not_change_what_exp053_itself_reported():
    """THE POINT OF ADDING IT NOW. EXP-053's measured contrasts are G vs control +0.0304 at
    p 0.1323 and G vs R +0.0350 at p 0.1167. The R contrast clears the +0.03 attribution bar
    but is NOT significant, so `beats_r` is False and the new row cannot fire. Editing a
    pre-registration after an arm has reported is the outcome-dependent editing the
    pre-registration exists to prevent; this pins that it did not happen."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.0304, 0.1323), g_vs_r=(0.0350, 0.1167))
    assert verdict == (
        "CLAIM 2 NOT CONFIRMED and CLAIM 3 NOT CONFIRMED. Encoder updates are redundant "
        "at this rate. The neuromorphic claim is REFUTED, not deferred. "
        "'We need a better gate' is NOT an available conclusion from this experiment.")
