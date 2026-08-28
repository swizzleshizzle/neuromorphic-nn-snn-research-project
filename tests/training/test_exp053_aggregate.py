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
    to allow it, the escape hatch the spec closed has been reopened."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.005, 0.9), g_vs_r=(0.002, 0.95))
    assert "REFUTED" in verdict
    assert "deferred" not in verdict.lower().replace("not deferred", "")


def test_a_large_but_nonsignificant_result_is_ambiguous_not_refuted():
    """Underpowered is not the same as null. Reporting an unresolved +0.10 as a
    refutation is the EXP-050 Claim 4 error, which this project has made once already."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.10, 0.06), g_vs_r=(0.02, 0.5))
    assert "AMBIGUOUS" in verdict
    assert "REFUTED" not in verdict


def test_a_genuinely_flat_result_is_still_refuted():
    """The escape hatch stays closed. A small delta with no significance is a real null
    and must still read REFUTED, or the ambiguity branch has swallowed the whole table."""
    m = _module()
    verdict = m.claim3_verdict(g_vs_control=(0.005, 0.9), g_vs_r=(0.002, 0.95))
    assert "REFUTED" in verdict
    assert "AMBIGUOUS" not in verdict
