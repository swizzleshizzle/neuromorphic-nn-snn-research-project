"""EXP-056's interpretive rules, tested as CONDITIONS rather than as prose.

The validity gate is the point. EXP-052 named a shape from indistinguishable means, EXP-054
printed a verdict from a rank correlation over noise, and both aggregators had the right words
in their docstrings. Only a condition stopped it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "experiments" / "056_flattened_critic" / "aggregate.py"


def _module():
    spec = importlib.util.spec_from_file_location("exp056_agg", MODULE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_gate_needs_only_one_stage_to_pass():
    m = _module()
    rows = [{"ratio": 0.001}, {"ratio": 0.30}, {"ratio": 0.002}]
    assert m.gate_passed(rows) is True
    assert m.gate_passed([{"ratio": 0.049}, {"ratio": 0.0}]) is False


def test_a_failed_gate_makes_claim_1_uninterpretable_even_when_significant():
    """THE ORDERING. The gate is checked BEFORE the verdict. If flattening removed nothing, a
    significant delta cannot be read as evidence about state-dependence, because the arms did
    not differ in the way the claim is about. A verdict computed first and gated afterwards
    would print a confident reading here."""
    m = _module()
    out = m.claim1_verdict([0.09] * 12, p=0.001, gate_ok=False)
    assert "UNINTERPRETABLE" in out
    assert "inert" in out.lower()
    assert "ABOVE B" not in out and "BELOW B" not in out


def test_a_significant_negative_delta_is_reported_as_f_below_b_not_as_a_null():
    """Claim 1 is NOT directional. The spec pre-registers F below B as the most surprising
    outcome, so a significant negative must not collapse into 'indistinguishable'."""
    m = _module()
    out = m.claim1_verdict([-0.08] * 12, p=0.01, gate_ok=True)
    assert "F BELOW B" in out
    assert "indistinguishable" not in out.lower()


def test_a_significant_positive_delta_is_read_as_harmful_noise_removed():
    m = _module()
    out = m.claim1_verdict([0.08] * 12, p=0.01, gate_ok=True)
    assert "F ABOVE B" in out
    assert "NOISE" in out


def test_a_non_significant_claim_1_is_a_bound_and_never_an_equivalence():
    m = _module()
    diffs = [0.05, -0.04, 0.03, -0.05, 0.04, -0.03, 0.02, -0.02, 0.01, -0.01, 0.0, 0.005]
    out = m.claim1_verdict(diffs, p=0.90, gate_ok=True)
    assert "BOUND, NOT AN EQUIVALENCE" in out
    assert "interval" in out.lower()
    for forbidden in ("equivalent", "no different", "as good as", "identical"):
        assert forbidden not in out.lower(), f"wording claims equivalence: {out}"
