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
        within={"E10": 0.7, "E20": 0.6, "E40": 0.8, "E80": 0.5}, between=-0.9)
    assert "RETIRED" in verdict
    assert "fifth inverted instrument" in verdict


def test_agreeing_signs_keep_the_metric():
    verdict = _module().claim4_verdict(
        within={"E10": 0.7, "E20": 0.6, "E40": 0.8, "E80": 0.5}, between=0.9)
    assert "RETIRED" not in verdict


def test_a_mixed_within_arm_picture_does_not_silently_pass():
    """If the within-arm correlations disagree with EACH OTHER, there is no coherent within
    sign to compare against, and the aggregator must say so rather than pick one."""
    verdict = _module().claim4_verdict(
        within={"E10": 0.7, "E20": -0.6, "E40": 0.8, "E80": -0.5}, between=0.9)
    assert "INCONCLUSIVE" in verdict
    assert "RETIRED" not in verdict
