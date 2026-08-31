"""EXP-055: the aggregator's interpretive rules, tested before any number exists.

Two rules here are responses to specific failures in this project's own history, and both are
implemented as CONDITIONS rather than as conventions in the prose:

  - a non-significant contrast is reported as a BOUND, never as an equivalence
  - a shape word may not be emitted for a contrast that is not significant
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

AGG_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "055_pretraining_left_edge" / "aggregate.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp055_agg", AGG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_permutation_p_on_a_known_case():
    """Twelve all-positive differences is the most extreme two-sided outcome: 2/4096."""
    assert _module().permutation_p([0.1] * 12) == pytest.approx(2 / 4096)


def test_a_non_significant_small_contrast_reports_an_interval_not_a_bound():
    """REPLACES test_a_non_significant_small_contrast_is_reported_as_a_bound, and for a
    correctness reason, not a wording preference: that test asserted "bound" in out.lower(),
    which encoded a FALSE claim. The reviewer measured the paired-difference sd on this
    project's real arms: e10 vs e20 sd 0.137 (se 0.040), e10 vs e40 sd 0.102 (se 0.029). At
    n=12 those give a non-significant result that is consistent with true effects up to
    roughly +0.09 - nearly twice the +0.05 bar - so "BOUNDS the effect below +0.05" is not a
    conservative simplification, it is wrong. The honest output names an interval and says
    plainly that n=12 does not resolve the question; it must still avoid the forbidden
    equivalence phrasing the old test also checked for.
    """
    # Realistic spread (sd about 0.14, matching e10 vs e20) with a small mean, so the interval
    # crosses the +0.05 bar and demonstrates why a bound claim would be false here.
    diffs = [0.15, -0.14, 0.13, -0.12, 0.16, -0.15, 0.09, -0.08, 0.07, -0.055, 0.03, 0.001]
    assert abs(sum(diffs) / len(diffs)) < 0.05, "fixture must stay under the bar to hit this branch"
    out = _module().describe_contrast(diffs, p=0.82, bar=0.05, alpha=0.05)
    assert "interval" in out.lower()
    assert "does not resolve" in out.lower()
    assert "indistinguishable" in out.lower()
    for forbidden in ("as good as", "equal", "equivalent", "no different", "bounds the"):
        assert forbidden not in out.lower(), f"the wording claims a bound or equivalence: {out}"


def test_a_confirming_contrast_says_confirmed():
    out = _module().describe_contrast([0.08] * 12, p=0.01, bar=0.05, alpha=0.05)
    assert "CONFIRMED" in out


def test_a_significant_negative_delta_is_not_reported_as_confirmed():
    """THE CLAIM 1 DIRECTION RULE. Claim 1 is directional: `e10 - e1 >= +0.05`. The old
    implementation used `abs(delta) >= bar`, so a significant -0.08 would have printed
    "CONFIRMED ... clearing the +0.05 bar" - the opposite finding. This must fail against that
    code and pass against the fix."""
    out = _module().describe_contrast([-0.08] * 12, p=0.01, bar=0.05, alpha=0.05)
    assert "CONFIRMED" not in out
    assert "opposite" in out.lower()
    assert "significant" in out.lower()


def test_a_large_but_non_significant_contrast_is_not_a_bound_below_the_bar():
    """delta 0.09 at p 0.30 does NOT bound the effect below 0.05 - it is simply unresolved,
    and saying otherwise would invert the finding."""
    out = _module().describe_contrast([0.09] * 12, p=0.30, bar=0.05, alpha=0.05)
    assert "unresolved" in out.lower()
    assert "bound" not in out.lower()


def test_shape_word_is_refused_when_a_contrast_is_not_significant():
    """THE CLAIM 3 GATE. EXP-052 named a monotone shape from indistinguishable means, and
    EXP-054's aggregator repeated it three days later. The gate is a condition, not a habit."""
    m = _module()
    assert m.shape_word(delta=0.04, p=0.40, alpha=0.05) == "indistinguishable"
    assert m.shape_word(delta=-0.04, p=0.40, alpha=0.05) == "indistinguishable"


def test_shape_word_is_allowed_when_significant():
    m = _module()
    assert m.shape_word(delta=0.06, p=0.01, alpha=0.05) == "rises"
    assert m.shape_word(delta=-0.06, p=0.01, alpha=0.05) == "falls"


def test_load_s_populates_epoch_10_from_exp054_paired_by_seed():
    """THE CLAIM 4 FIX. Claim 4 asks whether S saturates at the same epoch as policy, and the
    comparison that actually answers it is e5 -> e10 - but EXP-055 only measures S on its own
    four new encoders (1, 2, 3, 5), so epoch 10 was absent and ADJACENT's (5, 10) entry hit a
    silent `continue`. EXP-054 already measured S, with the same statistic and the same twelve
    seeds, on the identical exp052_encoder_e10_s*.pt encoders. This asserts the loader actually
    picks those twelve records up, keyed by seed, so the e5 -> e10 contrast is genuinely
    paired rather than silently skipped."""
    m = _module()
    s_by_epoch = m.load_s(m.HERE / "outputs")
    assert 10 in s_by_epoch, "epoch 10 must be populated from EXP-054's records"
    assert sorted(s_by_epoch[10]) == list(range(12))
    for seed, r in s_by_epoch[10].items():
        assert r["S"] is not None
        assert isinstance(r["S_cross"], float)
        assert isinstance(r["level"], float)
