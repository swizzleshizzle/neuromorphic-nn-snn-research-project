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


def test_a_non_significant_small_contrast_is_reported_as_a_bound():
    """THE CLAIM 1 RULE. A non-significant difference is not evidence of equality, and n=12 is
    exactly where that conversion is tempting."""
    out = _module().describe_contrast(delta=0.004, p=0.82, bar=0.05, alpha=0.05)
    assert "bound" in out.lower()
    assert "indistinguishable" in out.lower()
    for forbidden in ("as good as", "equal", "equivalent", "no different"):
        assert forbidden not in out.lower(), f"the wording claims equivalence: {out}"


def test_a_confirming_contrast_says_confirmed():
    out = _module().describe_contrast(delta=0.08, p=0.01, bar=0.05, alpha=0.05)
    assert "CONFIRMED" in out


def test_a_large_but_non_significant_contrast_is_not_a_bound_below_the_bar():
    """delta 0.09 at p 0.30 does NOT bound the effect below 0.05 - it is simply unresolved,
    and saying otherwise would invert the finding."""
    out = _module().describe_contrast(delta=0.09, p=0.30, bar=0.05, alpha=0.05)
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
