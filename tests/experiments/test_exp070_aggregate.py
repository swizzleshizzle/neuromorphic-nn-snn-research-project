"""EXP-070 aggregator. The gates and the unresolved band are verdicts, not warnings (EXP-068)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "exp070_aggregate",
    Path(__file__).resolve().parents[2] / "experiments" / "070_lookahead_existing" / "aggregate.py",
)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def test_one_sided_p_all_positive_twelve_is_one_over_4096():
    """Catches a two-sided test or a sampled approximation: 12 equal positive diffs reach the
    observed sum only under the identity flip."""
    assert agg.one_sided_p([0.1] * 12) == pytest.approx(1 / 4096)


def test_one_sided_p_of_zeros_is_one():
    """Catches a strict > comparison, which would call a null effect significant."""
    assert agg.one_sided_p([0.0] * 12) == 1.0


@pytest.mark.parametrize("e,p,expected", [
    (0.0, 0.0, "UNRESOLVED"), (0.019, 0.0199, "UNRESOLVED"),
    (0.019, 0.02, "RESOLVED"), (0.99, 0.985, "UNRESOLVED"), (0.98, 0.99, "RESOLVED"),
    (0.0, 0.3, "RESOLVED"),
])
def test_gate1_bands_including_edges(e, p, expected):
    """Catches floor-on-floor being read as a result (EXP-064) and pins edge inclusivity."""
    assert agg.gate1_verdict(e, p) == expected


def test_claim1_floor_on_floor_is_unresolved_even_with_a_positive_mean():
    """Catches the gate being applied after the significance test instead of before."""
    v, _ = agg.claim1_verdict([0.005] * 12, e_mean=0.001, p_mean=0.006)
    assert v == "UNRESOLVED"


def test_claim1_verdicts():
    """Catches a mis-ordered verdict ladder."""
    assert agg.claim1_verdict([0.05] * 12, 0.1, 0.15)[0] == "CONFIRMED"
    assert agg.claim1_verdict([-0.01] * 12, 0.1, 0.09)[0] == "REFUTED"
    assert agg.claim1_verdict([0.0] * 12, 0.1, 0.1)[0] == "REFUTED"
    mixed = [0.02, -0.02] * 6
    mixed[0] = 0.03
    assert agg.claim1_verdict(mixed, 0.1, 0.1008)[0] == "NOT SIGNIFICANT"


def test_wilson_interval_contains_the_point_and_is_inside_unit():
    """Catches a swapped or unclamped interval."""
    lo, hi = agg.wilson95(10, 200)
    assert 0.0 <= lo < 0.05 < hi <= 1.0


def test_gate0_refuses_to_run_before_the_preflight_fixes_its_form():
    """Catches the aggregator choosing Gate 0's form after seeing P numbers. Matches on
    'pre-flight' specifically: a plain `pytest.raises(SystemExit)` also passes if the None
    check is deleted and the call merely falls through to the unrelated 'unknown GATE0_FORM'
    raise at the bottom of the function, which names no pre-flight and is the wrong reason."""
    with pytest.raises(SystemExit, match="pre-flight"):
        agg.gate0_verdict(None, {}, {}, True)


def test_gate0_exact_fails_on_one_count_off():
    """Catches a tolerance sneaking into the exact form."""
    pub = {"d8_s0": {"solved": 23, "n": 200}}
    assert agg.gate0_verdict("exact", {"d8_s0": 23}, pub, True) == "PASS"
    assert agg.gate0_verdict("exact", {"d8_s0": 22}, pub, True) == "FAIL"


def test_gate0_fails_without_determinism_in_either_form():
    """Gate 0(a) is required in both forms."""
    pub = {"d8_s0": {"solved": 23, "n": 200}}
    assert agg.gate0_verdict("exact", {"d8_s0": 23}, pub, False) == "FAIL"
    assert agg.gate0_verdict("wilson", {"d8_s0": 23}, pub, False) == "FAIL"


def test_gate0_wilson_is_per_depth_on_pooled_counts():
    """Catches the Wilson form being applied per cell. A per-cell interval at n=200 is WIDER
    than the pooled one at n=2400, so per-cell is the LOOSER rule: every cell at 8/200 against a
    published 10/200 sits inside its own interval (about [0.027, 0.090]) but pooled 96/2400 =
    0.040 is below the pooled lower bound (about 0.042) and must FAIL."""
    pub = {f"d8_s{s}": {"solved": 10, "n": 200} for s in range(12)}
    near = {f"d8_s{s}": (10 if s else 8) for s in range(12)}
    far = {f"d8_s{s}": 5 for s in range(12)}
    shifted = {f"d8_s{s}": 8 for s in range(12)}
    assert agg.gate0_verdict("wilson", near, pub, True) == "PASS"
    assert agg.gate0_verdict("wilson", far, pub, True) == "FAIL"
    assert agg.gate0_verdict("wilson", shifted, pub, True) == "FAIL"
