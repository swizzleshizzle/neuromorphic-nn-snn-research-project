"""EXP-064's aggregator: the pre-registered rules, and the ways they could silently not hold.

The gate here is ASYMMETRIC and that is the interesting part. Arm C never trains the regions
and never reads the motor pathway, so `region_drift` and `motor_rate_mean` are None for it BY
CONSTRUCTION. Gating arm C on them would be a gate that cannot pass - EXP-058's exact mistake.
So the gate applies to arm P, and arm C is checked for the OPPOSITE: those readings must be
absent, because a control that recorded a drift is not the frozen-brain arm the spec describes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AGG = REPO / "experiments" / "064_motor_policy_path" / "aggregate.py"


@pytest.fixture(scope="module")
def agg():
    spec = importlib.util.spec_from_file_location("exp064_aggregate", AGG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _p(seed, *, success=0.30, drift=0.50, rate=0.12, trainable=15_540):
    return {"seed": seed, "tag": "exp064_motor_d5", "readout": "motor", "success_rate": success,
            "region_drift": drift, "motor_rate_mean": rate, "motor_silent_frac": 0.01,
            "trainable_params": trainable, "mean_train_entropy": 1.0, "revisit_rate": 0.3,
            "optimality": 0.6}


def _c(seed, *, success=0.30, trainable=15_484):
    return {"seed": seed, "tag": "exp064_mlp_d5", "readout": "concept", "success_rate": success,
            "region_drift": None, "motor_rate_mean": None, "motor_silent_frac": None,
            "trainable_params": trainable, "mean_train_entropy": 1.0, "revisit_rate": 0.3,
            "optimality": 0.6}


def _arm(fn, **kw):
    return {s: fn(s, **kw) for s in range(12)}


def test_permutation_is_exact_at_n12(agg):
    """4,096 flips is cheap, so nothing is sampled. Breaks switching to Monte Carlo, which
    would report an estimate where an exact proportion is available."""
    p, how = agg.permutation_p([0.1] * 12)
    assert "exact" in how and "4,096" in how
    assert p == pytest.approx(2 / 2 ** 12)


def test_t95_is_the_df11_multiplier_not_df23(agg):
    """n=12 here, against n=24 in EXP-059/061/063. Breaks copying 2.069 across, which would
    report every interval about 6% too NARROW."""
    assert agg.T95_DF11 == 2.201


def test_gates_pass_when_the_pathway_trained_and_fired(agg):
    """A gate must be able to PASS."""
    ok, lines = agg.check_gates(_arm(_p), _arm(_c))
    assert ok is True and sum("PASS" in ln for ln in lines) == 2


def test_gate_fails_on_a_frozen_pathway(agg):
    """And it must be able to FAIL. A pathway that never moved reads drift 0.0."""
    ok, _ = agg.check_gates(_arm(_p, drift=0.0), _arm(_c))
    assert ok is False


def test_gate_fails_on_a_silent_pathway(agg):
    """The specific vacuity this design risks: at depth 1 the motor region was silent on 67%
    of steps, which hands the head a constant zero vector."""
    ok, _ = agg.check_gates(_arm(_p, rate=0.0), _arm(_c))
    assert ok is False


def test_a_missing_reading_fails_rather_than_passes(agg):
    """Breaks the `None` -> 0.0 path that turns a floor gate into one that cannot fail."""
    P = _arm(_p)
    P[3]["region_drift"] = None
    ok, lines = agg.check_gates(P, _arm(_c))
    assert ok is False and any("MISSING" in ln for ln in lines)


def test_a_control_that_recorded_a_drift_fails_the_gate(agg):
    """THE ASYMMETRIC CHECK. Arm C must have NO drift reading; if it does, it trained the
    regions too and the contrast is not the one the spec registered.

    Breaks checking only arm P and ignoring the control entirely.
    """
    C = _arm(_c)
    for r in C.values():
        r["region_drift"] = 0.4
    ok, lines = agg.check_gates(_arm(_p), C)
    assert ok is False and any("arm C" in ln and "expected None" in ln for ln in lines)


def test_capacity_mismatch_is_caught_from_the_records(agg):
    """Verified from what the optimizer actually held, not from the spec's arithmetic.

    Breaks trusting the config: EXP-063 shipped a confound because a capacity claim was
    reasoned rather than read back.
    """
    ok, _ = agg.check_capacity(_arm(_p), _arm(_c))
    assert ok is True
    bad, lines = agg.check_capacity(_arm(_p, trainable=99_999), _arm(_c))
    assert bad is False and any("MISMATCH" in ln for ln in lines)


def test_verdict_is_three_way_and_not_directional(agg):
    """Both directions are interesting and the spec refuses to privilege one."""
    assert agg.verdict([0.08] * 12, 0.001)[0] == "CONFIRMED"
    assert agg.verdict([-0.08] * 12, 0.001)[0] == "REFUTED"
    assert agg.verdict([0.01] * 12, 0.001)[0] == "NULL"
    assert agg.verdict([0.08] * 12, 0.40)[0] == "NULL"


def test_a_null_is_labelled_a_bound(agg):
    """Fixed before dispatch so a null could not be upgraded into proof of equivalence."""
    _, why = agg.verdict([0.001] * 12, 0.9)
    assert "BOUND" in why and "NOT proof of equivalence" in why


def test_load_refuses_a_tag_carrying_the_wrong_readout(agg, tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_c(0) | {"tag": "exp064_motor_d5"}))
    with pytest.raises(SystemExit):
        agg.load(tmp_path, "exp064_motor_d5", "motor")
