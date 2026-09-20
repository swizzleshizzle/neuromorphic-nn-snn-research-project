"""EXP-066's aggregator: a ONE-SAMPLE reading, and the ways it could quietly stop being one.

EXP-064 failed because its capacity-matched control collapsed, both arms landed on the floor,
and the difference was 0 by construction. EXP-066 removes the control entirely and reads each
arm against a FIXED bar, so no other arm's behaviour can flatten the measurement.

The tests that matter most:

  * `test_verdict_requires_BOTH_the_bar_and_a_positive_interval` - a mean over the bar with an
    interval straddling zero is not evidence the pathway learned, and the spec pre-registered
    both conditions.
  * `test_a_uniformly_negative_sweep_is_reported_as_a_REAL_RESULT` - the wording was fixed
    before any number existed so an all-floor sweep could not later be called inconclusive.
  * `test_the_reference_is_never_given_a_verdict` - EXP-043 is context at a 40x capacity gap.
    Turning it into a control is EXP-063's mistake in the other direction.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AGG = REPO / "experiments" / "066_region_lr_sweep" / "aggregate.py"


@pytest.fixture(scope="module")
def agg():
    spec = importlib.util.spec_from_file_location("exp066_aggregate", AGG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell(seed, *, success=0.30, drift=0.25, rate=0.15, trainable=15_540):
    return {"seed": seed, "tag": "exp066_motor_lr1e3", "readout": "motor",
            "success_rate": success, "region_drift": drift, "motor_rate_mean": rate,
            "trainable_params": trainable, "mean_train_entropy": 1.0}


def _arm(**kw):
    return {s: _cell(s, **kw) for s in range(12)}


def test_t95_is_df11(agg):
    """n=12. Breaks copying 2.069 from the n=24 experiments, which narrows every interval."""
    assert agg.T95_DF11 == 2.201


def test_verdict_requires_BOTH_the_bar_and_a_positive_interval(agg):
    """Breaks dropping either condition. A mean of 0.05 built from one lucky seed has an
    interval through zero and is not evidence the pathway learned."""
    assert agg.verdict([0.30] * 12)[0] == "CONFIRMED"
    # mean clears 0.02 but the spread puts the lower bound below zero
    spiky = [0.60] + [0.0] * 11
    m = sum(spiky) / 12
    assert m >= agg.BAR
    v, why, _, (lo, _hi) = agg.verdict(spiky)
    assert lo < 0.0 and v == "REFUTED" and "interval includes zero" in why


def test_verdict_refuses_a_floor_arm(agg):
    """EXP-064's arms scored exactly 0.0000. That must read REFUTED, never 'competitive'."""
    v, why, m, _ = agg.verdict([0.0] * 12)
    assert v == "REFUTED" and m == 0.0 and "AT THE FLOOR" in why


def test_the_bar_is_exp062s_so_readings_stay_comparable(agg):
    """EXP-062 judged 0.0163 to be at the floor with 7/12 seeds above zero. Breaks quietly
    loosening the bar, which would make the two experiments incomparable."""
    assert agg.BAR == 0.02


def test_gates_pass_and_can_fail(agg):
    """A gate must be able to do both."""
    ok, _ = agg.check_gates("L3", _arm())
    assert ok is True
    frozen, _ = agg.check_gates("L3", _arm(drift=0.0))
    assert frozen is False
    silent, _ = agg.check_gates("L3", _arm(rate=0.0))
    assert silent is False


def test_there_is_no_upper_bound_on_drift(agg):
    """DELIBERATE. The 10k-episode drift at these lrs has never been measured, and a ceiling
    guessed from a 120-episode calibration is the regime error that killed EXP-057's gate.

    Breaks adding an upper bound, which would VOID an arm for being unexpectedly mobile.
    """
    ok, _ = agg.check_gates("L3", _arm(drift=9.9))
    assert ok is True, "drift must be reported and interpreted, never gated from above"


def test_a_missing_reading_fails_rather_than_passes(agg):
    arm = _arm()
    arm[4]["region_drift"] = None
    ok, lines = agg.check_gates("L3", arm)
    assert ok is False and any("MISSING" in ln for ln in lines)


def test_a_wrong_trainable_surface_fails_the_gate(agg):
    """Every cell must be the 15,540-parameter motor arm. Breaks a config drift that would
    make the sweep measure something other than region_lr."""
    ok, lines = agg.check_gates("L3", _arm(trainable=390))
    assert ok is False and any("not the motor arm" in ln for ln in lines)


def test_a_uniformly_negative_sweep_is_reported_as_a_REAL_RESULT(agg, capsys, monkeypatch):
    """The wording was fixed before any number existed. Breaks softening an all-floor sweep
    into 'inconclusive', which is exactly the retreat pre-registration exists to prevent."""
    floor = _arm(success=0.0)
    monkeypatch.setattr(agg, "load", lambda *a, **k: floor)
    agg.main()
    out = capsys.readouterr().out
    assert "ALL THREE ARMS ARE AT THE FLOOR" in out
    assert "REAL RESULT" in out
    assert "does not learn a policy here" in out


def test_the_reference_is_never_given_a_verdict(agg, capsys, monkeypatch):
    """EXP-043 is context at a 40x capacity gap. Breaks reporting a p-value or verdict for it."""
    monkeypatch.setattr(agg, "load", lambda *a, **k: _arm())
    agg.main()
    out = capsys.readouterr().out
    assert "CONTEXT, NOT A CONTROL" in out
    ref_block = out.split("SECONDARY and DESCRIPTIVE")[1].split("what this means")[0]
    assert "VERDICT" not in ref_block and "p " not in ref_block


def test_load_refuses_a_non_motor_record(agg, tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_cell(0) | {"readout": "concept"}))
    with pytest.raises(SystemExit):
        agg.load(tmp_path, "exp066_motor_lr1e3")
