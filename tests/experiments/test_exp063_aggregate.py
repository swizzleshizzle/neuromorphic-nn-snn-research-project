"""EXP-063's aggregator: the pre-registered rules, and the ways they could silently not hold.

Two gates guard this experiment and BOTH have a live path to the failure modes this project has
already paid for:

  * EXP-058's gate could not PASS - a threshold above the quantity's maximum attainable value.
    Gate 2 here is a fraction whose ceiling is bounded by episode length, and the depth-1
    curriculum stage runs under a 2-step cap that contributes exactly zero. The ceiling is
    computed in the spec; these tests check the aggregator reads the fraction it claims to.
  * The mirror is a gate that cannot FAIL, and `None` coerced to 0.0 is the live path to it:
    a missing reading would sail under a "must be at least X" floor while measuring nothing.

  * And EXP-059's aggregator was missing the PARTIAL case - one arm passing, the other failing.
    Only mutation testing found it. `test_partial_gate_failure_is_a_failure` is that case.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AGG = REPO / "experiments" / "063_learned_readout" / "aggregate.py"


@pytest.fixture(scope="module")
def agg():
    spec = importlib.util.spec_from_file_location("exp063_aggregate", AGG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell(seed, *, ratio=0.60, choice=300, steps=1000, entropy=0.8, success=0.30,
          tag="exp063_attn_d5", readout="memory_attn"):
    return {"seed": seed, "tag": tag, "readout": readout, "success_rate": success,
            "recall_concept_norm_ratio": ratio, "attn_choice_steps": choice,
            "train_steps": steps, "attn_entropy_norm": entropy, "attn_recency_mass": 0.4}


def _arm(**kw):
    return {s: _cell(s, **kw) for s in range(24)}


# ------------------------------------------------------------------------------- statistics

def test_t95_is_indexed_by_df_not_hardcoded_at_n12(agg):
    """EXP-055/056/057 all hardcode 2.201, the multiplier at df=11. This is n=24.

    Breaks copying that constant across, which reports every interval about 6% too wide.
    """
    assert agg.t95_for(24) == 2.069
    assert agg.t95_for(12) == 2.201
    assert agg.t95_for(24) != agg.t95_for(12)


def test_permutation_p_is_exact_for_small_n_and_sampled_above_the_cutoff(agg):
    """The METHOD must be reported, because an exact proportion and a Monte Carlo estimate are
    not interchangeable. Breaks silently sampling everywhere."""
    p_small, how_small = agg.permutation_p([0.1] * 6)
    assert "exact" in how_small
    assert p_small == pytest.approx(2 / 2 ** 6)          # only all-+ and all-- reach |obs|
    _, how_big = agg.permutation_p([0.1] * 24)
    assert "sampled" in how_big and "add-one" in how_big


def test_sampled_p_never_reports_zero(agg):
    """The add-one estimator. Breaks `hits / draws`, which prints 0.0000 and claims an
    exactness that sampling cannot support."""
    p, _ = agg.permutation_p([1.0] * 24)
    assert p > 0.0
    assert p == pytest.approx(1 / (agg.SAMPLED_DRAWS + 1))


def test_paired_refuses_a_none_field(agg):
    """Breaks pairing on a field one arm does not record, which would otherwise raise a
    TypeError deep in a subtraction or, worse, be coerced."""
    a, b = _arm(), _arm()
    a[3]["success_rate"] = None
    with pytest.raises(SystemExit):
        agg.paired(a, b, "success_rate")


# ------------------------------------------------------------------------------ gate reading

def test_choice_frac_is_none_when_the_denominator_is_zero(agg):
    """A cell that recorded no training steps is BROKEN, not a cell with a fraction of 0.0.

    Breaks `choice / (steps or 1)`, which would quietly drag an arm mean toward the floor and
    turn a broken run into a gate failure attributed to the science.
    """
    assert agg.choice_frac(_cell(0, choice=0, steps=0)) is None
    assert agg.choice_frac({"seed": 0}) is None
    assert agg.choice_frac(_cell(0, choice=250, steps=1000)) == pytest.approx(0.25)


def test_gate_reading_returns_missing_seeds_rather_than_skipping_them(agg):
    """THE GATE-THAT-CANNOT-FAIL TEST. Breaks dropping `None` from the mean, which would let an
    arm pass a floor on the strength of the seeds that happened to record a value."""
    arm = _arm(ratio=0.60)
    arm[5]["recall_concept_norm_ratio"] = None
    del arm[9]
    mean, missing, lo = agg.gate_reading(arm, lambda r: r.get("recall_concept_norm_ratio"))
    assert missing == [5, 9]
    assert mean == pytest.approx(0.60)
    assert lo == pytest.approx(0.60)


# --------------------------------------------------------------------------------- the gates

def test_both_gates_pass_when_both_arms_clear_both_floors(agg, monkeypatch):
    """A gate must be able to PASS. EXP-058's could not, and it voided the experiment."""
    monkeypatch.setattr(agg, "GATE1_MIN", 0.05)
    monkeypatch.setattr(agg, "GATE2_MIN", 0.10)
    ok, lines = agg.check_gates({"T": _arm(), "U": _arm(readout="memory_attn_noise",
                                                        tag="exp063_attnnoise_d5")})
    assert ok is True
    assert sum("PASS" in ln for ln in lines) == 4        # two gates x two arms


def test_both_gates_can_fail(agg, monkeypatch):
    """And a gate must be able to FAIL, measured on the same fixture shape."""
    monkeypatch.setattr(agg, "GATE1_MIN", 0.05)
    monkeypatch.setattr(agg, "GATE2_MIN", 0.10)
    ok, _ = agg.check_gates({"T": _arm(ratio=0.001, choice=1, steps=1000), "U": _arm()})
    assert ok is False


def test_partial_gate_failure_is_a_failure(agg, monkeypatch):
    """THE CASE EXP-059'S AGGREGATOR WAS MISSING, found only by mutation testing.

    One arm passing and the other failing is a FAILURE. Breaks any `any(...)` in place of
    `all(...)`, and breaks checking only the first arm.
    """
    monkeypatch.setattr(agg, "GATE1_MIN", 0.05)
    monkeypatch.setattr(agg, "GATE2_MIN", 0.10)
    ok, lines = agg.check_gates({"T": _arm(), "U": _arm(ratio=0.001)})
    assert ok is False
    assert any("FAIL" in ln for ln in lines) and any("PASS" in ln for ln in lines)


def test_a_missing_reading_fails_the_gate(agg, monkeypatch):
    """Breaks treating an absent measurement as a pass - the `None` -> 0.0 path, and the
    reason `gate_reading` returns its missing seeds instead of swallowing them."""
    monkeypatch.setattr(agg, "GATE1_MIN", 0.05)
    monkeypatch.setattr(agg, "GATE2_MIN", 0.10)
    arm = _arm()
    arm[2]["recall_concept_norm_ratio"] = None
    ok, lines = agg.check_gates({"T": arm, "U": _arm()})
    assert ok is False
    assert any("MISSING" in ln for ln in lines)


def test_an_uncalibrated_threshold_fails_rather_than_passes(agg, monkeypatch):
    """`GATE_MIN_* = None` means the floor was never calibrated. Breaks a `>=` against None
    (a TypeError at best) and breaks skipping an unset gate, which would silently drop a
    pre-registered condition."""
    monkeypatch.setattr(agg, "GATE1_MIN", None)
    monkeypatch.setattr(agg, "GATE2_MIN", 0.10)
    ok, lines = agg.check_gates({"T": _arm(), "U": _arm()})
    assert ok is False
    assert any("NO THRESHOLD SET" in ln for ln in lines)


# ------------------------------------------------------------------------------- the verdicts

def test_verdict_confirmed_requires_both_the_bar_and_the_alpha(agg):
    """Breaks reading significance alone as confirmation - an effect of +0.001 at p 0.001 is
    not what the spec pre-registered."""
    assert agg.verdict([0.08] * 24, 0.001, 0.05, 0.05)[0] == "CONFIRMED"
    assert agg.verdict([0.01] * 24, 0.001, 0.05, 0.05)[0] == "NULL"
    assert agg.verdict([0.08] * 24, 0.400, 0.05, 0.05)[0] == "NULL"


def test_verdict_refuted_is_the_wrong_direction_not_merely_a_null(agg):
    """The primary is DIRECTIONAL. Breaks collapsing "the control won" into "no difference",
    which would hide the added capacity actively hurting."""
    v, why = agg.verdict([-0.08] * 24, 0.001, 0.05, 0.05)
    assert v == "REFUTED" and "WRONG direction" in why


def test_a_null_is_labelled_a_bound_in_the_text_itself(agg):
    """The spec fixed this wording before dispatch so a null could not be upgraded afterwards
    into evidence of absence. Breaks softening the string."""
    _, why = agg.verdict([0.001] * 24, 0.9, 0.05, 0.05)
    assert "BOUND" in why and "NOT EVIDENCE OF ABSENCE" in why


def test_bonferroni_is_applied_to_the_secondaries(agg):
    """Three substantive contrasts. Breaks reading the secondaries at 0.05."""
    assert agg.N_CONTRASTS == 3
    assert agg.BONFERRONI == pytest.approx(0.05 / 3)
    assert agg.verdict([0.08] * 24, 0.03, agg.BONFERRONI, 0.05)[0] == "NULL"
    assert agg.verdict([0.08] * 24, 0.03, agg.ALPHA, 0.05)[0] == "CONFIRMED"


# ------------------------------------------------------------------------------------ loading

def test_load_refuses_a_tag_carrying_the_wrong_readout(agg, tmp_path):
    """Every arm shares `arm="regionalized"` and differs ONLY in the readout, so a mislabelled
    tag would contrast an arm against ITSELF with nothing in the numbers looking wrong."""
    (tmp_path / "a.json").write_text(json.dumps(_cell(0, readout="memory")))
    with pytest.raises(SystemExit):
        agg.load(tmp_path, "exp063_attn_d5", "memory_attn")


def test_load_ignores_other_tags_in_the_same_directory(agg, tmp_path):
    """Both new arms write into one outputs folder. Breaks a glob that pools them."""
    (tmp_path / "t.json").write_text(json.dumps(_cell(0)))
    (tmp_path / "u.json").write_text(json.dumps(
        _cell(1, tag="exp063_attnnoise_d5", readout="memory_attn_noise")))
    assert sorted(agg.load(tmp_path, "exp063_attn_d5", "memory_attn")) == [0]
