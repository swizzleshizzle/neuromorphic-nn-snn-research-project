"""EXP-054: the five arms must point at the right encoders, and E0 must be reconstructed.

A wrong path here would silently measure the wrong pretraining level, and the whole point of
the experiment is the epoch series. A config-level test costs seconds; discovering it after a
run costs the run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUN_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "054_sequence_blindness" / "run.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp054_run", RUN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_five_arms_with_the_pre_registered_epoch_levels():
    arms = _module().ARMS
    assert sorted(a["epochs"] for a in arms.values()) == [0, 10, 20, 40, 80]


def test_e0_is_reconstructed_not_loaded():
    """A random init is exactly reproducible from its seed and no file was ever saved, so E0
    must build rather than load. A path here would be a file that does not exist."""
    assert _module().ARMS["E0"]["path_fn"] is None


def test_each_trained_arm_points_at_its_own_experiment():
    arms = _module().ARMS
    expected = {
        "E10": "052_pretraining_optimum",
        "E20": "052_pretraining_optimum",
        "E40": "040_pretrained_encoder_policy",
        "E80": "050_objective_vs_gradient",
    }
    for name, frag in expected.items():
        path = str(arms[name]["path_fn"](0))
        assert frag in path, f"{name} resolves to {path}, which is not {frag}"


def test_e10_and_e20_are_distinguishable_paths():
    """Both live in the same directory and differ only by the epoch tag. A copy-paste error
    would make them the same file and the 10-vs-20 contrast would be exactly zero."""
    arms = _module().ARMS
    assert str(arms["E10"]["path_fn"](3)) != str(arms["E20"]["path_fn"](3))


def test_policy_values_match_the_pre_registered_series():
    arms = _module().ARMS
    assert arms["E0"]["policy"] == pytest.approx(0.0000)
    assert arms["E10"]["policy"] == pytest.approx(0.2012)
    assert arms["E20"]["policy"] == pytest.approx(0.1850)
    assert arms["E40"]["policy"] == pytest.approx(0.1800)
    assert arms["E80"]["policy"] == pytest.approx(0.0887)


def test_record_filenames_do_not_collide():
    m = _module()
    names = [m.record_filename(a, s) for a in m.ARMS for s in range(12)]
    assert len(set(names)) == len(names)


def test_per_seed_policy_has_within_arm_variance():
    """Claim 4 is a HARD disqualifier and it correlates S against WITHIN-arm policy.

    If policy were constant across an arm's seeds there would be no variance to correlate
    against, the rule could never fire, and a decorative hard rule is worse than none. This
    asserts the lookup returns 12 genuinely different values per trained arm.
    """
    m = _module()
    for arm in ("E10", "E20", "E40", "E80"):
        pol = m.policy_by_seed(arm)
        assert len(pol) >= 12, f"{arm} returned {len(pol)} seeds, expected at least 12"
        vals = [pol[s] for s in range(12)]
        assert len(set(vals)) > 1, (
            f"{arm} policy is constant across seeds ({vals[0]}); the tag lookup is wrong and "
            "Claim 4 would be uncomputable"
        )


def test_e0_policy_is_all_zero():
    """EXP-036 measured every seed at exactly 0.0000. That is why the spec excludes E0 from
    Claim 4 while keeping it for the Claim 3 floor."""
    pol = _module().policy_by_seed("E0")
    assert set(pol.values()) == {0.0}
