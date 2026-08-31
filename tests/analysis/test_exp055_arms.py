"""EXP-055: each arm must be EXP-052's Phase 2 with exactly one field changed.

EXP-052 copied EXP-043's depth-6 cell field for field and changed the encoder. EXP-055 does the
same. An arm carrying a second difference produces a paired delta that measures something
nobody chose, and a nine-point epoch curve would be silently wrong in one place.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUN_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "055_pretraining_left_edge" / "run.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp055_run", RUN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_arms_match_exp052_phase_two_field_for_field(tmp_path):
    """Every field except the encoder path and the tag must equal EXP-052's."""
    m = _module()
    for c in m.sweep_configs((0, 1), tmp_path, (1,)):
        assert c.depth == 6
        assert c.episodes == 10_000
        assert c.curriculum == (1, 2, 3, 4, 5, 6)
        assert c.max_steps_by_depth == ((1, 2),)
        assert c.entropy_beta == 0.0
        assert c.normalize_advantages is False
        assert c.max_depth == 6
        assert c.arm == "regionalized"
        assert c.readout == "concept"


def test_every_arm_is_frozen(tmp_path):
    """390 trainable, not 27,206. A fine-tuned arm is a different architecture and must never
    be tabulated with these."""
    m = _module()
    for c in m.sweep_configs((0,), tmp_path, m.EPOCH_ARMS):
        assert c.encoder_lr is None, "an arm is fine-tuning; every EXP-055 arm is frozen"
        assert c.plasticity_gate is None
        assert c.critic_lr is None


def test_each_arm_loads_its_own_epoch_encoder(tmp_path):
    m = _module()
    for e in m.EPOCH_ARMS:
        c = m.sweep_configs((3,), tmp_path, (e,))[0]
        assert f"exp055_encoder_e{e}_s3.pt" in str(c.encoder_state_path)


def test_tags_are_distinct_per_epoch_arm():
    """`record_filename` does not encode the encoder, so without a per-epoch tag the four arms
    would silently overwrite each other into one set of files."""
    m = _module()
    tags = [m.tag_for(e) for e in m.EPOCH_ARMS]
    assert len(set(tags)) == len(tags)
    assert tags[0] == "exp055_e1_d6"


def test_record_filenames_do_not_collide(tmp_path):
    from neuromorphic.training.cube_baseline import record_filename
    m = _module()
    names = [record_filename(c)
             for c in m.sweep_configs(range(12), tmp_path, m.EPOCH_ARMS)]
    assert len(set(names)) == len(names)


def test_anchors_point_at_the_experiments_that_measured_them():
    """0, 10, 20, 40 and 80 all exist and are NOT re-run. A wrong anchor would corrupt every
    contrast that uses it."""
    m = _module()
    assert set(m.ANCHORS) == {10, 20, 40, 80}
    assert m.ANCHORS[10][1] == "exp052_e10_d6"
    assert m.ANCHORS[20][1] == "exp052_e20_d6"
    assert m.ANCHORS[40][1] == "exp043_capped_d6"
    assert m.ANCHORS[80][1] == "exp050_pre2_d6"
    assert m.ANCHORS[10][2] == pytest.approx(0.2012)
    assert m.ANCHORS[40][2] == pytest.approx(0.1800)
    assert m.ANCHORS[80][2] == pytest.approx(0.0887)
