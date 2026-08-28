"""EXP-053: the three arms must differ from their controls in exactly ONE field each.

Every prior cube experiment copied its control's config field for field and changed one
thing. A config-level test is cheap and catches the failure that a 6-hour run would
otherwise reveal at the end: an arm that silently carries two differences, whose paired
delta then measures something nobody chose.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUN_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "053_neuromod_stage3" / "run.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp053_run", RUN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_arm_b_is_exp051_plus_a_critic(tmp_path):
    m = _module()
    cfgs = m.sweep_configs("B", (0, 1), tmp_path)
    assert len(cfgs) == 2
    for c in cfgs:
        assert c.depth == 7 and c.episodes == 10_000
        assert c.curriculum == (1, 2, 3, 4, 5, 6, 7)
        assert c.encoder_lr is None, "arm B is a FROZEN-encoder arm, as EXP-051 was"
        assert c.plasticity_gate is None
        # NOT `== m.SELECTED_CRITIC_LR`: comparing the module against its own constant is
        # an assertion that cannot fail. Pin it to the pilot grid instead, so setting it to
        # None, to 0.0, or to a value the pilot never measured all fail here.
        assert c.critic_lr in (1e-3, 1e-2, 1e-1), (
            f"arm B's critic_lr is {c.critic_lr!r}, which is not a value the pilot measured"
        )
        assert "047_encoder_finetuning" in str(c.encoder_state_path)


def test_arm_g_is_exp047_plus_a_gate(tmp_path):
    m = _module()
    for c in m.sweep_configs("G", (0, 1), tmp_path):
        assert c.depth == 6 and c.episodes == 10_000
        assert c.curriculum == (1, 2, 3, 4, 5, 6)
        assert c.encoder_lr == 1e-4, "arm G's control is EXP-047's fine-tuning arm"
        assert c.plasticity_gate == "dopamine"
        assert c.critic_lr is None, "the gate arm carries no critic; that is arm B"


def test_arm_r_matches_arm_g_except_for_the_gate(tmp_path):
    m = _module()
    g = {c.seed: c for c in m.sweep_configs("G", (0, 1), tmp_path)}
    r = {c.seed: c for c in m.sweep_configs("R", (0, 1), tmp_path,
                                            rates={0: 0.5, 1: 0.4})}
    for seed in (0, 1):
        gc, rc = g[seed], r[seed]
        differing = {f for f in gc.__dataclass_fields__
                     if getattr(gc, f) != getattr(rc, f)}
        assert differing == {"plasticity_gate", "gate_rate_by_seed", "tag"}, (
            f"arm R differs from arm G in {differing}, not just the gate. The attribution "
            "claim would then compare two things at once."
        )


def test_arm_r_refuses_to_build_without_rates(tmp_path):
    with pytest.raises(SystemExit, match="arm G"):
        _module().sweep_configs("R", (0, 1), tmp_path)


def test_record_filenames_do_not_collide(tmp_path):
    from neuromorphic.training.cube_baseline import record_filename
    m = _module()
    names = []
    for arm, kw in (("B", {}), ("G", {}), ("R", {"rates": {s: 0.5 for s in range(12)}})):
        names += [record_filename(c) for c in m.sweep_configs(arm, range(12), tmp_path, **kw)]
    assert len(set(names)) == len(names), "two cells would overwrite each other silently"
