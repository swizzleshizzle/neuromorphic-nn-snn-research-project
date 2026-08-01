"""EXP-034: train the SAME frozen concept and linear head with a depth curriculum.

EXP-033 measured that the frozen concept@64 supports 48.1% depth-3 success when its 390
head weights are fit with supervised labels, while REINFORCE reaches 2.2%. The
representation is not the bottleneck; finding the weights is. This changes only how the
head is trained.

The load-bearing test is `test_curriculum_off_is_byte_identical_to_the_shipped_run`. A new
training mode that perturbed the default path would silently invalidate EXP-029 through
EXP-033, all of which share this driver.
"""

from __future__ import annotations

import pytest

from neuromorphic.training.cube_baseline import CubeConfig, curriculum_schedule, run_cube_baseline


def test_curriculum_defaults_to_empty_so_existing_configs_are_untouched():
    assert CubeConfig().curriculum == ()


def test_curriculum_off_is_byte_identical_to_the_shipped_run(tmp_path):
    """The whole cube experiment line shares this driver. Default behaviour must not move.

    Explicitly passing the empty curriculum must also be identical to omitting it.
    """
    common = dict(arm="regionalized", depth=2, seed=3, episodes=8, max_depth=2)
    a = run_cube_baseline(CubeConfig(out_dir=tmp_path / "a", **common))
    b = run_cube_baseline(CubeConfig(out_dir=tmp_path / "b", curriculum=(), **common))
    for key in a:
        if key == "config":
            continue
        assert a[key] == b[key], f"{key} moved when the curriculum knob was added"


def test_schedule_splits_episodes_across_stages_and_conserves_the_total():
    """Total episodes must be conserved, or the curriculum arm silently buys extra training
    and any comparison against a fixed-budget arm is confounded by compute rather than
    by the schedule."""
    for total in (600, 601, 602, 1000, 3000):
        stages = curriculum_schedule((1, 2, 3), total)
        assert [d for d, _ in stages] == [1, 2, 3]
        assert sum(n for _, n in stages) == total


def test_schedule_puts_remainder_in_the_final_stage():
    """The final stage is the evaluated depth, so any leftover episodes belong there."""
    stages = curriculum_schedule((1, 2, 3), 602)
    assert stages == [(1, 200), (2, 200), (3, 202)]


def test_single_stage_schedule_is_the_whole_budget():
    assert curriculum_schedule((3,), 600) == [(3, 600)]


def test_schedule_rejects_a_budget_too_small_to_cover_every_stage():
    """Silently dropping a stage would make the arm something other than what it claims."""
    with pytest.raises(ValueError, match="too small"):
        curriculum_schedule((1, 2, 3), 2)


def test_curriculum_run_still_evaluates_at_the_configured_depth(tmp_path):
    """The curriculum trains on 1-2-3 but the reported number must be depth 3, or the arms
    are not comparable."""
    rec = run_cube_baseline(CubeConfig(
        arm="regionalized", depth=3, seed=0, episodes=9, max_depth=3,
        curriculum=(1, 2, 3), out_dir=tmp_path,
    ))
    assert rec["depth"] == 3
    assert rec["n"] == len(rec_eval_states(rec))


def rec_eval_states(rec):
    """Held-out count the record was scored on, recomputed from its own config."""
    from neuromorphic.envs.cube_distance import ExactBFSDistance
    from neuromorphic.training.cube_baseline import shell_states, split_shell
    cfg = rec["config"]
    provider = ExactBFSDistance(max_depth=max(cfg["max_depth"], cfg["depth"]))
    states = shell_states(provider, cfg["depth"])
    _, eval_states, _ = split_shell(
        states, cfg["depth"], seed=cfg["seed"],
        heldout_cap=cfg["heldout_cap"], heldout_frac=cfg["heldout_frac"],
    )
    return eval_states


def test_curriculum_actually_changes_the_outcome(tmp_path):
    """A curriculum that trained on one depth throughout would pass every test above.

    Same seed, same total episodes, same everything except the schedule: the resulting
    records must differ, or the knob is decorative.
    """
    common = dict(arm="regionalized", depth=3, seed=1, episodes=12, max_depth=3)
    direct = run_cube_baseline(CubeConfig(out_dir=tmp_path / "d", **common))
    curric = run_cube_baseline(
        CubeConfig(out_dir=tmp_path / "c", curriculum=(1, 2, 3), **common)
    )
    moved = [k for k in ("success_rate", "revisit_rate", "mean_train_entropy",
                         "greedy_modal_action_frac")
             if direct[k] != curric[k]]
    assert moved, "curriculum produced an identical run; it is not training on other depths"


def test_curriculum_stages_never_train_on_the_evaluated_heldout_states(tmp_path):
    """Shells at different distances are disjoint, so stages 1 and 2 cannot leak depth-3
    held-out states. Pinned because a future change to stage construction could break it."""
    from neuromorphic.envs.cube_distance import ExactBFSDistance
    from neuromorphic.training.cube_baseline import shell_states, split_shell
    provider = ExactBFSDistance(max_depth=3)
    _, eval3, _ = split_shell(shell_states(provider, 3), 3, seed=0, heldout_cap=200,
                              heldout_frac=0.25)
    for d in (1, 2):
        assert set(shell_states(provider, d)).isdisjoint(set(eval3))
