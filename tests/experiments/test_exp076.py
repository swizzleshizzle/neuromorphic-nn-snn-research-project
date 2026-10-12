"""EXP-076: a deeper look-ahead tree over EXP-075's committed X judges."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from neuromorphic.envs.cube_distance import ExactBFSDistance

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "076_deeper_tree"
E75 = REPO / "experiments" / "075_wide_region"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate = _load("exp076_evaluate", EXP / "evaluate.py")


def test_arms_are_the_spec_table():
    """Catches an arm wired to the wrong tree or goal depth (spec section 3), or J3V-X given an
    explicit goal depth instead of the default that Gate 0(b) checks."""
    assert evaluate.ARMS == {"J3V-X": (3, None), "J3VG4-X": (3, 4), "J4V-X": (4, 4)}
    assert evaluate.DEPTHS == (9, 11) and evaluate.RANK_K == 4


def test_j3v_x_here_equals_exp075s_harness_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH. Catches this harness drifting from EXP-075's: J3V-X through it
    equals EXP-075's own run_cell on the same states, committed judge, in every outcome field."""
    mine = evaluate.run_cell("J3V-X", 9, 0, tmp_path / "a", limit_states=2)
    ref = evaluate.e75.run_cell("J3V-X", 9, 0, tmp_path / "b", limit_states=2)
    for f in ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
              "greedy_modal_action_frac", "goal_fired_frac", "fallback_frac"):
        assert mine[f] == ref[f], f
    assert mine["k"] == 3 and mine["k_goal"] == 3


def test_new_arms_write_their_tree_and_goal_depths_and_fired4(tmp_path):
    """Catches a record that does not say which tree it ran (so the aggregator could not refuse
    a mislabelled one), and fired4 not taken from the first-fire histogram."""
    for arm, (k, kg) in (("J3VG4-X", (3, 4)), ("J4V-X", (4, 4))):
        rec = evaluate.run_cell(arm, 7, 0, tmp_path, limit_states=1)
        assert rec["arm"] == arm and rec["k"] == k and rec["k_goal"] == kg
        assert rec["mode"] == "C" and rec["no_revisit"] is True
        assert rec["fired4"] == rec["first_fire_len"].get("4", 0)
        assert rec["limit_states"] == 1 and rec["judge_updates"] == 4000
        on_disk = json.loads((tmp_path / f"exp076_{arm}_d7_s0.json").read_text())
        assert on_disk["k_goal"] == kg


def test_goal_check_arm_never_solves_less_than_j3v_on_a_slice(tmp_path):
    """THE GATE D INVARIANT ON THE REAL JUDGE. Catches the goal-check arm using a different
    stream from J3V-X: on the same states it must solve at least as many."""
    a = evaluate.run_cell("J3V-X", 9, 0, tmp_path, limit_states=6)
    b = evaluate.run_cell("J3VG4-X", 9, 0, tmp_path, limit_states=6)
    assert b["solved"] >= a["solved"]


def test_unknown_arm_and_missing_judge_exit_cleanly(tmp_path):
    """Catches a typo'd arm running some default, and a missing judge evaluating an untrained
    one."""
    with pytest.raises(SystemExit, match="unknown arm"):
        evaluate.run_cell("J5V-X", 9, 0, tmp_path, limit_states=1)
    with pytest.raises(SystemExit, match="judge.pt"):
        evaluate.run_cell("J4V-X", 9, 0, tmp_path, limit_states=1, judge_dir=tmp_path / "no")


def test_rank_cell_ranks_four_move_leaves(tmp_path, monkeypatch):
    """Catches Gate R measured on EXP-075's 3-move leaves instead of 4, and a rank record
    without its limit (so a smoke row could enter Gate R)."""
    seen = []
    real = evaluate.tree_levels

    def spy(state, k, n):
        seen.append(k)
        return real(state, k, n)

    monkeypatch.setattr(evaluate, "tree_levels", spy)
    rec = evaluate.rank_cell(7, 0, tmp_path, limit_states=2,
                             provider=ExactBFSDistance(max_depth=11))
    assert set(seen) == {4}
    assert rec["kind"] == "J-X4" and rec["k"] == 4 and rec["n"] == 2
    assert rec["limit_states"] == 2 and rec["judge_updates"] == 4000
    assert 0.0 < rec["chance"] < 1.0 and rec["hit"] in (0.0, 0.5, 1.0)
    assert (tmp_path / "exp076_rank_J-X4_d7_s0.json").exists()
