"""EXP-078: driver, aggregator and launcher. Each docstring names the bug it catches."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "experiments" / "078_full_five_move_tree"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("exp078_evaluate", HERE / "evaluate.py")


def test_arms_and_constants_are_the_spec():
    """Catches a drifted arm table, interval or forced-stop point (spec sections 3, 4, 7)."""
    assert ev.ARMS == {"J5V-X": ("C", 5, 5, None), "J4V-X": ("C", 4, 4, None)}
    assert ev.DEPTHS == (9, 11) and ev.CKPT_EVERY == 10 and ev.FORCED_STOP == 50
    assert ev.RANK_K == 5


def test_j4v_x_with_checkpointing_equals_exp076_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH. Catches checkpointing changing an outcome: J4V-X through this
    driver (checkpoint on) equals EXP-076's driver on the first 3 depth-9 states of seed 0."""
    e76 = _load("exp076_evaluate_t", REPO / "experiments" / "076_deeper_tree" / "evaluate.py")
    mine = ev.run_cell("J4V-X", 9, 0, tmp_path / "a", limit_states=3)
    ref = e76.run_cell("J4V-X", 9, 0, tmp_path / "b", limit_states=3)
    for f in ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
              "goal_fired_frac", "first_fire_len", "fallback_frac"):
        assert mine[f] == ref[f], f
    assert mine["resumes"] == 0


def test_stopped_cell_writes_no_record_and_resumes_to_the_same_record(tmp_path):
    """Catches a stopped cell leaving a record a launcher would skip as done, a deleted
    checkpoint before the record exists, and a resume that differs from a straight run."""
    straight = ev.run_cell("J4V-X", 9, 0, tmp_path / "s", limit_states=3)
    out = tmp_path / "r"
    assert ev.run_cell("J4V-X", 9, 0, out, limit_states=3, stop_after=2) is None
    assert not (out / ev.record_name("J4V-X", 9, 0)).exists()
    assert ev.ckpt_path(out, "J4V-X", 9, 0).exists()
    resumed = ev.run_cell("J4V-X", 9, 0, out, limit_states=3)
    assert (out / ev.record_name("J4V-X", 9, 0)).exists()
    assert not ev.ckpt_path(out, "J4V-X", 9, 0).exists()
    for f in ("solved", "success_rate", "mean_steps", "first_fire_len", "judge_evals"):
        assert resumed[f] == straight[f], f
    assert resumed["resumes"] == 1


def test_j5v_x_records_its_depths_and_fired5(tmp_path):
    """Catches a J5V-X record that does not say what ran (k 5, goal check 5, 7776 leaves)."""
    r = ev.run_cell("J5V-X", 9, 0, tmp_path, limit_states=1)
    assert (r["mode"], r["k"], r["k_goal"]) == ("C", 5, 5)
    assert r["judge_evals"] == 7776 and r["fired5"] == r["first_fire_len"].get("5", 0)


def test_unknown_arm_exits_cleanly(tmp_path):
    """Catches an arm typo running something unregistered."""
    with pytest.raises(SystemExit):
        ev.run_cell("J6V-X", 9, 0, tmp_path, limit_states=1)


def test_smoke_record_is_never_overwritten_or_treated_as_done(tmp_path):
    """Catches a --limit-states smoke record sitting under the real cell's name: run_cell must
    refuse a call with a different limit_states instead of overwriting or skipping it."""
    (tmp_path / ev.record_name("J4V-X", 9, 0)).write_text('{"limit_states": 3}')
    with pytest.raises(SystemExit):
        ev.run_cell("J4V-X", 9, 0, tmp_path, limit_states=None)


def test_stale_checkpoint_of_a_different_size_is_refused(tmp_path):
    """Catches a smoke checkpoint being resumed by a full run (or silently discarded)."""
    assert ev.run_cell("J4V-X", 9, 0, tmp_path, limit_states=2, stop_after=1) is None
    with pytest.raises(ValueError, match="n_states"):
        ev.run_cell("J4V-X", 9, 0, tmp_path, limit_states=3)
    assert ev.ckpt_path(tmp_path, "J4V-X", 9, 0).exists()
