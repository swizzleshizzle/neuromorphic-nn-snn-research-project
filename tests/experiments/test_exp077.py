"""EXP-077: driver, aggregator and launcher. Each docstring names the bug it catches."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "experiments" / "077_beam_search"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("exp077_evaluate", HERE / "evaluate.py")


def test_arms_are_the_spec_table():
    """Catches an arm whose mode, depth, goal check or width drifted from spec section 3."""
    assert ev.ARMS == {"J4V-X": ("C", 4, 4, None), "J4VG5-X": ("C", 4, 5, None),
                       "B5-X": ("B", 5, 5, 64)}
    assert ev.DEPTHS == (9, 11)


def test_j4v_x_here_equals_exp076s_record_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH. Catches mode B's seam changing mode C: J4V-X here, on the first
    3 depth-9 states of seed 0, equals EXP-076's driver on the same slice in every outcome
    field and in fallback_frac."""
    e76 = _load("exp076_evaluate_t", REPO / "experiments" / "076_deeper_tree" / "evaluate.py")
    mine = ev.run_cell("J4V-X", 9, 0, tmp_path / "a", limit_states=3)
    ref = e76.run_cell("J4V-X", 9, 0, tmp_path / "b", limit_states=3)
    for f in ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
              "goal_fired_frac", "first_fire_len", "fallback_frac"):
        assert mine[f] == ref[f], f


def test_new_arms_record_mode_depths_width_and_fired5(tmp_path):
    """Catches a record that does not say what ran: B5-X carries mode B, k 5, k_goal 5,
    beam_width 64 and fired5; J4VG5-X carries mode C, k 4, k_goal 5."""
    b = ev.run_cell("B5-X", 9, 0, tmp_path, limit_states=2)
    assert (b["mode"], b["k"], b["k_goal"], b["beam_width"]) == ("B", 5, 5, 64)
    assert b["fired5"] == b["first_fire_len"].get("5", 0)
    assert 0 < b["judge_evals"] <= 1026
    g = ev.run_cell("J4VG5-X", 9, 0, tmp_path, limit_states=2)
    assert (g["mode"], g["k"], g["k_goal"]) == ("C", 4, 5) and "beam_width" not in g
    assert g["judge_evals"] == 1296
    assert json.loads((tmp_path / ev.record_name("B5-X", 9, 0)).read_text())["arm"] == "B5-X"


def test_unknown_arm_exits_cleanly(tmp_path):
    """Catches an arm typo running something unregistered."""
    with pytest.raises(SystemExit):
        ev.run_cell("B6-X", 9, 0, tmp_path, limit_states=1)
