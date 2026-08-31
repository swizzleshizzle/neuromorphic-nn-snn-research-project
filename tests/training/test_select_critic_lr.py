"""EXP-053: the critic-lr selector must be BLIND to success rate.

`encoder_lr` had no prior either, and EXP-047 settled it with `select_lr.py`, a script that
read only the probe output and could not see a success rate. Same discipline: if the selector
can be swayed by the outcome it is meant to be chosen independently of, the pilot is not a
pilot, it is a tiny uncontrolled experiment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC_PATH = (Path(__file__).resolve().parents[2]
             / "experiments" / "053_neuromod_stage3" / "select_critic_lr.py")


def _module():
    spec = importlib.util.spec_from_file_location("select_critic_lr", SPEC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_picks_the_best_explained_variance_not_the_best_success():
    """The decisive test. The highest-EV lr here has the WORST success rate, so a selector
    that peeks at the outcome picks 1e-1 and fails."""
    records = [
        {"config": {"critic_lr": 1e-3}, "seed": 12, "critic_ev": 0.10, "critic_n": 5000,
         "success_rate": 0.90},
        {"config": {"critic_lr": 1e-3}, "seed": 13, "critic_ev": 0.12, "critic_n": 5000,
         "success_rate": 0.90},
        {"config": {"critic_lr": 1e-2}, "seed": 12, "critic_ev": 0.55, "critic_n": 5000,
         "success_rate": 0.05},
        {"config": {"critic_lr": 1e-2}, "seed": 13, "critic_ev": 0.57, "critic_n": 5000,
         "success_rate": 0.05},
        {"config": {"critic_lr": 1e-1}, "seed": 12, "critic_ev": 0.20, "critic_n": 5000,
         "success_rate": 0.99},
        {"config": {"critic_lr": 1e-1}, "seed": 13, "critic_ev": 0.22, "critic_n": 5000,
         "success_rate": 0.99},
    ]
    assert _module().select(records) == pytest.approx(1e-2)


def test_ties_break_toward_the_smaller_lr():
    records = [
        {"config": {"critic_lr": 1e-3}, "seed": 12, "critic_ev": 0.40, "critic_n": 5000},
        {"config": {"critic_lr": 1e-2}, "seed": 12, "critic_ev": 0.40, "critic_n": 5000},
    ]
    assert _module().select(records) == pytest.approx(1e-3)


def test_refuses_incomplete_cells():
    """A grid point missing a seed would be selected on a different sample than its rivals."""
    records = [
        {"config": {"critic_lr": 1e-3}, "seed": 12, "critic_ev": 0.40, "critic_n": 5000},
        {"config": {"critic_lr": 1e-3}, "seed": 13, "critic_ev": 0.40, "critic_n": 5000},
        {"config": {"critic_lr": 1e-2}, "seed": 12, "critic_ev": 0.90, "critic_n": 5000},
    ]
    with pytest.raises(ValueError, match="incomplete"):
        _module().select(records)


def test_refuses_a_cell_with_no_critic_data():
    """A degenerate stage yields critic_ev 0.0 with critic_n 0, indistinguishable from a
    critic that genuinely explains nothing. Selecting on it would silently halve that
    lr's mean. Refuse instead - a refusal cannot favour any learning rate."""
    records = [
        {"config": {"critic_lr": 1e-3}, "seed": 12, "critic_ev": 0.40, "critic_n": 5000},
        {"config": {"critic_lr": 1e-3}, "seed": 13, "critic_ev": 0.40, "critic_n": 5000},
        {"config": {"critic_lr": 1e-2}, "seed": 12, "critic_ev": 0.90, "critic_n": 5000},
        {"config": {"critic_lr": 1e-2}, "seed": 13, "critic_ev": 0.00, "critic_n": 0},
    ]
    with pytest.raises(ValueError, match="critic_n"):
        _module().select(records)


def test_a_real_zero_is_still_selectable():
    """A genuine 0.0 with data behind it is a valid measurement, not an error. If this
    test fails, the guard is rejecting real data and has become a re-ranking."""
    records = [
        {"config": {"critic_lr": 1e-3}, "seed": 12, "critic_ev": 0.00, "critic_n": 5000},
        {"config": {"critic_lr": 1e-2}, "seed": 12, "critic_ev": 0.50, "critic_n": 5000},
    ]
    assert _module().select(records) == pytest.approx(1e-2)
