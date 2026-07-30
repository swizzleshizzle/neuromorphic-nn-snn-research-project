import pytest
import torch

from neuromorphic.envs.cube import MOVE_LABELS, N_ACTIONS, SOLVED, apply_move
from neuromorphic.monitor.tasks import CubeAdapter, GridworldAdapter


def test_gridworld_header_keeps_grid_n():
    a = GridworldAdapter(grid_n=5)
    h = a.header_task()
    assert h["type"] == "gridworld"
    assert h["grid_n"] == 5
    assert h["action_labels"] == ["up", "right", "down", "left"]


def test_cube_header_omits_grid_n_and_declares_cube_n():
    h = CubeAdapter().header_task()
    assert h["type"] == "cube"
    assert "grid_n" not in h
    assert h["cube_n"] == 2


def test_cube_action_labels_match_move_count():
    # The 4-wide gridworld default IndexErrors on cube actions 4 and 5.
    a = CubeAdapter()
    assert len(a.action_labels) == N_ACTIONS
    assert list(a.action_labels) == list(MOVE_LABELS)
    for action in range(N_ACTIONS):
        assert a.action_labels[action] == MOVE_LABELS[action]


def test_cube_frame_task_has_facelets_and_no_coordinates():
    a = CubeAdapter()
    action = 3  # R'
    after = apply_move(SOLVED, action)
    info = {"solved": False, "scramble_depth": 2, "distance": 2,
            "move": action, "move_label": "R'"}
    t = a.frame_task(SOLVED, next_obs=after, action=action, reward=-1.0, total=-3.0,
                     terminated=False, truncated=False, info=info)
    assert len(t["facelets"]) == 24
    assert t["facelets"] == list(after)
    # The defect being fixed: facelet colors rendered as x/y coordinates.
    assert "agent" not in t
    assert "goal" not in t
    assert t["action_label"] == "R'"
    assert t["distance"] == 2


def test_cube_frame_task_distance_stays_none_without_provider():
    t = CubeAdapter().frame_task(
        SOLVED, next_obs=SOLVED, action=0, reward=-1.0, total=-1.0, terminated=False,
        truncated=False,
        info={"solved": False, "scramble_depth": 1, "distance": None,
              "move": 0, "move_label": "U"},
    )
    assert t["distance"] is None


def test_cube_frame_task_reports_next_obs_as_facelets_not_obs():
    """Pins the post-move contract: a cube frame describes the state AFTER its move.

    Before the fix, ``frame_task`` read ``obs`` (pre-move), which paired
    pre-move facelets with post-move ``solved``/``distance`` from ``info``.
    This must fail if the adapter ever reverts to reading ``obs``.
    """
    a = CubeAdapter()
    action = 3  # R'
    after = apply_move(SOLVED, action)
    assert after != SOLVED, "fixture move must actually change the state"
    info = {"solved": False, "scramble_depth": 1, "distance": 1,
            "move": action, "move_label": MOVE_LABELS[action]}
    t = a.frame_task(SOLVED, next_obs=after, action=action, reward=-1.0, total=-1.0,
                     terminated=False, truncated=False, info=info)
    assert tuple(t["facelets"]) == after
    assert tuple(t["facelets"]) != SOLVED


def test_gridworld_frame_task_ignores_next_obs_and_reads_obs():
    """Guards gridworld against ever silently switching to post-move semantics."""
    a = GridworldAdapter(grid_n=5)
    obs = [1, 2, 3, 4]
    next_obs = [9, 9, 9, 9]
    t = a.frame_task(obs, next_obs=next_obs, action=0, reward=-1.0, total=-1.0,
                     terminated=False, truncated=False, info={})
    assert t["agent"] == [1, 2]
    assert t["goal"] == [3, 4]


def test_cube_encoding_block_is_facelet_shaped():
    out = {"obs_spikes": torch.zeros(8, 1, 144)}
    enc = CubeAdapter().encoding(out)["sensory_input"]
    assert enc["cube_n"] == 2
    assert enc["n_colors"] == 6
    assert "grid_n" not in enc
    assert len(enc["spikes"]) == 8
    assert len(enc["spikes"][0]) == 144


def test_encoding_is_none_without_obs_spikes():
    assert CubeAdapter().encoding({}) is None
    assert GridworldAdapter(grid_n=5).encoding({}) is None
