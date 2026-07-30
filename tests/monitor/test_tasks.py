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
    info = {"solved": False, "scramble_depth": 2, "distance": 2,
            "move": 3, "move_label": "R'"}
    t = a.frame_task(SOLVED, action=3, reward=-1.0, total=-3.0,
                     terminated=False, truncated=False, info=info)
    assert len(t["facelets"]) == 24
    assert t["facelets"] == list(SOLVED)
    # The defect being fixed: facelet colors rendered as x/y coordinates.
    assert "agent" not in t
    assert "goal" not in t
    assert t["action_label"] == "R'"
    assert t["distance"] == 2


def test_cube_frame_task_distance_stays_none_without_provider():
    t = CubeAdapter().frame_task(
        SOLVED, action=0, reward=-1.0, total=-1.0, terminated=False,
        truncated=False,
        info={"solved": False, "scramble_depth": 1, "distance": None,
              "move": 0, "move_label": "U"},
    )
    assert t["distance"] is None


def test_cube_facelets_follow_the_applied_move():
    """Consecutive frame tasks must differ by exactly the move permutation."""
    a = CubeAdapter()
    before = SOLVED
    action = 3  # R'
    after = apply_move(before, action)
    info = {"solved": False, "scramble_depth": 1, "distance": 1,
            "move": action, "move_label": MOVE_LABELS[action]}
    t0 = a.frame_task(before, action=action, reward=-1.0, total=-1.0,
                      terminated=False, truncated=False, info=info)
    t1 = a.frame_task(after, action=0, reward=-1.0, total=-2.0,
                      terminated=False, truncated=False, info=info)
    assert tuple(t1["facelets"]) == apply_move(tuple(t0["facelets"]), action)
    assert tuple(t1["facelets"]) != tuple(t0["facelets"])


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
