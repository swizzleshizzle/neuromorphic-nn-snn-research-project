import importlib.util
import json
from pathlib import Path

import pytest

from neuromorphic.envs.cube import MOVE_LABELS, N_ACTIONS, apply_move

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "record_cube_trace.py"


def _load():
    spec = importlib.util.spec_from_file_location("record_cube_trace", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[0], lines[1:]


def test_recorded_cube_trace_header_is_cube_shaped(tmp_path):
    mod = _load()
    out = tmp_path / "cube.jsonl"
    mod.record(depth=1, seed=0, episodes=2, out_path=out)
    header, frames = _read(out)
    assert header["task"]["type"] == "cube"
    assert "grid_n" not in header["task"]
    assert len(header["task"]["action_labels"]) == N_ACTIONS
    assert header["task"]["action_labels"] == list(MOVE_LABELS)
    assert header["schema_version"] == "1.1"
    assert len(frames) >= 1
    assert frames[0]["task"]["distance"] is not None
    assert isinstance(frames[0]["task"]["distance"], int)


def test_recorded_frames_carry_facelets_that_follow_the_moves(tmp_path):
    """The strongest available check that frames describe the real episode."""
    mod = _load()
    out = tmp_path / "cube.jsonl"
    mod.record(depth=2, seed=0, episodes=2, out_path=out)
    _, frames = _read(out)
    for f in frames:
        assert len(f["task"]["facelets"]) == 24
        assert "agent" not in f["task"]
        assert "goal" not in f["task"]
    assert len(frames) >= 2, "need at least two frames to compare consecutive moves"
    for a, b in zip(frames, frames[1:]):
        expected = apply_move(tuple(a["task"]["facelets"]), a["task"]["action"])
        assert tuple(b["task"]["facelets"]) == expected, (
            "frame facelets do not follow the recorded move"
        )


def test_recorded_encoding_is_facelet_shaped(tmp_path):
    mod = _load()
    out = tmp_path / "cube.jsonl"
    mod.record(depth=1, seed=0, episodes=2, out_path=out)
    _, frames = _read(out)
    enc = frames[0]["encoding"]["sensory_input"]
    assert enc["cube_n"] == 2
    assert enc["n_colors"] == 6
    assert len(enc["spikes"][0]) == 144


def test_training_is_reproducible_at_a_fixed_seed(tmp_path):
    """Same seed, same trace. Guards the replicated seeding order against drift."""
    mod = _load()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    mod.record(depth=1, seed=3, episodes=2, out_path=a)
    mod.record(depth=1, seed=3, episodes=2, out_path=b)
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
