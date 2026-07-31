"""The adapter refactor must not change one gridworld byte except schema_version.

The fixture stores the pre-change header verbatim plus a sha256 digest per frame
(not the raw frames) to keep the fixture small. Each digest is taken over the
canonical form ``json.dumps(frame, sort_keys=True, separators=(",", ":"))`` so it
is key-order insensitive, matching the semantics of comparing parsed dicts.
"""
import hashlib
import json
from pathlib import Path

import torch

from neuromorphic.brain import Brain
from neuromorphic.envs import GridWorldEnv
from neuromorphic.monitor import FileSink, record_episode

FIXTURE = Path(__file__).parent / "fixtures" / "gridworld_reference_trace.json"


def _digest(frame: dict) -> str:
    return hashlib.sha256(json.dumps(frame, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _record(path) -> list[dict]:
    env = GridWorldEnv()
    brain = Brain(grid_n=env.size, seed=0)
    record_episode(brain, env, FileSink(path), seed=0,
                   generator=torch.Generator().manual_seed(0))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_gridworld_header_unchanged_except_version_and_hash(tmp_path):
    fixture = _load_fixture()
    old_h = fixture["header"]
    new = _record(tmp_path / "t.jsonl")
    assert len(new) == len(fixture["frame_digests"]) + 1, "frame count changed"

    new_h = new[0]
    assert new_h["schema_version"] == "1.1"
    assert old_h["schema_version"] == "1.0"
    # config_hash intentionally changes (n_obs replaces grid_n in the payload).
    assert new_h["task"] == old_h["task"], "gridworld task block must be identical"
    assert new_h["regions"] == old_h["regions"]
    assert new_h["pathways"] == old_h["pathways"]
    assert new_h["policy_regions"] == old_h["policy_regions"]
    for k in ("id", "seed", "T"):
        assert new_h["brain"][k] == old_h["brain"][k]


def test_gridworld_every_frame_is_field_identical(tmp_path):
    fixture = _load_fixture()
    old_digests = fixture["frame_digests"]
    new = _record(tmp_path / "t.jsonl")
    new_frames = new[1:]
    assert len(new_frames) == len(old_digests), "frame count changed"

    for i, (old_digest, frame) in enumerate(zip(old_digests, new_frames)):
        assert _digest(frame) == old_digest, f"frame {i} changed after the adapter refactor"
