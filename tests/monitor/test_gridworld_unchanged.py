"""The adapter refactor must not change one gridworld byte except schema_version."""
import json
from pathlib import Path

import torch

from neuromorphic.brain import Brain
from neuromorphic.envs import GridWorldEnv
from neuromorphic.monitor import FileSink, record_episode

FIXTURE = Path(__file__).parent / "fixtures" / "gridworld_reference_trace.jsonl"


def _record(path) -> list[dict]:
    env = GridWorldEnv()
    brain = Brain(grid_n=env.size, seed=0)
    record_episode(brain, env, FileSink(path), seed=0,
                   generator=torch.Generator().manual_seed(0))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_gridworld_header_unchanged_except_version_and_hash(tmp_path):
    old = [json.loads(l) for l in FIXTURE.read_text(encoding="utf-8").splitlines() if l.strip()]
    new = _record(tmp_path / "t.jsonl")
    assert len(new) == len(old), "frame count changed"

    old_h, new_h = old[0], new[0]
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
    old = [json.loads(l) for l in FIXTURE.read_text(encoding="utf-8").splitlines() if l.strip()]
    new = _record(tmp_path / "t.jsonl")
    for i, (o, n) in enumerate(zip(old[1:], new[1:])):
        assert n == o, f"frame {i} changed after the adapter refactor"
