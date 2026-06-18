import json

import torch

from neuromorphic.brain import Brain
from neuromorphic.envs import GridWorldEnv
from neuromorphic.monitor.runner import record_episode
from neuromorphic.monitor.sink import FileSink


def test_record_episode_writes_a_replayable_trace(tmp_path):
    env = GridWorldEnv()
    brain = Brain(grid_n=env.size, seed=0)
    sink = FileSink(tmp_path / "ep.jsonl")
    summary = record_episode(
        brain, env, sink, seed=0, max_steps=8,
        generator=torch.Generator().manual_seed(0),
    )

    assert summary["steps"] >= 1
    lines = (tmp_path / "ep.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == summary["steps"] + 1  # header + one frame per step

    header = json.loads(lines[0])
    assert header["schema_version"] == "1.0"
    assert [r["id"] for r in header["regions"]][0] == "sensory"

    frame = json.loads(lines[1])
    assert set(frame) >= {"task", "regions", "pathways", "router", "field", "encoding"}
    assert frame["task"]["action_label"] in ("up", "right", "down", "left")
    assert len(frame["field"]["hippocampus"]["spikes"][0]) == 150
    assert frame["encoding"]["sensory_input"]["grid_n"] == env.size
