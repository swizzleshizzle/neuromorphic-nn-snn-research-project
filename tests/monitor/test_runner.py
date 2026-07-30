import json

import torch

from neuromorphic.brain import Brain
from neuromorphic.envs import GridWorldEnv
from neuromorphic.monitor.runner import record_episode
from neuromorphic.monitor.sink import FileSink


def test_record_policy_episode_uses_head_and_pads_bypassed(tmp_path):
    import torch
    from neuromorphic.brain import Brain
    from neuromorphic.envs import GridWorldEnv
    from neuromorphic.monitor import FileSink, record_policy_episode
    from neuromorphic.monitor.frame import build_frame  # noqa: F401  (sanity import)
    from neuromorphic.training.reinforce import make_policy_head

    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    # Force the head to always choose action 1 (right), regardless of input.
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([0.0, 10.0, 0.0, 0.0]))

    env = GridWorldEnv(max_steps=6)
    sink = FileSink(tmp_path / "trace.jsonl")
    gen = torch.Generator().manual_seed(0)
    summary = record_policy_episode(brain, head, env, sink, seed=0, recall=False,
                                    policy_regions=("sensory",), generator=gen)

    assert summary["steps"] >= 1
    lines = [l for l in (tmp_path / "trace.jsonl").read_text().splitlines() if l.strip()]
    import json
    header = json.loads(lines[0]); frames = [json.loads(l) for l in lines[1:]]
    assert header["policy_regions"] == ["sensory"]
    # every action is the head's forced choice, not the brain's internal readout
    assert all(f["task"]["action"] == 1 for f in frames)
    # bypassed hippocampus recording is padded to a silent field of correct width
    n_hippo = brain.hippo.n_neurons
    spikes = frames[0]["field"]["hippocampus"]["spikes"]
    assert len(spikes[0]) == n_hippo
    assert all(v == 0 for row in spikes for v in row)


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
    assert header["schema_version"] == "1.1"
    assert [r["id"] for r in header["regions"]][0] == "sensory"

    frame = json.loads(lines[1])
    assert set(frame) >= {"task", "regions", "pathways", "router", "field", "encoding"}
    assert frame["task"]["action_label"] in ("up", "right", "down", "left")
    assert len(frame["field"]["hippocampus"]["spikes"][0]) == 150
    assert frame["encoding"]["sensory_input"]["grid_n"] == env.size
