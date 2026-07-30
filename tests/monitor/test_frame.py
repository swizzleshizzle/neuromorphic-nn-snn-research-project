import json

import torch

from neuromorphic.brain import Brain
from neuromorphic.monitor.frame import build_frame
from neuromorphic.monitor.tasks import GridworldAdapter


def _recorded_step():
    brain = Brain(grid_n=5, seed=0)
    gen = torch.Generator().manual_seed(0)
    out = brain.step([0, 0, 4, 4], record=True, recall=True, generator=gen)
    return brain, out


def _task():
    return {
        "agent": [0, 0], "goal": [4, 4],
        "action": 1, "action_label": "right",
        "reward": -1.0, "return": -1.0,
        "terminated": False, "truncated": False,
    }


def test_frame_has_all_blocks():
    _, out = _recorded_step()
    frame = build_frame(out, episode=0, step=3, t=3.0, task=_task(), store=False, recall=True)
    assert set(frame) >= {"episode", "step", "t", "task", "regions", "pathways", "router", "field"}
    assert frame["episode"] == 0 and frame["step"] == 3
    assert frame["task"]["action_label"] == "right"


def test_field_widths_match_topology():
    brain, out = _recorded_step()
    frame = build_frame(out, episode=0, step=0, t=0.0, task=_task(), store=False, recall=True)
    assert len(frame["field"]["hippocampus"]["spikes"]) == brain.T
    assert len(frame["field"]["hippocampus"]["spikes"][0]) == 150
    assert len(frame["field"]["sensory"]["spikes"][0]) == brain.content


def test_region_summary_ranges():
    _, out = _recorded_step()
    frame = build_frame(out, episode=0, step=0, t=0.0, task=_task(), store=False, recall=True)
    for r in ("sensory", "hippocampus", "prefrontal", "router", "motor"):
        s = frame["regions"][r]
        assert 0.0 <= s["rate"] <= 1.0
        assert 0.0 <= s["active_frac"] <= 1.0
        assert isinstance(s["spikes"], int)
        assert len(s["rate_t"]) == 32


def test_router_gate_open_is_one_minus_gate_closed():
    _, out = _recorded_step()
    frame = build_frame(out, episode=0, step=0, t=0.0, task=_task(), store=False, recall=True)
    expected = (1 - out["gate_closed"]).float().mean(dim=0)[0].tolist()
    got = frame["router"]["gate_open"]
    assert len(got) == len(expected)
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, expected))
    assert len(frame["router"]["utilities"]) == 4
    assert len(frame["router"]["gate_open_t"]) == 32


def test_pathway_gates_follow_flags():
    _, out = _recorded_step()
    frame = build_frame(out, episode=0, step=0, t=0.0, task=_task(), store=True, recall=False)
    assert frame["pathways"]["sens_hippo"]["gate_open"] == 1.0   # store=True
    assert frame["pathways"]["hippo_pfc"]["gate_open"] == 0.0    # recall=False
    assert "gate_open" not in frame["pathways"]["sens_pfc"]      # ungated edge


def test_step_returns_encoder_input():
    brain, out = _recorded_step()
    assert "obs_spikes" in out
    assert out["obs_spikes"].shape == (brain.T, 1, 2 * brain.grid_n * brain.grid_n)


def test_frame_encoding_block_is_truthful_grid():
    brain, out = _recorded_step()
    frame = build_frame(
        out, episode=0, step=0, t=0.0, task=_task(), store=False, recall=True,
        adapter=GridworldAdapter(brain.grid_n),
    )
    enc = frame["encoding"]["sensory_input"]
    assert enc["grid_n"] == brain.grid_n
    assert enc["planes"] == ["agent", "goal"]
    assert len(enc["spikes"]) == brain.T
    assert len(enc["spikes"][0]) == 2 * brain.grid_n * brain.grid_n


def test_encoding_omitted_without_grid_n():
    _, out = _recorded_step()
    frame = build_frame(out, episode=0, step=0, t=0.0, task=_task(), store=False, recall=True)
    assert "encoding" not in frame


def test_frame_is_json_serializable():
    brain, out = _recorded_step()
    frame = build_frame(
        out, episode=0, step=0, t=0.0, task=_task(), store=False, recall=True,
        adapter=GridworldAdapter(brain.grid_n),
    )
    json.dumps(frame)  # must not raise
