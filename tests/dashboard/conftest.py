"""Fixtures for dashboard tests: headless backend + synthetic traces."""

import json

import matplotlib

matplotlib.use("Agg")  # Headless backend — no GUI windows during pytest.

import pytest

# Tiny stand-ins for the real five-region topology (small N/T for speed).
_REGIONS = [
    {"id": "sensory", "label": "Sensory Cortex", "n_neurons": 6},
    {"id": "hippocampus", "label": "Hippocampus", "n_neurons": 8},
    {"id": "prefrontal", "label": "Prefrontal", "n_neurons": 4},
    {"id": "router", "label": "Thalamic Router", "n_neurons": 4},
    {"id": "motor", "label": "Motor Cortex", "n_neurons": 4},
]
_PATHWAYS = [
    {"id": "sens_hippo", "src": "sensory", "dst": "hippocampus", "gated": True},
    {"id": "sens_pfc", "src": "sensory", "dst": "prefrontal", "gated": False},
    {"id": "hippo_pfc", "src": "hippocampus", "dst": "prefrontal", "gated": True},
    {"id": "pfc_motor", "src": "prefrontal", "dst": "motor", "gated": True},
]
_T = 4


def _spikes(n_neurons, seed):
    """Deterministic [T][N] binary spike matrix (no torch dependency)."""
    out = []
    val = seed
    for _ in range(_T):
        row = []
        for _ in range(n_neurons):
            val = (val * 1103515245 + 12345) & 0x7FFFFFFF
            row.append(1 if (val >> 16) % 3 == 0 else 0)
        out.append(row)
    return out


def make_header():
    return {
        "schema_version": "1.0",
        "brain": {"id": "five-region", "config_hash": "deadbeef", "seed": 0, "T": _T},
        "task": {"type": "gridworld", "grid_n": 5,
                 "action_labels": ["up", "right", "down", "left"]},
        "regions": _REGIONS,
        "pathways": _PATHWAYS,
    }


def make_frames(n_episodes=1, steps_per_ep=5):
    """Build a flat list of frames spanning ``n_episodes`` episodes."""
    frames = []
    for ep in range(n_episodes):
        running = 0.0
        for step in range(steps_per_ep):
            reward = -1.0
            running += reward
            frames.append({
                "episode": ep,
                "step": step,
                "t": float(step),
                "task": {
                    "agent": [step % 5, ep % 5],
                    "goal": [4, 4],
                    "action": step % 4,
                    "action_label": ["up", "right", "down", "left"][step % 4],
                    "reward": reward,
                    "return": running,
                    "terminated": False,
                    "truncated": False,
                },
                "regions": {r["id"]: {"rate": 0.3, "spikes": 5, "active_frac": 0.4,
                                      "rate_t": [0.3] * _T} for r in _REGIONS},
                "pathways": {
                    "sens_hippo": {"intensity": 0.4, "gate_open": 0.0},
                    "sens_pfc": {"intensity": 0.3},
                    "hippo_pfc": {"intensity": 0.2, "gate_open": 0.0},
                    "pfc_motor": {"intensity": 0.1,
                                  "gate_open": [0.0, 0.0, 0.25, 0.0]},
                },
                "router": {"gate_open": [0.0, 0.0, 0.25, 0.0],
                           "gate_open_t": [[0.0, 0.0, 0.25, 0.0]] * _T,
                           "utilities": [0.1, 0.2, 0.3, 0.4]},
                "field": {r["id"]: {"spikes": _spikes(r["n_neurons"], ep * 97 + step * 7 + i)}
                          for i, r in enumerate(_REGIONS)},
            })
    return frames


@pytest.fixture
def header():
    return make_header()


@pytest.fixture
def single_episode_frames():
    return make_frames(n_episodes=1, steps_per_ep=5)


@pytest.fixture
def multi_episode_frames():
    return make_frames(n_episodes=3, steps_per_ep=2)


@pytest.fixture
def trace_file(tmp_path, header, single_episode_frames):
    """A real on-disk JSONL trace (with a trailing blank line)."""
    path = tmp_path / "trace.jsonl"
    lines = [json.dumps(header)] + [json.dumps(f) for f in single_episode_frames]
    path.write_text("\n".join(lines) + "\n\n")  # trailing blank line on purpose
    return path
