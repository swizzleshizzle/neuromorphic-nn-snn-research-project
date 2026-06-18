# Dashboard Data Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit a versioned, transport-agnostic JSONL trace (header + per-step Frames) from a running `Brain` episode, via a `TraceSink` abstraction, so any dashboard can replay a real five-region episode.

**Architecture:** A new `neuromorphic.monitor` package turns one `Brain.step(record=True)` into one JSON-serializable `Frame` and one run into a `Header`. A `TraceSink` interface (with `FileSink` now) writes header + frames as JSONL — the same object that a `WebSocketSink`/`RedisStreamSink` will later push over the wire. A `record_episode` runner drives env+brain and feeds the sink.

**Tech Stack:** Python 3.10+, PyTorch (already used), stdlib `json`/`hashlib`/`pathlib`, pytest. No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-06-17-stage2-dashboard-design.md`

---

## Scope notes (read before starting)

- **In scope:** schema constants, header builder, frame builder (task / regions / pathways / router / `field`), `TraceSink` + `FileSink`, `record_episode` runner, and one experiment script that produces a real trace file for the design team to load.
- **Out of scope (deferred, documented):** the `detail.membrane` block. No region currently records membrane voltage — adding it means a `_record("membrane", ...)` hook inside all five region `forward` loops (separate change). Until then the hero/Panel 5 use per-neuron **spikes** from the always-on `field` block, which this plan delivers. `WebSocketSink`/`RedisStreamSink` are also deferred — the `TraceSink` ABC is built so they drop in later.
- **Key fact that drives the code:** `region.n_neurons` is the region's *internal* size (e.g. sensory = hidden+concept, router = 2×n_actions). The hero renders each region's *output* spike train, whose width differs. The header therefore reports the **output width** (from `Brain` config), and `field`/`regions` read the per-region **output recording key**:

  | region id | output recording key | output width (header `n_neurons`) |
  |---|---|---|
  | `sensory` | `concept` | `brain.content` (64) |
  | `hippocampus` | `population` | `brain.hippo.n_neurons` (150) |
  | `prefrontal` | `utility` | `brain.n_actions` (4) |
  | `router` | `gate` | `brain.n_actions` (4) |
  | `motor` | `action` | `brain.n_actions` (4) |

- **Action ordering:** `GridWorldEnv` uses `0=up, 1=right, 2=down, 3=left`. Default `action_labels = ("up", "right", "down", "left")`.
- **Recordings shape:** `Brain.step(record=True)` returns `out["recordings"][region][key]` as a `[T, B, N]` tensor. Traces are single-agent (`B=1`); the builders squeeze batch index 0.

---

## File structure

- Create: `src/neuromorphic/monitor/__init__.py` — package exports
- Create: `src/neuromorphic/monitor/schema.py` — constants, `render_for_n`, `region_specs`, `PATHWAYS`, `build_header`
- Create: `src/neuromorphic/monitor/frame.py` — `build_frame` + private reducers
- Create: `src/neuromorphic/monitor/sink.py` — `TraceSink` ABC, `FileSink`
- Create: `src/neuromorphic/monitor/runner.py` — `record_episode`
- Create: `experiments/022_week11_dashboard_trace/run.py` — generates a real trace artifact
- Test: `tests/monitor/test_schema.py`, `tests/monitor/test_frame.py`, `tests/monitor/test_sink.py`, `tests/monitor/test_runner.py`

---

## Task 1: Schema constants + render ladder

**Files:**
- Create: `src/neuromorphic/monitor/__init__.py`
- Create: `src/neuromorphic/monitor/schema.py`
- Test: `tests/monitor/test_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/monitor/test_schema.py`:

```python
from neuromorphic.monitor.schema import REGION_OUTPUT_KEY, SCHEMA_VERSION, render_for_n


def test_schema_version_is_string():
    assert SCHEMA_VERSION == "1.0"


def test_region_output_key_map():
    assert REGION_OUTPUT_KEY == {
        "sensory": "concept",
        "hippocampus": "population",
        "prefrontal": "utility",
        "router": "gate",
        "motor": "action",
    }


def test_render_for_n_ladder():
    assert render_for_n(64) == "dots"
    assert render_for_n(2000) == "dots"
    assert render_for_n(2001) == "cloud"
    assert render_for_n(100_000) == "cloud"
    assert render_for_n(100_001) == "density"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/monitor/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuromorphic.monitor'`

- [ ] **Step 3: Write minimal implementation**

Create `src/neuromorphic/monitor/__init__.py`:

```python
"""``neuromorphic.monitor`` — server-side dashboard data contract.

Turns a running ``Brain`` episode into a versioned JSONL trace (header + per-step
Frames) via a ``TraceSink``. See docs/superpowers/specs/2026-06-17-stage2-dashboard-design.md.
"""

from neuromorphic.monitor.frame import build_frame
from neuromorphic.monitor.runner import record_episode
from neuromorphic.monitor.schema import SCHEMA_VERSION, build_header
from neuromorphic.monitor.sink import FileSink, TraceSink

__all__ = [
    "SCHEMA_VERSION",
    "build_header",
    "build_frame",
    "TraceSink",
    "FileSink",
    "record_episode",
]
```

Create `src/neuromorphic/monitor/schema.py` (header builder is added in Task 2 — for now just the constants the test needs):

```python
"""Schema constants and the trace header for the dashboard data contract."""

from __future__ import annotations

SCHEMA_VERSION = "1.0"

# region id -> the per-step recording key whose [T, B, N] tensor is the region's
# OUTPUT spike train (what the hero renders). Distinct from region.n_neurons.
REGION_OUTPUT_KEY = {
    "sensory": "concept",
    "hippocampus": "population",
    "prefrontal": "utility",
    "router": "gate",
    "motor": "action",
}


def render_for_n(n: int) -> str:
    """Hero representation hint as a function of output neuron count."""
    if n <= 2_000:
        return "dots"
    if n <= 100_000:
        return "cloud"
    return "density"
```

Note: `__init__.py` imports `frame`/`runner`/`sink` which do not exist yet, so it would fail to import. To keep Task 1 self-contained and green, temporarily make `__init__.py` export only the schema names:

```python
"""``neuromorphic.monitor`` — server-side dashboard data contract."""

from neuromorphic.monitor.schema import REGION_OUTPUT_KEY, SCHEMA_VERSION, render_for_n

__all__ = ["SCHEMA_VERSION", "REGION_OUTPUT_KEY", "render_for_n"]
```

(The full `__all__` above is restored in Task 5 once `frame`/`sink`/`runner` exist.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/monitor/test_schema.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/monitor/__init__.py src/neuromorphic/monitor/schema.py tests/monitor/test_schema.py
git commit -m "feat: monitor schema constants and neuron-count render ladder"
```

---

## Task 2: Trace header builder

**Files:**
- Modify: `src/neuromorphic/monitor/schema.py`
- Test: `tests/monitor/test_schema.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/monitor/test_schema.py`:

```python
from neuromorphic.brain import Brain
from neuromorphic.monitor.schema import build_header


def test_build_header_topology():
    brain = Brain(grid_n=5, seed=0)
    h = build_header(brain, seed=0, action_labels=("up", "right", "down", "left"))

    assert h["schema_version"] == "1.0"
    assert h["brain"]["T"] == brain.T
    assert h["brain"]["seed"] == 0
    assert isinstance(h["brain"]["config_hash"], str) and len(h["brain"]["config_hash"]) == 8

    assert h["task"]["type"] == "gridworld"
    assert h["task"]["grid_n"] == 5
    assert h["task"]["action_labels"] == ["up", "right", "down", "left"]

    ids = [r["id"] for r in h["regions"]]
    assert ids == ["sensory", "hippocampus", "prefrontal", "router", "motor"]

    by_id = {r["id"]: r for r in h["regions"]}
    assert by_id["sensory"]["n_neurons"] == brain.content
    assert by_id["hippocampus"]["n_neurons"] == 150
    assert by_id["prefrontal"]["n_neurons"] == brain.n_actions
    assert by_id["sensory"]["render"] == "dots"

    pathway_ids = [p["id"] for p in h["pathways"]]
    assert pathway_ids == ["sens_hippo", "sens_pfc", "hippo_pfc", "pfc_motor"]


def test_header_is_json_serializable():
    import json

    brain = Brain(grid_n=5, seed=0)
    h = build_header(brain, seed=0, action_labels=("up", "right", "down", "left"))
    json.dumps(h)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/monitor/test_schema.py -k header -v`
Expected: FAIL — `ImportError: cannot import name 'build_header'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/neuromorphic/monitor/schema.py`:

```python
import hashlib
import json

PATHWAYS = [
    {"id": "sens_hippo", "src": "sensory", "dst": "hippocampus", "gated": True, "label": "store/recall"},
    {"id": "sens_pfc", "src": "sensory", "dst": "prefrontal", "gated": False},
    {"id": "hippo_pfc", "src": "hippocampus", "dst": "prefrontal", "gated": True},
    {"id": "pfc_motor", "src": "prefrontal", "dst": "motor", "gated": True, "label": "router-gated"},
]


def region_specs(brain):
    """(id, label, output n_neurons, role) for each region, in signal-flow order.

    n_neurons is the region's OUTPUT width (what the hero renders), derived from
    brain config so it always matches the `field` tensor — not region.n_neurons.
    """
    return [
        ("sensory", "Sensory Cortex", brain.content, "input"),
        ("hippocampus", "Hippocampus", brain.hippo.n_neurons, "memory"),
        ("prefrontal", "Prefrontal", brain.n_actions, "planning"),
        ("router", "Thalamic Router", brain.n_actions, "control"),
        ("motor", "Motor Cortex", brain.n_actions, "output"),
    ]


def _config_hash(brain, seed: int) -> str:
    payload = {
        "content": brain.content,
        "n_actions": brain.n_actions,
        "n_hippo": brain.hippo.n_neurons,
        "T": brain.T,
        "grid_n": brain.grid_n,
        "seed": seed,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def build_header(brain, *, seed: int, action_labels, task_type: str = "gridworld", grid_n: int | None = None) -> dict:
    """Build the once-per-run trace header declaring brain topology + run context."""
    grid_n = brain.grid_n if grid_n is None else grid_n
    regions = [
        {"id": rid, "label": label, "n_neurons": n, "role": role, "render": render_for_n(n)}
        for rid, label, n, role in region_specs(brain)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "brain": {"id": "five-region", "config_hash": _config_hash(brain, seed), "seed": seed, "T": brain.T},
        "task": {"type": task_type, "grid_n": grid_n, "action_labels": list(action_labels)},
        "regions": regions,
        "pathways": [dict(p) for p in PATHWAYS],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/monitor/test_schema.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/monitor/schema.py tests/monitor/test_schema.py
git commit -m "feat: build_header — data-driven brain topology for the dashboard"
```

---

## Task 3: Frame builder

**Files:**
- Create: `src/neuromorphic/monitor/frame.py`
- Test: `tests/monitor/test_frame.py`

- [ ] **Step 1: Write the failing test**

Create `tests/monitor/test_frame.py`:

```python
import json

import torch

from neuromorphic.brain import Brain
from neuromorphic.monitor.frame import build_frame


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


def test_frame_is_json_serializable():
    _, out = _recorded_step()
    frame = build_frame(out, episode=0, step=0, t=0.0, task=_task(), store=False, recall=True)
    json.dumps(frame)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/monitor/test_frame.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuromorphic.monitor.frame'`

- [ ] **Step 3: Write minimal implementation**

Create `src/neuromorphic/monitor/frame.py`:

```python
"""Build one JSON-serializable Frame from one ``Brain.step(record=True)`` output."""

from __future__ import annotations

import torch

from neuromorphic.monitor.schema import REGION_OUTPUT_KEY


def _field_tensor(out: dict, region: str) -> torch.Tensor:
    """The region's output spike train, squeezed to [T, N] (single-agent trace)."""
    key = REGION_OUTPUT_KEY[region]
    rec = out["recordings"][region][key]  # [T, B, N]
    return rec[:, 0, :].float()


def _region_summary(field: torch.Tensor) -> dict:
    """Scalar activity summary for one region from its [T, N] output spikes."""
    return {
        "rate": float(field.mean()),
        "spikes": int(field.sum()),
        "active_frac": float((field.sum(dim=0) > 0).float().mean()),
        "rate_t": field.mean(dim=1).tolist(),  # [T]
    }


def _router_block(out: dict) -> dict:
    gate_open = (1 - out["gate_closed"]).float()  # [T, B, A]
    utilities = out["utilities"].float()          # [T, B, A]
    return {
        "gate_open": gate_open.mean(dim=0)[0].tolist(),   # [A] open-fraction per action
        "gate_open_t": gate_open[:, 0, :].tolist(),       # [T, A]
        "utilities": utilities.mean(dim=0)[0].tolist(),   # [A] utility rate per action
    }


def _pathways(region_rate: dict, out: dict, store: bool, recall: bool) -> dict:
    pfc_motor_open = (1 - out["gate_closed"]).float().mean(dim=0)[0].tolist()  # [A]
    return {
        "sens_hippo": {"intensity": region_rate["sensory"], "gate_open": 1.0 if store else 0.0},
        "sens_pfc": {"intensity": region_rate["sensory"]},
        "hippo_pfc": {"intensity": region_rate["hippocampus"], "gate_open": 1.0 if recall else 0.0},
        "pfc_motor": {"intensity": region_rate["prefrontal"], "gate_open": pfc_motor_open},
    }


def build_frame(out: dict, *, episode: int, step: int, t: float, task: dict, store: bool, recall: bool) -> dict:
    """Assemble one Frame. ``out`` must come from ``Brain.step(record=True)``."""
    fields = {r: _field_tensor(out, r) for r in REGION_OUTPUT_KEY}
    regions = {r: _region_summary(f) for r, f in fields.items()}
    region_rate = {r: regions[r]["rate"] for r in regions}
    return {
        "episode": episode,
        "step": step,
        "t": t,
        "task": task,
        "regions": regions,
        "pathways": _pathways(region_rate, out, store, recall),
        "router": _router_block(out),
        "field": {r: {"spikes": f.int().tolist()} for r, f in fields.items()},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/monitor/test_frame.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/monitor/frame.py tests/monitor/test_frame.py
git commit -m "feat: build_frame — per-step regions/pathways/router/field blocks"
```

---

## Task 4: TraceSink abstraction + FileSink

**Files:**
- Create: `src/neuromorphic/monitor/sink.py`
- Test: `tests/monitor/test_sink.py`

- [ ] **Step 1: Write the failing test**

Create `tests/monitor/test_sink.py`:

```python
import json

from neuromorphic.monitor.sink import FileSink


def test_filesink_writes_header_then_frames(tmp_path):
    path = tmp_path / "trace.jsonl"
    sink = FileSink(path)
    sink.open({"schema_version": "1.0", "regions": []})
    sink.write({"step": 0, "x": 1})
    sink.write({"step": 1, "x": 2})
    sink.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    header = json.loads(lines[0])
    assert header["schema_version"] == "1.0"
    assert json.loads(lines[1])["step"] == 0
    assert json.loads(lines[2])["step"] == 1


def test_filesink_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "trace.jsonl"
    sink = FileSink(path)
    sink.open({"schema_version": "1.0"})
    sink.close()
    assert path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/monitor/test_sink.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuromorphic.monitor.sink'`

- [ ] **Step 3: Write minimal implementation**

Create `src/neuromorphic/monitor/sink.py`:

```python
"""``TraceSink`` — interchangeable destinations for header + frames.

``FileSink`` writes JSONL (the system of record). ``WebSocketSink`` /
``RedisStreamSink`` will implement the same interface later — the same Frame
object, a different ``write``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path


class TraceSink(ABC):
    """Destination for one run: ``open(header)`` once, ``write(frame)`` per step, ``close()``."""

    @abstractmethod
    def open(self, header: dict) -> None: ...

    @abstractmethod
    def write(self, frame: dict) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class FileSink(TraceSink):
    """Append header + frames to a JSONL file (header = line 0)."""

    def __init__(self, path):
        self.path = Path(path)
        self._fh = None

    def open(self, header: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(json.dumps(header) + "\n")

    def write(self, frame: dict) -> None:
        if self._fh is None:
            raise RuntimeError("FileSink.write called before open()")
        self._fh.write(json.dumps(frame) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/monitor/test_sink.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/monitor/sink.py tests/monitor/test_sink.py
git commit -m "feat: TraceSink abstraction + JSONL FileSink"
```

---

## Task 5: Episode runner + restore package exports

**Files:**
- Create: `src/neuromorphic/monitor/runner.py`
- Modify: `src/neuromorphic/monitor/__init__.py`
- Test: `tests/monitor/test_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/monitor/test_runner.py`:

```python
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
    assert set(frame) >= {"task", "regions", "pathways", "router", "field"}
    assert frame["task"]["action_label"] in ("up", "right", "down", "left")
    assert len(frame["field"]["hippocampus"]["spikes"][0]) == 150
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/monitor/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuromorphic.monitor.runner'`

- [ ] **Step 3: Write minimal implementation**

Create `src/neuromorphic/monitor/runner.py`:

```python
"""Drive one Gymnasium episode with a ``Brain`` and stream Frames to a ``TraceSink``."""

from __future__ import annotations

from neuromorphic.monitor.frame import build_frame
from neuromorphic.monitor.schema import build_header

DEFAULT_ACTION_LABELS = ("up", "right", "down", "left")


def record_episode(
    brain,
    env,
    sink,
    *,
    seed: int = 0,
    action_labels=DEFAULT_ACTION_LABELS,
    max_steps: int | None = None,
    store_first: bool = True,
    recall: bool = True,
    generator=None,
) -> dict:
    """Record one episode to ``sink`` (header + one Frame per env step).

    Returns a summary dict: ``steps``, ``total_reward``, ``reached_goal``.
    """
    sink.open(build_header(brain, seed=seed, action_labels=action_labels))

    obs, _ = env.reset()
    if store_first:
        brain.remember(obs, generator=generator)

    total_reward = 0.0
    reached_goal = False
    steps = 0
    limit = max_steps if max_steps is not None else getattr(env, "max_steps", 100)

    while steps < limit:
        out = brain.step(obs, store=False, recall=recall, record=True, generator=generator)
        action = int(out["action"])
        next_obs, reward, terminated, truncated, _ = env.step(action)
        brain.learn(reward)
        total_reward += float(reward)

        task = {
            "agent": [int(obs[0]), int(obs[1])],
            "goal": [int(obs[2]), int(obs[3])],
            "action": action,
            "action_label": action_labels[action],
            "reward": float(reward),
            "return": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        frame = build_frame(
            out, episode=0, step=steps, t=float(steps), task=task, store=False, recall=recall
        )
        sink.write(frame)

        obs = next_obs
        steps += 1
        if terminated:
            reached_goal = True
            break
        if truncated:
            break

    sink.close()
    return {"steps": steps, "total_reward": total_reward, "reached_goal": reached_goal}
```

Then restore the full exports in `src/neuromorphic/monitor/__init__.py` (replace the temporary schema-only version from Task 1):

```python
"""``neuromorphic.monitor`` — server-side dashboard data contract.

Turns a running ``Brain`` episode into a versioned JSONL trace (header + per-step
Frames) via a ``TraceSink``. See docs/superpowers/specs/2026-06-17-stage2-dashboard-design.md.
"""

from neuromorphic.monitor.frame import build_frame
from neuromorphic.monitor.runner import record_episode
from neuromorphic.monitor.schema import REGION_OUTPUT_KEY, SCHEMA_VERSION, build_header, render_for_n
from neuromorphic.monitor.sink import FileSink, TraceSink

__all__ = [
    "SCHEMA_VERSION",
    "REGION_OUTPUT_KEY",
    "render_for_n",
    "build_header",
    "build_frame",
    "TraceSink",
    "FileSink",
    "record_episode",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/monitor/ -v`
Expected: PASS (all monitor tests, including the full suite importing the package)

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/monitor/runner.py src/neuromorphic/monitor/__init__.py tests/monitor/test_runner.py
git commit -m "feat: record_episode runner + full monitor package exports"
```

---

## Task 6: Experiment script — generate a real trace artifact

**Files:**
- Create: `experiments/022_week11_dashboard_trace/run.py`

This produces the real trace file the design team loads. It is a script (matching the repo's numbered-experiment convention), verified manually rather than by pytest.

- [ ] **Step 1: Write the experiment script**

Create `experiments/022_week11_dashboard_trace/run.py`:

```python
"""EXP-022 — generate a dashboard trace (Week-11 S2, L17).

Runs one untrained five-region episode and writes a versioned JSONL trace to
``outputs/week11_dashboard_trace.jsonl`` (header line + one Frame per step). This
is the real-data artifact the Stage-2 dashboard / design system loads.

Run:
    python experiments/022_week11_dashboard_trace/run.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from neuromorphic.brain import Brain
from neuromorphic.envs import GridWorldEnv
from neuromorphic.monitor import FileSink, record_episode

OUT = Path("outputs/week11_dashboard_trace.jsonl")


def main() -> None:
    env = GridWorldEnv()
    brain = Brain(grid_n=env.size, seed=0)
    sink = FileSink(OUT)
    summary = record_episode(
        brain, env, sink, seed=0, generator=torch.Generator().manual_seed(0)
    )
    print("EXP-022 — dashboard trace written")
    print(f"  file         : {OUT}")
    print(f"  steps        : {summary['steps']}")
    print(f"  total reward : {summary['total_reward']:.0f}")
    print(f"  reached goal : {summary['reached_goal']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `python experiments/022_week11_dashboard_trace/run.py`
Expected: prints a summary and writes `outputs/week11_dashboard_trace.jsonl`.

- [ ] **Step 3: Verify the artifact manually**

Run: `python -c "import json,pathlib; L=pathlib.Path('outputs/week11_dashboard_trace.jsonl').read_text().splitlines(); h=json.loads(L[0]); f=json.loads(L[1]); print('lines',len(L)); print('regions',[r['id'] for r in h['regions']]); print('frame keys',sorted(f)); print('hippo field TxN',len(f['field']['hippocampus']['spikes']),'x',len(f['field']['hippocampus']['spikes'][0]))"`

Expected output (shape, not exact values):
- `lines` ≥ 2 (header + ≥1 frame)
- `regions ['sensory', 'hippocampus', 'prefrontal', 'router', 'motor']`
- `frame keys ['episode', 'field', 'pathways', 'regions', 'router', 'step', 't', 'task']`
- `hippo field TxN 32 x 150`

- [ ] **Step 4: Confirm the trace is gitignored or intentionally committed**

Run: `git check-ignore outputs/week11_dashboard_trace.jsonl; git status --short experiments/022_week11_dashboard_trace/`
Decision: commit the **script**; the generated `.jsonl` may be large — check `.gitignore` for an `outputs/` rule and only commit the trace if you want it as a fixture. (The `outputs/` dir currently holds committed PNGs, so committing one small trace is acceptable.)

- [ ] **Step 5: Commit**

```bash
git add experiments/022_week11_dashboard_trace/run.py
git commit -m "exp 022: generate week-11 dashboard trace artifact"
```

---

## Self-review notes (already applied)

- **Spec coverage:** Frame `task`/`regions`/`pathways`/`router`/`field` blocks (spec §3.4) → Tasks 3/5. Header topology (§3.3) → Task 2. `TraceSink` + `FileSink` (§3.2) → Task 4. JSONL serialization (§3.1) → Task 4. Scaling `render` hint (§6) → Task 1. Real artifact for the design hand-off (§7 step 1) → Task 6.
- **Deferred with rationale (not gaps):** `detail.membrane` (needs region-level membrane hooks — no region records membrane today; hero/Panel 5 use `field` spikes meanwhile); `WebSocketSink`/`RedisStreamSink` (ABC built, implementations on demand per §7 steps 5–6).
- **Type consistency:** `REGION_OUTPUT_KEY` keys, header region ids, `field`/`regions` keys, and `out["recordings"]` keys all use the same five ids (`sensory`/`hippocampus`/`prefrontal`/`router`/`motor`). `gate_open = 1 - gate_closed` applied once, consistently, in `_router_block` and `_pathways`.
```
