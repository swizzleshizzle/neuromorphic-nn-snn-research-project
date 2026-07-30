# Dashboard Cube Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a 2x2 cube episode to a JSONL trace and display it in the NEURO-SCOPE dashboard, without breaking the existing gridworld path.

**Architecture:** A `TaskAdapter` protocol owns the three task-specific places in the trace contract (header task block, per-frame task block, sensory encoding block). `GridworldAdapter` reproduces today's output byte for byte; `CubeAdapter` is new. On the dashboard, the frame task type becomes a discriminated union stamped from the header at parse time, so committed traces keep loading and TypeScript refuses to read `task.agent` on a cube frame.

**Tech Stack:** Python 3.13 + PyTorch + Gymnasium (library, `.venv`), pytest. React 18 + TypeScript + zustand + vitest (dashboard).

## Global Constraints

- Run python only via `.venv\Scripts\python.exe`. Never bare `python`.
- **No em-dashes** anywhere in code, comments, docs, or commit messages.
- Commit messages are plain. **No `Co-Authored-By` trailer, no "Generated with" line.**
- Stage explicit paths. **Never `git add -A`.**
- **Never write an assertion that cannot fail.** Every test below must fail against pre-change code. Prefer a measured numeric threshold to a qualitative check.
- **Never weaken a passing threshold to make a change land.** If a change would require that, stop and report.
- Action-space width comes from `N_ACTIONS` / `MOVE_LABELS` / `env.action_space.n`, **never a literal 6**.
- Do not edit `src/neuromorphic/training/cube_baseline.py` or `experiments/030_memory_engagement/`. EXP-030 is running against this repo.
- Python tests: `.venv\Scripts\python.exe -m pytest tests/ -q -m "not slow"` (about 13 min). Full suite must stay at 354 or more.
- Dashboard tests: `cd dashboard; npm test` (vitest run).
- Branch: `week17-dashboard-cube`. Spec: `docs/superpowers/specs/2026-07-29-dashboard-cube-support-design.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/neuromorphic/monitor/tasks.py` | **new.** `TaskAdapter` protocol, `GridworldAdapter`, `CubeAdapter`. The only place task-specific trace shape lives. |
| `src/neuromorphic/monitor/schema.py` | Header assembly. Loses `task_type`/`grid_n` params, gains `adapter`. `_config_hash` stops reading `brain.grid_n`. |
| `src/neuromorphic/monitor/frame.py` | Frame assembly. Encoding block delegated to the adapter. |
| `src/neuromorphic/monitor/runner.py` | Episode drivers. Lose the hardcoded agent/goal task dict and the 4-wide action labels. |
| `src/neuromorphic/monitor/__init__.py` | Re-export the adapters. |
| `scripts/record_cube_trace.py` | **new.** Train a cube policy in-process at a seed, record one greedy episode. |
| `dashboard/src/contract.ts` | Three discriminated unions: `TaskState`, header task, `SensoryInput`. |
| `dashboard/src/source/parseTrace.ts` | Stamps `type` onto each frame task from the header. Rejects mismatched traces. |
| `dashboard/src/panels/cubeNet.ts` | **new, pure.** Facelet index 0-23 to unfolded-net (row, col). |
| `dashboard/src/panels/TaskState.tsx` | Narrows on type, delegates to `GridTaskView` / `CubeTaskView`. |
| `dashboard/src/hero/sensory.ts` | Adds `aggregateCubeFacelets`. |
| `dashboard/src/hero/overlays/SensoryGrid.tsx` | Renders whichever variant the frame carries. |

---

### Task 1: Task adapters (pure, unwired)

**Files:**
- Create: `src/neuromorphic/monitor/tasks.py`
- Test: `tests/monitor/test_tasks.py`

**Interfaces:**
- Consumes: `neuromorphic.envs.cube.MOVE_LABELS`, `N_ACTIONS`.
- Produces: `TaskAdapter` (Protocol), `GridworldAdapter(grid_n, action_labels=GRID_ACTION_LABELS)`, `CubeAdapter(cube_n=2, n_colors=6)`. All three expose `action_labels: tuple[str, ...]`, `header_task() -> dict`, `frame_task(obs, *, action, reward, total, terminated, truncated, info) -> dict`, `encoding(out) -> dict | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/monitor/test_tasks.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/monitor/test_tasks.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'neuromorphic.monitor.tasks'`

- [ ] **Step 3: Write the implementation**

```python
# src/neuromorphic/monitor/tasks.py
"""Per-task trace blocks, so the monitor itself stays task agnostic.

A TaskAdapter owns the three places where the dashboard data contract is task
specific: the header's task block, each frame's task block, and the sensory
encoding block. Adding a task means adding an adapter, not adding a branch to
each of schema.py, frame.py and runner.py.
"""

from __future__ import annotations

from typing import Protocol

from neuromorphic.envs.cube import MOVE_LABELS

GRID_ACTION_LABELS: tuple[str, ...] = ("up", "right", "down", "left")


class TaskAdapter(Protocol):
    """The task-specific half of the trace contract."""

    action_labels: tuple[str, ...]

    def header_task(self) -> dict: ...

    def frame_task(self, obs, *, action, reward, total, terminated, truncated, info) -> dict: ...

    def encoding(self, out: dict) -> dict | None: ...


class GridworldAdapter:
    """Reproduces the pre-adapter gridworld blocks exactly."""

    def __init__(self, grid_n: int, action_labels=GRID_ACTION_LABELS):
        self.grid_n = grid_n
        self.action_labels = tuple(action_labels)

    def header_task(self) -> dict:
        return {
            "type": "gridworld",
            "grid_n": self.grid_n,
            "action_labels": list(self.action_labels),
        }

    def frame_task(self, obs, *, action, reward, total, terminated, truncated, info) -> dict:
        return {
            "agent": [int(obs[0]), int(obs[1])],
            "goal": [int(obs[2]), int(obs[3])],
            "action": action,
            "action_label": self.action_labels[action],
            "reward": float(reward),
            "return": total,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def encoding(self, out: dict) -> dict | None:
        if "obs_spikes" not in out:
            return None
        return {
            "sensory_input": {
                "spikes": out["obs_spikes"][:, 0, :].int().tolist(),
                "grid_n": self.grid_n,
                "planes": ["agent", "goal"],
                "index": "y*grid_n + x",
            }
        }


class CubeAdapter:
    """Cube blocks. Width comes from MOVE_LABELS, never a literal."""

    def __init__(self, cube_n: int = 2, n_colors: int = 6):
        self.cube_n = cube_n
        self.n_colors = n_colors
        self.action_labels = tuple(MOVE_LABELS)

    def header_task(self) -> dict:
        return {
            "type": "cube",
            "cube_n": self.cube_n,
            "action_labels": list(self.action_labels),
        }

    def frame_task(self, obs, *, action, reward, total, terminated, truncated, info) -> dict:
        distance = info.get("distance")
        return {
            "facelets": [int(c) for c in obs],
            "solved": bool(info.get("solved", False)),
            "distance": None if distance is None else int(distance),
            "scramble_depth": int(info.get("scramble_depth", 0)),
            "move": action,
            "move_label": info.get("move_label"),
            "action": action,
            "action_label": self.action_labels[action],
            "reward": float(reward),
            "return": total,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def encoding(self, out: dict) -> dict | None:
        if "obs_spikes" not in out:
            return None
        return {
            "sensory_input": {
                "spikes": out["obs_spikes"][:, 0, :].int().tolist(),
                "cube_n": self.cube_n,
                "n_colors": self.n_colors,
                "index": "facelet*n_colors + color",
            }
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/monitor/test_tasks.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/monitor/tasks.py tests/monitor/test_tasks.py
git commit -m "feat(monitor): add task adapters for gridworld and cube trace blocks"
```

---

### Task 2: Wire the adapters in, proving gridworld unchanged

**Files:**
- Create: `tests/monitor/fixtures/gridworld_reference_trace.jsonl` (generated in Step 1)
- Create: `tests/monitor/test_gridworld_unchanged.py`
- Modify: `src/neuromorphic/monitor/schema.py`, `frame.py`, `runner.py`, `__init__.py`

**Interfaces:**
- Consumes: `GridworldAdapter`, `CubeAdapter` from Task 1.
- Produces: `build_header(brain, *, seed, adapter, policy_regions=None)`, `build_frame(out, *, episode, step, t, task, store, recall, adapter=None)`, `record_episode(..., adapter=None)`, `record_policy_episode(..., adapter=None)`. When `adapter is None` both runners default to `GridworldAdapter(brain.grid_n)`.

- [ ] **Step 1: Capture the pre-change reference trace**

This must run BEFORE any source edit. It is the ground truth the refactor is checked against.

```bash
.venv\Scripts\python.exe -c "import torch; from pathlib import Path; from neuromorphic.brain import Brain; from neuromorphic.envs import GridWorldEnv; from neuromorphic.monitor import FileSink, record_episode; p = Path('tests/monitor/fixtures/gridworld_reference_trace.jsonl'); p.parent.mkdir(parents=True, exist_ok=True); env = GridWorldEnv(); brain = Brain(grid_n=env.size, seed=0); print(record_episode(brain, env, FileSink(p), seed=0, generator=torch.Generator().manual_seed(0)))"
```

Expected: prints a summary dict, and the fixture file exists with a header line plus one line per step.

- [ ] **Step 2: Write the failing equality test**

```python
# tests/monitor/test_gridworld_unchanged.py
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
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/monitor/test_gridworld_unchanged.py -q`
Expected: FAIL on `assert new_h["schema_version"] == "1.1"` (it is still `"1.0"`).

- [ ] **Step 4: Edit `schema.py`**

Set `SCHEMA_VERSION = "1.1"`. Replace `_config_hash` and `build_header` with:

```python
def _config_hash(brain, seed: int, task_type: str) -> str:
    payload = {
        "content": brain.content,
        "n_actions": brain.n_actions,
        "n_hippo": brain.hippo.n_neurons,
        "T": brain.T,
        # n_obs is meaningful for every task; grid_n is meaningful for exactly one,
        # and a cube brain carries the Brain default of 5, which means nothing.
        "n_obs": brain.n_obs,
        "task": task_type,
        "seed": seed,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def build_header(brain, *, seed: int, adapter, policy_regions=None) -> dict:
    """Build the once-per-run trace header declaring brain topology + run context."""
    task = adapter.header_task()
    regions = [
        {"id": rid, "label": label, "n_neurons": n, "role": role, "render": render_for_n(n)}
        for rid, label, n, role in region_specs(brain)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "brain": {
            "id": "five-region",
            "config_hash": _config_hash(brain, seed, task["type"]),
            "seed": seed,
            "T": brain.T,
        },
        "task": task,
        "regions": regions,
        "pathways": [dict(p) for p in PATHWAYS],
        "policy_regions": list(policy_regions or []),
    }
```

- [ ] **Step 5: Edit `frame.py`**

Delete `_encoding_block`. Change the signature and the tail of `build_frame`:

```python
def build_frame(
    out: dict,
    *,
    episode: int,
    step: int,
    t: float,
    task: dict,
    store: bool,
    recall: bool,
    adapter=None,
) -> dict:
    """Assemble one Frame. ``out`` must come from ``Brain.step(record=True)``.

    ``adapter`` supplies the ``encoding`` block (needs ``out["obs_spikes"]``);
    omit it to skip the block.
    """
```

and replace the trailing `if grid_n is not None ...` with:

```python
    if adapter is not None:
        encoding = adapter.encoding(out)
        if encoding is not None:
            frame["encoding"] = encoding
    return frame
```

- [ ] **Step 6: Edit `runner.py`**

Add the import and default. In BOTH `record_episode` and `record_policy_episode`:

- add `adapter=None` to the keyword arguments and drop `action_labels=DEFAULT_ACTION_LABELS`
- immediately after the docstring, insert:

```python
    if adapter is None:
        adapter = GridworldAdapter(brain.grid_n)
```

- replace the `sink.open(build_header(...))` call with `sink.open(build_header(brain, seed=seed, adapter=adapter))` (keep `policy_regions=list(policy_regions)` in `record_policy_episode`)
- replace the inline `task = {...}` dict with:

```python
            task = adapter.frame_task(
                obs, action=action, reward=reward, total=total_reward,
                terminated=terminated, truncated=truncated, info=info,
            )
```

- capture `info` from the env: change `next_obs, reward, terminated, truncated, _ = env.step(action)` to `next_obs, reward, terminated, truncated, info = env.step(action)` in both functions
- replace `grid_n=brain.grid_n` in the `build_frame` call with `adapter=adapter`

Header import line becomes:

```python
from neuromorphic.monitor.schema import REGION_OUTPUT_KEY, build_header, region_specs
from neuromorphic.monitor.tasks import GRID_ACTION_LABELS, GridworldAdapter
```

Keep `DEFAULT_ACTION_LABELS = GRID_ACTION_LABELS` as a module-level alias so any external caller importing it still works.

- [ ] **Step 7: Export the adapters from `__init__.py`**

Add to the imports and `__all__`:

```python
from neuromorphic.monitor.tasks import CubeAdapter, GridworldAdapter, TaskAdapter
```

with `"TaskAdapter"`, `"GridworldAdapter"`, `"CubeAdapter"` appended to `__all__`.

- [ ] **Step 8: Run the monitor tests**

Run: `.venv\Scripts\python.exe -m pytest tests/monitor/ -q`
Expected: PASS. Existing tests calling `build_header(brain, seed=0, action_labels=...)` or `build_frame(..., grid_n=...)` will fail on the changed signature; update those call sites to pass `adapter=GridworldAdapter(brain.grid_n)`. Do NOT weaken any existing assertion to make this pass. `test_schema.py`'s `h["task"]["grid_n"] == 5` must still hold, because `GridworldAdapter` still emits it.

- [ ] **Step 9: Run the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q -m "not slow"`
Expected: all pass, count at or above the previous 353 non-slow.

- [ ] **Step 10: Commit**

```bash
git add src/neuromorphic/monitor/ tests/monitor/
git commit -m "refactor(monitor): route task-specific trace blocks through adapters

Gridworld frames stay field-identical; only schema_version and config_hash
change. config_hash now hashes n_obs and the task type instead of grid_n,
which a cube brain carries only as the Brain default of 5."
```

---

### Task 3: Cube trace recorder

**Files:**
- Create: `scripts/record_cube_trace.py`
- Test: `tests/monitor/test_record_cube_trace.py`

**Interfaces:**
- Consumes: `CubeAdapter`, `record_policy_episode`, `neuromorphic.training.cube_baseline` (read-only import of `CubeConfig`, `max_steps_for`, `make_agent`, `feature_width`, `shell_states`, `split_shell`), `neuromorphic.training.reinforce.train_episode`.
- Produces: `train_cube_policy(depth, seed, episodes) -> (brain, head)` and `record(depth, seed, episodes, out_path) -> dict` in `scripts/record_cube_trace.py`.

**Why the training loop is replicated here rather than imported:** `run_cube_baseline` returns a JSON record, not the trained `(brain, head)`, and the plan may not edit `cube_baseline.py` while EXP-030 runs. The replication mirrors that function's seeding order exactly, and Step 1's test is what guards the replication against drift.

- [ ] **Step 1: Write the failing test**

```python
# tests/monitor/test_record_cube_trace.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/monitor/test_record_cube_trace.py -q`
Expected: FAIL, the script file does not exist.

- [ ] **Step 3: Write the recorder**

```python
# scripts/record_cube_trace.py
"""Record one 2x2 cube episode to a dashboard JSONL trace.

Trains a cube policy in-process at a fixed seed, then records one greedy episode.
No checkpoint format is involved: ``checkpoints.load_trained`` hardcodes a
gridworld ``Brain`` and cannot rebuild a cube brain.

The training loop mirrors ``run_cube_baseline``'s seeding order exactly.
``tests/monitor/test_record_cube_trace.py`` guards that against drift.

Run:
    .venv\\Scripts\\python.exe scripts/record_cube_trace.py --depth 2 --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from neuromorphic.analysis.ablate import AblatedConcept
from neuromorphic.envs.cube import CubeEnv
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.monitor import CubeAdapter, FileSink, record_policy_episode
from neuromorphic.training.cube_baseline import (
    CubeConfig,
    MemoryReadout,
    ShellCubeEnv,
    feature_width,
    make_agent,
    max_steps_for,
    shell_states,
    split_shell,
)
from neuromorphic.training.reinforce import ema, policy_parameters, train_episode

DEFAULT_OUT = Path("outputs/cube_trace.jsonl")


def train_cube_policy(depth: int, seed: int, episodes: int, provider):
    """Train a concept-readout cube policy.

    Mirrors ``run_cube_baseline``'s seeding order and object construction exactly,
    including the ``AblatedConcept`` wrapper (used there even at sigma 0, with a
    ``None`` spec) and the concept-mode ``MemoryReadout``. Substituting a bare
    ``nn.Linear`` or ``feature_fn=None`` would look equivalent and silently consume
    the RNG stream differently.
    """
    cfg = CubeConfig(seed=seed, depth=depth, episodes=episodes, readout="concept")
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    states = shell_states(provider, cfg.depth)
    train_states, _, _ = split_shell(
        states, cfg.depth, seed=cfg.seed,
        heldout_cap=cfg.heldout_cap, heldout_frac=cfg.heldout_frac,
    )

    brain = make_agent(cfg)
    torch.manual_seed(cfg.seed)  # head init stream matched to run_cube_baseline
    readout = MemoryReadout(cfg.readout, random.Random(cfg.seed), brain)
    width = feature_width(cfg)
    head = AblatedConcept(nn.Linear(width, cfg.n_actions), None, width=width)
    optimizer = torch.optim.Adam(policy_parameters(head), lr=cfg.lr)
    env = ShellCubeEnv(
        train_states, random.Random(cfg.seed),
        scramble_depth=cfg.depth, max_steps=max_steps_for(cfg.depth),
    )

    baseline = 0.0
    for _ in range(cfg.episodes):
        readout.reset()
        stats = train_episode(
            brain, head, env, optimizer,
            gamma=cfg.gamma, baseline=baseline, generator=generator,
            max_steps=max_steps_for(cfg.depth),
            entropy_beta=cfg.entropy_beta,
            normalize_advantages=cfg.normalize_advantages,
            store=False, recall=False, feature_fn=readout,
        )
        baseline = ema(baseline, stats["mean_return"], cfg.baseline_beta)
    return brain, head, generator


def record(*, depth: int, seed: int, episodes: int, out_path) -> dict:
    """Train, then record one greedy episode to ``out_path``."""
    # Built once and shared: ExactBFSDistance expands the full 3,674,160-state table
    # (about 67 s). Building it separately for training and recording doubles that.
    provider = ExactBFSDistance(max_depth=max(6, depth))
    brain, head, generator = train_cube_policy(depth, seed, episodes, provider)
    env = CubeEnv(
        scramble_depth=depth,
        max_steps=max_steps_for(depth),
        scramble_seed=seed,
        distance_provider=provider,
    )
    summary = record_policy_episode(
        brain, head, env, FileSink(out_path),
        seed=seed,
        adapter=CubeAdapter(),
        max_steps=max_steps_for(depth),
        recall=False,
        policy_regions=("sensory",),
        generator=generator,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    summary = record(depth=args.depth, seed=args.seed,
                     episodes=args.episodes, out_path=args.out)
    print("cube trace written")
    print(f"  file         : {args.out}")
    print(f"  depth        : {args.depth}")
    print(f"  steps        : {summary['steps']}")
    print(f"  reached goal : {summary['reached_goal']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/monitor/test_record_cube_trace.py -q`
Expected: 4 passed. If `record_policy_episode` rejects `action_labels` removal or `info` capture, fix Task 2's edits rather than loosening a test.

- [ ] **Step 5: Generate a real trace for the dashboard**

Run: `.venv\Scripts\python.exe scripts/record_cube_trace.py --depth 2 --seed 0 --episodes 600 --out dashboard/public/cube_trace.jsonl`
Expected: prints a summary; the file has a header plus one line per step. This takes several minutes (600 training episodes at about 90 ms per `brain.step`).

- [ ] **Step 6: Commit**

```bash
git add scripts/record_cube_trace.py tests/monitor/test_record_cube_trace.py dashboard/public/cube_trace.jsonl
git commit -m "feat(monitor): add a cube trace recorder script"
```

---

### Task 4: Dashboard contract and parse-time stamping

**Files:**
- Modify: `dashboard/src/contract.ts`, `dashboard/src/source/parseTrace.ts`
- Test: `dashboard/src/source/parseTrace.test.ts`

**Interfaces:**
- Produces: `GridTask`, `CubeTask`, `TaskState = GridTask | CubeTask`, `GridTaskHeader`, `CubeTaskHeader`, `GridSensoryInput`, `CubeSensoryInput`. `parseTrace(text)` stamps `type` onto every frame task.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/src/source/parseTrace.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { parseTrace } from "./parseTrace";

const gridHeader = {
  schema_version: "1.1",
  brain: { id: "five-region", config_hash: "abc12345", seed: 0, T: 8 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [], pathways: [], policy_regions: [],
};
const cubeHeader = {
  ...gridHeader,
  task: { type: "cube", cube_n: 2, action_labels: ["U", "U'", "R", "R'", "F", "F'"] },
};
const jsonl = (h: unknown, fs: unknown[]) =>
  [JSON.stringify(h), ...fs.map((f) => JSON.stringify(f))].join("\n");

describe("parseTrace task stamping", () => {
  it("stamps gridworld frames from the header", () => {
    const t = parseTrace(jsonl(gridHeader, [{ step: 0, task: { agent: [1, 2], goal: [3, 4] } }]));
    expect(t.frames[0].task.type).toBe("gridworld");
  });

  it("stamps cube frames from the header", () => {
    const t = parseTrace(jsonl(cubeHeader, [{ step: 0, task: { facelets: new Array(24).fill(0) } }]));
    const task = t.frames[0].task;
    expect(task.type).toBe("cube");
    if (task.type === "cube") expect(task.facelets).toHaveLength(24);
  });

  it("throws when a cube header carries gridworld frames", () => {
    expect(() =>
      parseTrace(jsonl(cubeHeader, [{ step: 0, task: { agent: [1, 2], goal: [3, 4] } }])),
    ).toThrow(/cube.*agent|agent.*cube/i);
  });

  it("throws when a gridworld header carries cube frames", () => {
    expect(() =>
      parseTrace(jsonl(gridHeader, [{ step: 0, task: { facelets: new Array(24).fill(0) } }])),
    ).toThrow(/gridworld.*facelets|facelets.*gridworld/i);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard; npm test -- parseTrace`
Expected: FAIL, `task.type` is undefined.

- [ ] **Step 3: Rewrite the contract types**

In `dashboard/src/contract.ts`, replace the `TaskState` and `SensoryInput` interfaces and the header's `task` field:

```ts
export interface TaskCore {
  action: number;
  action_label: string;
  reward: number;
  return: number;
  terminated: boolean;
  truncated: boolean;
}

export interface GridTask extends TaskCore {
  type: "gridworld";
  agent: [number, number];
  goal: [number, number];
}

export interface CubeTask extends TaskCore {
  type: "cube";
  facelets: number[];
  solved: boolean;
  distance: number | null;
  scramble_depth: number;
  move: number;
  move_label: string | null;
}

export type TaskState = GridTask | CubeTask;

export interface GridTaskHeader {
  type: "gridworld";
  grid_n: number;
  action_labels: string[];
}

export interface CubeTaskHeader {
  type: "cube";
  cube_n: number;
  action_labels: string[];
}

export type TaskHeader = GridTaskHeader | CubeTaskHeader;

export interface GridSensoryInput {
  spikes: number[][];
  grid_n: number;
  planes: string[];
  index: string;
}

export interface CubeSensoryInput {
  spikes: number[][];
  cube_n: number;
  n_colors: number;
  index: string;
}

export type SensoryInput = GridSensoryInput | CubeSensoryInput;
```

and change `TraceHeader`'s field to `task: TaskHeader;`.

- [ ] **Step 4: Implement stamping in `parseTrace.ts`**

```ts
import type { Frame, Trace, TraceHeader } from "../contract";

/** Parse JSONL trace text: line 0 = header, each remaining non-blank line = one Frame.
 *
 * The wire format carries the task type once, in the header. Frames are stamped
 * with it here so the in-memory model is a real discriminated union and panels
 * cannot read the wrong variant. Traces written before the type existed still
 * load, because the stamp comes from their own header.
 */
export function parseTrace(text: string): Trace {
  const lines = text.split("\n").filter((l) => l.trim().length > 0);
  if (lines.length === 0) {
    throw new Error("parseTrace: empty trace (no header line)");
  }
  const header = JSON.parse(lines[0]) as TraceHeader;
  const type = header.task?.type ?? "gridworld";

  const frames = lines.slice(1).map((l, i) => {
    const frame = JSON.parse(l) as Frame;
    const task = frame.task as unknown as Record<string, unknown>;
    if (task) {
      if (type === "cube" && "agent" in task) {
        throw new Error(
          `parseTrace: header declares task type "cube" but frame ${i} carries gridworld "agent"`,
        );
      }
      if (type === "gridworld" && "facelets" in task) {
        throw new Error(
          `parseTrace: header declares task type "gridworld" but frame ${i} carries cube "facelets"`,
        );
      }
      frame.task = { ...task, type } as Frame["task"];
    }
    return frame;
  });
  return { header, frames };
}
```

- [ ] **Step 5: Stamp in the store as well, so live frames are covered**

`parseTrace` covers file traces only. `WebSocketTraceSource.handleMessage` calls
`this.onFrame?.(msg.data as Frame)` on the raw payload (see
`dashboard/src/source/WebSocketTraceSource.ts:79`), so a live frame never passes through
`parseTrace` and would reach the panels unstamped. Both sources funnel through the store,
so the store is the one place that covers both. Stamping twice is idempotent.

Write the failing test first, appended to `dashboard/src/store/traceStore.test.ts`:

```ts
it("stamps loaded frames with the header task type", () => {
  const header = {
    schema_version: "1.1",
    brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
    task: { type: "cube", cube_n: 2, action_labels: ["U"] },
    regions: [], pathways: [],
  } as unknown as TraceHeader;
  const frame = { episode: 0, step: 0, t: 0, task: { facelets: [] } } as unknown as Frame;
  useTraceStore.getState().load(header, [frame]);
  expect(useTraceStore.getState().frames[0].task.type).toBe("cube");
});

it("stamps appended live frames with the header task type", () => {
  const header = {
    schema_version: "1.1",
    brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
    task: { type: "cube", cube_n: 2, action_labels: ["U"] },
    regions: [], pathways: [],
  } as unknown as TraceHeader;
  useTraceStore.getState().load(header, []);
  useTraceStore.getState().appendFrame(
    { episode: 0, step: 0, t: 0, task: { facelets: [] } } as unknown as Frame,
  );
  expect(useTraceStore.getState().frames[0].task.type).toBe("cube");
});
```

Run `cd dashboard; npm test -- traceStore` and confirm both FAIL (`type` is undefined).

Then in `dashboard/src/store/traceStore.ts`, add the helper and use it in both ingest paths:

```ts
/** Stamp a frame's task with the run's task type from the header.
 *
 * The wire format carries the type once, in the header. parseTrace stamps file
 * traces, but WebSocketTraceSource hands raw frames straight to appendFrame, so
 * the store is the only point both paths share. Idempotent.
 */
const stamp = (frame: Frame, header?: TraceHeader): Frame => {
  const type = header?.task?.type;
  if (!type || !frame?.task) return frame;
  return { ...frame, task: { ...frame.task, type } as Frame["task"] };
};
```

```ts
  load: (header, frames) =>
    set({
      header,
      frames: frames.map((f) => stamp(f, header)),
      T: header.brain.T, envStep: 0, winTi: 0, playing: false,
    }),

  appendFrame: (frame) =>
    set((s) => {
      const frames = [...s.frames, stamp(frame, s.header)];
      return { frames, envStep: frames.length - 1 }; // follow-live (unconditional for MVP)
    }),
```

Run `cd dashboard; npm test -- traceStore` again. Expected: PASS.

**This also means no existing test needs editing.** `TaskState.test.tsx` builds its frame
without a `type` field and loads it through the store, so it gets stamped `"gridworld"`
from its own header and keeps passing unchanged. If you find yourself editing an existing
assertion to make this task pass, stop and report instead.

- [ ] **Step 6: Run the dashboard tests**

Run: `cd dashboard; npm test`
Expected: PASS. Type errors will surface in `TaskState.tsx` and `SensoryGrid.tsx`; those are fixed in Tasks 6 and 7. If the suite blocks on them, complete Tasks 5 to 7 before re-running the full suite, but `npm test -- parseTrace` and `npm test -- traceStore` must pass now.

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/contract.ts dashboard/src/source/parseTrace.ts dashboard/src/source/parseTrace.test.ts dashboard/src/store/traceStore.ts dashboard/src/store/traceStore.test.ts
git commit -m "feat(dashboard): discriminate task state by type, stamped on ingest

parseTrace covers file traces; the store covers both file and live, because
WebSocketTraceSource hands raw frames to appendFrame without parsing."
```

---

### Task 5: The cube net map (pure)

**Files:**
- Create: `dashboard/src/panels/cubeNet.ts`, `dashboard/src/panels/cubeNet.test.ts`

**Interfaces:**
- Produces: `NET_ROWS = 6`, `NET_COLS = 8`, `cubeNetPosition(facelet: number): { row: number; col: number }`, `FACE_OF = ["U","R","F","D","L","B"]`.

Face order matches `neuromorphic.envs.cube`: face `f` occupies facelets `[4f, 4f+4)` with `U=0, R=1, F=2, D=3, L=4, B=5`. Within a face, facelet `i` sits at `(i >> 1, i & 1)`.

- [ ] **Step 1: Write the failing tests**

```ts
// dashboard/src/panels/cubeNet.test.ts
import { describe, expect, it } from "vitest";
import { cubeNetPosition, NET_COLS, NET_ROWS } from "./cubeNet";

describe("cubeNetPosition", () => {
  it("is a bijection over all 24 facelets", () => {
    const seen = new Set<string>();
    for (let f = 0; f < 24; f++) {
      const { row, col } = cubeNetPosition(f);
      expect(row).toBeGreaterThanOrEqual(0);
      expect(row).toBeLessThan(NET_ROWS);
      expect(col).toBeGreaterThanOrEqual(0);
      expect(col).toBeLessThan(NET_COLS);
      seen.add(`${row},${col}`);
    }
    expect(seen.size).toBe(24);
  });

  it("places U in the top band above F", () => {
    // U = facelets 0-3, F = facelets 8-11, F sits directly below U.
    for (let i = 0; i < 4; i++) {
      const u = cubeNetPosition(i);
      const f = cubeNetPosition(8 + i);
      expect(u.col).toBe(f.col);
      expect(f.row - u.row).toBe(2);
    }
  });

  it("orders the middle band L F R B left to right", () => {
    const col = (f: number) => cubeNetPosition(f).col;
    expect(col(16)).toBeLessThan(col(8));  // L before F
    expect(col(8)).toBeLessThan(col(4));   // F before R
    expect(col(4)).toBeLessThan(col(20));  // R before B
  });

  it("keeps every face a contiguous 2x2 block", () => {
    for (let face = 0; face < 6; face++) {
      const pos = [0, 1, 2, 3].map((i) => cubeNetPosition(face * 4 + i));
      const rows = new Set(pos.map((p) => p.row));
      const cols = new Set(pos.map((p) => p.col));
      expect(rows.size).toBe(2);
      expect(cols.size).toBe(2);
      expect(Math.max(...rows) - Math.min(...rows)).toBe(1);
      expect(Math.max(...cols) - Math.min(...cols)).toBe(1);
    }
  });

  it("rejects an out-of-range facelet", () => {
    expect(() => cubeNetPosition(-1)).toThrow();
    expect(() => cubeNetPosition(24)).toThrow();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard; npm test -- cubeNet`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```ts
// dashboard/src/panels/cubeNet.ts
/** Unfolded-net geometry for a 2x2 cube.
 *
 * Face order matches neuromorphic.envs.cube: face f owns facelets [4f, 4f+4),
 * with U=0, R=1, F=2, D=3, L=4, B=5. The net is
 *
 *         U
 *   L  F  R  B
 *         D
 */
export const NET_ROWS = 6;
export const NET_COLS = 8;

export const FACE_OF = ["U", "R", "F", "D", "L", "B"] as const;

/** Top-left (row, col) of each face, indexed by face number. */
const FACE_ORIGIN: ReadonlyArray<readonly [number, number]> = [
  [0, 2], // U
  [2, 4], // R
  [2, 2], // F
  [4, 2], // D
  [2, 0], // L
  [2, 6], // B
];

export function cubeNetPosition(facelet: number): { row: number; col: number } {
  if (!Number.isInteger(facelet) || facelet < 0 || facelet > 23) {
    throw new Error(`cubeNetPosition: facelet must be an integer 0-23, got ${facelet}`);
  }
  const face = facelet >> 2;
  const i = facelet & 3;
  const [r0, c0] = FACE_ORIGIN[face];
  return { row: r0 + (i >> 1), col: c0 + (i & 1) };
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard; npm test -- cubeNet`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/panels/cubeNet.ts dashboard/src/panels/cubeNet.test.ts
git commit -m "feat(dashboard): add unfolded-net geometry for the 2x2 cube"
```

---

### Task 6: The cube task panel

**Files:**
- Modify: `dashboard/src/panels/TaskState.tsx`, `dashboard/src/panels/TaskState.test.tsx`

**Interfaces:**
- Consumes: `cubeNetPosition`, `NET_ROWS`, `NET_COLS` from Task 5; `TaskState` union from Task 4.
- Produces: the `TaskState` panel renders `data-cube-net` with 24 `data-facelet` cells for cube traces and the existing `data-cell` grid for gridworld.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/src/panels/TaskState.test.tsx`. The existing file loads state with
`useTraceStore.getState().load(header, [frame])`; use the same call. Do NOT modify the
existing gridworld test: the store stamps its frame from its own header, so it must keep
passing untouched.

```tsx
const cubeHeader = {
  schema_version: "1.1",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "cube", cube_n: 2, action_labels: ["U", "U'", "R", "R'", "F", "F'"] },
  regions: [], pathways: [],
} as unknown as TraceHeader;

const cubeFrame = (distance: number | null) => ({
  episode: 0, step: 0, t: 0,
  task: {
    facelets: Array.from({ length: 24 }, (_, i) => i % 6),
    solved: false, distance, scramble_depth: 2,
    move: 3, move_label: "R'",
    action: 3, action_label: "R'",
    reward: -1, return: -3, terminated: false, truncated: false,
  },
  regions: {}, pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
}) as unknown as Frame;

describe("TaskState cube", () => {
  it("renders 24 facelets in a net for a cube trace", () => {
    useTraceStore.getState().load(cubeHeader, [cubeFrame(2)]);
    const { container } = render(<TaskState />);
    expect(container.querySelectorAll("[data-facelet]")).toHaveLength(24);
    expect(container.querySelector("[data-cube-net]")).toBeTruthy();
    expect(container.querySelector("[data-cell]")).toBeNull();
    expect(container.textContent).toContain("R'");
    expect(container.textContent).toContain("distance 2");
  });

  it("shows a dash rather than the string null when distance is absent", () => {
    useTraceStore.getState().load(cubeHeader, [cubeFrame(null)]);
    const { container } = render(<TaskState />);
    expect(container.textContent).toContain("distance -");
    expect(container.textContent).not.toContain("null");
  });

  it("still renders the gridworld grid when the header says gridworld", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<TaskState />);
    expect(container.querySelectorAll("[data-cell]")).toHaveLength(25);
    expect(container.querySelector("[data-cube-net]")).toBeNull();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard; npm test -- TaskState`
Expected: FAIL, no `[data-facelet]` elements.

- [ ] **Step 3: Split the panel**

Rewrite `TaskState.tsx` so the exported `TaskState` reads the frame, narrows on `frame.task.type`, and delegates. Keep the existing gridworld markup exactly as it is today inside `GridTaskView`. Add:

```tsx
const FACELET_COLOR = [
  "#f2f2f2", "#e04a2f", "#2f7de0", "#f2c14a", "#3fae6a", "#c94ad6",
];

function CubeTaskView({ task }: { task: CubeTask }) {
  const cells = Array.from({ length: NET_ROWS * NET_COLS }, () => -1);
  task.facelets.forEach((color, f) => {
    const { row, col } = cubeNetPosition(f);
    cells[row * NET_COLS + col] = color;
  });
  const held = new Set([12, 16, 21].map((f) => {
    const { row, col } = cubeNetPosition(f);
    return row * NET_COLS + col;
  }));

  return (
    <>
      <div
        data-cube-net
        style={{ display: "grid", gridTemplateColumns: `repeat(${NET_COLS}, 1fr)`, gap: 2, maxWidth: 260, margin: "0 auto" }}
      >
        {cells.map((color, idx) =>
          color < 0 ? (
            <div key={idx} style={{ aspectRatio: "1" }} />
          ) : (
            <div
              key={idx}
              data-facelet
              style={{
                aspectRatio: "1",
                borderRadius: 2,
                background: FACELET_COLOR[color] ?? "var(--edge)",
                boxShadow: held.has(idx) ? "inset 0 0 0 2px var(--text-faint)" : "none",
              }}
            />
          ),
        )}
      </div>
      <div style={{ font: "11px monospace", color: "var(--text-dim)", marginTop: 10, display: "flex", flexDirection: "column", gap: 3 }}>
        <div>
          move {task.move_label ?? "-"} · distance {task.distance ?? "-"} · depth {task.scramble_depth}
        </div>
        <div>solved {task.solved ? "yes" : "no"}</div>
      </div>
    </>
  );
}
```

The reward and return lines stay shared: lift them out of the gridworld branch so both views render them from `TaskCore`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard; npm test -- TaskState`
Expected: all pass, including the pre-existing gridworld cases unchanged.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/panels/TaskState.tsx dashboard/src/panels/TaskState.test.tsx
git commit -m "feat(dashboard): render the cube as an unfolded net in the task panel"
```

---

### Task 7: Cube sensory overlay

**Files:**
- Modify: `dashboard/src/hero/sensory.ts`, `dashboard/src/hero/sensory.test.ts`, `dashboard/src/hero/overlays/SensoryGrid.tsx`

**Interfaces:**
- Produces: `aggregateCubeFacelets(encoding): number[] | null` returning 24 argmax color indices, alongside the existing `aggregateSensoryGrid`.

- [ ] **Step 1: Write the failing test**

```ts
// append to dashboard/src/hero/sensory.test.ts
import { aggregateCubeFacelets } from "./sensory";

it("takes the argmax color per facelet", () => {
  // 24 facelets x 6 colors = 144. Facelet f is color f % 6, spiking twice.
  const row = new Array(144).fill(0);
  for (let f = 0; f < 24; f++) row[f * 6 + (f % 6)] = 1;
  const enc = { sensory_input: { spikes: [row, row], cube_n: 2, n_colors: 6, index: "" } };
  const got = aggregateCubeFacelets(enc as never);
  expect(got).toHaveLength(24);
  for (let f = 0; f < 24; f++) expect(got![f]).toBe(f % 6);
});

it("returns null for a gridworld encoding", () => {
  const enc = { sensory_input: { spikes: [[0, 1]], grid_n: 1, planes: [], index: "" } };
  expect(aggregateCubeFacelets(enc as never)).toBeNull();
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard; npm test -- sensory`
Expected: FAIL, `aggregateCubeFacelets` is not exported.

- [ ] **Step 3: Implement**

```ts
// dashboard/src/hero/sensory.ts (append)
/** Sum cube sensory spikes over the window; return the argmax color per facelet. */
export function aggregateCubeFacelets(encoding: Frame["encoding"]): number[] | null {
  const si = encoding?.sensory_input;
  if (!si || !("cube_n" in si)) return null;
  const nFacelets = 6 * si.cube_n * si.cube_n;
  const sums = Array.from({ length: nFacelets }, () => new Array(si.n_colors).fill(0));
  for (const row of si.spikes ?? []) {
    for (let f = 0; f < nFacelets; f++) {
      for (let c = 0; c < si.n_colors; c++) {
        sums[f][c] += row[f * si.n_colors + c] || 0;
      }
    }
  }
  return sums.map((counts) => {
    let best = 0;
    for (let c = 1; c < counts.length; c++) if (counts[c] > counts[best]) best = c;
    return best;
  });
}
```

Guard `aggregateSensoryGrid` the same way, returning `null` unless `"grid_n" in si`.

- [ ] **Step 4: Update `SensoryGrid.tsx`**

Keep the existing gridworld body exactly as it is, moved into a `GridSensoryView`, and add
the cube branch. The component returns `null` when neither aggregate matches, so an unknown
task type degrades instead of crashing.

```tsx
import { useTraceStore } from "../../store/traceStore";
import { cubeNetPosition, NET_COLS, NET_ROWS } from "../../panels/cubeNet";
import { aggregateCubeFacelets, aggregateSensoryGrid } from "../sensory";

const FACELET_COLOR = [
  "#f2f2f2", "#e04a2f", "#2f7de0", "#f2c14a", "#3fae6a", "#c94ad6",
];

const shellStyle: React.CSSProperties = {
  position: "absolute",
  top: 44,
  left: 18,
  padding: "10px 11px",
  borderRadius: 10,
  background: "var(--panel)",
  border: "1px solid var(--edge)",
  backdropFilter: "var(--blur)",
  pointerEvents: "none",
  zIndex: 10,
};

const captionStyle: React.CSSProperties = {
  font: "600 8px/1 'IBM Plex Mono', monospace",
  color: "var(--text-faint)",
  letterSpacing: ".12em",
  marginBottom: 7,
};

function CubeSensoryView({ colors, cubeN }: { colors: number[]; cubeN: number }) {
  const cells = Array.from({ length: NET_ROWS * NET_COLS }, () => -1);
  colors.forEach((color, f) => {
    const { row, col } = cubeNetPosition(f);
    cells[row * NET_COLS + col] = color;
  });
  return (
    <div data-sensory-cube style={shellStyle}>
      <div style={captionStyle}>
        SENSORY INPUT · CUBE {cubeN}x{cubeN}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${NET_COLS}, 10px)`, gap: 2 }}>
        {cells.map((color, idx) =>
          color < 0 ? (
            <div key={idx} style={{ width: 10, height: 10 }} />
          ) : (
            <div
              key={idx}
              data-sensory-facelet
              style={{ width: 10, height: 10, borderRadius: 2, background: FACELET_COLOR[color] ?? "var(--edge)" }}
            />
          ),
        )}
      </div>
    </div>
  );
}

export function SensoryGrid() {
  const header = useTraceStore((s) => s.header);
  const envStep = useTraceStore((s) => s.envStep);
  const frames = useTraceStore((s) => s.frames);
  if (!header) return null;
  const frame = frames[envStep];

  const cube = aggregateCubeFacelets(frame?.encoding);
  if (cube) {
    const si = frame!.encoding!.sensory_input;
    return <CubeSensoryView colors={cube} cubeN={"cube_n" in si ? si.cube_n : 2} />;
  }

  const agg = aggregateSensoryGrid(frame?.encoding);
  if (!agg) return null;
  const si = frame!.encoding!.sensory_input;
  const g = "grid_n" in si ? si.grid_n : 0;
  const cells = Array.from({ length: g * g }, (_, c) => c);
  // ...existing gridworld markup, unchanged, using `cells`, `agg.agentCell`, `agg.goalCell`
}
```

Preserve the existing gridworld JSX verbatim in place of the final comment. Do not
restyle it.

- [ ] **Step 5: Run the full dashboard suite**

Run: `cd dashboard; npm test`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/hero/sensory.ts dashboard/src/hero/sensory.test.ts dashboard/src/hero/overlays/SensoryGrid.tsx
git commit -m "feat(dashboard): aggregate cube facelet encoding for the sensory overlay"
```

---

### Task 8: End-to-end verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-29-dashboard-cube-support-design.md` (mark success criteria)

- [ ] **Step 1: Full python suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: 354 or more passed, including the slow BFS test.

- [ ] **Step 2: Full dashboard suite and typecheck**

Run: `cd dashboard; npm test; npm run build`
Expected: tests pass, `tsc -b` reports no errors. A type error here is the discriminated union doing its job; fix the call site, do not cast it away.

- [ ] **Step 3: Load both traces in the browser**

Run: `cd dashboard; npm run dev`
Check: the committed `week11_dashboard_trace.jsonl` still renders the gridworld grid, and `cube_trace.jsonl` renders the net. Scrub the cube trace and confirm each step permutes exactly one face.

- [ ] **Step 4: Record the outcome**

Tick the success criteria in the spec, noting any that were not met and why.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-29-dashboard-cube-support-design.md
git commit -m "docs(dashboard): mark cube support success criteria"
```

---

## Self-Review

**Spec coverage:** section 1 adapters (Task 1), section 2 schema/frame/runner and the `_config_hash` break (Task 2), section 3 recorder (Task 3), section 4 contract/parseTrace/cubeNet/TaskState/sensory (Tasks 4 to 7), section 5 error handling (Task 4 mismatch throw, Task 6 null distance dash, Task 7 null-degrade), section 6 tests (all tasks), success criteria (Task 8). No gaps.

**Known deviation from the spec:** the spec's section 6 lists `test_cube_net_reflects_a_known_move` as a dashboard test. The move permutations live in Python and are not available to vitest, so the equivalent check runs server-side as `test_cube_facelets_follow_the_applied_move` (Task 1) and `test_recorded_frames_carry_facelets_that_follow_the_moves` (Task 3), which are stronger because they check real recorded episodes. The TypeScript side keeps geometry checks the map can actually fail: bijection, contiguous faces, and band ordering.

**Type consistency:** `cubeNetPosition` / `NET_ROWS` / `NET_COLS` / `FACE_OF` are named identically in Tasks 5, 6 and 7. `adapter` is the keyword in `build_header`, `build_frame`, `record_episode` and `record_policy_episode`. `frame_task` takes the same keyword set in both adapters and at the single call site in `runner.py`.
