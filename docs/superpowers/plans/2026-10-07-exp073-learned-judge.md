# EXP-073 Learned Judge: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a "moves to solved" judge J by approximate value iteration (arm A: spiking encoder plus head; arm B: head only), and evaluate J-driven search against P3V, with every gate the spec defines.

**Architecture:** A new module `src/neuromorphic/training/value_iteration.py` holds the judge, the target computation, one training step, and a resumable training loop with a probe instrument. It never reads the BFS table; the probe and exclusion lists are passed in. `experiments/073_learned_judge/` builds the per-seed lists, runs pilot and full training, evaluates J through the existing look-ahead procedure's critic scorer (`critic = -J`), and aggregates. Tasks 1 to 5 are implementer tasks on the VPS; Task 6 is the controller's laptop checklist.

**Tech Stack:** Python 3.10 venv, torch CPU, snnTorch via `SensoryCortex`, pytest. No scipy.

**Spec:** `docs/superpowers/specs/2026-10-07-exp073-learned-judge-design.md`. Read it fully before any task, especially sections 3, 4 and 6.

## Global Constraints

- Work ONLY in the worktree `/root/projects/.wt/exp-073` on branch `exp-073-judge`. Prefix every python/pytest command with `PYTHONPATH=src` (the venv's editable install points at main's `src`).
- Always pass an explicit Bash `timeout` (default is 120 s). Run tests in the FOREGROUND; 600000 ms covers every command here. Never run the whole suite.
- No em-dashes. Plain commit messages: no `Co-Authored-By`, no "Generated with". Stage explicit paths only.
- Action width from `n_actions` arguments or `env.action_space.n`, never a literal 6.
- `value_iteration.py` must NEVER import or read `neuromorphic.envs.cube_distance`. The BFS table builds probe sets and exclusion lists in the experiment folder only, and those reach the module as plain lists.
- **Training builds gradients through the encoder in arm A.** Never route the training forward through `imagined_values`, `action_distribution`, or `Brain.step`: all wrap the brain in `torch.no_grad()`. EXP-047's first version trained nothing because of exactly this.
- Tests whose subject is gradient flow must build their inputs WITH a live graph (EXP-063: a `no_grad` fixture disarms the test).
- Existing tests must keep passing unmodified, including `tests/training/test_lookahead_golden.py`.
- Seeds 12 and 13 are pilot-only; seeds 0 to 11 are evaluation seeds.

---

## Step 0 (controller)

```bash
cd /root/projects/neuromorphic-nn-snn-research-project
git worktree add /root/projects/.wt/exp-073 -b exp-073-judge
ln -s /root/projects/neuromorphic-nn-snn-research-project/.venv /root/projects/.wt/exp-073/.venv
```

## File Structure

| file | responsibility |
|---|---|
| `src/neuromorphic/training/value_iteration.py` (create) | `Judge`, `random_walk_states`, `bellman_targets`, `train_step`, `spearman`, `probe_judge`, `train_judge` (resumable) |
| `tests/training/test_value_iteration.py` (create) | unit tests for all of the above |
| `experiments/073_learned_judge/cells.py` (create) | per-seed exclusion list, probe set, judge construction from E1, paths |
| `experiments/073_learned_judge/train.py` (create) | pilot and full training driver, one record and checkpoint set per (arm, seed) |
| `experiments/073_learned_judge/evaluate.py` (create) | J search cells (via critic scorer), P3V continuity and depth-11 cells, R3V depth-11, J and policy rank checks |
| `experiments/073_learned_judge/aggregate.py` (create) | Gates 0, T, E, R, 1; Claims 1 and 2; secondaries; exploratory depth 11 |
| `experiments/073_learned_judge/launch073.ps1` (create) | laptop phases |
| `tests/experiments/test_exp073.py` (create) | cells, drivers, aggregator |

---

### Task 1: The judge, targets and one training step

**Files:** Create `src/neuromorphic/training/value_iteration.py`; Test `tests/training/test_value_iteration.py`.

**Interfaces (Produces):**
- `class Judge(nn.Module)`: `__init__(self, encoder_fn, sensory: nn.Module, T: int, content: int, hidden: int = 128)`; `concept(self, states, generator) -> Tensor[B, content]` (encode, sensory, mean over T; NO `no_grad`); `forward(self, states, generator) -> Tensor[B]` (softplus head, `>= 0`).
- `judge_from_brain(brain, hidden=128) -> Judge` using `brain._encoder`, `copy.deepcopy(brain.sensory)`, `brain.T`, `brain.content`.
- `random_walk_states(n, max_len, rng: random.Random, n_actions, exclude: set, *, start=SOLVED, apply_fn=apply_move) -> list[tuple]`: each state is a walk of length `rng.randint(1, max_len)`; a state in `exclude` is discarded and redrawn.
- `bellman_targets(states, jt: Judge, n_actions, generator, draws: int = 1, *, apply_fn=apply_move, goal_fn=is_solved) -> Tensor[B]` under `torch.no_grad()`: `0` for a solved state, else `1 + min_c Jt(c)` with `Jt(c) = 0` for solved children; `Jt` averaged over `draws` Poisson draws; all children of the batch encoded in ONE call per draw.
- `train_step(judge, jt, optimizer, states, n_actions, generator, draws=1) -> float` (MSE loss).
- `make_optimizer(judge, arm: str) -> torch.optim.Adam`: arm `"A"`: head lr 1e-3 and sensory lr 1e-4 as two param groups; arm `"B"`: head params only, lr 1e-3, and sets `requires_grad=False` on every sensory parameter.
- `sync_target(judge) -> Judge`: `copy.deepcopy(judge)` with every parameter `requires_grad=False` (encoder AND head).

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for value iteration (EXP-073). Each docstring names the bug it catches."""

from __future__ import annotations

import inspect
import random

import pytest
import torch
import torch.nn as nn

from neuromorphic.envs.cube import SOLVED, N_ACTIONS, apply_move, inverse_action, is_solved
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.cube_baseline import CubeConfig, make_agent


def _judge(seed=0):
    torch.manual_seed(seed)
    return vi.judge_from_brain(make_agent(CubeConfig(arm="regionalized", seed=seed)))


def test_module_never_imports_the_bfs_provider():
    """Catches an answer leak at the source level."""
    src = inspect.getsource(vi)
    assert "cube_distance" not in src and "ExactBFSDistance" not in src


def test_random_walks_respect_length_and_exclusions():
    """Catches walks longer than max_len, length-0 walks, or an ignored exclusion list."""
    rng = random.Random(0)
    one_move = {apply_move(SOLVED, a) for a in range(N_ACTIONS)}
    states = vi.random_walk_states(200, 1, rng, N_ACTIONS, exclude=set())
    assert set(states) <= one_move
    banned = {apply_move(SOLVED, 0)}
    states = vi.random_walk_states(200, 1, random.Random(1), N_ACTIONS, exclude=banned)
    assert not (set(states) & banned) and len(states) == 200


class _ConstJudge(nn.Module):
    """Jt stub: value looked up per state (default 5.0)."""

    def __init__(self, values):
        super().__init__()
        self.values = {tuple(k): v for k, v in values.items()}

    def forward(self, states, generator):
        return torch.tensor([float(self.values.get(tuple(s), 5.0)) for s in states])


def test_bellman_target_is_zero_at_solved_and_one_next_to_it():
    """Catches a solved state getting 1 + min, or a solved child not counted as 0."""
    s1 = apply_move(SOLVED, 2)
    y = vi.bellman_targets([SOLVED, s1], _ConstJudge({}), N_ACTIONS, torch.Generator())
    assert y.tolist() == [0.0, 1.0]


def test_bellman_target_takes_one_plus_the_minimum_child():
    """Catches max instead of min, or a missing +1."""
    s = apply_move(apply_move(SOLVED, 0), 2)
    child = apply_move(s, 4)
    y = vi.bellman_targets([s], _ConstJudge({child: 2.0}), N_ACTIONS, torch.Generator())
    assert y.item() == pytest.approx(3.0)


def test_jt_is_a_frozen_copy_of_encoder_and_head():
    """Catches Jt sharing the live encoder (targets would move every update) or keeping grads."""
    judge = _judge()
    jt = vi.sync_target(judge)
    assert all(not p.requires_grad for p in jt.parameters())
    before = [p.detach().clone() for p in jt.parameters()]
    opt = vi.make_optimizer(judge, "A")
    states = vi.random_walk_states(16, 4, random.Random(0), N_ACTIONS, exclude=set())
    vi.train_step(judge, jt, opt, states, N_ACTIONS, torch.Generator().manual_seed(0))
    after = list(jt.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    moved = sum(float((p - q).abs().sum()) for p, q in
                zip(judge.sensory.parameters(), jt.sensory.parameters()))
    assert moved > 0.0


def test_arm_a_gradient_reaches_the_encoder_and_arm_b_leaves_it_untouched():
    """THE GATE E MECHANISM. Catches the EXP-047 failure (encoder silently frozen in arm A) and a
    leaky control (encoder moving in arm B). Inputs carry a live graph: no no_grad fixture."""
    states = vi.random_walk_states(16, 4, random.Random(0), N_ACTIONS, exclude=set())
    a = _judge()
    jt_a = vi.sync_target(a)
    w0 = a.sensory.fc1.weight.detach().clone()
    vi.train_step(a, jt_a, vi.make_optimizer(a, "A"), states, N_ACTIONS,
                  torch.Generator().manual_seed(0))
    assert not torch.equal(a.sensory.fc1.weight, w0)

    b = _judge()
    jt_b = vi.sync_target(b)
    w0 = b.sensory.fc1.weight.detach().clone()
    h0 = b.head[0].weight.detach().clone()
    vi.train_step(b, jt_b, vi.make_optimizer(b, "B"), states, N_ACTIONS,
                  torch.Generator().manual_seed(0))
    assert torch.equal(b.sensory.fc1.weight, w0)
    assert not torch.equal(b.head[0].weight, h0)


def test_judge_output_is_nonnegative_and_batched():
    """Catches a missing softplus or a shape bug in the batched forward."""
    states = vi.random_walk_states(8, 6, random.Random(2), N_ACTIONS, exclude=set())
    v = _judge()(states, torch.Generator().manual_seed(0))
    assert v.shape == (8,) and bool((v >= 0).all())


def test_targets_use_one_batched_call_per_draw():
    """Catches per-child encoding (n times slower) by counting judge calls."""
    calls = []

    class Spy(_ConstJudge):
        def forward(self, states, generator):
            calls.append(len(states))
            return super().forward(states, generator)

    states = vi.random_walk_states(5, 4, random.Random(3), N_ACTIONS, exclude=set())
    vi.bellman_targets(states, Spy({}), N_ACTIONS, torch.Generator(), draws=2)
    assert calls == [5 * N_ACTIONS, 5 * N_ACTIONS]
```

- [ ] **Step 2: Run to verify failure** (`cd /root/projects/.wt/exp-073 && PYTHONPATH=src .venv/bin/python -m pytest tests/training/test_value_iteration.py -q`, timeout 600000). Expected: ImportError.

- [ ] **Step 3: Implement**

```python
"""Approximate value iteration for a "moves to solved" judge (EXP-073).

The judge learns from its OWN look-ahead through the simulator: a state's target is one move plus
the best child's value under a frozen copy of the judge. No distance table is read here; probe and
exclusion lists arrive as plain arguments. Training builds a graph through the spiking encoder in
arm A, so nothing in this module may route through Brain.step or the look-ahead helpers, which run
under no_grad.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromorphic.envs.cube import SOLVED, apply_move, is_solved


class Judge(nn.Module):
    def __init__(self, encoder_fn, sensory: nn.Module, T: int, content: int, hidden: int = 128):
        super().__init__()
        self.encoder_fn = encoder_fn
        self.sensory = sensory
        self.T = T
        self.head = nn.Sequential(nn.Linear(content, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def concept(self, states, generator):
        obs = torch.as_tensor(np.array(states), dtype=torch.long)
        spikes = self.encoder_fn(obs, T=self.T, generator=generator)
        return self.sensory(spikes).mean(dim=0)

    def forward(self, states, generator):
        return F.softplus(self.head(self.concept(states, generator))).squeeze(-1)


def judge_from_brain(brain, hidden: int = 128) -> Judge:
    return Judge(brain._encoder, copy.deepcopy(brain.sensory), brain.T, brain.content, hidden)


def random_walk_states(n, max_len, rng, n_actions, exclude, *, start=SOLVED,
                       apply_fn=apply_move):
    out = []
    while len(out) < n:
        s = start
        for _ in range(rng.randint(1, max_len)):
            s = apply_fn(s, rng.randrange(n_actions))
        if s not in exclude:
            out.append(s)
    return out


def bellman_targets(states, jt, n_actions, generator, draws=1, *, apply_fn=apply_move,
                    goal_fn=is_solved):
    with torch.no_grad():
        children = [apply_fn(s, a) for s in states for a in range(n_actions)]
        vals = torch.stack([jt(children, generator) for _ in range(draws)]).mean(dim=0)
        solved_child = torch.tensor([goal_fn(c) for c in children])
        vals = torch.where(solved_child, torch.zeros_like(vals), vals)
        best = vals.view(len(states), n_actions).min(dim=1).values
        y = 1.0 + best
        solved = torch.tensor([goal_fn(s) for s in states])
        return torch.where(solved, torch.zeros_like(y), y)


def sync_target(judge):
    jt = copy.deepcopy(judge)
    for p in jt.parameters():
        p.requires_grad_(False)
    return jt


def make_optimizer(judge, arm):
    if arm == "A":
        return torch.optim.Adam([{"params": judge.head.parameters(), "lr": 1e-3},
                                 {"params": judge.sensory.parameters(), "lr": 1e-4}])
    if arm == "B":
        for p in judge.sensory.parameters():
            p.requires_grad_(False)
        return torch.optim.Adam(judge.head.parameters(), lr=1e-3)
    raise ValueError(f"unknown arm {arm!r}")


def train_step(judge, jt, optimizer, states, n_actions, generator, draws=1):
    y = bellman_targets(states, jt, n_actions, generator, draws)
    loss = F.mse_loss(judge(states, generator), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss)
```

Verify before running: `SensoryCortex` exposes `fc1` (it does: `self.fc1` in `src/neuromorphic/regions/sensory_cortex.py`).

- [ ] **Step 4: Verify pass.** Same command.

- [ ] **Step 5: Mutation check** (revert each with `git checkout -- src/neuromorphic/training/value_iteration.py`, confirm clean; commit the real implementation FIRST so a revert cannot discard it):

| mutation | test that must fail |
|---|---|
| `concept` wrapped in `with torch.no_grad():` | `test_arm_a_gradient_reaches_the_encoder_and_arm_b_leaves_it_untouched` |
| `sync_target` returns `judge` itself | `test_jt_is_a_frozen_copy_of_encoder_and_head` |
| `sync_target` deep-copies only `judge.head` onto a shallow copy sharing `sensory` | `test_jt_is_a_frozen_copy_of_encoder_and_head` |
| `.min(dim=1)` becomes `.max(dim=1)` | `test_bellman_target_takes_one_plus_the_minimum_child` |
| `make_optimizer("B")` keeps sensory `requires_grad=True` and adds its params | the gradient test's arm B half |
| the solved-child override removed | `test_bellman_target_is_zero_at_solved_and_one_next_to_it` |

Strengthen any survivor; record outcomes in the commit body.

- [ ] **Step 6: Commit** `src/neuromorphic/training/value_iteration.py tests/training/test_value_iteration.py` with message "EXP-073: the judge, Bellman targets and a training step".

---

### Task 2: The resumable training loop and the probe instrument

**Files:** Modify `value_iteration.py`; Test `tests/training/test_value_iteration.py` (append).

**Interfaces (Produces):**
- `spearman(x: list[float], y: list[float]) -> float` (average ranks on ties, Pearson of ranks).
- `probe_judge(judge, probe: list[tuple[state, int]], generator) -> dict` with `spearman_all`, `spearman_7_11` (only entries with distance 7 to 11), `mean_j_by_distance` (dict of str(distance) -> mean J), `monotone_7_11` (bool: strictly increasing means 7..11 where present).
- `train_judge(judge, arm, *, n_updates, batch, max_len, n_actions, exclude, probe, seed, sync_every, probe_every, draws, ckpt_dir: Path, log=print) -> dict`: deterministic given `seed` (one `random.Random(seed)` for walks, one `torch.Generator().manual_seed(seed)` for encoding); syncs `Jt` every `sync_every` updates (and at update 0); probes and banks a checkpoint (`judge.pt`, `jt.pt`, `opt.pt`, `state.json` holding `update`, the Python RNG state, the torch generator state, and the probe history) every `probe_every` updates and at the end; RESUMES from `ckpt_dir` if `state.json` exists; returns `{"updates", "wall_s", "loss_last", "probes": [...], "final_probe": {...}}`.

- [ ] **Step 1: Failing tests** (append)

```python
from pathlib import Path


def test_spearman_known_values():
    """Catches Pearson-on-values instead of ranks, and wrong tie handling."""
    assert vi.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert vi.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert vi.spearman([1, 2, 3, 4], [1, 4, 9, 1000]) == pytest.approx(1.0)
    assert vi.spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.8944, abs=1e-4)


def test_probe_restricts_the_gate_quantity_to_distances_7_to_11():
    """Catches Gate T computed over all distances (dominated by the easy shallow end)."""

    class Fake(nn.Module):
        def forward(self, states, generator):
            # perfect below 7, inverted from 7 to 11
            return torch.tensor([float(s[0]) for s in states])

    probe = [((d if d < 7 else 18 - d,), d) for d in range(1, 12) for _ in range(3)]
    out = vi.probe_judge(Fake(), probe, torch.Generator())
    assert out["spearman_all"] > 0.5
    assert out["spearman_7_11"] < -0.9
    assert out["monotone_7_11"] is False


def _tiny(tmp_path, n_updates, seed=0):
    judge = _judge(seed)
    probe = [(apply_move(SOLVED, a), 1) for a in range(3)]
    return vi.train_judge(judge, "A", n_updates=n_updates, batch=8, max_len=3,
                          n_actions=N_ACTIONS, exclude=set(), probe=probe, seed=seed,
                          sync_every=2, probe_every=2, draws=1, ckpt_dir=tmp_path,
                          log=lambda *_: None), judge


def test_training_is_deterministic(tmp_path):
    """Catches unseeded randomness (global torch RNG, Python's random module)."""
    r1, j1 = _tiny(tmp_path / "a", 4)
    r2, j2 = _tiny(tmp_path / "b", 4)
    assert r1["loss_last"] == r2["loss_last"]
    assert all(torch.equal(p, q) for p, q in zip(j1.parameters(), j2.parameters()))


def test_resume_equals_an_uninterrupted_run(tmp_path):
    """Catches a resume that restarts the RNGs or the target net (a Windows Update restart
    would then silently change the run)."""
    full, jf = _tiny(tmp_path / "full", 4)
    _tiny(tmp_path / "split", 2)
    resumed, jr = _tiny(tmp_path / "split", 4)
    assert resumed["loss_last"] == full["loss_last"]
    assert all(torch.equal(p, q) for p, q in zip(jf.parameters(), jr.parameters()))
```

- [ ] **Step 2: Verify failure. Step 3: Implement** `spearman`, `probe_judge` (encode all probe states in ONE call under `torch.no_grad()`), and `train_judge` per the interface. For resume, when `state.json` exists, load `judge.pt` into `judge`, `jt.pt` into a fresh `sync_target(judge)`, `opt.pt` into a fresh `make_optimizer(judge, arm)`, `random.Random().setstate(...)` from the JSON (convert the list back to the tuple shape `getstate()` returns), and `generator.set_state(torch.tensor(..., dtype=torch.uint8))`. Save the torch generator state as a list of ints.
- [ ] **Step 4: Verify pass.**
- [ ] **Step 5: Mutation check**

| mutation | test that must fail |
|---|---|
| resume skips restoring the Python RNG | `test_resume_equals_an_uninterrupted_run` |
| resume re-syncs `Jt` from `judge` instead of loading `jt.pt` | `test_resume_equals_an_uninterrupted_run` |
| `random_walk_states` called with the module-level `random` instead of the seeded `Random` | `test_training_is_deterministic` |
| `spearman_7_11` computed over all entries | `test_probe_restricts_the_gate_quantity_to_distances_7_to_11` |

- [ ] **Step 6: Commit** with message "EXP-073: resumable value-iteration loop and the probe instrument".

---

### Task 3: Experiment cells and the training driver

**Files:** Create `experiments/073_learned_judge/cells.py`, `experiments/073_learned_judge/train.py`, `experiments/073_learned_judge/outputs/.gitkeep`; Test `tests/experiments/test_exp073.py`.

**Interfaces (Produces, `cells.py`):**
- `EVAL_SEEDS = tuple(range(12))`, `PILOT_SEEDS = (12, 13)`, `EVAL_DEPTHS = (7, 8, 9, 11)`, `ARMS_TRAIN = ("A", "B")`.
- `heldout_states(depth, seed, provider) -> list`: for depths 7 to 9, exactly EXP-070's `load_cell` split (`split_shell(shell_states(provider, depth), depth, seed=resolve_seed(published_config(depth, seed), "split"), heldout_cap=200, heldout_frac=0.25)[1]`); for depth 11 the same rule with `seed=seed`.
- `exclusion_set(seed, provider) -> set`: union of `heldout_states(d, seed, provider)` for `d` in `EVAL_DEPTHS`.
- `probe_set(seed, provider, per_distance=50) -> list[tuple[state, int]]`: for distance 1 to 11, the first `per_distance` states of `shell_states(provider, d)` after a `random.Random(10_000 + seed).shuffle`, skipping any state in `exclusion_set(seed, provider)`.
- `make_judge(seed) -> Judge`: `judge_from_brain(make_agent(published_config(7, seed)))` (EXP-070's config loads that seed's E1 encoder); `torch.manual_seed(seed)` immediately before construction so the head init is seeded.
- `ckpt_dir(arm, seed) -> Path` = `HERE / "outputs" / f"judge_{arm}_s{seed}"`; `record_name(arm, seed) -> str` = `f"exp073_train_{arm}_s{seed}.json"`.

`train.py`: `run(arm, seed, n_updates, batch, sync_every, probe_every, draws, out_dir) -> dict` builds the full BFS table once per process (`ExactBFSDistance(max_depth=None)`, about 65 s), the exclusion set and probe set, then `train_judge(...)`, and writes the record (all of `train_judge`'s return plus arm, seed, settings, `git_commit`, and the E1 drift `encoder_drift` = L2 norm of `judge.sensory` parameters minus the loaded E1 state dict). CLI: `--arms`, `--seeds`, `--n-updates`, `--batch`, `--sync-every`, `--probe-every`, `--draws`, `--workers`, `--pilot` (refuses any seed not in `PILOT_SEEDS`), and without `--pilot` refuses any seed not in `EVAL_SEEDS`.

- [ ] **Step 1: Failing tests**

```python
"""EXP-073: cells, training driver, evaluation and aggregator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube_distance import ExactBFSDistance

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "073_learned_judge"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp073_cells", "cells.py")
train = _load("exp073_train", "train.py")


@pytest.fixture(scope="module")
def provider():
    return ExactBFSDistance(max_depth=None)


def test_depth_7_to_9_heldout_sets_match_exp070(provider):
    """Catches evaluating or excluding a different split than the one every paired reference
    was measured on."""
    for d in (7, 8, 9):
        states = cells.c70.load_cell(d, 0)[2]
        assert cells.heldout_states(d, 0, provider) == states


def test_probe_and_training_never_touch_an_evaluation_state(provider):
    """Catches an evaluated position leaking into the probe (and so into Gate T)."""
    excl = cells.exclusion_set(0, provider)
    probe = cells.probe_set(0, provider, per_distance=5)
    assert len(excl) == 800
    assert not ({s for s, _ in probe} & excl)
    assert sorted({d for _, d in probe}) == list(range(1, 12))


def test_driver_refuses_seed_misuse():
    """Catches a pilot run on an evaluation seed, or an evaluation run on a pilot seed."""
    with pytest.raises(SystemExit):
        train.check_seeds([0], pilot=True)
    with pytest.raises(SystemExit):
        train.check_seeds([12], pilot=False)
    train.check_seeds([12, 13], pilot=True)
    train.check_seeds(list(range(12)), pilot=False)


def test_make_judge_starts_from_the_seeds_e1_encoder():
    """Catches a judge built on a random encoder (it would make arm A vs B meaningless)."""
    j = cells.make_judge(3)
    e1 = torch.load(cells.c70.published_config(7, 3).encoder_state_path, map_location="cpu")
    for k, v in j.sensory.state_dict().items():
        assert torch.equal(v, e1[k]), k
```

- [ ] **Steps 2 to 4:** verify failure, implement `cells.py` and `train.py` (add `check_seeds(seeds, pilot)` to `train.py`), verify pass. Then a smoke: `PYTHONPATH=src .venv/bin/python experiments/073_learned_judge/train.py --pilot --arms A --seeds 12 --n-updates 4 --batch 32 --sync-every 2 --probe-every 2 --workers 1 --out-dir /root/scratch/exp073-smoke` (timeout 600000). Paste the record into the report, including seconds per update. Do NOT run longer training; the pilot is the controller's.
- [ ] **Step 5: Mutation check:** probe built without the exclusion skip (test 2 must fail); `make_judge` without the E1 load (test 4 must fail).
- [ ] **Step 6: Commit** "EXP-073: cells and the training driver".

---

### Task 4: Evaluation through the existing search

**Files:** Create `experiments/073_learned_judge/evaluate.py`; Test `tests/experiments/test_exp073.py` (append).

**Interfaces (Produces):**
- `class NegJ(nn.Module)`: wraps a trained `Judge.head`; `forward(concept) -> -softplus(head(concept))` shaped `[B, 1]`, so the critic scorer's argmax picks the LOWEST J.
- `eval_agent(arm, seed) -> (agent, head, critic)`: the policy agent from EXP-070's `load_cell(depth, seed)`, with `agent.sensory` loaded from the trained judge's `sensory` state (arm A changes it; arm B leaves E1), the policy head unchanged (stream discipline only), and `critic = NegJ(trained head)`.
- `ARMS_EVAL`: `J1V-A, J1V-B, J3V-A, J3V-B` (mode `"C"`, `k` 1 or 3, `no_revisit=True`), `P3V` (mode `"P"`, `k` 3, `no_revisit=True`, the untouched policy agent), `R3V`.
- `run_cell(arm, depth, seed, out_dir, limit_states=None) -> dict` writing `exp073_{arm}_d{depth}_s{seed}.json`; P3V and R3V use the plain EXP-070 agent and head.
- `rank_cell(kind, depth, seed, out_dir) -> dict` writing `exp073_rank_{kind}_d{depth}_s{seed}.json` with per-seed mean `hit` and `chance` for top-leaf-closer-than-root at `k = 3`: `kind` in `("J-A", "J-B", "P")`. For J kinds the top leaf is the lowest J; for `"P"` it is the best summed policy log-probability sequence (reuse `lookahead.sequence_scores`). BFS (`max_depth = depth + 3`) is the yardstick here only.

- [ ] **Step 1: Failing tests** (append)

```python
evaluate = _load("exp073_evaluate", "evaluate.py")


def test_negj_makes_the_critic_scorer_pick_the_lowest_j():
    """Catches J used with the wrong sign (the search would head AWAY from solved)."""
    head = torch.nn.Sequential(torch.nn.Linear(2, 1))
    with torch.no_grad():
        head[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        head[0].bias.zero_()
    neg = evaluate.NegJ(head)
    concepts = torch.tensor([[5.0, 0.0], [0.1, 0.0], [3.0, 0.0]])
    assert int(neg(concepts).squeeze(-1).argmax()) == 1


def test_eval_arms_are_exactly_the_spec_list():
    assert [a for a in evaluate.ARMS_EVAL] == ["J1V-A", "J1V-B", "J3V-A", "J3V-B", "P3V", "R3V"]


def test_p3v_cell_reproduces_exp072_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH: P3V through this harness on 3 states must equal a direct
    evaluate_lookahead of EXP-072's configuration on the same states."""
    from neuromorphic.training.lookahead import evaluate_lookahead
    rec = evaluate.run_cell("P3V", 8, 0, tmp_path, limit_states=3)
    agent, head, states, ts = cells.c70.load_cell(8, 0)
    ref = evaluate_lookahead(agent, head, states[:3], depth=8, mode="P", k=3,
                             generator=torch.Generator().manual_seed(ts), rng_seed=ts,
                             imag_seed=ts, no_revisit=True)
    for f in ("solved", "success_rate", "mean_steps", "eval_revisit_rate"):
        assert rec[f] == ref[f], f
```

- [ ] **Steps 2 to 4:** verify failure, implement, verify pass. `run_cell` for J arms must raise `SystemExit` naming the missing checkpoint if `cells.ckpt_dir(arm, seed) / "judge.pt"` does not exist.
- [ ] **Step 5: Mutation check:** `NegJ` returns `+softplus` (test 1 fails).
- [ ] **Step 6: Commit** "EXP-073: evaluation through the look-ahead critic scorer, and the leaf-level rank checks".

---

### Task 5: Aggregator and launcher

**Files:** Create `experiments/073_learned_judge/aggregate.py`, `experiments/073_learned_judge/launch073.ps1`; Test `tests/experiments/test_exp073.py` (append).

**Interfaces (Produces, `aggregate.py`):** reuse EXP-071's `claim_verdict`, `one_sided_p`, `gate0b_verdict` via importlib. Constants set ONLY by the controller's dated amendments: `GATE_T_THRESHOLD: dict | None = None` (per arm), and none other. Functions:
- `gate_t_verdict(final_probes: list[dict], threshold: float) -> bool`: mean `spearman_7_11` over seeds `>= threshold` AND the seed-mean of `mean_j_by_distance` strictly increasing over 7..11.
- `gate_e_verdict(drift_a: list[float], drift_b: list[float]) -> bool`: every A drift `> 0` and every B drift `== 0.0`.
- `gate_r_verdict(rank_rows) -> bool`: exact one-sided p `< 0.05` and mean `hit - chance > 0`.
- `primary_verdicts(rates, gates) -> dict`: Claim 1 `("J3V-A", 9)` vs `("P3V", 9)`, Claim 2 `("J3V-A", 9)` vs `("J3V-B", 9)`; VOID per spec section 7.
- `main()` reads P3V and R3V at depths 7 to 9 from `experiments/071_critic_and_no_revisit/outputs` (depth 7) and `experiments/072_p3v_frontier/outputs` (depths 8, 9), and prints training gates, rank gates (with the policy reference), success tables for depths 7, 8, 9 and the exploratory depth 11, both primaries, the secondaries and the exploratory block, refusing claims if any required constant is `None`.

`launch073.ps1` phases (copy `launch072.ps1`'s checks, keep every warning comment): `check`, `pilot` (`train.py --pilot --arms A B --seeds 12 13 --workers 4` with the spec defaults and the pilot update count set inside the file), `train` (gated on `GATE_T_THRESHOLD` not being `None`: `train.py --arms A B --workers 20 --resume` over seeds 0 to 11), `rank` (J-A, J-B, P at depths 7, 8, 9), `cont` (P3V seed 0, depths 8 and 9), `eval` (the four J arms at depths 7, 8 and 9, plus J3V-A, J3V-B, P3V and R3V at depth 11; P3V and R3V at depths 7 to 9 are NOT re-run, they come from EXP-071 and EXP-072's committed records, re-checked by `cont`), `det` (J3V-A and J3V-B, seed 0, depth 9, into `C:\Users\mlgbr\exp073-det`). Check every gate read's `$LASTEXITCODE` and output format (the EXP-071 fix).

- [ ] **Step 1: Failing tests** (append)

```python
agg = _load("exp073_aggregate", "aggregate.py")


def test_gate_t_requires_monotone_means_not_just_correlation():
    """Catches a judge flat past distance 9 passing on correlation alone."""
    good = {"spearman_7_11": 0.6,
            "mean_j_by_distance": {"7": 6.0, "8": 7.0, "9": 8.0, "10": 9.0, "11": 9.5}}
    flat = {"spearman_7_11": 0.6,
            "mean_j_by_distance": {"7": 6.0, "8": 7.0, "9": 8.0, "10": 8.0, "11": 8.0}}
    assert agg.gate_t_verdict([good] * 12, 0.3) is True
    assert agg.gate_t_verdict([flat] * 12, 0.3) is False
    assert agg.gate_t_verdict([good] * 12, 0.7) is False


def test_gate_e_catches_a_frozen_arm_a_and_a_leaky_arm_b():
    """THE EXP-047 TRAP as a gate."""
    assert agg.gate_e_verdict([0.5] * 12, [0.0] * 12) is True
    assert agg.gate_e_verdict([0.5] * 11 + [0.0], [0.0] * 12) is False
    assert agg.gate_e_verdict([0.5] * 12, [0.0] * 11 + [1e-9]) is False


def test_claims_compare_the_registered_arms_at_depth_9():
    """Catches Claim 1 against J3V-B or Claim 2 against P3V, or the wrong depth."""
    r = lambda v: {s: v for s in range(12)}
    rates = {("J3V-A", 9): r(0.10), ("P3V", 9): r(0.06), ("J3V-B", 9): r(0.03)}
    out = agg.primary_verdicts(rates, {"ok": True})
    assert out["claim1"][4] == pytest.approx(0.06)
    assert out["claim2"][4] == pytest.approx(0.03)
```

Adjust `primary_verdicts`'s gate argument to whatever structure you implement, but keep the comparator assertions exactly.

- [ ] **Steps 2 to 5:** verify failure, implement, verify pass, mutation check (swap the claim comparators; drop the monotone condition from Gate T; make Gate E accept B drift `< 1e-6`). Commit "EXP-073: aggregator and launcher".

---

### Task 6 (CONTROLLER): verify, merge, pilot, amend, train, gate, evaluate

- [ ] **6.1** Chunked suite per CLAUDE.md with `PYTHONPATH=src`; `--collect-only` count into CLAUDE.md; merge `--no-ff` as `Merge exp-073-judge: <summary>`; push; remove the worktree and branch; sync the laptop; `-Phase check`.
- [ ] **6.2 Pilot:** `-Phase pilot`. Read the probe histories: updates per second, `spearman_7_11` and mean J per distance over time, and the min-bias symptom (means compressing past distance 8).
- [ ] **6.3 Dated amendment, BEFORE any seed 0-11 trains:** `batch`, `sync_every`, `draws`, `n_updates` (sized so 24 runs fit in about a day at 20 workers, from the pilot's measured rate), and `GATE_T_THRESHOLD` per arm = max(0.30, half the arm's pilot mean final `spearman_7_11`). Set the constant in `aggregate.py` and the training settings in `launch073.ps1` in the same commit.
- [ ] **6.4 Train:** `-Phase train`; poll checkpoint `state.json` counts rather than the ssh callback; resume with the same phase after any restart.
- [ ] **6.5 Gates before evaluation:** fetch the 24 training records; compute Gates T and E; run `-Phase rank` (J-A, J-B and the policy reference at depths 7 to 9) and compute Gate R; commit all of it, dated, before `-Phase eval`.
- [ ] **6.6 Evaluate:** `-Phase cont` (Gate 0(b) against EXP-072), then `-Phase eval`, then `-Phase det`.
- [ ] **6.7** Aggregate, force-add records and the 24 trained judges, `RESULTS.md`, roadmap outcome, handoff, docs-session message, `secretary log`.
