# Grid-World Generalization Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling to test whether the v1 grid-world agent (frozen brain + linear head) learned to navigate or memorized the path to (4,4), by training on random goals from a train set and evaluating on held-out goals, then run the experiment.

**Architecture:** Extend `GridWorldEnv` with random-goal sampling and optional potential-based reward shaping (both backward-compatible, default off). Add a `neuromorphic.training.generalization` module with the goal split, a greedy evaluator, and an importable `run_generalization` orchestrator that trains, logs a per-episode CSV, evaluates train-vs-held-out, and writes a summary JSON. A thin `experiments/024_grid_generalization/run.py` is the CLI. The brain stays frozen throughout; only the existing linear head trains.

**Tech Stack:** Python 3.11, PyTorch, Gymnasium, pytest. Spiking brain under `neuromorphic.*`.

## Global Constraints

- **The brain stays frozen.** Do not unfreeze any region or change `recall=False`. Only the existing `nn.Linear` head trains (ADR-0001 Amendment 1). This experiment changes the env and adds tooling, not the learning architecture.
- **Backward compatibility:** existing fixed-goal `GridWorldEnv()` construction and behavior must be byte-for-byte unchanged when the new options are not passed (no `goals`, `reward_shaping=False`).
- **Potential-based shaping** uses `Phi(s) = -manhattan(agent, goal)`; per-step shaped term is `shaping_gamma * Phi(s') - Phi(s)`. `shaping_gamma` defaults to `1.0` (a Manhattan progress reward); theory-exact invariance with the trainer's `gamma=0.99` is deferred and out of scope for this study.
- **Determinism:** goal sampling and the goal split must be deterministic for a given seed (use a dedicated `random.Random`, independent of Gym's `np_random`).
- **Goal split sizes:** 5x5 grid, start (0,0) → 24 candidate goal cells (all but start) → default 18 train / 6 held-out, disjoint.
- **Run commands from the repo root with the venv Python:** `.venv/Scripts/python.exe -m pytest ...` and `.venv/Scripts/python.exe experiments/024_grid_generalization/run.py ...`.
- **Commit style:** plain present-tense messages (`feat:`/`test:`/`docs:`), no `Co-Authored-By`, no "Generated with" trailer, no em-dashes.
- **Outputs** go under `outputs/` (a tracked directory).

---

## File Structure

```
src/neuromorphic/envs/gridworld.py            # MODIFY: manhattan(), random-goal sampling, potential-based shaping
src/neuromorphic/training/reinforce.py        # MODIFY: train_episode returns mean_entropy
src/neuromorphic/training/generalization.py   # NEW: split_goals, optimality, EvalResult, evaluate, GenConfig, run_generalization
experiments/024_grid_generalization/run.py    # NEW: thin CLI wrapper around run_generalization
tests/envs/__init__.py                        # NEW (empty, house style)
tests/envs/test_gridworld.py                  # NEW
tests/training/test_generalization.py         # NEW
tests/training/test_reinforce.py              # MODIFY: add entropy assertion
```

Note: `tests/envs/` is a new package named `envs`; it does not shadow anything (the source is `neuromorphic.envs`, not a top-level `envs`), so an `__init__.py` here is safe.

---

### Task 1: Env manhattan helper + random-goal sampling

**Files:**
- Modify: `src/neuromorphic/envs/gridworld.py`
- Create: `tests/envs/__init__.py` (empty), `tests/envs/test_gridworld.py`

**Interfaces:**
- Produces:
  - `manhattan(a, b) -> int` (module-level in `gridworld.py`): `abs(a0-b0) + abs(a1-b1)`.
  - `GridWorldEnv(..., goals: Sequence[tuple[int,int]] | None = None, goal_seed: int | None = None)`: when `goals` is given, `reset()` samples `self.goal` uniformly from `goals` using a dedicated `random.Random(goal_seed)`; otherwise the fixed `goal` is used unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/envs/__init__.py` (empty file). Create `tests/envs/test_gridworld.py`:

```python
from neuromorphic.envs.gridworld import GridWorldEnv, manhattan


def test_manhattan():
    assert manhattan((0, 0), (2, 3)) == 5
    assert manhattan((4, 4), (4, 4)) == 0


def test_fixed_goal_backward_compat():
    env = GridWorldEnv()
    obs, _ = env.reset()
    assert tuple(obs) == (0, 0, 4, 4)
    # action 1 = right; reward is the plain step penalty, goal unchanged
    obs, r, term, trunc, _ = env.step(1)
    assert tuple(obs) == (1, 0, 4, 4)
    assert r == -1.0 and term is False
    assert tuple(env.goal) == (4, 4)


def test_random_goal_sampled_from_set():
    goals = [(1, 2), (3, 4), (0, 1)]
    env = GridWorldEnv(goals=goals, goal_seed=0)
    seen = set()
    for _ in range(30):
        env.reset()
        g = tuple(env.goal)
        assert g in goals
        seen.add(g)
    assert len(seen) > 1  # actually samples more than one over 30 resets


def test_random_goal_deterministic_by_seed():
    goals = [(1, 2), (3, 4), (0, 1), (2, 0)]
    a = GridWorldEnv(goals=goals, goal_seed=7)
    b = GridWorldEnv(goals=goals, goal_seed=7)
    seq_a = [tuple((a.reset(), a.goal)[1]) for _ in range(10)]
    seq_b = [tuple((b.reset(), b.goal)[1]) for _ in range(10)]
    assert seq_a == seq_b
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/envs/test_gridworld.py -v`
Expected: FAIL (`manhattan` and the `goals`/`goal_seed` kwargs do not exist yet).

- [ ] **Step 3: Implement**

In `src/neuromorphic/envs/gridworld.py`, add `import random` near the top imports, add the module-level helper, and extend the constructor and `reset`:

```python
import random  # add with the other imports


def manhattan(a, b) -> int:
    """L1 distance between two (x, y) cells."""
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))
```

Change the `__init__` signature and body to add the two parameters (keep all existing params and lines):

```python
    def __init__(
        self,
        size: int = 5,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (4, 4),
        step_penalty: float = -1.0,
        goal_reward: float = 10.0,
        max_steps: int = 100,
        goals=None,
        goal_seed: int | None = None,
    ):
        super().__init__()
        self.size = size
        self.start = start
        self.goal = goal
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        self._goals = list(goals) if goals is not None else None
        self._goal_rng = random.Random(goal_seed)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(4,), dtype=np.int64
        )

        self._agent = np.array(start, dtype=np.int64)
        self._steps = 0
```

Change `reset` to sample a goal when a candidate set was provided:

```python
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._goals is not None:
            self.goal = self._goal_rng.choice(self._goals)
        self._agent = np.array(self.start, dtype=np.int64)
        self._steps = 0
        return self._obs(), {}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/envs/test_gridworld.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/envs/gridworld.py tests/envs/__init__.py tests/envs/test_gridworld.py
git commit -m "feat: random-goal sampling and manhattan helper in GridWorldEnv"
```

---

### Task 2: Potential-based reward shaping in the env

**Files:**
- Modify: `src/neuromorphic/envs/gridworld.py`
- Modify: `tests/envs/test_gridworld.py`

**Interfaces:**
- Consumes: `manhattan` (Task 1).
- Produces: `GridWorldEnv(..., reward_shaping: bool = False, shaping_gamma: float = 1.0)`. When `reward_shaping` is true, each `step` adds `shaping_gamma * Phi(s') - Phi(s)` to the reward, with `Phi(s) = -manhattan(agent, goal)`. Default off reproduces current reward exactly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/envs/test_gridworld.py`:

```python
def test_shaping_off_is_unchanged():
    env = GridWorldEnv(goal=(0, 2))  # reward_shaping defaults False
    env.reset()
    _, r, _, _, _ = env.step(2)  # action 2 = down -> (0,1)
    assert r == -1.0


def test_shaping_rewards_progress():
    env = GridWorldEnv(goal=(0, 2), reward_shaping=True)
    env.reset()  # agent (0,0), manhattan 2, Phi=-2
    # down -> (0,1), manhattan 1, Phi=-1: reward = -1 + (-1 - -2) = 0
    _, r_closer, _, _, _ = env.step(2)
    assert r_closer == 0.0
    # up -> (0,0), manhattan 2, Phi=-2: reward = -1 + (-2 - -1) = -2
    _, r_farther, _, _, _ = env.step(0)
    assert r_farther == -2.0
    assert r_closer > r_farther


def test_shaping_telescopes_over_path():
    env = GridWorldEnv(goal=(0, 3), reward_shaping=True)
    obs, _ = env.reset()
    start_phi = -manhattan((obs[0], obs[1]), env.goal)
    total = 0.0
    for action in (2, 2):  # two steps down, no goal reached
        obs, r, term, trunc, _ = env.step(action)
        total += r
    end_phi = -manhattan((obs[0], obs[1]), env.goal)
    base = -1.0 * 2  # two non-goal steps
    # shaped total minus base equals telescoped potential difference
    assert abs((total - base) - (end_phi - start_phi)) < 1e-9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/envs/test_gridworld.py -v`
Expected: FAIL (`reward_shaping` kwarg and shaping behavior do not exist).

- [ ] **Step 3: Implement**

Add the two parameters to `__init__` (after `goal_seed`) and initialize shaping state:

```python
        goals=None,
        goal_seed: int | None = None,
        reward_shaping: bool = False,
        shaping_gamma: float = 1.0,
    ):
        ...
        self._goals = list(goals) if goals is not None else None
        self._goal_rng = random.Random(goal_seed)
        self.reward_shaping = reward_shaping
        self.shaping_gamma = shaping_gamma
        self._prev_potential = 0.0
```

Set the potential at the end of `reset` (after the agent is placed and the goal is fixed):

```python
        self._agent = np.array(self.start, dtype=np.int64)
        self._steps = 0
        self._prev_potential = -manhattan(self._agent, self.goal)
        return self._obs(), {}
```

Add the shaping term in `step` just before the `return` (after `reward` is computed):

```python
        reward = self.goal_reward if terminated else self.step_penalty
        if self.reward_shaping:
            pot = -manhattan(self._agent, self.goal)
            reward += self.shaping_gamma * pot - self._prev_potential
            self._prev_potential = pot
        return self._obs(), float(reward), terminated, truncated, {}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/envs/test_gridworld.py -v`
Expected: PASS (7 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/envs/gridworld.py tests/envs/test_gridworld.py
git commit -m "feat: optional potential-based reward shaping in GridWorldEnv"
```

---

### Task 3: Goal split + optimality + EvalResult

**Files:**
- Create: `src/neuromorphic/training/generalization.py`
- Create: `tests/training/test_generalization.py`

**Interfaces:**
- Consumes: `manhattan` from `neuromorphic.envs.gridworld` (Task 1).
- Produces:
  - `split_goals(size: int, start: tuple[int,int], n_heldout: int, seed: int) -> tuple[list[tuple[int,int]], list[tuple[int,int]]]` returns `(train_goals, heldout_goals)`, disjoint, covering all cells except `start`, deterministic by seed.
  - `optimality(start, goal, steps: int) -> float`: `manhattan(start, goal) / steps` (0.0 if `steps <= 0`).
  - `@dataclass EvalResult` with fields `success_rate: float`, `mean_steps: float`, `optimality: float`, `n: int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/training/test_generalization.py`:

```python
from neuromorphic.training.generalization import EvalResult, optimality, split_goals


def test_split_goals_disjoint_sized_and_excludes_start():
    train, held = split_goals(size=5, start=(0, 0), n_heldout=6, seed=0)
    assert len(train) == 18 and len(held) == 6
    s_train, s_held = set(train), set(held)
    assert s_train.isdisjoint(s_held)
    assert (0, 0) not in s_train and (0, 0) not in s_held
    all_candidates = {(x, y) for x in range(5) for y in range(5)} - {(0, 0)}
    assert s_train | s_held == all_candidates


def test_split_goals_deterministic():
    a = split_goals(5, (0, 0), 6, seed=3)
    b = split_goals(5, (0, 0), 6, seed=3)
    c = split_goals(5, (0, 0), 6, seed=4)
    assert a == b
    assert a != c


def test_optimality():
    assert optimality((0, 0), (2, 0), 2) == 1.0
    assert optimality((0, 0), (2, 0), 4) == 0.5
    assert optimality((0, 0), (2, 0), 0) == 0.0


def test_eval_result_fields():
    r = EvalResult(success_rate=0.5, mean_steps=8.0, optimality=0.9, n=4)
    assert r.success_rate == 0.5 and r.n == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py -v`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement**

Create `src/neuromorphic/training/generalization.py`:

```python
"""Random-goal generalization experiment for the v1 grid-world policy (ADR-0001).

Tests whether the frozen-brain + linear-head policy learned to navigate or merely
memorized the fixed goal: train on a subset of goal cells, evaluate on held-out cells.
The brain stays frozen; only the existing head trains.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from neuromorphic.envs.gridworld import manhattan


def split_goals(
    size: int, start: tuple[int, int], n_heldout: int, seed: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Partition all cells except ``start`` into (train, held_out), deterministic by seed."""
    candidates = [(x, y) for x in range(size) for y in range(size) if (x, y) != start]
    random.Random(seed).shuffle(candidates)
    held_out = candidates[:n_heldout]
    train = candidates[n_heldout:]
    return train, held_out


def optimality(start: tuple[int, int], goal: tuple[int, int], steps: int) -> float:
    """Fraction of optimal: shortest-path length / steps taken (0.0 if no steps)."""
    if steps <= 0:
        return 0.0
    return manhattan(start, goal) / steps


@dataclass
class EvalResult:
    success_rate: float
    mean_steps: float
    optimality: float
    n: int
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/generalization.py tests/training/test_generalization.py
git commit -m "feat: goal split and eval primitives for generalization study"
```

---

### Task 4: Greedy evaluator

**Files:**
- Modify: `src/neuromorphic/training/generalization.py`
- Modify: `tests/training/test_generalization.py`

**Interfaces:**
- Consumes: `EvalResult`, `optimality` (Task 3); `GridWorldEnv` from `neuromorphic.envs`; `greedy_action` from `neuromorphic.training.reinforce`; `Brain`, `make_policy_head` for tests.
- Produces: `evaluate(brain, head, goals, *, size: int, start: tuple[int,int], max_steps: int, generator=None) -> EvalResult`. For each goal, build a fixed-goal `GridWorldEnv`, roll out the greedy policy from `start`, and aggregate: `success_rate` over all goals, `mean_steps` and `optimality` over reached goals (0.0 when none reached).

- [ ] **Step 1: Write the failing test**

Append to `tests/training/test_generalization.py`:

```python
import torch

from neuromorphic.brain import Brain
from neuromorphic.training.generalization import evaluate
from neuromorphic.training.reinforce import make_policy_head


def test_evaluate_smoke():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    gen = torch.Generator().manual_seed(0)
    res = evaluate(brain, head, [(1, 1), (2, 2)], size=5, start=(0, 0), max_steps=30, generator=gen)
    assert res.n == 2
    assert 0.0 <= res.success_rate <= 1.0
    assert res.mean_steps >= 0.0
    assert 0.0 <= res.optimality <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py::test_evaluate_smoke -v`
Expected: FAIL (`evaluate` not defined).

- [ ] **Step 3: Implement**

Add to `src/neuromorphic/training/generalization.py` (imports at top, function below `optimality`):

```python
import torch  # add to the imports block

from neuromorphic.envs.gridworld import GridWorldEnv, manhattan  # extend the existing import
from neuromorphic.training.reinforce import greedy_action
```

```python
def evaluate(
    brain,
    head,
    goals,
    *,
    size: int,
    start: tuple[int, int],
    max_steps: int,
    generator: "torch.Generator | None" = None,
) -> EvalResult:
    """Greedy rollouts from ``start`` to each goal; aggregate success, steps, optimality."""
    reached = 0
    steps_reached: list[int] = []
    opt_reached: list[float] = []
    for goal in goals:
        env = GridWorldEnv(size=size, start=start, goal=goal, max_steps=max_steps)
        obs, _ = env.reset()
        steps = 0
        done = False
        while steps < max_steps:
            with torch.no_grad():
                a = greedy_action(brain, head, obs, generator=generator)
            obs, _, term, trunc, _ = env.step(a)
            steps += 1
            if term:
                done = True
                break
            if trunc:
                break
        if done:
            reached += 1
            steps_reached.append(steps)
            opt_reached.append(optimality(start, goal, steps))
    n = len(list(goals)) if not isinstance(goals, list) else len(goals)
    mean_steps = sum(steps_reached) / len(steps_reached) if steps_reached else 0.0
    mean_opt = sum(opt_reached) / len(opt_reached) if opt_reached else 0.0
    return EvalResult(success_rate=reached / n if n else 0.0, mean_steps=mean_steps, optimality=mean_opt, n=n)
```

Note: `goals` is always passed as a list in this codebase; the `n` computation above is defensive. Keep the simple `len(goals)` form if you prefer, since all call sites pass lists.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/generalization.py tests/training/test_generalization.py
git commit -m "feat: greedy multi-goal evaluator for generalization study"
```

---

### Task 5: train_episode reports mean entropy

**Files:**
- Modify: `src/neuromorphic/training/reinforce.py`
- Modify: `tests/training/test_reinforce.py`

**Interfaces:**
- Produces: `train_episode(...)` returns the same dict plus a new key `"mean_entropy": float` (mean policy entropy over the episode's steps).

- [ ] **Step 1: Write the failing test**

Append to `tests/training/test_reinforce.py` (it already imports the trainer and builds a brain/head/env; reuse those helpers if present, otherwise this self-contained test):

```python
def test_train_episode_reports_mean_entropy():
    import torch
    from neuromorphic.brain import Brain
    from neuromorphic.envs import GridWorldEnv
    from neuromorphic.training.reinforce import make_policy_head, train_episode

    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    env = GridWorldEnv(max_steps=10)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    stats = train_episode(brain, head, env, opt, generator=gen, max_steps=10)
    assert "mean_entropy" in stats
    assert stats["mean_entropy"] >= 0.0
    assert stats["mean_entropy"] == stats["mean_entropy"]  # not NaN
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py::test_train_episode_reports_mean_entropy -v`
Expected: FAIL (`"mean_entropy"` not in stats).

- [ ] **Step 3: Implement**

In `src/neuromorphic/training/reinforce.py`, collect entropy in the episode loop and add it to the returned dict. In `train_episode`, add an `entropies` list and append each step's entropy:

```python
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    entropies: list[torch.Tensor] = []
    reached_goal = False
    ...
    while steps < limit:
        dist, _ = action_distribution(brain, head, obs, generator=generator)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        obs, reward, terminated, truncated, _ = env.step(int(action))
        ...
```

And in the returned dict:

```python
    return {
        "steps": steps,
        "total_reward": float(sum(rewards)),
        "mean_return": float(returns.mean()),
        "loss": float(loss.detach()),
        "reached_goal": reached_goal,
        "mean_entropy": float(torch.stack(entropies).mean()) if entropies else 0.0,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py -v`
Expected: PASS (all existing reinforce tests plus the new one).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/reinforce.py tests/training/test_reinforce.py
git commit -m "feat: train_episode reports mean policy entropy"
```

---

### Task 6: Config + run_generalization orchestrator

**Files:**
- Modify: `src/neuromorphic/training/generalization.py`
- Modify: `tests/training/test_generalization.py`

**Interfaces:**
- Consumes: `split_goals`, `evaluate` (Tasks 3-4); `Brain`; `GridWorldEnv` with `goals`/`reward_shaping` (Tasks 1-2); `make_policy_head`, `policy_parameters`, `train_episode`, `ema` from `reinforce` (Task 5).
- Produces:
  - `@dataclass GenConfig` with fields: `seed: int = 0`, `episodes: int = 600`, `lr: float = 1e-2`, `shaping: bool = True`, `n_heldout: int = 6`, `max_steps: int = 100`, `gamma: float = 0.99`, `baseline_beta: float = 0.1`, `size: int = 5`, `start: tuple[int,int] = (0,0)`, `tag: str = "shaped"`, `out_dir: Path = Path("outputs")`.
  - `run_generalization(cfg: GenConfig) -> dict`: trains the head on random train-set goals, writes the per-episode CSV and the summary JSON under `cfg.out_dir`, returns the summary dict. Summary keys: `config`, `train_goals`, `heldout_goals`, `eval` (with `train` and `heldout` sub-dicts of `EvalResult` fields), `generalization_gap`.

- [ ] **Step 1: Write the failing test**

Append to `tests/training/test_generalization.py`:

```python
def test_run_generalization_smoke(tmp_path):
    from neuromorphic.training.generalization import GenConfig, run_generalization

    cfg = GenConfig(seed=0, episodes=2, max_steps=20, n_heldout=6, tag="smoke", out_dir=tmp_path)
    summary = run_generalization(cfg)

    assert (tmp_path / "024_grid_generalization_smoke_metrics.csv").exists()
    assert (tmp_path / "024_grid_generalization_smoke_summary.json").exists()
    assert "generalization_gap" in summary
    assert "train" in summary["eval"] and "heldout" in summary["eval"]
    assert 0.0 <= summary["eval"]["train"]["success_rate"] <= 1.0

    # CSV has a header plus one row per episode
    lines = (tmp_path / "024_grid_generalization_smoke_metrics.csv").read_text().strip().splitlines()
    assert lines[0] == "episode,goal_x,goal_y,total_reward,steps,goal_reached,entropy"
    assert len(lines) == 1 + 2  # header + 2 episodes
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py::test_run_generalization_smoke -v`
Expected: FAIL (`GenConfig`/`run_generalization` not defined).

- [ ] **Step 3: Implement**

Extend the imports at the top of `src/neuromorphic/training/generalization.py`:

```python
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from neuromorphic.brain import Brain
from neuromorphic.training.reinforce import (
    ema,
    make_policy_head,
    policy_parameters,
    train_episode,
)
```

Add the config and orchestrator at the end of the file:

```python
@dataclass
class GenConfig:
    seed: int = 0
    episodes: int = 600
    lr: float = 1e-2
    shaping: bool = True
    n_heldout: int = 6
    max_steps: int = 100
    gamma: float = 0.99
    baseline_beta: float = 0.1
    size: int = 5
    start: tuple[int, int] = (0, 0)
    tag: str = "shaped"
    out_dir: Path = field(default_factory=lambda: Path("outputs"))


def run_generalization(cfg: GenConfig) -> dict:
    """Train the head on random train-set goals; eval train vs held-out; write CSV + JSON."""
    torch.manual_seed(cfg.seed)
    train_goals, heldout_goals = split_goals(cfg.size, cfg.start, cfg.n_heldout, cfg.seed)

    env = GridWorldEnv(
        size=cfg.size, start=cfg.start, goals=train_goals, goal_seed=cfg.seed,
        reward_shaping=cfg.shaping, max_steps=cfg.max_steps,
    )
    brain = Brain(grid_n=cfg.size, seed=cfg.seed)
    head = make_policy_head(brain)
    gen = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.Adam(policy_parameters(head), lr=cfg.lr)

    rows = []
    baseline = 0.0
    for ep in range(cfg.episodes):
        stats = train_episode(
            brain, head, env, opt, gamma=cfg.gamma, baseline=baseline,
            generator=gen, max_steps=cfg.max_steps,
        )
        baseline = ema(baseline, stats["mean_return"], cfg.baseline_beta)
        gx, gy = int(env.goal[0]), int(env.goal[1])
        rows.append((ep + 1, gx, gy, stats["total_reward"], stats["steps"],
                     int(stats["reached_goal"]), stats["mean_entropy"]))

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.out_dir / f"024_grid_generalization_{cfg.tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "goal_x", "goal_y", "total_reward", "steps", "goal_reached", "entropy"])
        w.writerows(rows)

    eval_train = evaluate(brain, head, train_goals, size=cfg.size, start=cfg.start,
                          max_steps=cfg.max_steps, generator=gen)
    eval_held = evaluate(brain, head, heldout_goals, size=cfg.size, start=cfg.start,
                         max_steps=cfg.max_steps, generator=gen)
    gap = eval_train.success_rate - eval_held.success_rate

    summary = {
        "config": asdict(cfg) | {"out_dir": str(cfg.out_dir)},
        "train_goals": train_goals,
        "heldout_goals": heldout_goals,
        "eval": {"train": asdict(eval_train), "heldout": asdict(eval_held)},
        "generalization_gap": gap,
    }
    json_path = cfg.out_dir / f"024_grid_generalization_{cfg.tag}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    return summary
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/generalization.py tests/training/test_generalization.py
git commit -m "feat: run_generalization orchestrator with CSV and summary logging"
```

---

### Task 7: Experiment CLI

**Files:**
- Create: `experiments/024_grid_generalization/run.py`

**Interfaces:**
- Consumes: `GenConfig`, `run_generalization` (Task 6).
- Produces: a CLI that parses args into a `GenConfig`, calls `run_generalization`, and prints the headline. Flags: `--episodes`, `--lr`, `--seed`, `--shaping/--no-shaping` (default shaping on), `--n-heldout`, `--max-steps`, `--tag`.

- [ ] **Step 1: Create the CLI**

Create `experiments/024_grid_generalization/run.py`:

```python
"""EXP-024 — grid-world generalization study (random goals, held-out eval).

Tests whether the v1 policy (frozen brain + linear head) learned to navigate or
memorized the fixed goal. Trains on a train subset of goal cells, evaluates greedily
on held-out cells, and reports the generalization gap. Brain stays frozen (ADR-0001).

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/024_grid_generalization/run.py --tag shaped
    .venv/Scripts/python.exe experiments/024_grid_generalization/run.py --no-shaping --tag sparse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch  # noqa: E402

from neuromorphic.training.generalization import GenConfig, run_generalization

torch.set_num_threads(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-024 grid-world generalization study")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-heldout", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--tag", type=str, default="shaped")
    p.add_argument("--shaping", action=argparse.BooleanOptionalAction, default=True,
                   help="potential-based distance-to-goal shaping (default on)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GenConfig(
        seed=args.seed, episodes=args.episodes, lr=args.lr, shaping=args.shaping,
        n_heldout=args.n_heldout, max_steps=args.max_steps, tag=args.tag,
        out_dir=Path("outputs"),
    )
    print(f"EXP-024 · {cfg.episodes} eps · shaping {cfg.shaping} · tag {cfg.tag}", flush=True)
    summary = run_generalization(cfg)
    et = summary["eval"]["train"]
    eh = summary["eval"]["heldout"]
    print(f"train  goals: success {et['success_rate']:.0%} · opt {et['optimality']:.2f}", flush=True)
    print(f"heldout goals: success {eh['success_rate']:.0%} · opt {eh['optimality']:.2f}", flush=True)
    print(f"generalization gap: {summary['generalization_gap']:+.2f} "
          f"(train success minus heldout success)", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the CLI**

Run: `.venv/Scripts/python.exe experiments/024_grid_generalization/run.py --episodes 5 --max-steps 20 --tag smoke`
Expected: prints the headline lines; writes `outputs/024_grid_generalization_smoke_metrics.csv` and `..._summary.json`. Then delete the smoke outputs: `rm outputs/024_grid_generalization_smoke_*`.

- [ ] **Step 3: Commit**

```bash
git add experiments/024_grid_generalization/run.py
git commit -m "feat: EXP-024 generalization study CLI"
```

---

### Task 8: Run the experiment and record results

**Files:** none changed (execution + results capture).

- [ ] **Step 1: Full test suite green**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all green (existing suite plus the new env, generalization, and reinforce tests).

- [ ] **Step 2: Shaped run (primary)**

Run: `.venv/Scripts/python.exe experiments/024_grid_generalization/run.py --tag shaped`
Capture the headline (train success, held-out success, gap) and the written `outputs/024_grid_generalization_shaped_summary.json`.

- [ ] **Step 3: Sparse ablation**

Run: `.venv/Scripts/python.exe experiments/024_grid_generalization/run.py --no-shaping --tag sparse`
Capture its headline and summary.

- [ ] **Step 4: Record the finding**

Note the generalization gap for both runs and the interpretation (navigates / memorized / did-not-learn, and whether shaping mattered). These outputs feed the Obsidian write-up. Commit the result artifacts:

```bash
git add outputs/024_grid_generalization_shaped_metrics.csv outputs/024_grid_generalization_shaped_summary.json outputs/024_grid_generalization_sparse_metrics.csv outputs/024_grid_generalization_sparse_summary.json
git commit -m "feat: EXP-024 generalization results (shaped + sparse)"
```

---

## Self-Review

**Spec coverage:**
- Env random goal + seedable RNG + fixed-goal backward compat: Task 1. ✓
- Potential-based shaping, default off reproduces current reward: Task 2. ✓
- `split_goals` deterministic 18/6 disjoint: Task 3. ✓
- Greedy `evaluate` with success/steps/optimality: Task 4. ✓
- Per-episode entropy for the CSV: Task 5. ✓
- `run_generalization` writes per-episode CSV + summary JSON with config and train-vs-held-out eval and the gap: Task 6. ✓
- CLI with shaping toggle and tags: Task 7. ✓
- Run shaped + sparse, report the gap: Task 8. ✓
- Stretch (Q-learning comparison) and deferred items (curriculum, brain unfreeze, W&B/TensorBoard): correctly out of scope, no tasks. ✓

**Placeholder scan:** none. Every code step shows complete code; the one defensive `n` computation in Task 4 is annotated with the simpler alternative.

**Type consistency:** `manhattan`, `split_goals`, `optimality`, `EvalResult`, `evaluate`, `GenConfig`, `run_generalization` are referenced with identical signatures across tasks. `train_episode`'s new `"mean_entropy"` key (Task 5) is consumed by `run_generalization` (Task 6). The env's new `goals`/`goal_seed`/`reward_shaping`/`shaping_gamma` kwargs (Tasks 1-2) are consumed by `run_generalization` and `evaluate`.

**Known shared-code touch:** Task 5 adds a key to `train_episode`'s returned dict; this is additive and does not affect EXP-023, which ignores the new key.
