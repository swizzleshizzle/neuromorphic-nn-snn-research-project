# REINFORCE Brain Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the five-region `Brain` actually learn the grid world — implement a surrogate-gradient REINFORCE training loop (ADR-0001) and a first-run experiment that trains for N episodes and emits a training curve + a dashboard trace.

**Architecture:** A new `neuromorphic.training.reinforce` module turns motor spike-counts (surrogate-differentiable) into a categorical action distribution, samples actions, and applies a Monte-Carlo policy-gradient update (REINFORCE with a running baseline). The memory path is bypassed (`recall=False`) per ADR-0001, so credit flows end-to-end through sensory → prefrontal → motor only. Training lives in a dedicated trainer, **not** in `Brain.learn` (REINFORCE is episodic; the per-step `learn`/dopamine hook stays reserved for the later R-STDP hybrid).

**Tech Stack:** Python 3.10+, PyTorch (`torch.optim.Adam`, `torch.distributions.Categorical`), snnTorch (surrogate gradients already in `snn.Leaky`), matplotlib (Agg), pytest. No new dependencies.

**Reference:** ADR-0001 (`docs/adr/0001-multi-region-training-strategy.md`).

## Global Constraints

- Run Python via the project venv: `.venv/Scripts/python.exe` (Windows). All `pytest`/script commands use it.
- Commit messages: plain, **no** "Co-Authored-By" / "Generated with" trailer (repo convention).
- Traces/policy are **single-agent** (`B = 1`).
- Memory bypassed in v1: every training/eval forward uses `recall=False`, `store=False` (ADR-0001).
- Reuse existing code: `neuromorphic.brain.Brain`, `neuromorphic.envs.GridWorldEnv`, `neuromorphic.monitor` (FileSink/record_episode). Do not modify region internals.
- Experiment scripts are standalone (constants + `main()`), matching EXP-020/021 — they do **not** use `ExperimentConfig`.
- matplotlib must use the `Agg` backend (set before importing pyplot).

---

## Key facts (verified against the code)

- `Brain.step(obs, *, store, recall, record, generator)` returns a dict; `out["action_spikes"]` is `[T, B, n_actions]`, **surrogate-differentiable** (snn.Leaky). `out["action"]` is the greedy argmax — REINFORCE ignores it and samples instead.
- Policy logits = motor spike-counts over the window: `out["action_spikes"].sum(dim=0)[0]` → `[n_actions]`, differentiable.
- With `recall=False`, `Brain` never calls the hippocampus, so `brain.hippo.fc_in/fc_out` receive **no gradient** (the bypass, made testable).
- Learnable policy params include `brain.sensory.fc1/fc2`, `brain.pfc.fc_transform/fc_utility`, `brain.motor.fc_in` (all `nn.Linear`). Lateral-inhibition matrices and `hippo.W_rec` are `register_buffer`s — never in `brain.parameters()`, never trained.
- `GridWorldEnv`: `reset(seed=...) -> (obs, {})`, `step(action) -> (obs, reward, terminated, truncated, {})`, reward `-1`/step, `+10` on goal, `max_steps=100`. Obs `(ax, ay, gx, gy)`.

## File structure

- Create: `src/neuromorphic/training/reinforce.py` — the trainer (pure helpers + `action_distribution` + `train_episode`)
- Create: `tests/training/__init__.py` (empty, matches `tests/viz/__init__.py` convention)
- Create: `tests/training/test_smoke.py` — the brief's "spikes flow through all regions" smoke
- Create: `tests/training/test_reinforce.py` — TDD for the trainer
- Create: `experiments/023_week11_brain_training/run.py` — first-run experiment (train + curve + trace)

Note: `src/neuromorphic/training/__init__.py` already exists (empty package). Leave it as-is.

---

## Task 1: Smoke test — spikes flow through all five regions

The brief's moment-of-truth check, made into a regression test. No new source — exercises the existing assembled `Brain`.

**Files:**
- Create: `tests/training/__init__.py` (empty)
- Create: `tests/training/test_smoke.py`

- [ ] **Step 1: Write the test**

`tests/training/__init__.py`: empty file.

`tests/training/test_smoke.py`:
```python
"""Brain assembly smoke: random input → spikes flow through all five regions."""

from __future__ import annotations

import torch

from neuromorphic.brain import Brain

# region id -> its primary output recording key (the spike train it emits)
PRIMARY = {
    "sensory": "concept",
    "hippocampus": "population",
    "prefrontal": "utility",
    "router": "select",
    "motor": "action",
}


def test_spikes_flow_through_all_regions():
    brain = Brain(grid_n=5, seed=0)
    gen = torch.Generator().manual_seed(0)
    # a random valid observation (coords in [0, grid_n))
    obs = torch.randint(0, 5, (4,), generator=gen).tolist()

    out = brain.step(obs, store=True, recall=True, record=True, generator=gen)
    recordings = out["recordings"]

    # every region produced recordings, and its primary output train has spikes
    for region, key in PRIMARY.items():
        assert region in recordings, f"{region} produced no recordings"
        train = recordings[region].get(key)
        assert train is not None, f"{region} has no '{key}' recording"
        assert train.sum().item() > 0, f"{region} ({key}) emitted zero spikes"


def test_step_produces_a_valid_action():
    brain = Brain(grid_n=5, seed=0)
    out = brain.step([0, 0, 4, 4], recall=False, generator=torch.Generator().manual_seed(0))
    assert isinstance(out["action"], int)
    assert 0 <= out["action"] < 4
```

- [ ] **Step 2: Run test to verify it passes (assembly already works)**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_smoke.py -v`
Expected: PASS (2 tests). If the `router`/`select` assertion fails because the router's selection train is silent for this obs, change the router key check to accept either: `recordings["router"].get("select")` or `recordings["router"].get("gate")` having spikes — but run first; with `store=True, recall=True` the selection layer should fire.

- [ ] **Step 3: Commit**

```bash
git add tests/training/__init__.py tests/training/test_smoke.py
git commit -m "test: brain assembly smoke — spikes flow through all five regions"
```

---

## Task 2: REINFORCE math helpers (returns + baseline)

**Files:**
- Create: `src/neuromorphic/training/reinforce.py`
- Create: `tests/training/test_reinforce.py`

**Interfaces:**
- Produces: `discounted_returns(rewards: list[float], gamma: float) -> list[float]` (returns-to-go); `ema(old: float, new: float, beta: float) -> float`.

- [ ] **Step 1: Write the failing test**

`tests/training/test_reinforce.py`:
```python
import math

import torch

from neuromorphic.training.reinforce import discounted_returns, ema


def test_discounted_returns_to_go():
    r = discounted_returns([0.0, 0.0, 1.0], gamma=0.9)
    assert len(r) == 3
    assert math.isclose(r[2], 1.0, rel_tol=1e-6)
    assert math.isclose(r[1], 0.9, rel_tol=1e-6)
    assert math.isclose(r[0], 0.81, rel_tol=1e-6)


def test_discounted_returns_undiscounted_sum():
    r = discounted_returns([1.0, 1.0, 1.0], gamma=1.0)
    assert r == [3.0, 2.0, 1.0]


def test_discounted_returns_empty():
    assert discounted_returns([], gamma=0.99) == []


def test_ema_blends():
    assert math.isclose(ema(0.0, 10.0, 0.1), 1.0, rel_tol=1e-6)
    assert math.isclose(ema(10.0, 10.0, 0.5), 10.0, rel_tol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuromorphic.training.reinforce'`.

- [ ] **Step 3: Write the implementation**

`src/neuromorphic/training/reinforce.py`:
```python
"""Surrogate-gradient REINFORCE training for the five-region Brain (ADR-0001).

Motor spike-counts over the inference window are surrogate-differentiable, so they
form a categorical policy: sample an action, weight its log-probability by the
(baseline-subtracted) discounted return, and backprop through the spiking layers.
The memory path is bypassed (``recall=False``) — credit flows sensory → PFC → motor.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Returns-to-go: ``G_t = r_t + gamma * G_{t+1}`` for each step."""
    out: list[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    return out


def ema(old: float, new: float, beta: float) -> float:
    """Exponential moving average: ``(1 - beta) * old + beta * new``."""
    return (1.0 - beta) * old + beta * new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/reinforce.py tests/training/test_reinforce.py
git commit -m "feat: REINFORCE return-to-go + EMA baseline helpers"
```

---

## Task 3: Differentiable action distribution

**Files:**
- Modify: `src/neuromorphic/training/reinforce.py`
- Modify: `tests/training/test_reinforce.py`

**Interfaces:**
- Consumes: `Brain.step` (returns `out["action_spikes"]` `[T,B,n_actions]`).
- Produces: `policy_logits(out: dict) -> torch.Tensor` (`[n_actions]`, differentiable); `action_distribution(brain, obs, *, generator=None) -> tuple[Categorical, torch.Tensor]` returning `(dist, logits)` with `recall=False`.

- [ ] **Step 1: Write the failing test**

Append to `tests/training/test_reinforce.py`:
```python
from neuromorphic.brain import Brain
from neuromorphic.training.reinforce import action_distribution, policy_logits


def test_policy_logits_shape_and_grad():
    brain = Brain(grid_n=5, seed=0)
    out = brain.step([0, 0, 4, 4], recall=False, generator=torch.Generator().manual_seed(0))
    logits = policy_logits(out)
    assert logits.shape == (4,)
    assert logits.requires_grad


def test_action_distribution_is_a_valid_policy():
    brain = Brain(grid_n=5, seed=0)
    dist, logits = action_distribution(brain, [0, 0, 4, 4], generator=torch.Generator().manual_seed(0))
    assert logits.shape == (4,)
    probs = dist.probs
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert (probs >= 0).all()


def test_log_prob_gradient_reaches_policy_but_not_memory():
    brain = Brain(grid_n=5, seed=0)
    dist, _ = action_distribution(brain, [0, 0, 4, 4], generator=torch.Generator().manual_seed(0))
    action = dist.sample()
    dist.log_prob(action).backward()
    # policy path received gradient
    assert brain.sensory.fc1.weight.grad is not None
    assert brain.motor.fc_in.weight.grad is not None
    # memory path bypassed (recall=False) → no gradient
    assert brain.hippo.fc_in.weight.grad is None
    assert brain.hippo.fc_out.weight.grad is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py -k "policy or distribution or gradient" -v`
Expected: FAIL — `cannot import name 'action_distribution'`.

- [ ] **Step 3: Write the implementation**

Append to `src/neuromorphic/training/reinforce.py`:
```python
def policy_logits(out: dict) -> torch.Tensor:
    """Differentiable action logits = motor spike-counts over the window (single agent)."""
    return out["action_spikes"].sum(dim=0)[0]  # [T,B,A] -> [A]


def action_distribution(
    brain, obs, *, generator: torch.Generator | None = None
) -> tuple[Categorical, torch.Tensor]:
    """One forward pass → a categorical policy over actions (memory bypassed)."""
    out = brain.step(obs, store=False, recall=False, record=False, generator=generator)
    logits = policy_logits(out)
    return Categorical(logits=logits), logits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py -v`
Expected: PASS (all — 4 math + 3 distribution).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/reinforce.py tests/training/test_reinforce.py
git commit -m "feat: differentiable categorical action policy from motor spikes"
```

---

## Task 4: The REINFORCE episode update

**Files:**
- Modify: `src/neuromorphic/training/reinforce.py`
- Modify: `tests/training/test_reinforce.py`

**Interfaces:**
- Consumes: `discounted_returns`, `action_distribution`; an `env` (Gymnasium `reset`/`step`), a `torch.optim.Optimizer`.
- Produces: `train_episode(brain, env, optimizer, *, gamma=0.99, baseline=0.0, generator=None, max_steps=None) -> dict` with keys `steps, total_reward, mean_return, loss, reached_goal`.

- [ ] **Step 1: Write the failing test**

Append to `tests/training/test_reinforce.py`:
```python
from neuromorphic.envs import GridWorldEnv
from neuromorphic.training.reinforce import train_episode


def test_train_episode_returns_stats_and_updates_policy():
    brain = Brain(grid_n=5, seed=0)
    env = GridWorldEnv()
    opt = torch.optim.Adam(brain.parameters(), lr=1e-2)
    before = brain.sensory.fc1.weight.detach().clone()

    stats = train_episode(
        brain, env, opt, gamma=0.99, baseline=0.0,
        generator=torch.Generator().manual_seed(0), max_steps=10,
    )

    assert set(stats) == {"steps", "total_reward", "mean_return", "loss", "reached_goal"}
    assert stats["steps"] >= 1
    assert isinstance(stats["reached_goal"], bool)
    # an optimizer step actually moved the policy weights
    after = brain.sensory.fc1.weight.detach()
    assert not torch.equal(before, after)


def test_train_episode_leaves_memory_weights_untouched():
    brain = Brain(grid_n=5, seed=0)
    env = GridWorldEnv()
    opt = torch.optim.Adam(brain.parameters(), lr=1e-2)
    hippo_before = brain.hippo.fc_in.weight.detach().clone()

    train_episode(brain, env, opt, generator=torch.Generator().manual_seed(0), max_steps=10)

    # recall=False → no grad → Adam never updates the memory afferent
    assert torch.equal(hippo_before, brain.hippo.fc_in.weight.detach())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py -k train_episode -v`
Expected: FAIL — `cannot import name 'train_episode'`.

- [ ] **Step 3: Write the implementation**

Append to `src/neuromorphic/training/reinforce.py`:
```python
def train_episode(
    brain,
    env,
    optimizer,
    *,
    gamma: float = 0.99,
    baseline: float = 0.0,
    generator: torch.Generator | None = None,
    max_steps: int | None = None,
) -> dict:
    """Run one episode, then apply one REINFORCE update. Memory bypassed (recall=False).

    Returns stats: ``steps``, ``total_reward`` (undiscounted), ``mean_return``
    (mean discounted return-to-go, for baseline tracking), ``loss``, ``reached_goal``.
    """
    obs, _ = env.reset()
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    reached_goal = False
    limit = max_steps if max_steps is not None else getattr(env, "max_steps", 100)

    steps = 0
    while steps < limit:
        dist, _ = action_distribution(brain, obs, generator=generator)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        obs, reward, terminated, truncated, _ = env.step(int(action))
        rewards.append(float(reward))
        steps += 1
        if terminated:
            reached_goal = True
            break
        if truncated:
            break

    returns = torch.tensor(discounted_returns(rewards, gamma), dtype=torch.float32)
    advantages = returns - baseline
    loss = -(torch.stack(log_probs) * advantages).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "steps": steps,
        "total_reward": float(sum(rewards)),
        "mean_return": float(returns.mean()),
        "loss": float(loss.detach()),
        "reached_goal": reached_goal,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py -v`
Expected: PASS (all — math + distribution + 2 train_episode).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/reinforce.py tests/training/test_reinforce.py
git commit -m "feat: REINFORCE episode update (sample, return-weighted policy gradient)"
```

---

## Task 5: First-run experiment (train + curve + dashboard trace)

The moment of truth: untrained baseline → train N episodes → training curve + a trained trace for the dashboard. Standalone script, verified by running it.

**Files:**
- Create: `experiments/023_week11_brain_training/run.py`

- [ ] **Step 1: Write the experiment script**

`experiments/023_week11_brain_training/run.py`:
```python
"""EXP-023 — first grid-world training run (Week-11 S3 hands-on, ADR-0001).

Trains the five-region Brain on the 5x5 grid world with surrogate-gradient REINFORCE
(memory bypassed, recall=False). Prints an untrained baseline, trains for EPISODES,
writes a training curve, and records one trained episode as a dashboard trace.

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/023_week11_brain_training/run.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from neuromorphic.brain import Brain  # noqa: E402
from neuromorphic.envs import GridWorldEnv  # noqa: E402
from neuromorphic.monitor import FileSink, record_episode  # noqa: E402
from neuromorphic.training.reinforce import ema, train_episode  # noqa: E402

EPISODES = 200
LR = 1e-2
GAMMA = 0.99
BASELINE_BETA = 0.1
SEED = 0
EVAL_EPISODES = 20

CURVE = Path("outputs/week11_training_curve.png")
TRACE = Path("outputs/week11_trained_trace.jsonl")


def eval_avg_reward(brain: Brain, env: GridWorldEnv, n: int, gen: torch.Generator) -> tuple[float, float]:
    """Average total reward and goal-reached fraction over n greedy episodes (no learning)."""
    total = 0.0
    goals = 0
    for _ in range(n):
        with torch.no_grad():
            summary = brain.run_episode(env, store_first=False, generator=gen)
        total += summary["total_reward"]
        goals += 1 if summary["reached_goal"] else 0
    return total / n, goals / n


def moving_avg(xs: list[float], k: int = 20) -> list[float]:
    out = []
    for i in range(len(xs)):
        window = xs[max(0, i - k + 1):i + 1]
        out.append(sum(window) / len(window))
    return out


def main() -> None:
    torch.manual_seed(SEED)
    env = GridWorldEnv()
    brain = Brain(grid_n=env.size, seed=SEED)
    gen = torch.Generator().manual_seed(SEED)

    pre_reward, pre_goals = eval_avg_reward(brain, env, EVAL_EPISODES, gen)
    print(f"untrained: avg reward {pre_reward:.1f} · reached goal {pre_goals:.0%}")

    opt = torch.optim.Adam(brain.parameters(), lr=LR)
    baseline = 0.0
    rewards: list[float] = []
    for ep in range(EPISODES):
        stats = train_episode(brain, env, opt, gamma=GAMMA, baseline=baseline, generator=gen)
        baseline = ema(baseline, stats["mean_return"], BASELINE_BETA)
        rewards.append(stats["total_reward"])
        if (ep + 1) % 20 == 0:
            recent = sum(rewards[-20:]) / min(20, len(rewards))
            print(f"  ep {ep + 1:4d} · reward {stats['total_reward']:6.1f} · "
                  f"avg20 {recent:6.1f} · goal {stats['reached_goal']}")

    post_reward, post_goals = eval_avg_reward(brain, env, EVAL_EPISODES, gen)
    print(f"trained:   avg reward {post_reward:.1f} · reached goal {post_goals:.0%}")
    print(f"delta:     {post_reward - pre_reward:+.1f} reward")

    CURVE.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.3, label="episode reward")
    plt.plot(moving_avg(rewards), label="moving avg (20)")
    plt.axhline(pre_reward, ls="--", c="gray", label="untrained baseline")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title("EXP-023 — REINFORCE on 5x5 grid world")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE, dpi=110)
    print(f"training curve -> {CURVE}")

    # record one trained episode for the NEURO·SCOPE dashboard (memory bypassed, as trained)
    sink = FileSink(TRACE)
    record_episode(brain, env, sink, seed=SEED, recall=False, generator=gen)
    print(f"trained trace  -> {TRACE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

Run: `.venv/Scripts/python.exe experiments/023_week11_brain_training/run.py`
Expected: prints an untrained baseline, per-20-episode progress, a trained summary line, and writes `outputs/week11_training_curve.png` + `outputs/week11_trained_trace.jsonl`. **Per the brief, the brain may not learn well** — success here is that it runs end-to-end, the loss/updates are applied, and a curve + trace are produced. Improvement over the untrained baseline is the hope, not the gate.

- [ ] **Step 3: Sanity-check the outputs**

Run: `.venv/Scripts/python.exe -c "import pathlib,json; t=pathlib.Path('outputs/week11_trained_trace.jsonl'); L=t.read_text().splitlines(); print('trace lines', len(L)); print('curve exists', pathlib.Path('outputs/week11_training_curve.png').exists())"`
Expected: `trace lines` ≥ 2 and `curve exists True`.

- [ ] **Step 4: Commit the script**

```bash
git add experiments/023_week11_brain_training/run.py
git commit -m "exp 023: first grid-world REINFORCE training run + curve + trace"
```
(The generated `outputs/*.png` and `*.jsonl` are gitignored — commit only the script.)

---

## Self-review notes (already applied)

- **ADR-0001 coverage:** REINFORCE objective (motor spikes → Categorical → log-prob × return) → Tasks 3–4; surrogate gradient reused (snn.Leaky, no new code) → verified in Task 3's grad test; memory bypassed `recall=False` → Tasks 3–4 + asserted (hippo grad None / weights untouched); end-to-end across sensory→PFC→motor → Task 3 grad test; baseline (variance reduction) → Tasks 2/4/5; "first run, may not work well" framing → Task 5.
- **Brief coverage:** brain assembly (already built) — confirmed live by Task 1 smoke; "spikes flow through all regions" → Task 1; reward-modulated weight update → Task 4; training loop 100–500 episodes → Task 5 (`EPISODES=200`, adjustable); connect to grid world + untrained vs trained → Task 5; watch the dashboard → Task 5 emits `week11_trained_trace.jsonl` for NEURO·SCOPE.
- **Design note (deviation from ADR wording):** ADR-0001 said "`Brain.learn` evolves into the REINFORCE update." On implementation, REINFORCE is **episodic** (needs the whole episode's log-probs), which does not fit the per-step `learn(reward)` signature. The trainer therefore lives in `neuromorphic.training.reinforce`; `Brain.learn`/the dopamine bus stay reserved for the per-step R-STDP hybrid (Option 3). This is the more correct factoring and is noted here as the source of truth.
- **Type consistency:** `discounted_returns`, `ema`, `policy_logits`, `action_distribution`, `train_episode` signatures and the stats-dict keys (`steps, total_reward, mean_return, loss, reached_goal`) are identical across Tasks 2–5. `mean_return` (not `total_reward`) feeds the EMA baseline so advantage scales match the discounted returns-to-go.
- **No new deps; venv-pinned commands; plain commit messages.**
```
