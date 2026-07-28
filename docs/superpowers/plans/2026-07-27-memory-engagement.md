# EXP-030 Memory Engagement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the hippocampus on the policy path and test whether episodic memory content improves cube solving, against a shuffle-null that holds head width fixed.

**Architecture:** Fix the hippocampus so it can hold more than one pattern and can report familiarity. Add a pluggable feature readout to `reinforce.py` so the policy head can read more than the sensory concept. Teach `CubeRunner` three readout modes plus revisit instrumentation. Then a thin driver.

**Tech Stack:** Python, gymnasium, numpy, torch, snntorch, pytest. Run python via `.venv/Scripts/python.exe`.

## Global Constraints

- **Reusable machinery under `src/neuromorphic/` only. `experiments/030_memory_engagement/` holds the driver and nothing importable.**
- Run python via `.venv/Scripts/python.exe`.
- Plain commit messages; NO `Co-Authored-By` trailer; NO em-dashes anywhere in code, docs, or commit messages.
- **The full suite is 340 passing at base and must never shrink.** `mode='concept'` must be bit-identical to today's behavior under a fixed generator, so every 024-029 driver is unaffected.
- **`reinforce.py` IS modified by this plan.** The "do not touch it" rule was specific to EXP-029. Engagement necessarily changes what the policy reads.
- TDD is mandatory: failing test first, confirmed failing for the stated reason, then implement, then green.
- Base: `main` at `af29e84`.

## Measured facts (do not re-derive)

Verified 2026-07-27 against a prototype of the accumulate fix:

- Broken today: recall pairwise cosine **0.998** (cube) and **0.9995** (grid). `store()` assigns, so only one pattern is ever held.
- After accumulating: cosine **0.666** at 2 patterns, **0.912** at 4, **0.962** at 8, **0.990** at 15, **0.994** at 21.
- Familiarity separation (visited minus novel) is **+1.04 to +1.81** across that whole range, stable where completion degrades.
- Five-region `Brain` at `n_actions=6` totals 510 neurons; `brain.step` costs about 90 ms.

## Design decisions this plan locks in

- **`arm` and `readout` are different axes.** `CubeConfig.arm` already means network topology (`regionalized` / `monolithic` / `random`) for EXP-029. EXP-030 adds `CubeConfig.readout` (`concept` / `memory` / `memory_shuffled`) and always uses `arm="regionalized"`. Do NOT overload `arm`.
- **`sigma=0.0` for all EXP-030 runs.** EXP-029 swept concept noise; here regularization is held fixed at zero across all three readouts so the only difference is the memory features.
- **Memory feature logic lives in `cube_baseline.py`, not `reinforce.py`.** `reinforce.py` gets one generic `feature_fn` parameter; the episode-scoped visited-concept cache and shuffle logic belong to the experiment runner.

## File structure

- `src/neuromorphic/regions/hippocampus.py` - accumulate, `clear()`, `familiarity()` (Task 1).
- `tests/regions/test_hippocampus.py` - replace two too-weak tests, add three (Task 1).
- `src/neuromorphic/training/reinforce.py` - `feature_fn` / `store` / `recall` passthrough (Task 2).
- `src/neuromorphic/training/cube_baseline.py` - `MemoryReadout`, `CubeConfig.readout`, revisit instrumentation (Task 3).
- `experiments/030_memory_engagement/{run.py,aggregate.py}` - driver only (Task 4).

---

### Task 1: Hippocampus becomes a real memory

**Files:**
- Modify: `src/neuromorphic/regions/hippocampus.py`
- Test: `tests/regions/test_hippocampus.py`

**Interfaces:**
- Produces:
  - `Hippocampus.store(content)` accumulates into `W_rec` instead of assigning.
  - `Hippocampus.clear() -> None` zeroes `W_rec` and the stored-pattern list.
  - `Hippocampus.familiarity(content: Tensor[B, content_dim]) -> Tensor[B]`, the Hopfield field alignment. Zeros when nothing is stored.
  - `Hippocampus.n_stored -> int`.

- [ ] **Step 1: Write the failing tests**

Replace `test_store_imprints_recurrent_weights` (it asserts only `count_nonzero(W_rec) > 0`, which cannot tell accumulate from overwrite) and `test_recall_is_content_specific` (it asserts only `not torch.equal(...)`, which passes at 99.8 percent identical). Delete both and add:

```python
def test_store_accumulates_rather_than_overwriting():
    """The defect this fixes: store() used to assign, so only one pattern survived."""
    h = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    g = torch.Generator().manual_seed(0)
    a = torch.rand(1, CONTENT, generator=g)
    b = torch.rand(1, CONTENT, generator=g)

    h.store(a)
    w_after_a = h.W_rec.clone()
    h.store(b)
    w_after_b = h.W_rec.clone()

    # B's contribution in isolation, from a fresh region with the same init
    h2 = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    h2.store(b)
    w_b_only = h2.W_rec.clone()

    assert not torch.allclose(w_after_b, w_b_only), "W_rec equals B alone: store still overwrites"
    assert torch.allclose(w_after_b, w_after_a + w_b_only, atol=1e-6)
    assert h.n_stored == 2


def test_clear_forgets_everything():
    h = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    h.store(torch.rand(1, CONTENT))
    assert torch.count_nonzero(h.W_rec) > 0
    h.clear()
    assert torch.count_nonzero(h.W_rec) == 0
    assert h.n_stored == 0


def test_recall_discriminates_between_stored_states():
    """Replaces the old not-equal assertion. Measured 0.912 at 4 patterns after the fix,
    versus 0.998 before it, so 0.95 is a threshold the broken code cannot pass."""
    import torch.nn.functional as F

    h = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    g = torch.Generator().manual_seed(0)
    contents = [torch.rand(1, CONTENT, generator=g) for _ in range(4)]
    for c in contents:
        h.store(c)

    recalls = torch.stack([h(c.unsqueeze(0).expand(T, 1, CONTENT)).mean(dim=0)[0] for c in contents])
    sims = [
        float(F.cosine_similarity(recalls[i], recalls[j], dim=0))
        for i in range(4) for j in range(i + 1, 4)
    ]
    assert sum(sims) / len(sims) < 0.95


def test_familiarity_separates_visited_from_novel():
    """Measured separation is +1.4 to +1.8 across loads; 0.3 is a conservative floor."""
    h = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    g = torch.Generator().manual_seed(0)
    visited = [torch.rand(1, CONTENT, generator=g) for _ in range(8)]
    novel = [torch.rand(1, CONTENT, generator=g) for _ in range(8)]
    for c in visited:
        h.store(c)

    fam_v = torch.cat([h.familiarity(c) for c in visited]).mean()
    fam_n = torch.cat([h.familiarity(c) for c in novel]).mean()
    assert float(fam_v - fam_n) > 0.3


def test_familiarity_is_zero_with_empty_memory():
    h = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    fam = h.familiarity(torch.rand(3, CONTENT))
    assert fam.shape == (3,)
    assert torch.allclose(fam, torch.zeros(3))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/regions/test_hippocampus.py -q`
Expected: FAIL. `test_store_accumulates_rather_than_overwriting` fails on the "store still overwrites" assertion, `test_recall_discriminates_between_stored_states` fails with a cosine near 0.998, and the `clear` / `familiarity` / `n_stored` tests fail with `AttributeError`.

- [ ] **Step 3: Implement the changes**

In `__init__`, after `self._stored_pattern: torch.Tensor | None = None`, add:

```python
        self._stored_patterns: list[torch.Tensor] = []
```

Replace the last three lines inside `store()`'s `torch.no_grad()` block:

```python
            # Accumulate, do not assign. Assigning kept only the most recent pattern,
            # which collapsed recall to a near-constant code (measured cosine 0.998).
            self.W_rec = self.W_rec + self.recurrent_gain * w
            self._stored_pattern = p
            self._stored_patterns.append(p)
```

Add these three members after `store()`:

```python
    @property
    def n_stored(self) -> int:
        """How many patterns are currently imprinted."""
        return len(self._stored_patterns)

    def clear(self) -> None:
        """Forget everything. Required for episodic memory: without it, imprints
        persist across episodes and accumulate into an uninterpretable mixture."""
        with torch.no_grad():
            self.W_rec = torch.zeros_like(self.W_rec)
            self._stored_pattern = None
            self._stored_patterns = []

    def familiarity(self, content: torch.Tensor) -> torch.Tensor:
        """``[B, content_dim]`` -> ``[B]`` Hopfield field alignment, one scalar per item.

        High when the content's sparse pattern sits near a stored attractor, so it
        answers "have I been here?". Reuses ``W_rec`` rather than a lookup table, so
        familiarity stays a property of the attractor. Zeros when nothing is stored.
        """
        with torch.no_grad():
            drive = self.fc_in(content)                       # [B, n_neurons]
            k = max(1, int(self.sparsity * self.n_neurons))
            idx = torch.topk(drive, k, dim=1).indices
            p = torch.zeros_like(drive)
            p.scatter_(1, idx, 1.0)
            s = 2.0 * p - 1.0
            return ((s @ self.W_rec) * s).sum(dim=1) / self.n_neurons
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/regions/test_hippocampus.py -q`
Expected: PASS.

**If `test_recall_discriminates_between_stored_states` still fails**, stop and report it. That would mean accumulation did not restore discriminability, which contradicts the measurement above and blocks EXP-030 entirely. Do not weaken the threshold to make it pass.

- [ ] **Step 5: Full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q -m "not slow"`
Expected: no fewer tests than base. `Brain.remember`/`step` call `store`, so if any existing test depends on single-pattern behavior it will surface here. Fix the seam, never the test.

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/regions/hippocampus.py tests/regions/test_hippocampus.py
git commit -m "fix(hippocampus): accumulate stored patterns, add clear and familiarity"
```

---

### Task 2: Pluggable policy readout in `reinforce.py`

**Files:**
- Modify: `src/neuromorphic/training/reinforce.py`
- Test: `tests/training/test_reinforce_readout.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `action_distribution(brain, head, obs, *, generator=None, store=False, recall=False, feature_fn=None)`.
  - `greedy_action(brain, head, obs, *, generator=None, store=False, recall=False, feature_fn=None)`.
  - `train_episode(..., store=False, recall=False, feature_fn=None)` passing all three through.
  - `feature_fn` defaults to `concept_rate`, so omitting it is bit-identical to today.

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_reinforce_readout.py
import torch

from neuromorphic.brain import Brain
from neuromorphic.training.reinforce import (
    action_distribution,
    concept_rate,
    make_policy_head,
)


def test_default_readout_is_bit_identical_to_concept_rate():
    """The compatibility gate: every 024-029 driver must be unaffected."""
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain, "linear")
    obs = [0, 0, 4, 4]

    _, logits_default = action_distribution(
        brain, head, obs, generator=torch.Generator().manual_seed(3)
    )
    _, logits_explicit = action_distribution(
        brain, head, obs, generator=torch.Generator().manual_seed(3), feature_fn=concept_rate
    )
    assert torch.equal(logits_default, logits_explicit)


def test_feature_fn_receives_the_step_output_and_its_result_drives_the_head():
    brain = Brain(grid_n=5, seed=0)
    seen = {}

    def fake_features(out):
        seen["keys"] = set(out.keys())
        return torch.zeros(brain.content)

    head = make_policy_head(brain, "linear")
    _, logits = action_distribution(
        brain, head, [0, 0, 4, 4],
        generator=torch.Generator().manual_seed(0), feature_fn=fake_features,
    )
    assert "concept" in seen["keys"]
    # zero features -> logits are exactly the head bias
    assert torch.allclose(logits, head.bias)


def test_recall_flag_reaches_the_brain():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain, "linear")
    captured = {}
    original = brain.step

    def spy(obs, **kwargs):
        captured.update(kwargs)
        return original(obs, **kwargs)

    brain.step = spy
    action_distribution(
        brain, head, [0, 0, 4, 4],
        generator=torch.Generator().manual_seed(0), store=True, recall=True,
    )
    assert captured["store"] is True
    assert captured["recall"] is True
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce_readout.py -q`
Expected: FAIL with `TypeError: action_distribution() got an unexpected keyword argument 'feature_fn'`.

- [ ] **Step 3: Implement the seam**

Replace `action_distribution` with:

```python
def action_distribution(
    brain,
    head: nn.Module,
    obs,
    *,
    generator: torch.Generator | None = None,
    store: bool = False,
    recall: bool = False,
    feature_fn=None,
) -> tuple[Categorical, torch.Tensor]:
    """One forward pass -> a categorical policy from the head on the chosen features.

    The brain runs under ``no_grad`` (frozen feature extractor; also avoids backprop
    through the spiking unroll). ``feature_fn`` selects what the head reads and defaults
    to the sensory concept, so omitting it reproduces v1 exactly. ``store``/``recall``
    engage the hippocampal pathway (both off by default, as in v1).
    """
    with torch.no_grad():
        out = brain.step(obs, store=store, recall=recall, record=False, generator=generator)
    features = concept_rate(out) if feature_fn is None else feature_fn(out)
    logits = head(features)
    return Categorical(logits=logits), logits
```

Replace `greedy_action` with:

```python
def greedy_action(
    brain,
    head: nn.Module,
    obs,
    *,
    generator: torch.Generator | None = None,
    store: bool = False,
    recall: bool = False,
    feature_fn=None,
) -> int:
    """The argmax-logit action (deterministic eval policy)."""
    _, logits = action_distribution(
        brain, head, obs, generator=generator,
        store=store, recall=recall, feature_fn=feature_fn,
    )
    return int(logits.argmax())
```

In `train_episode`, add three keyword parameters after `normalize_advantages: bool = False`:

```python
    store: bool = False,
    recall: bool = False,
    feature_fn=None,
```

and change its single `action_distribution` call to:

```python
        dist, _ = action_distribution(
            brain, head, obs, generator=generator,
            store=store, recall=recall, feature_fn=feature_fn,
        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce_readout.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q -m "not slow"`
Expected: no fewer tests than base. `tests/training/test_reinforce.py` exercises the old signatures and must still pass untouched.

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/training/reinforce.py tests/training/test_reinforce_readout.py
git commit -m "feat(training): pluggable policy readout and hippocampal flags"
```

---

### Task 3: Memory readout and revisit instrumentation in `CubeRunner`

**Files:**
- Modify: `src/neuromorphic/training/cube_baseline.py`
- Test: `tests/training/test_memory_readout.py`

**Interfaces:**
- Consumes: `Hippocampus.clear`/`familiarity`/`n_stored` (Task 1); `feature_fn`/`store`/`recall` (Task 2).
- Produces:
  - `CubeConfig.readout: str = "concept"` and `CubeConfig.n_revisit_probe: int = 0`.
  - `MemoryReadout(mode, rng)` with `.reset()`, `.__call__(out) -> Tensor`, `.width(content) -> int`.
  - `feature_width(cfg) -> int`.
  - `run_cube_baseline` records `readout`, `revisit_rate`, `mean_n_stored`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_memory_readout.py
import random

import pytest
import torch

from neuromorphic.brain import Brain
from neuromorphic.encoders import cube_encoder
from neuromorphic.training.cube_baseline import (
    CubeConfig,
    MemoryReadout,
    feature_width,
    run_cube_baseline,
)


def _out(brain):
    import numpy as np
    return brain.step(
        np.zeros(24, dtype=np.int64), store=True, recall=True,
        generator=torch.Generator().manual_seed(0),
    )


def test_feature_widths_per_mode():
    assert feature_width(CubeConfig(readout="concept")) == 64
    assert feature_width(CubeConfig(readout="memory")) == 129
    assert feature_width(CubeConfig(readout="memory_shuffled")) == 129


def _brain():
    return Brain(encoder=cube_encoder(), n_obs=144, obs_width=24, n_actions=6, seed=0)


def test_concept_mode_returns_only_the_concept():
    b = _brain()
    r = MemoryReadout("concept", random.Random(0), b)
    r.reset()
    assert r(_out(b)).shape == (64,)


def test_memory_mode_appends_recall_and_familiarity():
    b = _brain()
    r = MemoryReadout("memory", random.Random(0), b)
    r.reset()
    assert r(_out(b)).shape == (129,)


def test_shuffled_mode_matches_width_but_differs_in_content():
    """The shuffle-null must hold width fixed while destroying correspondence."""
    b = _brain()
    real = MemoryReadout("memory", random.Random(0), b)
    shuf = MemoryReadout("memory_shuffled", random.Random(0), b)
    real.reset()
    shuf.reset()
    outs = [_out(b) for _ in range(4)]
    f_real = torch.stack([real(o) for o in outs])
    f_shuf = torch.stack([shuf(o) for o in outs])
    assert f_real.shape == f_shuf.shape == (4, 129)
    # concept half identical, memory half differs once more than one state is cached
    assert torch.allclose(f_real[:, :64], f_shuf[:, :64])
    assert not torch.allclose(f_real[-1, 64:], f_shuf[-1, 64:])


def test_reset_clears_the_cache():
    b = _brain()
    r = MemoryReadout("memory_shuffled", random.Random(0), b)
    r.reset()
    r(_out(b))
    assert len(r._cache) == 1
    r.reset()
    assert len(r._cache) == 0


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown readout"):
        MemoryReadout("nonsense", random.Random(0), _brain()).reset()


def test_run_records_readout_and_revisit_rate():
    cfg = CubeConfig(arm="regionalized", readout="memory", depth=1, seed=0,
                     episodes=3, max_depth=1, sigma=0.0)
    rec = run_cube_baseline(cfg)
    assert rec["readout"] == "memory"
    assert 0.0 <= rec["revisit_rate"] <= 1.0
    assert rec["mean_n_stored"] >= 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_memory_readout.py -q`
Expected: FAIL with `ImportError: cannot import name 'MemoryReadout'`.

- [ ] **Step 3: Add the config fields**

In `CubeConfig`, after `n_actions: int = 6`, add:

```python
    readout: str = "concept"    # "concept" | "memory" | "memory_shuffled"
```

`arm` stays the network topology axis and is unchanged. EXP-030 always uses `arm="regionalized"`.

- [ ] **Step 4: Implement `MemoryReadout` and `feature_width`**

Add after `ShellCubeEnv`:

```python
FAMILIARITY_WIDTH = 1


def feature_width(cfg: CubeConfig) -> int:
    """Width of the policy head's input for a config's readout mode."""
    if cfg.readout == "concept":
        return cfg.content
    if cfg.readout in ("memory", "memory_shuffled"):
        return cfg.content * 2 + FAMILIARITY_WIDTH
    raise ValueError(f"unknown readout {cfg.readout!r}")


class MemoryReadout:
    """Builds the policy head's input from a ``brain.step`` output.

    ``concept``: the sensory concept alone, exactly v1.
    ``memory``: concept, the hippocampal recall code, and a familiarity scalar.
    ``memory_shuffled``: the same three, but recall and familiarity are computed from a
    DIFFERENT state visited earlier this episode. Both memory readouts stay real,
    in-distribution and correctly scaled; only the correspondence between the agent's
    current state and the memory it receives is destroyed. That isolates memory content
    from head width, which is the confound the control exists for.
    """

    def __init__(self, mode: str, rng: random.Random, brain):
        self.mode = mode
        self.rng = rng
        self.brain = brain
        self._cache: list[torch.Tensor] = []

    def reset(self) -> None:
        """Drop the episode's cached concepts. Call at every episode start."""
        if self.mode not in ("concept", "memory", "memory_shuffled"):
            raise ValueError(f"unknown readout {self.mode!r}")
        self._cache = []

    def __call__(self, out: dict) -> torch.Tensor:
        concept = concept_rate(out)                     # [content]
        if self.mode == "concept":
            return concept

        snapshot = out["concept"].mean(dim=0)           # [B, content]
        if self.mode == "memory_shuffled" and len(self._cache) >= 1:
            query = self.rng.choice(self._cache)
        else:
            query = snapshot
        self._cache.append(snapshot)

        recall = self.brain.hippo(
            query.unsqueeze(0).expand(self.brain.T, *query.shape)
        ).mean(dim=0)[0]                                # [content]
        fam = self.brain.hippo.familiarity(query)       # [B]
        return torch.cat([concept, recall, fam[:1]])
```

The brain is a constructor argument rather than a separate `bind()` call, so a readout can never be invoked before it has one.

Add `concept_rate` to the existing `reinforce` import line in this file.

- [ ] **Step 5: Wire it into `run_cube_baseline`**

Replace the trained-arm branch body between `agent = make_agent(cfg)` and the training loop:

```python
        agent = make_agent(cfg)
        torch.manual_seed(cfg.seed)  # head init and sampling stream matched across arms
        readout = MemoryReadout(cfg.readout, random.Random(cfg.seed), agent)
        use_memory = cfg.readout != "concept"
        spec = AblationSpec(kind="gaussian", dose=cfg.sigma, seed=cfg.seed) if cfg.sigma else None
        head = AblatedConcept(
            nn.Linear(feature_width(cfg), cfg.n_actions), spec, width=feature_width(cfg)
        )
        optimizer = torch.optim.Adam(policy_parameters(head), lr=cfg.lr)
        env = ShellCubeEnv(
            train_states, random.Random(cfg.seed),
            scramble_depth=cfg.depth, max_steps=max_steps_for(cfg.depth),
        )
        baseline = 0.0
        revisits, steps_total, stored_counts = 0, 0, []
        for _ in range(cfg.episodes):
            readout.reset()
            if use_memory:
                agent.hippo.clear()
            stats = train_episode(
                agent, head, env, optimizer,
                gamma=cfg.gamma, baseline=baseline, generator=generator,
                max_steps=max_steps_for(cfg.depth),
                entropy_beta=cfg.entropy_beta,
                normalize_advantages=cfg.normalize_advantages,
                store=use_memory, recall=use_memory, feature_fn=readout,
            )
            baseline = ema(baseline, stats["mean_return"], cfg.baseline_beta)
            steps_total += stats["steps"]
            revisits += len(env.visited) - len(set(env.visited))
            stored_counts.append(agent.hippo.n_stored if use_memory else 0)
```

Add `import torch.nn as nn` at the top of the file.

The head is built directly with `nn.Linear(feature_width(cfg), cfg.n_actions)` rather than `make_policy_head`, because `make_policy_head` hardcodes `brain.content` as the input width and the memory readouts are 129 wide.

**Revisit accounting needs the env to track states**, because the readout's own cache grows once per step and so can never show a repeat. Add to `ShellCubeEnv.__init__`:

```python
        self.visited: list[tuple[int, ...]] = []
```

At the end of `ShellCubeEnv.reset`, after the `super().reset(...)` result is obtained, record the start state and return:

```python
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options is None:
            options = {"state": self._pool_rng.choice(self._pool)}
        result = super().reset(seed=seed, options=options)
        self.visited = [self._state]
        return result
```

And add a `step` override that appends each state reached:

```python
    def step(self, action):
        result = super().step(action)
        self.visited.append(self._state)
        return result
```

`len(env.visited) - len(set(env.visited))` is then the number of repeat visits in the episode just finished.

- [ ] **Step 6: Extend the record**

Replace the `record` dict's opening keys to include:

```python
        "readout": cfg.readout,
        "revisit_rate": (revisits / steps_total) if steps_total else 0.0,
        "mean_n_stored": (sum(stored_counts) / len(stored_counts)) if stored_counts else 0.0,
```

For the `random` arm, set `revisits`, `steps_total` and `stored_counts` to `0`, `0` and `[]` before the record is built so those keys always exist.

- [ ] **Step 7: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_memory_readout.py tests/training/test_cube_baseline.py -q`
Expected: PASS. The existing EXP-029 tests must still pass, since `readout` defaults to `concept`.

- [ ] **Step 8: Commit**

```bash
git add src/neuromorphic/training/cube_baseline.py tests/training/test_memory_readout.py
git commit -m "feat(training): memory readout modes and revisit instrumentation"
```

---

### Task 4: EXP-030 driver

**Files:**
- Create: `experiments/030_memory_engagement/__init__.py` (empty)
- Create: `experiments/030_memory_engagement/run.py`
- Create: `experiments/030_memory_engagement/aggregate.py`

**Interfaces:**
- Consumes: `CubeConfig`, `run_cube_baseline` (Task 3).
- Produces: one `outputs/exp030_<readout>_d<depth>_s<seed>.json` per run, plus `outputs/030_curve.md`.

Drivers are not unit tested in this repo (024-029 have none); verification is the smoke run in Step 4.

- [ ] **Step 1: Write `run.py`**

```python
# experiments/030_memory_engagement/run.py
"""EXP-030 driver: does episodic memory content improve cube solving?

Phase 1 runs the concept arm alone and reports its revisit rate. That is a GATE: with a
2d+3 step budget, if the policy rarely revisits a state then cycle-avoidance memory has
nothing to do, and a null would say nothing about memory. Phase 2 runs the two memory
arms only after that number has been read.

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/030_memory_engagement/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import CubeConfig, run_cube_baseline

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
DEPTHS = [3, 4, 5, 6]
MEMORY_ARMS = ["memory", "memory_shuffled"]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def _fan_out(configs, workers: int) -> list[dict]:
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if i % 10 == 0 or i == len(configs):
                print(f"  {i}/{len(configs)}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-gate", action="store_true",
                    help="run the memory arms without pausing on the revisit gate")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(arm="regionalized", episodes=args.episodes, sigma=0.0,
                  out_dir=args.out_dir, tag="exp030")

    print(f"phase 1: concept arm, {len(args.depths) * len(args.seeds)} runs")
    concept_records = _fan_out(
        [CubeConfig(readout="concept", depth=d, seed=s, **common)
         for d in args.depths for s in args.seeds],
        args.workers,
    )

    print("\nREVISIT GATE (the number that decides whether this experiment can work):")
    for d in args.depths:
        rates = [r["revisit_rate"] for r in concept_records if r["depth"] == d]
        mean = sum(rates) / len(rates) if rates else 0.0
        print(f"  depth {d}: mean revisit rate {mean:.3f}")
    print("  If these are near zero there are no cycles to avoid, and a null in the")
    print("  memory arms would be a statement about the task, not about memory.\n")

    if not args.skip_gate:
        reply = input("continue to the memory arms? [y/N] ").strip().lower()
        if reply != "y":
            print("stopped at the gate. Records for the concept arm are already written.")
            return

    rest = [CubeConfig(readout=r, depth=d, seed=s, **common)
            for r in MEMORY_ARMS for d in args.depths for s in args.seeds]
    print(f"phase 2: memory arms, {len(rest)} runs")
    _fan_out(rest, args.workers)
    print(f"done. one record per run in {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `aggregate.py`**

```python
# experiments/030_memory_engagement/aggregate.py
"""Aggregate EXP-030 records.

The headline is the PAIRED memory minus memory_shuffled difference at matched
(depth, seed): both arms have identical head width, so a gap is memory content rather
than capacity. The revisit rate and mean stored-pattern count are reported alongside,
because a null with near-zero revisits is a statement about the task.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(out_dir.glob("exp030_*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=HERE / "outputs")
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "030_curve.md")
    args = ap.parse_args()

    records = load(args.runs)
    if not records:
        raise SystemExit(f"no run records found in {args.runs}")

    cells = defaultdict(list)
    revisit = defaultdict(list)
    stored = defaultdict(list)
    by_seed = defaultdict(dict)
    for r in records:
        cells[(r["readout"], r["depth"])].append(r["success_rate"])
        revisit[(r["readout"], r["depth"])].append(r["revisit_rate"])
        stored[(r["readout"], r["depth"])].append(r["mean_n_stored"])
        by_seed[(r["depth"], r["seed"])][r["readout"]] = r["success_rate"]

    lines = [
        "# EXP-030 memory engagement",
        "",
        "Primary test is the paired memory minus memory_shuffled column: identical head",
        "width, so a gap is memory content rather than capacity. A near-zero revisit rate",
        "means there were no cycles to avoid, and any null must be read that way.",
        "",
        "| depth | concept | memory | shuffled | paired mem-shuf | revisit rate | mean stored | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for depth in sorted({d for _, d in cells}):
        def mean(mode, table=cells, fmt="{:.0f}%", scale=100):
            vals = table.get((mode, depth), [])
            return fmt.format(scale * sum(vals) / len(vals)) if vals else "n/a"

        pairs = [v["memory"] - v["memory_shuffled"]
                 for (d, _), v in by_seed.items()
                 if d == depth and "memory" in v and "memory_shuffled" in v]
        diff = f"{100 * sum(pairs) / len(pairs):+.0f} pts" if pairs else "n/a"
        lines.append(
            f"| {depth} | {mean('concept')} | {mean('memory')} | {mean('memory_shuffled')} "
            f"| {diff} | {mean('concept', revisit, '{:.3f}', 1)} "
            f"| {mean('memory', stored, '{:.1f}', 1)} | {len(pairs)} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add `__init__.py` and confirm gitignore**

Create an empty `experiments/030_memory_engagement/__init__.py`.

Run: `git check-ignore -v experiments/030_memory_engagement/outputs/x.json`
Expected: prints the matching rule (`experiments/*/outputs/` already exists). If nothing prints, add it to `.gitignore`.

- [ ] **Step 4: Smoke the driver**

```bash
.venv/Scripts/python.exe experiments/030_memory_engagement/run.py --seeds 0 --episodes 5 --depths 3 --workers 2 --skip-gate
.venv/Scripts/python.exe experiments/030_memory_engagement/aggregate.py
```

Expected: phase 1 prints the revisit gate, phase 2 completes, and the aggregate table has a `paired mem-shuf` column plus revisit and stored columns. Numbers are garbage at 5 episodes; the point is the wiring.

Then delete the smoke records: `rm experiments/030_memory_engagement/outputs/exp030_*.json`

- [ ] **Step 5: Full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q -m "not slow"`
Expected: no fewer tests than base.

- [ ] **Step 6: Commit**

```bash
git add experiments/030_memory_engagement/
git commit -m "feat(exp030): memory engagement driver and aggregator"
```

---

## After the plan

The build ends at Task 4. Running the experiment is separate:

1. Run phase 1 and **read the revisit gate before continuing**. If revisit rates are near zero, stop and redesign (most likely longer step budgets); do not spend 96 more runs on a null that would say nothing about memory.
2. Pin the depth grid to EXP-029's knee once that experiment lands, replacing the 3-6 default.
3. Write `experiments/030_memory_engagement/RESULTS.md` against the spec's section-6 contract, marking each pre-registered claim confirmed or refuted, with provenance.

Pre-registered expectation to check against, from the measured capacity data: the completion code should contribute at depth 3 (9 steps) and be largely spent by depth 6 (15 steps), while familiarity should hold throughout. A memory win only at shallow depths is the predicted pattern, not a surprise.
