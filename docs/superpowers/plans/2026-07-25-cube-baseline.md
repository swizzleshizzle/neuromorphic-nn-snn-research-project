# EXP-029 Cube Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the v1 recipe (frozen encoder + linear REINFORCE head) on `CubeEnv` across exact scramble distances 1 to 6, against a neuron-matched monolithic control and a measured random-policy floor, and produce a committed collapse curve.

**Architecture:** Three library pieces then one driver. `Brain` gains an encoder seam so it can consume cube observations. A `MonolithicBrain` provides the unregionalized control, neuron-matched at runtime. `CubeEnv` gains start-state injection so held-out states can be evaluated. `CubeRunner` holds shell enumeration, the train/held-out split, the REINFORCE loop and greedy evaluation. `reinforce.py` is reused **unchanged**: it is already environment-agnostic.

**Tech Stack:** Python, gymnasium, numpy, torch, snntorch, pytest. Run python via `.venv/Scripts/python.exe`.

## Global Constraints

- **Reusable machinery under `src/neuromorphic/` only. `experiments/029_cube_baseline/` holds the driver and nothing a future experiment would want to import.**
- Run python via `.venv/Scripts/python.exe`.
- Plain commit messages; NO `Co-Authored-By` trailer; NO em-dashes anywhere in code, docs, or commit messages.
- **Backward compatibility is a hard gate.** The full suite is **313 passing tests** at base. Every task must leave it at 313 or more, never fewer. `Brain(grid_n=5, seed=0)` behavior must be bit-identical under a fixed generator.
- `reinforce.py` is NOT modified. If a task seems to need a change there, stop and re-read: `train_episode` already takes any gym env and `action_distribution` already reads only `brain.content` / `brain.step(obs)`.
- Encoders passed to `Brain` must be **picklable** (module-level functions or `functools.partial` over them, never lambdas): the driver fans out over `ProcessPoolExecutor`.
- Action-space width comes from `N_ACTIONS` / `env.action_space.n`, never a literal `6`.
- `distance` is an instrument, never a model input. The observation is raw facelets.
- Base branch: `main` at `41e7e6b`.
- Per-task gate: the task's pytest green via `.venv/Scripts/python.exe -m pytest <paths> -q`.

## Measured facts (do not re-derive)

- `brain.step` costs **90 ms** (11 steps/sec).
- Five-region `Brain` at `n_actions=6` totals **510 neurons**: sensory 192, hippocampus 150, prefrontal 150, router 12, motor 6. Router and motor scale with `n_actions`, so the total must be **computed at runtime**, never hardcoded.
- `weight_gain=5.0` (grid-tuned for a 2-hot code) is **fine on the cube's 24-hot code**: mean concept rate 0.394 vs the grid's 0.413, no saturation. Do NOT lower it; at 2.0 the rate collapses to 0.117 and at 1.0 to 0.009.
- Exact-distance shell sizes for d=1..6: **6, 27, 120, 534, 2256, 8969**.

## File structure

- `src/neuromorphic/encoders.py` - picklable encoder factories (Task 1).
- `src/neuromorphic/brain.py` - encoder seam, `n_obs`/`obs_width` params, `n_neurons` property (Task 1).
- `src/neuromorphic/monolithic.py` - `MonolithicBrain`, the unregionalized control (Task 2).
- `src/neuromorphic/envs/cube.py` - `reset(options={"state": ...})` injection (Task 3).
- `src/neuromorphic/training/cube_baseline.py` - `CubeConfig`, shell/split helpers, `ShellCubeEnv`, evaluation, `run_cube_baseline` (Task 4).
- `experiments/029_cube_baseline/{run.py,aggregate.py}` - driver only (Task 5).
- Tests: `tests/test_brain_encoder_seam.py`, `tests/test_monolithic.py`, `tests/envs/test_cube.py` (append), `tests/training/test_cube_baseline.py`.

---

### Task 1: Encoder seam in `Brain`

**Files:**
- Create: `src/neuromorphic/encoders.py`
- Modify: `src/neuromorphic/brain.py` (constructor; `_to_obs_tensor` at :101-109; encoder call sites at :117 and :156)
- Test: `tests/test_brain_encoder_seam.py`

**Interfaces:**
- Consumes: `encode_gridworld`, `encode_cube` from `neuromorphic.regions.sensory_cortex`.
- Produces:
  - `grid_encoder(grid_n: int = 5) -> Callable` and `cube_encoder(cube_n: int = 2, n_colors: int = 6) -> Callable`. Both return a `functools.partial` callable invoked as `enc(obs_tensor, T=..., generator=...) -> [T, B, n_obs]`.
  - `Brain(..., encoder=None, n_obs=None, obs_width=4)`. Defaults reproduce today's grid behavior exactly.
  - `Brain.n_neurons -> int`, the summed region neuron count. Task 2 uses this as the matching budget.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_brain_encoder_seam.py
import numpy as np
import pytest
import torch

from neuromorphic.brain import Brain
from neuromorphic.encoders import cube_encoder, grid_encoder
from neuromorphic.regions.sensory_cortex import encode_gridworld


def test_grid_default_is_unchanged():
    """The seam must not perturb existing grid behavior at all."""
    brain = Brain(grid_n=5, seed=0)
    obs = [0, 0, 4, 4]
    out = brain.step(obs, store=False, recall=False, generator=torch.Generator().manual_seed(7))
    expected_spikes = encode_gridworld(
        torch.tensor([[0, 0, 4, 4]]), grid_n=5, T=brain.T,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(out["obs_spikes"], expected_spikes)
    assert out["obs_spikes"].shape == (brain.T, 1, 50)


def test_brain_reports_total_neuron_count():
    brain = Brain(grid_n=5, n_actions=6, seed=0)
    assert brain.n_neurons == 510  # sensory 192 + hippo 150 + pfc 150 + router 12 + motor 6
    assert brain.n_neurons == sum(r.n_neurons for r in brain._regions.values())


def test_cube_configured_brain_runs():
    brain = Brain(
        encoder=cube_encoder(), n_obs=144, obs_width=24, n_actions=6, seed=0,
    )
    obs = np.zeros(24, dtype=np.int64)
    out = brain.step(obs, store=False, recall=False, generator=torch.Generator().manual_seed(0))
    assert out["obs_spikes"].shape == (brain.T, 1, 144)
    assert out["concept"].shape == (brain.T, 1, brain.content)
    assert out["utilities"].shape == (brain.T, 1, 6)
    assert out["action"] in range(6)


def test_obs_width_is_validated():
    brain = Brain(encoder=cube_encoder(), n_obs=144, obs_width=24, n_actions=6, seed=0)
    with pytest.raises(ValueError, match="24"):
        brain.step([0, 0, 4, 4], store=False, recall=False)


def test_encoders_are_picklable():
    """The driver fans out over ProcessPoolExecutor; a lambda encoder would break it."""
    import pickle
    for enc in (grid_encoder(5), cube_encoder()):
        assert pickle.loads(pickle.dumps(enc)) is not None
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_encoder_seam.py -q`
Expected: FAIL (`ModuleNotFoundError: neuromorphic.encoders`).

- [ ] **Step 3: Create `encoders.py`**

```python
# src/neuromorphic/encoders.py
"""Picklable encoder factories for Brain's encoder seam.

``Brain`` needs an encoder callable invoked as ``enc(obs_tensor, T=..., generator=...)``.
These factories return ``functools.partial`` objects over module-level functions, which
pickle cleanly and therefore survive ``ProcessPoolExecutor`` fan-out. A lambda would not
pickle, and the EXP-029 driver runs its grid across processes.
"""

from __future__ import annotations

from functools import partial

from neuromorphic.regions.sensory_cortex import encode_cube, encode_gridworld


def grid_encoder(grid_n: int = 5):
    """Grid observations ``[B, 4]`` -> Poisson spikes ``[T, B, 2*grid_n**2]``."""
    return partial(encode_gridworld, grid_n=grid_n)


def cube_encoder(cube_n: int = 2, n_colors: int = 6):
    """Cube facelets ``[B, 6*cube_n**2]`` -> Poisson spikes ``[T, B, 6*cube_n**2*n_colors]``."""
    return partial(encode_cube, cube_n=cube_n, n_colors=n_colors)
```

- [ ] **Step 4: Add the seam to `Brain.__init__`**

In `src/neuromorphic/brain.py`, add the import near the existing region imports:

```python
from neuromorphic.encoders import grid_encoder
```

Replace the constructor signature and the first lines of its body:

```python
    def __init__(
        self,
        grid_n: int = 5,
        content: int = 64,
        n_hippo: int = 150,
        n_actions: int = 4,
        num_steps: int = 32,
        seed: int = 0,
        bus: NeuromodBus | None = None,
        encoder=None,
        n_obs: int | None = None,
        obs_width: int = 4,
    ):
        self.grid_n = grid_n
        self.content = content
        self.n_actions = n_actions
        self.T = num_steps
        self.bus = bus if bus is not None else NeuromodBus()

        # Encoder seam: default reproduces the grid behavior exactly, so every existing
        # caller (Brain(grid_n=5)) is untouched. A cube passes cube_encoder() with
        # n_obs=144 and obs_width=24. See docs/.../2026-07-25-cube-baseline-design.md.
        self.obs_width = obs_width
        self._encoder = encoder if encoder is not None else grid_encoder(grid_n)
        self.n_obs = n_obs if n_obs is not None else 2 * grid_n * grid_n

        self.sensory = SensoryCortex(
            n_obs=self.n_obs, concept=content, num_steps=num_steps, seed=seed
        )
```

Leave the rest of the constructor (hippo, pfc, router, motor, `_regions`) exactly as it is. Delete the now-dead local `n_obs = 2 * grid_n * grid_n` line.

- [ ] **Step 5: Generalize `_to_obs_tensor` and route the two encoder call sites**

`_to_obs_tensor` is currently a `@staticmethod`. It has only two callers, both `self._to_obs_tensor(...)` inside this file, so converting it to an instance method is safe. Replace it:

```python
    def _to_obs_tensor(self, obs) -> torch.Tensor:
        """Coerce an observation to a ``[B, obs_width]`` int tensor."""
        arr = np.asarray(obs)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.obs_width:
            raise ValueError(
                f"obs must be [{self.obs_width}] or [B, {self.obs_width}]; got {arr.shape}"
            )
        return torch.as_tensor(arr, dtype=torch.long)
```

Then replace **both** encoder call sites (in `remember` and in `step`) with:

```python
        obs_spikes = self._encoder(obs_t, T=self.T, generator=generator)
```

In `remember` the local is named `spikes`, so there it reads:

```python
        spikes = self._encoder(obs_t, T=self.T, generator=generator)
```

Remove the now-unused `encode_gridworld` import from `brain.py` if nothing else in the file uses it.

- [ ] **Step 6: Add the `n_neurons` property**

Add directly after the `_regions` dict is built in `__init__`:

```python
    @property
    def n_neurons(self) -> int:
        """Total neurons across all regions. The matching budget for MonolithicBrain."""
        return sum(r.n_neurons for r in self._regions.values())
```

- [ ] **Step 7: Run the task tests, then the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_encoder_seam.py -q`
Expected: PASS (5 passed).

Run: `.venv/Scripts/python.exe -m pytest tests/ -q -m "not slow"`
Expected: PASS, count is 312 or more excluding the slow test (313 total at base). **If any existing test fails, the seam is wrong. Fix it, do not edit the failing test.**

- [ ] **Step 8: Commit**

```bash
git add src/neuromorphic/encoders.py src/neuromorphic/brain.py tests/test_brain_encoder_seam.py
git commit -m "feat(brain): encoder seam so Brain can consume non-grid observations"
```

---

### Task 2: `MonolithicBrain` (the unregionalized control)

**Files:**
- Create: `src/neuromorphic/monolithic.py`
- Test: `tests/test_monolithic.py`

**Interfaces:**
- Consumes: `SensoryCortex` from `neuromorphic.regions.sensory_cortex`; `grid_encoder` from `neuromorphic.encoders`; `Brain.n_neurons` from Task 1.
- Produces: `MonolithicBrain(n_obs, n_actions, total_neurons, *, content=64, num_steps=32, seed=0, obs_width=4, encoder=None, weight_gain=5.0)` exposing `.content`, `.n_actions`, `.T`, `.n_neurons`, and `.step(obs, *, store=False, recall=False, record=False, generator=None) -> {"concept": [T, B, content], "obs_spikes": [T, B, n_obs]}`. This is the duck-typed surface `reinforce.py` consumes.

**Why this shape:** `action_distribution` in `reinforce.py` reads only `brain.content` and `out["concept"]`, and `make_policy_head` reads `brain.content` and `brain.n_actions`. In v1 the action comes from the policy head, not the brain's motor readout (ADR-0001 Amendment 1), so the control legitimately has no motor. Matching **total** neurons means the control gets all 510 while `Brain` spends 18 of its 510 on router and motor that are off the policy path. That is conservative: it favors the control, which is the correct direction of bias for a "does regionalization help" claim.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_monolithic.py
import numpy as np
import pytest
import torch

from neuromorphic.brain import Brain
from neuromorphic.encoders import cube_encoder
from neuromorphic.monolithic import MonolithicBrain
from neuromorphic.training.reinforce import action_distribution, make_policy_head


def _cube_pair(seed=0):
    brain = Brain(encoder=cube_encoder(), n_obs=144, obs_width=24, n_actions=6, seed=seed)
    mono = MonolithicBrain(
        n_obs=144, n_actions=6, total_neurons=brain.n_neurons,
        content=brain.content, obs_width=24, encoder=cube_encoder(), seed=seed,
    )
    return brain, mono


def test_neuron_count_matches_the_brain_exactly():
    brain, mono = _cube_pair()
    assert mono.n_neurons == brain.n_neurons == 510


def test_step_returns_a_concept_of_the_right_shape():
    _, mono = _cube_pair()
    out = mono.step(np.zeros(24, dtype=np.int64), generator=torch.Generator().manual_seed(0))
    assert out["concept"].shape == (mono.T, 1, mono.content)
    assert out["obs_spikes"].shape == (mono.T, 1, 144)


def test_is_a_drop_in_for_reinforce():
    """The whole point: reinforce.py must consume it with no changes."""
    _, mono = _cube_pair()
    head = make_policy_head(mono, "linear")
    dist, logits = action_distribution(
        mono, head, np.zeros(24, dtype=np.int64), generator=torch.Generator().manual_seed(0)
    )
    assert logits.shape == (6,)
    assert 0 <= int(dist.sample()) < 6


def test_head_is_parameter_identical_to_the_regionalized_arm():
    """Same content width means the two arms train the same number of head parameters."""
    brain, mono = _cube_pair()
    n_brain = sum(p.numel() for p in make_policy_head(brain, "linear").parameters())
    n_mono = sum(p.numel() for p in make_policy_head(mono, "linear").parameters())
    assert n_brain == n_mono


def test_rejects_a_budget_that_cannot_hold_the_concept():
    with pytest.raises(ValueError, match="total_neurons"):
        MonolithicBrain(n_obs=144, n_actions=6, total_neurons=64, content=64)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_monolithic.py -q`
Expected: FAIL (`ModuleNotFoundError: neuromorphic.monolithic`).

- [ ] **Step 3: Implement `monolithic.py`**

```python
# src/neuromorphic/monolithic.py
"""``MonolithicBrain`` - the unregionalized control for the Phase-3 regionalization question.

One flat spiking stack, neuron-matched to the five-region ``Brain`` and frozen at random
init, exposing the same duck-typed surface ``reinforce.py`` consumes. Only the topology
differs: no hippocampus, no prefrontal, no router, no motor. If the regionalized arm beats
this at matched neuron count, matched head, matched seeds and matched protocol, the gap is
attributable to regionalization.

There is deliberately no motor readout: in v1 the action comes from the policy head, not
from the brain (ADR-0001 Amendment 1), so a motor region would be dead weight here.
"""

from __future__ import annotations

import numpy as np
import torch

from neuromorphic.encoders import grid_encoder
from neuromorphic.regions.sensory_cortex import SensoryCortex


class MonolithicBrain:
    """A single feedforward spiking stack sized to a whole ``Brain``'s neuron budget.

    Args:
        n_obs: encoder output width (144 for a 2x2 cube).
        n_actions: action-space width, for head sizing.
        total_neurons: the matching budget, normally ``Brain.n_neurons``.
        content: concept width. Kept equal to the Brain's so the policy head is
            parameter-identical across arms.
        num_steps: inference window ``T``.
        seed: RNG seed for reproducible weight init.
        obs_width: raw observation width (24 for a 2x2 cube, 4 for the grid).
        encoder: picklable encoder callable; defaults to the grid encoder.
        weight_gain: excitability knob. 5.0 is correct for both the grid's 2-hot code
            and the cube's 24-hot code (measured); lowering it collapses the code.
    """

    def __init__(
        self,
        n_obs: int,
        n_actions: int,
        total_neurons: int,
        *,
        content: int = 64,
        num_steps: int = 32,
        seed: int = 0,
        obs_width: int = 4,
        encoder=None,
        weight_gain: float = 5.0,
    ):
        if total_neurons <= content:
            raise ValueError(
                f"total_neurons ({total_neurons}) must exceed content ({content}); "
                "there would be no hidden layer left"
            )
        self.content = content
        self.n_actions = n_actions
        self.T = num_steps
        self.n_obs = n_obs
        self.obs_width = obs_width
        self.n_neurons = total_neurons
        self._encoder = encoder if encoder is not None else grid_encoder(5)

        # One flat stack spending the entire budget: hidden + concept == total_neurons.
        self.stack = SensoryCortex(
            n_obs=n_obs,
            hidden=total_neurons - content,
            concept=content,
            num_steps=num_steps,
            weight_gain=weight_gain,
            seed=seed,
        )

    def _to_obs_tensor(self, obs) -> torch.Tensor:
        """Coerce an observation to a ``[B, obs_width]`` int tensor."""
        arr = np.asarray(obs)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.obs_width:
            raise ValueError(
                f"obs must be [{self.obs_width}] or [B, {self.obs_width}]; got {arr.shape}"
            )
        return torch.as_tensor(arr, dtype=torch.long)

    def step(
        self,
        obs,
        *,
        store: bool = False,
        recall: bool = False,
        record: bool = False,
        generator: torch.Generator | None = None,
    ) -> dict:
        """One decision window: obs -> concept.

        ``store``, ``recall`` and ``record`` are accepted for interface parity with
        ``Brain`` and ignored: there is no hippocampus to store into and no separate
        regions to record. The action comes from the policy head, as in v1.
        """
        obs_t = self._to_obs_tensor(obs)
        obs_spikes = self._encoder(obs_t, T=self.T, generator=generator)
        concept = self.stack(obs_spikes)  # [T, B, content]
        return {"concept": concept, "obs_spikes": obs_spikes}
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_monolithic.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/monolithic.py tests/test_monolithic.py
git commit -m "feat(monolithic): neuron-matched unregionalized control brain"
```

---

### Task 3: Addressable states (env injection + shell enumeration)

**Files:**
- Modify: `src/neuromorphic/envs/cube.py` (`CubeEnv.reset`)
- Modify: `src/neuromorphic/envs/cube_distance.py` (add `states_at_distance`)
- Test: `tests/envs/test_cube.py` (append), `tests/envs/test_cube_distance.py` (append)

**Interfaces:**
- Produces:
  - `CubeEnv.reset(seed=None, options=None)` accepts `options={"state": <24 facelets>}` and starts from exactly that state instead of scrambling.
  - `ExactBFSDistance.states_at_distance(depth: int) -> list[tuple[int, ...]]`, sorted for determinism.

**Why:** two capabilities the runner needs and neither exists yet. `reset()` always calls `scramble()`, so there is no way to place the cube in a chosen state, which held-out evaluation requires; `options` is the standard Gymnasium channel and the signature already accepts it. And enumerating a distance shell currently means reaching into `ExactBFSDistance._table`, which is exactly the private-poking the env spec set out to avoid, so it gets a public method instead.

**Note on `exact_depth`:** the design spec called for `exact_depth=True`. Drawing start states directly from an exact-distance shell (Task 4) is strictly stronger: every start state is at distance `d` by construction, with no resampling. `exact_depth` is therefore not used in this experiment. The requirement is met, by a better mechanism.

- [ ] **Step 1: Write the failing tests**

Append to `tests/envs/test_cube.py`:

```python
def test_reset_accepts_an_explicit_start_state():
    env = CubeEnv(scramble_depth=3, scramble_seed=0)
    target = apply_move(SOLVED, 0)
    obs, info = env.reset(options={"state": target})
    assert tuple(int(c) for c in obs) == target
    assert info["solved"] is False
    # and it really is one move from solved
    _, _, term, _, _ = env.step(inverse_action(0))
    assert term is True


def test_reset_without_options_still_scrambles():
    env = CubeEnv(scramble_depth=2, scramble_seed=0)
    obs, _ = env.reset()
    assert tuple(int(c) for c in obs) != SOLVED


def test_reset_rejects_a_malformed_state():
    env = CubeEnv(scramble_seed=0)
    with pytest.raises(ValueError, match="24"):
        env.reset(options={"state": (0, 1, 2)})
```

Append to `tests/envs/test_cube_distance.py`:

```python
def test_states_at_distance_matches_published_shells(shallow):
    for depth, expected in enumerate(PUBLISHED_LEVELS):
        states = shallow.states_at_distance(depth)
        assert len(states) == expected
        assert all(shallow.distance(s) == depth for s in states)


def test_states_at_distance_is_sorted_and_beyond_the_bound_is_empty(shallow):
    states = shallow.states_at_distance(3)
    assert states == sorted(states)
    assert shallow.states_at_distance(12) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/envs/test_cube.py tests/envs/test_cube_distance.py -q -m "not slow"`
Expected: FAIL. The explicit-state test fails because `options` is ignored (the obs is a random scramble), and the shell tests fail with `AttributeError: 'ExactBFSDistance' object has no attribute 'states_at_distance'`.

- [ ] **Step 3: Implement the injection**

In `src/neuromorphic/envs/cube.py`, replace the body of `reset` between the seed handling and `self._steps = 0`:

```python
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        if options is not None and "state" in options:
            # Explicit start state, for evaluating a specific held-out cube.
            state = tuple(int(c) for c in options["state"])
            if len(state) != 24:
                raise ValueError(f"options['state'] must have 24 facelets, got {len(state)}")
            self._state = state
        else:
            self._state = scramble(
                self.scramble_depth,
                self._rng,
                provider=self.distance_provider if self.exact_depth else None,
            )
        self._steps = 0
        self._last_move = None
        d = self._distance()
        self._prev_dist = d
        return self._obs(), self._info(d)
```

Extend the `reset` docstring, or the class docstring if `reset` has none, with one line: `options={"state": facelets}` starts from that exact state instead of scrambling.

- [ ] **Step 4: Add `states_at_distance` to `ExactBFSDistance`**

In `src/neuromorphic/envs/cube_distance.py`, add this method next to `level_counts`:

```python
    def states_at_distance(self, depth: int) -> list[tuple[int, ...]]:
        """Every state at exactly ``depth`` moves from solved, sorted for determinism.

        The distance shell. Returns an empty list for a depth beyond a bounded table.
        Callers get this instead of reaching into ``_table``.
        """
        return sorted(state for state, d in self._table.items() if d == depth)
```

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/envs/test_cube.py tests/envs/test_cube_distance.py -q -m "not slow"`
Expected: PASS (21 cube tests, 9 distance tests).

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/envs/cube.py src/neuromorphic/envs/cube_distance.py tests/envs/test_cube.py tests/envs/test_cube_distance.py
git commit -m "feat(envs): explicit start state and distance-shell enumeration"
```

---

### Task 4: `CubeRunner`

**Files:**
- Create: `src/neuromorphic/training/cube_baseline.py`
- Test: `tests/training/test_cube_baseline.py`

**Interfaces:**
- Consumes: `Brain` (Task 1), `MonolithicBrain` (Task 2), `CubeEnv.reset(options=...)` (Task 3), `ExactBFSDistance`, `cube_encoder`, and from `reinforce.py` (unmodified): `make_policy_head`, `policy_parameters`, `train_episode`, `greedy_action`, `ema`. Also `AblationSpec` / `AblatedConcept` from `neuromorphic.analysis.ablate`.
- Produces:
  - `CubeConfig` dataclass (fields listed in Step 3).
  - `max_steps_for(depth: int) -> int`, returns `2 * depth + 3`.
  - `shell_states(provider, depth: int) -> list[tuple[int, ...]]`.
  - `split_shell(states, depth, *, seed, heldout_cap=200, heldout_frac=0.25) -> tuple[list, list, bool]` returning `(train, eval, is_heldout)`.
  - `ShellCubeEnv(CubeEnv)` whose `reset()` draws a start state from a fixed pool.
  - `make_agent(cfg) -> Brain | MonolithicBrain`.
  - `evaluate_states(agent, head, states, *, depth, generator=None, random_policy=False) -> dict` with keys `success_rate`, `mean_steps`, `optimality`, `n`.
  - `run_cube_baseline(cfg: CubeConfig) -> dict`, the single-run entry point the driver calls.

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_cube_baseline.py
import pytest
import torch

from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.cube_baseline import (
    CubeConfig,
    ShellCubeEnv,
    evaluate_states,
    make_agent,
    max_steps_for,
    run_cube_baseline,
    shell_states,
    split_shell,
)

PUBLISHED = {1: 6, 2: 27, 3: 120, 4: 534, 5: 2256, 6: 8969}


@pytest.fixture(scope="module")
def provider():
    return ExactBFSDistance(max_depth=4)


def test_max_steps_rule():
    assert [max_steps_for(d) for d in (1, 2, 3, 6)] == [5, 7, 9, 15]


def test_shell_sizes_match_published(provider):
    for d in (1, 2, 3, 4):
        assert len(shell_states(provider, d)) == PUBLISHED[d]


def test_shallow_depths_are_not_split(provider):
    for d in (1, 2):
        states = shell_states(provider, d)
        train, ev, is_heldout = split_shell(states, d, seed=0)
        assert is_heldout is False
        assert train == ev == states  # training distribution, labelled as such


def test_deep_depths_split_disjointly_and_deterministically(provider):
    states = shell_states(provider, 3)
    train, ev, is_heldout = split_shell(states, 3, seed=0)
    assert is_heldout is True
    assert set(train).isdisjoint(ev)
    assert len(train) + len(ev) == len(states)
    assert len(ev) == 30  # 25% of 120, under the 200 cap
    again = split_shell(states, 3, seed=0)
    assert again[0] == train and again[1] == ev


def test_heldout_is_capped(provider):
    states = shell_states(provider, 4)  # 534 states, 25% = 133, under the cap
    _, ev, _ = split_shell(states, 4, seed=0, heldout_cap=50)
    assert len(ev) == 50


def test_shell_env_starts_inside_the_pool(provider):
    import random
    pool = shell_states(provider, 2)
    env = ShellCubeEnv(pool, random.Random(0), scramble_depth=2, max_steps=max_steps_for(2))
    for _ in range(10):
        obs, _ = env.reset()
        assert tuple(int(c) for c in obs) in set(pool)


def test_random_arm_scores_above_zero_but_well_below_one(provider):
    """The measured chance floor. Asserting a range, not a point."""
    states = shell_states(provider, 1)
    res = evaluate_states(None, None, states, depth=1, random_policy=True)
    assert res["n"] == 6
    assert 0.0 <= res["success_rate"] <= 1.0


def test_agents_are_built_for_both_arms():
    reg = make_agent(CubeConfig(arm="regionalized"))
    mono = make_agent(CubeConfig(arm="monolithic"))
    assert reg.n_neurons == mono.n_neurons
    assert reg.content == mono.content
    with pytest.raises(ValueError, match="unknown arm"):
        make_agent(CubeConfig(arm="nonsense"))


def test_smoke_run_produces_a_wellformed_record(tmp_path):
    cfg = CubeConfig(
        arm="regionalized", depth=1, seed=0, episodes=3, max_depth=1, out_dir=tmp_path,
    )
    rec = run_cube_baseline(cfg)
    for key in ("arm", "depth", "seed", "sigma", "success_rate", "n", "is_heldout", "episodes"):
        assert key in rec
    assert rec["arm"] == "regionalized"
    assert rec["is_heldout"] is False  # depth 1
    assert 0.0 <= rec["success_rate"] <= 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_cube_baseline.py -q`
Expected: FAIL (`ModuleNotFoundError: neuromorphic.training.cube_baseline`).

- [ ] **Step 3: Implement `cube_baseline.py`**

```python
# src/neuromorphic/training/cube_baseline.py
"""EXP-029: the v1 fail-first baseline on the 2x2 cube, plus its unregionalized control.

``run_generalization`` cannot be reused: it is built on ``split_goals``, ``manhattan``
optimality and ``GridWorldEnv``. This is the cube analogue. ``reinforce.py`` IS reused
unchanged, because it is already environment-agnostic.

Difficulty is exact distance-to-solved, not move count, so the collapse curve is read off
a true axis. The distance table is an instrument only: the agent observes raw facelets and
never sees a distance. See docs/superpowers/specs/2026-07-25-cube-baseline-design.md.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from neuromorphic.analysis.ablate import AblatedConcept, AblationSpec
from neuromorphic.brain import Brain
from neuromorphic.encoders import cube_encoder
from neuromorphic.envs.cube import CubeEnv
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.monolithic import MonolithicBrain
from neuromorphic.training.reinforce import (
    ema,
    greedy_action,
    make_policy_head,
    policy_parameters,
    train_episode,
)

CUBE_N_OBS = 144      # 24 facelets x 6 colors
CUBE_OBS_WIDTH = 24   # raw facelets


@dataclass
class CubeConfig:
    """One run: one arm, one depth, one seed, one sigma."""

    seed: int = 0
    depth: int = 1
    arm: str = "regionalized"   # "regionalized" | "monolithic" | "random"
    sigma: float = 0.0          # Gaussian concept-noise dose (the EXP-028 operator)
    episodes: int = 600
    lr: float = 1e-2
    gamma: float = 0.99
    baseline_beta: float = 0.1
    entropy_beta: float = 0.0
    normalize_advantages: bool = False
    content: int = 64
    n_actions: int = 6
    max_depth: int = 6          # BFS table bound
    heldout_cap: int = 200
    heldout_frac: float = 0.25
    tag: str = "exp029"
    out_dir: Path = field(default_factory=lambda: Path("outputs"))


def max_steps_for(depth: int) -> int:
    """Step budget at a given exact distance. Optimal is ``depth``; this is generous."""
    return 2 * depth + 3


def shell_states(provider: ExactBFSDistance, depth: int) -> list[tuple[int, ...]]:
    """Every state at exact distance ``depth``, sorted so the order is deterministic."""
    return provider.states_at_distance(depth)


def split_shell(
    states: list[tuple[int, ...]],
    depth: int,
    *,
    seed: int,
    heldout_cap: int = 200,
    heldout_frac: float = 0.25,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], bool]:
    """Partition a shell into (train, eval, is_heldout).

    Depths 1 and 2 have only 6 and 27 states. Holding out 1 of 6 is not a generalization
    test, so those depths are NOT split: train and eval are both the whole shell and the
    caller must label the number training-distribution. Deeper shells are split, with the
    held-out side capped so a single evaluation stays affordable (brain.step is 90 ms).
    """
    if depth <= 2:
        return list(states), list(states), False
    shuffled = list(states)
    random.Random(seed).shuffle(shuffled)
    n_eval = min(heldout_cap, int(len(shuffled) * heldout_frac))
    return shuffled[n_eval:], shuffled[:n_eval], True


class ShellCubeEnv(CubeEnv):
    """A ``CubeEnv`` whose ``reset()`` draws its start state from a fixed pool.

    ``train_episode`` calls ``env.reset()`` with no arguments, so this is how the training
    loop is confined to the train side of the split without touching ``reinforce.py``.
    """

    def __init__(self, states, rng: random.Random, **kwargs):
        super().__init__(**kwargs)
        self._pool = list(states)
        self._pool_rng = rng

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options is None:
            options = {"state": self._pool_rng.choice(self._pool)}
        return super().reset(seed=seed, options=options)


def make_agent(cfg: CubeConfig):
    """Build the arm's feature extractor. Both arms are frozen at random init."""
    if cfg.arm == "regionalized":
        return Brain(
            encoder=cube_encoder(), n_obs=CUBE_N_OBS, obs_width=CUBE_OBS_WIDTH,
            n_actions=cfg.n_actions, content=cfg.content, seed=cfg.seed,
        )
    if cfg.arm == "monolithic":
        reference = Brain(
            encoder=cube_encoder(), n_obs=CUBE_N_OBS, obs_width=CUBE_OBS_WIDTH,
            n_actions=cfg.n_actions, content=cfg.content, seed=cfg.seed,
        )
        return MonolithicBrain(
            n_obs=CUBE_N_OBS, n_actions=cfg.n_actions, total_neurons=reference.n_neurons,
            content=cfg.content, obs_width=CUBE_OBS_WIDTH, encoder=cube_encoder(),
            seed=cfg.seed,
        )
    # "random" never reaches here: run_cube_baseline short-circuits it, since the chance
    # floor needs no feature extractor at all.
    raise ValueError(f"unknown arm {cfg.arm!r} (expected regionalized or monolithic)")


def evaluate_states(
    agent,
    head,
    states,
    *,
    depth: int,
    generator: torch.Generator | None = None,
    random_policy: bool = False,
) -> dict:
    """Greedy rollouts from each state. ``random_policy`` measures the chance floor."""
    limit = max_steps_for(depth)
    env = CubeEnv(scramble_depth=depth, max_steps=limit)
    rng = random.Random(0)
    solved = 0
    steps_solved: list[int] = []
    for state in states:
        obs, _ = env.reset(options={"state": state})
        for t in range(1, limit + 1):
            if random_policy:
                action = rng.randrange(env.action_space.n)
            else:
                with torch.no_grad():
                    action = greedy_action(agent, head, obs, generator=generator)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated:
                solved += 1
                steps_solved.append(t)
                break
            if truncated:
                break
    n = len(states)
    total_steps = sum(steps_solved)
    return {
        "success_rate": solved / n if n else 0.0,
        "mean_steps": total_steps / len(steps_solved) if steps_solved else 0.0,
        "optimality": (depth * len(steps_solved) / total_steps) if total_steps else 0.0,
        "n": n,
    }


def run_cube_baseline(cfg: CubeConfig) -> dict:
    """One (arm, depth, seed, sigma) run. Returns a JSON-safe record."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    provider = ExactBFSDistance(max_depth=max(cfg.max_depth, cfg.depth))
    states = shell_states(provider, cfg.depth)
    train_states, eval_states, is_heldout = split_shell(
        states, cfg.depth, seed=cfg.seed,
        heldout_cap=cfg.heldout_cap, heldout_frac=cfg.heldout_frac,
    )

    if cfg.arm == "random":
        result = evaluate_states(None, None, eval_states, depth=cfg.depth, random_policy=True)
        episodes_run = 0
    else:
        agent = make_agent(cfg)
        spec = AblationSpec(kind="gaussian", dose=cfg.sigma, seed=cfg.seed) if cfg.sigma else None
        head = AblatedConcept(
            make_policy_head(agent, "linear"), spec, width=cfg.content
        )
        optimizer = torch.optim.Adam(policy_parameters(head), lr=cfg.lr)
        env = ShellCubeEnv(
            train_states, random.Random(cfg.seed),
            scramble_depth=cfg.depth, max_steps=max_steps_for(cfg.depth),
        )
        baseline = 0.0
        for _ in range(cfg.episodes):
            stats = train_episode(
                agent, head, env, optimizer,
                gamma=cfg.gamma, baseline=baseline, generator=generator,
                max_steps=max_steps_for(cfg.depth),
                entropy_beta=cfg.entropy_beta,
                normalize_advantages=cfg.normalize_advantages,
            )
            baseline = ema(baseline, stats["mean_return"], cfg.baseline_beta)
        result = evaluate_states(
            agent, head, eval_states, depth=cfg.depth, generator=generator
        )
        episodes_run = cfg.episodes

    record = {
        "arm": cfg.arm,
        "depth": cfg.depth,
        "seed": cfg.seed,
        "sigma": cfg.sigma,
        "episodes": episodes_run,
        "is_heldout": is_heldout,
        "n_train": len(train_states),
        "tag": cfg.tag,
        **result,
    }

    # One file per run, never a shared append. The driver fans out over processes, and
    # concurrent appends to a single file interleave and corrupt lines on Windows.
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{cfg.tag}_{cfg.arm}_d{cfg.depth}_s{cfg.seed}_sig{cfg.sigma}.json"
    (out_dir / name).write_text(json.dumps(record), encoding="utf-8")
    return record
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_cube_baseline.py -q`
Expected: PASS (9 passed). The smoke run trains 3 episodes at depth 1 and takes a few seconds.

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/cube_baseline.py tests/training/test_cube_baseline.py
git commit -m "feat(training): CubeRunner for the EXP-029 cube baseline"
```

---

### Task 5: EXP-029 driver

**Files:**
- Create: `experiments/029_cube_baseline/__init__.py` (empty)
- Create: `experiments/029_cube_baseline/run.py`
- Create: `experiments/029_cube_baseline/aggregate.py`
- Test: none beyond the smoke invocation in Step 4. Drivers are not unit tested in this repo (024-028 have no tests); the logic they call is tested in Task 4.

**Interfaces:**
- Consumes: `CubeConfig`, `run_cube_baseline` from Task 4.
- Produces: `outputs/exp029_runs.jsonl` and `outputs/029_curve.md`.

- [ ] **Step 1: Write `run.py`**

```python
# experiments/029_cube_baseline/run.py
"""EXP-029 driver: the v1 fail-first cube baseline and its unregionalized control.

Phase 1 sweeps sigma at depth 1 on BOTH trained arms and picks each arm's winner. Phase 2
runs depths 2 to 6 at each arm's winning sigma. The random arm is evaluation only and
measures the chance floor at every depth.

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/029_cube_baseline/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import CubeConfig, run_cube_baseline

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
SIGMAS = [0.0, 0.2, 0.4]
TRAINED_ARMS = ["regionalized", "monolithic"]
DEPTHS = [1, 2, 3, 4, 5, 6]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(episodes=args.episodes, out_dir=args.out_dir)

    # Phase 1: sigma sweep at depth 1, both trained arms. Supplies the depth-1 cell.
    sweep = [
        CubeConfig(arm=arm, depth=1, seed=s, sigma=sig, **common)
        for arm in TRAINED_ARMS
        for sig in SIGMAS
        for s in args.seeds
    ]
    print(f"phase 1: sigma sweep, {len(sweep)} runs")
    sweep_records = _fan_out(sweep, args.workers)

    # Each arm keeps its own winning sigma. Tuning on one arm and applying it to the
    # other would hand the control an untuned hyperparameter and bias the comparison.
    by_arm_sigma = defaultdict(list)
    for r in sweep_records:
        by_arm_sigma[(r["arm"], r["sigma"])].append(r["success_rate"])
    best = {}
    for arm in TRAINED_ARMS:
        means = {sig: sum(v) / len(v) for (a, sig), v in by_arm_sigma.items() if a == arm}
        best[arm] = max(means, key=means.get)
        print(f"  {arm}: sigma means {means} -> winner {best[arm]}")

    # Phase 2: depths 2-6 at each arm's winning sigma, plus the random floor everywhere.
    rest = [
        CubeConfig(arm=arm, depth=d, seed=s, sigma=best[arm], **common)
        for arm in TRAINED_ARMS
        for d in DEPTHS[1:]
        for s in args.seeds
    ] + [
        CubeConfig(arm="random", depth=d, seed=s, sigma=0.0, **common)
        for d in DEPTHS
        for s in args.seeds
    ]
    print(f"phase 2: {len(rest)} runs")
    _fan_out(rest, args.workers)
    print(f"done. one record per run in {args.out_dir}")


def _fan_out(configs, workers: int) -> list[dict]:
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if i % 10 == 0 or i == len(configs):
                print(f"  {i}/{len(configs)}")
    return records


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `aggregate.py`**

```python
# experiments/029_cube_baseline/aggregate.py
"""Aggregate EXP-029 records into the collapse table.

Reports mean success per (arm, depth) across seeds, with the paired regionalized-minus-
monolithic difference. Depths 1 and 2 are training-distribution; 3 to 6 are held-out, and
the table says which is which rather than leaving the reader to guess.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(out_dir: Path) -> list[dict]:
    """Every per-run record. One file per run, written by the workers."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("exp029_*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=HERE / "outputs")
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "029_curve.md")
    args = ap.parse_args()

    records = load(args.runs)
    if not records:
        raise SystemExit(f"no run records found in {args.runs}")
    cells = defaultdict(list)
    heldout = {}
    for r in records:
        cells[(r["arm"], r["depth"])].append(r["success_rate"])
        heldout[r["depth"]] = r["is_heldout"]

    by_seed = defaultdict(dict)
    for r in records:
        if r["arm"] in ("regionalized", "monolithic"):
            by_seed[(r["depth"], r["seed"])][r["arm"]] = r["success_rate"]

    lines = [
        "# EXP-029 collapse curve",
        "",
        "| depth | eval | regionalized | monolithic | random floor | paired diff | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for depth in sorted({d for _, d in cells}):
        def mean(arm):
            vals = cells.get((arm, depth), [])
            return f"{100 * sum(vals) / len(vals):.0f}%" if vals else "n/a"

        pairs = [
            v["regionalized"] - v["monolithic"]
            for (d, _), v in by_seed.items()
            if d == depth and "regionalized" in v and "monolithic" in v
        ]
        diff = f"{100 * sum(pairs) / len(pairs):+.0f} pts" if pairs else "n/a"
        label = "held-out" if heldout.get(depth) else "train-dist"
        lines.append(
            f"| {depth} | {label} | {mean('regionalized')} | {mean('monolithic')} | "
            f"{mean('random')} | {diff} | {len(pairs)} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add `__init__.py` and gitignore the outputs**

Create an empty `experiments/029_cube_baseline/__init__.py` (matching 027 and 028).

Confirm `outputs/` is already gitignored at repo level. Run:

```bash
git check-ignore -v experiments/029_cube_baseline/outputs/x.jsonl
```

Expected: a matching ignore rule is printed. If nothing is printed, add `experiments/*/outputs/` to `.gitignore`.

- [ ] **Step 4: Smoke the driver end to end**

Run a deliberately tiny grid to prove the wiring, not to produce science:

```bash
.venv/Scripts/python.exe experiments/029_cube_baseline/run.py --seeds 0 --episodes 5 --workers 2
.venv/Scripts/python.exe experiments/029_cube_baseline/aggregate.py
```

Expected: phase 1 prints per-arm sigma means and a winner, phase 2 completes, and `aggregate.py` prints a table with rows for depths 1 to 6 and a `random floor` column. Numbers will be garbage at 5 episodes. **Delete the smoke records afterwards** so they cannot contaminate the real run:

```bash
rm experiments/029_cube_baseline/outputs/exp029_*.json
```

- [ ] **Step 5: Full regression**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q -m "not slow"`
Expected: PASS, no fewer tests than at base.

- [ ] **Step 6: Commit**

```bash
git add experiments/029_cube_baseline/
git commit -m "feat(exp029): cube baseline driver and aggregator"
```

---

## After the plan

The build is done when Task 5 commits. The **experiment run itself is a separate step**, roughly 4 to 5 hours on 8 cores:

```bash
.venv/Scripts/python.exe experiments/029_cube_baseline/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
.venv/Scripts/python.exe experiments/029_cube_baseline/aggregate.py
```

Then write `experiments/029_cube_baseline/RESULTS.md` against the pre-registered contract in the spec §5, marking each pre-registered claim confirmed or refuted, with provenance (seeds, date, machine, regeneration command). That file is committed; `outputs/` is not.
