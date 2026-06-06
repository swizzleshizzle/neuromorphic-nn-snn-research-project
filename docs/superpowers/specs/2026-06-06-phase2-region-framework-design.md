# Phase 2 — Multi-Region Framework (Step 2.1/2.2) — Design

**Status:** Approved · ready for implementation plan
**Date:** 2026-06-06 · Week 9 hands-on
**Phase:** 2 — Multi-region brain · **Step 2.1/2.2** — region base class, inter-region protocol, first concrete region
**Parent spec:** `docs/architecture-spec-v1.md` (five-region architecture v1)
**Tracking plan:** `docs/superpowers/plans/2026-06-05-phase2-architecture-spec-plan.md`

---

## 0. Scope

First implementation pass of the five-region architecture. Three deliverables:

1. `BrainRegion` abstract base class — the standard interface every region implements.
2. `Projection` — sparse random, delayed inter-region connection.
3. `SensoryCortex` — first concrete region; feedforward Leaky stack with a grid-world encoder.

This feeds spec corrections back (architecture-spec §4 build-order step 1, "Sensory alone").
Out of scope today: the other four regions, the router gates, the neuromod bus, any training.

### Resolved decisions (this session)

- **Coding scheme** (spec open-question §5.4): **rate / Poisson** for the sensory input.
- **Package path:** modules live under the installed `neuromorphic` package —
  `src/neuromorphic/regions/`, `src/neuromorphic/connections/` (not literal top-level `src/regions/`).
- **Grid-world observation:** agent + goal, **one-hot per cell**. For an `N×N` grid,
  `N_obs = 2 * N*N` (agent one-hot ⊕ goal one-hot). Bring-up grid = **5×5** → `N_obs = 50`.
- **Verification:** both pytest contract gates **and** a viz demo in a numbered experiment folder.

### Conventions inherited (locked, repo-wide)

- Tensor contract `[T, B, N]` for spikes/currents; weights `[N_post, N_pre]`.
- snnTorch primitives; per-`T` forward loop with `init_*()`, stack to `[T, B, N]`.
- `from __future__ import annotations`; numpy-style docstrings (Args/Returns); strict
  `ValueError` shape checks; plain-pytest tests.
- Leaky params from the week-7 locked config: `beta=0.9, threshold=1.0,
  reset_mechanism="subtract"`. Inference window `T = 32`.

---

## 1. `BrainRegion` ABC — `src/neuromorphic/regions/base_region.py`

Subclasses `nn.Module` **and** `abc.ABC`: parameters/devices work, and the interface is
enforced (cannot instantiate without overriding the abstract methods).

**Uniform contract:** input spikes `[T, B, N_in]` → output spikes `[T, B, N_out]`.

```python
class BrainRegion(nn.Module, ABC):
    def __init__(self, name: str, n_neurons: int)

    # --- contract (abstract) ---
    @abstractmethod
    def forward(self, input_spikes: Tensor) -> Tensor       # [T,B,N_in] -> [T,B,N_out]
    @abstractmethod
    def reset(self, batch_size: int | None = None,
              device: torch.device | None = None) -> None    # (re)init neuron states
    @abstractmethod
    def get_state(self) -> dict[str, Tensor]                 # current membrane/spk state

    # --- built-in logging hooks for the viz toolkit (opt-in) ---
    def enable_recording(self, flag: bool = True) -> None
    def clear_recording(self) -> None
    def _record(self, key: str, value: Tensor) -> None       # subclass calls in the T-loop
    def get_recording(self, key: str | None = None)          # -> [T,B,N] (or dict)
```

### Logging-hook semantics

- Recording is **off by default** — `_record` is a no-op until `enable_recording(True)`,
  so the hot loop has zero overhead in production.
- A subclass calls `self._record("concept", spk)` once per time step inside its forward
  loop, passing a `[B, N]` tensor. `get_recording("concept")` stacks the per-step list along
  dim 0 → `[T, B, N]`, the canonical viz contract — directly consumable by `spike_raster`,
  `population_rate`, `psth`. Recorded tensors are detached.
- Weight matrices need no special hook: Linear `weight` params are already `[N_post, N_pre]`
  and feed `weight_heatmap` / `weight_histogram` via standard `named_parameters()`.

### State semantics

- Neuron states (membranes, recurrent spikes) live on `self`, created/zeroed by `reset()`,
  updated in `forward`, exposed read-only via `get_state()`.
- This lets future recurrent regions (Hippocampus) **carry state across `T`-windows**.
  Feedforward regions simply re-init each window. `forward` lazily calls `reset()` if state
  is uninitialized; callers should `reset()` between independent sequences.

---

## 2. `Projection` — `src/neuromorphic/connections/projection.py`

Sparse random, delayed link between two regions. An `nn.Module` holding a weight matrix
`[N_tgt, N_src]`, a fixed boolean sparsity mask of the same shape, and an integer delay Δ.

**Outputs current `[T, B, N_tgt]`** — projections route weighted current, never spikes. The
consuming region's neurons convert current → spikes.

```python
class Projection(nn.Module):
    def __init__(self, n_source: int, n_target: int,
                 sparsity: float = 0.1,        # fraction of connections KEPT (density p)
                 delay: int = 1,               # Δ in T-steps (spec §5.5 placeholder)
                 weight_scale: float = 1.0,    # std of N(0, scale) for nonzero weights
                 seed: int | None = None)

    def forward(self, src_spikes: Tensor) -> Tensor   # [T,B,N_src] -> [T,B,N_tgt]
```

- **Connectivity:** a seeded boolean mask keeps ~`sparsity` of the `N_tgt × N_src` entries;
  weights are drawn (seeded) and multiplied by the mask, so masked-out synapses stay 0.
- **Delay:** output shifted forward by Δ time steps, front zero-padded (so a spike at `t`
  affects the target at `t+Δ`). Δ=0 is a pass-through. Δ must be `0 ≤ Δ < T`.
- **Reproducibility:** mask and weights derive from `seed` via a local generator.
- **Shape checks:** raises `ValueError` if `src_spikes` is not 3D or `N_src` mismatches.

---

## 3. `SensoryCortex` — `src/neuromorphic/regions/sensory_cortex.py`

First concrete region (spec §2.1). Two-stage feedforward Leaky compression
`N_obs → 128 → 64` (concept code). Plus a **standalone grid-world encoder** kept separate
from the region so the ABC contract stays "spikes in → spikes out".

```python
def encode_gridworld(obs, grid_n: int, T: int = 32,
                     max_rate: float = 0.5,   # per-step spike prob for an active one-hot cell
                     generator: torch.Generator | None = None) -> Tensor:
    """Agent+goal one-hots -> Poisson spikes [T, B, N_obs], N_obs = 2*grid_n*grid_n."""

class SensoryCortex(BrainRegion):
    def __init__(self, n_obs: int, hidden: int = 128, concept: int = 64,
                 beta: float = 0.9, threshold: float = 1.0,
                 reset_mechanism: str = "subtract", num_steps: int = 32)

    def forward(self, input_spikes: Tensor) -> Tensor   # [T,B,N_obs] -> [T,B,concept]
        # records "hidden" ([T,B,128]) and "concept" ([T,B,64]) when recording is on
```

- **Encoder:** agent `(x,y)` and goal `(x,y)` → two `grid_n*grid_n` one-hot vectors,
  concatenated → length-`N_obs` rate vector scaled by `max_rate`; sampled as Poisson spikes
  over `T` steps with a seeded generator. Input batched: `obs` shape `[B, 4]` (agent_x,
  agent_y, goal_x, goal_y) → `[T, B, N_obs]`.
- **Region:** `fc1: Linear(N_obs, 128) → Leaky → fc2: Linear(128, 64) → Leaky`, per-`T`
  loop, returns `[T, B, 64]`.

---

## 4. Verification

### pytest — `tests/regions/`, `tests/connections/`

- **BrainRegion:** instantiating the ABC raises `TypeError`; a minimal subclass missing an
  abstract method also fails; recording off → `get_recording` empty; recording on → stacks
  to `[T,B,N]`.
- **Projection:** output shape `[T,B,N_tgt]`; mask density ≈ `sparsity` (within tolerance);
  Δ-shift correctness (a single input spike at `t` appears at `t+Δ`); same seed →
  identical weights/mask/output.
- **SensoryCortex:** `encode_gridworld` output shape `[T,B,N_obs]` and binary; forward output
  `[T,B,64]`; output is **not dead** (fires) given an active input; **determinism** under a
  fixed generator; **selectivity** — two distinct agent positions produce measurably
  different concept codes (spike-count distance > 0).

### viz demo — `experiments/013_week9_sensory_bringup/`

Numbered experiment folder (continues the EXP-NNN pattern). Encodes a 5×5 grid observation,
runs `SensoryCortex`, and saves via the viz toolkit:

- `outputs/concept_raster.png` — `spike_raster` of the concept code.
- `outputs/population_rate.png` — `population_rate` of the concept layer.

This is the "produces meaningful output spikes" gate. A short `results.md` records the
shapes, firing rate, and selectivity observation (week-7 verification-gate discipline).

---

## 5. File manifest

```
src/neuromorphic/regions/__init__.py
src/neuromorphic/regions/base_region.py        # BrainRegion ABC
src/neuromorphic/regions/sensory_cortex.py     # SensoryCortex + encode_gridworld
src/neuromorphic/connections/__init__.py
src/neuromorphic/connections/projection.py     # Projection
tests/regions/test_base_region.py
tests/regions/test_sensory_cortex.py
tests/connections/test_projection.py
experiments/013_week9_sensory_bringup/run.py   # viz demo + results.md
```
