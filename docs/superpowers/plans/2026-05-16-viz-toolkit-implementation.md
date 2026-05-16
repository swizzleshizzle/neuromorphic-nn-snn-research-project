# Stage 1 Viz Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 7-function Stage 1 visualization toolkit at `src/neuromorphic/viz/`, with per-function pytest smoke tests and a live eyeball script against the week-5 SNN checkpoint.

**Architecture:** A `viz` subpackage under `neuromorphic` with one module per data type (`spikes`, `membrane`, `weights`, `training`) plus a `core` module for shared axis/sample helpers. Every public function follows a strict `[T, B, N]` (or `[N_post, N_pre]`) input contract and returns `(Figure, Axes)`. Two test layers: pytest smoke tests (`tests/viz/test_smoke.py`) for contract verification, and a one-off `experiments/008_week5_snn_mnist_baseline/viz_smoke.py` for visual correctness against real model output.

**Tech Stack:** Python 3.11, PyTorch, snnTorch 0.9.4, matplotlib 3.10, pytest (to be installed). No seaborn, sklearn, or plotly.

**Parent spec:** `docs/design/2026-05-16-viz-toolkit-implementation.md`
**Stage 1 design doc:** `docs/design/stage-1-visualization-toolkit.md`

**Discipline reminders (from parent spec §9):**
- Implement one function → run its pytest smoke → eyeball one render against week-5 data → move to next. **Not** all-7-then-test-at-end.
- Before each function: state expected output shape, what the plot should look like, what would indicate it's broken. Run. Compare prediction to reality.
- Verified contract violation: `splt.traces` does NOT accept an `ax`; it uses `GridSpec` internally. `membrane_trace` is therefore implemented from scratch, not wrapped. `splt.raster` does accept an `ax` and is wrap-able.

---

## File Structure

**Created by this plan:**

| File | Responsibility |
|---|---|
| `src/neuromorphic/viz/__init__.py` | Re-exports the 7 public functions as a flat API |
| `src/neuromorphic/viz/core.py` | `_ensure_ax`, `_select_sample`, shape validation helpers |
| `src/neuromorphic/viz/spikes.py` | `spike_raster`, `population_rate`, `psth` |
| `src/neuromorphic/viz/membrane.py` | `membrane_trace` |
| `src/neuromorphic/viz/weights.py` | `weight_histogram`, `weight_heatmap`, `_prep_weight` helper |
| `src/neuromorphic/viz/training.py` | `training_curve` |
| `tests/viz/__init__.py` | Empty — marks `tests/viz/` as a package |
| `tests/viz/conftest.py` | Forces matplotlib `Agg` backend, defines shared fixtures |
| `tests/viz/test_smoke.py` | One smoke test per public function |
| `experiments/008_week5_snn_mnist_baseline/viz_smoke.py` | Loads week-5 checkpoint, renders all 7 plots with the new toolkit |

**Modified by this plan:**
- `requirements.txt` — add `pytest>=8.0` (currently missing).

**NOT created today** (deferred per parent spec §6, §10):
`tuning.py`, `memory.py`, `composite.py`.

---

## Task 0: Project setup

**Files:**
- Create: `src/neuromorphic/viz/__init__.py` (empty for now)
- Create: `tests/viz/__init__.py` (empty)
- Create: `tests/viz/conftest.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create the viz package directory with a placeholder `__init__.py`**

```bash
mkdir -p src/neuromorphic/viz
```

Write `src/neuromorphic/viz/__init__.py`:

```python
"""Visualization toolkit for spiking neural networks.

All functions return ``(matplotlib.figure.Figure, matplotlib.axes.Axes)`` and
accept tensors in the canonical ``[T, B, N]`` shape. See
``docs/design/stage-1-visualization-toolkit.md``.
"""
```

(Public-API re-exports are added at the end of the build in Task 9.)

- [ ] **Step 2: Create the tests/viz package**

```bash
mkdir -p tests/viz
```

Write `tests/viz/__init__.py` as an empty file.

- [ ] **Step 3: Create the pytest conftest**

Write `tests/viz/conftest.py`:

```python
"""Shared fixtures and matplotlib backend setup for viz smoke tests."""

import matplotlib

matplotlib.use("Agg")  # Headless backend — no GUI windows during pytest.

import pytest
import torch


@pytest.fixture(scope="session")
def small_shape():
    """Canonical small synthetic shape used by every smoke test."""
    return {"T": 20, "B": 4, "N": 50}


@pytest.fixture(scope="session")
def spk(small_shape):
    """Synthetic [T, B, N] spike tensor at ~20% firing rate."""
    torch.manual_seed(0)
    T, B, N = small_shape["T"], small_shape["B"], small_shape["N"]
    return (torch.rand(T, B, N) < 0.2).float()


@pytest.fixture(scope="session")
def mem(small_shape):
    """Synthetic [T, B, N] membrane-potential tensor."""
    torch.manual_seed(1)
    T, B, N = small_shape["T"], small_shape["B"], small_shape["N"]
    return torch.randn(T, B, N) * 0.5


@pytest.fixture(scope="session")
def W():
    """Synthetic [N_post=100, N_pre=50] weight matrix."""
    torch.manual_seed(2)
    return torch.randn(100, 50) * 0.1


@pytest.fixture(scope="session")
def history():
    """Synthetic training history dict with loss and accuracy series."""
    return {
        "train_loss": [3.0 - 0.01 * i for i in range(100)],
        "test_loss":  [3.0 - 0.02 * i for i in range(10)],
        "test_acc":   [0.1 + 0.08 * i for i in range(10)],
    }
```

- [ ] **Step 4: Add pytest to requirements**

Modify `requirements.txt` — add the line `pytest>=8.0` under the "Notebook + dev convenience" block:

```diff
 # Notebook + dev convenience
 jupyter>=1.0
 ipykernel>=6.29
+pytest>=8.0
```

- [ ] **Step 5: Install pytest into the active venv**

```powershell
.venv\Scripts\python.exe -m pip install "pytest>=8.0"
```

Expected output: line ending in `Successfully installed pytest-8.x.x ...`.

- [ ] **Step 6: Create `outputs/` directory for per-task eyeball renders**

The per-function eyeball PNGs in Tasks 2–7 land at `outputs/viz_smoke_*.png`. The directory doesn't exist yet, and following the repo's existing pattern (`data/`, `checkpoints/`, `runs/` are all `dir-tracked-with-.gitkeep-contents-ignored`), do the same here.

```bash
mkdir -p outputs
```

Write `outputs/.gitkeep` as an empty file.

Modify `.gitignore` — append under the "Project-specific" block (after the `runs/*` / `!runs/.gitkeep` pair):

```diff
 checkpoints/*
 !checkpoints/.gitkeep
 runs/*
 !runs/.gitkeep
+outputs/*
+!outputs/.gitkeep
 wandb/
```

- [ ] **Step 7: Smoke check that the package imports cleanly**

```powershell
.venv\Scripts\python.exe -c "import neuromorphic.viz; print(neuromorphic.viz.__doc__)"
```

Expected: the docstring prints, no traceback.

- [ ] **Step 8: Commit**

```bash
git add src/neuromorphic/viz/__init__.py tests/viz/__init__.py tests/viz/conftest.py requirements.txt outputs/.gitkeep .gitignore
git commit -m "viz: scaffold viz package + test conftest, add pytest dep, outputs/ dir"
```

---

## Task 1: `core.py` — shared helpers

**Files:**
- Create: `src/neuromorphic/viz/core.py`
- Modify: `tests/viz/test_smoke.py` (create with first tests)

**Predict before executing:**
- `_ensure_ax(None)` → returns a `(Figure, Axes)` tuple with a fresh fig.
- `_ensure_ax(existing_ax)` → returns `(existing_ax.figure, existing_ax)`.
- `_select_sample(spk_3d)` with default `sample_idx=0` on a `[20, 4, 50]` tensor → returns `[20, 50]`.
- `_select_sample(spk_2d)` on a `[20, 50]` tensor → returns `[20, 50]` unchanged.
- `_select_sample(spk_1d)` → raises `ValueError` (we only support 2D/3D).

- [ ] **Step 1: Write the failing tests**

Write `tests/viz/test_smoke.py`:

```python
"""Smoke tests for neuromorphic.viz. Each public function gets one test
asserting it returns a (Figure, Axes) tuple without raising."""

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ---------- core helpers ----------

def test_ensure_ax_creates_new_when_none():
    from neuromorphic.viz.core import _ensure_ax
    fig, ax = _ensure_ax(None)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_ensure_ax_reuses_existing():
    from neuromorphic.viz.core import _ensure_ax
    fig, ax = plt.subplots()
    fig2, ax2 = _ensure_ax(ax)
    assert ax2 is ax
    assert fig2 is fig
    plt.close(fig)


def test_select_sample_indexes_3d():
    from neuromorphic.viz.core import _select_sample
    t = torch.zeros(20, 4, 50)
    t[:, 2, :] = 1.0
    out = _select_sample(t, sample_idx=2)
    assert out.shape == (20, 50)
    assert (out == 1.0).all()


def test_select_sample_passes_2d_through():
    from neuromorphic.viz.core import _select_sample
    t = torch.zeros(20, 50)
    out = _select_sample(t, sample_idx=0)
    assert out.shape == (20, 50)
    assert out is t  # no copy


def test_select_sample_rejects_1d():
    from neuromorphic.viz.core import _select_sample
    with pytest.raises(ValueError, match="2D or 3D"):
        _select_sample(torch.zeros(20), sample_idx=0)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -v
```

Expected: 5 errors / failures, all variants of `ModuleNotFoundError: No module named 'neuromorphic.viz.core'`.

- [ ] **Step 3: Implement `core.py`**

Write `src/neuromorphic/viz/core.py`:

```python
"""Shared helpers used by every viz function.

These exist so that every public function can:
- accept ``ax=None`` and produce a fresh fig/ax on demand
- accept either ``[T, B, N]`` or ``[T, N]`` for single-sample plots
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _ensure_ax(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``; create a new pair if ``ax`` is None."""
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.figure, ax


def _select_sample(tensor: torch.Tensor, sample_idx: int = 0) -> torch.Tensor:
    """Reduce a ``[T, B, N]`` tensor to ``[T, N]`` by picking one sample.

    A ``[T, N]`` tensor is returned unchanged. Other ranks raise ``ValueError``.
    """
    if tensor.ndim == 3:
        return tensor[:, sample_idx, :]
    if tensor.ndim == 2:
        return tensor
    raise ValueError(
        f"_select_sample expects a 2D or 3D tensor, got shape {tuple(tensor.shape)}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/viz/core.py tests/viz/test_smoke.py
git commit -m "viz: add core helpers (_ensure_ax, _select_sample) + tests"
```

---

## Task 2: `spike_raster`

**Files:**
- Create: `src/neuromorphic/viz/spikes.py`
- Modify: `tests/viz/test_smoke.py` (append test)

**Predict before executing:**
- Input `spk` is `[20, 4, 50]`, ~20% firing rate → after `_select_sample(spk, 0)` it's `[20, 50]`.
- `splt.raster` produces a scatter where each (time, neuron) coordinate of a 1.0 is a dot.
- We expect roughly `20 * 50 * 0.2 = 200` dots scattered across `t ∈ [0, 19]`, `n ∈ [0, 49]`.
- Broken signal: no dots (sample selection wrong, or `splt.raster` got a 3D tensor and silently did nothing).

- [ ] **Step 1: Write the failing test**

Append to `tests/viz/test_smoke.py`:

```python
# ---------- spike_raster ----------

def test_spike_raster_returns_fig_ax(spk):
    from neuromorphic.viz import spike_raster
    fig, ax = spike_raster(spk)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_spike_raster_respects_provided_ax(spk):
    from neuromorphic.viz import spike_raster
    fig, ax = plt.subplots()
    fig2, ax2 = spike_raster(spk, ax=ax)
    assert ax2 is ax
    assert fig2 is fig
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py::test_spike_raster_returns_fig_ax -v
```

Expected: `ImportError` — `spike_raster` not exposed by `neuromorphic.viz`.

- [ ] **Step 3: Implement `spikes.py` (spike_raster only for now)**

Write `src/neuromorphic/viz/spikes.py`:

```python
"""Spike-train plotting: rasters, PSTH, population rate."""

from __future__ import annotations

import snntorch.spikeplot as splt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuromorphic.viz.core import _ensure_ax, _select_sample


def spike_raster(
    spk: torch.Tensor,
    ax: Axes | None = None,
    sample_idx: int = 0,
    **scatter_kwargs,
) -> tuple[Figure, Axes]:
    """Spike raster: one dot per (time, neuron) spike.

    Wraps ``snntorch.spikeplot.raster``, which calls ``ax.scatter`` internally.

    Args:
        spk: ``[T, B, N]`` or ``[T, N]``. If 3D, sample ``sample_idx`` is shown.
        ax: matplotlib Axes. If None, a fresh fig/ax is created.
        sample_idx: which batch element to plot (ignored if ``spk`` is 2D).
        scatter_kwargs: forwarded to ``ax.scatter`` via ``splt.raster``
            (e.g. ``s=1.5``, ``c="black"``).

    Returns:
        ``(fig, ax)``. The caller decides whether to show, save, or embed.
    """
    fig, ax = _ensure_ax(ax)
    spk_2d = _select_sample(spk, sample_idx)
    # splt.raster signature: raster(data, ax, **kwargs); data shape [T, N].
    scatter_kwargs.setdefault("s", 2)
    scatter_kwargs.setdefault("c", "black")
    splt.raster(spk_2d, ax, **scatter_kwargs)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Neuron index")
    return fig, ax
```

- [ ] **Step 4: Expose `spike_raster` from the package init**

Modify `src/neuromorphic/viz/__init__.py` — append:

```python
from neuromorphic.viz.spikes import spike_raster

__all__ = ["spike_raster"]
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py::test_spike_raster_returns_fig_ax tests/viz/test_smoke.py::test_spike_raster_respects_provided_ax -v
```

Expected: `2 passed`.

- [ ] **Step 6: Eyeball — render against the synthetic fixture**

```powershell
.venv\Scripts\python.exe -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from neuromorphic.viz import spike_raster

torch.manual_seed(0)
spk = (torch.rand(20, 4, 50) < 0.2).float()
fig, ax = spike_raster(spk)
ax.set_title('spike_raster smoke (synthetic)')
fig.savefig('outputs/viz_smoke_spike_raster.png', dpi=100, bbox_inches='tight')
print('saved outputs/viz_smoke_spike_raster.png')
"
```

Open `outputs/viz_smoke_spike_raster.png` in an image viewer. Confirm: scattered dots, x-axis 0–19, y-axis 0–49, roughly the firing-rate density predicted above.

- [ ] **Step 7: Commit**

```bash
git add src/neuromorphic/viz/spikes.py src/neuromorphic/viz/__init__.py tests/viz/test_smoke.py
git commit -m "viz: add spike_raster wrapping splt.raster"
```

---

## Task 3: `membrane_trace`

**Files:**
- Create: `src/neuromorphic/viz/membrane.py`
- Modify: `src/neuromorphic/viz/__init__.py`
- Modify: `tests/viz/test_smoke.py`

**Why we are NOT wrapping `splt.traces`:** confirmed at plan-write time that `splt.traces(data, spk, dim=(3,3), ...)` builds its own multi-panel `GridSpec` and does not accept an `ax`. Wrapping it would either ignore our `ax=` argument (silent contract violation) or require us to monkey-patch matplotlib's current-figure state. Implementing the trace plot directly is shorter, honest, and matches the week-5 reference (`experiments/008_week5_snn_mnist_baseline/plots.py::plot_membrane_traces`).

**Predict before executing:**
- Input `mem` is `[20, 4, 50]`. After sample selection → `[20, 50]`. 50 line traces on one axis.
- With `show_threshold=True, threshold=1.0`, a horizontal dashed line at y=1.0 appears.
- Broken signal: no horizontal threshold line; or all 50 lines are identical (sample selection bug).

- [ ] **Step 1: Write the failing tests**

Append to `tests/viz/test_smoke.py`:

```python
# ---------- membrane_trace ----------

def test_membrane_trace_returns_fig_ax(mem):
    from neuromorphic.viz import membrane_trace
    fig, ax = membrane_trace(mem)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_membrane_trace_draws_threshold_line(mem):
    from neuromorphic.viz import membrane_trace
    fig, ax = membrane_trace(mem, threshold=1.0, show_threshold=True)
    # axhline adds a Line2D; check at least one line is at y=1.0.
    threshold_lines = [
        ln for ln in ax.get_lines()
        if ln.get_ydata().size == 2 and (ln.get_ydata() == 1.0).all()
    ]
    assert len(threshold_lines) >= 1
    plt.close(fig)


def test_membrane_trace_omits_threshold_when_disabled(mem):
    from neuromorphic.viz import membrane_trace
    fig, ax = membrane_trace(mem, threshold=1.0, show_threshold=False)
    threshold_lines = [
        ln for ln in ax.get_lines()
        if ln.get_ydata().size == 2 and (ln.get_ydata() == 1.0).all()
    ]
    assert len(threshold_lines) == 0
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -k membrane_trace -v
```

Expected: `ImportError` — `membrane_trace` not exposed.

- [ ] **Step 3: Implement `membrane.py`**

Write `src/neuromorphic/viz/membrane.py`:

```python
"""Membrane-potential trace plotting.

Implemented from scratch rather than wrapping ``snntorch.spikeplot.traces``
because the latter does not accept an ``ax`` argument and uses ``GridSpec``
internally — incompatible with the toolkit's ``(fig, ax)`` contract.
"""

from __future__ import annotations

import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuromorphic.viz.core import _ensure_ax, _select_sample


def membrane_trace(
    mem: torch.Tensor,
    spk: torch.Tensor | None = None,
    ax: Axes | None = None,
    sample_idx: int = 0,
    threshold: float = 1.0,
    show_threshold: bool = True,
    spk_height: float = 1.5,
) -> tuple[Figure, Axes]:
    """Line plot of membrane potential ``u_i(t)`` for each neuron in a layer.

    Args:
        mem: ``[T, B, N]`` or ``[T, N]`` membrane potentials.
        spk: optional ``[T, B, N]`` or ``[T, N]`` spike train. When provided,
            ``spk_height`` is added to ``mem`` wherever the neuron spiked, so
            spikes "ride on" the trace (Friedemann Zenke style).
        ax: matplotlib Axes. If None, a fresh fig/ax is created.
        sample_idx: which batch element to plot.
        threshold: y-value at which to draw the (optional) threshold line.
        show_threshold: if True, draw a dashed horizontal line at ``threshold``.
        spk_height: vertical bump applied where ``spk == 1`` (if ``spk`` given).

    Returns:
        ``(fig, ax)``.
    """
    fig, ax = _ensure_ax(ax)
    mem_2d = _select_sample(mem, sample_idx).detach().cpu()

    if spk is not None:
        spk_2d = _select_sample(spk, sample_idx).detach().cpu()
        data = mem_2d + spk_height * spk_2d
    else:
        data = mem_2d

    T, N = data.shape
    time = torch.arange(T)
    for n in range(N):
        ax.plot(time, data[:, n], linewidth=0.8, alpha=0.7)

    if show_threshold:
        ax.axhline(y=threshold, color="black", linestyle="--",
                   linewidth=1.0, alpha=0.6, label=f"threshold = {threshold}")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Time step")
    ax.set_ylabel(r"Membrane potential $U[t]$")
    ax.grid(alpha=0.3)
    return fig, ax
```

- [ ] **Step 4: Expose `membrane_trace`**

Modify `src/neuromorphic/viz/__init__.py`:

```python
"""Visualization toolkit for spiking neural networks.

All functions return ``(matplotlib.figure.Figure, matplotlib.axes.Axes)`` and
accept tensors in the canonical ``[T, B, N]`` shape. See
``docs/design/stage-1-visualization-toolkit.md``.
"""

from neuromorphic.viz.membrane import membrane_trace
from neuromorphic.viz.spikes import spike_raster

__all__ = ["spike_raster", "membrane_trace"]
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -k membrane_trace -v
```

Expected: `3 passed`.

- [ ] **Step 6: Eyeball — render against the synthetic fixture**

```powershell
.venv\Scripts\python.exe -c "
import matplotlib
matplotlib.use('Agg')
import torch
from neuromorphic.viz import membrane_trace

torch.manual_seed(1)
mem = torch.randn(20, 4, 50) * 0.5
fig, ax = membrane_trace(mem, threshold=1.0)
ax.set_title('membrane_trace smoke (synthetic)')
fig.savefig('outputs/viz_smoke_membrane_trace.png', dpi=100, bbox_inches='tight')
print('saved outputs/viz_smoke_membrane_trace.png')
"
```

Open the PNG. Confirm: 50 wiggly lines, dashed threshold line at y=1.0, x-axis 0–19.

- [ ] **Step 7: Commit**

```bash
git add src/neuromorphic/viz/membrane.py src/neuromorphic/viz/__init__.py tests/viz/test_smoke.py
git commit -m "viz: add membrane_trace (impl from scratch, not wrapping splt.traces)"
```

---

## Task 4: `weight_histogram` and `weight_heatmap`

**Files:**
- Create: `src/neuromorphic/viz/weights.py`
- Modify: `src/neuromorphic/viz/__init__.py`
- Modify: `tests/viz/test_smoke.py`

**Predict before executing:**
- `weight_histogram(W)` on `[100, 50]` random N(0, 0.1) → roughly Gaussian-shaped histogram centered at 0, span ~ ±0.4.
- `weight_heatmap(W)` → 100×50 image with a centered diverging colormap; bluish-red speckle, no obvious structure.
- Broken signal: histogram all in one bin (vmin/vmax bug); heatmap clipped to one color (vmin/vmax not symmetric).

- [ ] **Step 1: Write the failing tests**

Append to `tests/viz/test_smoke.py`:

```python
# ---------- weights ----------

def test_weight_histogram_returns_fig_ax(W):
    from neuromorphic.viz import weight_histogram
    fig, ax = weight_histogram(W)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_weight_histogram_bins_count(W):
    from neuromorphic.viz import weight_histogram
    fig, ax = weight_histogram(W, bins=20)
    # ax.hist returns BarContainer; count its patches.
    assert len(ax.patches) == 20
    plt.close(fig)


def test_weight_heatmap_returns_fig_ax(W):
    from neuromorphic.viz import weight_heatmap
    fig, ax = weight_heatmap(W)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_weight_heatmap_uses_symmetric_color_limits(W):
    from neuromorphic.viz import weight_heatmap
    fig, ax = weight_heatmap(W, center=0.0)
    images = ax.get_images()
    assert len(images) == 1
    vmin, vmax = images[0].get_clim()
    assert abs(vmin + vmax) < 1e-6, f"expected symmetric clim, got ({vmin}, {vmax})"
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -k "weight_" -v
```

Expected: `ImportError` on all 4.

- [ ] **Step 3: Implement `weights.py`**

Write `src/neuromorphic/viz/weights.py`:

```python
"""Weight-matrix visualization: 1D histogram of values and 2D heatmap.

The two public functions share a small ``_prep_weight`` helper for shape
validation and symmetric color-limit computation. Keep them in this module
so the helper stays private.
"""

from __future__ import annotations

import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuromorphic.viz.core import _ensure_ax


def _prep_weight(W: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Validate a weight tensor and return ``(detached_cpu, abs_max)``.

    Used by both ``weight_histogram`` (for axis-limit hints) and
    ``weight_heatmap`` (for symmetric vmin/vmax around zero).
    """
    if W.ndim != 2:
        raise ValueError(
            f"weight tensor must be 2D [N_post, N_pre], got shape {tuple(W.shape)}"
        )
    W_cpu = W.detach().cpu()
    abs_max = float(W_cpu.abs().max().item())
    return W_cpu, abs_max


def weight_histogram(
    W: torch.Tensor,
    bins: int = 50,
    ax: Axes | None = None,
    log_y: bool = False,
) -> tuple[Figure, Axes]:
    """1D histogram of synaptic weight values.

    Use this to spot dead synapses (mass at 0), saturation (mass at bounds),
    or pre/post-training distribution shifts.

    Args:
        W: ``[N_post, N_pre]`` weight matrix.
        bins: number of histogram bins.
        ax: matplotlib Axes. If None, a fresh fig/ax is created.
        log_y: log-scale the y-axis (useful when most weights cluster).

    Returns:
        ``(fig, ax)``.
    """
    fig, ax = _ensure_ax(ax)
    W_cpu, abs_max = _prep_weight(W)
    ax.hist(W_cpu.flatten().numpy(), bins=bins, color="C0",
            edgecolor="black", linewidth=0.4)
    if log_y:
        ax.set_yscale("log")
    ax.axvline(x=0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count" + (" (log)" if log_y else ""))
    ax.set_title(
        f"Weight distribution — n={W_cpu.numel()} synapses, |w|max={abs_max:.3f}"
    )
    return fig, ax


def weight_heatmap(
    W: torch.Tensor,
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    center: float = 0.0,
) -> tuple[Figure, Axes]:
    """2D heatmap of a weight matrix with diverging colormap centered at ``center``.

    Red = positive (excitatory), blue = negative (inhibitory) under the default
    ``RdBu_r`` colormap. Color limits are symmetric around ``center`` so sign
    information is preserved.

    Args:
        W: ``[N_post, N_pre]`` weight matrix.
        ax: matplotlib Axes. If None, a fresh fig/ax is created.
        cmap: matplotlib colormap name. Diverging maps recommended.
        center: value that should map to the colormap's center color.

    Returns:
        ``(fig, ax)``.
    """
    fig, ax = _ensure_ax(ax)
    W_cpu, abs_max = _prep_weight(W)
    span = max(abs_max, 1e-12)
    im = ax.imshow(
        W_cpu.numpy(),
        cmap=cmap,
        vmin=center - span,
        vmax=center + span,
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Weight value")
    ax.set_xlabel("Presynaptic neuron")
    ax.set_ylabel("Postsynaptic neuron")
    ax.set_title(
        f"Weight matrix — {W_cpu.shape[0]} × {W_cpu.shape[1]}, |w|max={abs_max:.3f}"
    )
    return fig, ax
```

- [ ] **Step 4: Expose both functions**

Modify `src/neuromorphic/viz/__init__.py`:

```python
"""Visualization toolkit for spiking neural networks.

All functions return ``(matplotlib.figure.Figure, matplotlib.axes.Axes)`` and
accept tensors in the canonical ``[T, B, N]`` shape. See
``docs/design/stage-1-visualization-toolkit.md``.
"""

from neuromorphic.viz.membrane import membrane_trace
from neuromorphic.viz.spikes import spike_raster
from neuromorphic.viz.weights import weight_heatmap, weight_histogram

__all__ = ["spike_raster", "membrane_trace", "weight_histogram", "weight_heatmap"]
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -k "weight_" -v
```

Expected: `4 passed`.

- [ ] **Step 6: Eyeball — render against the synthetic fixture**

```powershell
.venv\Scripts\python.exe -c "
import matplotlib
matplotlib.use('Agg')
import torch
from neuromorphic.viz import weight_histogram, weight_heatmap

torch.manual_seed(2)
W = torch.randn(100, 50) * 0.1

fig, ax = weight_histogram(W, bins=30)
fig.savefig('outputs/viz_smoke_weight_histogram.png', dpi=100, bbox_inches='tight')

fig, ax = weight_heatmap(W)
fig.savefig('outputs/viz_smoke_weight_heatmap.png', dpi=100, bbox_inches='tight')

print('saved 2 weight PNGs')
"
```

Open both PNGs. Histogram: roughly Gaussian, centered at 0, no spike at any single bin. Heatmap: blue/red speckle, colorbar shows symmetric range.

- [ ] **Step 7: Commit**

```bash
git add src/neuromorphic/viz/weights.py src/neuromorphic/viz/__init__.py tests/viz/test_smoke.py
git commit -m "viz: add weight_histogram and weight_heatmap (shared _prep_weight)"
```

---

## Task 5: `training_curve`

**Files:**
- Create: `src/neuromorphic/viz/training.py`
- Modify: `src/neuromorphic/viz/__init__.py`
- Modify: `tests/viz/test_smoke.py`

**Predict before executing:**
- `history = {"train_loss": [100 vals], "test_loss": [10 vals], "test_acc": [10 vals]}`, `log_interval=10`.
- `train_loss` plots at iterations `0..99` on the left axis.
- `test_loss` plots at iterations `0, 10, 20, ..., 90` on the left axis.
- `test_acc` plots at iterations `0, 10, ..., 90` on a twin right axis with y-range [0, 1].
- Broken signal: only one series visible (key matching wrong); accuracy plotted on left axis.

- [ ] **Step 1: Write the failing tests**

Append to `tests/viz/test_smoke.py`:

```python
# ---------- training_curve ----------

def test_training_curve_returns_fig_ax(history):
    from neuromorphic.viz import training_curve
    fig, ax = training_curve(history, log_interval=10)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_training_curve_creates_twin_axis_when_acc_present(history):
    from neuromorphic.viz import training_curve
    fig, ax = training_curve(history, log_interval=10)
    # A twin y-axis shares the same x-axis but has its own ylim.
    twins = [a for a in fig.axes if a is not ax]
    assert len(twins) >= 1, "expected a twin axis for accuracy series"
    plt.close(fig)


def test_training_curve_skips_missing_series():
    from neuromorphic.viz import training_curve
    only_loss = {"train_loss": [1.0, 0.5, 0.3]}
    fig, ax = training_curve(only_loss)
    twins = [a for a in fig.axes if a is not ax]
    assert len(twins) == 0, "no acc keys → no twin axis"
    plt.close(fig)


def test_training_curve_warns_on_unknown_key():
    import warnings
    from neuromorphic.viz import training_curve
    weird = {"train_loss": [1.0, 0.5], "mystery_metric": [0.2, 0.4]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, _ = training_curve(weird)
        assert any("mystery_metric" in str(w.message) for w in caught)
    import matplotlib.pyplot as plt
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -k training_curve -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `training.py`**

Write `src/neuromorphic/viz/training.py`:

```python
"""Training history plotting.

``training_curve`` takes a flexible ``{name: [values]}`` dict so the same
function works for loss-only runs and richer comparisons.
"""

from __future__ import annotations

import warnings

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuromorphic.viz.core import _ensure_ax


def _is_loss_key(name: str) -> bool:
    return "loss" in name.lower()


def _is_acc_key(name: str) -> bool:
    n = name.lower()
    return "acc" in n or "accuracy" in n


def _smooth(values: list[float], window: int) -> np.ndarray:
    """Centered rolling-mean smoothing. ``window=None`` or ``<2`` returns input."""
    if window is None or window < 2:
        return np.asarray(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def training_curve(
    history: dict[str, list[float]],
    log_interval: int = 1,
    smoothing_window: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot training-history series.

    Keys containing ``loss`` go on the left y-axis. Keys containing ``acc`` or
    ``accuracy`` go on a twin right y-axis with a fixed [0, 1] range. Keys
    matching neither prefix go on the left axis with a ``warnings.warn``.

    X-axis units are training iterations. Series whose name starts with
    ``train_`` are assumed to be sampled every iteration. All other series are
    assumed to be sampled every ``log_interval`` iterations.

    Args:
        history: ``{series_name: list_of_values}``. Empty or missing keys
            are silently skipped — only what's in ``history`` is drawn.
        log_interval: spacing (in iterations) of non-``train_`` series.
        smoothing_window: optional rolling-mean window applied to every series.
        ax: matplotlib Axes. If None, a fresh fig/ax is created.

    Returns:
        ``(fig, ax)`` where ``ax`` is the left (loss) axis. The accuracy twin
        axis, if any, is accessible via ``fig.axes``.
    """
    fig, ax = _ensure_ax(ax)
    twin = None
    color_cycle_loss = ["C0", "C1", "C4", "C5"]
    color_cycle_acc = ["C2", "C3", "C6", "C7"]
    loss_i = acc_i = 0

    for name, values in history.items():
        if not values:
            continue
        ys = _smooth(values, smoothing_window)
        n = len(ys)
        stride = 1 if name.startswith("train_") else log_interval
        xs = np.arange(n) * stride

        if _is_loss_key(name):
            ax.plot(xs, ys, label=name, color=color_cycle_loss[loss_i % len(color_cycle_loss)])
            loss_i += 1
        elif _is_acc_key(name):
            if twin is None:
                twin = ax.twinx()
                twin.set_ylabel("Accuracy")
                twin.set_ylim(0.0, 1.0)
            twin.plot(xs, ys, label=name, linestyle="--",
                      color=color_cycle_acc[acc_i % len(color_cycle_acc)])
            acc_i += 1
        else:
            warnings.warn(
                f"training_curve: key '{name}' matches neither 'loss' nor 'acc' "
                "prefixes; plotting on the loss axis."
            )
            ax.plot(xs, ys, label=name, alpha=0.6)

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)

    # Single combined legend across both axes.
    handles, labels = ax.get_legend_handles_labels()
    if twin is not None:
        h2, l2 = twin.get_legend_handles_labels()
        handles += h2
        labels += l2
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8)

    return fig, ax
```

- [ ] **Step 4: Expose `training_curve`**

Modify `src/neuromorphic/viz/__init__.py`:

```python
"""Visualization toolkit for spiking neural networks.

All functions return ``(matplotlib.figure.Figure, matplotlib.axes.Axes)`` and
accept tensors in the canonical ``[T, B, N]`` shape. See
``docs/design/stage-1-visualization-toolkit.md``.
"""

from neuromorphic.viz.membrane import membrane_trace
from neuromorphic.viz.spikes import spike_raster
from neuromorphic.viz.training import training_curve
from neuromorphic.viz.weights import weight_heatmap, weight_histogram

__all__ = [
    "spike_raster",
    "membrane_trace",
    "weight_histogram",
    "weight_heatmap",
    "training_curve",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -k training_curve -v
```

Expected: `4 passed`.

- [ ] **Step 6: Eyeball — render against the synthetic fixture**

```powershell
.venv\Scripts\python.exe -c "
import matplotlib
matplotlib.use('Agg')
from neuromorphic.viz import training_curve

history = {
    'train_loss': [3.0 - 0.01 * i for i in range(100)],
    'test_loss':  [3.0 - 0.02 * i for i in range(10)],
    'test_acc':   [0.1 + 0.08 * i for i in range(10)],
}
fig, ax = training_curve(history, log_interval=10)
ax.set_title('training_curve smoke (synthetic)')
fig.savefig('outputs/viz_smoke_training_curve.png', dpi=100, bbox_inches='tight')
print('saved outputs/viz_smoke_training_curve.png')
"
```

Open the PNG. Confirm: descending train_loss on left axis 0–99, descending test_loss with markers at 0/10/.../90, ascending test_acc on right axis with [0,1] range.

- [ ] **Step 7: Commit**

```bash
git add src/neuromorphic/viz/training.py src/neuromorphic/viz/__init__.py tests/viz/test_smoke.py
git commit -m "viz: add training_curve with twin-axis loss/acc routing"
```

---

## Task 6: `population_rate`

**Files:**
- Modify: `src/neuromorphic/viz/spikes.py`
- Modify: `src/neuromorphic/viz/__init__.py`
- Modify: `tests/viz/test_smoke.py`

**Predict before executing:**
- Input `spk` is `[20, 4, 50]` at ~20% firing rate.
- `spk.mean(dim=(1, 2))` → `[20]` vector, each value ≈ 0.2.
- Output: flat-ish line near y=0.2, x-axis 0–19.
- With `smoothing_window=5`, output length shrinks to 16 via valid convolution.
- Broken signal: line at constant 0 (sum vs mean confusion); or wrong length.

- [ ] **Step 1: Write the failing tests**

Append to `tests/viz/test_smoke.py`:

```python
# ---------- population_rate ----------

def test_population_rate_returns_fig_ax(spk):
    from neuromorphic.viz import population_rate
    fig, ax = population_rate(spk)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_population_rate_line_length_matches_T(spk):
    from neuromorphic.viz import population_rate
    fig, ax = population_rate(spk)
    lines = ax.get_lines()
    assert len(lines) == 1
    assert lines[0].get_xdata().size == spk.shape[0]
    plt.close(fig)


def test_population_rate_smoothing_shrinks_line(spk):
    from neuromorphic.viz import population_rate
    fig, ax = population_rate(spk, smoothing_window=5)
    lines = ax.get_lines()
    # Valid convolution: length T - window + 1.
    assert lines[0].get_xdata().size == spk.shape[0] - 5 + 1
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -k population_rate -v
```

Expected: `ImportError`.

- [ ] **Step 3: Append `population_rate` to `spikes.py`**

First, add `numpy` to the imports at the top of `src/neuromorphic/viz/spikes.py`:

```diff
 from __future__ import annotations

+import numpy as np
 import snntorch.spikeplot as splt
 import torch
 from matplotlib.axes import Axes
 from matplotlib.figure import Figure

 from neuromorphic.viz.core import _ensure_ax, _select_sample
```

Then append the new function at the bottom:

```python
def population_rate(
    spk: torch.Tensor,
    smoothing_window: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Mean firing rate of an entire layer over time.

    For each time step, compute the fraction of (batch, neuron) pairs that
    spiked. Plot as a single line over time.

    Use this as the load-bearing diagnostic for layer-level activity (see
    parent design doc §3.4): saturation → broken inhibition; collapse to
    zero → dead layer.

    Args:
        spk: ``[T, B, N]`` spike tensor.
        smoothing_window: optional rolling-mean width (in time steps).
        ax: matplotlib Axes. If None, a fresh fig/ax is created.

    Returns:
        ``(fig, ax)``.
    """
    if spk.ndim != 3:
        raise ValueError(
            f"population_rate expects a [T, B, N] tensor, got shape {tuple(spk.shape)}"
        )
    fig, ax = _ensure_ax(ax)
    rate = spk.float().mean(dim=(1, 2)).detach().cpu().numpy()

    if smoothing_window is not None and smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        rate = np.convolve(rate, kernel, mode="valid")

    x = np.arange(len(rate))
    ax.plot(x, rate, color="C0", linewidth=1.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean firing rate (spikes / neuron / step)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    return fig, ax
```

- [ ] **Step 4: Expose `population_rate`**

Modify `src/neuromorphic/viz/__init__.py`:

```python
"""Visualization toolkit for spiking neural networks.

All functions return ``(matplotlib.figure.Figure, matplotlib.axes.Axes)`` and
accept tensors in the canonical ``[T, B, N]`` shape. See
``docs/design/stage-1-visualization-toolkit.md``.
"""

from neuromorphic.viz.membrane import membrane_trace
from neuromorphic.viz.spikes import population_rate, spike_raster
from neuromorphic.viz.training import training_curve
from neuromorphic.viz.weights import weight_heatmap, weight_histogram

__all__ = [
    "spike_raster",
    "membrane_trace",
    "weight_histogram",
    "weight_heatmap",
    "training_curve",
    "population_rate",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -k population_rate -v
```

Expected: `3 passed`.

- [ ] **Step 6: Eyeball — render against the synthetic fixture**

```powershell
.venv\Scripts\python.exe -c "
import matplotlib
matplotlib.use('Agg')
import torch
from neuromorphic.viz import population_rate

torch.manual_seed(0)
spk = (torch.rand(20, 4, 50) < 0.2).float()
fig, ax = population_rate(spk)
ax.set_title('population_rate smoke (synthetic, target ~0.2)')
fig.savefig('outputs/viz_smoke_population_rate.png', dpi=100, bbox_inches='tight')
print('saved outputs/viz_smoke_population_rate.png')
"
```

Open the PNG. Confirm: jagged line hovering near y=0.2, x-axis 0–19.

- [ ] **Step 7: Commit**

```bash
git add src/neuromorphic/viz/spikes.py src/neuromorphic/viz/__init__.py tests/viz/test_smoke.py
git commit -m "viz: add population_rate (layer mean firing rate over time)"
```

---

## Task 7: `psth`

**Files:**
- Modify: `src/neuromorphic/viz/spikes.py`
- Modify: `src/neuromorphic/viz/__init__.py`
- Modify: `tests/viz/test_smoke.py`

**Predict before executing:**
- `spk` is `[20, 4, 50]`. Total spikes per time step = `spk.sum(dim=(1, 2))` → `[20]`. Expected ≈ `4 * 50 * 0.2 = 40` spikes/step.
- With `bin_size=5`, we get 4 bins, each pooling 5 time steps → expected counts ≈ `5 * 40 = 200` per bin.
- We plot bars at bin centers (2.5, 7.5, 12.5, 17.5) — heights ≈ 200 each.
- Broken signal: bin count wrong (T not divisible by bin_size handling); heights off by factor of 5 (bin pooling missed).

- [ ] **Step 1: Write the failing tests**

Append to `tests/viz/test_smoke.py`:

```python
# ---------- psth ----------

def test_psth_returns_fig_ax(spk):
    from neuromorphic.viz import psth
    fig, ax = psth(spk, bin_size=5)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_psth_correct_bin_count(spk):
    from neuromorphic.viz import psth
    # T=20, bin_size=5 → 4 bins.
    fig, ax = psth(spk, bin_size=5)
    assert len(ax.patches) == 4
    plt.close(fig)


def test_psth_truncates_when_T_not_divisible():
    from neuromorphic.viz import psth
    # T=22 with bin_size=5: only 4 full bins; last 2 time steps dropped.
    spk = (torch.rand(22, 4, 50) < 0.2).float()
    fig, ax = psth(spk, bin_size=5)
    assert len(ax.patches) == 4
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -k "psth" -v
```

Expected: `ImportError`.

- [ ] **Step 3: Append `psth` to `spikes.py`**

Edit `src/neuromorphic/viz/spikes.py` — add at the bottom:

```python
def psth(
    spk: torch.Tensor,
    bin_size: int = 5,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Peri-stimulus time histogram: total population spikes per time bin.

    Pools across batch and neuron axes into ``bin_size``-wide time bins, then
    bar-plots the sum per bin. If ``T`` is not divisible by ``bin_size``,
    trailing time steps that don't fill a full bin are dropped.

    Args:
        spk: ``[T, B, N]`` spike tensor.
        bin_size: width of each bin in time steps.
        ax: matplotlib Axes. If None, a fresh fig/ax is created.

    Returns:
        ``(fig, ax)``.
    """
    if spk.ndim != 3:
        raise ValueError(
            f"psth expects a [T, B, N] tensor, got shape {tuple(spk.shape)}"
        )
    if bin_size < 1:
        raise ValueError(f"bin_size must be >= 1, got {bin_size}")

    fig, ax = _ensure_ax(ax)
    per_step = spk.sum(dim=(1, 2)).detach().cpu()  # [T]
    T = per_step.shape[0]
    n_bins = T // bin_size
    truncated = per_step[: n_bins * bin_size].reshape(n_bins, bin_size).sum(dim=1)
    bin_centers = (torch.arange(n_bins) + 0.5) * bin_size

    ax.bar(bin_centers.numpy(), truncated.numpy(), width=bin_size * 0.9,
           color="C0", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Time step")
    ax.set_ylabel(f"Total spikes per {bin_size}-step bin")
    ax.set_title(f"PSTH — {n_bins} bins of width {bin_size}")
    return fig, ax
```

- [ ] **Step 4: Expose `psth`**

Modify `src/neuromorphic/viz/__init__.py`:

```python
"""Visualization toolkit for spiking neural networks.

All functions return ``(matplotlib.figure.Figure, matplotlib.axes.Axes)`` and
accept tensors in the canonical ``[T, B, N]`` shape. See
``docs/design/stage-1-visualization-toolkit.md``.
"""

from neuromorphic.viz.membrane import membrane_trace
from neuromorphic.viz.spikes import population_rate, psth, spike_raster
from neuromorphic.viz.training import training_curve
from neuromorphic.viz.weights import weight_heatmap, weight_histogram

__all__ = [
    "spike_raster",
    "membrane_trace",
    "weight_histogram",
    "weight_heatmap",
    "training_curve",
    "population_rate",
    "psth",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/test_smoke.py -k "psth" -v
```

Expected: `3 passed`.

- [ ] **Step 6: Eyeball — render against the synthetic fixture**

```powershell
.venv\Scripts\python.exe -c "
import matplotlib
matplotlib.use('Agg')
import torch
from neuromorphic.viz import psth

torch.manual_seed(0)
spk = (torch.rand(20, 4, 50) < 0.2).float()
fig, ax = psth(spk, bin_size=5)
ax.set_title('psth smoke (synthetic, target ~200/bin)')
fig.savefig('outputs/viz_smoke_psth.png', dpi=100, bbox_inches='tight')
print('saved outputs/viz_smoke_psth.png')
"
```

Open the PNG. Confirm: 4 bars centered at x=2.5, 7.5, 12.5, 17.5 each near height 200.

- [ ] **Step 7: Commit**

```bash
git add src/neuromorphic/viz/spikes.py src/neuromorphic/viz/__init__.py tests/viz/test_smoke.py
git commit -m "viz: add psth (population spikes per time bin)"
```

---

## Task 8: Full test sweep and `__init__.py` confirmation

**Files:**
- (No new files. Final verification + commit if anything moves.)

- [ ] **Step 1: Run the full smoke suite**

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/ -v
```

Expected: all tests pass. Count should be: 5 (core) + 2 (spike_raster) + 3 (membrane_trace) + 4 (weights) + 4 (training_curve) + 3 (population_rate) + 3 (psth) = **24 passed**.

If anything fails, fix in place — do not move on with a red suite.

- [ ] **Step 2: Confirm the public API surface**

```powershell
.venv\Scripts\python.exe -c "
from neuromorphic.viz import (
    spike_raster, membrane_trace,
    weight_histogram, weight_heatmap,
    training_curve, population_rate, psth,
)
import neuromorphic.viz as v
print('public:', v.__all__)
"
```

Expected output:

```
public: ['spike_raster', 'membrane_trace', 'weight_histogram', 'weight_heatmap', 'training_curve', 'population_rate', 'psth']
```

- [ ] **Step 3: No commit needed (init was finalized in Task 7).** Proceed to Task 9.

---

## Task 9: `viz_smoke.py` — live eyeball against the week-5 checkpoint

**Files:**
- Create: `experiments/008_week5_snn_mnist_baseline/viz_smoke.py`

This is the layer-B test from the spec §7.2 — pytest passes on valid-but-visually-broken output; this script renders real model data so a human can sanity-check.

**Predict before executing:**
- `spike_raster` on input layer (784 neurons, 25 time steps, rate=0.13 on average for normalized MNIST) → sparse dots, mostly empty.
- `membrane_trace` on the 10 output neurons → 10 lines rising toward threshold, one (the target class) reaching it most.
- `weight_histogram` on `fc1.weight` (1000×784) → roughly Gaussian, slightly shifted by training.
- `weight_heatmap` on `fc1.weight` → noisy speckle; structure (if any) is subtle.
- `training_curve` on the checkpoint's `loss_hist`/`test_loss_hist`/`test_acc_hist` → loss descending, accuracy ascending.
- `population_rate` on hidden-layer spikes → average activity around 0.1–0.3.
- `psth` on output-layer spikes → 5 bins (25 / 5) with bars showing when output neurons fired.

- [ ] **Step 1: Write the eyeball script**

Write `experiments/008_week5_snn_mnist_baseline/viz_smoke.py`:

```python
"""Render every neuromorphic.viz function against the week-5 SNN checkpoint.

This is the human-eyeball test mandated by the parent spec §7.2 — pytest
asserts the contract, this script asserts that the contract produces
visually sensible plots on real data.

Usage:
    python experiments/008_week5_snn_mnist_baseline/viz_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Make the existing run2.py importable so we can reconstruct the network.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run2 import FeedforwardSNN, encode_rate  # noqa: E402

from neuromorphic.config import load_config  # noqa: E402
from neuromorphic.utils import get_device, set_seed  # noqa: E402
from neuromorphic.viz import (  # noqa: E402
    membrane_trace,
    population_rate,
    psth,
    spike_raster,
    training_curve,
    weight_heatmap,
    weight_histogram,
)


def _capture_full_forward(net, spk_in):
    """Mirror the helper from week-5 plots.py — record spikes from every layer."""
    net.eval()
    with torch.no_grad():
        mem1, mem2, mem3 = (net.lif1.init_leaky(),
                            net.lif2.init_leaky(),
                            net.lif3.init_leaky())
        s1, s2, s3, m3 = [], [], [], []
        for step in range(net.num_steps):
            cur1 = net.fc1(spk_in[step]); spk1, mem1 = net.lif1(cur1, mem1)
            cur2 = net.fc2(spk1);          spk2, mem2 = net.lif2(cur2, mem2)
            cur3 = net.fc3(spk2);          spk3, mem3 = net.lif3(cur3, mem3)
            s1.append(spk1); s2.append(spk2); s3.append(spk3); m3.append(mem3)
    return (torch.stack(s1), torch.stack(s2),
            torch.stack(s3), torch.stack(m3))


def main():
    config = load_config(HERE / "config.yaml")
    set_seed(config.seed)
    device = get_device()

    out_dir = HERE / "outputs" / "snn_mnist_baseline" / "viz_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = HERE / "outputs" / "snn_mnist_baseline" / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Week-5 checkpoint missing at {ckpt_path}. Run run2.py first."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    net = FeedforwardSNN(
        num_inputs=config.num_inputs,
        hidden_dims=tuple(config.hidden_dims),
        num_outputs=config.num_outputs,
        beta=config.beta,
        threshold=config.threshold,
        reset_mechanism=config.reset_mechanism,
        num_steps=config.num_steps,
    ).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    print(f"Loaded week-5 net, final test acc was {ckpt['final_accuracy']*100:.2f}%")

    # ---- Fetch one batch of MNIST for the spike/membrane plots ----
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])
    test_set = datasets.MNIST(config.data_root, train=False, download=False,
                              transform=transform)
    loader = DataLoader(test_set, batch_size=8, shuffle=True)
    data, targets = next(iter(loader))
    data_flat = data.view(data.size(0), -1).to(device)
    spk_in = encode_rate(data_flat, num_steps=config.num_steps, gain=config.gain)
    spk1, spk2, spk3, mem3 = _capture_full_forward(net, spk_in)
    print(f"sample target={targets[0].item()}, "
          f"pred={spk3[:, 0, :].sum(dim=0).argmax().item()}")

    # ---- Render every plot ----
    plots = []

    fig, ax = spike_raster(spk_in, sample_idx=0)
    ax.set_title("input-layer raster (784 neurons, sample 0)")
    plots.append(("01_spike_raster_input.png", fig))

    fig, ax = spike_raster(spk1, sample_idx=0)
    ax.set_title("hidden-1 raster (1000 neurons, sample 0)")
    plots.append(("02_spike_raster_hidden1.png", fig))

    fig, ax = membrane_trace(mem3, sample_idx=0, threshold=config.threshold)
    ax.set_title(f"output membrane traces (target={targets[0].item()})")
    plots.append(("03_membrane_trace_output.png", fig))

    fig, ax = weight_histogram(net.fc1.weight)
    plots.append(("04_weight_histogram_fc1.png", fig))

    fig, ax = weight_heatmap(net.fc3.weight)
    plots.append(("05_weight_heatmap_fc3.png", fig))

    history = {
        "train_loss": ckpt["loss_hist"],
        "test_loss":  ckpt["test_loss_hist"],
        "test_acc":   ckpt["test_acc_hist"],
    }
    fig, ax = training_curve(history, log_interval=config.log_interval)
    ax.set_title("week-5 training history")
    plots.append(("06_training_curve.png", fig))

    fig, ax = population_rate(spk2, smoothing_window=3)
    ax.set_title("hidden-2 population rate")
    plots.append(("07_population_rate_hidden2.png", fig))

    fig, ax = psth(spk3, bin_size=5)
    ax.set_title("output-layer PSTH")
    plots.append(("08_psth_output.png", fig))

    for filename, fig in plots:
        path = out_dir / filename
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")

    print(f"\nAll 8 plots saved under {out_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```powershell
.venv\Scripts\python.exe experiments/008_week5_snn_mnist_baseline/viz_smoke.py
```

Expected: prints loaded-checkpoint line, sample target/pred, then 8 `saved …` lines.

- [ ] **Step 3: Open each PNG and compare to predictions**

Open every file under `experiments/008_week5_snn_mnist_baseline/outputs/snn_mnist_baseline/viz_smoke/`. For each, write one sentence (in your head or in a scratch note) about whether the plot matches the prediction. If anything looks wrong (no spikes, all-flat membrane, histogram collapsed to one bin, training curve flat), stop and diagnose — do not commit a broken toolkit.

- [ ] **Step 4: Commit**

```bash
git add experiments/008_week5_snn_mnist_baseline/viz_smoke.py
git commit -m "viz: add week-5 eyeball script rendering all 7 toolkit functions"
```

---

## Done criteria

Toolkit is complete when:

1. `pytest tests/viz/ -v` reports `24 passed` with no warnings about deprecated APIs.
2. `python experiments/008_week5_snn_mnist_baseline/viz_smoke.py` runs without error and produces 8 PNGs.
3. All 8 PNGs visually match the predictions in Task 9.
4. `from neuromorphic.viz import spike_raster, membrane_trace, weight_histogram, weight_heatmap, training_curve, population_rate, psth` succeeds.

If any of the four fails, fix before declaring the morning session done. The afternoon optimization work depends on `training_curve` + `weight_histogram` in particular.
