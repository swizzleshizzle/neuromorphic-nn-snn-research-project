# Stage 1 Viz Toolkit — Session Implementation Spec

**Date:** 2026-05-16 (Saturday hands-on, Week 6)
**Status:** Approved by Mike, ready for implementation.
**Parent spec:** [`stage-1-visualization-toolkit.md`](stage-1-visualization-toolkit.md)
**Implementing agent context:** This doc reconciles the parent design spec with today's session goals and locks the exact contract, signatures, files, and build order. Follow it verbatim during the build.

---

## 1. Why this doc exists

The parent Stage 1 spec is the long-form design. This is the session-scoped implementation contract for today's 90-minute build slot. Two reconciliations resolved:

1. **Scope** — session goals named 4 functions (`spike_raster`, `membrane_trace`, `weight_histogram`, `training_curve`); the parent doc's MVP slice named 5 different ones (`plot_raster`, `plot_membrane`, `plot_population_rate`, `plot_psth`, plus stretch `plot_weights`). Decision: build the **union** as 7 functions today (see §3).
2. **Naming** — parent doc uses `plot_*` prefix; session goals use bare names. Decision: **bare names** across the toolkit (module name `viz` is enough context).

Tensor-shape contract `[T, B, N]` from the parent doc is **verified**: `experiments/008_week5_snn_mnist_baseline/run2.py:92-119` returns `(torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0))` where each appended element is `[B, N]` — final shape is `[T, B, N]`. No `forward_pass()` helper exists in the repo yet; the network's own `forward` is the source of truth.

---

## 2. Module location

```
src/neuromorphic/viz/
```

Imported as:

```python
from neuromorphic.viz import (
    spike_raster, membrane_trace,
    weight_histogram, weight_heatmap,
    training_curve, population_rate, psth,
)
```

Selected because: matches parent doc verbatim, shorter import path than `visualization/`, and consistent with the parent doc's file split (`spikes.py`, `weights.py`, etc.).

---

## 3. Functions in scope today (7)

| # | Function | Source | Type |
|---|---|---|---|
| 1 | `spike_raster` | session + parent §3.1 | thin wrap of `splt.raster` |
| 2 | `membrane_trace` | session + parent §3.2 | thin wrap of `splt.traces` + threshold line |
| 3 | `weight_histogram` | session only (new) | `ax.hist(W.flatten())` |
| 4 | `weight_heatmap` | parent §3.7 (renamed from `plot_weights`) | `ax.imshow(W, cmap='RdBu_r', center=0)` |
| 5 | `training_curve` | session only (new) | matplotlib-native; see §5 |
| 6 | `population_rate` | parent §3.4 | `spk.mean(dim=(1,2))` line plot |
| 7 | `psth` | parent §3.3 | binned population-firing histogram |

Deferred (parent doc lists, not built today): `plot_tuning_curve`, `plot_overlaps`, `plot_layer_rasters`, `plot_isi`, `plot_output_counts`, `plot_confusion`.

---

## 4. Contracts (load-bearing)

These are non-negotiable. Surface violations loudly; do not paper over with `try/except`.

| Contract | Rule |
|---|---|
| **Tensor shape** | Spikes/membranes are `[T, B, N]`. Conv-layer outputs `[T, B, C, H, W]` are the caller's responsibility to flatten via `.view(T, B, -1)` before passing. |
| **Return type** | Every function returns `(fig, ax)`. Never `plt.show()`, never `fig.savefig()`. |
| **Sample indexing** | Functions that visualize one sample (`spike_raster`, `membrane_trace`) accept either pre-indexed `[T, N]` or `[T, B, N]` + `sample_idx=0`. If 3D and no index, default to sample 0. |
| **Axis injection** | Every function accepts `ax=None`. If None, `_ensure_ax()` in `core.py` produces a fresh `(fig, ax)`. If provided, function draws on it and returns `(ax.figure, ax)`. |
| **Dependencies** | `matplotlib`, `numpy`, `torch`, `snntorch`. No `seaborn`, no `plotly`, no `sklearn` today (not building `plot_confusion`). |

---

## 5. Function signatures

```python
# core.py
def _ensure_ax(ax) -> tuple[Figure, Axes]: ...
def _select_sample(tensor: Tensor, sample_idx: int = 0) -> Tensor: ...
    # Accepts [T, B, N] or [T, N]; if 3D, returns tensor[:, sample_idx, :].

# spikes.py
def spike_raster(spk, ax=None, sample_idx=0, **kwargs) -> tuple[Figure, Axes]
def psth(spk, bin_size: int = 5, ax=None) -> tuple[Figure, Axes]
def population_rate(spk, smoothing_window: int | None = None,
                    ax=None) -> tuple[Figure, Axes]

# membrane.py
def membrane_trace(mem, spk=None, ax=None, sample_idx: int = 0,
                   threshold: float = 1.0,
                   show_threshold: bool = True) -> tuple[Figure, Axes]

# weights.py
def weight_histogram(W, bins: int = 50, ax=None,
                     log_y: bool = False) -> tuple[Figure, Axes]
def weight_heatmap(W, ax=None, cmap: str = 'RdBu_r',
                   center: float = 0.0) -> tuple[Figure, Axes]
# Both call a shared _prep_weight(W) for shape validation and symmetric vmin/vmax.

# training.py
def training_curve(history: dict[str, list[float]],
                   log_interval: int = 1,
                   smoothing_window: int | None = None,
                   ax=None) -> tuple[Figure, Axes]
```

**`training_curve` semantics:**
- `history` keys containing `loss` → left y-axis, log-scale optional.
- `history` keys containing `acc` or `accuracy` → twin right y-axis, fixed [0, 1] range.
- Keys matching neither prefix → left axis with a `warnings.warn()`.
- `log_interval` applies to test/eval series (assumed sampled every `log_interval` train iters). Train-series keys plotted per-iteration.
- Missing series silently skipped — only what's in `history` gets drawn.

---

## 6. File layout

```
src/neuromorphic/viz/
├── __init__.py     # flat re-export of all 7 public functions
├── core.py         # _ensure_ax, _select_sample, shape validators
├── spikes.py       # spike_raster, psth, population_rate
├── membrane.py     # membrane_trace
├── weights.py      # weight_histogram, weight_heatmap, _prep_weight
└── training.py     # training_curve
```

`tuning.py`, `memory.py`, `composite.py` from the parent doc are **not created today**.

---

## 7. Testing strategy (two layers, both required)

### 7.1 Pytest smoke tests — `tests/viz/test_smoke.py`

One fixture file produces synthetic tensors at module scope:

```python
T, B, N = 20, 4, 50
spk = (torch.rand(T, B, N) < 0.2).float()
mem = torch.randn(T, B, N)
W   = torch.randn(100, 50)
history = {"train_loss": [...], "test_loss": [...], "test_acc": [...]}
```

Every plot function gets exactly one test that asserts:
1. Returns a `(Figure, Axes)` tuple.
2. Does not raise.
3. `plt.close(fig)` succeeds (catches any backend issues).

Run: `pytest tests/viz/ -q`.

### 7.2 Live visual check against week-5 checkpoint

A one-off `experiments/008_week5_snn_mnist_baseline/viz_smoke.py` that:
1. Loads `outputs/snn_mnist_baseline/checkpoint.pt`.
2. Reconstructs the FeedforwardSNN, runs one batch, captures spk/mem.
3. Renders all 7 plots and saves to `outputs/snn_mnist_baseline/viz_smoke/`.

This is the human-eyeball test — pytest passes on technically-valid-but-visually-broken output.

### 7.3 Cadence

Per the parent doc's "non-negotiable" rule: **implement one function → run its pytest smoke → eyeball one render against week-5 data → move to next.** Not all-7-then-test-at-end.

---

## 8. Build order (90 min budget)

```
0:00  core.py — _ensure_ax, _select_sample, shape validators            ~10 min
0:10  spike_raster + smoke + eyeball                                    ~10 min
0:20  membrane_trace + smoke + eyeball                                  ~10 min
0:30  weight_histogram + weight_heatmap + smoke + eyeball               ~15 min
0:45  training_curve + smoke + eyeball                                  ~15 min
1:00  population_rate + smoke + eyeball                                 ~10 min
1:10  psth + smoke + eyeball                                            ~15 min
1:25  full test run, __init__.py finalize, viz_smoke.py finalize        ~5 min
1:30  done
```

**Cut order if running long:** drop `psth`, then `population_rate`. These have no immediate consumer in the afternoon optimization work. Functions 1–5 directly serve the architecture-comparison table.

---

## 9. Predict-before-execute discipline

Parent doc §"Agent behavior expectations" mandates this. Before each function:
1. State expected output shape of any intermediate tensor.
2. State what the plot should look like.
3. State what would indicate it's broken.
4. Run it. Compare prediction to result. Note divergences.

This is the project-wide pattern from Week 4 RL onward.

---

## 10. Non-goals for today

From parent doc §"What NOT to do" and reaffirmed here:

- No new dependencies beyond matplotlib/numpy/torch/snntorch.
- No animation work — `splt.animator` exists; custom animation is Phase 2.
- No expansion to the 11-plot scope mid-build. Ideas go to a "Phase 2 candidate" list, not into today's work.
- No stylesheet, no colormap discovery work, no `save_path=` kwargs. Parent doc §6 open questions stay open — decide in-context only if forced.

---

## 11. After the toolkit (afternoon work, brainstorm separately)

Session goal 2 is "optimize MNIST SNN to ≥95%+ via spiking conv layers / num_steps / beta sweeps, produce architecture × accuracy × training-time comparison table, save best checkpoint." This is **not** covered by the parent design doc or this implementation spec. Brainstorm as a separate design exercise once toolkit work completes.

---

## Revision history

| Date | Change |
|---|---|
| 2026-05-16 | Initial draft. Reconciles session goals with parent Stage 1 spec; locks 7 functions, signatures, file layout, build order, testing strategy. |
