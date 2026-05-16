
# Stage 1 Visualization Toolkit — Design Doc

**Phase 1, Module L12** — Visualization toolkit design (build target: Saturday 2026-05-16)
**Status:** Design only. No code written. Implementation tracked separately.
**Related:** [[week-06-optimize-visualize#Session 3 — 2026-05-15 (Fri)|Week 6 Session 3]] · [[week-05-first-snn|Week 5 — First SNN]] · [[week-06-optimize-visualize|Week 6 — Optimize & Visualize]]

---

## Agent Hand-Off Context — Read This First

This design doc is the **blueprint**, not the full briefing. When you (the agent) start the Saturday implementation session, surface the following before writing any code:

### Required additional context

1. **Repo structure and conventions** — open `src/` in the `neuromorphic-nn-snn-research-project` repo. Pay attention to:
   - The `ExperimentConfig` dataclass + YAML loading pattern established in Week 3. The viz toolkit should *compose* with this, not duplicate it.
   - The src-layout (`src/neuromorphic/...`). The new viz module lives at `src/neuromorphic/viz/`.
   - Existing test conventions in `tests/`. Match them in `tests/viz/`.

2. **Tensor source — `forward_pass()` from Week 6 Session 1** — the design doc claims `[T, B, N]` is the canonical shape because `forward_pass(net, num_steps, data)` produces it. **Verify this yourself.** Read [[week-06-optimize-visualize#2.4 Forward Pass with forward_pass() Helper|Week 6 §2.4]] and trace the actual return shape (`torch.stack(spk_rec)` over `num_steps` iterations of a network that may have `init_hidden=True`). If the real shape differs from the contract, flag it before locking the API.

3. **Hopfield/overlap context (Phase 2 prep)** — the `plot_overlaps` function (§3.11) ties to memory-region design from [[week-06-optimize-visualize#Session 2|Week 6 Session 2]] / Gerstner Ch. 17. Skim that session before designing the overlap function signature — `patterns` shape `[M, N]` and the binary-state assumption come from there.

4. **Open decision: `[T, B, N]` vs `[B, T, N]`** — the doc currently locks `[T, B, N]` (snnTorch-native), but PyTorch's general convention is batch-first. **Argue a position before writing code.** The right answer depends on whether the toolkit is downstream-of-snnTorch (`[T, B, N]` wins, no permute) or downstream-of-general-PyTorch (`[B, T, N]` wins, matches `DataLoader` output). State the trade-off, commit, move on.

### Agent behavior expectations

Mike's preferred learning/build mode is **scaffolded with Socratic checkpoints**. Specifically:

- **Don't bang out all 5 MVP plots and test at the end.** Implement one, smoke-test it, confirm the (`fig`, `ax`) tuple contract works against a synthetic `[T, B, N]` tensor, *then* move to the next. The "smoke test per item" in §5 is non-negotiable, not a stretch goal.
- **Predict before executing.** Before each function: state what shape you expect to come out, what the plot should look like, and what would indicate it's broken. Then run it. Compare prediction to result. This is the project-wide pattern from Week 4 RL onward.
- **Surface contract violations loudly.** If a wrapped `snntorch.spikeplot` function doesn't fit cleanly into the `(fig, ax)` return contract — say so. Don't paper over it with `try/except` or silent conversion. The contract is the load-bearing design decision.
- **Defer the open questions in §6 to in-context decisions during the build.** Don't try to resolve them upfront. The colormap / save / time-axis / animation / stylesheet decisions become obvious once two or three functions exist.

### What NOT to do

- Don't expand scope. The 11 plot types are fixed. New plot ideas during the build go into a "Phase 2 candidate" list, not into Saturday's work.
- Don't pre-build the animation versions. `splt.animator` and `splt.spike_count(animate=True)` already exist; custom animations are explicitly deferred to Phase 2 dashboard work.
- Don't introduce new dependencies beyond `matplotlib`, `numpy`, `torch`, `snntorch`, and `sklearn` (for confusion matrix). If a plot tempts you toward `seaborn` or `plotly`, resist — Phase 1 is matplotlib-only by design.
- Don't lock the `[T, B, N]` vs `[B, T, N]` decision without checking actual `forward_pass()` output first.

---

## Purpose & Scope

The Phase 1 viz toolkit has one job: **let you look at a single SNN's behavior and understand what it's doing.** Phase 2–3 will upgrade this to a multi-region live dashboard. Stage 1 = static matplotlib, single network, post-hoc analysis.

**In scope (Stage 1):**
- Static matplotlib plots of spike trains, membrane potentials, weights, and population dynamics
- Wrapping `snntorch.spikeplot` where built-ins exist
- Neuroscience-standard plots `snntorch.spikeplot` doesn't provide (PSTH, ISI, tuning curves, population rate)
- A consistent tensor-shape contract across all functions

**Out of scope (deferred to Phase 2–3):**
- Live / streaming dashboard
- Multi-region composite displays beyond simple multi-layer rasters
- Custom animations beyond what `splt.animator` provides
- Interactive plotting (plotly, bokeh, etc.)

---

## Design Philosophy

Three principles drive the design:

1. **Wrap, don't replace, `snntorch.spikeplot`.** It already covers raster, membrane traces, and animations. Reimplementing them is wasted effort. The toolkit's value is in the *neuroscience-standard* plots snnTorch *doesn't* provide (PSTH, tuning curves, ISI histograms, population activity), plus a consistent input contract across everything.

2. **One canonical input shape.** All toolkit functions accept `[T, B, N]` tensors — time, batch, neurons. This matches what snnTorch produces natively from `forward_pass(net, num_steps, data)`. Conv-layer outputs (`[T, B, C, H, W]`) require an explicit `.view(T, B, -1)` flatten before plotting. No magic shape detection.

3. **Return the figure, don't show it.** Every function returns `(fig, ax)`. The caller decides whether to `plt.show()`, save, embed in a notebook, or pipe into the Phase 2 dashboard later. This is the same separation-of-concerns logic from `forward_pass()` — the network knows about one step, the time loop lives outside. Here: the plotting function knows about one figure, the display loop lives outside.

### SCADA bridge — viz toolkit as HMI faceplate library #bridge/scada

This is exactly the same design problem as building **faceplates** for a SCADA HMI. You define a small set of standardized display widgets (analog gauge, trend chart, alarm summary, bar graph), give each one a strict signal contract (input tag types, units, scaling), and instantiate them throughout the screens. The faceplates don't know what they're displaying — they just render whatever tag is wired in.

The viz toolkit is a faceplate library for neural data:
- **Signal contract = `[T, B, N]` tensor**
- **Faceplate = one plotting function** (raster, PSTH, tuning curve, etc.)
- **HMI screen = a composition of faceplates** (the Phase 2 dashboard later)

Lock the contract now, and the Phase 2 live-dashboard work becomes "swap the data source from a saved tensor to a streaming source." The faceplates don't change.

---

## Section 1 — snnTorch Built-Ins Inventory

What `snntorch.spikeplot` ships with (verified from the 0.9.4 docs and source):

| Function | What it does | Input shape | Notes |
|---|---|---|---|
| `splt.raster(data, ax, **kwargs)` | Spike raster via `plt.scatter` | `[T, N]` (single sample) | Pass an `ax`, get a scatter plot. Used in Week 6 Session 1. |
| `splt.traces(mem, spk=None)` | Membrane potential traces, optional spike overlay | `[T, N]` mem, optional `[T, N]` spk | Used in Tutorial 3 and Week 6 Session 1. Good as-is. |
| `splt.spike_count(data, fig, ax, labels, animate=False, ...)` | Horizontal bar plot of summed spikes per neuron, optional animation | `[T, N_out]` | Used in Week 6 Session 1 §4.2. Output-layer focused. |
| `splt.animator(data, fig, ax)` | Animate 2D data over time (e.g., MNIST input spikes) | `[T, H, W]` | Returns a `FuncAnimation`. Requires ffmpeg for video export. |

**Verdict per plot type:**
- **Raster** — wrap `splt.raster`. Add multi-layer / multi-region overlay capability.
- **Membrane traces** — wrap `splt.traces`. Add y-axis labeling and threshold-line annotation.
- **Output spike counts** — wrap `splt.spike_count`. Adjust labels for non-MNIST tasks.
- **2D input animator** — wrap `splt.animator` for sensory-input visualization (will matter when Phase 2 has real spatially-structured input).

What snnTorch does **not** provide and the toolkit must build from scratch:
- PSTH (peri-stimulus time histogram)
- ISI (inter-spike interval) histogram
- Tuning curve (firing rate vs. input feature)
- Population activity over time (mean firing rate of an entire layer)
- Weight matrix heatmap
- Hopfield-style overlap variables $m^\mu(t)$ (load-bearing for Phase 2 — see [[week-06-optimize-visualize#Session 2|Week 6 Session 2]])

---

## Section 2 — Canonical Tensor Contract

**Every toolkit function consumes one of these shapes. No exceptions.**

| Symbol | Shape | Source | Meaning |
|---|---|---|---|
| `spk` | `[T, B, N]` | `forward_pass()` output, stacked | Spike trains: 0/1 per neuron per time step |
| `mem` | `[T, B, N]` | `forward_pass()` output, stacked | Membrane potential per neuron per time step |
| `weights` | `[N_post, N_pre]` | `layer.weight.data` | Synaptic weight matrix |
| `patterns` | `[M, N]` | Stored memory patterns (Phase 2+) | $M$ stored patterns of $N$ neurons (Hopfield) |

**Notation:** $T$ = time steps, $B$ = batch size, $N$ = number of neurons (in a layer), $M$ = number of stored patterns, $N_{\rm pre}$ / $N_{\rm post}$ = pre- and postsynaptic neuron counts.

**Conv-layer outputs** (`[T, B, C, H, W]`) require explicit flattening: `spk.view(T, B, -1)` before passing to toolkit functions. Reason: the toolkit doesn't know whether the user wants per-channel rasters, per-spatial-location rasters, or fully-flattened. Make them choose explicitly.

**Single-sample plots** (raster, traces) index the batch dim themselves: `spk[:, sample_idx, :]` → `[T, N]`. Toolkit functions accept either `[T, B, N]` and a `sample_idx` argument, or pre-indexed `[T, N]`. Default behavior: if a 3D tensor is passed, plot `sample_idx=0`.

---

## Section 3 — Plot Types In Scope (Stage 1)

Selected scope: **maximum beneficial viz**. Below are all eleven plot types and what each is *for*. Saturday MVP slice is in Section 5.

### 3.1 Spike Raster — `plot_raster(spk, ax=None, sample_idx=0, **kwargs) → (fig, ax)`

Each row = one neuron. Each dot = one spike. X-axis = time step. Y-axis = neuron index.

**Use:** the bread-and-butter SNN plot. First thing to look at after a forward pass. Tells you whether neurons are spiking at all, how synchronous activity is, and which neurons are "dead."

**Implementation:** wrap `splt.raster`. If `ax=None`, create one. Index `spk[:, sample_idx, :]` if 3D.

### 3.2 Membrane Trace — `plot_membrane(mem, spk=None, ax=None, sample_idx=0, threshold=1.0) → (fig, ax)`

Line plot of $u_i(t)$ for each neuron in a layer. Optional spike-marker overlay. Optional dashed horizontal line at threshold.

**Use:** debug the dead-neuron problem (Week 5). If membranes never approach threshold, weights are too small. If they saturate far above, weights too large or reset broken.

**Implementation:** wrap `splt.traces`. Add threshold line as an `ax.axhline()`.

### 3.3 PSTH — `plot_psth(spk, bin_size=5, ax=None) → (fig, ax)`

**Peri-stimulus time histogram.** Pool spikes across the batch (or across repeated trials) and across neurons into time bins. Y-axis = mean firing rate (spikes/bin). X-axis = time step.

**Use:** detect time-locked responses to stimuli. Standard in neuroscience papers. Shows when the *population* fires, abstracting away which individual neurons.

**Implementation:**
```python
binned = spk.sum(dim=(1, 2)).reshape(num_bins, bin_size).sum(dim=1)
ax.bar(bin_centers, binned)
```

### 3.4 Population Activity — `plot_population_rate(spk, smoothing_window=None, ax=None) → (fig, ax)`

Mean firing rate of the population over time: `spk.mean(dim=(1, 2))` → line plot of length $T$.

**Use:** at-a-glance "how active is this layer right now." For Phase 2 hippocampus debugging, this is the **load-bearing diagnostic** — [[week-06-optimize-visualize#Section 17.3|Week 6 Session 2 §17.3]] flagged this explicitly: if total activity saturates, your inhibitory circuit is broken. #insight

**Implementation:** simple mean over batch and neuron axes. Optional Gaussian or moving-average smoothing.

**SCADA bridge:** This is a **process trend chart** for the layer. One scalar over time, smoothed, with optional setpoint overlay (target activity level). Identical to plotting `flow_rate` over a shift. #bridge/scada

### 3.5 ISI Histogram — `plot_isi(spk, ax=None) → (fig, ax)`

Inter-spike interval distribution. Compute time-between-spikes for every (neuron, batch) pair, pool, histogram.

**Use:** diagnose firing regularity. Exponential ISI distribution = Poisson-like irregular firing (cortical-realistic). Sharp peak at one value = pathological synchronization. Bimodal = bursting.

**Implementation:**
```python
spike_times = spk.nonzero()  # returns indices where spikes occurred
isis = compute_intervals_per_neuron(spike_times)  # diff of consecutive times per neuron
ax.hist(isis, bins=50)
```

### 3.6 Tuning Curve — `plot_tuning_curve(spike_counts_per_stimulus, stimulus_values, ax=None) → (fig, ax)`

Mean firing rate vs. input feature value. Inputs: an array of stimulus values (e.g., orientation angles, input intensities, cube state encodings) and corresponding spike counts per neuron.

**Use:** show what each neuron is "tuned to." Standard neuroscience plot. Will matter in Phase 2 when the sensory cortex region needs to demonstrate feature selectivity.

**Implementation:** for each neuron, line plot of `spike_count vs stimulus_value`. Multi-panel grid for multiple neurons.

### 3.7 Weight Matrix Heatmap — `plot_weights(W, ax=None, cmap='RdBu_r', center=0) → (fig, ax)`

Synaptic weight matrix as a 2D heatmap. Red = positive (excitatory), blue = negative (inhibitory). Center colormap at zero so sign is visually obvious.

**Use:** before/after training comparison; spot dead synapses (rows of zeros); see whether Hebbian patterns have formed (Phase 2 — should see block structure once memories are stored).

**Implementation:** `ax.imshow(W, cmap=cmap, vmin=-|W|.max(), vmax=|W|.max())`.

### 3.8 Output Spike Count Bar — `plot_output_counts(spk, labels=None, ax=None, sample_idx=0) → (fig, ax)`

Static version of `splt.spike_count`. Total spikes per output neuron, one bar per class.

**Use:** for classification tasks, this is your "prediction certainty" plot. One tall bar = confident prediction. Two close bars = network unsure between two classes.

**Implementation:** wrap `splt.spike_count` with `animate=False`. Add class labels.

### 3.9 Confusion Matrix — `plot_confusion(predictions, targets, ax=None, normalize=True) → (fig, ax)`

Standard classifier diagnostic: predicted vs. true class as a heatmap.

**Use:** post-evaluation breakdown of where the network errs. Not SNN-specific but indispensable.

**Implementation:** wrap `sklearn.metrics.confusion_matrix`. Annotate cells with counts.

### 3.10 Multi-Layer Raster Grid — `plot_layer_rasters(spk_dict, sample_idx=0) → fig`

Composite plot: one raster per layer, arranged vertically with shared time axis. `spk_dict` is `{layer_name: spk_tensor}`.

**Use:** see the spike pattern propagating through the network from input to output. For Phase 2+ multi-region work, this generalizes to "one raster per region."

**Implementation:** `plt.subplots(num_layers, 1, sharex=True)`, call `plot_raster` on each.

### 3.11 Hopfield Overlap — `plot_overlaps(spk, patterns, ax=None) → (fig, ax)`

For Phase 2 onward. Compute $m^\mu(t) = \frac{1}{N}\sum_i p_i^\mu S_i(t)$ for each stored pattern $\mu$, plot all $M$ overlaps as line traces over time.

**Use:** **the** diagnostic for the Hippocampus region. If memory retrieval works, exactly one $m^\mu$ should go to ~1 and stay there after the cue is removed. Multiple high overlaps = spurious mixture state. None high = retrieval failed. Lifted directly from [[week-06-optimize-visualize#Section 17.3|Week 6 Session 2's Ch. 17 design requirements]].

**Implementation:** matrix multiply `spk.mean(dim=1) @ patterns.T / N` → `[T, M]` → line plot.

#insight

---

## Section 4 — File Structure

Working location for the toolkit (Saturday's build):

```
src/neuromorphic/viz/
├── __init__.py        # exposes the public API
├── core.py            # tensor shape utilities, sample indexing, fig/ax handling
├── spikes.py          # plot_raster, plot_psth, plot_population_rate, plot_isi
├── membrane.py        # plot_membrane
├── tuning.py          # plot_tuning_curve, plot_output_counts, plot_confusion
├── weights.py         # plot_weights
├── memory.py          # plot_overlaps (Phase 2 prep)
└── composite.py       # plot_layer_rasters (multi-panel)
```

**Public API (`__init__.py`):** every plot function is importable as `from neuromorphic.viz import plot_raster, plot_psth, ...`. Internal organization is by data type, not by user concern.

**Test harness:** `tests/viz/` with one fixture file producing a small synthetic `[T, B, N]` tensor (random spikes, random membranes). Every plot function gets one smoke test that asserts it returns a `(Figure, Axes)` tuple without raising.

---

## Section 5 — Saturday Build Order

Strict order. Stop when out of time. The earlier items have the most reuse value.

**MVP slice (must finish Saturday):**
1. `core.py` — shape utilities + `_ensure_ax(ax)` helper that creates a fig/ax if `ax=None`
2. `plot_raster` (3.1) — wrap `splt.raster`, return `(fig, ax)`. Smoke test.
3. `plot_membrane` (3.2) — wrap `splt.traces` + threshold line. Smoke test.
4. `plot_population_rate` (3.4) — most useful Phase 2 diagnostic. Smoke test.
5. `plot_psth` (3.3) — neuroscience-standard. Smoke test.

**Stretch (if MVP done with time left):**

6. `plot_output_counts` (3.8)
7. `plot_weights` (3.7)
8. `plot_isi` (3.5)
9. `plot_layer_rasters` (3.10) — composite, depends on #2 working
10. `plot_confusion` (3.9)

**Deferred to Phase 2:**

11. `plot_tuning_curve` (3.6) — no stimulus-feature variable to plot against yet
12. `plot_overlaps` (3.11) — no stored patterns until hippocampus region exists

**Rationale for MVP order:** items 1–5 cover the *most common* questions you'll have about any SNN: "are they spiking?" (raster), "what are membranes doing?" (membrane), "how active is the layer overall?" (population rate), "when do they fire relative to the stimulus?" (PSTH). With those four plots, you can debug almost anything in a single-region SNN. The rest are nice-to-haves.

---

## Section 6 — Open Design Questions

These are intentionally deferred — the answer becomes obvious once you start writing code.

1. **Colormap defaults.** Stick with matplotlib defaults or pick a colorblind-safe palette (e.g., `viridis`, `cividis`) from the start? Probably the latter — viz is the *point*.
2. **Saving convention.** Add an optional `save_path` arg to every function, or rely on the caller to do `fig.savefig()`? Caller-saves keeps the API smaller — likely the right call.
3. **Time-axis units.** Always plot in "time steps" (integer)? Or accept an optional `dt` to convert to ms? For Phase 1, time steps. For Phase 2 (real biological time constants), `dt` becomes useful.
4. **Animation vs. static.** Each function gets a sibling `animate_*` version, or a single `animate=True` flag? snnTorch went with the flag. Match them.
5. **Style sheet.** Define a `viz/style.mplstyle` and apply it on import? Keeps plot aesthetics consistent without per-function arg bloat.

Decide quickly during the build — none of these block the MVP.

---

## Section 7 — Deferred Items

- **r/neuroscience and r/MachineLearning browse** — calendar suggested it as inspiration, not blocking. 10 minutes during coffee tomorrow.
- **Animation work** — `splt.animator` and `splt.spike_count(animate=True)` already exist. Defer custom animation until Phase 2 dashboard work demands it.
- **Live / streaming dashboard** — explicitly Phase 2–3 work. Static matplotlib is fine for Stage 1.

---

## Review Notes

*[Space for Mike's design-doc review:*

*- Does the canonical `[T, B, N]` contract feel right, or are there cases where it forces awkward indexing?*
*- Should `plot_overlaps` be a Phase 1 build (forward-looking) or strictly Phase 2 (when patterns exist)?*
*- Is the SCADA "faceplate library" analogy load-bearing or decoration? If it doesn't help write the code Saturday, drop it.*
*- Saturday MVP slice realistic for one session (~2 hr)?]*

---

## Revision History

| Date | Change | Notes |
|---|---|---|
| 2026-05-15 | Initial draft | Created during Week 6 Session 3. Extracted from week-06 weekly notes into standalone design doc. |
| 2026-05-15 | Added Agent Hand-Off Context section | Top-of-doc context block for Saturday's implementation agent — repo conventions, tensor-source verification, build behavior expectations, and explicit non-goals. |

---

## Inline Tag Index

#concept/visualization-toolkit · #concept/canonical-tensor-shape · #concept/spike-raster · #concept/psth · #concept/population-activity · #concept/isi-histogram · #concept/tuning-curve · #concept/weight-heatmap · #concept/hopfield-overlap · #bridge/scada · #insight
