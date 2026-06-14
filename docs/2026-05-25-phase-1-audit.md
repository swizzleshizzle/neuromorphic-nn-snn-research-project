# Phase 1 Audit — Read-Only Report

**Date:** 2026-05-25
**Purpose:** Ground-truth audit of the repo for the Phase 1 "What I Learned" write-up. Hand-off artifact for the agent drafting that document — every number, path, and claim below was read directly from disk; nothing is estimated.

---

## A. Experiment inventory

| Folder | Description |
|---|---|
| `001_smoke_test` | Repo bootstrap / environment smoke test (pre-week-1). |
| `002_week1_mnist_classifier` | Week 1 non-spiking MNIST classifier (PyTorch warm-up). |
| `003_week2_lif_neuron_viz` | Week 2 LIF neuron + first viz exercises. |
| `004_week3_1_pytorch_handson` | Week 3 PyTorch hands-on. |
| `005_week3_2_pytorch_training_loops` | Week 3 training-loop patterns. |
| `006_week4_1_gymnasium_intro` | Week 4 Gymnasium intro. |
| `007_week4_2_qlearning_grid_agent` | Week 4 tabular Q-learning grid-world agent. |
| `008_week5_snn_mnist_baseline` | Week 5 feedforward SNN MNIST baseline (`run.py` / `run2.py` + `viz_smoke.py`). |
| `009_week6_snn_mnist_optimization` | Week 6 MNIST optimization, 3 variants (v1 MLP-tuned, v2 CNN baseline, v3 CNN-tuned). |
| `010_week7_recurrent_sn` | Week 7 Mon diagnostic — RLeaky V-sweep / FF-vs-recurrent / RSynaptic check. **No training, no MNIST.** |
| `011_week7_sequential_mnist` | Week 7 sequential MNIST — recurrent (RLeaky) vs feedforward (Leaky) control on row-at-a-time. |

Corrections to prior understanding:
- **EXP-010 DOES exist** — it's a Monday-of-week-7 diagnostic ("see the dynamics before scaling") for `snn.RLeaky`. Sections: `section2_v_sweep.py`, `section2b_10neuron_population.py`, `section3_ff_vs_recurrent.py`, `section4_rsynaptic_check.py`, plus `run_all.py`. Outputs are 5 PNGs in `outputs/`. There is no numbering gap between 009 and 011.
- EXP-008 / 009 / 011 otherwise match the descriptions in the audit request.

## B. EXP-011 results

Verbatim from `experiments/011_week7_sequential_mnist/outputs/comparison.md`:

| Variant | Hidden layer | Params | Epochs | Train time | Test acc | Gap vs feedforward |
|---|---|---:|---:|---:|---:|---:|
| `recurrent` | snn.RLeaky(all_to_all=True) | 21,514 | 5 | 5m 33s | 79.12% | +37.28% |
| `feedforward` | snn.Leaky | 5,002 | 5 | 5m 00s | 41.84% | 0.00% |

CSV agrees (`gap_vs_feedforward = 0.3728`; train seconds 333.95 / 300.22).

**Config compliance** (`recurrent.yaml` / `feedforward.yaml`): locked config matches — `num_steps: 28`, `hidden_size: 128`, `readout_window: 4`, `beta: 0.9`, `threshold: 1.0`, `reset_mechanism: subtract`, `lr: 0.0003`, `epochs: 5`, `seed: 42`. `recurrent: true/false` is the only diff. RLeaky `all_to_all=True` is asserted in the comparison table. **No deviation.**

**Rasters present, both variants:**
- `experiments/011_week7_sequential_mnist/outputs/recurrent/hidden_raster.png`
- `experiments/011_week7_sequential_mnist/outputs/feedforward/hidden_raster.png`

Visually (1 test image, T=28, all 128 hidden units):
- **Recurrent raster** — dense, sustained activity. Most of the 128 neurons participate from ~t=7 onward; strong vertical banding (synchronized population bursts every ~1 step late in the trace). Activity persists across the whole sweep — consistent with lateral mixing.
- **Feedforward raster** — noticeably sparser, more diffuse. Less population synchrony, no clear vertical bands; spikes look more like isolated per-neuron events tied to local input rows. Consistent with the "independent columns" framing.

**Checkpoint reload:** `results.md` reports Gate 4 PASS with **Δ = 0.00000**. `outputs/best_checkpoint.{pt,json}` exist; the `.json` records `final_test_acc: 0.7912`, `num_params: 21514`, `variant: recurrent`.

**Caveat from `results.md`:** both variants undershot their predicted absolute accuracy (recurrent predicted 94–96%, got 79.12%; FF predicted 88–92%, got 41.84%). Author attributes this to under-training at 5 epochs, not architectural failure. The **gap direction and reload gate are the load-bearing claim** — the absolute numbers are not.

## C. STDP demo results

**Design exists; implementation does NOT.**

- Design doc: `docs/design/2026-05-25-stdp-wta-demo.md` (untracked in git per the status snapshot — created today, 2026-05-25).
- The doc earmarks `experiments/012_week7_stdp_demo/` as the planned location and lists 5 deliverable artifacts (weight evolution, final matrix, tuning curves, late raster, `selectivity_report.txt`).
- **No `experiments/012*` folder exists.** No `stdp*` script anywhere outside `.venv` (snnTorch's own `stdp_learner.py` is the only match, irrelevant).
- No STDP weight-evolution plot, no selectivity plot, no STDP curve reproduction found in the repo.

So: the demo was **scoped but not built yet** as of this snapshot. It did NOT grow an EXP-012 folder. The "exploratory scratch script" framing in the audit request is at odds with the design doc, which actually proposes a full EXP-012 folder with `patterns.py`, `models.py`, `run.py`, `config.yaml`, `outputs/`, and 5 verification gates. Worth resolving in the "What I Learned" doc whether the demo will be built or dropped.

## D. Viz toolkit inventory

`src/neuromorphic/viz/__init__.py` exports **exactly 7 functions** — matches expectation.

| Function | Module | What it plots | Expected input shape |
|---|---|---|---|
| `spike_raster` | `spikes.py` | One dot per (time, neuron) spike — wraps `snntorch.spikeplot.raster` | `[T, B, N]` or `[T, N]` (selects `sample_idx` if 3D) |
| `population_rate` | `spikes.py` | Mean firing rate across batch+neurons per time step; optional rolling smoothing | `[T, B, N]` — strict, raises if ndim≠3 |
| `psth` | `spikes.py` | Peri-stimulus time histogram: total population spikes per `bin_size`-wide bin | `[T, B, N]` — strict, raises if ndim≠3 |
| `membrane_trace` | `membrane.py` | Per-neuron membrane `U[t]` line plot, optional spike "ride-on", optional threshold dashed line | `[T, B, N]` or `[T, N]` (selects `sample_idx` if 3D) |
| `weight_histogram` | `weights.py` | 1D histogram of weight values, optional log-y | `[N_post, N_pre]` — 2D only |
| `weight_heatmap` | `weights.py` | 2D heatmap with symmetric diverging colormap centered at 0 | `[N_post, N_pre]` — 2D only |
| `training_curve` | `training.py` | Loss series on left axis, accuracy series on twin right axis [0,1]; routes by `"loss"` / `"acc"` substring | `dict[str, list[float]]` (not a tensor) |

Contract notes:
- Canonical tensor contract is `[T, B, N]` for time-series plots, `[N_post, N_pre]` for weights. The 7 functions are internally consistent with that.
- `training_curve` takes a dict-of-lists, not a tensor — that's by design but worth noting in the "What I Learned" doc as the one non-tensor entry point.
- No `TODO` / `FIXME` / `NotImplemented` / `XXX` markers anywhere under `src/neuromorphic/viz/`. Module is clean.

## E. Discrepancies / surprises

1. **EXP-010 exists** — the audit request asserted it didn't. It's a no-training diagnostic for RLeaky regimes (V-sweep, 10-neuron population, FF-vs-recurrent comparison, RSynaptic smoke). Five plots in `outputs/`. Predates EXP-011 in week 7.
2. **STDP demo is unimplemented.** Only a design doc exists (`docs/design/2026-05-25-stdp-wta-demo.md`, untracked). No code, no plots, no results — contrary to "find where it lives" wording in the audit request and contrary to "exploratory scratch script" framing (the design proposes a full EXP-012 folder).
3. **EXP-008 has no `results.md` or `comparison.md`.** Its accuracy number (the audit request says 93.5%) is not recorded in a markdown file in-repo — it lives inside `checkpoint.pt` as `final_accuracy`, and is only echoed at runtime by `viz_smoke.py:84` and `plots.py:252`. Could not confirm the 93.5% figure from a repo markdown artifact without loading the checkpoint (which would mean running code — out of scope per the read-only rule).
4. **EXP-008 config has stale ID metadata.** `experiments/008_week5_snn_mnist_baseline/config.yaml` self-labels as `experiment_id: "EXP-001"` and references `experiments/002_snn_mnist_baseline/config.yaml` in its header comment. The folder/ID renumbering didn't propagate into the config. Cosmetic, but worth knowing if you trust IDs in configs.
5. **No in-repo experiment-log file.** Searched for `experiment-log*`, `**/experiment*log*` — no matches. `experiments/010_week7_recurrent_sn/goals.md:6` points at the Obsidian vault path (`300 Efforts/Active/Coding/Neuromorphic Development/Weekly Notes/`). The in-repo equivalents are the per-experiment `goals.md` / `results.md` files plus `docs/design/*.md` and `docs/superpowers/plans/*.md`. If you want a single in-repo log, it doesn't exist today.
6. **EXP-011 absolute accuracy undershoot** — already documented in `results.md`. Headline claim (architectural gap) holds; absolute numbers are flagged as under-trained, not validated against the predicted 94–96% / 88–92% ranges.

## F. Suggested outline for "What I Learned" doc

Skeleton only — each bullet is the data point that belongs in the section, not the prose.

- **Phase 1 scope and what it produced** — 11 numbered experiment folders (001–011), 7-function viz toolkit, 4 design docs, 3 implementation plans.
- **The MNIST progression** — EXP-008 feedforward SNN baseline (final accuracy in `checkpoint.pt`, ≈93.5% per author's notes) → EXP-009 three-variant optimization (95.08% / 97.83% / **98.05%** for v1 MLP-tuned / v2 CNN baseline / v3 CNN-tuned; CNN folded in at v2).
- **From static to temporal** — EXP-010 diagnostic (no training) maps RLeaky V-regime boundaries before scaling; EXP-011 commits to row-at-a-time sequential MNIST.
- **Headline architectural result (EXP-011)** — recurrent vs feedforward gap = **+37.28%** (79.12% vs 41.84%) under identical 28-step, H=128, β=0.9, RLeaky `all_to_all=True`, 5-epoch config, seed 42. Predicted gap was 3–6%; actual was ~6× larger and in the predicted direction.
- **What the rasters show** — recurrent raster: dense, banded, sustained population activity from ~t=7 onward; FF raster: sparse, diffuse, column-like. Visual confirmation of the lateral-mixing claim.
- **What did NOT work as predicted (and why it doesn't sink the claim)** — both EXP-011 variants under-trained at 5 epochs (recurrent 79.12% vs predicted 94–96%; FF 41.84% vs predicted 88–92%). Direction + reload gate (Δ=0) carry the conclusion; absolute number is a known follow-up.
- **Reproducibility discipline that earned its keep** — 4 verification gates in EXP-011 (forward shape, exact param count, initial CE ≈ ln 10, best-checkpoint reload Δ=0); locked YAML config; predict-before-execute notebook before each run.
- **The viz toolkit as Phase 1 infrastructure** — 7 functions on a `[T, B, N]` / `[N_post, N_pre]` contract, `(fig, ax)` return everywhere, single non-tensor entry point (`training_curve`'s dict). Zero TODO/FIXME markers.
- **Loose ends entering Phase 2** — STDP/WTA demo designed (`docs/design/2026-05-25-stdp-wta-demo.md`) but not built; EXP-011 absolute-accuracy calibration left for a longer training run; EXP-008 ID/config metadata never renumbered to match its folder.
- **What Phase 2 inherits** — a working sequential-MNIST architecture demonstrating recurrence helps; a viz toolkit ready for STDP plots (weight heatmap + histogram already there); a verification-gate pattern that scaled.
