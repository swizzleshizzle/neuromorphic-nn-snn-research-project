# STDP Competitive-WTA Demo — Design

**Date:** 2026-05-25
**Experiment ID:** EXP-012 → `experiments/012_week7_stdp_demo/`
**Companion to:** EXP-011 sequential MNIST (week 7 hands-on, Step 1.4)
**Time budget:** ~60 min build, including viz.

---

## 1. One-sentence goal

Build a small unsupervised SNN where competitive STDP causes output neurons to
develop selectivity for distinct input patterns — and produce visualizations
that make the emergent selectivity legible at a glance.

## 2. What this experiment is NOT

- Not a benchmark. No accuracy target, no comparison run, no test set.
- Not a generic STDP library. Single-file, demo-only, hardcoded to this setup.
- Not using snnTorch's autograd path. STDP is a local plasticity rule
  applied with `torch.no_grad()` — no backprop, no loss function.
- Not reusing `ExperimentTracker` or YAML config loader. Standalone script
  with a small hardcoded config block. Rationale: this is a one-off demo,
  not part of the tracked experiment grid.

## 3. Network architecture (locked)

- **12 input neurons** (Poisson spike sources)
- **4 output LIF neurons** (`snn.Leaky`, β=0.9, threshold=1.0,
  reset_mechanism="subtract")
- **Fully-connected input→output synapses** — weight matrix `W` of shape
  `[12, 4]`, initialized uniform random in `[0, w_max]`
- **Hard WTA**: at each timestep, after computing membrane potentials,
  identify the output neurons that crossed threshold; if multiple, keep
  only the one with the highest membrane potential as the "winner";
  zero the others' spike output AND reset their membrane to 0 for that
  step. (Simpler than lateral inhibition; no extra tunable parameters.)

## 4. Input patterns

Three fixed spatial patterns over 12 input neurons:

| Pattern | Active input indices | Rate (active) | Rate (inactive) |
|---|---|---:|---:|
| A | {0, 1, 2, 3} | 100 Hz | 5 Hz |
| B | {4, 5, 6, 7} | 100 Hz | 5 Hz |
| C | {8, 9, 10, 11} | 100 Hz | 5 Hz |

Each **trial**:
1. Sample one pattern uniformly at random (A, B, or C)
2. Present it for **20 timesteps** at dt=2.5 ms (so 50 ms pattern window)
3. Inter-trial gap of **8 timesteps** at background-only rate

Train for **1000 trials** total (~28,000 timesteps).

Inputs are sampled per-timestep as Bernoulli draws: `p_spike = rate * dt`.
At 100 Hz × 2.5 ms = 0.25 probability per active step.

## 5. STDP rule (locked)

Pair-based STDP with exponential eligibility traces, updated every timestep
within `torch.no_grad()`:

**State:**
- `x_pre[i]` — pre-synaptic trace, one per input neuron, shape `[12]`
- `y_post[j]` — post-synaptic trace, one per output neuron, shape `[4]`

**Per-timestep update (after computing spikes):**
1. Decay traces: `x_pre *= exp(-dt / tau_pre)`, `y_post *= exp(-dt / tau_post)`
2. Where pre neuron i spiked: `x_pre[i] += 1`, AND for all j: `W[i,j] += A_minus * y_post[j]` (LTD: pre after post)
3. Where post neuron j spiked: `y_post[j] += 1`, AND for all i: `W[i,j] += A_plus * x_pre[i]` (LTP: pre before post)
4. Clip: `W = W.clamp(0, w_max)`

**Hyperparameters (locked):**
| Param | Value | Notes |
|---|---|---|
| `tau_pre` | 20 ms | classic Bi-Poo timescale |
| `tau_post` | 20 ms | symmetric |
| `A_plus` | 0.005 | LTP step |
| `A_minus` | -0.0055 | slight LTD bias to suppress runaway |
| `w_max` | 1.0 | weight clip ceiling |
| `w_init` range | [0, 1.0] uniform | every weight starts random |

## 6. File layout

```
experiments/012_week7_stdp_demo/
  __init__.py                 # empty marker
  patterns.py                 # PoissonPatternGenerator class
  models.py                   # STDPLayer + run_step helpers
  run.py                      # main loop + visualizations + selectivity summary
  config.yaml                 # numeric knobs (read in run.py, no ExperimentConfig)
```

Outputs (gitignored under `outputs/` per existing rule):

```
outputs/
  weight_matrix_evolution.png    # 4-snapshot grid
  weight_matrix_final.png        # labeled final matrix
  tuning_curves.png              # per-output mean rate by pattern
  spike_raster_late.png          # ~200ms window late in training
  selectivity_report.txt         # which output prefers which pattern
```

## 7. Visualizations (deliverables)

1. **Weight matrix evolution** — 2×2 grid of heatmaps at trial 0 / 250 / 500 / 1000.
   X-axis: 4 output neurons. Y-axis: 12 inputs. Colormap: viridis,
   shared scale `[0, w_max]`. Group separators between input blocks
   {0–3, 4–7, 8–11} so the pattern structure is visible.

2. **Final weight matrix** — single heatmap, with row labels showing pattern
   group ("A"/"B"/"C") and column labels showing each output's preferred
   pattern (computed by argmax over per-pattern response rates).

3. **Tuning curves** — bar chart, 4 panels (one per output neuron). Each
   panel: bars for mean firing rate when pattern A, B, C is shown
   (post-training, no STDP updates, average over 50 trials per pattern).

4. **Spike raster late in training** — single 200 ms window (80 timesteps)
   sampled from the last 50 trials. Two stacked panels: input spikes
   (12 rows) on top, output spikes (4 rows) below. Vertical lines mark
   pattern onsets and offsets.

5. **`selectivity_report.txt`** — plain text summary:
   ```
   Output 0: prefers pattern X (rate Y Hz vs other patterns Z Hz)
   ...
   Patterns covered: {A, B, C}  |  Spare/unselective outputs: 0–1
   ```

## 8. Verification gates (inline, no pytest)

| # | Gate | When | Pass criterion |
|---|---|---|---|
| 1 | Init sanity | After construction | `W.shape == (12, 4)`; values in `[0, 1]` |
| 2 | Poisson rate check | After generating 1000 ms of pattern A | active inputs fire 80–120 spikes; background inputs fire 0–15 spikes |
| 3 | Pre-training random selectivity | Before training | Every output responds within 30% of mean rate to all patterns (no built-in bias) |
| 4 | Post-training selectivity | After training | At least 2 of 4 outputs have a clear preferred pattern (preferred-pattern rate ≥ 2× any other pattern) |
| 5 | Weight statistics | After training | Weight matrix is no longer uniform: std(W) post-training > 2× std(W) at init |

If gate 4 fails, the demo's headline claim is unsupported. Triage in order:
LR (A_plus too low), threshold (firing rate too low), trial count (too few),
WTA selectivity (if all outputs always lose to one, lower its initial weights).

## 9. Predict-before-execute

- Outputs will partition into 3 selective + 1 "spare" (might be silent or
  echo another output) OR 4 selective with two outputs sharing a pattern.
- Weight matrix block structure will appear by trial 500 (visible bands).
- Selectivity ratio (preferred-pattern rate / max-other-pattern rate)
  ≥ 3× for the selective outputs.
- Final weight std ≥ 3× initial std (most weights pushed to 0 or w_max).

## 10. Out of scope

- No homeostatic threshold adjustment
- No weight normalization (just clip)
- No triplet STDP, no voltage-dependent STDP, no STP
- No metaplasticity
- No comparison to a non-STDP control (the demo IS the result)
- No CUDA optimization — small network, CPU is fine

## 11. Done criteria

- [ ] All 5 visualization artifacts produced
- [ ] At least 2 of 4 outputs show clear pattern selectivity
- [ ] `selectivity_report.txt` written and human-readable
- [ ] Source committed in 1–2 scoped commits
- [ ] `outputs/` gitignored (existing rule)
- [ ] Brief `results.md` in experiment folder noting actual selectivity
