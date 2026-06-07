# Five-Region Architecture Specification — v1

**Status:** Draft · v1 (starting point — expected to evolve)
**Date:** 2026-06-05 · Week 9, Session 3
**Phase:** 2 — Multi-region brain · **Step 2.1** — architecture specification
**Author session:** drafted with Claude (Cowork)
**Parent notes:**
- Obsidian — `300 Efforts/Active/Coding/Neuromorphic Development/Weekly Notes/week-09-brain-architectures` (Sessions 1–3)
- L9 *Architectural Decisions Catalogue* (Spaun + NEF mapping)
- L10 *Communication Protocol Spec* (driver/modulator buses, gated channels)
**Tracking plan:** `docs/superpowers/plans/2026-06-05-phase2-architecture-spec-plan.md`

---

## 0. Purpose and scope

This document is the first concrete, buildable description of the five-region spiking
architecture. It turns the L9/L10 reading notes into per-region numbers and wiring. It is
deliberately *small* — 50–200 neurons per region — so the whole brain can be stood up,
debugged, and visualized before any scaling. Absolute counts and codings here are starting
points chosen to be defensible, not final.

The downstream task (Phase 2) is a **grid-world** navigation problem; Phase 3 swaps in the
**2×2 Rubik's Cube**. The action set referenced below assumes the grid-world's 4 discrete
moves (`up, down, left, right`); the spec is written so only `N_actions` and the sensory
input dimension change when the task changes.

### Conventions (locked, repo-wide)

- **Tensor contract:** every spike signal is `[T, B, N]` — time, batch, neuron — matching the
  viz toolkit's canonical contract. Weight matrices are `[N_post, N_pre]`.
- **Spikes are binary** `{0, 1}` events. "Vectors" passed between regions are *population /
  rate codes* over a neuron group, not decoded floats on a wire — decoding (if any) happens
  inside the consuming region.
- **Inference window:** `T = 32` time steps per decision (follows the week-7 sequential-MNIST
  precedent of `T = 28`; rounded for headroom). One environment step = one `T`-window.
- **Neuron primitives** map to snnTorch: **Leaky** = `snn.Leaky` (LIF), **Synaptic** =
  `snn.Synaptic` (LIF + separate synaptic-current state, longer memory), **Recurrent** =
  `snn.RLeaky` (LIF with a recurrent weight matrix). The week-7 result (recurrent vs
  feedforward gap **+37.28%** on sequential MNIST) is the evidence base for using recurrence
  where sustained/temporal state matters.

---

## 1. Region overview

| # | Region | Neurons | Neuron type | Role | Weights |
|---|---|---:|---|---|---|
| 1 | Sensory Cortex | 200 | Leaky | Encode + compress observation → concept code | Learned (surrogate grad) |
| 2 | Hippocampus (Memory) | 150 | Recurrent (RLeaky) | Hold patterns over a delay (attractor) | Hand-designed dynamics, optional fine-tune |
| 3 | Prefrontal (Planning) | 150 | Recurrent + Leaky | Integrate sensory + memory → subgoal/utilities | Learned value/relation; recurrent state-hold |
| 4 | Motor Cortex (Action) | 100 | Leaky + lateral inhibition | Decompress + winner-take-all action | WTA hand-designed; decompression learned |
| 5 | Thalamic Router | 50 | Leaky | Select (BG-like WTA) + gate (thalamus-like relay) | Hand-derived, no training |
| | **Total** | **650** | | | |

Plus a global **neuromodulatory bus** (not a region): two broadcast scalars — **dopamine**
(reward / learning-enable) and **ACh** (gain / precision) — read by every region as global
parameters. No addressing, one-to-all.

---

## 2. Region specifications

### 2.1 Sensory Cortex

- **Neuron count:** 200 — a 2-stage feedforward compression stack: input projection ≈128 →
  compressed concept layer ≈64 (≈192, budgeted at 200).
- **Neuron type:** **Leaky** (`snn.Leaky`). Pure feedforward; no recurrence needed. Weights
  are **learned** with surrogate gradients — we don't know the right features a priori, so
  this region is trained, not hand-derived (L9 catalogue: "learned").
- **Internal connectivity:** hierarchical, dense feedforward between stages
  (`input → 128 → 64`). No lateral or recurrent connections in v1. This is Spaun's visual
  hierarchy (V1→…→IT) collapsed to two stages at our scale.
- **Input spike format:** environment observation encoded as spikes, `[T, B, N_obs]`,
  rate-coded (Poisson) or latency-coded. For grid-world, the observation (agent position,
  goal position, wall/occupancy) is flattened and spike-encoded. Enters through a **1st-order
  relay gate** (the only sensory input gate — see §3).
- **Output spike format:** `[T, B, 64]` — the compressed *concept code*. A population/rate
  code, not a decoded vector.
- **Connects FROM:** external environment (sensor input).
- **Connects TO:** Prefrontal (driver, content, ungated); optionally Hippocampus later.

### 2.2 Hippocampus (Memory)

- **Neuron count:** 150 — single recurrent population (attractor substrate).
- **Neuron type:** **Recurrent** (`snn.RLeaky`, `all_to_all=True`). Recurrence is the whole
  point: it sustains activity across the delay after input is removed. **Synaptic**
  (`snn.Synaptic`) is the fallback if RLeaky's decay is too fast to hold a trace for the
  required window — its separate synaptic-current state lengthens the memory time constant.
- **Internal connectivity:** recurrent attractor — recurrent excitation plus global
  inhibition to stabilize a **point or line attractor**. The held state is a stored
  pattern / list vector that persists with no input (sustained-activity regime observed in
  the week-7 recurrent raster: dense, banded, persistent).
- **Input spike format:** `[T, B, 64]` content code from Prefrontal (driver), **gated** — the
  *store* command opens this pathway.
- **Output spike format:** `[T, B, 150]` sustained spike pattern; the *recall* read-out is a
  ~64-D population code back to Prefrontal.
- **Connects FROM:** Prefrontal (store, gated driver).
- **Connects TO:** Prefrontal (recall, gated driver). Both pathways gated by the Thalamic
  Router.
- **OPEN CHOICE (carried from L9):** memory encoding scheme —
  *bind-and-superpose* (`content ⊗ position`) **vs** *recurrent sequence-chaining*. Pick one
  before implementation; v1 leaves both on the table.

### 2.3 Prefrontal Cortex (Planning)

- **Neuron count:** 150 — split: ~100 recurrent state-hold + ~50 feedforward transform.
- **Neuron type:** **Recurrent + Leaky.** The state-hold sub-population is `snn.RLeaky`
  (holds the current goal/context/task-state vector, NEF Principle 3 dynamics); the
  transformation sub-population is `snn.Leaky` with **learned** weights computing the
  value/relation function (L9: this region's value/relation map is learned, the state-hold is
  hand-derived dynamics).
- **Internal connectivity:** recurrent state-holding loop feeding a feedforward learned
  transform. The transform integrates the sensory concept and the hippocampal recall into a
  current subgoal and a vector of action utilities.
- **Input spike format:** sensory concept `[T, B, 64]` (driver, ungated) **+** hippocampal
  recall `[T, B, ~64]` (driver, gated). Also reads dopamine/ACh from the neuromod bus.
- **Output spike format:** `[T, B, 150]` internally; two decoded read-outs —
  (a) action-utility code `[T, B, N_actions]` to Motor and to the Router's selection stage,
  (b) a *store* content code `[T, B, 64]` to Hippocampus.
- **Connects FROM:** Sensory (driver), Hippocampus (recall, gated driver), neuromod bus.
- **Connects TO:** Motor (driver, gated), Hippocampus (store, gated), Thalamic Router
  (utilities → selection).

### 2.4 Motor Cortex (Action)

- **Neuron count:** 100 — a decompression stage (~64) feeding a small WTA output layer
  (`N_actions`, =4 for grid-world).
- **Neuron type:** **Leaky** (`snn.Leaky`) with **lateral inhibition** on the output layer.
- **Internal connectivity:** **winner-take-all via lateral inhibition** — every output neuron
  inhibits the others, so the strongest action suppresses its competitors and a single winner
  emerges (the inhibition motif is *hand-designed*, L9 catalogue). Upstream, a decompression
  stack maps the low-D action intent up to the motor command (decompression learned/derived).
- **Input spike format:** `[T, B, N_actions]` candidate utilities from Prefrontal (driver),
  **gated** by the router (action-enable).
- **Output spike format:** near one-hot spike code `[T, B, N_actions]` — the winning action,
  read out by spike count over the `T`-window and applied to the environment.
- **Connects FROM:** Prefrontal (driver, gated).
- **Connects TO:** external environment (action). ACh gain from the neuromod bus sharpens the
  WTA.

### 2.5 Thalamic Router

- **Neuron count:** 50 — two internal stages, kept small (no learning, pure control).
- **Neuron type:** **Leaky** (`snn.Leaky`), hand-derived throughout.
- **Internal connectivity (two stages):**
  - **Stage A — Selection (basal-ganglia-like):** WTA over the action/route utilities via
    lateral inhibition. Picks the winning option.
  - **Stage B — Gating (thalamus-like):** gate populations that **tonically inhibit** relay
    channels and *release* (disinhibit) only the channel Stage A selected. Modes: `off` /
    `tonic`, per L10.
- **Input spike format:** utility code from Prefrontal `[T, B, utility-dim]` (control-class
  signal, low bandwidth).
- **Output spike format:** **control lines** — per-pathway gate/gain signals (inhibitory),
  *not* content. These open/close the gated pathways listed in §3.
- **Connects FROM:** Prefrontal (utilities).
- **Connects TO:** the gates on the *few* gated content pathways (Prefrontal→Hippocampus
  store, Hippocampus→Prefrontal recall, Prefrontal→Motor). It never carries content itself.
- **OPEN CHOICE (carried from L9):** router granularity — **one region / two internal stages
  (recommended at this scale)** vs two separate regions (BG + thalamus, Spaun-faithful). v1
  commits to one region, two distinctly-designed stages.

---

## 3. Inter-region wiring (the pathway table)

Three signal classes on **separate paths** (L10): **content/driver** (region→region,
high-bandwidth), **control** (router→pathway, low-bandwidth, inhibitory), **neuromodulatory**
(broadcast, one-to-all, scalar). Draw lines in three passes — never all at once.

**Content (driver) pathways:**

| # | Source | Target | Class | Topology | Delay Δ | Gated? |
|---|---|---|---|---|---|---|
| 1 | Environment | Sensory | driver | encode | 0 | yes (1st-order relay) |
| 2 | Sensory | Prefrontal | driver | structured (dense) | 1 | no |
| 3 | Prefrontal | Hippocampus | driver | random(p) | 1 | **yes (store)** |
| 4 | Hippocampus | Prefrontal | driver | random(p) | 1 | **yes (recall)** |
| 5 | Prefrontal | Motor | driver | structured | 1 | **yes (action-enable)** |
| 6 | Motor | Environment | driver | decode | 0 | no |

*Delay Δ is in `T`-steps; `random(p)` = sparse random connectivity at density p (set per
region during implementation). The three gated rows are exactly L9's gated-channel primitive.*

**Control pathways:** Thalamic Router → gates of pathways 3, 4, 5. Router input = Prefrontal
utilities. Selection (Stage A) picks the winner; gating (Stage B) applies it.

**Neuromodulatory bus:** a single broadcast bar touching every region — two scalars,
**dopamine** (reward / learning-enable) and **ACh** (gain / precision). No addressing.

```
            ┌──────────  NEUROMOD BUS (broadcast: dopamine, ACh)  ──────────┐
            ▼            ▼            ▼            ▼            ▼
        ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
 env ─▶ │SENSORY │─▶│ (PFC)  │  │ HIPPO  │  │ MOTOR  │  │ ROUTER │
 (gate) └────────┘  │PREFRONT│◀▶│ MEMORY │  │ ACTION │  │ (A→B)  │
                    └───┬────┘  └────────┘  └───▲────┘  └───┬────┘
                        │  (2)        (3/4)        (5)      │
                        └── utilities ───────────────────▶ │ selects + gates 3,4,5
                                                            └── control (off/tonic)
                        Motor (5) ──▶ env (action)
```

---

## 4. Build order (v1 → working brain)

Recommended bring-up sequence (smallest dependency surface first):

1. **Sensory** alone — encode a grid-world observation, confirm a stable concept code.
2. **Motor + WTA** — feed it hand-made utilities, confirm a single winner emerges.
3. **Prefrontal** — wire Sensory→PFC→Motor end-to-end, no memory, no gating (open loop).
4. **Thalamic Router** — add Stage A/B; gate pathway 5 (action-enable) first.
5. **Hippocampus** — add store/recall (pathways 3/4) under router gates.
6. **Neuromod bus** — add dopamine/ACh once the closed loop runs.

Each step gets a verification gate in the spirit of the week-7 discipline (forward-shape
check, single-winner check, attractor-persistence check, reload Δ=0 where applicable).

---

## 5. Open questions (tracked, not resolved in v1)

1. **Memory encoding scheme** — bind-and-superpose vs recurrent sequence-chaining (§2.2).
2. **Router granularity** — one region/two stages (chosen) vs two regions (§2.5) — revisit if
   selection and gating prove hard to co-tune.
3. **Hippocampus neuron primitive** — RLeaky vs Synaptic, pending a delay-hold measurement.
4. **Coding scheme** — rate vs latency for the sensory input and the inter-region codes.
5. **Delays** — placeholder Δ=1 on most links; real values set when ring-buffers are wired.

---

*v1 — this is the starting point. Revise counts, codings, and gating as the bring-up in §4
surfaces real constraints. Bump to v2 when the closed loop runs on grid-world.*
