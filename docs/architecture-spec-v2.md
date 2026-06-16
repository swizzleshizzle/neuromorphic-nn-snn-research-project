# Five-Region Architecture Specification — v2

**Status:** Authoritative · v2 (post-bring-up — grounded in a working implementation)
**Date:** 2026-06-06 · Week 9 hands-on
**Phase:** 2 — Multi-region brain · **Step 2.1/2.2** — architecture + first implementation pass
**Supersedes:** `docs/architecture-spec-v1.md` (v1, archived — the pre-implementation starting point)
**Design spec:** `docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md`
**Tracking plan:** `docs/superpowers/plans/2026-06-05-phase2-architecture-spec-plan.md`
**Code:** `src/neuromorphic/regions/`, `src/neuromorphic/connections/`, `src/neuromorphic/neuromod.py`
**Bring-up evidence:** EXP-013 … EXP-018 (`experiments/013…018_week9_*`); Week-10 EXP-019 (pattern
completion), EXP-020 (closed loop), EXP-021 (R-STDP taste)

---

## 0. Purpose and what changed from v1

v1 was a defensible *starting point* drawn from the L9/L10 reading notes. v2 records what was
**actually built and verified** in the first implementation pass: real neuron counts, the
resolved open choices, the realized wiring, the excitability constants that made each region
behave, and the honest limits. Every number below is the implemented default
(`src/neuromorphic/…`), cross-checked against the region constructors.

The task is still **grid-world** navigation (4 actions); Phase 3 swaps in the 2×2 Rubik's cube.
Only `N_actions` and the sensory input dimension change when the task changes.

### Headline deltas from v1

- **Coding scheme resolved** → **rate / Poisson** for sensory input and inter-region codes (§5.4).
- **Memory encoding resolved** → **recurrent attractor with a one-shot Hebbian (Hopfield) imprint**,
  not bind-and-superpose (§5.1).
- **Hippocampus primitive resolved** → **`snn.Leaky` + explicit hand-designed Hopfield recurrence**
  (direct control of the attractor weights), not RLeaky/Synaptic (§5.3).
- **Router granularity held** → one region, two stages; gating is **per-action disinhibition with a
  do-nothing floor** (§2.5).
- **Composition model fixed** → regions are `spikes → spikes` and **own their afferent**; inter-region
  `Projection` (current out) is owned *inside* the consuming region so the uniform contract holds.
- **Counts are leaner than v1's 650 budget**: Motor's learned decompression stack and a large Router
  are **deferred** (no training yet), so Motor and Router are minimal at this stage.

### Conventions (locked, repo-wide — unchanged from v1)

- **Tensor contract:** every spike/current signal is `[T, B, N]`. Weight matrices are `[N_post, N_pre]`.
- **Spikes are binary** `{0,1}`; inter-region "vectors" are population/rate codes, not decoded floats.
- **Inference window:** `T = 32` steps per decision; one environment step = one `T`-window.
- **Neuron params (week-7 locked):** `beta=0.9, threshold=1.0, reset_mechanism="subtract"`.
- **Verification-gate discipline:** every region has a bring-up experiment with explicit gates.

---

## 1. Region overview (as built)

| # | Region | Class | Neurons (impl) | Primitive | Weights | Status |
|---|---|---|---:|---|---|---|
| 1 | Sensory Cortex | `SensoryCortex` | 192 (128→64) | Leaky | random init (untrained) | ✅ EXP-013 |
| 2 | Hippocampus | `Hippocampus` | 150 | Leaky + Hopfield recurrence | one-shot Hebbian imprint | ✅ EXP-017 |
| 3 | Prefrontal | `Prefrontal` | 150 (100 RLeaky + 50 Leaky) | RLeaky + Leaky | random init (untrained) | ✅ EXP-015 |
| 4 | Motor Cortex | `MotorCortex` | `N_actions` (=4) | Leaky + lateral inhibition | identity drive + fixed WTA | ✅ EXP-014 |
| 5 | Thalamic Router | `ThalamicRouter` | 2·`N_actions` (=8) | Leaky (2 stages) | hand-derived | ✅ EXP-016 |

Plus the **neuromodulatory bus** `NeuromodBus` (not a region): broadcast scalars **dopamine**
(reward / learning-enable) and **ACh** (gain / precision), one-to-all (§6).

Connection primitives (`src/neuromorphic/connections/`):
- **`Projection`** — sparse random, delayed; spikes → current `[T,B,N_tgt]`; `[N_tgt,N_src]` masked
  weights + integer delay Δ; seeded. Never spikes.
- **`apply_gate(signal, gate_closed)`** — `signal · (1 − gate_closed)`; releases a content pathway
  through the router's inhibitory control lines.

The shared `BrainRegion` ABC (`regions/base_region.py`) defines the contract `forward(spikes)→spikes`,
`reset`, `get_state`, plus opt-in recording hooks (`enable_recording`/`_record`/`get_recording`) that
emit `[T,B,N]` straight into the viz toolkit.

---

## 2. Region specifications (as built)

### 2.1 Sensory Cortex — `SensoryCortex(n_obs, hidden=128, concept=64, weight_gain=5.0)`

- **Structure:** two-stage feedforward Leaky compression `N_obs → 128 → 64` (concept code). Pure
  feedforward, untrained random weights.
- **Encoder:** `encode_gridworld(obs, grid_n, T=32, max_rate=0.5)` — agent + goal one-hots →
  `N_obs = 2·grid_n²` (5×5 → 50) → **Poisson** spikes `[T,B,N_obs]`. Kept separate from the region so
  the contract stays spikes-in.
- **I/O:** `[T,B,N_obs]` → concept `[T,B,64]`.
- **Excitability:** `weight_gain=5.0` — the sparse 2-hot input needs it to drive the hierarchy.
- **Finding (EXP-013):** concept rate ≈ 0.41, 45/64 neurons active, **position-selective**
  (pairwise L1 ≈ 475–773 between grid positions). Stable, non-saturated distributed code.
- **Connects TO (resolved 2026-06-08):** Prefrontal (pathway 2, content) **and now directly to
  Hippocampus** (store content, router-gated — the EC perforant-path analog). The position-selective
  concept code *is* the place-code-like snapshot the hippocampus stores. See §2.2 / §3 pathway 3.

### 2.2 Hippocampus — `Hippocampus(content_dim=64, n_neurons=150, sparsity=0.2, input_gain=3.0, recurrent_gain=2.0)`

- **Encoding (resolved):** recurrent attractor, **one-shot Hebbian imprint**. `store(content)` picks
  the top-`sparsity·N` driven neurons as a sparse pattern `p`, writes the Hopfield outer-product
  `W_rec = recurrent_gain · (s sᵀ)/N` (s = 2p−1, zero diagonal) — making `p` a stable fixed point.
- **Dynamics:** `cur = input_gain·fc_in(content) + (2·spk_prev−1)·W_recᵀ`; Leaky neurons; readout
  `fc_out → Leaky` gives the ~64-D recall code.
- **I/O:** content `[T,B,64]` → recall `[T,B,64]`; population `[T,B,150]` via recording.
- **Finding (EXP-017):** 30/150 pattern **held at rate 1.00, leak 0.00 across the full delay** with no
  input — a clean fixed-point attractor. Recall is content-specific (27/64 read-out units differ
  between two stored patterns).
- **Finding (EXP-019, pattern completion):** a partial cue with up to **90% of content dims masked**
  still recovers the full pattern — late-window **held 1.00 / leak 0.00**, a **+0.89 lift** over the
  recurrence-off control (which sits at a ~0.10 `fc_in` bias floor, not zero). Confirms the memory is
  **content-addressable** (partial input → full pattern), not just persistent. See
  `experiments/019_week10_pattern_completion/`.
- **Store input (resolved 2026-06-08):** the stored content is the **Sensory concept code, delivered
  directly** (Sensory→Hippocampus content pathway), *not* the PFC output — PFC emits action utilities,
  not a snapshot. PFC/Router supply only the **store command** (when to store) by opening the gate.
  This mirrors the biology: the entorhinal/sensory perforant path is the content route into CA3, while
  PFC acts as a controller. Interface unchanged — `fc_in` still takes a 64-D content code; only the
  *source* feeding it changes. PFC↔Hippo content is now one-way (recall, pathway 4).
- **Gating:** store (pathway 3) and recall (pathway 4) released by `apply_gate` (router-driven).

### 2.3 Prefrontal — `Prefrontal(concept_dim=64, n_state=100, n_transform=50, n_actions=4, weight_gain=2.0, sparsity=1.0, delay=1)`

- **Structure:** **owns its afferent** `Projection(64→100, dense, Δ=1)` (pathway 2), feeding a
  **RLeaky** state-hold (100, `all_to_all`) → **Leaky** transform (50) → **Leaky** utility read-out
  (`N_actions`).
- **I/O:** concept `[T,B,64]` → action utilities `[T,B,N_actions]`. Records `state`/`transform`/`utility`.
- **Excitability (critical finding):** `weight_gain=2.0` kept **moderate on purpose**. The untrained
  utility read-out **saturates at higher gain** — the favoured action fires every step and washes out
  all upstream selectivity. At 2.0 the read-out stays in the responsive regime and the utility code
  **discriminates 5/5 grid positions** (EXP-015).
- **Honest limit:** untrained → the argmax action is a fixed structural favourite; task-appropriate
  selection needs training + reward (Phase 2+).
- **Multi-source integration (designed 2026-06-08, Week 10 S2):** PFC gains a **second afferent**
  `Projection(recall_dim=64 → n_state)` for the hippocampal recall (pathway 4), **summed** into the
  RLeaky state-hold alongside the sensory afferent — `state_drive = afferent_sensory(concept) +
  afferent_memory(recall)`. Two summed afferents (not concatenation) keep the streams separable, so the
  router can gate memory independently; the memory afferent receives the **router-gated** recall and
  reads zero when the gate is closed. `forward` becomes `forward(concept, recall=None)` — `recall=None`
  → zeros, keeping the sensory-only open loop (EXP-015) backward-compatible. Memory-afferent scale
  (`memory_gain`, default 1.0) starts ≤ `weight_gain` so the sensory stream keeps driving (a tuning
  knob). **Built 2026-06-14 (EXP-020):** second afferent `Projection(recall_dim=64→n_state)` summed
  into the state-hold; `recall=None` reproduces the EXP-015 output byte-for-byte (golden regression),
  a non-zero recall shifts the utility code, `mem_afferent` recorded for viz.

### 2.4 Motor Cortex — `MotorCortex(n_actions=4, input_gain=2.0, inhibition=3.0, ach_gain=1.0, bus=None)`

- **Structure:** single WTA layer of `N_actions` Leaky neurons. Action-aligned (identity) input drive
  + **lateral inhibition** (`w_inh`: −inhibition off-diagonal, zero self). Winner by spike count.
- **I/O:** utilities `[T,B,N_actions]` → near one-hot action `[T,B,N_actions]`.
- **ACh:** if a `NeuromodBus` is attached, `bus.ach` scales the inhibition (sharper WTA).
- **Finding (EXP-014):** clean case → winner share **0.97**; inhibition sweep sharpens monotonically
  (0.27→0.85). Under tight Poisson margins the WTA can lock an early leader (expected hysteresis).
- **Deferred from v1:** the learned ~64-wide decompression stack (needs training); WTA-focused for now.

### 2.5 Thalamic Router — `ThalamicRouter(n_actions=4, input_gain=0.7, select_bias=0.15, inhibition=3.0, tonic_drive=1.5, gate_inhibition=3.0, mode="tonic")`

- **Granularity (resolved):** one region, two stages; **per-action disinhibition with a do-nothing floor**.
- **Stage A — Selection (BG-like):** lateral-inhibition WTA over utilities. A constant `select_bias`
  makes a single spike sub-threshold, so selection needs *sustained* utility — the **do-nothing floor**
  (floor rate ≈ `select_bias/input_gain` ≈ 0.21).
- **Stage B — Gating (thalamus-like):** gate neurons fire tonically (= channel **closed**); the winner
  inhibits its gate neuron → **disinhibits** (opens) only that channel. `mode="off"` removes tonic
  drive → all open.
- **Output:** gate-closed control lines `[T,B,N_actions]` (inhibitory, no content). Apply with
  `apply_gate`; convert with `open_mask = 1 − gate_closed`.
- **Finding (EXP-016):** selects the strong channel, Motor follows the selection, and below-floor
  utilities **veto** the action (0 channels open, Motor silent). Gate raster shows clean disinhibition.
- **Gain refinement (designed L11, 2026-06-12; built 2026-06-14):** the gate is **multiplicative** —
  biologically thalamic gain modulation, and `g·(Wx) = (gW)x`, so gating the signal ≡ scaling the
  projection weights. The binary `apply_gate` (open/closed) is now joined by **`apply_gain(signal, g)`**
  — a per-pathway **gain** `g ∈ [0, g_max]` (`0` off · `<1` suppress · `1` pass · `>1` amplify) so the
  router can *amplify* as well as *veto*; the binary gate is the special case `g = 1 − gate_closed`
  (back-compat preserved). L10 tonic/burst maps onto this (burst = transient high gain).

---

## 3. Inter-region wiring (realized status)

| # | Source → Target | Class | Topology | Δ | Gated? | Realized |
|---|---|---|---|---:|---|---|
| 1 | Environment → Sensory | encode | Poisson encoder | 0 | yes (relay) | encoder ✅; relay gate ⬜ deferred |
| 2 | Sensory → Prefrontal | driver | dense (PFC-owned `Projection`) | 1 | no | ✅ EXP-015 |
| 3 | Sensory → Hippocampus (store content) | driver | dense | 1 | **yes (store)** | gate ✅ (EXP-017/019); content source ✅ (EXP-020 — Sensory snapshot) |
| 4 | Hippocampus → Prefrontal (recall) | driver | dense | 1 | **yes (recall)** | read-out gated ✅; PFC memory afferent wired ✅ (EXP-020) |
| 5 | Prefrontal → Motor (action-enable) | driver | structured | — | **yes (router)** | ✅ EXP-016 |
| 6 | Motor → Environment | decode | winner read-out | 0 | no | ✅ (spike-count winner) |

**Control:** `ThalamicRouter` → gates of pathways 3/4/5 via `apply_gate`. Selection (Stage A) →
gating (Stage B). For pathway 3 the **store command** (when to store the Sensory snapshot) originates
as a PFC store-utility → Router → store gate — content (Sensory→Hippo) and command (PFC→Router→gate)
travel on separate paths, per the L10 content/control split. **Neuromod:** `NeuromodBus` broadcast to
every region (§6).

Composition rule (v2): a region takes input **spikes** and owns its afferent weights (a `Projection`
or `Linear`); inter-region `Projection`s live *inside* the consumer. Standalone gated pathways use
`apply_gate` on the spike stream.

**Full assembly (Week-11 S1, 2026-06-15):** `src/neuromorphic/brain.py` (`Brain`) instantiates all
five regions + `NeuromodBus` and runs pathways 2/3/4/5 per `step(obs) → action` — **window-batched**
(each region consumes a full `[T,B,N]` window; the EXP-020 pattern). `learn(reward)` wires reward onto
the dopamine bus (the R-STDP third factor; **plasticity deferred**). Driven by the Gymnasium
`GridWorldEnv` (`src/neuromorphic/envs/gridworld.py`); smoke test `tests/integration/test_brain.py`.
v1 simplifications: store/recall (p3/p4) use explicit `store`/`recall` flags, not yet router-issued
commands; the router still gates p5 (PFC→Motor).

---

## 4. Build order — completed

All six bring-up steps done, each with a passing verification gate (120 tests total):

1. ✅ **Sensory** alone — stable selective concept code (EXP-013).
2. ✅ **Motor + WTA** — single winner = max utility (EXP-014).
3. ✅ **Prefrontal** — Sensory→PFC→Motor open loop, 5/5 distinct utility codes (EXP-015).
4. ✅ **Thalamic Router** — Stage A/B, gates pathway 5, do-nothing veto (EXP-016).
5. ✅ **Hippocampus** — store/hold/recall under gated pathways 3/4 (EXP-017).
6. ✅ **Neuromod bus** — ACh sharpens Motor WTA; dopamine learning-enable hook (EXP-018).

---

## 5. Resolved decisions (were open in v1)

1. **Memory encoding** — recurrent attractor + one-shot Hebbian imprint (§2.2).
2. **Router granularity** — one region, two stages; per-action disinhibition + floor (§2.5).
3. **Hippocampus primitive** — Leaky + explicit Hopfield recurrence (§2.2).
4. **Coding scheme** — rate / Poisson (§2.1, inter-region codes).
5. **Delays** — Δ=1 realized on pathway 2 via the PFC-owned `Projection`; other links still
   placeholder (Δ=0/1) pending ring-buffer standardization.

---

## 6. Neuromodulatory bus — `NeuromodBus(dopamine=0.0, ach=1.0, learning_threshold=0.5)`

- **ACh (live):** gain/precision. Motor reads `bus.ach` to scale lateral inhibition — higher ACh ⇒
  sharper WTA (EXP-018: winner share 0.55→0.85, then saturates). Only meaningful where channels
  compete (post-router the selection is already crisp).
- **Dopamine (hook):** reward / learning-enable, broadcast one-to-all. `learning_enabled =
  dopamine ≥ learning_threshold`. No plasticity exists yet, so it is a signal + hook for future
  STDP / reward-modulated learning.
- **Learning rule (designed L11, 2026-06-12 — R-STDP / three-factor):** the planned plasticity is
  **reward-modulated STDP** — `Δw_ij = β·(R − b)·e_ij` (Izhikevich 2007; Frémaux & Gerstner 2016).
  The dopamine scalar **is the third factor** `(R − b)` (reward − baseline/expected); `e_ij` is a
  per-synapse **eligibility trace** (STDP-fed, slow decay `τ_e`) added to each *trainable* `Projection`,
  which bridges the distal-reward credit-assignment gap. At reward: `Δw = β·dopamine·e`. First target =
  the PFC utility readout (closest to reward); this is what trains selection past its fixed-favourite
  limit (§2.3, §7). See week-10 note Session 3.
- **First taste built 2026-06-14 (EXP-021):** the three-factor rule on the PFC utility readout alone
  (everything upstream frozen; a one-off, *not* yet baked into the shared `Projection`). A
  normalised, STDP-fed eligibility trace + dopamine `(R−b)` from the `NeuromodBus`, ε-greedy
  exploration on a toy reward: the rewarded (non-favourite) action's readout weights grow, its utility
  rises, the old favourite is depressed, and greedy selection flips to the target — the first evidence
  the network can move selection in the reward direction. Stable under eligibility normalisation +
  weight clipping. Next build: an eligibility-bearing trainable `Projection`.

---

## 7. Known limits / open for the next phase

- **Learning: first taste only.** All learned regions (Sensory, PFC transform, Motor decompression)
  still use random init, so steady-state selection is **not yet task-meaningful**. The R-STDP
  three-factor rule (§6) now has a **working one-off on the PFC readout** (EXP-021 — moves selection
  past the fixed favourite), but it is not yet baked into the shared `Projection` or run on the full
  closed loop. Generalising plasticity to a trainable `Projection` is the headline next step.
- **Closed loop now runs (EXP-020):** hippocampal recall is fed back into PFC's second afferent and
  measurably shifts the utility code (pathway 4 wired; pathway 3 content source = the Sensory snapshot).
  Still open: the Sensory relay gate (pathway 1) is not implemented, and the loop is untrained.
- **Counts deferred:** Motor decompression stack and a fuller Router are deferred until trainable.
- **Single-item memory:** the attractor holds one pattern; multi-item addressable memory
  (bind-and-superpose) remains a future option (v1 §5.1 alternative).
- **Delays:** most pathway delays are still placeholders.

---

*v2 — grounded in the working five-region implementation. Promote to v3 once training/plasticity
makes selection task-meaningful and the closed loop (recall→PFC, reward→learning) runs on grid-world.*
