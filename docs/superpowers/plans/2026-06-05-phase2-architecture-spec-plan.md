# Phase 2 — Five-Region Architecture Spec — Tracking Plan

> **For agentic workers:** this is the living tracker for the five-region architecture
> specification. The spec itself is the deliverable; this doc tracks its state across Phase 2.
> Steps use checkbox (`- [ ]`) syntax. Update the boxes as the spec evolves — do not rewrite
> the spec here, link to it.

**Goal:** Define and iterate a buildable specification of the five-region spiking brain
(Sensory, Hippocampus, Prefrontal, Motor, Thalamic Router) — neuron counts, neuron types,
internal connectivity, I/O spike formats, and inter-region wiring — from a small v1 starting
point through a closed-loop v2 that runs on grid-world.

**Spec under management:** `docs/architecture-spec-v1.md`

**Parent design notes:**
- Obsidian — `300 Efforts/.../Weekly Notes/week-09-brain-architectures` (Sessions 1–3)
- L9 Architectural Decisions Catalogue (Spaun + NEF)
- L10 Communication Protocol Spec (driver/control/neuromod buses, gated channels)

**Phase / step:** Phase 2 (Multi-region brain) · Step 2.1 (architecture specification)

---

## Status snapshot

| Item | State | Date |
|---|---|---|
| v1 spec drafted (5 regions, wiring, build order) | **DONE** | 2026-06-05 |
| Open questions logged | **DONE** | 2026-06-05 |
| Decisions resolved (encoding, router granularity) | open | — |
| v2 spec (counts/codings tuned from bring-up) | not started | — |

---

## Task 1: v1 specification — *the starting point*

- [x] **Region roster fixed** — Sensory (200, Leaky), Hippocampus (150, RLeaky), Prefrontal
  (150, RLeaky+Leaky), Motor (100, Leaky+WTA), Thalamic Router (50, Leaky). Total 650.
- [x] **Per-region spec** — for each: neuron count, neuron type, internal connectivity,
  input spike format, output spike format, connects-TO / connects-FROM.
- [x] **Inter-region pathway table** — 6 content/driver pathways with class, topology, delay,
  gated-flag; control lines from the router; neuromod bus (dopamine, ACh).
- [x] **Build order** — 6-step bring-up sequence (Sensory → Motor → PFC → Router → Hippo →
  neuromod), each with a verification gate.
- [x] **Conventions locked** — `[T, B, N]` tensor contract, binary spikes, `T = 32` window.

## Task 2: Resolve the open architectural choices

- [x] **Memory encoding scheme** — **resolved 2026-06-06: recurrent attractor with a
  one-shot Hebbian (Hopfield-style) imprint** (not bind-and-superpose). Single-pattern
  working memory; holds the stored concept as a stable fixed point (spec §2.2 / §5.1).
- [ ] **Router granularity** — confirm one-region/two-stage vs two-region split holds after a
  selection+gating co-tune (spec §2.5 / §5.2).
- [x] **Hippocampus primitive** — **resolved 2026-06-06: `snn.Leaky` + an explicit
  hand-designed Hopfield recurrence** (outer-product `W_rec`), rather than RLeaky/Synaptic
  — gives direct control of the attractor weights for the one-shot imprint. Held the
  pattern at rate 1.00 across the full delay, so no longer-time-constant primitive was
  needed (spec §5.3).
- [ ] **Coding scheme** — rate vs latency for sensory input and inter-region codes (spec §5.4).
- [ ] **Delays** — replace placeholder Δ=1 with measured/chosen values once ring-buffers
  exist (spec §5.5).

## Task 3: First implementation pass (feeds spec corrections back)

- [x] **Sensory** standalone — encode a grid-world observation; confirm a stable concept code.
  *(2026-06-06: built `SensoryCortex` + `encode_gridworld`; EXP-013 PASS — rate 0.41,
  45/64 active, position-selective. See `experiments/013_week9_sensory_bringup/results.md`.)*
- [x] **Motor + WTA** — single-winner check on hand-made utilities.
  *(2026-06-06: built `MotorCortex` (WTA via lateral inhibition, decompression
  deferred); EXP-014 PASS — winner=argmax utility, share 0.97, inhibition sweep
  sharpens monotonically. See `experiments/014_week9_motor_wta_bringup/results.md`.)*
- [x] **Prefrontal** — Sensory→PFC→Motor open loop end-to-end.
  *(2026-06-06: built `Prefrontal` (owns afferent `Projection` for pathway 2 Δ=1;
  RLeaky state-hold → Leaky transform → utility readout). EXP-015 PASS — end-to-end
  flow, every stage alive, utility code discriminates 5/5 agent positions. Untrained
  argmax stays a fixed favourite (expected). See
  `experiments/015_week9_open_loop_bringup/results.md`.)*
- [x] **Thalamic Router** — Stage A/B; gate action-enable (pathway 5) first.
  *(2026-06-06: built `ThalamicRouter` (Stage A WTA select + Stage B tonic-inhibition/
  disinhibition gating, do-nothing floor) + `apply_gate` primitive. EXP-016 PASS —
  selects ch2, gates pathway 5 so Motor follows the selection, vetoes action below the
  floor. See `experiments/016_week9_router_gating_bringup/results.md`.)*
- [x] **Hippocampus** — store/recall (pathways 3/4) under router gates.
  *(2026-06-06: built `Hippocampus` — recurrent attractor, one-shot Hebbian imprint.
  EXP-017 PASS — 30/150 pattern held at rate 1.00 / leak 0.00 across the delay,
  content-specific recall (27/64), pathways 3/4 gated via `apply_gate`. See
  `experiments/017_week9_hippocampus_bringup/results.md`.)*
- [ ] **Neuromod bus** — dopamine/ACh on the closed loop.

## Task 4: Promote to v2

- [ ] Fold bring-up findings (real counts, codings, delays, resolved choices) into a
  `docs/architecture-spec-v2.md`; archive v1; update this tracker and the Obsidian note.

---

## Change log

- **2026-06-06 (Week 9, hands-on, cont.⁴):** built `Hippocampus` (build-order step 5)
  — recurrent attractor memory with a one-shot Hebbian (Hopfield) imprint; resolves the
  §5.1 encoding choice (attractor, not bind-and-superpose) and the §5.3 primitive choice
  (Leaky + explicit Hopfield recurrence). EXP-017: store→hold→recall under gated pathways
  3/4 — 30/150 pattern held at rate 1.00 / leak 0.00 across the delay, content-specific
  recall. 9 new tests; 113 total passing.
- **2026-06-06 (Week 9, hands-on, cont.³):** built `ThalamicRouter` (build-order
  step 4) — Stage A basal-ganglia-like WTA selection + Stage B thalamus-like
  tonic-inhibition/disinhibition gating, with a constant-bias do-nothing floor.
  Added the `apply_gate` control primitive (`connections/gating.py`). EXP-016 gates
  pathway 5 (PFC→Motor): the router selects a channel, disinhibits only that relay,
  Motor follows the selection, and below-floor utilities veto the action. 19 new
  tests (router + gating + gated-loop integration); 104 total passing.
- **2026-06-06 (Week 9, hands-on, cont.²):** built `Prefrontal` and wired the
  Sensory→PFC→Motor **open loop** (build-order step 3). PFC owns its afferent
  `Projection` (pathway 2, Δ=1) per the agreed composition model; RLeaky state-hold
  → Leaky transform → utility readout. EXP-015 confirms end-to-end spike flow with
  every stage alive and the utility code discriminating all 5 agent positions.
  Surfaced + tuned the untrained-readout saturation trap (excitability kept moderate
  so upstream selectivity propagates). 15 new tests (incl. open-loop integration);
  85 total passing.
- **2026-06-06 (Week 9, hands-on, cont.):** built `MotorCortex` — WTA action
  selection via lateral inhibition (decompression stage deferred until trainable).
  EXP-014 bring-up confirms a single winner = argmax utility (share 0.97) and
  monotonic sharpening under the inhibition sweep (Task 3, Motor step). 11 new
  tests (70 total passing).
- **2026-06-06 (Week 9, hands-on):** first implementation pass began. Built the
  multi-region framework foundation — `BrainRegion` ABC
  (`src/neuromorphic/regions/base_region.py`), sparse delayed `Projection`
  (`src/neuromorphic/connections/projection.py`), and the first concrete region
  `SensoryCortex` + `encode_gridworld` (`src/neuromorphic/regions/sensory_cortex.py`),
  all TDD (31 new tests, 59 total passing). EXP-013 bring-up confirms a stable,
  position-selective concept code (Task 3, Sensory step). Design spec:
  `docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md`.
- **2026-06-05 (Week 9, Session 3):** v1 spec drafted (`docs/architecture-spec-v1.md`).
  Roster, per-region specs, pathway table, build order, and open questions all in place.
  Tracked in the Obsidian week-9 note (Session 3) and here.
