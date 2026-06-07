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

- [ ] **Memory encoding scheme** — pick bind-and-superpose vs recurrent sequence-chaining for
  the hippocampus (spec §2.2 / §5.1).
- [ ] **Router granularity** — confirm one-region/two-stage vs two-region split holds after a
  selection+gating co-tune (spec §2.5 / §5.2).
- [ ] **Hippocampus primitive** — RLeaky vs Synaptic, decided by a delay-hold measurement
  (spec §5.3).
- [ ] **Coding scheme** — rate vs latency for sensory input and inter-region codes (spec §5.4).
- [ ] **Delays** — replace placeholder Δ=1 with measured/chosen values once ring-buffers
  exist (spec §5.5).

## Task 3: First implementation pass (feeds spec corrections back)

- [x] **Sensory** standalone — encode a grid-world observation; confirm a stable concept code.
  *(2026-06-06: built `SensoryCortex` + `encode_gridworld`; EXP-013 PASS — rate 0.41,
  45/64 active, position-selective. See `experiments/013_week9_sensory_bringup/results.md`.)*
- [ ] **Motor + WTA** — single-winner check on hand-made utilities.
- [ ] **Prefrontal** — Sensory→PFC→Motor open loop end-to-end.
- [ ] **Thalamic Router** — Stage A/B; gate action-enable (pathway 5) first.
- [ ] **Hippocampus** — store/recall (pathways 3/4) under router gates.
- [ ] **Neuromod bus** — dopamine/ACh on the closed loop.

## Task 4: Promote to v2

- [ ] Fold bring-up findings (real counts, codings, delays, resolved choices) into a
  `docs/architecture-spec-v2.md`; archive v1; update this tracker and the Obsidian note.

---

## Change log

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
