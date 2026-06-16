# Week 10 Hands-On — Region Build Day (Phase 2, Step 2.1/2.2)

> **For agentic workers (Claude Code):** implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax. **Read the two source-of-truth docs first** before touching code:
> `docs/architecture-spec-v2.md` (the as-built spec + all designed upgrades) and the Obsidian
> week-10 note Sessions 1–3 (the design rationale). This plan is self-contained; the brief that
> spawned it is paraphrased below.

**Session:** Saturday 2026-06-13, 9:00am–12:00pm (3 h). **Phase 2, Step 2.1/2.2.**

---

## ⚠️ GROUND TRUTH — read before you do anything

The original brief says *"implement the remaining 4 regions: hippocampus, prefrontal_cortex,
motor_cortex, thalamic_router."* **Those regions already exist and are tested.** They were built
in the Week-9 sprint (EXP-013…018):

| File (actual path) | Class | Status | Verified |
|---|---|---|---|
| `src/neuromorphic/regions/sensory_cortex.py` | `SensoryCortex` | built | EXP-013 |
| `src/neuromorphic/regions/hippocampus.py` | `Hippocampus` | built + pattern completion | EXP-017, EXP-019 |
| `src/neuromorphic/regions/prefrontal.py` | `Prefrontal` | built (single-source) | EXP-015 |
| `src/neuromorphic/regions/motor_cortex.py` | `MotorCortex` | **complete, no work** | EXP-014 |
| `src/neuromorphic/regions/thalamic_router.py` | `ThalamicRouter` | built | EXP-016 |

There is a full test suite (`tests/regions/*`, `tests/connections/*`, `tests/neuromod/*`,
`tests/integration/*`) that is green. **DO NOT rebuild these files from scratch — that destroys
working, tested code.** Note also: the real package path is `src/neuromorphic/regions/`, not
`src/regions/` as the brief wrote.

**What this session actually builds = the designed-but-unimplemented upgrades** from the Week-10
design sessions (1–3). Each still satisfies the brief's per-region acceptance criteria. Work the
upgrades, not a rewrite.

---

## Acceptance criteria (the brief's bar, applied to every task)

Each touched region/pathway must still:

- **Accept input spikes** via the projection protocol (`Projection` / `apply_gate`, `[T,B,N]`).
- **Process internally** (its own neuron dynamics).
- **Produce output spikes** (`[T,B,N]`).
- **Log state for visualization** (`enable_recording` + `self._record(...)` → `get_recording`).
- **Be tested in isolation** (a bring-up experiment + unit test) **before** any wiring.

Discipline: match the existing TDD pattern (write/extend the `tests/` unit test first), keep the
full suite green, and add a numbered `experiments/NNN_.../` bring-up with a `results.md` and a
verification-gate block (the EXP-013…019 house style). New experiment numbers continue from 019:
EXP-020, EXP-021.

---

## Task 0 — Pre-flight (10 min)

- [ ] `git status` clean (only `.claude/` untracked expected).
- [ ] Run the full suite to capture the green baseline: `pytest -q` (expect all pass).
- [ ] Confirm imports: `python -c "from neuromorphic.regions import Hippocampus, Prefrontal, MotorCortex, ThalamicRouter, SensoryCortex; from neuromorphic.connections import Projection, apply_gate"`.

---

## Task 1 — Prefrontal multi-source integration (45 min) · spec §2.3

The one real "region build." PFC currently integrates only the sensory concept; add the
hippocampal recall as a **second summed afferent** (design decided Week-10 S2 — summed, *not*
concatenated, so memory stays independently router-gateable).

**Files:** edit `src/neuromorphic/regions/prefrontal.py`; extend `tests/regions/test_prefrontal.py`.

- [ ] Add a second afferent `Projection(recall_dim=64 → n_state)` alongside the existing sensory
  afferent. New `__init__` args: `recall_dim=64`, `memory_gain` (start `≤ weight_gain` so sensory
  keeps driving — see spec §2.3 tuning note).
- [ ] Change `forward(self, input_spikes, recall_spikes=None)`: when `recall_spikes is None` use
  zeros (keeps the EXP-015 sensory-only open loop byte-for-byte compatible); otherwise
  `state_drive = afferent_sensory(concept) + afferent_memory(recall)` summed into the RLeaky
  state-hold.
- [ ] Keep recording `state` / `transform` / `utility`; add a `mem_afferent` record if cheap.
- [ ] **Unit tests:** (a) `recall=None` path reproduces the prior single-source output (regression);
  (b) shape contract `[T,B,64]×2 → [T,B,N_actions]`; (c) a non-zero recall **shifts** the utility
  code vs `recall=None` (integration of the second source actually changes the output).
- [ ] **Isolation gate:** feed a fixed sensory concept + a fixed recall code; confirm the utility
  code differs from the sensory-only case. (This becomes part of EXP-020 below.)

**Acceptance:** spikes in (two projection streams) → internal RLeaky/Leaky → utility spikes out →
state logged → isolation test green; full suite still green.

---

## Task 2 — Thalamic Router continuous gain (30 min) · spec §2.5

Generalise the gate from binary to a continuous multiplicative gain so the router can **amplify**
as well as suppress (design decided Week-10 S3 / L11).

**Files:** edit `src/neuromorphic/connections/gating.py`; extend `tests/connections/test_gating.py`.

- [ ] Extend `apply_gate` (or add `apply_gain`) so the control value is a per-pathway gain
  `g ∈ [0, g_max]` applied multiplicatively to the signal: `y = g · signal` (`g=0` off, `0<g<1`
  suppress, `g=1` pass, `g>1` amplify). Keep the current binary `gate_closed` behaviour as the
  default path so EXP-016 and existing tests are unchanged.
- [ ] **Unit tests:** `g=0` ⇒ zeros; `g=1` ⇒ identity (matches old open behaviour); `g=2` ⇒ 2×;
  back-compat test for the existing `gate_closed` call signature.
- [ ] No region internals change here — this is a connection-primitive refinement.

**Acceptance:** existing router/gating tests still green; new gain behaviour covered.

---

## Task 3 — Sensory→Hippocampus store-content rewire (20 min) · spec §2.1/§2.2/§3 pathway 3

Decision from Week-10 S1: the hippocampus stores the **sensory** concept snapshot directly (the
content), with the store **gated** by the router on a PFC command. The `Hippocampus` class itself
does **not** change (`fc_in` still takes a 64-D content code) — only the *source* feeding it.

**Files:** wherever the store content is composed (the EXP-020 bring-up below; no region-internal
edit). Confirm `Hippocampus.store()` / `forward()` are content-source-agnostic (they are).

- [ ] In the closed-loop wiring (Task 4), feed the **Sensory** concept (not PFC output) as the store
  content, through the router-gated store pathway (`apply_gate`).
- [ ] Sanity gate: a store driven by the sensory snapshot recalls a content-specific pattern (reuse
  the EXP-017/019 recall metric).

**Acceptance:** store content originates from Sensory; recall still content-specific.

---

## Task 4 — Closed-loop bring-up EXP-020 (40 min)

Wire the pieces and verify end-to-end **after** each region passed its isolation test.

**Files:** `experiments/020_week10_closed_loop/run.py` (+ `results.md`); optional
`tests/integration/test_closed_loop.py`.

- [ ] Wire: Sensory → PFC (pathway 2) **+** gated Hippocampus recall → PFC (pathway 4, via
  `apply_gate`) **+** gated Sensory → Hippocampus store (pathway 3). Router drives the gates.
- [ ] Verify: every stage emits spikes; PFC utility code **changes when a memory is recalled** vs
  not (this is the payoff of Task 1 + Task 3 together); Motor still selects a single winner.
- [ ] Log state at each region (rasters for sensory concept, hippo attractor, PFC utility, motor
  winner) — the viz toolkit consumes `[T,B,N]` directly.
- [ ] `results.md` with a verification-gate block in the EXP-019 house style.

**Acceptance:** closed loop runs; recall measurably shifts PFC utilities; rasters saved.

---

## Task 5 — STRETCH: R-STDP first taste, EXP-021 (30–45 min) · spec §6

Only if Tasks 1–4 are green with time left. Smallest possible taste of the learning engine designed
in L11 (Week-10 S3): an eligibility trace on **one** trainable projection — the PFC utility readout
(closest to reward, smallest).

**Files:** `experiments/021_week10_rstdp_taste/` (keep it a one-off; do not yet bake plasticity into
the shared `Projection`).

- [ ] Add a per-synapse **eligibility trace** `e` (STDP-fed, decay `τ_e`) on the PFC readout weights.
- [ ] At a toy reward, apply `Δw = β · dopamine · e` (dopamine from `NeuromodBus`; `R−b` = reward −
  running-baseline).
- [ ] Gate: after a few rewarded trials favouring one action, that action's readout weights **grow**
  and its utility rises — i.e. selection moves in the reward direction (past the EXP-015
  fixed-favourite limit).

**Acceptance:** weight change tracks reward sign on the toy task. (If short on time, this is the
first thing to cut.)

---

## Task 6 — Verification & hand-back (15 min)

- [ ] Full `pytest -q` green (baseline + new tests).
- [ ] Every touched region: state-logging confirmed + an isolation test exists.
- [ ] Each new EXP has a `results.md` with a gate block.
- [ ] Flip the realized-status flags in `docs/architecture-spec-v2.md` §3 (pathway 4 "wiring ⬜" →
  ✅; pathway 3 content-source ⬜ → ✅) and §2.3 (PFC multi-source "deferred" → built).
- [ ] Add a change-log entry + status row to
  `docs/superpowers/plans/2026-06-05-phase2-architecture-spec-plan.md`.
- [ ] Note the session in the Obsidian week-10 note (a Session 4 "build" entry) — or leave a
  one-line pointer for the user to sync.

---

## Time budget & cut order (3 h hard cap)

Pre-flight 10 · **Task 1 PFC 45** · **Task 2 Router 30** · **Task 3 Hippo rewire 20** ·
**Task 4 closed loop 40** · Task 5 R-STDP 30–45 · Verify 15.

**Cut order if running long:** drop **Task 5** first (it's a stretch), then trim Task 4's
integration test (keep the run + rasters). **Never cut** Tasks 1–3 or their isolation tests — those
are the load-bearing region work. Motor needs no work at all.

---

## What "done" looks like

PFC integrates sensory + gated memory; the router gate can amplify/suppress (not just on/off); the
hippocampus stores the sensory snapshot under router gating; the closed loop runs end-to-end with
recall visibly shifting PFC utilities; the full test suite is green; and (stretch) one projection
has learned from reward. All four brief regions satisfy the interface criteria — because they
already did, and now their designed upgrades do too.
