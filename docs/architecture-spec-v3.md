# Five-Region Architecture Specification — v3

**Status:** Authoritative. Folds the **as-trained** configuration into the region tables and **retargets
the architecture to the Phase-3 2×2 Rubik's cube**. Regions + wiring inherit from v2 (still code-grounded);
training/plasticity is governed by **ADR-0001** (Amendments 1–6, authoritative).
**Date:** 2026-07-17 · Week 15/16, Phase-2 → Phase-3 rotation
**Phase:** transition doc for **Phase 3 — 2×2 Rubik's cube capstone** (plan timeline Jul 25 – Sep 27)
**Supersedes:** `docs/architecture-spec-v2.md` (v2 — as-built snapshot; its own promotion condition, "promote
to v3 once training makes selection task-meaningful," has now fired)
**Grounding:**
- `docs/adr/0001-multi-region-training-strategy.md` (Amendments 1–6 — the actual training strategy + results)
- `docs/phase2-honest-assessment.md` (the checkpoint scorecard + disclosed limitations)
- `docs/phase2-to-phase3-transition.md` (what carries forward / what is new for the cube)
- EXP-024 … EXP-028 (`experiments/024…028_*`; EXP-027/028 carry a committed `RESULTS.md` — the habit
  adopted at the 2026-07-13 audit)
- Tag `phase-2-complete` (2026-07-14) — the substrate this spec builds on

---

> [!important] How to read v3
> v2 described the brain **as built** (Week-9 bring-up: real regions, real wiring, "untrained / random
> init"). v3 does two things v2 deferred:
>
> 1. **Folds in the as-trained truth.** The brain was trained and evaluated on grid navigation across
>    EXP-023…EXP-028. What actually trains is **narrow**, and this spec now says so in the region tables
>    themselves rather than in a warning banner.
> 2. **Retargets to the cube.** v2's optimistic "only `N_actions` and the sensory input dimension change
>    when the task changes" is **superseded**. The 2×2 cube breaks the shortcuts that made a
>    frozen-extractor + linear-head policy sufficient on a trivial fully-observable grid — so v3 records,
>    per region, both the Phase-2 as-trained state **and** the Phase-3 target.
>
> This is a **transition/scoping** doc, not the full Phase-3 spec. The full Phase-3 spec (env details,
> training recipe, falsifiers) is authored at the Phase-3 kickoff (~Jul 25).

## 0. What changed from v2

**v2 → v3 delta, in one paragraph.** v2 was accurate on regions and wiring but stale on training — it
said "no plasticity exists yet / untrained / first taste only," which was true at 2026-06-06 bring-up and
false by the time the brain was trained and characterized. The honest Phase-2 result: the brain runs as a
**frozen feature extractor** under `no_grad`; a single trainable `nn.Linear` **policy head** on the sensory
`concept[64]` learns via surrogate-gradient **REINFORCE**; memory is **bypassed** (`recall=False`); PFC,
router, and motor are computed and visualized but **off the policy path** — so **1 of 5 regions drives the
action**. Region-local R-STDP plasticity is designed-but-deferred (EXP-021 taste only); `Brain.learn()`
still only writes the reward to the dopamine bus — *reward never changes a synapse in the brain*. The one
region that was genuinely engaged — the sensory encoder, pre-trained on goal-relative displacement (EXP-026)
— **specializes through learning** and lifts the held-out cap (EXP-027 characterization; EXP-028 causal
dose-response). That is the scientific win of Phase 2, and the launch point for Phase 3.

### Headline deltas from v2

- **As-trained config is now first-class** (§1, §2, §4): the region tables carry an *on the policy path?*
  and *what trains?* column, not just "as built."
- **The nav cap is optimization-limited, not encoder-limited** (EXP-028, ADR Amdt 6): moderate input noise
  *doubled* held-out success (43% → 83% at σ=0.4, all 12 seeds). So the first Phase-3 lever on the cap is
  **regularization / exploration**, not richer encoders (§4.3).
- **The cube forces the deferred regions on-path** (§1, §3): hippocampus (move history) and PFC (subgoal
  decomposition) can no longer be bypassed; the action space grows (4 → 6) and sensory input goes from
  4 ints to 24 facelets × 6 colors.
- **New Phase-3 methodology commitments** carried from the honest assessment: a **monolithic
  same-neuron-count baseline** (the "does regionalization help?" control), **curriculum learning**
  (1→2→3-move scrambles), and the standing **committed-`RESULTS.md`-per-experiment** habit.

### Conventions (locked, repo-wide — unchanged from v1/v2)

- **Tensor contract:** every spike/current signal is `[T, B, N]`. Weight matrices are `[N_post, N_pre]`.
- **Spikes are binary** `{0,1}`; inter-region "vectors" are population/rate codes, not decoded floats.
- **Inference window:** `T = 32` steps per decision; one environment step = one `T`-window.
- **Neuron params (week-7 locked):** `beta=0.9, threshold=1.0, reset_mechanism="subtract"`.
- **Verification-gate discipline:** every region has a bring-up experiment with explicit gates; every
  experiment commits an in-repo `RESULTS.md` (habit adopted 2026-07-13).

---

## 1. Region overview — as built · as trained · Phase-3 target

The substrate (classes, primitives, wiring) is unchanged from v2 and **carries forward** — do NOT rebuild
it. What changes is (a) the honest record of what actually trained on the grid, and (b) the target
configuration for the cube.

| # | Region | Class | As-built primitive (v2) | On the v1 policy path? (as trained, grid) | Phase-3 target (2×2 cube) |
|---|---|---|---|---|---|
| 1 | Sensory Cortex | `SensoryCortex` | Leaky compression `N_obs→128→64` | **Yes — the only trained region.** Pre-trained on displacement (EXP-026), then frozen; the linear policy head reads its `concept[64]`. | Re-target input: **24 facelets × 6 colors → population coding**; re-shape/-train the encoder on cube state. |
| 2 | Hippocampus | `Hippocampus` | Leaky + Hopfield recurrence, one-shot Hebbian imprint | **No — fully bypassed** (`recall=False`). Built, tested (EXP-017/019), never exercised by the policy. | **Engaged** — track move history, recognize visited states (avoid cycles). Memory can no longer be bypassed. |
| 3 | Prefrontal | `Prefrontal` | RLeaky state-hold + Leaky transform + utility readout | **No — off policy path.** Runs and is visualized; its utility readout is degenerate at init (decodability collapses PFC state[100] R²0.76 → utility[4] 0.03). | **Engaged** — subgoal decomposition / lookahead. The cube needs structure a reactive policy can't give. |
| 4 | Motor Cortex | `MotorCortex` | `N_actions` Leaky WTA + lateral inhibition | **No — off policy path.** Action comes from the linear head, not `motor.winner` (argmax saturates → zero-grad, ADR Amdt 1). | **Expand to 6 actions** (cube moves; 12 on a 3×3); re-enter the policy path once it has a non-degenerate, trainable readout. |
| 5 | Thalamic Router | `ThalamicRouter` | Leaky, 2 stages, per-action disinhibition + floor | **No — spectator.** Gates are computed but the trained policy is `sensory → head`. | **Task-phase-dependent gating** — route by cube-solving phase (e.g. scramble-recognition vs subgoal execution). |

Plus the **neuromodulatory bus** `NeuromodBus` (not a region): broadcasts **dopamine** (reward /
learning-enable) and **ACh** (gain / precision). Dopamine remains a **hook** — no plasticity consumes it yet
(§6).

**The honest one-liner (carry it loud into Phase 3):** Phase 2 delivered *an honestly-characterized
frozen-feature-extractor brain with a demonstrably specialized sensory encoder* — **not** a learning
multi-region brain. The cube is where regions 2–5 have to earn their place.

---

## 2. Region specifications — as-trained notes + Phase-3 changes

Full as-built specs (constructors, excitability constants, bring-up findings) live in **v2 §2** and are
unchanged. This section records only what training revealed and what the cube changes.

### 2.1 Sensory Cortex — the one engaged region

- **As trained (grid):** the encoder was **pre-trained** on goal-relative displacement — a scratch
  `Linear(concept→2)` predicts `(gx−ax, gy−ay)` from the concept rate, backprop through the spiking encoder
  (surrogate gradients); the readout is discarded, the shaped encoder frozen for RL (EXP-026, ADR Amdt 3).
- **What it bought:** held-out navigation lifted ~2.5–5× over a random encoder (paired 19/24 seed-cells,
  +28/+34 pts). Characterization (EXP-027, Amdt 4/5): the trained concept decodes displacement at **R² 0.86–0.90**,
  beats every non-sensory spectator on both task targets in **100% of 12 seeds** (shuffle-null + PCA-matched
  controls), and the code is **distributed** (participation ratio ~24 of 64) **and causally robust** (graceful
  dropout curve; dropping half the units costs ~11 pts).
- **Cube change:** input goes from 4 ints (`ax,ay,gx,gy`) to **24 facelets × 6 colors**. Encode via
  **population coding** (per plan) rather than the grid one-hot Poisson encoder. The `encode_gridworld`
  helper is replaced by a cube encoder; the region contract (spikes-in → `concept[64]`) is unchanged.
  Whether displacement-style supervised pre-training has a cube analog (e.g. distance-to-solved) is an open
  Phase-3 design question.

### 2.2 Hippocampus — bypassed → engaged

- **As trained (grid):** a true zero on the policy path. `recall=False`; store/recall and the Hebbian
  attractor are built and tested (EXP-017 fixed-point hold; EXP-019 pattern completion, +0.89 lift over the
  recurrence-off control) but **never exercised** by the policy. Reported honestly as bypassed in EXP-027.
- **Why it was defensible:** the 5×5 grid is fully observable — a reactive policy provably suffices, so
  memory was not required (ADR "key enabling fact").
- **Cube change:** the cube **benefits from move history** (avoid cycles, recognize visited states), so the
  attractor's content-addressable recall gets a real job. Bringing memory into the loop is **step 1** of the
  ADR's migration path (enable `recall=True`, train the recall readout while keeping `W_rec` Hebbian).
  Single-item attractor may need extending toward multi-item addressable memory (v1 §5.1 alternative).

### 2.3 Prefrontal — spectator → subgoal engine

- **As trained (grid):** off the policy path. The untrained utility readout is a fixed structural favourite;
  decodability attenuates sharply through it (state[100] R² 0.76 → utility[4] 0.03 → router/motor ~0,
  EXP-027). It was excluded from the v1 policy precisely because its readout is degenerate at init (ADR Amdt 1).
- **Cube change:** solving even 2–3-move scrambles needs **lookahead / subgoal structure** a reactive policy
  can't express, so PFC has to earn its place. Giving it a non-degenerate, trainable readout so it can
  re-enter the policy path is core Phase-3 work.

### 2.4 Motor Cortex — 4 → 6 actions

- **As trained (grid):** off the policy path. The original "motor spike-counts ARE the logits" premise was
  broken two ways (saturation → zero-gradient absorbing state; degenerate one-hot readout) — see ADR Amdt 1.
  The action is read from the linear head instead.
- **Cube change:** action space grows **4 → 6** (the cube's face turns). Expand the WTA layer to 6; the
  deferred learned decompression stack (v2 §2.4) may finally be needed. Re-entering the policy path depends
  on a trainable, non-saturating readout.
- **Why 6 and not 12** (revised 2026-07-24, verified): a 2×2 has no centers, so turning a face is the same
  physical act as counter-turning the opposite face (`U == D'`, `R == L'`, `F == B'`). A 12-move action space
  is exactly 2× redundant. The env holds the DLB corner still and turns only U, R, F, which keeps all
  3,674,160 states and God's number 14 while making every action distinct. **Chance on a depth-1 scramble is
  therefore 1/6, not 1/12.** This is 2×2-only: a 3×3 has fixed centers, so all six faces are genuinely
  distinct and its action space is 12 quarter turns (18 with half-turns). Read the width from
  `neuromorphic.envs.cube.N_ACTIONS`, never a literal. See `docs/superpowers/specs/2026-07-24-cube-env-design.md` §1.

### 2.5 Thalamic Router — spectator → task-phase gating

- **As trained (grid):** spectator. Gates are computed and visualized but the trained policy bypasses them.
- **Cube change:** **task-phase-dependent gating** — the router earns a role by gating pathways according to
  the solving phase (e.g. read-state vs plan vs execute). Reward-modulating the gate is **step 3** of the ADR
  migration path.

---

## 3. Inter-region wiring — realized (grid) vs Phase-3 target

The wiring topology (v2 §3, pathways 1–6) is unchanged and carries forward. What changes is **which
pathways the policy actually uses**.

| Pathway | v2 realized (as built) | As trained on grid (v1 policy) | Phase-3 target (cube) |
|---|---|---|---|
| 1 Env → Sensory | encoder ✅; relay gate deferred | grid Poisson encoder | **cube population encoder** (24 facelets × 6 colors) |
| 2 Sensory → Prefrontal | ✅ EXP-015 | computed, **off policy path** | **on path** — PFC engaged for subgoals |
| 3 Sensory → Hippocampus (store) | ✅ EXP-017/020 | **bypassed** (`recall=False`) | **engaged** — store visited cube states |
| 4 Hippocampus → Prefrontal (recall) | ✅ EXP-020 | **bypassed** | **engaged** — recall feeds subgoal choice |
| 5 Prefrontal → Motor (action-enable) | ✅ EXP-016 | router computes gates, **policy bypasses** | **on path** — routed, task-phase gated |
| 6 Motor → Environment | ✅ spike-count winner | **replaced** by linear-head argmax | 6-action motor readout re-enters the loop |

**The v1 policy path, stated once:** `env → sensory encoder (frozen) → concept[64] → trainable nn.Linear
head → action`. Everything else runs, is recorded, and is visualized — but is not where the learnable
decision lives. **The Phase-3 goal is to widen this path** so pathways 2–5 carry causal weight.

**Full assembly** (`src/neuromorphic/brain.py`, `Brain`) is unchanged: instantiates all five regions +
`NeuromodBus`, runs window-batched, driven by the Gymnasium `GridWorldEnv`. Phase 3 adds a sibling **2×2
cube Gymnasium env** (mirror the `GridWorldEnv` structure) and extends the monitor JSONL schema for cube
state rather than reinventing it.

---

## 4. Training & plasticity — authoritative pointer + as-trained config

> **ADR-0001 is authoritative on training strategy, credit assignment, and all results.** This section
> summarizes the as-trained configuration and the Phase-3 plan; it does not restate the ADR.

### 4.1 As-trained config (v1, grid) — what actually learned

- **Objective:** surrogate-gradient **REINFORCE** (Monte-Carlo policy gradient) with a return baseline.
- **Learnable parameters:** a single **`nn.Linear` policy head** reading the frozen sensory `concept[64]` →
  action logits. The brain runs under `no_grad` as a **frozen feature extractor** (ADR Amdt 1).
- **Stabilizers (default-off, byte-identical when unused):** an **entropy bonus** (`entropy_beta`) and
  **per-episode advantage normalization** — together they eliminate the REINFORCE entropy-collapse failure
  mode (ADR Amdt 2).
- **Memory:** bypassed (`recall=False`).
- **Plasticity in the brain:** **none.** `Brain.learn(reward)` writes reward to the dopamine bus; the
  weight-update hook was never filled. R-STDP is a designed-but-deferred "taste" (EXP-021 on the PFC readout
  only).

### 4.2 What training established (the Phase-2 science)

1. **Readout capacity is not the cap** — an MLP head does not beat linear on the frozen concept (EXP-025, Amdt 2).
2. **Engaging the encoder lifts the cap** — displacement pre-training, ~2.5–5× held-out (EXP-026, Amdt 3).
3. **The engaged region specializes through learning** — decodable, distributed, causally robust (EXP-027, Amdt 4/5).
4. **The remaining cap is optimization-limited, not encoder-limited** — moderate input noise *doubled*
   held-out success (43% → 83% at σ=0.4; EXP-028, Amdt 6). Structural unit-drop declines monotonically
   (confirms the distributed code); Gaussian noise *inverts* the fidelity prediction — the policy is
   under-regularized.

### 4.3 Phase-3 training plan (scoping — full recipe at kickoff)

- **Regularization / exploration is the cheap first lever on the cap** (from EXP-028). Carry the
  noise-regularization trick into the cube baseline so the v1 baseline is the *best* v1 can do, not an
  under-tuned strawman.
- **Curriculum learning** — increasing scramble depth (1 → 2 → 3 moves); experience replay adapted for SNNs.
- **Engage the deferred regions** (memory, PFC, router) and **plausibly turn on plasticity** — follow the
  ADR migration path (recall → local plastic memory → reward-modulated router → Option-3 hybrid). Open
  kickoff decision: engage *before* attempting the cube, or run the v1 recipe first and let it fail
  informatively (recommendation: one quick v1-recipe baseline on 1-move scrambles to quantify the gap, then
  engage).
- **Monolithic same-neuron-count single-region baseline** — the key **"does regionalization help?"** control,
  stood up early so the question is answerable from the start.

---

## 5. What carries forward vs what is new (Phase-3 rotation)

**Carry forward — do NOT rebuild** (from the transition doc):
- The **five-region substrate** (`src/neuromorphic/`) — architecture and wiring are sound.
- The **training + analysis toolkit** — REINFORCE + entropy/advantage-norm, the generalization harness,
  checkpoints, and the probe/ablation stack (`src/neuromorphic/analysis/probes.py`,
  `src/neuromorphic/analysis/ablate.py`; REINFORCE in `src/neuromorphic/training/reinforce.py`).
- The **methodology** — paired per-seed tests, shuffle-null + PCA-matched controls, de-noising at n≥12,
  pre-registration with falsifiers, adversarial verification, and a committed `RESULTS.md` per experiment.
- The **observability contract** — the monitor JSONL trace + dashboard (extend the schema, don't reinvent).

**New for Phase 3** (per plan Step 3.1–3.4):

| Area | Phase 2 (grid) | Phase 3 (2×2 cube) |
|---|---|---|
| Environment | 5×5 GridWorldEnv, fully observable | 2×2 cube Gymnasium env, 24 facelets, 6 moves, curriculum |
| Sensory input | 4-int obs | 24 facelets × 6 colors — population coding |
| Motor | 4 actions | 6 actions (12 on a 3×3) |
| Hippocampus | bypassed | **engaged** — move history, visited states |
| Prefrontal | off policy path | **engaged** — subgoal decomposition |
| Router | spectator | task-phase-dependent gating |
| Training | single goal-set, REINFORCE on frozen encoder | curriculum learning, SNN-adapted replay, regularization-first |
| Baseline | random vs pre-trained encoder | **monolithic same-neuron-count single-region baseline** |
| Analysis | decodability + dropout | decoding, ablation, representational similarity analysis (RSA) |

---

## 6. Neuromodulatory bus — unchanged design, plasticity still deferred

`NeuromodBus` is as v2 §6. **ACh (live):** scales Motor lateral inhibition (sharper WTA where channels
compete). **Dopamine (hook):** reward / learning-enable, broadcast one-to-all; `learning_enabled =
dopamine ≥ learning_threshold`. The R-STDP three-factor rule (`Δw = β·(R−b)·e`, EXP-021 taste on the PFC
readout) remains **designed-but-deferred** — no plasticity consumes dopamine yet. Turning it on for memory
and routing is the Phase-3 stretch that completes the ADR's Option-3 hybrid.

---

## 7. Phase-3 checkpoint (the bar) + open kickoff decisions

**Checkpoint (aiming at):** (1) regionalized SNN solves 2×2 cubes from ≥1-move scrambles (3-move is
stretch); (2) comparison vs monolithic on ≥1 metric; (3) interpretability analysis; (4) clear documentation.

**Open decisions for the Phase-3 kickoff (~Jul 25):**
1. **Engage-first or fail-first?** Rethink the training strategy before the cube, or run the v1 recipe on
   1-move scrambles first and let it fail informatively? (Lean: quick v1 baseline to quantify the gap, then engage.)
2. **Cube encoder** — is there a supervised pre-training proxy analogous to grid displacement (e.g.
   distance-to-solved), or unfreeze end-to-end?
3. **How far to push plasticity** — recall-in-loop only, or all the way to reward-modulated memory + router
   (ADR migration steps 1–3)?

**Housekeeping to finish the rotation:**
- ~~Land EXP-028 → ADR Amdt 6 → `RESULTS.md`.~~ **DONE** (input noise regularizes; fidelity is not the cap).
- ~~Merge `week14` + `week15` to mainline, tag Phase 2 complete.~~ **DONE** (`c292cda`, tag `phase-2-complete`).
- ~~Promote `architecture-spec-v2.md` → v3.~~ **This document.**

---

*v3 — the as-trained reality folded into the substrate, retargeted at the 2×2 cube. Promote to v4 when the
cube forces real multi-region participation and/or plasticity, and the region tables above stop reading
"engaged (target)" and start reading "engaged (verified)."*
