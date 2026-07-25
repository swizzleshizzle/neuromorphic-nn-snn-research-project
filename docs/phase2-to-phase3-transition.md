# Phase 2 → Phase 3 Transition Scoping

**Date:** 2026-07-13 (Week 15) · **Status:** scoping (not a full Phase-3 spec — that comes at the
Phase-3 kickoff, ~Jul 25). Grounds the rotation in the Phase-2 honest assessment
(`docs/phase2-honest-assessment.md`) and the project plan's Phase 3 section.

Phase 3 = the **2×2 Rubik's Cube challenge** (plan timeline Jul 25 – Sep 27). This doc answers: what do
we carry forward, what is genuinely new, and what strategic decision the honest Phase-2 result forces
before we start.

## The strategic pivot (read this first)

Phase 2's honest conclusion: *we built and characterized the five-region substrate, but only **one**
region (sensory) is on the policy path, the brain never learns (frozen extractor + trainable linear
head), and memory is bypassed.* On a trivial fully-observable 5×5 grid, that sufficed.

**The cube breaks every one of those shortcuts:**
- It is **not trivially reactive** — solving needs lookahead / subgoal structure → the PFC has to earn
  its place.
- It **benefits from move history** (avoid cycles, recognize visited states) → the hippocampus/memory
  can no longer be bypassed.
- The action space **grows** (4 → 6 moves; revised from 12 on 2026-07-24, see the cube-env spec §1) and
  the sensory input is **richer** (24 facelets × 6 colors) → a single frozen encoder + linear head is
  very unlikely to reach the checkpoint.

So Phase 3 is where the v1 architecture's deferred pieces (more regions on the path; plausibly real
plasticity) stop being optional. **Open decision for the kickoff:** do we rethink the training strategy
(engage more regions / turn on R-STDP) *before* attempting the cube, or discover the need empirically by
first trying the v1 recipe and letting it fail informatively? (Recommendation: one quick v1-recipe
baseline on 1-move scrambles to quantify the gap, then engage.)

**EXP-028 sharpens this.** The remaining grid-nav cap turned out to be **optimization-limited, not
encoder-limited** — moderate input noise alone doubled held-out success (43%→83%), evidence the
frozen-extractor + REINFORCE policy is under-regularized (ADR Amendment 6). Implication for the cube:
**regularization / exploration is a cheap first lever** to squeeze before concluding "we need more
regions / plasticity." Bake it into the v1-recipe baseline (carry the noise-regularization trick over),
so the baseline is the *best* v1 can do, not an under-tuned strawman — a fairer measure of the real gap.

## What carries forward (do NOT rebuild)

- **The five-region substrate** — SensoryCortex, Hippocampus, Prefrontal, ThalamicRouter, MotorCortex,
  NeuromodBus (`src/neuromorphic/`). Architecture and wiring are sound (spec §1–§6).
- **The training + analysis toolkit** — REINFORCE + entropy/advantage-norm (`training/reinforce.py`),
  the generalization harness, checkpoints, and the probe/ablation analysis stack
  (`analysis/probes.py`, `analysis/ablate.py`) that made specialization measurable.
- **The methodology** — paired per-seed tests, shuffle-null + PCA-matched controls, de-noising at n≥12,
  pre-registration with falsifiers, adversarial verification of specs and results, and the new standing
  habit: a committed `RESULTS.md` per experiment.
- **The observability contract** — the monitor JSONL trace + dashboard; extend the schema for cube
  state rather than reinventing it.

## What is new for Phase 3 (per plan Step 3.1–3.4)

| Area | Phase 2 (grid) | Phase 3 (2×2 cube) |
|---|---|---|
| Environment | 5×5 GridWorldEnv, fully observable | 2×2 cube Gymnasium env, 24 facelets, 6 moves, curriculum 1→2→3-move scrambles |
| Sensory input | 4-int obs (ax,ay,gx,gy) | 24 facelets × 6 colors — population coding (plan L13) |
| Motor | 4 actions | 6 actions (expand MotorCortex; 12 on a 3×3) |
| Hippocampus | bypassed (`recall=False`) | **engaged** — track move history, recognize visited states |
| Prefrontal | off policy path | **engaged** — subgoal decomposition |
| Router | spectator | task-phase-dependent gating |
| Training | single goal-set, REINFORCE on frozen encoder | **curriculum learning** (increasing scramble depth), experience replay adapted for SNNs |
| Baseline | random vs pretrained encoder | **monolithic same-neuron-count single-region baseline** (the key "does regionalization help?" comparison) |
| Analysis | decodability + dropout | decoding, ablation, representational similarity analysis (RSA) |

## Phase 3 checkpoint (the bar we are aiming at)

> (1) Regionalized SNN solves 2×2 cubes from at least 1-move scrambles (3-move is stretch). (2)
> Comparison vs monolithic on at least one metric. (3) Interpretability analysis. (4) Clear documentation.

## Suggested first moves at Phase-3 kickoff (not started yet)

1. **Build the cube env** (Gymnasium, 24-facelet state, 6 moves, scramble-depth curriculum) + tests —
   mirror the GridWorldEnv structure.
2. **v1-recipe baseline** on 1-move scrambles to quantify how far the frozen-extractor + linear-head
   approach gets — an informative failure that motivates engagement.
3. **Decide the engagement strategy** from that baseline (more regions on the path / plasticity), and
   stand up the **monolithic baseline** early so the "regionalization helps" question is answerable from
   the start.

## Housekeeping to finish the Phase-2 → 3 rotation

- ~~Land EXP-028 → ADR Amendment 6 → its own `RESULTS.md`.~~ **DONE** (9602265): input noise regularizes;
  fidelity is not the cap. Threaded into the honest assessment (§ limitation 5, 12) and above.
- Merge `week14-encoder-characterization` + `week15-phase2-closeout` to mainline and tag Phase 2 complete.
- ~~Promote `architecture-spec-v2.md` → v3 (fold in the as-trained config) as the first Phase-3 doc, since
  the region tables change materially for the cube anyway.~~ **DONE** (2026-07-17): `docs/architecture-spec-v3.md`
  — as-trained config folded into the region tables, retargeted at the 2×2 cube; v2 superseded.
