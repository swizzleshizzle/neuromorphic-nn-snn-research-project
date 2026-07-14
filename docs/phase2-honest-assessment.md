# Phase 2 — Honest Assessment (checkpoint criterion 5)

**Date:** 2026-07-13 (Week 15, final week of Phase 2) · **Method:** adversarial checkpoint audit
(evidence-gatherer + skeptic per criterion + devil's-advocate pass), findings cross-checked against the
repo, ADR-0001, experiments 024–028, and the vault. This document is deliberately unsparing; it is the
record we hold ourselves to before declaring Phase 2 done.

## Checkpoint scorecard

| Criterion | Grade | Note |
|---|---|---|
| (1) Architecture spec document | **Partial** | Regions + wiring spec is excellent and code-accurate; training strategy lives only in ADR-0001, and `architecture-spec-v2.md` was stale on training until the 2026-07-13 supersession banner. |
| (2) Working multi-region SNN on grid navigation | **Partial** | Assembled, wired, navigates held-out goals — but only 1 of 5 regions is on the policy path and the brain itself does not learn. |
| (3) Monitoring dashboard | **Met** | Renders real multi-region traces; writer↔reader contract verified. Caveat: it is a post-hoc replay tool, not a live monitor (streaming sinks are stubs). |
| (4) Evidence of regional specialization | **Partial→strong** | Decodability + geometry + causal dropout all now committed (EXP-027 RESULTS.md, ADR Amdt 5). "Through learning" borrows EXP-026's trained-vs-random contrast; a residual upstream-position confound remains. |
| (5) Honest assessment | **This document** | — |

**Verdict:** Phase 2 delivered an *honestly-characterized frozen-feature-extractor brain with a
demonstrably specialized sensory encoder* — **not** a learning multi-region brain. That is a real,
defensible result for a research phase whose stated ethos is honest assessment; it is not the "solved"
story a generous reading might tell.

## What genuinely works

- **The five-region spiking architecture is real and runs.** SensoryCortex, Hippocampus, Prefrontal,
  ThalamicRouter, MotorCortex + NeuromodBus are assembled, wired per spec, and produce a valid grid
  action each step with signal flowing through all regions (`brain.py`; `tests/integration/`).
- **It navigates held-out goals, and generalizes rather than memorizes** (EXP-024): small/negative
  train-vs-heldout gaps. Engaging the sensory encoder (EXP-026) lifts held-out success ~2.5–5× over a
  random encoder (paired 19/24 seed-cells, +28/+34 pts).
- **A region specializes through learning** — the strongest scientific result of the phase. The trained
  sensory concept decodes task structure (disp R² 0.86–0.90) and beats every non-sensory spectator in
  100% of 12 seeds, with shuffle-null + PCA-matched controls; the code is distributed (PR ~24/64) and
  causally robust (graceful dropout curve). See `experiments/027_encoder_characterization/RESULTS.md`.
- **Observability exists and is honest.** The dashboard renders real traces; the monitor↔reader data
  contract is key-matched and tested; the "honest trace" work stopped the dashboard from flattering a
  degenerate policy.
- **The methodology is rigorous.** Paired per-seed tests, shuffle-null bands, PCA-matched controls,
  de-noising when n=5 misled, pre-registration with falsifiers, and adversarial verification of both
  specs and results.

## What does NOT work — the limitations we must disclose

### Structural (the deliberate v1 design ceiling — per ADR-0001, but must be stated plainly)
1. **The brain does not learn.** Only a single `nn.Linear` head trains (REINFORCE); the brain runs under
   `no_grad` as a frozen extractor, and `Brain.learn()` only writes to the dopamine bus — the plasticity
   hook was never filled. *Reward never changes a synapse in the brain.* (`brain.py:192-202`; ADR-0001.)
2. **Four of five regions are inert spectators on the policy path.** The action is
   `sensory concept → linear head`; PFC/router/motor outputs are computed and visualized but discarded,
   and decodability collapses from PFC state[100] R² 0.76 → utility[4] 0.03 → router/motor ~0. Only 1 of
   5 regions causally drives behavior.
3. **The hippocampus/memory is fully bypassed and untrained** (`recall=False`). Store/recall and the
   Hebbian attractor are built but never exercised; memory contributes nothing.
4. **R-STDP and biological plausibility are designed but deferred.** Only an eligibility-trace "taste"
   (EXP-021) exists; credit assignment is non-biological global backprop, chosen for expediency. The
   biological-plausibility goal is explicitly unmet.

### Performance / statistics
5. **The navigation ceiling is low.** Best (pretrained-encoder) held-out success is only ~42–45%; a
   nonlinear MLP head does not beat linear, so the frozen encoder is a hard representational cap.
6. **It fails many *trained* goals too** (~53% train success) — not merely a generalization gap.
7. **Extreme per-seed variance** (0–100% swings). Every positive claim is an average-only claim; a single
   run is unreliable.
8. **Statistical fragility.** The EXP-026 verdict flipped between n=5 (read as failure) and n=12; the
   EXP-025 "cap" numbers rest on n=5 and were downgraded to "no MLP advantage," not a precise figure.
9. **EXP-026's pre-registered "clears the band" criterion was NOT met**; the positive verdict rests on a
   paired per-seed sign test chosen (correctly) as the better analysis — flagged transparently, not hidden.

### Scope / open threads
10. **The task is a trivial, fully-observable 5×5 grid** where a reactive policy suffices and memory is
    provably unnecessary — nothing here exercises memory, routing, or partial observability.
11. **The engaged region learned via a hand-picked supervised proxy** (goal-relative displacement), then
    was frozen; it never adapts to the actual RL reward. Whether a richer target or end-to-end unfreezing
    raises the ceiling is untested.
12. **The second causal test (EXP-028 re-training dose-response) is still running** — its result (whether
    degrading the concept and re-training the head drops navigation) is not yet in.

### Records hygiene (being fixed this week)
13. The Component B result had lived only in a vault note (gitignored, laptop-only) while the committed
    record said "pending" — now corrected via `RESULTS.md` + ADR Amendment 5. The broader Phase-2 work
    still sits on an unmerged branch; merging to mainline is part of the Phase-2 → 3 rotation.

## What this means for Phase 3

Phase 3 (Rubik's cube) will *expose* limitations 1–3 immediately: a 2×2 cube is not fully observable in
the trivial sense, benefits from move-history memory (hippocampus), and needs subgoal decomposition (PFC)
— i.e. the four "inert" regions have to actually earn their place, and a frozen-extractor + linear-head
policy will very likely not suffice. The honest Phase-2 conclusion — *we built and characterized the
substrate, but only engaged one region and never turned on plasticity* — is precisely the gap Phase 3
must close. Carry forward the methodology (paired seeds, controls, adversarial verification, committed
RESULTS.md); carry forward the substrate; do **not** carry forward the assumption that the architecture
alone buys multi-region behavior.
