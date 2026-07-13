# EXP-028 — Sensory-Code Ablation (dose-response) — Design Spec

- **Status:** DRAFT / scoped (not yet built). Working-tree only, uncommitted.
- **Date:** 2026-07-10 (Week 14, Session 3 candidate)
- **Depends on:** EXP-026 (encoder pre-training lifts the cap; ADR-0001 Amdt 3), EXP-027 (sensory region
  specializes AND its code is distributed; ADR-0001 Amdt 4). Component B closed the *causal
  distributedness* question via masking a **trained** head. This ablation asks the complementary
  question at the **policy-learning** level.

## 1. Motivation — what NEW thing does this prove?

EXP-027 was diagnostic on a *frozen, already-trained* policy: it showed the concept code carries task
structure (Aim 1) and that masking units degrades a fixed head gracefully (Aim 2B). Neither re-trains
the policy under degradation. The open question ablation answers:

> If we **degrade the sensory concept and then let the linear head train against that degraded code**,
> how much does held-out navigation fall, and how gracefully? I.e. is the *pre-trained sensory code* a
> load-bearing input to policy learning, in a dose-dependent way?

This is the classic lesion/dose-response that the original Week-14 calendar item asked for — now
*meaningful* because EXP-026/027 established the sensory code both matters and is distributed. A flat
(no-effect) curve would undercut the EXP-026 claim; a smooth dose-response *quantifies* how much of the
lift is attributable to concept fidelity.

**Not redundant with prior work (the EXP-027 non-redundancy argument, updated):**
- Removing sensory entirely = chance (trivial, already known).
- Freezing = it is already frozen.
- Random-vs-trained encoder = that is EXP-026 (binary: 0% vs 100% of the code).
- **This ablation = the *dose* axis between those two endpoints**, measured through re-training, not
  through a fixed head. That is the new information.

## 2. Hypothesis (pre-registered)

Held-out navigation success is a **monotone decreasing, graceful** function of sensory-code corruption:
small corruption ≈ EXP-026 pre-trained baseline (~40–45%), full corruption → the random-encoder floor
(~8–17%, the EXP-026 random arm). Graceful (not a cliff) because EXP-027 showed the code is distributed.

**Falsifiers:** (a) flat curve = concept fidelity does not drive the lift (contradicts EXP-026 causal
story) → investigate. (b) cliff at tiny corruption = brittleness (contradicts EXP-027 Aim 2)
→ reconcile against Component B before publishing.

## 3. Design

### 3.1 Ablation operators (the "dose")
Applied to the **pre-trained, frozen** sensory concept output, *before* the trainable head reads it, for
the whole of head training + eval (deterministic per seed):

1. **Additive Gaussian noise** on the concept rate: `c' = c + N(0, sigma^2)`, dose = sigma over a grid
   e.g. `{0, 0.05, 0.1, 0.2, 0.4, 0.8}` (0 = EXP-026 baseline anchor). *Primary operator* — continuous,
   smooth, standard.
2. **Unit dropout (structural)** — zero a fraction `p` of concept units for the whole run, dose =
   `p in {0, 0.1, 0.25, 0.5, 0.75, 0.9}`. Two sub-modes reusing EXP-027 `unit_importance`:
   `random` (neutral) and `top` (drop most-important-first — the adversarial dose). Bottom-k optional.

Rationale for two operators: noise degrades *fidelity* continuously; unit-drop degrades *dimensionality*
structurally. Agreement across both = robust claim; divergence is itself informative.

### 3.2 Protocol
- Base config = EXP-026 winning arm: `pretrain_sensory=True`, shaped, 600 episodes, grid 5,
  held-out 10, **12 seeds** (de-noise lesson from EXP-026 — n=5 lied; do not go below 12).
- For each (operator, dose, seed): pretrain+freeze sensory → apply ablation wrapper on the concept →
  train the linear head fresh → eval held-out nav success (+ train success, optimality, gap).
- **Paired by seed** across the dose axis (same seed → same pretrained encoder → same goal split), so
  the analysis is per-seed paired deltas vs the dose=0 anchor, not cross-seed means. (EXP-026 lesson.)
- Anchors to sanity-check endpoints land where expected:
  - dose=0 should ≈ EXP-026 pre-trained held-out (~40–45%).
  - max dose (Gaussian sigma high / p→0.9) should approach the EXP-026 **random-encoder** arm (~8–17%).
  If endpoints do not bracket, the operator is miscalibrated — fix before reading the middle.

### 3.3 Metric & analysis
- Primary: held-out nav success vs dose (mean + per-seed paired band across 12 seeds).
- Report the **dose at which success drops halfway** between baseline and floor (an "ED50"-style number)
  — a single interpretable summary of how much fidelity the policy needs.
- Secondary: train-vs-held gap vs dose (does corruption hurt generalization more than training?).
- Ordering check for unit-drop: success(random) ≥ success(top) at matched p (importance is real).

## 4. What to build

Small — leans hard on existing pieces, mirrors EXP-027's structure:
- `analysis/ablate.py` (or extend `probes.py`): `AblatedConcept` wrapper — a callable that perturbs the
  concept rate (gaussian / unit-drop) with a fixed per-seed generator. Reuse
  `probes.unit_importance` for the top-k ordering (already built + tested).
- A `concept_ablation` hook in the generalization harness, **default-off / byte-identical when off**
  (same discipline as the `pretrain_sensory` and `entropy_beta` hooks — do not perturb the existing
  code path when the feature is disabled). Likely a `GenConfig.ablation: AblationSpec | None = None`
  consulted only inside the forward that reads the concept.
- `experiments/028_sensory_ablation/{run.py, aggregate.py}` — dose×seed driver
  (`ProcessPoolExecutor`, `set_num_threads(1)` per worker, `--workers` flag — same as 026/027;
  respect Mike's "don't pin the gaming PC" rule, offer reduced workers), writing `028_curve.md` +
  `028_summary.json`.
- Tests: wrapper is identity at dose=0 (byte-identical guard), monotone-ish sanity on a toy, mask
  plumbing, endpoint anchors. Follow EXP-027's rigor — the whole-branch review caught 10 defects there;
  budget for a spec-verification pass + whole-branch review again.

## 5. Cost / logistics
- Grid: 2 operators × ~6 doses × 12 seeds ≈ 144 head-trainings (Gaussian) + unit-drop set. Each head
  train ≈ the EXP-026 per-seed cost. This is **bigger than Component B** — plan a multi-hour run;
  strong candidate for the laptop-over-SSH pattern established 2026-07-09, or trim the dose grid first
  (a 4-dose pilot at 12 seeds to confirm the shape, then fill in).
- Reuses the checkpoint-mint path from EXP-027 Component B (`checkpoint_path`) if we want to ablate a
  *cached* pretrained encoder rather than re-pretrain per dose — **recommended**, cuts cost ~6x: mint 12
  pretrained encoders once, then sweep doses re-training only the cheap linear head. Wire `run.py` to
  load a cached encoder per seed and only re-train the head.

## 6. Resolved decisions (2026-07-12)
- **Operators: BOTH from the start** — Gaussian-noise (continuous fidelity dose) AND structural
  unit-drop (random + top-k). Agreement across the two = the stronger claim.
- **Cache encoders once (6x cheaper)** — mint 12 pretrained+frozen encoders once (reuse EXP-027
  `checkpoint_path`), then for each (operator, dose, seed) reload the cached encoder and re-train only
  the cheap linear head against the ablated concept. This is what makes the full sweep feasible overnight.
- **Full 6-point dose grid** at 12 seeds in one run (no pilot): Gaussian `sigma in {0,.05,.1,.2,.4,.8}`;
  unit-drop `p in {0,.1,.25,.5,.75,.9}` x {random, top}.
- **Separate EXP-028** (`experiments/028_sensory_ablation/`), its own writeup + ADR-0001 Amendment 5.

Run-size sanity: 12 encoder pretrains (once) + head-retrains = Gaussian 6x12=72 + unit-drop 6x2x12=144
= ~216 head-trainings. Head-train (linear head, 600 eps) is the cheap leg; multi-hour, overnight on the
laptop via the established SSH pattern, `--workers` tuned to spare the gaming CPU.
