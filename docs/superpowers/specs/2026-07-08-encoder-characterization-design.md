# EXP-027: Encoder Characterization (regional task-structure + distributedness)

Date: 2026-07-08
Status: Approved (design settled in session; synthesized via a design workflow), ready for implementation planning

## Motivation

EXP-026 (ADR-0001 Amendment 3) showed that pre-training the sensory encoder lifts held-out
navigation, but only as a *navigation-success* comparison (trained vs random encoder). This
experiment **characterizes the now-engaged encoder** to produce honest **regional-specialization
evidence** for the Jul-19 Phase-2 checkpoint, and to quantify how load-bearing / distributed the
learned representation is. It adds information beyond EXP-026 by analyzing the *internal
representation* across all five regions (plus the never-examined sensory `hidden[128]`) and a
behavioral load curve, rather than re-running a trained-vs-random navigation comparison.

Straight remove/freeze/random-vs-trained ablation is explicitly excluded: removing sensory =
head gets no input = chance; freezing = it is already frozen; random-vs-trained navigation = that
IS EXP-026. None of those are part of this design.

## The honesty backbone (load-bearing)

"Frozen spectator" is **not uniform**, and the design must say so:
- **Hippocampus** is genuinely bypassed under `recall=False` (`brain.step` only calls `self.hippo`
  when `recall=True`), so it produces **no recording at all** (`get_recording()` returns an empty
  dict) -> the probe **zero-fills** its row, and it is reported as "bypassed on the policy path
  (memory off, `W_rec` zero)", NOT as "lacks structure".
- **PFC / router / motor** still execute every step (they are driven by the forwarded concept), so
  they carry real, nonzero, frozen-weight activity that is at most a **lossy, attenuated image** of
  the concept's structure. The specialization claim is therefore a **gradient** (concentrated in
  sensory, attenuated downstream) reported with **effect sizes**, never a strawman "spectators are
  silent". Because spiking is lossy and these regions are frozen functions of the concept, their
  linear rate-decodability is **expected to attenuate** relative to the concept - an empirical
  tendency, NOT a hard bound (the data-processing inequality bounds an optimal decoder's mutual
  information, not a linear probe's R2 on time-averaged rates, and downstream rates depend on the
  full concept spike train, not the T-averaged concept rate the sensory probe reads).

Two controls neutralize the region-width confound (regions are 64/128/150/100/4 wide):
- **Shuffle-label null:** permute targets and refit to get an empirical per-region chance band.
- **PCA-matched dimensionality:** reduce every region's rate matrix to k=4 and k=8 PCs before
  probing, so a wide region cannot win by overfit capacity alone.

## Aim 1 - regional task-structure contrast (decodability matrix)

For each of the 12 EXP-026 seeds, pre-train a sensory encoder (`pretrain_sensory`, no head, no RL -
regime-independent), then over the encoder's **pre-training held-out state split** (the exact
`split_states(seed)` states the encoder never trained on - this, not the RL goal split, is what makes
"held-out decodability" honest) build a **region x target held-out decodability matrix**:

- Regions (recordings are keyed by the `Brain._regions` names, not `pfc`/`sensory_cortex`): sensory
  `concept[64]` and `hidden[128]`; prefrontal (PFC) `utility[4]` and internal `state[100]`; router
  `gate[4]`; motor `action[4]`; hippocampus `population[150]` (zero-filled because bypassed under
  `recall=False`, with one optional `recall=True` fairness pass).
- Targets: (a) **goal-relative displacement** `(gx-ax, gy-ay)` via a ridge linear probe -> held-out
  R2; (b) **per-action optimality** - four independent linear probes, one per action, predicting
  whether that action strictly reduces Manhattan distance to the goal (computed from the
  action->next-cell transition, `GridWorldEnv` deltas with grid clipping; `manhattan` alone is
  insufficient), scored by held-out accuracy/AUC. **Chance is each probe's shuffle-null band, not a
  literal 25%** - states have 1 or 2 optimal actions (2 when the goal is strictly diagonal), so a
  single 4-class softmax target is ill-defined.
- Each cell reported with its shuffle-null band and its PCA-matched (k=4, k=8) values.

Specialization is substantiated if the trained sensory concept's held-out decodability exceeds
every frozen spectator's for both targets, paired across seeds (win-fraction + sign test + bootstrap
CI, in the EXP-026 19/24 idiom), surviving both the shuffle-null band and the PCA-matched control.

## Aim 2 - distributedness / load-bearing (two corroborating readouts)

**(A) Representation geometry (correlational, encoder-only, cheap):** on the trained concept's
`[M, 64]` state x unit rate matrix - participation ratio / effective dimensionality
`(sum lambda)^2 / sum(lambda^2)`, per-unit displacement selectivity (single-unit R2) distribution,
the fraction of the 64 units needed to reach 90% of full-population decode R2, and keep-k / drop-k
R2 curves with units ranked by probe-weight magnitude.

**(B) Causal load curve (behavioral):** a `MaskedHead` wraps the trained linear head and zeroes k of
the 64 concept units before the head reads them (`head(x * mask)`), evaluated through the existing
`evaluate` / `greedy_action` rollout as held-out navigation degradation. Swept over k with
**random-k** (multiple draws per seed) and **importance-ordered top-k vs bottom-k** masks. Graceful
degradation + low participation ratio + small top/bottom gap => distributed/robust code; a cliff on
top-k => a few load-bearing units. (A) is correlational geometry, (B) is causal behavior; **agreeing
verdicts are the strong claim.**

## Checkpoint save (new infrastructure, enables Component B cheaply)

EXP-026 saved no trained models, so Component B would otherwise re-train per mask config. Add
lightweight serialization to the training path: after `run_generalization` trains, optionally save
`brain.sensory.state_dict()` and `head.state_dict()` (plus the config) to a checkpoint file keyed by
the run tag. Component B then **reloads** a trained `(brain.sensory, head)` per seed and runs the
many dropout-mask evals with no retraining. The ~1h training cost is paid **once** to mint the 12
checkpoints; every dropout sweep afterward is eval-only. This also unblocks all future work that
wants to reload trained models.

## Components

- `src/neuromorphic/analysis/probes.py` (NEW, brain/training-agnostic primitives):
  `region_rate_matrix(brain, states, *, region_key, recall, T)` (one batched
  `brain.step(states[M,4], record=True, recall=False)` -> `recordings[region_key][key].mean(dim=0)`
  `[M,N]`; region keys are the `Brain._regions` names `sensory`/`hippocampus`/`prefrontal`/`router`/
  `motor` - NOT `pfc`/`sensory_cortex`; **zero-fills a `[M, width]` row when a region produced no
  recording** - the bypassed hippocampus under `recall=False`, mirroring `_pad_bypassed_recordings`;
  the sensory path delegates to `pretrain.concept_rate_batch` so it matches the head input exactly);
  `task_targets(states, grid_n)` (displacement via `pretrain.displacement_target`; per-action
  optimality via the `GridWorldEnv` action->next-cell deltas + grid clipping + `gridworld.manhattan`);
  `ridge_probe` (**explicit L2-penalized** - augmented `[X; sqrt(lambda)*I]` through `torch.linalg.
  lstsq`, or normal equations `(XtX + lambda*I)^-1 XtY`, with a **shared lambda across regions**,
  stated - NOT bare `lstsq`, which is unregularized OLS); `peraction_probe` (four linear probes ->
  per-action held-out accuracy/AUC); `shuffle_null`; `pca_reduce(X, k)`; `participation_ratio(X)`;
  `unit_importance(X, Y)`; `keepk_curve(X, Y, order)`.
- Checkpoint save: a small addition to `training/generalization.py` (a `checkpoint_dir` /
  `save_checkpoint` option on `GenConfig` / `run_generalization`, default off = byte-identical) plus
  a `load_trained(path) -> (brain, head)` helper (in `pretrain.py` or a new `checkpoints.py`).
- `experiments/027_encoder_characterization/probe.py` (Component A): per seed, pre-train encoder,
  extract every region's rate matrix, fit displacement + optimal-action probes with shuffle-null and
  PCA-matched controls, compute the Aim-2A geometry. Encoder-only, runs in seconds/seed.
- `experiments/027_encoder_characterization/dropout_eval.py` (Component B): `MaskedHead`,
  `random_mask(k, seed)`, `importance_mask(probe_weights, k, mode)`, `masked_evaluate` reusing
  `evaluate` + `greedy_action` unchanged; loads the trained `(brain.sensory, head)` checkpoint. The
  importance ordering **must be computed from the checkpoint encoder that is being masked** (compute
  the concept probe weights on the reloaded encoder, or serialize the probe weights alongside the
  checkpoint) - not from a separately-pretrained encoder - so the ranked top-k/bottom-k units are the
  units actually masked. Component B depends on the checkpoint step running first.
- `experiments/027_encoder_characterization/run.py` + `aggregate.py` (driver): fork the EXP-026
  `ProcessPoolExecutor` fan-out over the 12 seeds; aggregate keyed by `(region, target)` and
  `(mask_policy, k)` with `paired_delta` / `win_count` / sign test; emit `027_summary.json` and
  `027_table.md` (decodability matrix with paired win-fractions; geometry table; dropout curve).
- `tests/training/test_probes.py` (NEW): trained-concept displacement R2 exceeds every spectator's
  and clears its own shuffle-null band; per-action optimality decodable above its shuffle-null band;
  probes deterministic under fixed seed; `MaskedHead(k=0)` reproduces the unmasked eval exactly;
  `region_rate_matrix` returns a zero `[M,150]` row for the bypassed hippocampus (`recall=False`)
  rather than raising; checkpoint round-trip (`save` then `load_trained` reproduces identical eval).
- `src/dashboard/decodability_panel.py` (OPTIONAL, checkpoint figure): a NEW sibling panel (does not
  edit `render_dashboard`): region x target decodability heatmap, participation-ratio gauge, keep-k
  curve, dropout curve. Build only if the Jul-19 checkpoint wants a rendered figure.

## Decisions (settled)

- **Component B included**, with the **checkpoint-save** so it reloads rather than re-trains.
- **Probe width:** canonical output keys + sensory `hidden[128]` + PFC `state[100]` (cheap, new info).
- **Targets:** displacement + per-action optimality (four per-action probes) only.
- **Seeds:** reuse EXP-026's exact 12 (0-11). Component A is regime-independent (12 encoders, not 24
  seed-regime cells). Component B uses a single regime (shaped) for the trained heads.
- **Omit** a random-encoder decodability reference row (keeps the within-brain sensory-vs-spectator
  contrast constraint-clean).
- **Defer** the intermediate-dose RL sweep (epochs 25/50/100) - the dropout curve gives the causal
  load-bearing verdict without ~72 extra 600-episode runs that reproduce EXP-026 endpoints.

## Success criteria

The specialization claim holds if, paired across the 12 seeds, the trained sensory concept's held-out
decodability for **both** displacement and per-action optimality exceeds every frozen spectator's by a
margin that survives (i) each region's shuffle-null band and (ii) the PCA-matched (k=4, 8) control,
reported as a paired win-fraction + sign test with per-seed spread. The distributedness verdict is
delivered if the geometry readouts and the causal dropout-on-navigation curve **agree** on
distributed-vs-brittle. The result is honest only if hippocampus is reported as zero-filled (bypassed)
and PFC/router/motor as an attenuated lossy image of concept (a gradient with effect sizes).

## Risks

- Decodability is **correlational** - a region decoding displacement does not prove it is *used* (the
  head reads only concept). Phrase Aim 1 as "carries linearly-accessible task structure"; Component B
  is the causal complement for the encoder itself.
- **Spectator inheritance:** PFC/router/motor are deterministic frozen functions of concept -> may
  decode somewhat above chance; report the gradient, not a binary. Only hippocampus is zero-filled
  (bypassed under `recall=False` it produces no recording, so the probe fills a zero row - it is not
  evidence of "no structure", just of being off the policy path).
- **Width heterogeneity** makes raw cross-region R2 non-comparable; the PCA-matched control mitigates
  but adds a modelling choice (k) to defend.
- **Eval-time concept masking is off-distribution** for a head trained on the full 64-d concept, so
  the dropout curve conflates "representation not distributed" with "head not robust to masking"; the
  top-vs-bottom-k contrast controls for this but does not fully remove it.
- **Variance:** EXP-026's high per-run variance means probe/geometry metrics are noisy across 12
  seeds - needs the held-out state split, shared ridge regularization, shuffle null, and paired stats
  to be trustworthy.
- **Linear-only:** probes capture only linearly-decodable structure; partially mitigated by probing
  `hidden[128]` and reporting participation ratio, but the claim is bounded to linear accessibility.

## Scope boundaries (explicitly out)

- No remove/freeze/random-vs-trained-navigation ablation (forbidden by the task constraint,
  redundant with EXP-026).
- No encoder-weight lesioning, retraining, or gradient-saliency on `fc1`/`fc2`; distributedness is
  measured on activity + eval-time masks, leaving frozen weights untouched.
- No nonlinear probes / full RSA (linear probes + participation ratio for v1).
- No MLP-head variants (v1 ground truth is the linear head).
- No editing `render_dashboard` or the JSONL trace contract (the optional figure is a new sibling
  panel only).
- No absolute-position / 25-class cell-identity targets in v1.
