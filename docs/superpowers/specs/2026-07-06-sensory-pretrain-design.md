# EXP-026: Sensory Pre-Training (engage the encoder)

Date: 2026-07-06
Status: Approved (design settled in session), ready for implementation planning

## Motivation

EXP-025 (ADR-0001 Amendment 2) concluded that the ~30-50% held-out navigation cap is set by
the **frozen sensory encoder**, not the linear readout: a fair, collapse-free MLP head did not
beat the linear head. In v1 the encoder is random-init and frozen, so the policy head has only
ever read a **random projection** of the state. This experiment engages the encoder: pre-train
it on a supervised state-encoding task so the 64-d concept becomes a decodable state code, then
freeze it and re-run the generalization eval to see whether the cap finally moves.

This is the Phase-2->3 bridge and the first change that lets a region *specialize through
learning* — which is also what makes the deferred ablation studies meaningful.

## The pre-training objective (settled)

**Goal-relative displacement.** Pre-train a readout on the concept rate to predict the vector
pointing from agent to goal, `(goal_x - agent_x, goal_y - agent_y)`, normalized to `[-1, 1]`
by dividing by `grid_n - 1`. This is exactly the quantity a navigation policy needs; if a linear
map can then decode "which way is the goal" off the frozen concept, the cap should lift. It
directly targets the EXP-024 finding (the linear head could not decode goal-conditioned
direction from the random-projection concept). The readout is a scratch `Linear(concept -> 2)`
used only to shape the encoder; it is discarded after pre-training.

## Two-stage protocol (settled)

**Stage 1 — did the encoder learn? (cheap gate).** After pre-training, measure the displacement
readout's error on **held-out grid states**. If the pre-trained concept cannot linearly predict
goal-direction on held-out states, pre-training failed and Stage 2 is skipped. Report the
random-encoder readout error as the reference point (the EXP-025 hypothesis says a random
concept does *not* expose displacement linearly). Gate passes when the pre-trained held-out
displacement error is clearly below the random-encoder baseline and low in absolute terms
(target: held-out mean-absolute error < ~0.75 normalized units, i.e. the readout points the
right general direction).

**Stage 2 — did it lift the cap? (the real test, ~1h).** Load the pre-trained encoder frozen,
train the linear policy head, run the generalization harness: paired 5 seeds, shaped + sparse,
held-out success. **Success = the pre-trained encoder's held-out mean clears the random-encoder
linear band** (EXP-025 beta=0 baseline: shaped 23%, sparse 27% held-out).

Verdict logic:
- Clears the band -> engaging the encoder lifted the cap. Option B validated; regions now
  specialize through learning -> ablation studies become meaningful (next work).
- Does not clear the band -> the bottleneck is deeper than the sensory encoder (the whole
  frozen-feature-extractor framing, or the task) -> rethink.

## Integration (settled)

Pre-train **in-harness, per seed**, so everything stays paired and directly comparable to the
EXP-025 baseline at each seed:

- Add `pretrain_sensory: bool = False` (plus `pretrain_epochs`, `pretrain_lr`) to `GenConfig`.
  Default `False` leaves `run_generalization` byte-identical (EXP-023/024/025 reproducible).
- When `pretrain_sensory=True`, `run_generalization` pre-trains `brain.sensory` (seeded by
  `cfg.seed`) *before* the RL loop. The existing `no_grad` RL path then freezes it automatically.
- The summary records `pretrain_sensory` and the Stage-1 gate metric (held-out displacement
  error) alongside the existing eval block.

## Components

`src/neuromorphic/training/pretrain.py` (new):
- `displacement_target(obs, grid_n) -> Tensor[B, 2]` — normalized `(gx-ax, gy-ay) / (grid_n-1)`.
- `enumerate_states(grid_n) -> Tensor[M, 4]` — all `(agent, goal)` pairs with `agent != goal`
  (600 for 5x5).
- `split_states(states, frac_heldout, seed) -> (train, heldout)` — deterministic state split
  for the Stage-1 gate.
- `concept_rate_batch(sensory, obs, grid_n, T, generator) -> Tensor[B, concept]` — encode a
  batch of states (`encode_gridworld` -> `sensory.forward` **with grad**) and return the
  mean-over-`T` concept firing rate. (The existing `brain.step` wraps the encoder in `no_grad`;
  pre-training must call the encoder directly so gradients reach `fc1`/`fc2`.)
- `pretrain_sensory(sensory, *, grid_n, epochs, lr, frac_heldout, seed, T, generator) -> dict` —
  trains a scratch `Linear(concept -> 2)` readout on `concept_rate_batch` toward
  `displacement_target` with MSE + Adam over the train states, backprop updating the encoder
  weights (`fc1`, `fc2`) in place. Returns `{"heldout_disp_error": float, "train_disp_error":
  float, "epochs": int}`. The readout is not returned (discarded).

`src/neuromorphic/training/generalization.py`:
- `GenConfig` gains `pretrain_sensory: bool = False`, `pretrain_epochs: int = 200`,
  `pretrain_lr: float = 1e-3` (values are starting points; the plan may tune).
- `run_generalization`: if `cfg.pretrain_sensory`, call `pretrain_sensory(brain.sensory, ...)`
  after building `brain`, before the RL loop; store the returned gate dict under
  `summary["pretrain"]`.

`experiments/026_sensory_pretrain/` (new):
- `run.py` — runs the paired sweep with `pretrain_sensory=True` across `{shaped, sparse} x 5
  seeds` (linear head only; the head-type question is closed), tagged so outputs do not collide
  with EXP-025, reusing the EXP-025 aggregator for the held-out table. Prints the Stage-1 gate
  metrics per seed and the Stage-2 aggregate table.

## Data flow

Per seed: build `Brain(seed)` -> `pretrain_sensory` updates `brain.sensory` weights (Stage 1,
gate recorded) -> RL loop trains the linear head on random train goals with the now-frozen
encoder -> greedy eval on train vs held-out goals (Stage 2) -> summary carries both the
`pretrain` gate block and the `eval` block. Determinism: `cfg.seed` drives the encoder init, the
state split, the Poisson generator, the goal split, and the head training, so a fixed seed is
fully reproducible and paired against the EXP-025 random-encoder run.

## Testing

- `displacement_target`: correct normalized vectors for known obs (e.g. agent (0,0) goal (4,4)
  -> `(1.0, 1.0)`; agent (4,0) goal (0,0) -> `(-1.0, 0.0)`).
- `enumerate_states`: length `grid_n**2 * (grid_n**2 - 1)`, no `agent == goal`.
- `split_states`: disjoint, sized, deterministic by seed.
- `concept_rate_batch`: returns `[B, concept]`, and is differentiable (a backward pass populates
  `sensory.fc1.weight.grad`) — this guards the "encoder must receive gradient" requirement.
- `pretrain_sensory`: reduces train displacement error over epochs; changes `sensory.fc1`/`fc2`
  weights; deterministic by seed; returns the gate dict with finite errors.
- Integration: `GenConfig().pretrain_sensory is False`; `pretrain_sensory=True` records a
  `pretrain` block in the summary and yields a `brain.sensory` that differs from the random-init
  encoder; `pretrain_sensory=False` leaves `run_generalization` output byte-identical.
- Keep the existing suite green (228 tests).

## Deliverable

- The pre-training module + harness integration + EXP-026 runner, tests green.
- EXP-026 outputs: Stage-1 gate metrics (per-seed held-out displacement error vs random
  baseline) and the Stage-2 held-out success table (pre-trained vs the EXP-025 random-encoder
  band), plus the verdict.
- Session-3 findings appended to the Week-13/14 obsidian note; ADR-0001 amended only if the
  result is decisive.

## Scope boundaries (explicitly out)

- No unfreeze-during-RL (end-to-end fine-tuning) — that is the other Option-B flavor and a
  separate experiment; this one isolates pre-train-then-freeze for clean attribution.
- No ablation studies yet — they follow only after a region is shown to specialize.
- No new regions and no PFC/motor re-integration — one lever at a time.
- No MLP head — the head-type question is closed (linear head, per EXP-025).
