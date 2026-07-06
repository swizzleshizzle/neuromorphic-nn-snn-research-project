# Entropy Bonus for REINFORCE (unblock EXP-025)

Date: 2026-07-01
Status: Approved (design settled in session), ready for implementation planning

## Motivation

EXP-025 (head-capacity probe) was inconclusive: the MLP head is confounded by entropy
collapse. Half the MLP seeds saturate to a zero-entropy one-hot policy by ~episode 150 and
die (reach 0%); the linear head never collapses. The mechanical result "MLP did not beat
linear" is therefore invalid, because the MLP never got a fair test.

The root cause is concrete: `train_episode` already **computes** per-step policy entropy
(returned as `mean_entropy`) but **never adds it to the loss**. With no entropy
regularization, the higher-capacity ReLU head is free to saturate. This was the fix queued
since Week-11 Session 3; it was deferred because the low-capacity linear head was stable
enough without it. Giving the head real capacity made the deferred bug the blocker.

This spec adds an entropy bonus to the REINFORCE objective so that a fair head-vs-encoder
re-run of EXP-025 becomes possible.

## Design (settled)

- **Objective change:** the REINFORCE loss becomes
  `loss = -(log_probs * advantages).sum() - beta * entropies.sum()`
  i.e. subtract `beta * sum_t H(pi_t)` (maximizing entropy = subtracting the entropy term
  from a minimized loss). The entropy term is summed over trajectory steps, matching the
  scale of the summed policy-gradient term.
- **beta value:** a single conservative `0.01` for the re-run (one variable at a time).
- **Permanent, not a probe-only flag:** add `entropy_beta: float = 0.0` as a parameter of
  `train_episode` and a field of `GenConfig`, threaded to the trainer. Default `0.0`
  reproduces today's behavior exactly, so EXP-023 / EXP-024 stay byte-identical; the
  EXP-025 re-run sets `entropy_beta=0.01`.

## Scope

In scope:
- `entropy_beta` parameter on `train_episode` (default 0.0), added to the loss.
- `entropy_beta` field on `GenConfig` (default 0.0), threaded to `train_episode` in
  `run_generalization`.
- Re-run the exact EXP-025 sweep with `entropy_beta=0.01`, re-read the verdict.

Out of scope (parked):
- No LR change, no advantage normalization (the other queued levers) unless the
  entropy-bonus re-run is still unstable (Branch A3 in the Week-13 note).
- No encoder unfreezing (that is the next step, gated on the re-run verdict).
- No beta sweep in the baseline re-run (escalate only if 0.01 is ambiguous).

## Components and changes

`src/neuromorphic/training/reinforce.py` — `train_episode`:
- Add keyword-only param `entropy_beta: float = 0.0`.
- After computing `entropies` (already collected per step), change the loss line to
  subtract `entropy_beta * torch.stack(entropies).sum()`. When `entropy_beta == 0.0` the
  numeric loss is bit-identical to today (subtracting `0.0 * x`); guard the byte-identity
  with a test.
- `mean_entropy` in the returned stats is unchanged.

`src/neuromorphic/training/generalization.py`:
- Add `entropy_beta: float = 0.0` to `GenConfig` (next to `head_type` / `hidden`).
- Pass `entropy_beta=cfg.entropy_beta` into the `train_episode` call in the episode loop.
- `asdict(cfg)` already records it in the summary `config` block.

`experiments/025_head_capacity/run.py`:
- `build_configs` sets `entropy_beta=0.01` on every config for the re-run. (Keep the
  original zero-beta outputs as the EXP-025 baseline of record; the re-run writes new
  tagged files, e.g. tag suffix `_b01`, so nothing is overwritten.)

## Testing

- **Default byte-identity:** with `entropy_beta=0.0`, `train_episode` returns the same
  `loss` (to full float equality) as the pre-change path for a fixed seed/episode. Guards
  EXP-023/024 reproducibility.
- **Bonus lowers the loss:** with `entropy_beta > 0` and a non-degenerate policy
  (entropy > 0), the returned `loss` is strictly less than the `entropy_beta=0.0` loss on
  the same trajectory (the bonus subtracts a positive quantity).
- **Gradient still flows:** a 1-episode `train_episode` with `entropy_beta=0.01` runs and
  changes head params.
- **Config pass-through:** `GenConfig(entropy_beta=0.01)` is recorded in the summary
  config; determinism holds for fixed seed + beta.
- Keep the existing suite green.

## Deliverable

- The entropy-bonus trainer change + config, tests green.
- EXP-025 re-run outputs (`entropy_beta=0.01`), a refreshed evidence table, and the
  updated verdict (head-limited / encoder-limited / still-unstable). Findings appended to
  the Week-13 obsidian note (Session 2), ADR-0001 amended only if the re-run is decisive.
