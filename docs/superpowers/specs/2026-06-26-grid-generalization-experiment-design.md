# Grid-World Generalization Experiment Design

**Date:** 2026-06-26
**Status:** Approved (pending implementation plan)
**Session:** Week 12, Grid Navigation Training Deep Dive
**Author context:** v1 trains only a linear policy head on the frozen sensory concept (ADR-0001 Amendment 1). The fixed-goal 5x5 is already solved at 100% with optimal 8-step paths (EXP-023).

## 1. Question

Did the v1 agent learn to **navigate**, or did it **memorize** the single path to the fixed goal (4,4)?

The observation is `[ax, ay, gx, gy]` (agent and goal coordinates), so the head can in principle use the goal coordinates. With a fixed goal, `gx, gy` were constant and the head never needed them. Randomizing the goal forces the question. The experiment trains on a subset of goals and evaluates on goals never seen in training; the gap between train-goal and held-out-goal success is the headline result.

This is a controlled study within v1 (the brain stays frozen). It does not engage or unfreeze any region; that is the separate next arc.

## 2. Background

Confirmed by code inspection:
- Brain runs under `torch.no_grad()` (`src/neuromorphic/training/reinforce.py:54`); only `nn.Linear(brain.content=64, brain.n_actions=4)` trains (`run.py` optimizer over `policy_parameters(head)`).
- Pathways are static `register_buffer`s; no STDP/Hebbian in the training loop. Memory bypassed (`recall=False`).
- Env (`src/neuromorphic/envs/gridworld.py`): 5x5, start (0,0), goal (4,4), `-1.0` per step, `+10.0` on goal, truncate at 100 steps. No random goal, no reward shaping, no curriculum today.
- No experiment tracker is wired in; training logs to stdout plus one end-of-run curve and one monitor trace.

## 3. Components

### 3.1 Env changes (`src/neuromorphic/envs/gridworld.py`), backward-compatible

- **Random goal:** the env accepts an optional set of candidate goals and a seed. On `reset()`, if a candidate set is provided, it samples a goal uniformly from that set using the env's own seeded RNG; otherwise it uses the existing fixed `goal`. The start stays (0,0). Existing fixed-goal construction and behavior are unchanged.
- **Potential-based reward shaping (optional, default off):** with shaping enabled, the per-step reward gains `Phi(s') - Phi(s)` where `Phi(s) = -manhattan(agent, goal)`. This is theory-preserving (Ng, Harada, Russell 1999): it does not change the optimal policy, only densifies the signal toward the current goal. The base reward (`-1` per step, `+10` on goal, truncation) is unchanged; shaping is added on top. Default off preserves current behavior.

### 3.2 Generalization helpers (`src/neuromorphic/training/generalization.py`, new, importable + testable)

- `split_goals(size, start, n_heldout, seed) -> (train_goals, heldout_goals)`: candidate cells are all grid cells except `start`; a seeded shuffle partitions them into disjoint train and held-out lists. Deterministic for a given seed. For 5x5 with one start cell: 24 candidates, default 18 train / 6 held-out.
- `EvalResult` (small dataclass): `success_rate`, `mean_steps`, `optimality` (mean over reached goals of `manhattan(start, goal) / steps`, 1.0 = optimal), `n`.
- `evaluate(brain, head, goals, max_steps, ...) -> EvalResult`: greedy (argmax-logit) rollout from the fixed start to each goal in `goals`; a goal counts as success if reached within `max_steps`. Pure with respect to the policy (no training, no gradient).

### 3.3 Experiment (`experiments/024_grid_generalization/`, new; 023 untouched)

- `run.py`: builds the brain and the linear head, draws the goal split, trains the head with REINFORCE on **random train-set goals** (shaping per CLI flag), and after training evaluates greedily on the **train goals** and the **held-out goals** separately.
- **Per-episode CSV** (`outputs/024_grid_generalization_<tag>_metrics.csv`): columns `episode, goal_x, goal_y, total_reward, steps, goal_reached, entropy`.
- **Summary JSON** (`outputs/024_grid_generalization_<tag>_summary.json`): the full run config (seed, episodes, lr, shaping on/off, the train/held-out split) plus the eval results (train and held-out `success_rate`, `mean_steps`, `optimality`) and the computed generalization gap. This satisfies "document every experiment configuration."
- CLI flags include `--episodes`, `--lr`, `--seed`, `--shaping/--no-shaping`, `--tag`.

## 4. Protocol (runs tonight)

1. **Shaped random-goal** training (primary): shaping on, train on the 18 train goals, eval train vs held-out.
2. **Sparse random-goal** training (ablation): shaping off, otherwise identical, to show how much the shaping contributed.
3. The existing **fixed-goal** EXP-023 result is the reference point (already 100% / optimal).

**Headline number:** generalization gap = train-goal success_rate minus held-out success_rate.
- Near zero with high success on both: the head learned goal-conditioned navigation.
- Large gap (high on train, low on held-out): it memorized trained goals.
- Low on both: it did not learn the random-goal task (shaping vs sparse tells us whether signal was the bottleneck).

## 5. Testing

TDD on the pure pieces:
- Goal sampling: sampled goals are in the candidate set, never equal the start, and the sequence is deterministic for a fixed seed.
- `split_goals`: train and held-out are disjoint, cover all candidates, have the requested sizes, and are deterministic by seed.
- Shaping: `Phi(s') - Phi(s)` is positive when a step reduces Manhattan distance and negative when it increases; over any full path the shaping sum telescopes to `Phi(end) - Phi(start)`; with shaping off the reward equals the current behavior exactly (backward compat).
- `evaluate`: success detection and the optimality calculation are correct on a hand-built tiny case.

The REINFORCE training loop gets a fast smoke/integration test (a few episodes run without error and produce a CSV and summary), not unit assertions on stochastic learning outcomes.

## 6. Scope

- **Core (tonight):** sections 3 and 4 in full, including both the shaped run and the sparse ablation, with the held-out eval and the documented CSV plus summary.
- **Stretch (if time):** a learning-speed comparison against the Phase-0 Q-learning agent on random goals.
- **Deferred:** the 3x3 to 5x5 curriculum; unfreezing or pre-training any brain region (the Option-B arc); wiring W&B/TensorBoard (the CSV plus summary is enough for this study).

## 7. Acceptance criteria

1. The env samples random goals from a provided candidate set (start excluded) with a seedable, deterministic RNG, and fixed-goal behavior is unchanged when no candidate set is given.
2. Potential-based shaping can be toggled on; off reproduces current reward exactly; on densifies toward the goal without changing the optimal policy.
3. `split_goals` produces a deterministic, disjoint 18/6 train/held-out partition for 5x5.
4. A training run trains the head on random train-set goals and writes the per-episode CSV and the summary JSON with the config and the train-vs-held-out eval.
5. The generalization gap is reported, and we can state from the numbers whether v1 navigates or memorized.
6. All pure helpers are unit-tested; the full Python suite stays green.
