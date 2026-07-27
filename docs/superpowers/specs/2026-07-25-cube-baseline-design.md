# EXP-029 Cube Baseline (design)

**Date:** 2026-07-25 · **Phase:** 3 (first cube experiment) · **Grounds:** `docs/phase3-kickoff-brief.md`, `docs/superpowers/specs/2026-07-24-cube-env-design.md`, ADR-0001 and its amendments, EXP-024 through EXP-028.

## Goal

Run the v1 recipe (frozen sensory encoder + linear REINFORCE head) on `CubeEnv` and let it fail informatively, alongside a neuron-matched monolithic control so that the Phase-3 checkpoint question "does regionalization help?" is answerable from this experiment rather than retrofitted later.

This is a **fail-first baseline**. The deliverable is a collapse curve (success vs. exact scramble distance) and an honest reading of where it breaks, not a working cube solver.

## Placement (architectural rule)

Same rule as the env: reusable machinery lives under `src/neuromorphic/`, and `experiments/029_cube_baseline/` holds only the driver. Two library pieces are required before the driver can exist (§1, §2); the driver is §3.

## Locked inputs (do NOT relitigate)

From `docs/phase3-kickoff-brief.md` and the 2026-07-24 design audit:

1. **Fail-first.** Best-v1, not a strawman. Carries noise regularization.
2. **Honest encoding, no pre-training proxy.** 144 one-hot Poisson via `encode_cube`. No distance-to-solved or per-facelet pre-training target. The sensory encoder is therefore **frozen at random init**: unlike the grid's v1, there is no displacement pre-training to inherit. This is deliberate and is the single most likely reason for a weak depth-1 number, so it is stated up front rather than discovered in the post-mortem.
3. **Pure v1.** `recall=False`; recall-in-loop is the first engagement step AFTER this baseline.
4. **Distance is our instrument, never the model's.** The observation is raw facelets. The model never sees distance.
5. **Reward shaping is break-glass only.** Not used in this experiment.

## 1. Encoder seam in `Brain` (library)

`Brain` currently hardcodes `encode_gridworld` at `brain.py:117` and `:156`, and derives `n_obs = 2 * grid_n**2`. It cannot represent a cube.

Add two optional constructor parameters, defaulting to today's behavior:

```python
Brain(grid_n=5)                                        # unchanged
Brain(encoder=cube_encoder, n_obs=144, n_actions=6)    # cube
```

- `encoder`: a callable `(obs_tensor, T, generator) -> [T, B, n_obs]` spikes. Defaults to a partial of `encode_gridworld` bound to `grid_n`.
- `n_obs`: input width. Defaults to `2 * grid_n**2` when not supplied.
- Both call sites route through `self._encoder`.

`MotorCortex`, `ThalamicRouter` and `Prefrontal` already accept `n_actions`, so the 4 to 6 action change is an argument, not a code change.

**Backward compatibility is a hard requirement:** the full 313-test suite and every 024-028 driver must pass untouched. This lands as its own library commit, verified before anything cube-specific is built on it.

## 2. `CubeRunner` (library)

`run_generalization` cannot be reused: it is built on `split_goals`, `manhattan` optimality, and `GridWorldEnv`. The cube needs its own run loop, in `src/neuromorphic/training/cube_baseline.py`.

`reinforce.py` **is** reused unchanged. It is already environment-agnostic (`make_policy_head` reads `brain.content` and `brain.n_actions`; `action_distribution` only calls `brain.step(obs)`).

Responsibilities:

- **Shell enumeration.** Given `ExactBFSDistance`, collect all states at exact distance `d`.
- **Train/held-out split.** Deterministic by seed. See §4 for the sizing rule.
- **Training loop.** REINFORCE on the linear head over the frozen concept, sampling start states from the train split.
- **Greedy evaluation.** Argmax rollouts from each eval state, `max_steps = 2d + 3`.
- **Config dataclass** mirroring `GenConfig` in shape and naming.

## 3. `experiments/029_cube_baseline/` (driver)

Matches the 027/028 layout: `run.py`, `aggregate.py`, `outputs/` (gitignored), `RESULTS.md` (committed). Parallelised with `ProcessPoolExecutor` as EXP-028 did. No logic that a future experiment would want to reuse.

## 4. Protocol

| Axis | Value |
|---|---|
| Arms | `regionalized`, `monolithic`, `random` (see below) |
| Depths | 1 to 6, `exact_depth=True`, one shared `ExactBFSDistance(max_depth=6)` |
| Seeds | 0 to 11 (n >= 12 standing rule; n=5 lied to us in EXP-026) |
| Head | Linear, over the frozen sensory concept |
| Recall | `False` |
| `max_steps` | `2d + 3` (optimal is `d`) |

**Arms.** `regionalized` is the five-region `Brain`. `monolithic` is an unregionalized spiking stack matched on **total neuron count**, frozen at random init, with the identical head, optimizer, seeds and depth grid. `random` is a uniform-policy arm, evaluation only and no training, which measures the true chance floor (see §5).

Total-neuron matching does NOT put the two arms on equal footing on the path that actually feeds the policy head. With `recall=False`, the regionalized policy path is the sensory region only (144 -> 128 hidden -> 64 concept); hippocampus (150 neurons) never runs, and prefrontal (150), router (12) and motor (6) run but their outputs are discarded before the head. The monolithic arm spends its whole budget on the path (144 -> 446 hidden -> 64 concept). Effective policy-path width is therefore **128 (regionalized) vs 446 (monolithic)**, not matched, and any gap between arms is confounded with this width difference, not attributable to topology alone.

The matched neuron count is **computed from the live region sizes at construction time**, not hardcoded: sum the neuron counts of sensory (hidden + concept), hippocampus, prefrontal, router and motor for the cube configuration, and size the flat stack to the same total. A test asserts the two totals are equal, so the match cannot silently drift if a region's default changes.

**Evaluation.** The state shell at exact distance `d` has 6, 27, 120, 534, 2256, 8969 members for `d = 1..6`.

- `d = 1, 2`: no split. Training samples start states from the whole shell, and evaluation is **exhaustive** over that same whole shell. Reported and labelled explicitly as **training-distribution**, not generalization. Holding out 1 of 6 states is not a generalization test.
- `d >= 3`: per-shell train/held-out split, held-out capped at `min(200, 25% of shell)`. Training samples only from the train side. Report held-out success.

The cap exists because `brain.step` costs **90 ms** (measured, 11 steps/sec): an exhaustive `d = 6` evaluation would take about 3.4 hours on its own. Capping at 200 keeps a single evaluation near 4 minutes.

**Noise.** EXP-028's regularizer is Gaussian noise on the sensory **concept** (the encoder's output, via `AblationSpec` / `AblatedConcept`), applied at both train and eval, at sigma 0.4 on the grid, where it doubled held-out navigation from 43% to 83%. That sigma is grid-tuned and its effect was a surprise nobody predicted, so it is **not** imported as a constant. Sweep sigma in `{0, 0.2, 0.4}` at depth 1, then fix each arm's winner for depths 2-6.

**The sweep runs on BOTH trained arms, and each arm keeps its own winning sigma.** Tuning sigma on `regionalized` and applying that value to `monolithic` would hand the regionalized arm a tuned hyperparameter and the control an untuned one, biasing the very comparison the experiment exists to make. The depth-1 cell of the main grid is then simply each arm's winning-sigma run, not a separate run.

Both train-and-eval application is retained so the number stays comparable to EXP-028; EXP-028's own caveat about that choice carries over and is restated in `RESULTS.md`.

## 5. Pre-registered interpretation contract

Written before the numbers exist, per the EXP-028 lesson (where pre-registration mattered precisely because the headline result refuted it).

- **Chance is measured, not assumed.** The `random` arm gives the floor at each depth. 1/6 is the probability the *first* action is correct; with a `2d + 3` step budget a random walk can stumble into solved, so the real floor is higher and depth-dependent. Every "near-chance" claim is made against the measured `random` curve.
- **Expected shape:** solid success at depth 1, sharp degradation at depth 2, near-chance by depth 3. The informative number is **where the knee is**, not any single cell.
- **A weak depth-1 result does NOT indict the architecture.** Depth 1 is a 6-way classification with exactly one correct action. Failing it indicts the encoding or the training setup, and must be debugged before any regionalization conclusion is drawn. Note the frozen-random encoder (locked input 2) as the leading suspect.
- **The regionalization comparison is paired per seed.** `regionalized` vs `monolithic` at matched seed and depth, reported as a paired test, not as two independent means.
- **A null result is a result.** If the two arms are indistinguishable, that is the finding, and it is reported as such rather than explored until something separates.
- **A monolithic win is confounded with policy-path width.** Effective policy-path width is 128 (regionalized) vs 446 (monolithic; see §4), so a monolithic win reads as "a wider random feature bank wins," not "regionalization does not help." Only a regionalized win, achieved at roughly a third the effective width, is informative about topology.
- **The depth-1 cell is optimistically biased.** It is selected by max over three swept sigmas on the same data it reports, while depths 2 to 6 use a single pre-selected sigma. Depth 1 is therefore not directly comparable in bias to the rest of the curve.

## 6. Outputs

- `outputs/*.jsonl`: one record per (arm, depth, seed, sigma) run. Gitignored.
- `aggregate.py` -> `outputs/029_curve.md`: the collapse table.
- `RESULTS.md`: committed, curated, with provenance (seeds, date, machine, regeneration command), per the standing habit adopted after the 2026-07-13 audit. The interpretation contract in §5 is committed **before** the numbers land.

## 7. Cost (measured)

Based on `brain.step` at 90 ms:

| Item | Runs | Single-core |
|---|---|---|
| Sigma sweep, depth 1 (2 arms x 3 sigmas x 12 seeds); supplies the depth-1 cell | 72 | ~6 h |
| Depths 2-6 at each arm's winning sigma (2 arms x 5 depths x 12 seeds) | 120 | ~26 h |
| `random` arm (eval only, no training) | 72 | negligible |
| **Total trained runs** | **192** | **~32 h** |

About **4 to 5 hours on 8 cores** with `ProcessPoolExecutor`. If it must shrink, the honest levers are episodes (600 to 400) or depths (1-6 to 1-4). **Seeds are not a lever.**

## 8. Testing

- **Encoder seam:** existing `Brain(grid_n=5)` behavior is bit-identical under a fixed generator; the full 313-test suite passes; a cube-configured `Brain` produces `[T, B, 144]` obs spikes and 6 action logits.
- **`CubeRunner`:** shell enumeration returns exactly the published shell sizes; the train/held-out split is deterministic by seed and disjoint; the held-out cap is honored; `max_steps` follows `2d + 3`.
- **Driver:** a smoke run at depth 1, 1 seed, few episodes completes and writes a well-formed record.

## Non-goals

- Making the cube actually get solved. This is the fail-first baseline.
- Recall-in-loop, R-STDP, or any engagement step (that is the experiment after this one).
- The distance-to-solved pre-training proxy (locked input 2).
- 3x3 anything.
- Dashboard or monitor wiring for cube traces (separate follow-up; the env's `info` already satisfies the trace contract).

## Success criteria

1. A collapse curve over depths 1-6 for both trained arms, with the measured `random` floor alongside.
2. The regionalized vs monolithic comparison reported as a paired per-seed test at n = 12.
3. Depth 1 and 2 numbers labelled training-distribution; depth 3-6 numbers labelled held-out.
4. Every claim in the §5 contract either confirmed or explicitly refuted in `RESULTS.md`.
5. `Brain`'s encoder seam ships without changing any existing behavior (313 tests green).
6. All reusable machinery under `src/neuromorphic/`; only the driver under `experiments/`.
