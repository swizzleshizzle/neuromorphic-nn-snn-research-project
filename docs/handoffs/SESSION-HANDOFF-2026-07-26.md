# Session Handoff - 2026-07-26 (Sun) -> Week 17

> Single-page pickup point. Everything below is committed and pushed to `origin/main` (`3070921`);
> nothing is stranded. Repo is `main`-only, working tree clean, 340 tests passing.

## 0. Start here (2 minutes)

```bash
git fetch --all --prune
git status                      # expect: on main, clean, up to date with origin/main
.venv/Scripts/python.exe -m pytest tests/ -q -m "not slow"   # expect 339 passed, 1 deselected
```

The full suite including the slow BFS test is 340 and takes about 16 minutes. The fast inner loop is `-m "not slow"`.

## 1. What shipped this weekend

**Phase 3 now has a complete cube stack, library plus harness.** Two merges:

- `d48e943` - the 2x2 cube environment (`CubeEnv`, `ExactBFSDistance`, `encode_cube`)
- `3070921` - the EXP-029 baseline harness (encoder seam, `MonolithicBrain`, `CubeRunner`, driver)

New library surface:

| Piece | Path |
|---|---|
| Cube env + exact distance | `src/neuromorphic/envs/cube.py`, `cube_distance.py` |
| Picklable encoder factories | `src/neuromorphic/encoders.py` |
| `Brain` encoder seam | `src/neuromorphic/brain.py` (`encoder`, `n_obs`, `obs_width`, `n_neurons`) |
| Unregionalized control | `src/neuromorphic/monolithic.py` |
| Cube run loop | `src/neuromorphic/training/cube_baseline.py` |
| Driver | `experiments/029_cube_baseline/` |

Design docs: `docs/superpowers/specs/2026-07-25-cube-baseline-design.md`, plan at `docs/superpowers/plans/2026-07-25-cube-baseline.md`.

## 2. THE FINDING THAT CHANGES WEEK 17

**EXP-029 as designed cannot answer "does regionalization help?" Do not run it expecting that answer.**

Verified empirically, not argued. With `recall=False` (locked kickoff decision 3), the regionalized policy path is the sensory region ONLY: `144 -> 128 -> 64`. Hippocampus (150 neurons) never executes; prefrontal (150), router (12) and motor (6) run but their outputs are discarded before the linear head reads `concept`. So **318 of the five-region brain's 510 neurons are off the policy path.**

Two consequences:

1. **The neuron-matched monolithic arm is width-confounded.** It spends its whole 510 budget on the path (`144 -> 446 -> 64`), so the comparison is a 128-wide frozen stack against a 446-wide one: 26,816 vs 93,278 policy-path parameters. If monolithic wins, the honest reading is "a wider random feature bank wins," not "regionalization does not help." Only a regionalized win, at roughly a third the effective width, would be informative about topology.
2. **A path-matched third arm is not a control, it is the same network.** Measured: `MonolithicBrain(total_neurons=192)` constructs the identical `SensoryCortex(144 -> 128 -> 64)` at the same seed. All four weight tensors compare equal and the emitted concept is bit-identical to the regionalized arm. Adding that arm would cost about 72 runs to produce a guaranteed 0.000 difference.

**Regionalization only becomes testable once the other regions are ON the policy path.** That is the engagement step the kickoff brief already defers to "after the baseline" (recall-in-loop first, R-STDP later). Week 17's real question is that step, not another baseline variant.

Both caveats are written into the pre-registered contract (spec section 5) BEFORE any numbers exist, which is the point of pre-registration.

## 3. What EXP-029 IS still good for

It is a genuine fail-first baseline and the collapse curve is worth having:

- Where does v1 collapse as a function of **exact** scramble distance (not move count)?
- How far above the **measured** random floor is it at each depth?
- Does a wider frozen random feature bank help at all? (the monolithic arm answers this, honestly labelled)

Run it, read it as a capability floor and a width probe, and do not write a regionalization verdict from it.

## 4. Running the experiment (not yet done)

```bash
.venv/Scripts/python.exe experiments/029_cube_baseline/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
.venv/Scripts/python.exe experiments/029_cube_baseline/aggregate.py
```

About 4 to 5 hours on 8 cores (192 trained runs; `brain.step` is 90 ms and dominates). Phase 1 sweeps sigma at depth 1 on both trained arms and writes `outputs/029_winners.json`; phase 2 runs depths 2 to 6 at each arm's winner plus the evaluation-only `random` arm at every depth.

Then write `experiments/029_cube_baseline/RESULTS.md` against the spec's section-5 contract, marking each pre-registered claim confirmed or refuted, with provenance (seeds, date, machine, regeneration command). Standing habit since the 2026-07-13 audit: results live in-repo, never only in `outputs/` or a vault note.

Reading notes for whoever writes it:
- Depths 1 and 2 are **training-distribution** (shells of 6 and 27, deliberately unsplit). Depths 3 to 6 are held-out. The table labels which is which; keep that distinction in the prose.
- The depth-1 cell is selected by max over three sigmas on the data it reports, so it is optimistically biased relative to depths 2 to 6. The aggregator prints the unselected all-sigma mean alongside it. Quote both.
- Chance is the measured `random` arm, not 1/6. A random walk with a `2d+3` budget can stumble into solved.

## 5. Parked test debt (deliberate, logged, not forgotten)

Two items adjudicated as real but not load-bearing at the end of the review loop:

1. `test_random_arm_scores_above_zero_but_well_below_one` asserts `0.0 <= x <= 1.0`, which cannot fail. It is why a hardcoded `random.Random(0)` in `evaluate_states` survived five task reviews before the final review caught it. The underlying bug is fixed and now covered by `test_different_rng_seeds_give_different_random_arm_results`; the stale assertion is redundant.
2. `test_both_arms_get_identical_head_init_at_a_fixed_seed` re-implements the reseed inline instead of exercising `run_cube_baseline`, so deleting the production fix would not fail it. The fix at `cube_baseline.py` is correct and was verified at the call site.

Neither blocks the run. Both are cheap to close.

## 6. Suggested Week 17 shape

1. **Run EXP-029 and write RESULTS.md** (half a day, mostly unattended). Locks the capability floor.
2. **Then the engagement step**, which is where the real Phase-3 question lives: put the deferred regions on the policy path. Recall-in-loop (`recall=True`) is the first and cheapest lever, and it is the one that finally makes a regionalized-vs-monolithic comparison mean something, because the hippocampal path stops being dead weight.
3. Dashboard cube task panel remains unstarted and is still the Phase-3 registry blocker. The env's `info` already satisfies the trace contract (`facelets`, `solved`, `scramble_depth`, `distance`, `move`, `move_label`); nothing wires it into `neuromorphic.monitor` yet, and `build_header` still defaults `task_type="gridworld"`.

## 7. Pointers

- Cube env design: `docs/superpowers/specs/2026-07-24-cube-env-design.md`
- EXP-029 design + pre-registration: `docs/superpowers/specs/2026-07-25-cube-baseline-design.md`
- Kickoff decisions: `docs/phase3-kickoff-brief.md`
- Architecture (as-trained + cube retarget): `docs/architecture-spec-v3.md`
- Previous handoff: `docs/handoffs/SESSION-HANDOFF-2026-07-24.md`
