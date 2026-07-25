# Session Handoff — 2026-07-24 (Fri) -> Phase 3 kickoff (Sat 2026-07-25)

> Single-page pickup point. Everything below is committed and pushed to `origin/main`;
> nothing is stranded. Tomorrow is the Phase-3 build block: **execute the cube-env plan.**

## AMENDMENT (later on 2026-07-24, design audit): read this first

The spec and plan were audited before execution and **revised**. Three things below are now stale;
the spec and plan themselves are correct and up to date.

1. **The move set is 6, not 12.** A 2x2 has no centers, so a face turn is the same physical act as the
   opposite face's counter-turn (`U == D'`, `R == L'`, `F == B'`) -- a 12-move action space is exactly
   2x redundant and gives every depth-1 scramble two solving moves. The env holds the DLB corner still
   and turns only U, R, F. All 3,674,160 states and God's number 14 are unchanged. **Chance at depth 1
   is 1/6, not 1/12.** `docs/architecture-spec-v3.md`, `phase3-kickoff-brief.md` and
   `phase2-to-phase3-transition.md` were updated to match.
2. **No BFS canonicalization, and the cost estimate was wrong.** The old 12-move design needed
   canonicalization over 24 rotations and measured **~33 minutes** to build (not "1-2 min"), times four
   because each test rebuilt it. The 6-move set makes the raw facelet tuple canonical: full build is
   **~67s**, and `ExactBFSDistance(max_depth=6)` builds in 0.04s for tests.
3. **`scramble_depth` is a move count, not a distance.** A `k`-move walk can land closer than `k`
   (measured 0% at depths 1-2, ~3.6% at depth 3, ~15% at depth 6). `scramble` now never returns a solved
   cube, and `exact_depth=True` (with a provider) makes the difficulty axis exact.

Also fixed in the plan: `info` now carries the full trace contract (`move`, `move_label`), `CubeEnv` is
exported from `envs/__init__.py`, `terminated`/`truncated` are mutually exclusive, and
`ExactBFSDistance` has a public API. **All plan code has been executed and its tests run green**
(31 fast + 1 slow).

## 0. Start here (2 minutes)

```bash
git fetch --all --prune
git status                      # expect: on main, clean, up to date with origin/main
.venv/Scripts/python.exe -m pytest tests/envs tests/regions -q   # baseline before building cube (should pass; cube tests do not exist yet)
```

Then execute the plan (see §3): `docs/superpowers/plans/2026-07-24-cube-env.md`.

## 1. Git state

- `main` == `origin/main`, clean. Repo is `main`-only (no feature branches open).
- Last relevant commits: the cube-env design-audit revision (see the Amendment), `3eca8e6` this handoff, `4d2d584` cube-env PLAN, `e7dba74` cube-env SPEC, `71c5900` Phase-2-Live/dashboard-honesty epilogue merges.
- Phase-2 tag `phase-2-complete` sits on `c292cda` (unchanged; the cube work is Phase 3).

## 2. What this session did (Fri Jul 24 study block)

Design-only block: brainstormed -> spec'd -> planned the **2x2 cube environment** (`CubeEnv`). No code shipped; the deliverable is a build-ready plan with **pre-verified cube math**.

- **Spec:** `docs/superpowers/specs/2026-07-24-cube-env-design.md`
- **Plan:** `docs/superpowers/plans/2026-07-24-cube-env.md`

The load-bearing risk (cube move permutations) was retired up front: the move permutations embedded in the plan were derived from a 3D cubie model and **verified to reproduce the published 2x2 quarter-turn distance distribution** `[1,6,27,120,534,2256,8969]`. That verification was independently reproduced during the audit. They are correct; copy them verbatim, do not re-derive. (The audit reduced the move set from 12 to 6 and dropped the rotation generators entirely -- see the Amendment.)

## 3. THE JOB TOMORROW: execute the cube-env plan

Plan `docs/superpowers/plans/2026-07-24-cube-env.md`, 4 TDD tasks, base = current `main` (the design-audit revision commit). **Recommended: run via sdd-runner** (subagent-driven, same loop as EXP-027/028 and the pulse rework). Branch first: `git checkout -b week16-cube-env` (or similar), then sdd-runner on that branch.

| Task | Builds | Key correctness gate |
|---|---|---|
| 1 | `src/neuromorphic/envs/cube.py` primitives (6 moves, `is_solved`, `scramble`) | property tests: order-4, CW/CCW inverse, scramble round-trip, bijection, anti-redundancy, fixed DLB corner |
| 2 | `CubeEnv(gym.Env)` (appended to cube.py) | mirrors GridWorldEnv; Discrete(6), Box(24), sparse reward, optional distance, full trace-contract `info` |
| 3 | `src/neuromorphic/envs/cube_distance.py` `ExactBFSDistance` | BFS levels == `[1,6,27,120,534,2256,8969]`; full table == **3,674,160** states, max distance == **14** (full build ~67s, in a `slow`-marked test) |
| 4 | `encode_cube` in `src/neuromorphic/regions/sensory_cortex.py` | size-generic (test checks 3x3 width too); 144-wide one-hot Poisson |

**Constraints (in the plan's Global Constraints, restated):** everything under `src/neuromorphic/`, NOTHING under `experiments/`; run python via `.venv/Scripts/python.exe`; plain commits, no Co-Authored-By, no em-dashes; move-apply convention `tuple(f[P[i]] for i in range(24))`; distance is optional/nullable end-to-end.

## 4. The three Phase-3 decisions already locked (do NOT relitigate)

From `docs/phase3-kickoff-brief.md` (Mon Jul 20). The cube-env plan already honors these; they matter for the NEXT step (the baseline):

1. **Fail-first baseline.** After the env exists, run the v1 recipe (frozen encoder + linear REINFORCE head) on **1-move scrambles** and let it fail informatively. It MUST carry EXP-028's **input-noise regularization** so it is best-v1, not a strawman.
2. **Honest encoding, no pre-training proxy yet.** 144 one-hot Poisson (`encode_cube`). Do NOT add a distance-to-solved / per-facelet pre-training proxy yet (deferred until evidence says the encoder binds; per EXP-028 the grid cap was optimization-limited, not encoder-limited). Categorical colors are NOT a population-coding (tuning-curve) target.
3. **Recall-in-loop as the engagement ceiling** (R-STDP deferred). And: the baseline runs **pure v1** (`recall=False`); recall-in-loop is the FIRST engagement step AFTER the baseline. Principled: a 1-move scramble is solvable reactively.

**Pre-registered baseline expectation (revised per the Amendment):** depth-1 is effectively **6**-way classification (chance **1/6**), so v1 SHOULD do well; expect a sharp fall at depth 2, near-chance by depth 3 -- the informative number is WHERE it collapses. A depth-1 failure indicts the encoding/setup, NOT the architecture, and must be debugged before any regionalization conclusion. If the collapse-vs-depth curve is the headline number, run with `exact_depth=True` so the depth axis is true distance.

## 5. After the env (the rest of Phase 3, not tomorrow-morning)

- **v1-recipe baseline** as an experiment DRIVER under `experiments/NNN_cube_baseline/` (the env is library; the run is an experiment). Carry noise regularization; 12 seeds (n>=12 standing rule); commit a `RESULTS.md` (standing habit).
- **Monolithic same-neuron-count baseline** early, so "does regionalization help?" (Phase-3 checkpoint criterion 2) is answerable from day one.
- **Distance provider is our instrument, not the model's.** The observation is raw facelets only; the model never sees distance. Exact distance is a 2x2 luxury that does NOT generalize to 3x3 (~4.3e19 states) -- keep it optional/nullable everywhere. Sparse "learn from facelets" is also the path that generalizes.
- **Reward shaping (warmer/colder via distance) is BREAK-GLASS ONLY** -- reserved for if traditional sparse sensory RL genuinely cannot learn. Not a casual toggle.

## 6. Extensibility contract (why the env is built the way it is)

Designed for 2x2 -> 3x3 -> NxN. Two size-specific seams: the **move set** and the **distance provider**. Everything else (facelet obs, `encode_cube`, scramble, solved-detection, sparse reward, trace contract) is size-generic. A future `Cube3x3Env` = new move permutations + a different/absent distance provider on the same contract. Full delta table in the spec §7.

## 7. Dashboard note (not blocking the env)

The cube **trace/task contract** is defined in the spec §6 (`facelets`, `solved`, `scramble_depth`, `distance` nullable, `move`, `move_label`, ...). The env's `info` already exposes the data; wiring it into the monitor + a **cube task panel** in the dashboard is a separate follow-up (the panel is currently grid-hardcoded in both `_grid_world` and React `TaskState`). Not needed to build or run the env; needed before cube traces render meaningfully.

## 8. Pointers

- Spec: `docs/superpowers/specs/2026-07-24-cube-env-design.md`
- Plan: `docs/superpowers/plans/2026-07-24-cube-env.md`
- Kickoff decisions: `docs/phase3-kickoff-brief.md`
- Architecture (as-trained + cube retarget): `docs/architecture-spec-v3.md`
- Grid env to mirror: `src/neuromorphic/envs/gridworld.py`; encoder to mirror: `encode_gridworld` in `src/neuromorphic/regions/sensory_cortex.py`
- Vault: `Weekly Notes/week-16-phase3-prep.md` (Fri Jul 24 session + Saturday plan)
