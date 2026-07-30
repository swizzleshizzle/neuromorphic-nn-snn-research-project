# Dashboard Cube Support (design)

**Date:** 2026-07-29 · **Phase:** 3 · **Grounds:** `docs/superpowers/specs/2026-06-27-honest-dashboard-trace-design.md`,
`docs/superpowers/specs/2026-07-25-cube-baseline-design.md`, `docs/handoffs/SESSION-HANDOFF-2026-07-27.md` section 5 item 3.

## Goal

Let the NEURO-SCOPE dashboard display a Rubik's cube episode. Today it can display gridworld only.
This is the Phase-3 registry blocker named in the 2026-07-27 handoff.

Scope is **post-hoc replay**: record a cube episode to a JSONL trace, load it in the dashboard, scrub it.
Live streaming during training is a deliberate follow-up (see Non-goals).

## The gap, measured 2026-07-29

The handoff described this as "the cube task panel is grid-hardcoded". It is wider than that. Five
independent breaks, any one of which is fatal on its own:

| Layer | Site | What is grid-shaped |
|---|---|---|
| Emission | `training/cube_baseline.py` | No reference to `monitor`, `trace` or `jsonl` anywhere. A cube run emits **no frames at all**. |
| Header | `monitor/schema.py:65,75` | `task_type` defaults to `"gridworld"`; the task block carries `grid_n`. |
| Header | `monitor/schema.py:53` | `_config_hash` reads `brain.grid_n`. A cube brain is built with `encoder=cube_encoder(), n_obs=144`, so `grid_n` keeps its **default of 5**, and a meaningless 5 is hashed into the run identity. |
| Frame | `monitor/frame.py:46` | `_encoding_block` slices `[T, 2*grid_n**2]` agent/goal planes. Cube encoding is `[T, 24*6]` one-hot facelet-colors. |
| Runner | `monitor/runner.py:135,49` | The task dict hardcodes `agent: [obs[0], obs[1]]` and `goal: [obs[2], obs[3]]`. On a cube those indices are facelet **colors**. `DEFAULT_ACTION_LABELS` is 4 wide; the cube has `N_ACTIONS == 6`. |
| Contract | `dashboard/src/contract.ts:50,61` | `TaskState` requires `agent`/`goal`; `SensoryInput` requires `grid_n`. |
| UI | `panels/TaskState.tsx`, `hero/overlays/SensoryGrid.tsx`, `hero/sensory.ts` | All render `grid_n x grid_n` cells derived from agent/goal. |

**The runner break is the dangerous one.** Reading facelet colors as x/y coordinates does not raise; it
renders a plausible-looking wrong picture. The action-label break, by contrast, is a hard `IndexError` at
actions 4 and 5.

**What already works.** `CubeEnv._info()` returns `solved`, `scramble_depth`, `distance`, `move`,
`move_label`, which is exactly the payload the trace contract wants. That half was built to spec.

## 1. Task adapters (library)

New file `src/neuromorphic/monitor/tasks.py`.

```python
class TaskAdapter(Protocol):
    def header_task(self) -> dict: ...
    def frame_task(self, obs, action, reward, total, terminated, truncated, info) -> dict: ...
    def encoding(self, out: dict) -> dict | None: ...
```

Two implementations:

- **`GridworldAdapter(grid_n)`** reproduces today's behavior exactly: header `{"type": "gridworld",
  "grid_n": n, "action_labels": [...]}`, frame task with `agent`/`goal`, encoding from the two
  `encode_gridworld` planes.
- **`CubeAdapter(cube_n=2)`** emits header `{"type": "cube", "cube_n": 2, "action_labels": MOVE_LABELS}`
  and a frame task of `facelets` (the 24 observed colors) merged with the env `info` fields. It reads
  width from `N_ACTIONS` / `MOVE_LABELS`, never a literal 6, per the standing action-space invariant.

Rejected alternatives: `if task_type == "cube"` branches across three files (the same conditional
triplicated, and drift renders wrong data rather than crashing), and an env-supplied `trace_task()`
method (couples every env to the dashboard wire format and requires retrofitting `GridWorldEnv`).

## 2. Schema, frame, runner (library)

- `build_header(brain, *, seed, adapter, policy_regions=None)`. The `task_type` and `grid_n` parameters
  go away; the task block comes from the adapter.
- `_config_hash` replaces `grid_n` with `n_obs` plus the task type. `n_obs` is meaningful for both tasks;
  `grid_n` is meaningful for only one.
- `SCHEMA_VERSION` becomes `"1.1"`.
- `build_frame` delegates the encoding block to the adapter.
- `record_policy_episode` takes an adapter, defaulting to `GridworldAdapter` so existing callers are
  unaffected. The hardcoded agent/goal task dict and `DEFAULT_ACTION_LABELS` are removed from the
  cube path.

**Deliberate break:** changing the `_config_hash` payload changes the `config_hash` of gridworld traces.
No test asserts a literal hash (`tests/monitor/test_schema.py:34` checks only type and length) and the
value is informational, but it is recorded here rather than discovered later.

## 3. Recorder (`scripts/record_cube_trace.py`, new)

Takes `--depth`, `--seed`, `--out`. Runs the seeded cube training path in-process, then records one
greedy episode through `record_policy_episode` with a `CubeAdapter`.

No checkpoint format is added. `checkpoints.load_trained` hardcodes `Brain(grid_n=..., seed=...)` with no
encoder or `n_obs`, so it structurally cannot rebuild a cube brain; extending it is out of scope here.
Training in-process is reproducible because EXP-030 measured 36/36 concept records byte-identical across
two independent invocations, so a given seed reproduces its exact policy.

`cube_baseline.py` is not touched. EXP-030 is running from this repo on the laptop while this work
proceeds.

## 4. Dashboard

- **`contract.ts`**: three types change, all discriminated on `type`.
  - `TaskState` becomes `GridTask | CubeTask`. A panel cannot read `task.agent` without first narrowing,
    which is the compile-time version of the `runner.py` defect.
  - `TraceHeader["task"]` becomes `GridTaskHeader | CubeTaskHeader`, so `grid_n` exists only on the
    gridworld variant and `cube_n` only on the cube one. Leaving `grid_n` non-optional here would force
    every cube trace to carry a meaningless value, which is the same defect as the `_config_hash` one.
  - `SensoryInput` becomes `GridSensoryInput | CubeSensoryInput`: the grid variant keeps `grid_n` and its
    two planes, the cube variant carries `cube_n` and `n_colors`.
- **`source/parseTrace.ts`**: the wire format keeps `type` in the header only. `parseTrace` reads it once
  and stamps each frame's task while building the in-memory model. The JSONL stays lean, the in-app model
  is a real discriminated union, and the committed `week11_dashboard_trace.jsonl` loads untouched because
  its stamp comes from its own header.
- **`panels/cubeNet.ts`** (new, pure): facelet index 0-23 to net (row, col). No React, so it is unit
  testable in isolation.
- **`panels/TaskState.tsx`**: narrows on `type` and delegates to `GridTaskView` or `CubeTaskView`.
  The cube view renders the unfolded net (U above, L F R B across the middle, D below) with the held DLB
  corner (facelets 12, 16, 21) marked, plus `move_label`, `distance`, `scramble_depth`, reward and return.
- **`hero/sensory.ts` and `SensoryGrid.tsx`**: aggregate cube encoding as a per-facelet argmax over the
  6 color channels, instead of two grid planes.

## 5. Error handling

- **Mismatch is loud.** If the header declares `cube` but frames carry `agent`/`goal`, `parseTrace` throws
  naming the trace path and the conflicting fields. Silence here is worse than a crash: the current code
  would render colors as coordinates and look believable.
- `distance: null` (no `DistanceProvider`) renders as `-`, never the string `null`.
- An unknown `task.type` falls back to a status-only readout rather than crashing, so a future task type
  degrades instead of breaking the page.

## 6. Testing

Per the repo's test-strength rule, every test below must fail against current code.

| Test | What it catches |
|---|---|
| `test_cube_action_labels_match_move_count` | Asserts `len(action_labels) == N_ACTIONS`. Today's 4-wide default raises `IndexError` at actions 4 and 5. |
| `test_cube_frame_task_has_facelets_and_no_coordinates` | Asserts 24 facelets present **and** `agent`/`goal` absent. Today both exist and carry colors. |
| `test_cube_header_omits_grid_n` | Today `build_header` always emits `grid_n`. |
| `test_cube_net_map_is_a_bijection` | The 24 net positions are distinct and cover exactly 24 cells. Catches a transposed or overlapping face. |
| `test_cube_net_reflects_a_known_move` | Apply `R'` to `SOLVED` and assert the net cells that change are exactly those the verified permutation moves. A plausible-but-wrong face map fails this; a shape-only check would not. |
| `test_gridworld_header_and_frames_unchanged_except_version` | Field-by-field equality against a pre-refactor recording at a fixed seed. Same discipline that proved EXP-030's `concept` arm bit-identical. |
| `parseTrace` stamps type from header; committed week11 trace still loads | Back-compat tested rather than assumed. |

The full suite stays at 354 or more.

## Non-goals

- **Live streaming during training.** Chosen as a follow-up spec. It needs a decision about which of 144
  parallel worker runs emits, and it puts per-step cost on a hot path where `brain.step` already costs
  about 90 ms.
- Any edit to `cube_baseline.py` or the EXP-030 driver.
- A cube checkpoint format.
- 3x3 anything. `CubeAdapter` takes `cube_n` so a 3x3 needs no new class, but nothing here is tested
  against one.
- Hero 3D changes beyond the sensory overlay.

## Success criteria

Verified 2026-07-30 (Task 8, final end-to-end check on `week17-dashboard-cube`, commit 7bde5d1).

1. **MET.** `dashboard/public/cube_trace.jsonl` (depth 2, 2 frames, solves) and
   `dashboard/public/cube_trace_d3.jsonl` (depth 3, 9 frames, cycles) both load and are
   internally consistent: header `task.type == "cube"` with no `grid_n` and exactly 6
   `action_labels` (`U U' R R' F F'`, matching `N_ACTIONS`); every frame's `task.facelets`
   has 24 entries with no `agent`/`goal` key; the post-move invariant
   `apply_move(facelets[i], action[i+1]) == facelets[i+1]` holds across all 9 consecutive
   pairs checked; `task.solved` agrees with `is_solved(facelets)` on all 11 frames across
   both files (this is the regression guard for the pre-fix defect where a scrambled cube
   was labelled solved); `encoding.sensory_input` carries `cube_n`/`n_colors` with 144-wide
   spike rows on every frame. Checked with a throwaway script, not committed.
2. **VERIFIED BY TEST ONLY, not confirmed in a browser.** `cubeNet.test.ts` confirms the
   24-facelet net mapping is a bijection, that U/D sit above/below F, that the middle band
   orders L F R B, and that each face is a contiguous 2x2 block. `TaskState.test.tsx`
   confirms a cube trace renders 24 `[data-facelet]` nodes inside a `[data-cube-net]`
   container (not the gridworld `[data-cell]` grid) and shows the move label and distance.
   **Residual gap, and what it actually is.** This spec's section 6 named
   `test_cube_net_reflects_a_known_move` as a dashboard test. It does not exist in
   TypeScript, by a decision recorded in the implementation plan's self-review: the move
   permutations live in `neuromorphic.envs.cube` and vitest cannot reach them, so the
   move-correctness check was moved server-side, where it is stronger. Two Python tests
   cover it against real data rather than a lookup table:
   `test_cube_facelets_follow_the_applied_move` (`tests/monitor/test_tasks.py`) and
   `test_recorded_frames_carry_facelets_that_follow_the_moves`
   (`tests/monitor/test_record_cube_trace.py`), the latter asserting the invariant across
   every consecutive pair of a real recorded episode.

   So move correctness IS tested, and net geometry IS tested (bijection, U above F, D
   below F, band order L F R B, contiguous 2x2 faces). What no test covers is the JOIN
   between them: that a facelet moved by a real permutation lands in the geometrically
   correct net cell. Both halves are pinned independently; their composition is not.
   Combined with not being able to drive `npm run dev` in a browser, "a recorded move
   visibly permutes the correct facelets" rests on those two verified halves rather than
   on a direct check.

   This is a different gap from `progress.md`'s Task 6 deferred minor (the held DLB corner
   highlight renders on facelets 12/16/21 but no test asserts it lands on exactly those
   three cells). Both are open; they are not the same item.
3. **MET.** `dashboard/public/week11_dashboard_trace.jsonl` still parses under the new
   contract: header `task.type == "gridworld"` with `grid_n: 5`, `schema_version` is the
   old `"1.0"` (expected: parse-time stamping is what lets old traces load), and frames
   carry `agent`/`goal`. `test_gridworld_header_unchanged_except_version_and_hash` and
   `test_gridworld_every_frame_is_field_identical` (`tests/monitor/test_gridworld_unchanged.py`)
   pin this at the library level; `parseTrace.test.ts` ("stamps gridworld frames from the
   header") and `TaskState.test.tsx` ("still renders the gridworld grid when the header
   says gridworld") pin it at the dashboard level. Visual rendering in a live browser is
   not separately confirmed (see criterion 2's caveat).
4. **MOSTLY MET, one gap.** `test_cube_action_labels_match_move_count`,
   `test_cube_frame_task_has_facelets_and_no_coordinates`, and
   `test_cube_header_omits_grid_n_and_declares_cube_n` all exist in
   `tests/monitor/test_tasks.py` and pass. The `cubeNet` bijection/band/contiguity tests
   and the out-of-range-facelet guard exist and pass. The gridworld-unchanged test and the
   parseTrace stamping/back-compat tests exist and pass. `parseTrace.test.ts` also has the
   "mismatch is loud" tests (throws when a cube header carries gridworld frames and vice
   versa). The one test named in section 6 that does not exist as specified is
   `test_cube_net_reflects_a_known_move`; see criterion 2. Full suites are green: python
   369 passed (>= 354 floor), dashboard 20 files / 79 tests passed, `npm run build` zero
   TypeScript errors.
5. **MET.** `src/neuromorphic/monitor/tasks.py` (adapters), `schema.py`, `frame.py`, and
   `runner.py` hold all reusable machinery; `scripts/record_cube_trace.py` is the only new
   file under `scripts/`, and it contains no logic beyond CLI plumbing into
   `record_policy_episode`.
