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

1. `scripts/record_cube_trace.py` writes a loadable cube trace with 6 action labels and 24 facelets per frame.
2. The dashboard renders the unfolded net, and a recorded move visibly permutes the correct facelets.
3. The committed gridworld trace still loads and renders unchanged.
4. Every test in section 6 fails against pre-change code and passes after.
5. Reusable machinery under `src/neuromorphic/monitor/`; only the recorder under `scripts/`.
