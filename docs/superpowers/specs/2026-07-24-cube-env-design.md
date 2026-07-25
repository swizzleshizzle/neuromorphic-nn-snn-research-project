# CubeEnv — 2x2 Pocket Cube Environment (design)

**Date:** 2026-07-24 · **Phase:** 3 kickoff (env slice) · **Grounds:** `docs/phase3-kickoff-brief.md`, `docs/architecture-spec-v3.md`, mirrors `src/neuromorphic/envs/gridworld.py`.

## Goal

A 2x2 Rubik's cube (Pocket Cube) as a Gymnasium environment that the five-region brain can drive, mirroring `GridWorldEnv`. It is the Phase-3 task environment: the agent learns to solve scrambled cubes from **raw facelet observations alone**. Built as reusable core architecture designed to extend to 3x3 and larger cubes.

## Placement (architectural rule)

This is **core library code**, not an experiment. It lives under `src/neuromorphic/`, exactly like `gridworld.py` and `encode_gridworld`. The `experiments/NNN_*` folders hold only one-off experiment *drivers* (e.g. the 024-028 runners); the env, the encoder, and the distance machinery are reusable infrastructure every future experiment imports. Nothing in this design goes under `experiments/`.

## The 2x2 cube (facts this design relies on)

- 8 corner cubies, no edges/centers. **24 facelets** (6 faces x 4 stickers).
- **6 moves** = 3 faces (U, R, F) x {CW, CCW} quarter turns. See "why 6, not 12" below.
- **3,674,160** reachable states. God's number = 11 (half-turn metric), 14 (quarter-turn metric).

### Why 6 moves, not 12 (a 2x2-only fact, verified)

A 2x2 has no centers and no middle layer, so a face turn is the **same physical act** as the
opposite face's counter-turn. Verified exhaustively over the 12-move set:

```
U = D'      R = L'      F = B'
U' = D      R' = L      F' = B
```

A 12-move action space is therefore exactly 2x redundant: every depth-1 scramble has **two**
solving actions (checked over 200 states, always exactly 2). That corrupts the action space
(two labels for one act), corrupts the baseline's chance level, and lets a scramble drift the
whole cube's orientation.

The standard fix, adopted here: **hold one corner still and turn only the three faces that
touch its opposite corner.** With the U/R/F move set, the DLB corner facelets `12, 16, 21`
are fixed by every move (verified), the orientation never drifts, and no two moves are
equivalent. The reachable state count is **unchanged at 3,674,160** and God's number is
**still 14** in the quarter-turn metric -- the redundancy was in the *labels*, not the states.

**This does not carry to 3x3.** A 3x3 has fixed centers, so all six faces are distinguishable
and D/L/B are genuinely distinct moves there. See §7.

## Design

### 1. State and moves — facelet representation

State is a length-24 array of color indices (0-5); the **observation is the state**. Facelet layout convention: faces ordered `U, R, F, D, L, B`; within a face, row-major from a fixed viewing orientation; face `f` occupies indices `[4f, 4f+4)`. **Solved** = every face uniform (`facelets[4f : 4f+4]` all equal, for each f).

Each of the 6 moves is a **fixed permutation of the 24 facelet indices**. The 3 clockwise face turns (U, R, F) are defined directly over the layout (a face turn 4-cycles that face's own 4 stickers and cycles the adjacent belt of 8 stickers); the 3 counter-clockwise moves are their inverses. Because the DLB corner is never touched, the solved state is unique among reachable states -- no whole-cube rotation is reachable.

*Why facelet, not cubie:* the observation falls out directly (no cubie->facelet layer), it is transparent, and it matches the 24-facelet sensory input exactly. It also generalizes to NxN (a bigger cube is just 6N^2 facelets). The cubie (permutation+orientation) representation is more compact but adds abstraction we do not need.

*Correctness strategy (where cube bugs hide):* the permutation tables are verified by **properties**, not eyeballing:
- `move` applied 4 times == identity (quarter turn has order 4);
- `move` then its inverse (CW then CCW of the same face) == identity;
- a scramble followed by the exact reversed-inverse move sequence returns to solved;
- every move is a bijection over `range(24)` (permutation invariant);
- **no two moves are physically identical** (the anti-redundancy property: for every pair
  `a != b`, applying `a` then `b`-inverse does not yield a solved cube);
- the DLB corner facelets `12, 16, 21` are fixed by every move;
- BFS level counts from solved match the published quarter-turn distribution
  `[1, 6, 27, 120, 534, 2256, 8969]`. This one test catches essentially any permutation error.

### 2. Env API — mirror `GridWorldEnv`

`gymnasium.Env` subclass:
- `action_space = spaces.Discrete(6)`; `observation_space = spaces.Box(low=0, high=5, shape=(24,), dtype=int64)`.
- `reset(seed, options) -> (obs, info)`: set state to a scramble of `scramble_depth` random moves from solved (seeded); return facelet obs + info.
- `step(move) -> (obs, reward, terminated, truncated, info)`: apply the move permutation; `terminated = solved`; `truncated = (not terminated) and steps >= max_steps` (Gymnasium treats the two as mutually exclusive).
- Constructor params mirror the grid: `scramble_depth: int = 1`, `step_penalty: float = -1.0`, `solve_reward: float = 10.0`, `max_steps: int = 50`, `reward_shaping: bool = False` (see §3), `shaping_gamma: float = 1.0`, plus `scramble_seed` for reproducible scrambles and `exact_depth: bool = False` (see below).

**Curriculum boundary:** the env takes a single `scramble_depth`; the **training loop drives the 1 -> 2 -> 3 curriculum** by constructing envs at increasing depth, exactly as the grid's loop drove the goal-set. The env stays dumb about curriculum.

**`scramble_depth` is a move count, not a distance (measured).** A random walk of `depth` moves can
revisit shorter-distance states -- `U U U` is three moves but distance 1. Filtering the immediate
self-inverse is not enough. With the 6-move set, measured true-distance contamination over 5000
samples per depth:

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| true distance < depth | 0% | 0% | 3.6% | 7.2% | 10% | 15% | 30% |

Depths 1 and 2 are exact for free; contamination grows with depth. This matters because the
Phase-3 baseline is read off *where performance collapses as a function of depth* -- a smeared
depth axis smears that reading. Two mitigations, both in scope:

- `scramble` **never returns an already-solved cube** (redraw). Unconditional, no distance needed.
- `exact_depth=True` requires a `distance_provider` and redraws until the true distance equals
  `depth`, giving an exact difficulty axis. Off by default; mirrors the `reward_shaping` guard.
  Without a provider, `scramble_depth` is documented as an **upper bound** on true distance.

`info` carries `{"solved": bool, "scramble_depth": int, "distance": int | None, "move": int | None, "move_label": str | None}` -- the §6 trace contract, so the monitor needs no cube-specific glue (`move`/`move_label` are `None` at `reset`). See §4 for distance.

### 3. Reward

Sparse by default, mirroring the grid: `step_penalty` (-1) per move, `solve_reward` (+10) on solved. `reward_shaping=False` by default.

`reward_shaping=True` is a **break-glass fallback only** — a potential-based warmer/colder signal from the distance table (§4), the cube analog of the grid's `manhattan` shaping. It is reserved for the case where traditional sparse sensory RL genuinely cannot learn; it is not a casual per-experiment toggle. Documented as such at the flag. The agent's *observation* never includes distance either way — shaping only enriches the reward.

### 4. Distance-to-solved — an optional, pluggable capability

A one-time **breadth-first search from the solved state** computes exact distance for every reachable state, keyed on the raw facelet tuple.

**No canonicalization is needed**, because the fixed-corner move set (§1) already quotients out whole-cube rotations: the reachable graph is exactly the 3,674,160 physically-distinct states, and no rotation of a state is reachable from it. This is a direct dividend of the 6-move decision. An earlier draft canonicalized every state under the 24 whole-cube rotations to undo the 12-move set's rotation drift; that cost 24 permutation applications *per generated state* and measured **~33 minutes** to build the table. Without it the same table builds in **~67 seconds** (measured), and `_ROT_X/_ROT_Y/_ROT_Z`, `_compose`, `_rotation_group` and `_canon` all disappear.

**Depth-bounded builds.** `ExactBFSDistance(max_depth=None)` optionally stops BFS at a depth. Bounded builds are effectively free and are what tests and shallow curricula should use; the full table is only needed to assert God's number. Measured:

| `max_depth` | 6 | 7 | 8 | `None` (full) |
|---|---|---|---|---|
| states | 11,913 | 44,971 | 159,120 | 3,674,160 |
| build | 0.04s | 0.17s | 0.67s | 67s |

A bounded table returns `None` for states beyond the bound -- which is exactly what the `int | None` provider contract already allows, so nothing special is needed downstream. Anything that *consumes* a distance must handle `None` (this includes `reward_shaping`, which must not arithmetic on it).

**This is deliberately behind a seam, because it does not generalize.** Define a minimal provider interface:

```python
class DistanceProvider(Protocol):
    def distance(self, facelets) -> int | None: ...
```

- 2x2: `ExactBFSDistance` builds (and caches) the table; `distance()` returns exact distance. It also exposes `table_size` and `max_distance` as **public** read-only properties, so tests and instrumentation never reach into `_table`.
- Future 3x3 (~4.3e19 states): exact BFS is impossible; a 3x3 supplies a heuristic provider or `None`. The rest of the system must not assume exact distance exists.

The env holds an **optional** distance provider. `info["distance"]` and the trace's `distance` are the provider's result or `None`. **Nothing downstream (training loop, encoder, dashboard) may require distance to be present.** This is the rule that stops the 2x2 convenience from becoming a load-bearing assumption that breaks on scale-up.

Note (reinforces §3): since exact distance cannot transfer to 3x3, the sparse "learn from facelets" path is also the one that generalizes; a distance-shaped crutch would not even be available on a bigger cube.

### 5. `encode_cube` — mirrors `encode_gridworld`, size-generic

Rate-encode the facelet observation into Poisson spikes, exactly as `encode_gridworld` does for the grid. **Written generically over cube size**, not hardcoded to 24: `n_facelets = 6 * n^2`, one-hot over `n_colors = 6`, giving `6 * n^2 * 6` inputs (144 for 2x2), Poisson-sampled at the same `max_rate` / `T` as the grid encoder. Signature parallels `encode_gridworld(obs, grid_n, T, max_rate, generator)`.

### 6. Cube trace / task contract (size-agnostic)

What the monitor emits per step so a cube trace renders (unblocks the dashboard cube task panel later). Kept generic over cube size and tolerant of a missing distance:

```
task = {
  "facelets": [int] * (6 * n^2),   # the raw sticker colors
  "solved": bool,
  "scramble_depth": int,
  "distance": int | null,          # provider result or null
  "move": int, "move_label": str,  # e.g. "R'"
  "reward": float, "return": float,
}
```

`facelets` is a list of any length (24 here, 54 for 3x3); `distance` is nullable. No field hardcodes the 2x2 size.

### 7. Extensibility: the 2x2 -> 3x3 delta (design-for, do not build now)

Built for 2x2 now; **not** building a speculative NxN move-generator tonight (YAGNI; 3x3 moves genuinely differ). The two size-specific pieces are isolated behind clean seams so 3x3 is an extension, not a rewrite:

| Concern | Generalizes as-is? | 3x3 delta |
|---|---|---|
| Facelet state / obs / solved / scramble | Yes (6N^2 facelets) | none (more stickers) |
| `encode_cube` | Yes (size-generic) | none (54 facelets -> 324 inputs) |
| Reward (sparse) | Yes | none |
| Trace/task contract | Yes (nullable distance, any-length facelets) | none |
| Gym API | Yes | none |
| **Move set** (size-specific seam #1) | No | **6 -> 12** quarter turns (all six faces become distinct); 18 if half-turns are included |
| **Action-space width** (follows the move set) | No | `Discrete(6)` -> `Discrete(12)`; Motor Cortex width follows |
| **Distance provider** (size-specific seam #2) | No | exact BFS impossible; supply heuristic provider or `None` |

**The 6-move restriction is 2x2-only and must not be carried forward.** It works here purely
because a 2x2 has no centers, which is what makes D/L/B redundant with U'/R'/F' (§1). A 3x3 has
**fixed centers**, so the six faces are distinguishable and D, L, B are genuinely distinct moves
with genuinely distinct effects -- a 3x3 agent must be able to turn all six faces, so its action
space is 12 quarter turns (or 18 in the half-turn metric). Anything that hardcodes `6` outside the
move table is a bug waiting for the 3x3. `n_actions` is read from `len(MOVES)`, never written as a
literal.

So a future `Cube3x3Env` is: a new move-permutation set (12/18 moves) + a different (or absent) distance provider, on the same env / obs / encode / trace contract.

## Testing

- **Move permutations:** the properties in §1 (order-4, inverse, scramble round-trip, bijection, **anti-redundancy**, fixed DLB corner), per move.
- **Env:** `reset` at depth k yields a scrambled (non-solved) state; a solving move terminates with `solve_reward`; `max_steps` truncates and `terminated`/`truncated` are never both true; obs shape/space; reproducibility under a fixed `scramble_seed`; `exact_depth=True` without a provider raises.
- **Distance provider:** solved -> 0; every depth-1 scramble -> distance 1; **BFS level counts match `[1,6,27,120,534,2256,8969]`** (the strongest single check); a bounded table returns `None` past its bound. Full-table facts (`table_size == 3,674,160`, `max_distance == 14`, all of `range(15)` present) live in one `@pytest.mark.slow` test, since that build is ~67s.
- **`encode_cube`:** output shape `[T, B, 6*n^2*6]`; only the active color bit per facelet fires; rate respects `max_rate`; reproducible under a fixed generator.

## File structure

- `src/neuromorphic/envs/cube.py` - `CubeEnv`, the 6 move permutations, scramble, solved-detection.
- `src/neuromorphic/envs/cube_distance.py` - `DistanceProvider` protocol + `ExactBFSDistance` (plain BFS, no canonicalization). Isolated so the expensive, size-specific piece is separable.
- `src/neuromorphic/envs/__init__.py` - export `CubeEnv` alongside `GridWorldEnv` (and add it to `__all__`).
- `src/neuromorphic/regions/sensory_cortex.py` - add `encode_cube` next to `encode_gridworld`.
- `pyproject.toml` - register the `slow` pytest marker.
- Tests: `tests/envs/test_cube.py`, `tests/envs/test_cube_distance.py`, `tests/regions/test_encode_cube.py` (mirroring existing test locations).

## Non-goals

- The v1-recipe baseline run itself (a separate experiment driver under `experiments/`, next step after the env exists).
- Any 3x3 implementation (design-for only).
- The distance-to-solved pre-training proxy (deferred per kickoff-brief D2 until evidence says the encoder binds).
- Dashboard cube task-panel rendering (separate follow-up; this spec only fixes the trace contract it will consume).

## Success criteria

1. `CubeEnv` solves-and-terminates on hand-constructed solutions; scrambles/round-trips are correct (property tests green).
2. **No two moves are physically identical**, and a depth-1 scramble has exactly one solving action.
3. `ExactBFSDistance` reproduces the published BFS level counts; the full 3,674,160-state table and God's number 14 are asserted in a `slow`-marked test.
4. `encode_cube` produces the 144-wide Poisson encoding, size-generically.
5. Distance is optional end-to-end (env runs, and the trace is well-formed, with `distance=None`).
6. `info` carries the full §6 trace contract, so the monitor needs no cube-specific glue.
7. All new code under `src/neuromorphic/`; nothing under `experiments/`.
