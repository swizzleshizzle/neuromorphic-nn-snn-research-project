# CubeEnv — 2x2 Pocket Cube Environment (design)

**Date:** 2026-07-24 · **Phase:** 3 kickoff (env slice) · **Grounds:** `docs/phase3-kickoff-brief.md`, `docs/architecture-spec-v3.md`, mirrors `src/neuromorphic/envs/gridworld.py`.

## Goal

A 2x2 Rubik's cube (Pocket Cube) as a Gymnasium environment that the five-region brain can drive, mirroring `GridWorldEnv`. It is the Phase-3 task environment: the agent learns to solve scrambled cubes from **raw facelet observations alone**. Built as reusable core architecture designed to extend to 3x3 and larger cubes.

## Placement (architectural rule)

This is **core library code**, not an experiment. It lives under `src/neuromorphic/`, exactly like `gridworld.py` and `encode_gridworld`. The `experiments/NNN_*` folders hold only one-off experiment *drivers* (e.g. the 024-028 runners); the env, the encoder, and the distance machinery are reusable infrastructure every future experiment imports. Nothing in this design goes under `experiments/`.

## The 2x2 cube (facts this design relies on)

- 8 corner cubies, no edges/centers. **24 facelets** (6 faces x 4 stickers).
- **12 moves** = 6 faces x {CW, CCW} quarter turns (matches the arch-spec's "motor 4 -> 12").
- **3,674,160** reachable states (physically distinct, quotienting whole-cube rotations). God's number = 11 (half-turn metric), 14 (quarter-turn metric).

## Design

### 1. State and moves — facelet representation

State is a length-24 array of color indices (0-5); the **observation is the state**. Facelet layout convention: faces ordered `U, R, F, D, L, B`; within a face, row-major from a fixed viewing orientation; face `f` occupies indices `[4f, 4f+4)`. **Solved** = every face uniform (`facelets[4f : 4f+4]` all equal, for each f).

Each of the 12 moves is a **fixed permutation of the 24 facelet indices**. The 6 clockwise face turns are defined directly over the layout (a face turn 4-cycles that face's own 4 stickers and cycles the adjacent belt of 8 stickers); the 6 counter-clockwise moves are their inverses.

*Why facelet, not cubie:* the observation falls out directly (no cubie->facelet layer), it is transparent, and it matches the 24-facelet sensory input exactly. It also generalizes to NxN (a bigger cube is just 6N^2 facelets). The cubie (permutation+orientation) representation is more compact but adds abstraction we do not need.

*Correctness strategy (where cube bugs hide):* the permutation tables are verified by **properties**, not eyeballing:
- `move` applied 4 times == identity (quarter turn has order 4);
- `move` then its inverse (CW then CCW of the same face) == identity;
- a scramble followed by the exact reversed-inverse move sequence returns to solved;
- every move is a bijection over `range(24)` (permutation invariant).

### 2. Env API — mirror `GridWorldEnv`

`gymnasium.Env` subclass:
- `action_space = spaces.Discrete(12)`; `observation_space = spaces.Box(low=0, high=5, shape=(24,), dtype=int64)`.
- `reset(seed, options) -> (obs, info)`: set state to a scramble of `scramble_depth` random moves from solved (seeded); return facelet obs + info.
- `step(move) -> (obs, reward, terminated, truncated, info)`: apply the move permutation; `terminated = solved`; `truncated = steps >= max_steps`.
- Constructor params mirror the grid: `scramble_depth: int = 1`, `step_penalty: float = -1.0`, `solve_reward: float = 10.0`, `max_steps: int = 50`, `reward_shaping: bool = False` (see §3), plus `scramble_seed` for reproducible scrambles.

**Curriculum boundary:** the env takes a single `scramble_depth`; the **training loop drives the 1 -> 2 -> 3 curriculum** by constructing envs at increasing depth, exactly as the grid's loop drove the goal-set. The env stays dumb about curriculum.

`info` carries `{"solved": bool, "scramble_depth": int, "distance": int | None}` (see §4 for distance).

### 3. Reward

Sparse by default, mirroring the grid: `step_penalty` (-1) per move, `solve_reward` (+10) on solved. `reward_shaping=False` by default.

`reward_shaping=True` is a **break-glass fallback only** — a potential-based warmer/colder signal from the distance table (§4), the cube analog of the grid's `manhattan` shaping. It is reserved for the case where traditional sparse sensory RL genuinely cannot learn; it is not a casual per-experiment toggle. Documented as such at the flag. The agent's *observation* never includes distance either way — shaping only enriches the reward.

### 4. Distance-to-solved — an optional, pluggable capability

A one-time **breadth-first search from the solved state** computes exact distance for every reachable state. To keep it at the true 3.67M physically-distinct states (the 6-face move set makes whole-cube rotations reachable, which would inflate the raw facelet-string graph ~24x), each state is **canonicalized under the 24 whole-cube rotations** (canonical form = lexicographically-minimal facelet string over the 24 rotations). BFS explores canonical states; a distance query canonicalizes the state, then looks it up.

**This is deliberately behind a seam, because it does not generalize.** Define a minimal provider interface:

```python
class DistanceProvider(Protocol):
    def distance(self, facelets) -> int | None: ...
```

- 2x2: `ExactBFSDistance` builds (and caches) the table; `distance()` returns exact distance.
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
| `encode_cube` | Yes (size-generic) | none |
| Reward (sparse) | Yes | none |
| Trace/task contract | Yes (nullable distance, any-length facelets) | none |
| Gym API | Yes | none |
| **Move set** (size-specific seam #1) | No | 18+ moves (add half-turns; optionally slice moves for N>=3); new permutation tables |
| **Distance provider** (size-specific seam #2) | No | exact BFS impossible; supply heuristic provider or `None` |

So a future `Cube3x3Env` is: a new move-permutation set + a different (or absent) distance provider, on the same env / obs / encode / trace contract.

## Testing

- **Move permutations:** the four properties in §1 (order-4, inverse, scramble round-trip, bijection), per move.
- **Env:** `reset` at depth k yields a scrambled (non-solved for k>=1) state; solving move sequence terminates with `solve_reward`; `max_steps` truncates; obs shape/space; reproducibility under a fixed `scramble_seed`.
- **Distance provider:** solved -> 0; a depth-1 scramble -> distance 1; all distances in `[0, 14]` (quarter-turn God's number); canonicalization maps a state and its 24 rotations to the same distance; table covers 3,674,160 states.
- **`encode_cube`:** output shape `[T, B, 6*n^2*6]`; only the active color bit per facelet fires; rate respects `max_rate`; reproducible under a fixed generator.

## File structure

- `src/neuromorphic/envs/cube.py` — `CubeEnv`, the 12 move permutations, scramble, solved-detection.
- `src/neuromorphic/envs/cube_distance.py` — `DistanceProvider` protocol + `ExactBFSDistance` (BFS + canonicalization). Isolated so the expensive, size-specific piece is separable.
- `src/neuromorphic/regions/sensory_cortex.py` — add `encode_cube` next to `encode_gridworld`.
- Tests: `tests/envs/test_cube.py`, `tests/envs/test_cube_distance.py`, `tests/regions/test_encode_cube.py` (mirroring existing test locations).

## Non-goals

- The v1-recipe baseline run itself (a separate experiment driver under `experiments/`, next step after the env exists).
- Any 3x3 implementation (design-for only).
- The distance-to-solved pre-training proxy (deferred per kickoff-brief D2 until evidence says the encoder binds).
- Dashboard cube task-panel rendering (separate follow-up; this spec only fixes the trace contract it will consume).

## Success criteria

1. `CubeEnv` solves-and-terminates on hand-constructed solutions; scrambles/round-trips are correct (property tests green).
2. `ExactBFSDistance` builds the full 3,674,160-state table; distances validated against known anchors.
3. `encode_cube` produces the 144-wide Poisson encoding, size-generically.
4. Distance is optional end-to-end (env runs, and the trace is well-formed, with `distance=None`).
5. All new code under `src/neuromorphic/`; nothing under `experiments/`.
