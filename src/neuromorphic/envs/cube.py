"""2x2 Pocket Cube primitives: facelet state, the 6 moves, solved-detection, scramble.

State is a length-24 tuple of color indices (0-5); the observation is the state.
Faces are ordered U, R, F, D, L, B; face f occupies indices [4f, 4f+4).

Only the three faces U, R, F are turnable. A 2x2 has no centers, so turning a face is
the same physical act as counter-turning the opposite face (U == D', R == L', F == B');
offering all six faces would make the action space exactly 2x redundant. Holding the DLB
corner still (facelets 12, 16, 21 never move) removes that redundancy without removing any
state: the reachable set is still all 3,674,160 positions and God's number is still 14.
This is a 2x2-only simplification; a 3x3 has fixed centers and needs all six faces.

The move permutations below are pre-verified: they reproduce the published 2x2 quarter-turn
BFS level counts [1, 6, 27, 120, 534, 2256, 8969]. Applying permutation P to facelets f
yields ``tuple(f[P[i]] for i in range(24))``.
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Solved coloring: face f is uniformly color f.
SOLVED: tuple[int, ...] = (
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
)

# Clockwise quarter-turn of each turnable face (verified permutations).
_CW = {
    "U": (2, 0, 3, 1, 4, 5, 10, 11, 8, 9, 18, 19, 12, 13, 14, 15, 16, 17, 22, 23, 20, 21, 6, 7),
    "R": (0, 22, 2, 20, 6, 4, 7, 5, 8, 1, 10, 3, 12, 9, 14, 11, 16, 17, 18, 19, 15, 21, 13, 23),
    "F": (6, 4, 2, 3, 14, 5, 15, 7, 10, 8, 11, 9, 12, 13, 19, 17, 16, 0, 18, 1, 20, 21, 22, 23),
}

_FACE_ORDER = ["U", "R", "F"]

# Facelets no move touches (the held DLB corner). Asserted in tests.
FIXED_FACELETS: tuple[int, ...] = (12, 16, 21)


def _invert(P: tuple[int, ...]) -> tuple[int, ...]:
    Q = [0] * 24
    for i, p in enumerate(P):
        Q[p] = i
    return tuple(Q)


def _build_moves() -> tuple[list[tuple[int, ...]], list[str]]:
    moves: list[tuple[int, ...]] = []
    labels: list[str] = []
    for face in _FACE_ORDER:
        moves.append(_CW[face])
        labels.append(face)
        moves.append(_invert(_CW[face]))
        labels.append(face + "'")
    return moves, labels


# 6 moves as (CW, CCW) pairs per face -> actions 0..5.
MOVES, MOVE_LABELS = _build_moves()

# Action-space width. Read this, never the literal 6: a 3x3 move set is 12 or 18 wide.
N_ACTIONS: int = len(MOVES)


def inverse_action(action: int) -> int:
    """The action that undoes ``action`` (moves are stored as CW/CCW pairs)."""
    return action - 1 if action % 2 else action + 1


def apply_move(facelets: tuple[int, ...], action: int) -> tuple[int, ...]:
    """Apply move ``action`` to ``facelets``."""
    P = MOVES[action]
    return tuple(facelets[P[i]] for i in range(24))


def is_solved(facelets: tuple[int, ...]) -> bool:
    """True iff every face is a single color."""
    return all(
        facelets[f * 4] == facelets[f * 4 + 1] == facelets[f * 4 + 2] == facelets[f * 4 + 3]
        for f in range(6)
    )


def scramble(
    depth: int,
    rng: random.Random,
    provider=None,
    max_tries: int = 1000,
) -> tuple[int, ...]:
    """Return a state ``depth`` random moves from ``SOLVED``, never already solved.

    ``depth`` is a MOVE COUNT, not a distance. A random walk can revisit shorter-distance
    states (``U U U`` is three moves but distance 1), and skipping the immediate self-inverse
    does not prevent it. Measured contamination without a provider: 0% at depths 1-2, ~3.6%
    at depth 3, ~15% at depth 6. So without ``provider``, ``depth`` is an UPPER BOUND on the
    true distance.

    Args:
        depth: number of random moves to apply.
        rng: seeded RNG, for reproducible scrambles.
        provider: optional ``DistanceProvider``. When given, redraw until the true distance
            equals ``depth``, making the difficulty axis exact.
        max_tries: give up after this many redraws (raises).
    """
    if depth < 0:
        raise ValueError(f"scramble depth must be >= 0, got {depth}")
    if depth == 0:
        return SOLVED

    for _ in range(max_tries):
        f = SOLVED
        last = -1
        for _ in range(depth):
            blocked = inverse_action(last) if last >= 0 else -1
            a = rng.choice([m for m in range(N_ACTIONS) if m != blocked])
            f = apply_move(f, a)
            last = a
        if is_solved(f):
            continue
        if provider is None or provider.distance(f) == depth:
            return f
    raise RuntimeError(
        f"could not draw a scramble at depth {depth} in {max_tries} tries"
        + (" (exact depth may exceed God's number 14)" if provider is not None else "")
    )


class CubeEnv(gym.Env):
    """2x2 Pocket Cube as a Gymnasium env (mirrors GridWorldEnv).

    Observation is the 24 facelet colors. Action is one of the 6 quarter-turn moves.
    Sparse reward by default: ``step_penalty`` per move, ``solve_reward`` on solved.

    Args:
        scramble_depth: moves applied to build the start state (see ``scramble``).
        step_penalty: reward per non-solving step.
        solve_reward: reward on reaching solved.
        max_steps: episode truncates after this many steps.
        scramble_seed: seed for reproducible scrambles.
        reward_shaping: BREAK-GLASS FALLBACK ONLY. Potential-based warmer/colder from the
            distance provider, reserved for if sparse sensory RL genuinely cannot learn.
            Not a casual toggle. Requires a ``distance_provider``.
        shaping_gamma: discount for the shaping potential (mirrors GridWorldEnv).
        distance_provider: optional ``DistanceProvider``. When absent, ``info["distance"]``
            is ``None`` and nothing downstream may require it.
        exact_depth: redraw scrambles until the true distance equals ``scramble_depth``.
            Requires a ``distance_provider``. Off by default.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scramble_depth: int = 1,
        step_penalty: float = -1.0,
        solve_reward: float = 10.0,
        max_steps: int = 50,
        scramble_seed: int | None = None,
        reward_shaping: bool = False,
        shaping_gamma: float = 1.0,
        distance_provider=None,
        exact_depth: bool = False,
    ):
        super().__init__()
        if reward_shaping and distance_provider is None:
            raise ValueError(
                "reward_shaping=True requires a distance_provider (break-glass fallback)"
            )
        if exact_depth and distance_provider is None:
            raise ValueError("exact_depth=True requires a distance_provider")

        self.scramble_depth = scramble_depth
        self.step_penalty = step_penalty
        self.solve_reward = solve_reward
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.shaping_gamma = shaping_gamma
        self.distance_provider = distance_provider
        self.exact_depth = exact_depth

        self._rng = random.Random(scramble_seed)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=5, shape=(24,), dtype=np.int64)

        self._state: tuple[int, ...] = SOLVED
        self._steps = 0
        self._last_move: int | None = None
        self._prev_dist: int | None = None

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        return np.array(self._state, dtype=np.int64)

    def _distance(self) -> int | None:
        if self.distance_provider is None:
            return None
        return self.distance_provider.distance(self._state)

    def _info(self, distance: int | None) -> dict:
        """The spec's cube trace contract, so the monitor needs no cube-specific glue."""
        return {
            "solved": is_solved(self._state),
            "scramble_depth": self.scramble_depth,
            "distance": distance,
            "move": self._last_move,
            "move_label": None if self._last_move is None else MOVE_LABELS[self._last_move],
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        self._state = scramble(
            self.scramble_depth,
            self._rng,
            provider=self.distance_provider if self.exact_depth else None,
        )
        self._steps = 0
        self._last_move = None
        d = self._distance()
        self._prev_dist = d
        return self._obs(), self._info(d)

    def step(self, action: int):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"invalid action {action!r}; expected 0..{N_ACTIONS - 1}")

        self._last_move = int(action)
        self._state = apply_move(self._state, self._last_move)
        self._steps += 1

        terminated = is_solved(self._state)
        # Gymnasium treats terminated/truncated as mutually exclusive.
        truncated = (not terminated) and self._steps >= self.max_steps
        reward = self.solve_reward if terminated else self.step_penalty

        d = self._distance()
        if self.reward_shaping and not terminated:
            # Potential-based, mirroring GridWorldEnv's -manhattan potential.
            if d is not None and self._prev_dist is not None:
                reward += self.shaping_gamma * (-d) - (-self._prev_dist)
            self._prev_dist = d

        return self._obs(), float(reward), terminated, truncated, self._info(d)

    def render(self):
        return None
