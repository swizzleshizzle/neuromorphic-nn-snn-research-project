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
