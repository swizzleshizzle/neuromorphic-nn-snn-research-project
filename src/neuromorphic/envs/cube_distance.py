# src/neuromorphic/envs/cube_distance.py
"""Exact distance-to-solved for the 2x2 cube, behind a pluggable provider.

This is a small-cube luxury: the 3,674,160 reachable states are enumerable by BFS from
solved. Because the move set holds the DLB corner still (see ``cube``), whole-cube rotations
are unreachable and the raw facelet tuple is already a canonical key -- no canonicalization
pass is needed, which is what keeps the full build at ~67s instead of ~33 minutes.

Larger cubes (3x3 has ~4.3e19 states) supply a different provider or None; nothing downstream
may assume exact distance exists.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

from neuromorphic.envs.cube import MOVES, SOLVED, apply_move


class DistanceProvider(Protocol):
    def distance(self, facelets) -> int | None: ...


class ExactBFSDistance:
    """Exact quarter-turn distance-to-solved via BFS from solved.

    Args:
        max_depth: stop BFS at this depth. States beyond it return ``None`` from
            ``distance()`` -- allowed by the provider contract. Bounded builds are
            near-free (depth 6: 0.04s / 11,913 states) versus the full table
            (67s / 3,674,160 states), so tests and shallow curricula should bound it.
            ``None`` builds the full table.
    """

    def __init__(self, max_depth: int | None = None) -> None:
        self.max_depth = max_depth
        self._table: dict[tuple[int, ...], int] = {SOLVED: 0}
        frontier = deque([SOLVED])
        while frontier:
            state = frontier.popleft()
            d = self._table[state]
            if max_depth is not None and d >= max_depth:
                continue
            for a in range(len(MOVES)):
                nxt = apply_move(state, a)
                if nxt not in self._table:
                    self._table[nxt] = d + 1
                    frontier.append(nxt)

    @property
    def table_size(self) -> int:
        """Number of states in the table (3,674,160 for an unbounded build)."""
        return len(self._table)

    @property
    def max_distance(self) -> int:
        """Deepest distance in the table (14 for an unbounded build: God's number, QTM)."""
        return max(self._table.values())

    def level_counts(self) -> list[int]:
        """States at each distance, index = distance. Compared against published counts."""
        counts = [0] * (self.max_distance + 1)
        for d in self._table.values():
            counts[d] += 1
        return counts

    def distance(self, facelets) -> int | None:
        """Exact distance to solved, or ``None`` if beyond a bounded table."""
        return self._table.get(tuple(facelets))
