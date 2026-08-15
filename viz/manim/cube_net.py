"""Unfolded-net geometry for a 2x2 cube, and a self-check that it is actually right.

This is the Python twin of `dashboard/src/panels/cubeNet.ts`. The two must agree, because a
viewer who sees the dashboard and the video should be looking at the same cube.

    face order matches neuromorphic.envs.cube: face f owns facelets [4f, 4f+4),
    U=0, R=1, F=2, D=3, L=4, B=5

            U
      L  F  R  B
            D

WITHIN a face the model's row order is INVERTED relative to the net's, uniformly across all six
faces - hence `1 - (i >> 1)`. Plain row-major is wrong for five of the six, which is the bug
fixed on 2026-08-15 after sitting pinned behind an `it.fails` marker since 2026-08-03. It hid
because placement tests pin WHERE a face's block sits and never the orientation WITHIN it.

`verify()` re-derives the check rather than trusting the comment: on a 2x2 every facelet is a
corner sticker, so two facelets touching across a net border must belong to the same physical
corner. Run this file to check.
"""

from __future__ import annotations

NET_ROWS, NET_COLS = 6, 8

# Top-left (row, col) of each face's 2x2 block, indexed by face number.
FACE_ORIGIN = [(0, 2), (2, 4), (2, 2), (4, 2), (2, 0), (2, 6)]
FACE_OF = ("U", "R", "F", "D", "L", "B")

# The eight corners, one sticker per face. Derived 2026-08-03 from the move permutations in
# src/neuromorphic/envs/cube.py by closing the orbit of one corner under U/R/F and adding the
# fixed DLB corner, which no move touches. Copied from cubeNet.test.ts, which derived it.
CORNERS = ((0, 10, 19), (1, 6, 11), (2, 18, 23), (3, 7, 22),
           (4, 9, 15), (5, 13, 20), (8, 14, 17), (12, 16, 21))

# Faces sharing a border, as (above-or-left, below-or-right, kind).
BORDERS = ((0, 2, "v"), (2, 3, "v"), (4, 2, "h"), (2, 1, "h"), (1, 5, "h"))


def net_position(facelet: int) -> tuple[int, int]:
    """(row, col) of a facelet in the 6x8 net grid."""
    if not isinstance(facelet, int) or not 0 <= facelet <= 23:
        raise ValueError(f"facelet must be an integer 0-23, got {facelet!r}")
    face, i = facelet >> 2, facelet & 3
    r0, c0 = FACE_ORIGIN[face]
    return r0 + 1 - (i >> 1), c0 + (i & 1)


def verify() -> None:
    """Raise unless the net is a bijection, keeps faces contiguous, and joins corners correctly."""
    seen = {}
    for f in range(24):
        pos = net_position(f)
        if pos in seen:
            raise AssertionError(f"facelets {seen[pos]} and {f} both land on {pos}")
        seen[pos] = f
    if len(seen) != 24:
        raise AssertionError(f"expected 24 distinct cells, got {len(seen)}")

    for face in range(6):
        rows = {net_position(face * 4 + i)[0] for i in range(4)}
        cols = {net_position(face * 4 + i)[1] for i in range(4)}
        if len(rows) != 2 or len(cols) != 2:
            raise AssertionError(f"face {FACE_OF[face]} is not a contiguous 2x2 block")

    corner_of = {f: n for n, c in enumerate(CORNERS) for f in c}
    if len(corner_of) != 24:
        raise AssertionError("CORNERS does not cover all 24 facelets exactly once")

    for fa, fb, kind in BORDERS:
        pairs = []
        for ia in range(4):
            ra, ca = net_position(fa * 4 + ia)
            for ib in range(4):
                rb, cb = net_position(fb * 4 + ib)
                touching = (rb - ra == 1 and ca == cb) if kind == "v" else (
                    cb - ca == 1 and ra == rb)
                if touching:
                    pairs.append((fa * 4 + ia, fb * 4 + ib))
        if len(pairs) != 2:
            raise AssertionError(
                f"border {FACE_OF[fa]}/{FACE_OF[fb]} has {len(pairs)} touching pairs, expected 2")
        for a, b in pairs:
            if corner_of[a] != corner_of[b]:
                raise AssertionError(
                    f"facelets {a} and {b} touch across {FACE_OF[fa]}/{FACE_OF[fb]} "
                    f"but are different corners - the net is mis-oriented")


if __name__ == "__main__":
    verify()
    print("net geometry OK: bijection, contiguous faces, every border joins matching corners")
    for r in range(NET_ROWS):
        row = []
        for c in range(NET_COLS):
            f = next((f for f in range(24) if net_position(f) == (r, c)), None)
            row.append(f"{f:>2}" if f is not None else " .")
        print(" ".join(row))
