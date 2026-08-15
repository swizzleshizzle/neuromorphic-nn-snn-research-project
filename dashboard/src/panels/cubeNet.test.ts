import { describe, expect, it } from "vitest";
import { cubeNetPosition, NET_COLS, NET_ROWS } from "./cubeNet";

describe("cubeNetPosition", () => {
  it("is a bijection over all 24 facelets", () => {
    const seen = new Set<string>();
    for (let f = 0; f < 24; f++) {
      const { row, col } = cubeNetPosition(f);
      expect(row).toBeGreaterThanOrEqual(0);
      expect(row).toBeLessThan(NET_ROWS);
      expect(col).toBeGreaterThanOrEqual(0);
      expect(col).toBeLessThan(NET_COLS);
      seen.add(`${row},${col}`);
    }
    expect(seen.size).toBe(24);
  });

  it("places U in the top band above F", () => {
    // U = facelets 0-3, F = facelets 8-11, F sits directly below U.
    for (let i = 0; i < 4; i++) {
      const u = cubeNetPosition(i);
      const f = cubeNetPosition(8 + i);
      expect(u.col).toBe(f.col);
      expect(f.row - u.row).toBe(2);
    }
  });

  it("places D in the bottom band below F", () => {
    for (let i = 0; i < 4; i++) {
      const f = cubeNetPosition(8 + i);
      const d = cubeNetPosition(12 + i);
      expect(d.col).toBe(f.col);
      expect(d.row - f.row).toBe(2);
    }
  });

  it("orders the middle band L F R B left to right", () => {
    const col = (f: number) => cubeNetPosition(f).col;
    expect(col(16)).toBeLessThan(col(8));  // L before F
    expect(col(8)).toBeLessThan(col(4));   // F before R
    expect(col(4)).toBeLessThan(col(20));  // R before B
  });

  it("keeps every face a contiguous 2x2 block", () => {
    for (let face = 0; face < 6; face++) {
      const pos = [0, 1, 2, 3].map((i) => cubeNetPosition(face * 4 + i));
      const rows = new Set(pos.map((p) => p.row));
      const cols = new Set(pos.map((p) => p.col));
      expect(rows.size).toBe(2);
      expect(cols.size).toBe(2);
      expect(Math.max(...rows) - Math.min(...rows)).toBe(1);
      expect(Math.max(...cols) - Math.min(...cols)).toBe(1);
    }
  });

  it("rejects an out-of-range facelet", () => {
    expect(() => cubeNetPosition(-1)).toThrow();
    expect(() => cubeNetPosition(24)).toThrow();
    expect(() => cubeNetPosition(1.5)).toThrow();
    expect(() => cubeNetPosition(NaN)).toThrow();
  });
});

/**
 * Corner geometry, derived from the cube model rather than from the picture.
 *
 * The six tests above pin WHERE each face's 2x2 block sits and that it stays contiguous.
 * None of them pins the orientation of facelets WITHIN a face, so every one of them passes
 * whether a face is rendered upright, flipped or mirrored. That is the gap these add.
 *
 * CORNERS below is not hand-read off a diagram. It was derived on 2026-08-03 from the
 * pre-verified move permutations in src/neuromorphic/envs/cube.py, by closing the orbit of
 * one corner under the U/R/F position permutations and adding the fixed DLB corner (which no
 * move touches, so the orbit cannot reach it). It validates exactly: the eight triples use
 * three mutually non-opposite faces each, one sticker per face, covering all 24 facelets, and
 * match the eight geometric corners of a cube.
 */
const CORNERS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 10, 19],   // U F L
  [1, 6, 11],    // U R F
  [2, 18, 23],   // U L B
  [3, 7, 22],    // U R B
  [4, 9, 15],    // R F D
  [5, 13, 20],   // R D B
  [8, 14, 17],   // F D L
  [12, 16, 21],  // D L B  <- the held corner, no move touches it
];

const cornerOf = (facelet: number): number =>
  CORNERS.findIndex((c) => c.includes(facelet));

/** Faces sharing a border in the L F R B / U / D net, as [above-or-left, below-or-right]. */
const BORDERS: ReadonlyArray<readonly [number, number, "v" | "h"]> = [
  [0, 2, "v"],   // U above F
  [2, 3, "v"],   // F above D
  [4, 2, "h"],   // L left of F
  [2, 1, "h"],   // F left of R
  [1, 5, "h"],   // R left of B
];

function touchingPairs(faceA: number, faceB: number, kind: "v" | "h") {
  const pairs: Array<[number, number]> = [];
  for (let ia = 0; ia < 4; ia++) {
    const a = cubeNetPosition(faceA * 4 + ia);
    for (let ib = 0; ib < 4; ib++) {
      const b = cubeNetPosition(faceB * 4 + ib);
      const adjacent =
        kind === "v"
          ? b.row - a.row === 1 && a.col === b.col
          : b.col - a.col === 1 && a.row === b.row;
      if (adjacent) pairs.push([faceA * 4 + ia, faceB * 4 + ib]);
    }
  }
  return pairs;
}

describe("cube net corner geometry", () => {
  it("derives eight corners covering every facelet exactly once", () => {
    const seen = CORNERS.flatMap((c) => [...c]);
    expect(seen.length).toBe(24);
    expect(new Set(seen).size).toBe(24);
    expect(CORNERS.length).toBe(8);
  });

  it("puts the held DLB corner on three different faces", () => {
    // FIXED_FACELETS in src/neuromorphic/envs/cube.py. TaskState.tsx hardcodes the same
    // literal to draw the held-corner highlight, so this pins the two copies together.
    const faces = [12, 16, 21].map((f) => f >> 2);
    expect(new Set(faces).size).toBe(3);
    expect(cornerOf(12)).toBe(cornerOf(16));
    expect(cornerOf(16)).toBe(cornerOf(21));
  });

  it("gives every border exactly two touching facelet pairs", () => {
    for (const [a, b, kind] of BORDERS) {
      expect(touchingPairs(a, b, kind).length).toBe(2);
    }
  });

  /**
   * FIXED 2026-08-15. This carried `it.fails` from 2026-08-03 to pin the defect while the suite
   * stayed green; the marker is gone because the geometry is now right.
   *
   * On a 2x2 every facelet is a corner sticker, so two facelets touching across a net border are
   * the same physical corner. The old code mapped every face row-major (i -> [i>>1, i&1]), which
   * satisfies that for B alone. Worked example, the U/F border: U's F-side stickers are facelets
   * 0 and 1 (corners UFL and UFR), so they belong on U's BOTTOM row; row-major put them on the
   * top. F's U-side stickers are 10 and 11, which row-major put on the bottom rather than the
   * top. Inverting the within-face row fixes both, and the same single rule fixes every face.
   *
   * NOTE: the 2026-08-02 handoff recorded this as "B and D are probably mis-oriented". That was
   * close to backwards. Solving the border constraints gives 32 consistent assignments; B is the
   * ONLY face for which row-major appears in any of them, because this net constrains B through
   * R alone. U, R, F, D and L were all inconsistent - it was never a two-face problem.
   *
   * Of those 32, eight also keep the U/F and F/D column-alignment tests above green, and exactly
   * one of the eight is a uniform rule rather than a per-face table. No test was weakened to land
   * the fix, which was the thing worth checking before touching anything.
   */
  it("places facelets touching across a net border on the same corner", () => {
    for (const [a, b, kind] of BORDERS) {
      for (const [fa, fb] of touchingPairs(a, b, kind)) {
        expect(cornerOf(fa)).toBe(cornerOf(fb));
      }
    }
  });
});
