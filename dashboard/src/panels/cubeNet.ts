/** Unfolded-net geometry for a 2x2 cube.
 *
 * Face order matches neuromorphic.envs.cube: face f owns facelets [4f, 4f+4),
 * with U=0, R=1, F=2, D=3, L=4, B=5. The net is
 *
 *         U
 *   L  F  R  B
 *         D
 */
export const NET_ROWS = 6;
export const NET_COLS = 8;

export const FACE_OF = ["U", "R", "F", "D", "L", "B"] as const;

/** Top-left (row, col) of each face, indexed by face number. */
const FACE_ORIGIN: ReadonlyArray<readonly [number, number]> = [
  [0, 2], // U
  [2, 4], // R
  [2, 2], // F
  [4, 2], // D
  [2, 0], // L
  [2, 6], // B
];

/** Facelet index -> (row, col) in the net.
 *
 * WITHIN a face the model's row order is INVERTED relative to the net's, uniformly across all
 * six faces, which is why the row is `1 - (i >> 1)` and not `i >> 1`.
 *
 * That is not a guess. Every facelet on a 2x2 is a corner sticker, so two facelets touching
 * across a net border must be the same physical corner - and plain row-major satisfies that for
 * B alone. Searching all 24 within-face permutations per face against the corner constraints
 * derived from the move permutations in `src/neuromorphic/envs/cube.py` leaves 32 consistent
 * assignments; 8 of those also keep U and F (and F and D) index-aligned by column, and of those
 * exactly one is a single uniform rule applied to every face. This is it.
 *
 * The worked example, U/F border: U's F-side stickers are facelets 0 and 1 (corners UFL and
 * UFR), so they belong on U's BOTTOM row. F's U-side stickers are 10 and 11, which belong on F's
 * TOP row. Row-major puts both on the wrong side; inverting the row puts both right.
 */
export function cubeNetPosition(facelet: number): { row: number; col: number } {
  if (!Number.isInteger(facelet) || facelet < 0 || facelet > 23) {
    throw new Error(`cubeNetPosition: facelet must be an integer 0-23, got ${facelet}`);
  }
  const face = facelet >> 2;
  const i = facelet & 3;
  const [r0, c0] = FACE_ORIGIN[face];
  return { row: r0 + 1 - (i >> 1), col: c0 + (i & 1) };
}
