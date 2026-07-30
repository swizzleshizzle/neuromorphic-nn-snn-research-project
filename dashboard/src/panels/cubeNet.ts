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

export function cubeNetPosition(facelet: number): { row: number; col: number } {
  if (!Number.isInteger(facelet) || facelet < 0 || facelet > 23) {
    throw new Error(`cubeNetPosition: facelet must be an integer 0-23, got ${facelet}`);
  }
  const face = facelet >> 2;
  const i = facelet & 3;
  const [r0, c0] = FACE_ORIGIN[face];
  return { row: r0 + (i >> 1), col: c0 + (i & 1) };
}
