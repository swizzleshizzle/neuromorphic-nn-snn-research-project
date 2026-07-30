import type { Frame } from "../contract";

/** Sum sensory_input spikes over the window; return the most-active agent and goal cell. */
export function aggregateSensoryGrid(
  encoding: Frame["encoding"],
): { agentCell: number; goalCell: number } | null {
  const si = encoding?.sensory_input;
  if (!si || !("grid_n" in si)) return null;
  const cells = si.grid_n * si.grid_n;
  const aSum = new Array(cells).fill(0);
  const gSum = new Array(cells).fill(0);
  for (const row of si.spikes ?? []) {
    for (let c = 0; c < cells; c++) {
      aSum[c] += row[c] || 0;
      gSum[c] += row[cells + c] || 0;
    }
  }
  let aMax = 0;
  let aCell = -1;
  let gMax = 0;
  let gCell = -1;
  for (let c = 0; c < cells; c++) {
    if (aSum[c] > aMax) {
      aMax = aSum[c];
      aCell = c;
    }
    if (gSum[c] > gMax) {
      gMax = gSum[c];
      gCell = c;
    }
  }
  return { agentCell: aMax > 0 ? aCell : -1, goalCell: gMax > 0 ? gCell : -1 };
}

/** Sum cube sensory spikes over the window; return the argmax color per facelet. */
export function aggregateCubeFacelets(encoding: Frame["encoding"]): number[] | null {
  const si = encoding?.sensory_input;
  if (!si || !("cube_n" in si)) return null;
  const nFacelets = 6 * si.cube_n * si.cube_n;
  const sums = Array.from({ length: nFacelets }, () => new Array(si.n_colors).fill(0));
  for (const row of si.spikes ?? []) {
    for (let f = 0; f < nFacelets; f++) {
      for (let c = 0; c < si.n_colors; c++) {
        sums[f][c] += row[f * si.n_colors + c] || 0;
      }
    }
  }
  return sums.map((counts) => {
    let best = 0;
    for (let c = 1; c < counts.length; c++) if (counts[c] > counts[best]) best = c;
    return best;
  });
}
