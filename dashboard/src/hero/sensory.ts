import type { Frame } from "../contract";

/** Sum sensory_input spikes over the window; return the most-active agent and goal cell. */
export function aggregateSensoryGrid(
  encoding: Frame["encoding"],
): { agentCell: number; goalCell: number } | null {
  const si = encoding?.sensory_input;
  if (!si) return null;
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
