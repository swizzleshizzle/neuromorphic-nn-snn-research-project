import type { Frame, TraceHeader } from "../contract";

export interface NeuronPoint {
  region: string;
  idx: number;
  x: number; // world X (flow axis), region-ordered left -> right
  y: number; // world Y, stacked within the region's column
}

/**
 * Lay out every neuron of every region as a point. Phase-0 layout: regions in a
 * left-to-right column per `header.regions` order; neurons stacked vertically,
 * centered. Real shapes (disc, grid, cloud) are Phase 1.
 */
export function buildNeurons(header: TraceHeader): NeuronPoint[] {
  const points: NeuronPoint[] = [];
  const n = header.regions.length;
  header.regions.forEach((region, ri) => {
    const x = n > 1 ? (ri / (n - 1)) * 2 - 1 : 0; // [-1, 1]
    const count = region.n_neurons;
    for (let idx = 0; idx < count; idx++) {
      const y = count > 1 ? (idx / (count - 1) - 0.5) * 1.6 : 0;
      points.push({ region: region.id, idx, x, y });
    }
  });
  return points;
}

/** True if neuron `idx` of `region` spikes at window step `ti` in this frame. */
export function isSpiking(frame: Frame, region: string, ti: number, idx: number): boolean {
  const spikes = frame.field?.[region]?.spikes;
  const row = spikes?.[ti];
  return !!row && row[idx] === 1;
}
