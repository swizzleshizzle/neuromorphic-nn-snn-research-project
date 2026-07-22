import type { PathwayState, TraceHeader } from "../contract";
import type { Vec3 } from "./layout";

export interface HeroEdge {
  id: string;
  src: string;
  dst: string;
  gated: boolean;
}

export interface EdgeState {
  inten: number;
  open: boolean;
  quiescent: boolean;
}

export function edgeState(pw: PathwayState | undefined, gated: boolean): EdgeState {
  if (!pw) return { inten: 0, open: !gated, quiescent: gated };
  const inten = pw.intensity || 0;
  if (pw.gate_open === undefined) return { inten, open: true, quiescent: false };
  const g = pw.gate_open;
  const frac = Array.isArray(g) ? Math.max(...g) : g;
  const open = frac > 0.5;
  return { inten, open, quiescent: !open };
}

export function buildEdges(header: TraceHeader): HeroEdge[] {
  return header.pathways.map((p) => ({ id: p.id, src: p.src, dst: p.dst, gated: p.gated }));
}

/** Quadratic bezier; control point = midpoint pushed perpendicular in the XY plane by `bow`. */
export function quadPoint(a: Vec3, b: Vec3, bow: number, t: number): Vec3 {
  const mx = (a[0] + b[0]) / 2;
  const my = (a[1] + b[1]) / 2;
  const mz = (a[2] + b[2]) / 2;
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  const px = -dy / len;
  const py = dx / len;
  const c: Vec3 = [mx + px * bow, my + py * bow, mz];
  const u = 1 - t;
  return [
    u * u * a[0] + 2 * u * t * c[0] + t * t * b[0],
    u * u * a[1] + 2 * u * t * c[1] + t * t * b[1],
    u * u * a[2] + 2 * u * t * c[2] + t * t * b[2],
  ];
}

/**
 * How many pulses an edge should show for a given intensity.
 * 0 below `thresh` (quiet edge is still); otherwise 1..maxPulses,
 * scaling with intensity above the threshold.
 */
export function pulseCount(inten: number, thresh: number, maxPulses: number): number {
  if (inten < thresh) return 0;
  const span = 1 - thresh;
  const norm = span > 0 ? Math.min(1, (inten - thresh) / span) : 1;
  return 1 + Math.round((maxPulses - 1) * norm);
}

/**
 * Position in [0,1) along the edge (src -> dst) for pulse `k` of `count`,
 * driven by the playback playhead `winTi` (0..T-1). Because it reads `winTi`
 * and not a wall clock, pulses freeze when playback is paused. `edgeOffset`
 * is a static per-edge phase offset for visual spacing only (encodes no data).
 */
export function pulsePhase(
  winTi: number,
  T: number,
  k: number,
  count: number,
  edgeOffset: number,
): number {
  const base = T > 0 ? winTi / T : 0;
  const stagger = count > 0 ? k / count : 0;
  let pp = (base + stagger + edgeOffset) % 1;
  if (pp < 0) pp += 1;
  return pp;
}
