import type { Frame, TraceHeader } from "../contract";

/** True if neuron `idx` of `region` spikes at window step `ti` in this frame. */
export function isSpiking(frame: Frame, region: string, ti: number, idx: number): boolean {
  const spikes = frame.field?.[region]?.spikes;
  const row = spikes?.[ti];
  return !!row && row[idx] === 1;
}

export type Vec3 = [number, number, number];
export type Shape = "col" | "grid" | "disc";

export interface HeroNeuron {
  region: string;
  idx: number;
  cloudPos: Vec3; // 3D cloud layout, mean-centered
  flowPos: Vec3;  // flat z=0 flow layout
  r3: number;     // relative size hint by shape
}

const GOLDEN = 2.39996323;

/** Deterministic RNG (mulberry32), so layouts are stable across runs and in tests. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shapeOf(count: number): Shape {
  if (count <= 8) return "col";
  return Number.isInteger(Math.sqrt(count)) ? "grid" : "disc";
}

export function buildHeroNeurons(header: TraceHeader): HeroNeuron[] {
  const regions = header.regions;
  const n = regions.length;
  const lr = rng(99);
  const neurons: HeroNeuron[] = [];

  regions.forEach((reg, ri) => {
    const cnt = reg.n_neurons;
    const sh = shapeOf(cnt);
    const cxX = -1.0 + 2.0 * (n > 1 ? ri / (n - 1) : 0.5);
    const cyY = ri % 2 ? 0.24 : -0.18;
    const czZ = (((ri * 0.41) % 1) - 0.5) * 0.7;
    const g = Math.round(Math.sqrt(cnt));
    const gden = Math.max(1, g - 1);
    const cden = Math.max(1, cnt - 1);

    for (let i = 0; i < cnt; i++) {
      let cloud: Vec3;
      let flow: Vec3;
      let r3: number;

      if (sh === "grid") {
        const r = Math.floor(i / g);
        const c = i % g;
        cloud = [cxX + (lr() - 0.5) * 0.05, cyY + (r / gden - 0.5) * 0.72, czZ + (c / gden - 0.5) * 0.72];
        flow = [cxX + (c / gden - 0.5) * 0.34, (r / gden - 0.5) * 0.72, 0];
        r3 = 2.3;
      } else if (sh === "disc") {
        const y0 = 1 - (i / cden) * 2;
        const rad = Math.sqrt(Math.max(0, 1 - y0 * y0));
        const th = i * GOLDEN;
        const jr = 0.45 + 0.55 * lr();
        cloud = [cxX + rad * Math.cos(th) * 0.44 * jr, cyY + y0 * 0.5 * jr, czZ + rad * Math.sin(th) * 0.5 * jr];
        const fr = Math.sqrt((i + 0.5) / cnt) * 0.26;
        flow = [cxX + fr * Math.cos(th), fr * Math.sin(th), 0];
        r3 = 1.9;
      } else {
        const ang = (i / cnt) * Math.PI * 2;
        cloud = [cxX + Math.cos(ang) * 0.13, cyY + Math.sin(ang) * 0.13, czZ + (i % 2 ? 0.09 : -0.09)];
        flow = [cxX, (i / cden - 0.5) * 0.72, 0];
        r3 = 5.0;
      }
      neurons.push({ region: reg.id, idx: i, cloudPos: cloud, flowPos: flow, r3 });
    }
  });

  // mean-center the cloud so it orbits about its own middle
  const mean: Vec3 = [0, 0, 0];
  for (const nn of neurons) for (let a = 0; a < 3; a++) mean[a] += nn.cloudPos[a];
  for (let a = 0; a < 3; a++) mean[a] /= neurons.length || 1;
  for (const nn of neurons) for (let a = 0; a < 3; a++) nn.cloudPos[a] -= mean[a];

  return neurons;
}

export function clusterCentroids(neurons: HeroNeuron[], which: "cloud" | "flow"): Map<string, Vec3> {
  const acc = new Map<string, { sum: Vec3; n: number }>();
  for (const nn of neurons) {
    const p = which === "cloud" ? nn.cloudPos : nn.flowPos;
    const e = acc.get(nn.region) ?? { sum: [0, 0, 0] as Vec3, n: 0 };
    e.sum[0] += p[0];
    e.sum[1] += p[1];
    e.sum[2] += p[2];
    e.n += 1;
    acc.set(nn.region, e);
  }
  const out = new Map<string, Vec3>();
  for (const [id, e] of acc) out.set(id, [e.sum[0] / e.n, e.sum[1] / e.n, e.sum[2] / e.n]);
  return out;
}
