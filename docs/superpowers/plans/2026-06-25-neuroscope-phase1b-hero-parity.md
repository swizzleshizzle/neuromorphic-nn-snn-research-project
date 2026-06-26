# NEURO·SCOPE Phase 1b: Hero Parity (real-3D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the minimal Phase-0 hero with a full real-3D R3F neuron cloud: glowing additive neurons with spike flash, pathway edges with travelling pulses, region/motor labels, a sensory-input overlay, and a smooth camera-morph transition to a flat Flow Map switchable by an on-stage Cloud/Flow toggle.

**Architecture:** All geometry and frame-state logic lives in pure, unit-tested helper functions under `dashboard/src/hero/`. The R3F component tree only wires those helpers to three.js objects. A single shared `morphRef` (eased each frame by a `MorphDriver`) drives neuron positions, edge geometry, and the camera together; `0` = Cloud, `1` = Flow. The imperative `useFrame` loops read `useTraceStore.getState()` and never call `setState`, so the canvas never re-renders per frame; only DOM overlays and labels subscribe reactively to `envStep`.

**Tech Stack:** Vite + React 18 + TypeScript 5 + Zustand 4 + react-three-fiber 8 + three 0.169 + `@react-three/drei` 9 (new) + Vitest 2 + Playwright 1.

**Design source:** `docs/superpowers/specs/2026-06-25-neuroscope-phase1b-hero-parity-design.md`. Hero reference code: `docs/handoffs/claude_design/NEURO-SCOPE Dashboard.dc.html` (`_layout`, `_drawCloud`, `_drawFlow`, `_edgeState`, `_neuronGlow`).

## Global Constraints

- **Data-driven only.** No hardcoded region ids, counts, or pathway ids in any logic or render path. Everything derives from the trace `header` and `Frame`. The five-region palette is the only region-keyed constant, and it has a hash fallback for unknown ids.
- **Pure helpers carry the tests.** All geometry/state functions (Tasks 1-4) are pure and unit-tested with Vitest, RED then GREEN. The R3F components (Tasks 5-8) cannot mount in jsdom (no WebGL / ResizeObserver), so they are verified by `npx tsc -b` + `npm run build` + the Playwright e2e + manual sign-off, NOT by unit tests. A missing unit test on a WebGL component is expected, not a defect.
- **Keep the `vi.mock` in `App.test.tsx`.** `vi.mock("./hero/Hero", () => ({ Hero: () => null }))` must remain; the real `<Canvas>` crashes jsdom.
- **Preserve the imperative/reactive split.** `useFrame` loops read `useTraceStore.getState()` and mutate three.js objects directly. They never call `setState`. Only `envStep`-scoped reactive subscriptions are allowed, and only in DOM overlays/labels (per-step, never per-frame).
- **Typecheck is a required gate.** Vitest/esbuild does not typecheck; run `npx tsc -b` (from `dashboard/`) on every task that touches `.ts`/`.tsx`.
- **Commit style:** plain messages, present-tense `feat:` / `test:` / `chore:` prefix. No `Co-Authored-By`, no "Generated with" trailers. No em-dashes in commit messages.
- **Coordinate convention:** all layout helpers emit normalized world coordinates (roughly `[-1, 1]` per axis), never pixels. three.js handles projection.
- **All commands run from `dashboard/`** unless stated otherwise.

---

## File Structure

```
dashboard/src/hero/
  layout.ts            # (MODIFY) shapeOf, buildHeroNeurons, clusterCentroids, types; keep isSpiking
  layout.test.ts       # (MODIFY) add cases for new helpers
  interp.ts            # (NEW) neuronGlow, lerpVec3, damp
  interp.test.ts       # (NEW)
  edges.ts             # (NEW) edgeState, buildEdges, quadPoint
  edges.test.ts        # (NEW)
  sensory.ts           # (NEW) aggregateSensoryGrid
  sensory.test.ts      # (NEW)
  glowTexture.ts       # (NEW) makeGlowTexture (browser-only canvas util, not unit-tested)
  Hero.tsx             # (REWRITE) Canvas + Scene + DOM overlays container
  Scene.tsx            # (NEW) composes in-canvas children, owns neuron memo + morphRef
  NeuronField.tsx      # (NEW) instanced additive neurons, useFrame
  Pathways.tsx         # (NEW) edges + travelling pulses, useFrame
  RegionLabels.tsx     # (NEW) drei <Html> region + motor labels, reactive to envStep
  CameraRig.tsx        # (NEW) OrbitControls + camera ease by morph
  MorphDriver.tsx      # (NEW) eases morphRef toward the store target each frame
  overlays/
    CloudFlowToggle.tsx # (NEW) on-stage Cloud/Flow segmented control
    SensoryGrid.tsx     # (NEW) sensory-input overlay, reactive to envStep
    HeroCaption.tsx     # (NEW) NEURON FIELD caption, reactive to layout/envStep
dashboard/src/store/traceStore.ts   # (MODIFY) add heroLayout + setHeroLayout
dashboard/e2e/smoke.spec.ts         # (MODIFY) assert toggle present + switches
```

Note on consolidation: the spec lists `buildCloudLayout` and `buildFlowLayout` separately. The plan consolidates them into a single `buildHeroNeurons(header)` pass that emits both `cloudPos` and `flowPos` per neuron. This is DRY (the shape-by-count logic is shared) and removes any risk of the two arrays drifting out of index order.

---

### Task 1: Layout helpers (`shapeOf`, `buildHeroNeurons`, `clusterCentroids`)

**Files:**
- Modify: `dashboard/src/hero/layout.ts`
- Test: `dashboard/src/hero/layout.test.ts`

**Interfaces:**
- Consumes: `TraceHeader` from `../contract`.
- Produces:
  - `type Vec3 = [number, number, number]`
  - `type Shape = "col" | "grid" | "disc"`
  - `interface HeroNeuron { region: string; idx: number; cloudPos: Vec3; flowPos: Vec3; r3: number }`
  - `function shapeOf(count: number): Shape`
  - `function buildHeroNeurons(header: TraceHeader): HeroNeuron[]`
  - `function clusterCentroids(neurons: HeroNeuron[], which: "cloud" | "flow"): Map<string, Vec3>`
  - (keep existing `isSpiking(frame, region, ti, idx)` exactly as-is)

- [ ] **Step 1: Write the failing tests**

Add to `dashboard/src/hero/layout.test.ts`. Leave the existing `buildNeurons` and `isSpiking` test blocks in place — `buildNeurons` stays exported until Task 5 rewrites `Hero.tsx`, so `tsc` stays clean through Tasks 1-4. Append the new blocks:

```ts
import { describe, expect, it } from "vitest";
import type { TraceHeader } from "../contract";
import { buildHeroNeurons, clusterCentroids, shapeOf } from "./layout";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 4 },
  task: { type: "gridworld", grid_n: 5, action_labels: [] },
  regions: [
    { id: "sensory", label: "Sensory Cortex", n_neurons: 4, role: "input", render: "dots" },
    { id: "router", label: "Thalamic Router", n_neurons: 9, role: "gating", render: "dots" },
    { id: "motor", label: "Motor", n_neurons: 5, role: "output", render: "dots" },
  ],
  pathways: [],
} as unknown as TraceHeader;

describe("shapeOf", () => {
  it("columns for small regions, grid for perfect squares, disc otherwise", () => {
    expect(shapeOf(5)).toBe("col");   // <= 8
    expect(shapeOf(8)).toBe("col");
    expect(shapeOf(9)).toBe("grid");  // perfect square > 8
    expect(shapeOf(12)).toBe("disc"); // non-square > 8
  });
});

describe("buildHeroNeurons", () => {
  const ns = buildHeroNeurons(header);

  it("emits one entry per neuron across all regions in header order", () => {
    expect(ns).toHaveLength(18); // 4 + 9 + 5
    expect(ns.filter((n) => n.region === "router")).toHaveLength(9);
    expect(ns[0]).toMatchObject({ region: "sensory", idx: 0 });
  });

  it("gives every neuron finite cloud and flow coordinates", () => {
    for (const n of ns) {
      for (const v of [...n.cloudPos, ...n.flowPos]) expect(Number.isFinite(v)).toBe(true);
      expect(n.flowPos[2]).toBe(0); // flow is a z=0 plane
    }
  });

  it("mean-centers the cloud near the origin", () => {
    const sum = ns.reduce((s, n) => [s[0] + n.cloudPos[0], s[1] + n.cloudPos[1], s[2] + n.cloudPos[2]], [0, 0, 0]);
    for (const axis of sum) expect(Math.abs(axis / ns.length)).toBeLessThan(1e-6);
  });

  it("spreads regions left-to-right by header order on the flow X axis", () => {
    const cx = (id: string) => {
      const pts = ns.filter((n) => n.region === id);
      return pts.reduce((s, n) => s + n.flowPos[0], 0) / pts.length;
    };
    expect(cx("sensory")).toBeLessThan(cx("router"));
    expect(cx("router")).toBeLessThan(cx("motor"));
  });
});

describe("clusterCentroids", () => {
  it("returns one centroid per region", () => {
    const c = clusterCentroids(buildHeroNeurons(header), "cloud");
    expect(c.size).toBe(3);
    expect(c.get("router")).toHaveLength(3);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/hero/layout.test.ts`
Expected: FAIL (`shapeOf`/`buildHeroNeurons`/`clusterCentroids` not exported).

- [ ] **Step 3: Implement the helpers**

In `dashboard/src/hero/layout.ts`, keep `isSpiking`, `buildNeurons`, and `NeuronPoint` exactly as they are (the Phase-0 `Hero.tsx` still imports them until Task 5). Append the new exports:

```ts
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/hero/layout.test.ts`
Expected: PASS.

- [ ] **Step 5: Typecheck**

Run: `npx tsc -b`
Expected: clean. `buildNeurons` and `isSpiking` remain exported, so the Phase-0 `Hero.tsx` still typechecks. Any error must be fixed now.

- [ ] **Step 6: Commit**

```bash
git add src/hero/layout.ts src/hero/layout.test.ts
git commit -m "feat: hero layout helpers (cloud + flow positions, centroids)"
```

---

### Task 2: Interpolation helpers (`neuronGlow`, `lerpVec3`, `damp`)

**Files:**
- Create: `dashboard/src/hero/interp.ts`
- Test: `dashboard/src/hero/interp.test.ts`

**Interfaces:**
- Consumes: `Frame` from `../contract`, `Vec3` from `./layout`.
- Produces:
  - `function neuronGlow(frame: Frame, region: string, idx: number, ti: number, T: number): { sp: number; act: number }`
  - `function lerpVec3(a: Vec3, b: Vec3, t: number): Vec3`
  - `function damp(current: number, target: number, lambda: number, dt: number): number`

- [ ] **Step 1: Write the failing tests**

Create `dashboard/src/hero/interp.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { Frame } from "../contract";
import { damp, lerpVec3, neuronGlow } from "./interp";

const frame = {
  field: { sensory: { spikes: [[1, 0], [0, 0], [0, 1], [0, 0]] } },
} as unknown as Frame;

describe("neuronGlow", () => {
  it("0.06 baseline, +0.5 on spike at ti, +0.18 on spike at ti-1 (wrapped)", () => {
    expect(neuronGlow(frame, "sensory", 0, 0, 4)).toMatchObject({ sp: 1 });
    expect(neuronGlow(frame, "sensory", 0, 0, 4).act).toBeCloseTo(0.56); // sp at 0, prev = ti-1 = 3 -> 0
    expect(neuronGlow(frame, "sensory", 0, 1, 4).act).toBeCloseTo(0.24); // ti=1 no spike, prev ti=0 spikes -> 0.06+0.18
    expect(neuronGlow(frame, "missing", 0, 0, 4)).toEqual({ sp: 0, act: 0.06 });
  });
});

describe("lerpVec3", () => {
  it("interpolates componentwise", () => {
    expect(lerpVec3([0, 0, 0], [2, 4, 6], 0.5)).toEqual([1, 2, 3]);
    expect(lerpVec3([1, 1, 1], [3, 3, 3], 0)).toEqual([1, 1, 1]);
  });
});

describe("damp", () => {
  it("stays at current when dt=0 and approaches target as dt grows", () => {
    expect(damp(0, 1, 6, 0)).toBe(0);
    expect(damp(0, 1, 6, 10)).toBeCloseTo(1, 5);
    const mid = damp(0, 1, 6, 0.1);
    expect(mid).toBeGreaterThan(0);
    expect(mid).toBeLessThan(1);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/hero/interp.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `dashboard/src/hero/interp.ts`:

```ts
import type { Frame } from "../contract";
import type { Vec3 } from "./layout";

/** Spike state + afterglow for one neuron at window step ti (ports the comp's _neuronGlow). */
export function neuronGlow(
  frame: Frame,
  region: string,
  idx: number,
  ti: number,
  T: number,
): { sp: number; act: number } {
  const arr = frame.field?.[region]?.spikes;
  if (!arr) return { sp: 0, act: 0.06 };
  const sp = arr[ti]?.[idx] ?? 0;
  const prev = (ti - 1 + T) % T;
  const spPrev = arr[prev]?.[idx] ?? 0;
  const act = 0.06 + (sp ? 0.5 : 0) + (spPrev ? 0.18 : 0);
  return { sp, act };
}

export function lerpVec3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/** Frame-rate-independent exponential approach toward target. */
export function damp(current: number, target: number, lambda: number, dt: number): number {
  return target + (current - target) * Math.exp(-lambda * dt);
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run src/hero/interp.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hero/interp.ts src/hero/interp.test.ts
git commit -m "feat: hero interpolation helpers (neuronGlow, lerpVec3, damp)"
```

---

### Task 3: Edge helpers (`edgeState`, `buildEdges`, `quadPoint`)

**Files:**
- Create: `dashboard/src/hero/edges.ts`
- Test: `dashboard/src/hero/edges.test.ts`

**Interfaces:**
- Consumes: `TraceHeader`, `PathwayState` from `../contract`; `Vec3` from `./layout`.
- Produces:
  - `interface HeroEdge { id: string; src: string; dst: string; gated: boolean }`
  - `interface EdgeState { inten: number; open: boolean; quiescent: boolean }`
  - `function edgeState(pw: PathwayState | undefined, gated: boolean): EdgeState`
  - `function buildEdges(header: TraceHeader): HeroEdge[]`
  - `function quadPoint(a: Vec3, b: Vec3, bow: number, t: number): Vec3` (quadratic bezier; control = midpoint offset perpendicular in XY by `bow`; `bow = 0` yields the straight segment)

- [ ] **Step 1: Write the failing tests**

Create `dashboard/src/hero/edges.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { PathwayState, TraceHeader } from "../contract";
import { buildEdges, edgeState, quadPoint } from "./edges";

describe("edgeState", () => {
  it("no pathway frame: open iff not gated", () => {
    expect(edgeState(undefined, false)).toEqual({ inten: 0, open: true, quiescent: false });
    expect(edgeState(undefined, true)).toEqual({ inten: 0, open: false, quiescent: true });
  });
  it("ungated pathway frame is always open", () => {
    expect(edgeState({ intensity: 0.4 } as PathwayState, false)).toEqual({ inten: 0.4, open: true, quiescent: false });
  });
  it("gate_open > 0.5 opens; array takes the max; <= 0.5 is quiescent", () => {
    expect(edgeState({ intensity: 0.2, gate_open: 0.8 } as PathwayState, true).open).toBe(true);
    expect(edgeState({ intensity: 0.2, gate_open: [0.1, 0.9] } as PathwayState, true).open).toBe(true);
    expect(edgeState({ intensity: 0.2, gate_open: 0.3 } as PathwayState, true)).toMatchObject({ open: false, quiescent: true });
  });
});

describe("buildEdges", () => {
  it("maps header pathways to edges", () => {
    const header = { pathways: [{ id: "a_b", src: "a", dst: "b", gated: true }] } as unknown as TraceHeader;
    expect(buildEdges(header)).toEqual([{ id: "a_b", src: "a", dst: "b", gated: true }]);
  });
});

describe("quadPoint", () => {
  it("endpoints at t=0 and t=1; bow=0 gives the straight midpoint", () => {
    expect(quadPoint([0, 0, 0], [2, 0, 0], 0, 0)).toEqual([0, 0, 0]);
    expect(quadPoint([0, 0, 0], [2, 0, 0], 0, 1)).toEqual([2, 0, 0]);
    expect(quadPoint([0, 0, 0], [2, 0, 0], 0, 0.5)).toEqual([1, 0, 0]);
  });
  it("bow offsets the midpoint perpendicular in XY", () => {
    const mid = quadPoint([0, 0, 0], [2, 0, 0], 1, 0.5);
    expect(mid[0]).toBeCloseTo(1);
    expect(Math.abs(mid[1])).toBeGreaterThan(0); // bowed off the straight line
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/hero/edges.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `dashboard/src/hero/edges.ts`:

```ts
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
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run src/hero/edges.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hero/edges.ts src/hero/edges.test.ts
git commit -m "feat: hero edge helpers (edgeState, buildEdges, quadPoint)"
```

---

### Task 4: Sensory aggregation (`aggregateSensoryGrid`)

**Files:**
- Create: `dashboard/src/hero/sensory.ts`
- Test: `dashboard/src/hero/sensory.test.ts`

**Interfaces:**
- Consumes: `Frame["encoding"]` from `../contract`.
- Produces: `function aggregateSensoryGrid(encoding: Frame["encoding"]): { agentCell: number; goalCell: number } | null` (returns `null` when no `sensory_input`; cell indices are `-1` when a plane never fires).

- [ ] **Step 1: Write the failing tests**

Create `dashboard/src/hero/sensory.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { Frame } from "../contract";
import { aggregateSensoryGrid } from "./sensory";

describe("aggregateSensoryGrid", () => {
  it("returns null when encoding is absent", () => {
    expect(aggregateSensoryGrid(undefined)).toBeNull();
  });

  it("argmax of the agent plane (first g*g) and goal plane (next g*g) over the window", () => {
    // grid_n = 2 -> 4 cells per plane, 8 columns total. agent fires cell 1, goal fires cell 3.
    const enc = {
      sensory_input: {
        grid_n: 2,
        planes: ["agent", "goal"],
        index: "row-major",
        spikes: [
          [0, 1, 0, 0, /*goal*/ 0, 0, 0, 1],
          [0, 1, 0, 0, /*goal*/ 0, 0, 0, 1],
        ],
      },
    } as unknown as Frame["encoding"];
    expect(aggregateSensoryGrid(enc)).toEqual({ agentCell: 1, goalCell: 3 });
  });

  it("reports -1 for a plane that never fires", () => {
    const enc = {
      sensory_input: { grid_n: 2, planes: ["agent", "goal"], index: "row-major", spikes: [[0, 0, 0, 0, 0, 0, 0, 0]] },
    } as unknown as Frame["encoding"];
    expect(aggregateSensoryGrid(enc)).toEqual({ agentCell: -1, goalCell: -1 });
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/hero/sensory.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `dashboard/src/hero/sensory.ts`:

```ts
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
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run src/hero/sensory.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hero/sensory.ts src/hero/sensory.test.ts
git commit -m "feat: hero sensory-input aggregation helper"
```

---

### Task 5: Cloud rendering (drei dep, NeuronField, Scene, Hero, CameraRig, fog)

**Verification note:** this task and Tasks 6-8 build WebGL components that cannot mount in jsdom. They are verified by `npx tsc -b` + `npm run build` + Playwright e2e + manual sign-off, not unit tests (see Global Constraints).

**Files:**
- Create: `dashboard/src/hero/glowTexture.ts`, `dashboard/src/hero/NeuronField.tsx`, `dashboard/src/hero/CameraRig.tsx`, `dashboard/src/hero/Scene.tsx`
- Rewrite: `dashboard/src/hero/Hero.tsx`
- Modify: `dashboard/package.json` (add `@react-three/drei`)

**Interfaces:**
- Consumes: `buildHeroNeurons`, `HeroNeuron` (Task 1); `neuronGlow`, `lerpVec3` (Task 2); `useTraceStore`.
- Produces: `<Hero />` default export shape unchanged (`export function Hero()`), so `Shell.tsx` keeps working. `Scene` accepts `{ neurons: HeroNeuron[]; morphRef: MutableRefObject<number> }`. `NeuronField` and `CameraRig` accept the same `morphRef`.

- [ ] **Step 1: Add the drei dependency**

Run: `npm install @react-three/drei@^9.114.0`
Expected: installs without peer-dependency errors against `@react-three/fiber@^8.17` and `three@^0.169`.

- [ ] **Step 2: Glow texture util**

Create `dashboard/src/hero/glowTexture.ts`:

```ts
import * as THREE from "three";

/** A soft radial-gradient sprite texture for additive neuron glow. Browser-only (uses a 2D canvas). */
export function makeGlowTexture(): THREE.Texture {
  const size = 64;
  const cv = document.createElement("canvas");
  cv.width = cv.height = size;
  const ctx = cv.getContext("2d")!;
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.35, "rgba(255,255,255,0.55)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(cv);
  tex.needsUpdate = true;
  return tex;
}
```

- [ ] **Step 3: NeuronField (instanced additive neurons)**

Create `dashboard/src/hero/NeuronField.tsx`:

```tsx
import { useFrame, useThree } from "@react-three/fiber";
import { type MutableRefObject, useMemo, useRef } from "react";
import * as THREE from "three";
import { useTraceStore } from "../store/traceStore";
import { makeGlowTexture } from "./glowTexture";
import { neuronGlow } from "./interp";
import type { HeroNeuron } from "./layout";

const HUE: Record<string, string> = {
  sensory: "#3fd2ff",
  hippocampus: "#ad8bff",
  prefrontal: "#ffd24a",
  router: "#ff5a8a",
  motor: "#46f0a0",
};
const hueFor = (id: string, i: number) => HUE[id] ?? `hsl(${(i * 67) % 360} 88% 66%)`;
const UNIT = 0.012; // world units per r3 unit
const white = new THREE.Color("#ffffff");

export function NeuronField({ neurons, morphRef }: { neurons: HeroNeuron[]; morphRef: MutableRefObject<number> }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const tex = useMemo(makeGlowTexture, []);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const col = useMemo(() => new THREE.Color(), []);
  const { camera } = useThree();

  const baseColors = useMemo(() => {
    const regionIndex = new Map<string, number>();
    neurons.forEach((n) => regionIndex.set(n.region, regionIndex.get(n.region) ?? regionIndex.size));
    return neurons.map((n) => new THREE.Color(hueFor(n.region, regionIndex.get(n.region) ?? 0)));
  }, [neurons]);

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh || neurons.length === 0) return;
    const { frames, envStep, winTi, T } = useTraceStore.getState();
    const frame = frames[envStep];
    const morph = morphRef.current;
    for (let i = 0; i < neurons.length; i++) {
      const n = neurons[i];
      dummy.position.set(
        n.cloudPos[0] + (n.flowPos[0] - n.cloudPos[0]) * morph,
        n.cloudPos[1] + (n.flowPos[1] - n.cloudPos[1]) * morph,
        n.cloudPos[2] + (n.flowPos[2] - n.cloudPos[2]) * morph,
      );
      dummy.quaternion.copy(camera.quaternion); // billboard
      const glow = frame ? neuronGlow(frame, n.region, i - 0, winTi, T) : { sp: 0, act: 0.06 };
      // glow.act uses idx within region; recompute idx-correct value:
      const g = frame ? neuronGlow(frame, n.region, n.idx, winTi, T) : { sp: 0, act: 0.06 };
      const flash = g.sp ? 1 : 0;
      const size = n.r3 * UNIT * (1 + flash * 1.3);
      dummy.scale.setScalar(size);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      col.copy(baseColors[i]).lerp(white, flash * 0.6).multiplyScalar(Math.min(1, g.act + flash * 0.95));
      mesh.setColorAt(i, col);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  if (neurons.length === 0) return null;
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, neurons.length]}>
      <planeGeometry args={[1, 1]} />
      <meshBasicMaterial
        map={tex}
        transparent
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        toneMapped={false}
      />
    </instancedMesh>
  );
}
```

Note: remove the dead `i - 0` line during implementation; it is shown only to flag that glow must use `n.idx` (per-region index), not the global loop index `i`. Final loop should compute `const g = neuronGlow(frame, n.region, n.idx, winTi, T)` once.

- [ ] **Step 4: CameraRig (OrbitControls auto-rotate, morph-aware)**

Create `dashboard/src/hero/CameraRig.tsx`:

```tsx
import { OrbitControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { type MutableRefObject, useRef } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

const CLOUD_POS = [0, 0, 3.2] as const;
const FLOW_POS = [0, 0, 3.0] as const;

export function CameraRig({ morphRef }: { morphRef: MutableRefObject<number> }) {
  const ref = useRef<OrbitControlsImpl>(null);
  const { camera } = useThree();

  useFrame(() => {
    const m = morphRef.current;
    const ctrl = ref.current;
    if (!ctrl) return;
    // auto-rotate fades out as we approach flow; drag-orbit disabled in flow
    ctrl.autoRotate = m < 0.05;
    ctrl.enabled = m < 0.5;
    if (m > 0.001) {
      // ease the camera toward a locked front view while in/approaching flow
      camera.position.x += (FLOW_POS[0] - camera.position.x) * m * 0.2;
      camera.position.y += (FLOW_POS[1] - camera.position.y) * m * 0.2;
      camera.position.z += (FLOW_POS[2] - camera.position.z) * m * 0.2;
      ctrl.target.x += (0 - ctrl.target.x) * m * 0.2;
      ctrl.target.y += (0 - ctrl.target.y) * m * 0.2;
    }
    ctrl.update();
  });

  return (
    <OrbitControls
      ref={ref}
      enablePan={false}
      autoRotate
      autoRotateSpeed={0.6}
      enableDamping
      minDistance={2}
      maxDistance={6}
      // start framing
      makeDefault
    />
  );
}
```

(Set the initial camera position via the `<Canvas camera>` prop in Hero, Step 6: `position: CLOUD_POS`.)

- [ ] **Step 5: Scene (composes in-canvas children)**

Create `dashboard/src/hero/Scene.tsx`:

```tsx
import { type MutableRefObject, useMemo } from "react";
import * as THREE from "three";
import { useTraceStore } from "../store/traceStore";
import { CameraRig } from "./CameraRig";
import { buildHeroNeurons } from "./layout";
import { NeuronField } from "./NeuronField";

export function Scene({ morphRef }: { morphRef: MutableRefObject<number> }) {
  const header = useTraceStore((s) => s.header);
  const neurons = useMemo(() => (header ? buildHeroNeurons(header) : []), [header]);
  return (
    <>
      <fog attach="fog" args={[new THREE.Color("#05060a"), 3.5, 7.5]} />
      <NeuronField neurons={neurons} morphRef={morphRef} />
      <CameraRig morphRef={morphRef} />
    </>
  );
}
```

- [ ] **Step 6: Rewrite Hero.tsx**

Replace `dashboard/src/hero/Hero.tsx` with:

```tsx
import { Canvas } from "@react-three/fiber";
import { useRef } from "react";
import { Scene } from "./Scene";

export function Hero() {
  const morphRef = useRef(0);
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <Canvas camera={{ position: [0, 0, 3.2], fov: 50 }} style={{ background: "var(--bg)" }} dpr={[1, 2]}>
        <Scene morphRef={morphRef} />
      </Canvas>
    </div>
  );
}
```

- [ ] **Step 7: Remove the obsolete Phase-0 layout helper**

Now that no component imports them, delete `buildNeurons` and the `NeuronPoint` interface from `dashboard/src/hero/layout.ts`, and remove their `describe("buildNeurons", ...)` block from `layout.test.ts`. Keep `isSpiking` and its test (still exported for reference; harmless).

- [ ] **Step 8: Typecheck and build**

Run: `npx tsc -b`
Expected: clean (no remaining `buildNeurons` references anywhere).

Run: `npm run build`
Expected: succeeds.

- [ ] **Step 9: Unit suite still green (App mock intact)**

Run: `npx vitest run`
Expected: PASS, including `App.test.tsx` with its `vi.mock("./hero/Hero")` unchanged.

- [ ] **Step 10: e2e + manual**

Run: `npm run e2e`
Expected: existing smoke passes (canvas visible).

Manual: `npm run dev`, confirm a glowing 3D neuron cloud orbits, flashes on spikes, and can be dragged to spin.

- [ ] **Step 11: Commit**

```bash
git add src/hero package.json package-lock.json
git commit -m "feat: real-3D neuron cloud hero (instanced additive neurons, orbit camera, fog)"
```

---

### Task 6: Pathways, pulses, region + motor labels

**Files:**
- Create: `dashboard/src/hero/Pathways.tsx`, `dashboard/src/hero/RegionLabels.tsx`
- Modify: `dashboard/src/hero/Scene.tsx` (mount the two new children)

**Interfaces:**
- Consumes: `buildEdges`, `edgeState`, `quadPoint` (Task 3); `clusterCentroids`, `HeroNeuron`, `Vec3` (Task 1); `lerpVec3` (Task 2); `useTraceStore`; drei `<Html>`.
- Produces: `<Pathways neurons morphRef />`, `<RegionLabels neurons />`.

- [ ] **Step 1: Pathways (edges + travelling pulses)**

Create `dashboard/src/hero/Pathways.tsx`:

```tsx
import { useFrame } from "@react-three/fiber";
import { type MutableRefObject, useMemo, useRef } from "react";
import * as THREE from "three";
import { useTraceStore } from "../store/traceStore";
import { buildEdges, edgeState, quadPoint } from "./edges";
import { clusterCentroids, type HeroNeuron, type Vec3 } from "./layout";

const HUE: Record<string, string> = {
  sensory: "#3fd2ff",
  hippocampus: "#ad8bff",
  prefrontal: "#ffd24a",
  router: "#ff5a8a",
  motor: "#46f0a0",
};
const SEGMENTS = 24;
const PULSES = 3;

export function Pathways({ neurons, morphRef }: { neurons: HeroNeuron[]; morphRef: MutableRefObject<number> }) {
  const header = useTraceStore((s) => s.header);
  const edges = useMemo(() => (header ? buildEdges(header) : []), [header]);
  const cloudC = useMemo(() => clusterCentroids(neurons, "cloud"), [neurons]);
  const flowC = useMemo(() => clusterCentroids(neurons, "flow"), [neurons]);

  const lineRefs = useRef<(THREE.Line | null)[]>([]);
  const pulseRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useFrame(({ clock }) => {
    const { frames, envStep } = useTraceStore.getState();
    const frame = frames[envStep];
    const m = morphRef.current;
    const t = clock.elapsedTime;
    let pulseI = 0;
    const pmesh = pulseRef.current;

    edges.forEach((e, ei) => {
      const ca = cloudC.get(e.src);
      const cb = cloudC.get(e.dst);
      const fa = flowC.get(e.src);
      const fb = flowC.get(e.dst);
      if (!ca || !cb || !fa || !fb) return;
      const a: Vec3 = [ca[0] + (fa[0] - ca[0]) * m, ca[1] + (fa[1] - ca[1]) * m, ca[2] + (fa[2] - ca[2]) * m];
      const b: Vec3 = [cb[0] + (fb[0] - cb[0]) * m, cb[1] + (fb[1] - cb[1]) * m, cb[2] + (fb[2] - cb[2]) * m];
      const bow = 0.5 * m; // straight in cloud, arced in flow
      const st = edgeState(frame?.pathways?.[e.id], e.gated);

      const line = lineRefs.current[ei];
      if (line) {
        const pts: THREE.Vector3[] = [];
        for (let s = 0; s <= SEGMENTS; s++) {
          const p = quadPoint(a, b, bow, s / SEGMENTS);
          pts.push(new THREE.Vector3(p[0], p[1], p[2]));
        }
        line.geometry.setFromPoints(pts);
        const mat = line.material as THREE.LineBasicMaterial;
        mat.color.set(HUE[e.src] ?? "#8aa");
        mat.opacity = st.quiescent ? 0.16 : 0.06 + st.inten * 0.3;
      }

      if (pmesh && !st.quiescent) {
        for (let k = 0; k < PULSES; k++) {
          let pp = (t * 0.18 + k / PULSES + ei * 0.21) % 1;
          if (pp < 0) pp += 1;
          const p = quadPoint(a, b, bow, pp);
          dummy.position.set(p[0], p[1], p[2]);
          dummy.scale.setScalar(0.02 + st.inten * 0.05);
          dummy.updateMatrix();
          pmesh.setMatrixAt(pulseI++, dummy.matrix);
        }
      }
    });
    if (pmesh) {
      // park unused instances at the origin with zero scale
      for (let z = pulseI; z < edges.length * PULSES; z++) {
        dummy.position.set(0, 0, 0);
        dummy.scale.setScalar(0);
        dummy.updateMatrix();
        pmesh.setMatrixAt(z, dummy.matrix);
      }
      pmesh.instanceMatrix.needsUpdate = true;
    }
  });

  return (
    <>
      {edges.map((e, ei) => (
        // @ts-expect-error three line element
        <line key={e.id} ref={(el) => (lineRefs.current[ei] = el as unknown as THREE.Line)}>
          <bufferGeometry />
          <lineBasicMaterial transparent depthWrite={false} blending={THREE.AdditiveBlending} />
        </line>
      ))}
      <instancedMesh ref={pulseRef} args={[undefined, undefined, Math.max(1, edges.length * PULSES)]}>
        <sphereGeometry args={[1, 6, 6]} />
        <meshBasicMaterial color="#ffffff" transparent depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
      </instancedMesh>
    </>
  );
}
```

If the `<line>` ref/material typing proves troublesome, an acceptable equivalent is `<primitive object={new THREE.Line(...)} />` built in a memo. Either renders the edges; choose whichever typechecks cleanly.

- [ ] **Step 2: RegionLabels (drei Html, reactive to envStep)**

Create `dashboard/src/hero/RegionLabels.tsx`:

```tsx
import { Html } from "@react-three/drei";
import { useMemo } from "react";
import { useTraceStore } from "../store/traceStore";
import { clusterCentroids, type HeroNeuron } from "./layout";

const HUE: Record<string, string> = {
  sensory: "#3fd2ff",
  hippocampus: "#ad8bff",
  prefrontal: "#ffd24a",
  router: "#ff5a8a",
  motor: "#46f0a0",
};

export function RegionLabels({ neurons }: { neurons: HeroNeuron[] }) {
  const header = useTraceStore((s) => s.header);
  const envStep = useTraceStore((s) => s.envStep);
  const frames = useTraceStore((s) => s.frames);
  const centroids = useMemo(() => clusterCentroids(neurons, "cloud"), [neurons]);
  if (!header) return null;
  const frame = frames[envStep];

  return (
    <>
      {header.regions.map((r) => {
        const c = centroids.get(r.id);
        if (!c) return null;
        const rate = frame?.regions?.[r.id]?.rate ?? 0;
        const hue = HUE[r.id] ?? "var(--text-dim)";
        return (
          <Html key={r.id} position={[c[0], c[1] - 0.45, c[2]]} center style={{ pointerEvents: "none" }}>
            <div style={{ textAlign: "center", whiteSpace: "nowrap" }}>
              <div style={{ font: "600 10px/1 'Space Grotesk', sans-serif", color: hue }}>
                {r.label.toUpperCase().split(" ")[0]}
              </div>
              <div style={{ font: "500 8px/1 'IBM Plex Mono', monospace", color: "var(--text-faint)", marginTop: 4 }}>
                {r.n_neurons}N · {rate.toFixed(2)}
              </div>
            </div>
          </Html>
        );
      })}
    </>
  );
}
```

(Motor per-action labels — selected `◂` / gated-shut `✕` — are added here in the same component, reading `frame.router.gate_open` and `frame.task.action`, anchored at the motor region centroid. Keep them in this file.)

- [ ] **Step 3: Mount in Scene**

Modify `dashboard/src/hero/Scene.tsx` to render `<Pathways neurons morphRef />` and `<RegionLabels neurons />` alongside `<NeuronField/>`.

- [ ] **Step 4: Typecheck, build, suite, e2e**

Run: `npx tsc -b` → clean.
Run: `npm run build` → succeeds.
Run: `npx vitest run` → PASS (App mock intact).
Run: `npm run e2e` → smoke passes.

Manual: `npm run dev`, confirm glowing edges connect clusters, pulses travel along them scaled by intensity, gated-closed edges are dim, and region/motor labels track the step.

- [ ] **Step 5: Commit**

```bash
git add src/hero/Pathways.tsx src/hero/RegionLabels.tsx src/hero/Scene.tsx
git commit -m "feat: hero pathways with travelling pulses and region/motor labels"
```

---

### Task 7: Flow morph + Cloud/Flow toggle

**Files:**
- Modify: `dashboard/src/store/traceStore.ts` (add `heroLayout` + `setHeroLayout`)
- Test: `dashboard/src/store/traceStore.test.ts` (add a case)
- Create: `dashboard/src/hero/MorphDriver.tsx`, `dashboard/src/hero/overlays/CloudFlowToggle.tsx`
- Modify: `dashboard/src/hero/Scene.tsx` (mount `<MorphDriver/>`), `dashboard/src/hero/Hero.tsx` (mount the toggle over the canvas)

**Interfaces:**
- Consumes: `damp` (Task 2); `useTraceStore`.
- Produces: store gains `heroLayout: "cloud" | "flow"` (default `"cloud"`) and `setHeroLayout(v)`. `<MorphDriver morphRef />` eases `morphRef.current` toward `heroLayout === "flow" ? 1 : 0`. `<CloudFlowToggle />` is a DOM control.

- [ ] **Step 1: Write the failing store test**

Add to `dashboard/src/store/traceStore.test.ts` (create the import if the file does not yet test default state):

```ts
import { useTraceStore } from "./traceStore";

it("toggles hero layout", () => {
  expect(useTraceStore.getState().heroLayout).toBe("cloud");
  useTraceStore.getState().setHeroLayout("flow");
  expect(useTraceStore.getState().heroLayout).toBe("flow");
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/store/traceStore.test.ts`
Expected: FAIL (`heroLayout` undefined).

- [ ] **Step 3: Extend the store**

In `dashboard/src/store/traceStore.ts`, add to the interface and the store object:

```ts
// interface:
heroLayout: "cloud" | "flow";
setHeroLayout: (v: "cloud" | "flow") => void;

// store body (initial state):
heroLayout: "cloud",
setHeroLayout: (v) => set({ heroLayout: v }),
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run src/store/traceStore.test.ts`
Expected: PASS.

- [ ] **Step 5: MorphDriver**

Create `dashboard/src/hero/MorphDriver.tsx`:

```tsx
import { useFrame } from "@react-three/fiber";
import type { MutableRefObject } from "react";
import { useTraceStore } from "../store/traceStore";
import { damp } from "./interp";

export function MorphDriver({ morphRef }: { morphRef: MutableRefObject<number> }) {
  useFrame((_, dt) => {
    const target = useTraceStore.getState().heroLayout === "flow" ? 1 : 0;
    morphRef.current = damp(morphRef.current, target, 4, dt);
  });
  return null;
}
```

- [ ] **Step 6: CloudFlowToggle (on-stage DOM control)**

Create `dashboard/src/hero/overlays/CloudFlowToggle.tsx`:

```tsx
import { useTraceStore } from "../../store/traceStore";

export function CloudFlowToggle() {
  const layout = useTraceStore((s) => s.heroLayout);
  const set = useTraceStore((s) => s.setHeroLayout);
  const seg = (active: boolean): React.CSSProperties => ({
    padding: "6px 11px",
    borderRadius: 6,
    border: "none",
    cursor: "pointer",
    font: "600 10px/1 'Space Grotesk', sans-serif",
    background: active ? "var(--c-router)" : "transparent",
    color: active ? "var(--text-bright)" : "var(--text-dim)",
  });
  return (
    <div
      data-hero-toggle
      style={{
        position: "absolute",
        top: 14,
        right: 18,
        display: "flex",
        gap: 3,
        padding: 3,
        borderRadius: 8,
        background: "var(--panel)",
        border: "1px solid var(--edge)",
        backdropFilter: "var(--blur)",
        zIndex: 10,
      }}
    >
      <button style={seg(layout === "cloud")} onClick={() => set("cloud")}>
        3D Cloud
      </button>
      <button style={seg(layout === "flow")} onClick={() => set("flow")}>
        Flow Map
      </button>
    </div>
  );
}
```

- [ ] **Step 7: Wire MorphDriver into Scene and the toggle into Hero**

Modify `Scene.tsx` to render `<MorphDriver morphRef={morphRef} />`. Modify `Hero.tsx` to render `<CloudFlowToggle />` as a sibling of `<Canvas>` inside the positioned wrapper div (so it overlays the canvas):

```tsx
return (
  <div style={{ position: "absolute", inset: 0 }}>
    <Canvas camera={{ position: [0, 0, 3.2], fov: 50 }} style={{ background: "var(--bg)" }} dpr={[1, 2]}>
      <Scene morphRef={morphRef} />
    </Canvas>
    <CloudFlowToggle />
  </div>
);
```

- [ ] **Step 8: Typecheck, build, suite, e2e**

Run: `npx tsc -b` → clean.
Run: `npm run build` → succeeds.
Run: `npx vitest run` → PASS.
Run: `npm run e2e` → smoke passes.

Manual: `npm run dev`, click Flow Map; confirm the cloud smoothly flattens into the left-to-right flow layout, edges arc, the camera eases to a front-on view, and clicking 3D Cloud morphs back with auto-rotate resuming.

- [ ] **Step 9: Commit**

```bash
git add src/store/traceStore.ts src/store/traceStore.test.ts src/hero/MorphDriver.tsx src/hero/overlays/CloudFlowToggle.tsx src/hero/Scene.tsx src/hero/Hero.tsx
git commit -m "feat: cloud/flow camera-morph toggle for the hero"
```

---

### Task 8: Sensory-grid and caption overlays

**Files:**
- Create: `dashboard/src/hero/overlays/SensoryGrid.tsx`, `dashboard/src/hero/overlays/HeroCaption.tsx`
- Modify: `dashboard/src/hero/Hero.tsx` (mount both over the canvas)

**Interfaces:**
- Consumes: `aggregateSensoryGrid` (Task 4); `useTraceStore`.
- Produces: `<SensoryGrid />`, `<HeroCaption />` (both DOM, reactive to `envStep`).

- [ ] **Step 1: SensoryGrid overlay**

Create `dashboard/src/hero/overlays/SensoryGrid.tsx`:

```tsx
import { useTraceStore } from "../../store/traceStore";
import { aggregateSensoryGrid } from "../sensory";

export function SensoryGrid() {
  const header = useTraceStore((s) => s.header);
  const envStep = useTraceStore((s) => s.envStep);
  const frames = useTraceStore((s) => s.frames);
  if (!header) return null;
  const frame = frames[envStep];
  const agg = aggregateSensoryGrid(frame?.encoding);
  if (!agg) return null;
  const g = frame!.encoding!.sensory_input.grid_n;
  const cells = Array.from({ length: g * g }, (_, c) => c);

  return (
    <div
      data-sensory-grid
      style={{
        position: "absolute",
        top: 44,
        left: 18,
        padding: "10px 11px",
        borderRadius: 10,
        background: "var(--panel)",
        border: "1px solid var(--edge)",
        backdropFilter: "var(--blur)",
        pointerEvents: "none",
        zIndex: 10,
      }}
    >
      <div style={{ font: "600 8px/1 'IBM Plex Mono', monospace", color: "var(--text-faint)", letterSpacing: ".12em", marginBottom: 7 }}>
        SENSORY INPUT · {g}×{g}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${g}, 10px)`, gap: 2 }}>
        {cells.map((c) => {
          const isA = c === agg.agentCell;
          const isG = c === agg.goalCell;
          return (
            <div
              key={c}
              style={{
                width: 10,
                height: 10,
                borderRadius: 2,
                background: isA ? "var(--c-sensory)" : "var(--edge)",
                boxShadow: isA ? "0 0 8px var(--c-sensory)" : isG ? "inset 0 0 0 1.5px var(--c-motor)" : "none",
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: HeroCaption overlay**

Create `dashboard/src/hero/overlays/HeroCaption.tsx`:

```tsx
import { useTraceStore } from "../../store/traceStore";

export function HeroCaption() {
  const header = useTraceStore((s) => s.header);
  const layout = useTraceStore((s) => s.heroLayout);
  if (!header) return null;
  const caption =
    layout === "cloud"
      ? "distributed neuron cloud · field spikes · auto-orbit"
      : header.regions.map((r) => r.label.split(" ")[0].toLowerCase()).join(" → ");
  return (
    <div
      data-hero-caption
      style={{
        position: "absolute",
        top: 14,
        left: 18,
        display: "flex",
        alignItems: "center",
        gap: 9,
        font: "500 9px/1 'IBM Plex Mono', monospace",
        color: "var(--text-faint)",
        letterSpacing: ".1em",
        pointerEvents: "none",
        zIndex: 10,
      }}
    >
      <span style={{ color: "var(--text-dim)" }}>NEURON FIELD</span>
      <span>{caption}</span>
    </div>
  );
}
```

- [ ] **Step 3: Mount both in Hero**

Modify `Hero.tsx` to render `<SensoryGrid />` and `<HeroCaption />` as siblings of `<Canvas>` (after `<CloudFlowToggle />`).

- [ ] **Step 4: Typecheck, build, suite, e2e**

Run: `npx tsc -b` → clean.
Run: `npm run build` → succeeds.
Run: `npx vitest run` → PASS.
Run: `npm run e2e` → smoke passes.

Manual: `npm run dev`, confirm the sensory-input grid highlights the agent cell (filled) and goal cell (ringed) and updates per step, and the caption switches text between Cloud and Flow.

- [ ] **Step 5: Commit**

```bash
git add src/hero/overlays/SensoryGrid.tsx src/hero/overlays/HeroCaption.tsx src/hero/Hero.tsx
git commit -m "feat: hero sensory-input grid and caption overlays"
```

---

### Task 9: E2E coverage for the toggle + final verification gate

**Files:**
- Modify: `dashboard/e2e/smoke.spec.ts`

**Interfaces:** none new.

- [ ] **Step 1: Extend the smoke spec**

Add to `dashboard/e2e/smoke.spec.ts`, inside the existing test (after the canvas assertion):

```ts
  // Hero Cloud/Flow toggle is present and switches
  const toggle = page.locator("[data-hero-toggle]");
  await expect(toggle).toBeVisible();
  await expect(toggle.getByRole("button", { name: "Flow Map" })).toBeVisible();
  await toggle.getByRole("button", { name: "Flow Map" }).click();
  // sensory-input overlay renders from encoding.sensory_input
  await expect(page.locator("[data-sensory-grid]")).toBeVisible();
```

- [ ] **Step 2: Run the e2e**

Run: `npm run e2e`
Expected: PASS (1 test).

- [ ] **Step 3: Full verification gate**

Run: `npx vitest run` → all green (helper suites + App mock intact).
Run: `npx tsc -b` → clean.
Run: `npm run build` → succeeds.
Run: `npm run e2e` → PASS.

- [ ] **Step 4: Commit**

```bash
git add e2e/smoke.spec.ts
git commit -m "test: e2e covers hero cloud/flow toggle and sensory overlay"
```

---

## Self-Review

**Spec coverage:**
- 3D Cloud (neurons, glow, flash, fog, orbit + drag): Tasks 1, 2, 5. ✓
- Pathways + pulses + gated-closed dashed/dim: Tasks 3, 6. ✓
- Region + motor labels: Task 6. ✓
- Flow layout + camera morph + on-stage toggle: Tasks 1, 7. ✓
- Sensory-input overlay + caption: Tasks 4, 8. ✓
- Data-driven, imperative/reactive split, jsdom isolation, typecheck gate: Global Constraints + every task's verify steps. ✓
- Dropped effects (grid, trail-fade) and hardcoded knob defaults: honored (no tasks add them). ✓
- Encoding key / theme / focus deferred to 1c: not in any task. ✓

**Placeholder scan:** the only non-final code is the explicitly-flagged `i - 0` teaching line in Task 5 Step 3 (with removal instructions) and the two named fallbacks (Task 6 `<primitive>` line alternative; spec §11 edge-opacity crossfade). No TBD/TODO/"handle edge cases".

**Type consistency:** `HeroNeuron`/`Vec3`/`Shape` (Task 1) are consumed with the same names in Tasks 2, 3, 5, 6. `buildHeroNeurons`, `clusterCentroids(which)`, `neuronGlow(frame, region, idx, ti, T)`, `edgeState(pw, gated)`, `quadPoint(a, b, bow, t)`, `aggregateSensoryGrid(encoding)`, `damp(current, target, lambda, dt)`, `lerpVec3(a, b, t)`, and the store `heroLayout`/`setHeroLayout` are referenced identically across tasks.

**Cross-task `tsc` cleanliness:** `buildNeurons`/`NeuronPoint` stay exported through Tasks 1-4 (the Phase-0 `Hero.tsx` still imports them), so every task's `npx tsc -b` gate is green. Task 5 Step 7 deletes them during the Hero rewrite, once nothing imports them. No task is left in a knowingly-broken typecheck state.
