# NEURO·SCOPE Honest Pulse System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the hero's pathway pulses depict real activity — phase driven by the playback playhead (freezes on pause) and presence/count gated on real per-pathway intensity — instead of a wall-clock loop.

**Architecture:** Two pure, unit-tested helpers (`pulseCount`, `pulsePhase`) in `dashboard/src/hero/edges.ts` carry the math; `dashboard/src/hero/Pathways.tsx` calls them inside its `useFrame` loop, reading `winTi`/`T` from the Zustand store instead of `clock.elapsedTime`. Everything else in the hero is untouched.

**Tech Stack:** TypeScript (ES2020 target), React, @react-three/fiber (R3F), three.js, Zustand, Vitest.

## Global Constraints

- All dashboard commands run from the `dashboard/` directory.
- TypeScript target is **ES2020** — do NOT use ES2022 APIs (e.g. `Array.prototype.at()`). Only `npx tsc -b` catches this; vitest does not.
- R3F `<Canvas>` cannot render in jsdom — all testable logic goes in pure helpers, never asserted through the component. The component is covered only by the existing Playwright e2e (render-without-crash).
- Commit messages: plain, no `Co-Authored-By` trailer, no em-dashes.
- Constants: `THRESH = 0.05`, `PULSES = 3` (existing).
- Base branch: `main` at `f6b57a8`.
- Gates that must pass before the final commit: `npx tsc -b` clean, `npx vitest run` green, `npm run build` succeeds.

---

### Task 1: Pure pulse-math helpers in `edges.ts`

**Files:**
- Modify: `dashboard/src/hero/edges.ts` (append two exported functions)
- Test: `dashboard/src/hero/edges.test.ts` (append a `describe` block; file already exists)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pulseCount(inten: number, thresh: number, maxPulses: number): number` — returns `0` when `inten < thresh`, else `1..maxPulses`, monotonically non-decreasing in `inten`.
  - `pulsePhase(winTi: number, T: number, k: number, count: number, edgeOffset: number): number` — position in `[0, 1)` along the edge for pulse `k` of `count`, from the playhead `winTi`. `winTi = 0, k = 0 → 0`; identical `winTi` → identical output (freeze); always wrapped into `[0, 1)`.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/src/hero/edges.test.ts` (the file already imports `{ describe, expect, it } from "vitest"` at the top; do not re-import). Add:

```ts
import { pulseCount, pulsePhase } from "./edges";

describe("pulseCount", () => {
  it("returns 0 below threshold", () => {
    expect(pulseCount(0.04, 0.05, 3)).toBe(0);
    expect(pulseCount(0, 0.05, 3)).toBe(0);
  });

  it("returns 1 exactly at threshold", () => {
    expect(pulseCount(0.05, 0.05, 3)).toBe(1);
  });

  it("returns maxPulses at full intensity", () => {
    expect(pulseCount(1, 0.05, 3)).toBe(3);
  });

  it("is monotonically non-decreasing in intensity", () => {
    let prev = -1;
    for (let x = 0; x <= 1.0001; x += 0.05) {
      const c = pulseCount(x, 0.05, 3);
      expect(c).toBeGreaterThanOrEqual(prev);
      prev = c;
    }
  });
});

describe("pulsePhase", () => {
  it("is 0 at winTi=0 for pulse 0", () => {
    expect(pulsePhase(0, 32, 0, 3, 0)).toBe(0);
  });

  it("freezes: identical winTi gives identical phase", () => {
    const a = pulsePhase(7, 32, 1, 3, 0.21);
    const b = pulsePhase(7, 32, 1, 3, 0.21);
    expect(a).toBe(b);
  });

  it("always returns a value in [0, 1)", () => {
    for (let winTi = 0; winTi < 32; winTi++) {
      for (let k = 0; k < 3; k++) {
        const p = pulsePhase(winTi, 32, k, 3, 0.6);
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThan(1);
      }
    }
  });

  it("advances with the playhead", () => {
    expect(pulsePhase(16, 32, 0, 3, 0)).toBeCloseTo(0.5);
  });

  it("guards T=0 (no divide by zero)", () => {
    expect(pulsePhase(5, 0, 0, 1, 0)).toBe(0);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (from `dashboard/`): `npx vitest run src/hero/edges.test.ts`
Expected: FAIL — `pulseCount` / `pulsePhase` are not exported from `./edges`.

- [ ] **Step 3: Implement the helpers**

Append to `dashboard/src/hero/edges.ts` (after the existing `quadPoint` function):

```ts
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (from `dashboard/`): `npx vitest run src/hero/edges.test.ts`
Expected: PASS (all new `pulseCount` + `pulsePhase` cases green, existing edges tests still green).

- [ ] **Step 5: Typecheck**

Run (from `dashboard/`): `npx tsc -b`
Expected: clean (no output / exit 0). Confirms no ES2022 API slipped in.

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/hero/edges.ts dashboard/src/hero/edges.test.ts
git commit -m "feat(dashboard): pure pulse-math helpers (playhead phase + intensity count)"
```

---

### Task 2: Drive `Pathways.tsx` pulses from the playhead + intensity

**Files:**
- Modify: `dashboard/src/hero/Pathways.tsx` (the `useFrame` body + its store read + the import line)

**Interfaces:**
- Consumes: `pulseCount`, `pulsePhase` from Task 1; the store's `winTi` and `T` (already on the store: `useTraceStore.getState()` returns `{ frames, envStep, winTi, T, ... }`).
- Produces: nothing consumed by later tasks (final task).

- [ ] **Step 1: Add `THRESH` and import the helpers**

In `dashboard/src/hero/Pathways.tsx`, change the edges import to include the two helpers:

```ts
import { buildEdges, edgeState, pulseCount, pulsePhase, quadPoint } from "./edges";
```

And add a threshold constant next to the existing `SEGMENTS` / `PULSES` constants near the top of the file:

```ts
const SEGMENTS = 24;
const PULSES = 3;
const THRESH = 0.05; // below this per-pathway intensity, an edge shows no pulses (quiet = still)
```

- [ ] **Step 2: Rewrite the `useFrame` body to use the playhead + helpers**

Replace the entire `useFrame(({ clock }) => { ... });` block with the version below. The changes: read `winTi` and `T` from the store; drop `clock` and the `t` wall-clock; per edge compute `count = pulseCount(...)` and place `count` pulses via `pulsePhase(...)`; keep the line rendering, morph/bow, size, and instance-parking exactly as they were.

```tsx
  useFrame(() => {
    const { frames, envStep, winTi, T } = useTraceStore.getState();
    const frame: Frame | undefined = frames[envStep];
    const m = morphRef.current;
    let pulseI = 0;
    const pmesh = pulseRef.current;

    edges.forEach((e, ei) => {
      const ca = cloudC.get(e.src);
      const fa = flowC.get(e.src);
      const cb = cloudC.get(e.dst);
      const fb = flowC.get(e.dst);
      if (!ca || !fa || !cb || !fb) return;

      const a = lerpVec3(ca, fa, m);
      const b = lerpVec3(cb, fb, m);
      const bow = 0.5 * m; // straight in cloud, arced in flow

      const st = edgeState(frame?.pathways?.[e.id], e.gated);

      const lineObj = lineObjects[ei];
      if (lineObj) {
        const pts: THREE.Vector3[] = [];
        for (let s = 0; s <= SEGMENTS; s++) {
          const p = quadPoint(a, b, bow, s / SEGMENTS);
          pts.push(new THREE.Vector3(p[0], p[1], p[2]));
        }
        lineObj.geometry.setFromPoints(pts);
        lineObj.computeLineDistances(); // required for dashes after geometry changes
        const mat = lineObj.material as THREE.LineDashedMaterial;
        mat.color.set(hueFor(e.src, ei));
        mat.opacity = st.quiescent ? 0.16 : 0.06 + st.inten * 0.3;
        mat.gapSize = st.quiescent ? 0.05 : 0; // dashed when gated-closed, solid otherwise
      }

      // Pulses: phase from the playhead (winTi) so they freeze on pause; count
      // gated on real intensity so quiet pathways show none. edgeOffset is a
      // static per-edge phase offset for visual spacing only (encodes no data).
      const count = st.quiescent ? 0 : pulseCount(st.inten, THRESH, PULSES);
      if (pmesh && count > 0) {
        const edgeOffset = ei * 0.21;
        for (let k = 0; k < count; k++) {
          const pp = pulsePhase(winTi, T, k, count, edgeOffset);
          const p = quadPoint(a, b, bow, pp);
          dummy.position.set(p[0], p[1], p[2]);
          dummy.scale.setScalar(0.02 + st.inten * 0.05);
          dummy.updateMatrix();
          pmesh.setMatrixAt(pulseI++, dummy.matrix);
        }
      }
    });

    if (pmesh) {
      // Park unused instances at the origin with zero scale.
      for (let z = pulseI; z < edges.length * PULSES; z++) {
        dummy.position.set(0, 0, 0);
        dummy.scale.setScalar(0);
        dummy.updateMatrix();
        pmesh.setMatrixAt(z, dummy.matrix);
      }
      pmesh.instanceMatrix.needsUpdate = true;
    }
  });
```

- [ ] **Step 3: Typecheck**

Run (from `dashboard/`): `npx tsc -b`
Expected: clean (exit 0). If it complains that `clock` or `t` is unused, confirm they were fully removed from the `useFrame` signature and body (the new signature is `useFrame(() => {` with no argument).

- [ ] **Step 4: Run the full vitest suite**

Run (from `dashboard/`): `npx vitest run`
Expected: all green (the Task-1 helper tests plus every pre-existing suite; no test imports `Pathways` directly, so nothing else should change).

- [ ] **Step 5: Verify the production build**

Run (from `dashboard/`): `npm run build`
Expected: succeeds (tsc + vite build, exit 0).

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/hero/Pathways.tsx
git commit -m "feat(dashboard): pulses driven by playhead + intensity, not wall-clock"
```

---

## Manual verification (after both tasks, optional but recommended)

From `dashboard/`: `npm run dev`, load a trace, and confirm:
1. Pausing playback freezes all pulses; resuming moves them at the data tempo.
2. A pathway with intensity below `THRESH` shows only its static line (no pulses); higher-intensity pathways show more/bigger pulses.

(This is the same behavior the spec's success criteria 1-2 describe; it is not automatable through the jsdom test path.)
