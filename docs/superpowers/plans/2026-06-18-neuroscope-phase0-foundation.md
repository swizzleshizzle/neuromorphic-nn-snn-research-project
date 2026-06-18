# NEURO·SCOPE Phase 0 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A real Vite + React + TypeScript app under `dashboard/` that replays the real `week11_dashboard_trace.jsonl` through the `TraceSource`/`TraceStore` seams and renders a data-driven shell, a minimal WebGL (R3F) hero, and two real panels.

**Architecture:** Frontend twin of the Python `TraceSink`: a `TraceSource` yields `{header, frames}`; a Zustand `TraceStore` holds replay/playback state; the hero runs an imperative `requestAnimationFrame` loop reading the store (never re-rendering React), while panels subscribe reactively to `envStep`. Everything renders from `header.regions[]`/`pathways[]` — no hardcoded region names.

**Tech Stack:** Vite 5, React 18, TypeScript 5, Zustand 4, three.js + @react-three/fiber 8, Vitest 2 + @testing-library/react + jsdom, Playwright 1.

**Reference spec:** `docs/superpowers/specs/2026-06-18-neuroscope-platform-design.md` (§9 is the Phase 0 scope).

---

## Conventions (read first)

- All commands run from `dashboard/` unless noted. The Python repo root is its parent.
- Package manager: **npm**. Node ≥ 20.
- Test command: `npm test` (= `vitest run`). Pure-logic tasks are TDD. React DOM components use `@testing-library/react`. The R3F hero (canvas) is covered by the Playwright smoke, not jsdom.
- Commit convention for this repo: **plain messages, no "Co-Authored-By" / no "Generated with" trailer** (matches existing repo style).
- If an exact dependency version fails to resolve, install the nearest compatible latest of the same major and note it in the commit — do not change major versions.
- The real trace lives at `outputs/week11_dashboard_trace.jsonl` (repo root, gitignored). Task 1 wires a sync script that copies it into `dashboard/public/`; if it is missing, regenerate via `python experiments/022_week11_dashboard_trace/run.py` from the repo root.

## File structure (what each file owns)

```
dashboard/
  package.json  tsconfig.json  tsconfig.node.json  vite.config.ts  index.html  .gitignore
  playwright.config.ts
  scripts/sync-trace.mjs            # copy repo-root trace into public/
  public/week11_dashboard_trace.jsonl   # synced, gitignored
  src/
    main.tsx                        # React entry
    App.tsx                         # loads TraceSource -> store, renders Shell
    vite-env.d.ts                   # import.meta.env typing
    test/setup.ts                   # jest-dom matchers
    contract.ts                     # Header/Frame TS types (mirror python schema)
    source/
      TraceSource.ts                # interface
      parseTrace.ts                 # pure JSONL parser
      FileTraceSource.ts            # fetch + parse + replay
      parseTrace.test.ts
    store/
      traceStore.ts                 # Zustand store
      traceStore.test.ts
    playback/
      advance.ts                    # pure window/episode advance reducer
      advance.test.ts
      usePlayback.ts                # rAF hook driving store.tickWindow
    hero/
      layout.ts                     # buildNeurons(header) + isSpiking(frame,...)
      layout.test.ts
      Hero.tsx                      # R3F canvas, imperative loop
    shell/
      Shell.tsx  TopBar.tsx  Scrubber.tsx
      TopBar.test.tsx
    panels/
      RegionActivity.tsx  TaskState.tsx
      RegionActivity.test.tsx  TaskState.test.tsx
  e2e/
    smoke.spec.ts                   # Playwright: boot on real trace
```

---

## Task 1: Scaffold `dashboard/` (Vite + React + TS + Vitest)

**Files:**
- Create: `dashboard/package.json`, `dashboard/tsconfig.json`, `dashboard/tsconfig.node.json`, `dashboard/vite.config.ts`, `dashboard/index.html`, `dashboard/.gitignore`, `dashboard/src/main.tsx`, `dashboard/src/vite-env.d.ts`, `dashboard/src/test/setup.ts`, `dashboard/src/sanity.test.ts`, `dashboard/scripts/sync-trace.mjs`

- [ ] **Step 1: Create `dashboard/package.json`**

```json
{
  "name": "neuroscope-dashboard",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "sync:trace": "node scripts/sync-trace.mjs",
    "e2e": "playwright test"
  },
  "dependencies": {
    "@react-three/fiber": "^8.17.10",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "three": "^0.169.0",
    "zustand": "^4.5.5"
  },
  "devDependencies": {
    "@playwright/test": "^1.48.0",
    "@testing-library/jest-dom": "^6.5.0",
    "@testing-library/react": "^16.0.1",
    "@types/react": "^18.3.11",
    "@types/react-dom": "^18.3.0",
    "@types/three": "^0.169.0",
    "@vitejs/plugin-react": "^4.3.2",
    "jsdom": "^25.0.1",
    "typescript": "^5.6.2",
    "vite": "^5.4.8",
    "vitest": "^2.1.2"
  }
}
```

- [ ] **Step 2: Create the config files**

`dashboard/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "types": ["vite/client", "vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

`dashboard/tsconfig.node.json`:
```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "noEmit": true
  },
  "include": ["vite.config.ts", "playwright.config.ts", "scripts"]
}
```

`dashboard/vite.config.ts`:
```ts
/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    exclude: ["e2e/**", "node_modules/**"],
  },
});
```

`dashboard/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NEURO·SCOPE</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`dashboard/.gitignore`:
```
node_modules
dist
.vite
public/*.jsonl
playwright-report
test-results
```

- [ ] **Step 3: Create the entry + test setup + a sanity test**

`dashboard/src/vite-env.d.ts`:
```ts
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TRACE_URL?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

`dashboard/src/main.tsx`:
```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

`dashboard/src/test/setup.ts`:
```ts
import "@testing-library/jest-dom";
```

`dashboard/src/sanity.test.ts`:
```ts
import { describe, expect, it } from "vitest";

describe("toolchain", () => {
  it("runs vitest", () => {
    expect(1 + 1).toBe(2);
  });
});
```

Note: `App` does not exist yet — it is created in Task 6. To keep Task 1 self-contained and green, create a **temporary placeholder** `dashboard/src/App.tsx` now (Task 6 replaces it):
```tsx
export function App() {
  return null;
}
```

- [ ] **Step 4: Create the trace sync script**

`dashboard/scripts/sync-trace.mjs`:
```js
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../outputs/week11_dashboard_trace.jsonl");
const dst = resolve(here, "../public/week11_dashboard_trace.jsonl");

if (!existsSync(src)) {
  console.error(`trace not found at ${src}\nrun: python experiments/022_week11_dashboard_trace/run.py`);
  process.exit(1);
}
mkdirSync(dirname(dst), { recursive: true });
copyFileSync(src, dst);
console.log(`synced trace -> ${dst}`);
```

- [ ] **Step 5: Install, sync trace, run the sanity test**

```bash
cd dashboard
npm install
npm run sync:trace
npm test
```
Expected: `npm install` succeeds; `sync:trace` prints `synced trace -> …/public/week11_dashboard_trace.jsonl`; `npm test` shows 1 passed (`sanity.test.ts`).

- [ ] **Step 6: Commit**

```bash
git add dashboard/package.json dashboard/package-lock.json dashboard/tsconfig*.json dashboard/vite.config.ts dashboard/index.html dashboard/.gitignore dashboard/src/main.tsx dashboard/src/App.tsx dashboard/src/vite-env.d.ts dashboard/src/test/setup.ts dashboard/src/sanity.test.ts dashboard/scripts/sync-trace.mjs
git commit -m "feat(dashboard): scaffold Vite+React+TS app with Vitest"
```

---

## Task 2: Contract types + JSONL parser + FileTraceSource

**Files:**
- Create: `dashboard/src/contract.ts`, `dashboard/src/source/TraceSource.ts`, `dashboard/src/source/parseTrace.ts`, `dashboard/src/source/FileTraceSource.ts`, `dashboard/src/source/parseTrace.test.ts`

- [ ] **Step 1: Write the failing test**

`dashboard/src/source/parseTrace.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { parseTrace } from "./parseTrace";

const HEADER = {
  schema_version: "1.0",
  brain: { id: "five-region", config_hash: "ab12cd34", seed: 0, T: 32 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [{ id: "sensory", label: "Sensory Cortex", n_neurons: 64, role: "input", render: "dots" }],
  pathways: [{ id: "sens_pfc", src: "sensory", dst: "prefrontal", gated: false }],
};
const FRAME = { episode: 0, step: 0, t: 0, task: {}, regions: {}, pathways: {}, router: {}, field: {} };

describe("parseTrace", () => {
  it("splits header (line 0) from frames", () => {
    const text = JSON.stringify(HEADER) + "\n" + JSON.stringify(FRAME) + "\n" + JSON.stringify(FRAME) + "\n";
    const trace = parseTrace(text);
    expect(trace.header.schema_version).toBe("1.0");
    expect(trace.header.regions[0].id).toBe("sensory");
    expect(trace.frames).toHaveLength(2);
  });

  it("ignores blank trailing lines", () => {
    const text = JSON.stringify(HEADER) + "\n" + JSON.stringify(FRAME) + "\n\n";
    expect(parseTrace(text).frames).toHaveLength(1);
  });

  it("throws on empty input", () => {
    expect(() => parseTrace("")).toThrow();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- parseTrace`
Expected: FAIL — cannot find module `./parseTrace`.

- [ ] **Step 3: Write the implementation**

`dashboard/src/contract.ts`:
```ts
export type RenderHint = "dots" | "cloud" | "density";

export interface Region {
  id: string;
  label: string;
  n_neurons: number;
  role: string;
  render: RenderHint;
}

export interface Pathway {
  id: string;
  src: string;
  dst: string;
  gated: boolean;
  label?: string;
}

export interface TraceHeader {
  schema_version: string;
  brain: { id: string; config_hash: string; seed: number; T: number };
  task: { type: string; grid_n: number; action_labels: string[] };
  regions: Region[];
  pathways: Pathway[];
}

export interface RegionState {
  rate: number;
  spikes: number;
  active_frac: number;
  rate_t: number[];
}

export interface PathwayState {
  intensity: number;
  gate_open?: number | number[];
}

export interface RouterState {
  gate_open: number[];
  gate_open_t: number[][];
  utilities: number[];
}

export interface FieldState {
  spikes: number[][]; // [T][N]
}

export interface TaskState {
  agent: [number, number];
  goal: [number, number];
  action: number;
  action_label: string;
  reward: number;
  return: number;
  terminated: boolean;
  truncated: boolean;
}

export interface SensoryInput {
  spikes: number[][]; // [T][2*grid_n^2]
  grid_n: number;
  planes: string[];
  index: string;
}

export interface Frame {
  episode: number;
  step: number;
  t: number;
  task: TaskState;
  regions: Record<string, RegionState>;
  pathways: Record<string, PathwayState>;
  router: RouterState;
  field: Record<string, FieldState>;
  encoding?: { sensory_input: SensoryInput };
}

export interface Trace {
  header: TraceHeader;
  frames: Frame[];
}
```

`dashboard/src/source/parseTrace.ts`:
```ts
import type { Frame, Trace, TraceHeader } from "../contract";

/** Parse JSONL trace text: line 0 = header, each remaining non-blank line = one Frame. */
export function parseTrace(text: string): Trace {
  const lines = text.split("\n").filter((l) => l.trim().length > 0);
  if (lines.length === 0) {
    throw new Error("parseTrace: empty trace (no header line)");
  }
  const header = JSON.parse(lines[0]) as TraceHeader;
  const frames = lines.slice(1).map((l) => JSON.parse(l) as Frame);
  return { header, frames };
}
```

`dashboard/src/source/TraceSource.ts`:
```ts
import type { Frame, TraceHeader } from "../contract";

/** Frontend twin of the Python TraceSink. The render path is agnostic to origin. */
export interface TraceSource {
  /** Resolve the run header. */
  open(): Promise<TraceHeader>;
  /** Emit frames: all-at-once for a file, live for a websocket. Call after open(). */
  subscribe(onFrame: (frame: Frame) => void): void;
  /** Release resources. */
  close(): void;
}
```

`dashboard/src/source/FileTraceSource.ts`:
```ts
import type { Frame, Trace, TraceHeader } from "../contract";
import { parseTrace } from "./parseTrace";
import type { TraceSource } from "./TraceSource";

/** Loads a JSONL trace by URL and replays its frames synchronously on subscribe. */
export class FileTraceSource implements TraceSource {
  private trace?: Trace;

  constructor(private readonly url: string) {}

  async open(): Promise<TraceHeader> {
    const res = await fetch(this.url);
    if (!res.ok) {
      throw new Error(`FileTraceSource: fetch ${this.url} failed (${res.status})`);
    }
    this.trace = parseTrace(await res.text());
    return this.trace.header;
  }

  subscribe(onFrame: (frame: Frame) => void): void {
    if (!this.trace) {
      throw new Error("FileTraceSource: call open() before subscribe()");
    }
    for (const frame of this.trace.frames) {
      onFrame(frame);
    }
  }

  close(): void {
    this.trace = undefined;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- parseTrace`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/contract.ts dashboard/src/source/
git commit -m "feat(dashboard): contract types, JSONL parser, FileTraceSource"
```

---

## Task 3: Zustand TraceStore

**Files:**
- Create: `dashboard/src/store/traceStore.ts`, `dashboard/src/store/traceStore.test.ts`
- Depends on: Task 4's `advancePlayback` is NOT needed here; the store's `tickWindow` is added in Task 4. This task ships `load`/`setEnvStep`/`play`/`pause`/`setWinTi`.

- [ ] **Step 1: Write the failing test**

`dashboard/src/store/traceStore.test.ts`:
```ts
import { beforeEach, describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "./traceStore";

const header = {
  schema_version: "1.0",
  brain: { id: "five-region", config_hash: "ab12cd34", seed: 0, T: 32 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [],
  pathways: [],
} as TraceHeader;

const frame = (step: number) =>
  ({ episode: 0, step, t: step, task: {}, regions: {}, pathways: {}, router: {}, field: {} }) as unknown as Frame;

describe("traceStore", () => {
  beforeEach(() => useTraceStore.getState().reset());

  it("load sets header, frames, T, and resets playhead", () => {
    useTraceStore.getState().load(header, [frame(0), frame(1)]);
    const s = useTraceStore.getState();
    expect(s.header?.brain.T).toBe(32);
    expect(s.frames).toHaveLength(2);
    expect(s.T).toBe(32);
    expect(s.envStep).toBe(0);
    expect(s.winTi).toBe(0);
  });

  it("setEnvStep clamps to frame range", () => {
    useTraceStore.getState().load(header, [frame(0), frame(1)]);
    useTraceStore.getState().setEnvStep(5);
    expect(useTraceStore.getState().envStep).toBe(1);
    useTraceStore.getState().setEnvStep(-3);
    expect(useTraceStore.getState().envStep).toBe(0);
  });

  it("play/pause toggle the flag", () => {
    useTraceStore.getState().play();
    expect(useTraceStore.getState().playing).toBe(true);
    useTraceStore.getState().pause();
    expect(useTraceStore.getState().playing).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- traceStore`
Expected: FAIL — cannot find module `./traceStore`.

- [ ] **Step 3: Write the implementation**

`dashboard/src/store/traceStore.ts`:
```ts
import { create } from "zustand";
import type { Frame, TraceHeader } from "../contract";

interface TraceStore {
  header?: TraceHeader;
  frames: Frame[];
  T: number;
  envStep: number;
  winTi: number;
  playing: boolean;

  load: (header: TraceHeader, frames: Frame[]) => void;
  setEnvStep: (i: number) => void;
  setWinTi: (ti: number) => void;
  play: () => void;
  pause: () => void;
  reset: () => void;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export const useTraceStore = create<TraceStore>((set, get) => ({
  frames: [],
  T: 1,
  envStep: 0,
  winTi: 0,
  playing: false,

  load: (header, frames) =>
    set({ header, frames, T: header.brain.T, envStep: 0, winTi: 0, playing: false }),

  setEnvStep: (i) => {
    const max = Math.max(0, get().frames.length - 1);
    set({ envStep: clamp(i, 0, max) });
  },

  setWinTi: (ti) => set({ winTi: clamp(ti, 0, Math.max(0, get().T - 1)) }),

  play: () => set({ playing: true }),
  pause: () => set({ playing: false }),

  reset: () => set({ header: undefined, frames: [], T: 1, envStep: 0, winTi: 0, playing: false }),
}));
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- traceStore`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/store/
git commit -m "feat(dashboard): Zustand TraceStore (load, playhead, play/pause)"
```

---

## Task 4: Playback advance reducer + rAF hook

**Files:**
- Create: `dashboard/src/playback/advance.ts`, `dashboard/src/playback/advance.test.ts`, `dashboard/src/playback/usePlayback.ts`
- Modify: `dashboard/src/store/traceStore.ts` (add `tickWindow` action)

- [ ] **Step 1: Write the failing test**

`dashboard/src/playback/advance.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { advancePlayback } from "./advance";

describe("advancePlayback", () => {
  it("advances the window playhead within T", () => {
    const r = advancePlayback({ winTi: 3, envStep: 0, T: 32, frameCount: 10 });
    expect(r).toEqual({ winTi: 4, envStep: 0 });
  });

  it("wraps the window and advances the episode at T boundary", () => {
    const r = advancePlayback({ winTi: 31, envStep: 0, T: 32, frameCount: 10 });
    expect(r).toEqual({ winTi: 0, envStep: 1 });
  });

  it("wraps the episode back to 0 at the last frame", () => {
    const r = advancePlayback({ winTi: 31, envStep: 9, T: 32, frameCount: 10 });
    expect(r).toEqual({ winTi: 0, envStep: 0 });
  });

  it("is safe with zero frames", () => {
    const r = advancePlayback({ winTi: 31, envStep: 0, T: 32, frameCount: 0 });
    expect(r).toEqual({ winTi: 0, envStep: 0 });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- advance`
Expected: FAIL — cannot find module `./advance`.

- [ ] **Step 3: Write the implementation**

`dashboard/src/playback/advance.ts`:
```ts
export interface PlaybackState {
  winTi: number;
  envStep: number;
  T: number;
  frameCount: number;
}

/** Advance the T-window playhead by one; wrap to the next episode frame at the boundary. */
export function advancePlayback(s: PlaybackState): { winTi: number; envStep: number } {
  const winTi = s.winTi + 1;
  if (winTi >= s.T) {
    const envStep = s.frameCount > 0 ? (s.envStep + 1) % s.frameCount : 0;
    return { winTi: 0, envStep };
  }
  return { winTi, envStep: s.envStep };
}
```

Add `tickWindow` to `dashboard/src/store/traceStore.ts`. Add the import at the top:
```ts
import { advancePlayback } from "../playback/advance";
```
Add `tickWindow: () => void;` to the `TraceStore` interface (after `pause`), and add the action in the store body (after `pause`):
```ts
  tickWindow: () =>
    set((s) => advancePlayback({ winTi: s.winTi, envStep: s.envStep, T: s.T, frameCount: s.frames.length })),
```

`dashboard/src/playback/usePlayback.ts`:
```ts
import { useEffect } from "react";
import { useTraceStore } from "../store/traceStore";

const STEP_HZ = 7; // ~7 window steps / second

/** Drives the store's window playhead with a single rAF loop while `playing`. */
export function usePlayback(): void {
  const playing = useTraceStore((s) => s.playing);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const stepDur = 1 / STEP_HZ;

    const tick = (now: number) => {
      acc += (now - last) / 1000;
      last = now;
      while (acc >= stepDur) {
        acc -= stepDur;
        useTraceStore.getState().tickWindow();
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- advance traceStore`
Expected: PASS (advance: 4, traceStore: 3 — the store still compiles with the new action).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/playback/ dashboard/src/store/traceStore.ts
git commit -m "feat(dashboard): playback advance reducer + rAF hook"
```

---

## Task 5: Hero layout helpers

**Files:**
- Create: `dashboard/src/hero/layout.ts`, `dashboard/src/hero/layout.test.ts`

These pure helpers are the testable core of the hero; the R3F component (Task 9) is a thin shell over them.

- [ ] **Step 1: Write the failing test**

`dashboard/src/hero/layout.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { buildNeurons, isSpiking } from "./layout";

const header = {
  schema_version: "1.0",
  brain: { id: "five-region", config_hash: "x", seed: 0, T: 4 },
  task: { type: "gridworld", grid_n: 5, action_labels: [] },
  regions: [
    { id: "sensory", label: "Sensory", n_neurons: 2, role: "input", render: "dots" },
    { id: "motor", label: "Motor", n_neurons: 3, role: "output", render: "dots" },
  ],
  pathways: [],
} as TraceHeader;

describe("buildNeurons", () => {
  it("emits one entry per neuron across all regions, in flow order", () => {
    const ns = buildNeurons(header);
    expect(ns).toHaveLength(5); // 2 + 3
    expect(ns.filter((n) => n.region === "sensory")).toHaveLength(2);
    expect(ns[0]).toMatchObject({ region: "sensory", idx: 0 });
    // x increases left-to-right by region order
    const sx = ns.find((n) => n.region === "sensory")!.x;
    const mx = ns.find((n) => n.region === "motor")!.x;
    expect(mx).toBeGreaterThan(sx);
  });
});

describe("isSpiking", () => {
  const frame = {
    field: { sensory: { spikes: [[1, 0], [0, 0], [0, 1], [0, 0]] } },
  } as unknown as Frame;

  it("reads field[region].spikes[ti][idx]", () => {
    expect(isSpiking(frame, "sensory", 0, 0)).toBe(true);
    expect(isSpiking(frame, "sensory", 1, 0)).toBe(false);
    expect(isSpiking(frame, "sensory", 1, 2)).toBe(true);
  });

  it("returns false when the region or index is absent", () => {
    expect(isSpiking(frame, "missing", 0, 0)).toBe(false);
    expect(isSpiking(frame, "sensory", 0, 9)).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- layout`
Expected: FAIL — cannot find module `./layout`.

- [ ] **Step 3: Write the implementation**

`dashboard/src/hero/layout.ts`:
```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- layout`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/hero/layout.ts dashboard/src/hero/layout.test.ts
git commit -m "feat(dashboard): hero layout helpers (buildNeurons, isSpiking)"
```

---

## Task 6: App data-load wiring

**Files:**
- Modify: `dashboard/src/App.tsx` (replace the Task 1 placeholder)
- Create: `dashboard/src/App.test.tsx`

- [ ] **Step 1: Write the failing test**

`dashboard/src/App.test.tsx`:
```tsx
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { useTraceStore } from "./store/traceStore";

const header = {
  schema_version: "1.0",
  brain: { id: "five-region", config_hash: "ab12cd34", seed: 0, T: 32 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [{ id: "sensory", label: "Sensory Cortex", n_neurons: 64, role: "input", render: "dots" }],
  pathways: [],
};
const frame = { episode: 0, step: 0, t: 0, task: { agent: [0, 0], goal: [4, 4], action: 0, action_label: "up", reward: -1, return: -1, terminated: false, truncated: false }, regions: { sensory: { rate: 0.2, spikes: 1, active_frac: 0.1, rate_t: [] } }, pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: { sensory: { spikes: [[0]] } } };

afterEach(() => {
  useTraceStore.getState().reset();
  vi.restoreAllMocks();
});

describe("App", () => {
  it("loads the trace through a TraceSource into the store and renders the shell", async () => {
    const body = JSON.stringify(header) + "\n" + JSON.stringify(frame) + "\n";
    // Plain response-like object — avoids depending on a global Response in jsdom.
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, status: 200, text: async () => body })));

    render(<App />);

    await waitFor(() => expect(useTraceStore.getState().header).toBeDefined());
    expect(useTraceStore.getState().frames).toHaveLength(1);
    // shell renders the brain id somewhere
    expect(await screen.findByText(/five-region/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- App`
Expected: FAIL — `App` renders `null` (placeholder) and `Shell`/`TopBar` do not exist yet.

- [ ] **Step 3: Write the implementation**

Replace `dashboard/src/App.tsx`:
```tsx
import { useEffect } from "react";
import { FileTraceSource } from "./source/FileTraceSource";
import { Shell } from "./shell/Shell";
import { useTraceStore } from "./store/traceStore";

const TRACE_URL = import.meta.env.VITE_TRACE_URL ?? "/week11_dashboard_trace.jsonl";

export function App() {
  const header = useTraceStore((s) => s.header);
  const load = useTraceStore((s) => s.load);

  useEffect(() => {
    const source = new FileTraceSource(TRACE_URL);
    let cancelled = false;
    source
      .open()
      .then((hdr) => {
        const frames: Parameters<typeof load>[1] = [];
        source.subscribe((f) => frames.push(f));
        if (!cancelled) load(hdr, frames);
      })
      .catch((err) => console.error("trace load failed:", err));
    return () => {
      cancelled = true;
      source.close();
    };
  }, [load]);

  if (!header) {
    return <div style={{ color: "#9aa3b6", font: "13px monospace", padding: 24 }}>Loading trace…</div>;
  }
  return <Shell />;
}
```

This imports `Shell`, which is created in Task 7. To keep this task runnable in isolation, create minimal `dashboard/src/shell/Shell.tsx` and `dashboard/src/shell/TopBar.tsx` now; Task 7 fleshes them out. Minimal `Shell.tsx`:
```tsx
import { TopBar } from "./TopBar";

export function Shell() {
  return (
    <div className="shell">
      <TopBar />
    </div>
  );
}
```
Minimal `TopBar.tsx`:
```tsx
import { useTraceStore } from "../store/traceStore";

export function TopBar() {
  const header = useTraceStore((s) => s.header);
  if (!header) return null;
  return <header className="topbar">{header.brain.id}</header>;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- App`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/App.tsx dashboard/src/App.test.tsx dashboard/src/shell/Shell.tsx dashboard/src/shell/TopBar.tsx
git commit -m "feat(dashboard): wire FileTraceSource into the store on mount"
```

---

## Task 7: Shell — data-driven TopBar + Scrubber

**Files:**
- Modify: `dashboard/src/shell/Shell.tsx`, `dashboard/src/shell/TopBar.tsx`
- Create: `dashboard/src/shell/Scrubber.tsx`, `dashboard/src/shell/TopBar.test.tsx`

- [ ] **Step 1: Write the failing test**

`dashboard/src/shell/TopBar.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { TopBar } from "./TopBar";

const header = {
  schema_version: "1.0",
  brain: { id: "five-region", config_hash: "ab12cd34", seed: 7, T: 32 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [],
  pathways: [],
} as TraceHeader;

describe("TopBar", () => {
  it("renders the run topology from the header", () => {
    useTraceStore.getState().load(header, []);
    render(<TopBar />);
    expect(screen.getByText(/five-region/i)).toBeInTheDocument();
    expect(screen.getByText(/T\s*32/i)).toBeInTheDocument();
    expect(screen.getByText(/seed\s*7/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- TopBar`
Expected: FAIL — the minimal TopBar shows only `header.brain.id`, not T/seed.

- [ ] **Step 3: Write the implementation**

Replace `dashboard/src/shell/TopBar.tsx`:
```tsx
import { useTraceStore } from "../store/traceStore";

export function TopBar() {
  const header = useTraceStore((s) => s.header);
  if (!header) return null;
  const { brain } = header;
  return (
    <header
      className="topbar"
      style={{ display: "flex", gap: 16, alignItems: "center", height: 56, padding: "0 16px", font: "12px monospace", color: "#e9edf6", background: "#080a12", borderBottom: "1px solid rgba(255,255,255,.075)" }}
    >
      <strong style={{ font: "700 14px sans-serif" }}>NEURO·SCOPE</strong>
      <span>{brain.id}</span>
      <span>· {brain.config_hash}</span>
      <span>· seed {brain.seed}</span>
      <span>· T {brain.T}</span>
    </header>
  );
}
```

Replace `dashboard/src/shell/Shell.tsx`:
```tsx
import { usePlayback } from "../playback/usePlayback";
import { Hero } from "../hero/Hero";
import { RegionActivity } from "../panels/RegionActivity";
import { TaskState } from "../panels/TaskState";
import { Scrubber } from "./Scrubber";
import { TopBar } from "./TopBar";

export function Shell() {
  usePlayback();
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "#05060a" }}>
      <TopBar />
      <div style={{ display: "grid", gridTemplateColumns: "316px minmax(0,1fr) 336px", flex: 1, minHeight: 0 }}>
        <aside style={{ overflow: "auto", borderRight: "1px solid rgba(255,255,255,.06)" }}>
          <RegionActivity />
        </aside>
        <main style={{ position: "relative", minWidth: 0 }}>
          <Hero />
          <Scrubber />
        </main>
        <aside style={{ overflow: "auto", borderLeft: "1px solid rgba(255,255,255,.06)" }}>
          <TaskState />
        </aside>
      </div>
    </div>
  );
}
```
(`Hero`, `RegionActivity`, `TaskState` are created in Tasks 8–9. Create empty stubs now so this compiles: a file each exporting `export function Hero(){return null}` etc. Tasks 8–9 replace them.)

`dashboard/src/shell/Scrubber.tsx`:
```tsx
import { useTraceStore } from "../store/traceStore";

export function Scrubber() {
  const winTi = useTraceStore((s) => s.winTi);
  const T = useTraceStore((s) => s.T);
  const playing = useTraceStore((s) => s.playing);
  const play = useTraceStore((s) => s.play);
  const pause = useTraceStore((s) => s.pause);

  return (
    <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 62, display: "flex", alignItems: "center", gap: 12, padding: "0 16px", font: "11px monospace", color: "#9aa3b6", background: "rgba(8,10,16,.86)" }}>
      <button onClick={playing ? pause : play} style={{ font: "11px monospace" }}>
        {playing ? "❚❚" : "▶"}
      </button>
      <span>
        t {winTi}/{T}
      </span>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- TopBar`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/shell/
git commit -m "feat(dashboard): data-driven TopBar + Scrubber + Shell layout"
```

---

## Task 8: Panel 01 Region Activity + Panel 03 Task State

**Files:**
- Create: `dashboard/src/panels/RegionActivity.tsx`, `dashboard/src/panels/TaskState.tsx`, `dashboard/src/panels/RegionActivity.test.tsx`, `dashboard/src/panels/TaskState.test.tsx` (replacing any stubs from Task 7)

- [ ] **Step 1: Write the failing tests**

`dashboard/src/panels/RegionActivity.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { RegionActivity } from "./RegionActivity";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "gridworld", grid_n: 5, action_labels: [] },
  regions: [
    { id: "sensory", label: "Sensory Cortex", n_neurons: 64, role: "input", render: "dots" },
    { id: "motor", label: "Motor Cortex", n_neurons: 4, role: "output", render: "dots" },
  ],
  pathways: [],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: {} as Frame["task"],
  regions: { sensory: { rate: 0.21, spikes: 4, active_frac: 0.3, rate_t: [] }, motor: { rate: 0.62, spikes: 9, active_frac: 0.25, rate_t: [] } },
  pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
} as unknown as Frame;

describe("RegionActivity", () => {
  it("renders one row per header region with its live rate", () => {
    useTraceStore.getState().load(header, [frame]);
    render(<RegionActivity />);
    expect(screen.getByText("Sensory Cortex")).toBeInTheDocument();
    expect(screen.getByText("Motor Cortex")).toBeInTheDocument();
    expect(screen.getByText("0.21")).toBeInTheDocument();
    expect(screen.getByText("0.62")).toBeInTheDocument();
  });
});
```

`dashboard/src/panels/TaskState.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { TaskState } from "./TaskState";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [],
  pathways: [],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: { agent: [2, 3], goal: [4, 4], action: 1, action_label: "right", reward: -1, return: -5, terminated: false, truncated: false },
  regions: {}, pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
} as unknown as Frame;

describe("TaskState", () => {
  it("renders a grid_n x grid_n grid and the action/coords readout", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<TaskState />);
    // 25 cells for grid_n=5
    expect(container.querySelectorAll("[data-cell]")).toHaveLength(25);
    expect(screen.getByText(/right/i)).toBeInTheDocument();
    expect(screen.getByText(/2,\s*3/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- panels`
Expected: FAIL — panels are empty stubs / missing.

- [ ] **Step 3: Write the implementations**

`dashboard/src/panels/RegionActivity.tsx`:
```tsx
import { useTraceStore } from "../store/traceStore";

export function RegionActivity() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;

  return (
    <section style={{ padding: 13, font: "12px sans-serif", color: "#e9edf6" }}>
      <h3 style={{ font: "600 8.5px monospace", letterSpacing: ".12em", color: "#5b6378", textTransform: "uppercase" }}>
        Panel 01 · Region Activity
      </h3>
      {header.regions.map((r) => {
        const rate = frame?.regions[r.id]?.rate ?? 0;
        return (
          <div key={r.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 0" }}>
            <span style={{ flex: 1 }}>{r.label}</span>
            <div style={{ width: 80, height: 6, background: "rgba(255,255,255,.06)", borderRadius: 3 }}>
              <div style={{ width: `${Math.min(100, rate * 100)}%`, height: "100%", background: "#3fd2ff", borderRadius: 3 }} />
            </div>
            <span style={{ font: "11px monospace", width: 34, textAlign: "right" }}>{rate.toFixed(2)}</span>
          </div>
        );
      })}
    </section>
  );
}
```

`dashboard/src/panels/TaskState.tsx`:
```tsx
import { useTraceStore } from "../store/traceStore";

export function TaskState() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;
  const n = header.task.grid_n;
  const task = frame?.task;

  const cells = [];
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const isAgent = task && task.agent[0] === x && task.agent[1] === y;
      const isGoal = task && task.goal[0] === x && task.goal[1] === y;
      cells.push(
        <div
          key={`${x},${y}`}
          data-cell
          style={{ aspectRatio: "1", border: "1px solid rgba(255,255,255,.06)", background: isAgent ? "#3fd2ff" : "transparent", boxShadow: isGoal ? "inset 0 0 0 2px #46f0a0" : "none" }}
        />,
      );
    }
  }

  return (
    <section style={{ padding: 13, font: "12px sans-serif", color: "#e9edf6" }}>
      <h3 style={{ font: "600 8.5px monospace", letterSpacing: ".12em", color: "#5b6378", textTransform: "uppercase" }}>
        Panel 03 · Task State
      </h3>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${n}, 1fr)`, gap: 2, maxWidth: 200 }}>{cells}</div>
      {task && (
        <div style={{ font: "11px monospace", color: "#9aa3b6", marginTop: 8 }}>
          <div>agent {task.agent[0]},{task.agent[1]} · goal {task.goal[0]},{task.goal[1]}</div>
          <div>action {task.action_label} · reward {task.reward} · return {task.return}</div>
        </div>
      )}
    </section>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- panels`
Expected: PASS (RegionActivity: 1, TaskState: 1).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/panels/
git commit -m "feat(dashboard): Region Activity + Task State panels (data-driven)"
```

---

## Task 9: Minimal R3F hero

**Files:**
- Modify/Create: `dashboard/src/hero/Hero.tsx` (replace the Task 7 stub)

The hero has no jsdom unit test (canvas/WebGL); its logic lives in the Task 5 helpers, and the Playwright smoke (Task 10) verifies it mounts. Keep it minimal: instanced points, one imperative `useFrame` loop.

- [ ] **Step 1: Write the implementation**

`dashboard/src/hero/Hero.tsx`:
```tsx
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useTraceStore } from "../store/traceStore";
import { buildNeurons, isSpiking } from "./layout";

const REGION_HUE: Record<string, string> = {
  sensory: "#3fd2ff",
  hippocampus: "#ad8bff",
  prefrontal: "#ffd24a",
  router: "#ff5a8a",
  motor: "#46f0a0",
};

function NeuronCloud() {
  const header = useTraceStore((s) => s.header);
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);

  const neurons = useMemo(() => (header ? buildNeurons(header) : []), [header]);
  const baseColors = useMemo(
    () => neurons.map((n) => new THREE.Color(REGION_HUE[n.region] ?? "#8aa")),
    [neurons],
  );

  // Imperative loop: read the store directly; never triggers a React render.
  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh || neurons.length === 0) return;
    const { frames, envStep, winTi } = useTraceStore.getState();
    const frame = frames[envStep];
    neurons.forEach((nrn, i) => {
      dummy.position.set(nrn.x, nrn.y, 0);
      const lit = frame ? isSpiking(frame, nrn.region, winTi, nrn.idx) : false;
      const s = lit ? 0.05 : 0.025;
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      color.copy(baseColors[i]).multiplyScalar(lit ? 1 : 0.28);
      mesh.setColorAt(i, color);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  if (neurons.length === 0) return null;
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, neurons.length]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

export function Hero() {
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <Canvas camera={{ position: [0, 0, 3], fov: 50 }} style={{ background: "#05060a" }}>
        <NeuronCloud />
      </Canvas>
    </div>
  );
}
```

- [ ] **Step 2: Verify it type-checks and the unit suite is unaffected**

Run: `npm run build`
Expected: `tsc -b` passes (no type errors) and `vite build` completes. (If `vite build` warns about chunk size, that's fine.)

Run: `npm test`
Expected: all prior unit tests still PASS (the hero has no unit test; nothing regressed).

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/hero/Hero.tsx
git commit -m "feat(dashboard): minimal R3F neuron-field hero (imperative loop)"
```

---

## Task 10: Playwright smoke test on the real trace

**Files:**
- Create: `dashboard/playwright.config.ts`, `dashboard/e2e/smoke.spec.ts`

- [ ] **Step 1: Write the Playwright config**

`dashboard/playwright.config.ts`:
```ts
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  use: { baseURL: "http://localhost:4173" },
  webServer: {
    command: "npm run sync:trace && npm run build && npm run preview -- --port 4173",
    url: "http://localhost:4173",
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
```

- [ ] **Step 2: Write the smoke test**

`dashboard/e2e/smoke.spec.ts`:
```ts
import { expect, test } from "@playwright/test";

test("boots on the real trace and renders the data-driven shell", async ({ page }) => {
  await page.goto("/");

  // TopBar shows the brain id from the real header
  await expect(page.getByText(/five-region/i)).toBeVisible();

  // Region Activity renders rows from the header topology
  await expect(page.getByText("Sensory Cortex")).toBeVisible();
  await expect(page.getByText("Hippocampus")).toBeVisible();

  // Task State renders the 5x5 gridworld (25 cells)
  await expect(page.locator("[data-cell]")).toHaveCount(25);

  // The hero canvas mounted
  await expect(page.locator("canvas")).toBeVisible();
});
```

- [ ] **Step 3: Install browsers and run the smoke test**

```bash
npx playwright install chromium
npm run e2e
```
Expected: 1 passed. (The `webServer` block syncs the trace, builds, and serves on :4173; the test asserts the real header topology rendered.)

- [ ] **Step 4: Commit**

```bash
git add dashboard/playwright.config.ts dashboard/e2e/smoke.spec.ts
git commit -m "test(dashboard): Playwright smoke booting on the real trace"
```

---

## Task 11: Manual verification

- [ ] **Step 1: Run the dev server and confirm the replay**

```bash
cd dashboard
npm run sync:trace
npm run dev
```
Open the printed URL (default http://localhost:5173). Confirm by inspection:
- Top bar shows `NEURO·SCOPE · five-region · <hash> · seed 0 · T 32`.
- Left rail lists all five regions (Sensory Cortex … Motor Cortex) with rate bars.
- Right rail shows a 5×5 grid with the agent (filled) and goal (ring) cells and a coords/action readout.
- The hero shows five vertical clusters of points on a near-black field.
- Pressing ▶ animates spikes flashing across the hero and advances the playhead; the panels update as `envStep` advances; no console errors.

- [ ] **Step 2: Confirm the imperative/reactive split**

In React DevTools (or by adding a temporary `console.count("panel render")` in `RegionActivity`), confirm panels re-render only when `envStep` changes (on window wrap), not every animation frame — proving the hero's rAF loop is not driving React renders. Remove any temporary logging before finishing.

---

## Self-review notes (already applied)

- **Spec coverage (§9):** scaffold → Task 1; `contract.ts` → Task 2; `TraceSource`/`FileTraceSource` → Task 2; `TraceStore` → Task 3; playback driver → Task 4; data-driven Shell → Tasks 6–7; minimal R3F hero (imperative loop) → Tasks 5+9; Panels 01 + 03 → Task 8; tests (parse, store transitions, Playwright smoke) → Tasks 2/3/10; trace reachability (`VITE_TRACE_URL` + sync script) → Task 1; the imperative/reactive split → Tasks 4/9 + verified in Task 11.
- **Out of scope (per §9):** Panels 02/04/05, both full hero treatments, focus mode, themes, scrubber polish, live/WebSocket, registry — none appear as tasks. Correct.
- **Type consistency:** `TraceSource.open/subscribe/close`, `useTraceStore` action names (`load`, `setEnvStep`, `setWinTi`, `play`, `pause`, `tickWindow`, `reset`), `buildNeurons`/`isSpiking` signatures, and the `contract.ts` field names are used identically across Tasks 2–11.
- **Stub ordering:** Tasks 1/6/7 intentionally create minimal placeholders (`App`, `Shell`, `TopBar`, `Hero`, panels) so each task compiles in isolation; the later task that owns each file replaces the stub. Flagged in-place so an out-of-order reader isn't surprised.
