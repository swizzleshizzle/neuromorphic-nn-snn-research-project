# NEURO·SCOPE Phase 1a — Panels Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the NEURO·SCOPE dashboard's panel layer to design-spec parity — all five panels rendering the real trace, data-driven from the header, on a CSS theme-token foundation.

**Architecture:** A single `tokens.css` (Observatory theme) defines CSS custom properties; a `regionHue` helper maps region/pathway ids → hue tokens; a shared `Panel` chrome primitive wraps every panel; each panel is a small React component reading `frames[envStep]` + `header` from the Zustand store reactively (the Phase-0 pattern). Existing chrome (TopBar/Scrubber/Shell) and panels (01/03) migrate onto tokens.

**Tech Stack:** Vite 5, React 18, TypeScript 5, Zustand 4, Vitest 2 + @testing-library/react + jsdom, Playwright 1. SVG for the comm-flow graph and spike rasters (no canvas in 1a).

**Reference spec:** `docs/superpowers/specs/2026-06-24-neuroscope-phase1a-panels-parity-design.md`.

## Global Constraints

- All commands run from `dashboard/`. Package manager: **npm**, Node ≥ 20.
- Commit convention: **plain messages, no "Co-Authored-By" / no "Generated with" trailer** (matches repo style).
- Verify every task with BOTH `npm test` (vitest) AND `npx tsc -b` — vitest uses esbuild and does not typecheck. `npm run build` must stay green.
- **Data-driven always:** iterate `header.regions[]` / `header.pathways[]` / `header.task.action_labels[]`; never hardcode the five region ids in render logic (the `regionHue` palette lookup with a neutral fallback is the one allowed id→color map).
- **Reconciliation rules:** PFC→Motor pathway id is `pfc_motor`; `gate_open` is a fraction in [0,1], threshold `> 0.5` → OPEN; v1 is flash-only (Panel 05 reads `field[r].spikes`, no membrane); the `pfc_motor` `gate_open` is a per-action array — aggregate with `max(...)` for a single tag.
- Tests render components against a loaded store and mock nothing but the store; follow the existing `panels/*.test.tsx` style.
- Single theme only (Observatory). The theme **toggle** + Clinical set are slice 1c — out of scope here.

---

## Task 1: Theme tokens + `regionHue` helper

**Files:**
- Create: `dashboard/src/theme/tokens.css`
- Create: `dashboard/src/theme/regionHue.ts`
- Create: `dashboard/src/theme/regionHue.test.ts`
- Modify: `dashboard/src/main.tsx` (import the stylesheet)

**Interfaces:**
- Produces: `regionHue(regionId: string): string` — returns a CSS color token string (e.g. `"var(--c-sensory)"`), neutral `"var(--text-dim)"` for unknown ids. Consumed by Tasks 4, 6.
- Produces: CSS custom properties on `:root` (Observatory). Consumed by every later task's styles.

- [ ] **Step 1: Write the failing test**

`dashboard/src/theme/regionHue.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { regionHue } from "./regionHue";

describe("regionHue", () => {
  it("maps known region ids to their CSS hue token", () => {
    expect(regionHue("sensory")).toBe("var(--c-sensory)");
    expect(regionHue("hippocampus")).toBe("var(--c-hippocampus)");
    expect(regionHue("motor")).toBe("var(--c-motor)");
  });

  it("falls back to a neutral token for unknown ids", () => {
    expect(regionHue("mystery")).toBe("var(--text-dim)");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- regionHue`
Expected: FAIL — cannot find module `./regionHue`.

- [ ] **Step 3: Write the implementation**

`dashboard/src/theme/regionHue.ts`:
```ts
const HUE_VAR: Record<string, string> = {
  sensory: "var(--c-sensory)",
  hippocampus: "var(--c-hippocampus)",
  prefrontal: "var(--c-prefrontal)",
  router: "var(--c-router)",
  motor: "var(--c-motor)",
};

/** CSS color token for a region/pathway-source id; neutral for unknown ids. */
export function regionHue(regionId: string): string {
  return HUE_VAR[regionId] ?? "var(--text-dim)";
}
```

`dashboard/src/theme/tokens.css`:
```css
/* NEURO·SCOPE — Observatory theme (Phase 1a). Clinical set + toggle land in slice 1c. */
:root {
  --bg: #05060a;
  --bg2: #080a12;
  --bar: rgba(8, 10, 16, 0.86);
  --panel: rgba(13, 16, 24, 0.66);
  --panel2: rgba(22, 26, 37, 0.55);
  --edge: rgba(255, 255, 255, 0.075);
  --edge2: rgba(255, 255, 255, 0.045);
  --text: #e9edf6;
  --text-dim: #9aa3b6;
  --text-faint: #5b6378;
  --blur: blur(9px);

  --c-sensory: #3fd2ff;
  --c-hippocampus: #ad8bff;
  --c-prefrontal: #ffd24a;
  --c-router: #ff5a8a;
  --c-motor: #46f0a0;

  --gate-open: #46f0a0;
  --gate-closed: #ff5a8a;
  --reward-pos: #46f0a0;
  --reward-neg: #ffb37a;
  --return-neg: #ff8aa6;
}
```

Modify `dashboard/src/main.tsx` — add the stylesheet import at the top (after the React imports):
```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./theme/tokens.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

- [ ] **Step 4: Run test + typecheck to verify they pass**

Run: `npm test -- regionHue`
Expected: PASS (2 tests).
Run: `npx tsc -b`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/theme/ dashboard/src/main.tsx
git commit -m "feat(dashboard): Observatory theme tokens + regionHue helper"
```

---

## Task 2: Shared `Panel` chrome primitive

**Files:**
- Create: `dashboard/src/panels/Panel.tsx`
- Create: `dashboard/src/panels/Panel.test.tsx`

**Interfaces:**
- Produces: `Panel({ kicker, title, accent?, children })` — card chrome with a header (kicker + title + optional accent dot). Consumed by Tasks 4–8.

- [ ] **Step 1: Write the failing test**

`dashboard/src/panels/Panel.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Panel } from "./Panel";

describe("Panel", () => {
  it("renders kicker, title, and children", () => {
    render(
      <Panel kicker="PANEL 0X · TEST" title="My Panel">
        <div>body content</div>
      </Panel>,
    );
    expect(screen.getByText("PANEL 0X · TEST")).toBeInTheDocument();
    expect(screen.getByText("My Panel")).toBeInTheDocument();
    expect(screen.getByText("body content")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- Panel`
Expected: FAIL — cannot find module `./Panel`.

- [ ] **Step 3: Write the implementation**

`dashboard/src/panels/Panel.tsx`:
```tsx
import type { ReactNode } from "react";

interface PanelProps {
  kicker: string;
  title: string;
  accent?: string; // CSS color for the accent dot
  children: ReactNode;
}

export function Panel({ kicker, title, accent, children }: PanelProps) {
  return (
    <section
      style={{
        background: "var(--panel)",
        border: "1px solid var(--edge)",
        borderRadius: 11,
        backdropFilter: "var(--blur)",
        margin: 13,
        color: "var(--text)",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "12px 13px 10px",
          borderBottom: "1px solid var(--edge2)",
        }}
      >
        <div style={{ flex: 1 }}>
          <div
            style={{
              font: "600 8px/1 monospace",
              letterSpacing: ".12em",
              color: "var(--text-faint)",
              textTransform: "uppercase",
            }}
          >
            {kicker}
          </div>
          <div style={{ font: "600 13px sans-serif", marginTop: 4 }}>{title}</div>
        </div>
        {accent && (
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: accent,
              boxShadow: `0 0 7px ${accent}`,
              flex: "none",
            }}
          />
        )}
      </header>
      <div style={{ padding: "11px 13px 13px" }}>{children}</div>
    </section>
  );
}
```

- [ ] **Step 4: Run test + typecheck**

Run: `npm test -- Panel`
Expected: PASS (1 test).
Run: `npx tsc -b`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/panels/Panel.tsx dashboard/src/panels/Panel.test.tsx
git commit -m "feat(dashboard): shared Panel chrome primitive"
```

---

## Task 3: Migrate chrome (TopBar / Scrubber / Shell) onto tokens

**Files:**
- Modify: `dashboard/src/shell/TopBar.tsx`
- Modify: `dashboard/src/shell/Scrubber.tsx`
- Modify: `dashboard/src/shell/Shell.tsx`

**Interfaces:**
- Consumes: tokens from Task 1. No public API change; behavior identical, only inline hex → `var(--…)`.

This is a pure styling refactor — existing tests (`TopBar.test.tsx`, `App.test.tsx`) cover it; no new test.

- [ ] **Step 1: Replace inline hex with tokens in `TopBar.tsx`**

`dashboard/src/shell/TopBar.tsx`:
```tsx
import { useTraceStore } from "../store/traceStore";

export function TopBar() {
  const header = useTraceStore((s) => s.header);
  if (!header) return null;
  const { brain } = header;
  return (
    <header
      className="topbar"
      style={{
        display: "flex",
        gap: 16,
        alignItems: "center",
        height: 56,
        padding: "0 16px",
        font: "12px monospace",
        color: "var(--text)",
        background: "var(--bg2)",
        borderBottom: "1px solid var(--edge)",
      }}
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

- [ ] **Step 2: Replace inline hex with tokens in `Scrubber.tsx`**

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
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: 0,
        height: 62,
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "0 16px",
        font: "11px monospace",
        color: "var(--text-dim)",
        background: "var(--bar)",
        backdropFilter: "var(--blur)",
      }}
    >
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

- [ ] **Step 3: Replace inline hex with tokens in `Shell.tsx`** (panel mounts unchanged for now — Task 9 adds the new panels)

`dashboard/src/shell/Shell.tsx`:
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
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--bg)" }}>
      <TopBar />
      <div style={{ display: "grid", gridTemplateColumns: "316px minmax(0,1fr) 336px", flex: 1, minHeight: 0 }}>
        <aside style={{ overflow: "auto", borderRight: "1px solid var(--edge)" }}>
          <RegionActivity />
        </aside>
        <main style={{ position: "relative", minWidth: 0 }}>
          <Hero />
          <Scrubber />
        </main>
        <aside style={{ overflow: "auto", borderLeft: "1px solid var(--edge)" }}>
          <TaskState />
        </aside>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Verify existing tests + typecheck stay green**

Run: `npm test`
Expected: all tests PASS (no behavior change).
Run: `npx tsc -b`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/shell/
git commit -m "refactor(dashboard): migrate chrome to theme tokens"
```

---

## Task 4: Panel 01 — Region Activity (upgrade)

**Files:**
- Modify: `dashboard/src/panels/RegionActivity.tsx`
- Modify: `dashboard/src/panels/RegionActivity.test.tsx`

**Interfaces:**
- Consumes: `Panel` (Task 2), `regionHue` (Task 1), store selectors `header`, `frames[envStep]`.
- Reads frame fields: `regions[r].{rate, rate_t, active_frac, spikes}`.

- [ ] **Step 1: Extend the test for the upgrade (write the failing assertions)**

Replace `dashboard/src/panels/RegionActivity.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { RegionActivity } from "./RegionActivity";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 3 },
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
  regions: {
    sensory: { rate: 0.21, spikes: 4, active_frac: 0.3, rate_t: [0.1, 0.2, 0.21] },
    motor: { rate: 0.62, spikes: 9, active_frac: 0.25, rate_t: [0.5, 0.6, 0.62] },
  },
  pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
} as unknown as Frame;

describe("RegionActivity", () => {
  it("renders one row per region with rate, active-frac, spikes, and a sparkline", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<RegionActivity />);
    expect(screen.getByText("Sensory Cortex")).toBeInTheDocument();
    expect(screen.getByText("Motor Cortex")).toBeInTheDocument();
    expect(screen.getByText("0.21")).toBeInTheDocument();
    expect(screen.getByText("0.62")).toBeInTheDocument();
    // active-frac + spike count readout
    expect(screen.getByText(/active 30% · 4 spikes/)).toBeInTheDocument();
    // one sparkline polyline per region
    expect(container.querySelectorAll("polyline")).toHaveLength(2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- RegionActivity`
Expected: FAIL — the current component has no active-frac text and no `<polyline>`.

- [ ] **Step 3: Write the implementation**

`dashboard/src/panels/RegionActivity.tsx`:
```tsx
import { useTraceStore } from "../store/traceStore";
import { regionHue } from "../theme/regionHue";
import { Panel } from "./Panel";

/** Build SVG polyline points for a sparkline scaled to a w×h box. */
function sparkPoints(series: number[], w: number, h: number): string {
  if (series.length === 0) return "";
  const max = Math.max(...series, 1e-6);
  const n = series.length;
  return series
    .map((v, i) => {
      const x = n > 1 ? (i / (n - 1)) * w : 0;
      const y = h - (v / max) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function RegionActivity() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;

  return (
    <Panel kicker="PANEL 01 · ACTIVITY" title="Region Activity">
      {header.regions.map((r) => {
        const rs = frame?.regions[r.id];
        const rate = rs?.rate ?? 0;
        const hue = regionHue(r.id);
        return (
          <div key={r.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 0" }}>
            <span
              style={{ width: 7, height: 7, borderRadius: "50%", background: hue, boxShadow: `0 0 6px ${hue}`, flex: "none" }}
            />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between", font: "12px sans-serif" }}>
                <span>{r.label}</span>
                <span style={{ font: "11px monospace", color: "var(--text-dim)" }}>{rate.toFixed(2)}</span>
              </div>
              <svg
                width="100%"
                height="16"
                viewBox="0 0 100 16"
                preserveAspectRatio="none"
                style={{ marginTop: 3, display: "block" }}
              >
                <polyline points={sparkPoints(rs?.rate_t ?? [], 100, 16)} fill="none" stroke={hue} strokeWidth="1.2" />
              </svg>
              <div style={{ font: "9px monospace", color: "var(--text-faint)", marginTop: 2 }}>
                active {((rs?.active_frac ?? 0) * 100).toFixed(0)}% · {rs?.spikes ?? 0} spikes
              </div>
            </div>
          </div>
        );
      })}
    </Panel>
  );
}
```

- [ ] **Step 4: Run test + typecheck**

Run: `npm test -- RegionActivity`
Expected: PASS (1 test).
Run: `npx tsc -b`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/panels/RegionActivity.tsx dashboard/src/panels/RegionActivity.test.tsx
git commit -m "feat(dashboard): Panel 01 Region Activity — sparkline, active-frac, spikes"
```

---

## Task 5: Panel 04 — Thalamic Router State (new)

**Files:**
- Create: `dashboard/src/panels/ThalamicRouter.tsx`
- Create: `dashboard/src/panels/ThalamicRouter.test.tsx`

**Interfaces:**
- Consumes: `Panel` (Task 2), store selectors.
- Reads: `header.task.action_labels`, `frame.router.{utilities, gate_open}`, `frame.task.action`.

- [ ] **Step 1: Write the failing test**

`dashboard/src/panels/ThalamicRouter.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { ThalamicRouter } from "./ThalamicRouter";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [], pathways: [],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: { agent: [0, 0], goal: [4, 4], action: 2, action_label: "down", reward: -1, return: -1, terminated: false, truncated: false },
  regions: {}, pathways: {},
  router: { gate_open: [0.0, 0.0, 0.7, 0.0], gate_open_t: [], utilities: [0.1, 0.2, 0.9, 0.3] },
  field: {},
} as unknown as Frame;

describe("ThalamicRouter", () => {
  it("renders one row per action with utility, gate pill, and selected highlight", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<ThalamicRouter />);
    expect(container.querySelectorAll("[data-action-row]")).toHaveLength(4);
    // gate pills: action 2 open (0.7 > 0.5), the other three closed
    expect(screen.getByText("OPEN")).toBeInTheDocument();
    expect(screen.getAllByText("CLOSED")).toHaveLength(3);
    // utility value for the selected action
    expect(screen.getByText("0.90")).toBeInTheDocument();
    // selected-action footer
    expect(screen.getByText(/selected action ▸ down/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- ThalamicRouter`
Expected: FAIL — cannot find module `./ThalamicRouter`.

- [ ] **Step 3: Write the implementation**

`dashboard/src/panels/ThalamicRouter.tsx`:
```tsx
import { useTraceStore } from "../store/traceStore";
import { Panel } from "./Panel";

export function ThalamicRouter() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;

  const labels = header.task.action_labels;
  const utilities = frame?.router?.utilities ?? [];
  const gates = frame?.router?.gate_open ?? [];
  const maxUtil = Math.max(...utilities, 1e-6);
  const selected = frame?.task?.action;

  return (
    <Panel kicker="PANEL 04 · GATING" title="Thalamic Router" accent="var(--c-router)">
      <div
        style={{ display: "flex", gap: 9, font: "500 8px/1 monospace", color: "var(--text-faint)", letterSpacing: ".08em", padding: "0 8px 6px" }}
      >
        <span style={{ width: 40 }}>ACTION</span>
        <span style={{ flex: 1 }}>UTILITY</span>
        <span style={{ width: 48, textAlign: "center" }}>GATE</span>
      </div>
      {labels.map((label, a) => {
        const util = utilities[a] ?? 0;
        const open = (gates[a] ?? 0) > 0.5;
        const isSel = selected === a;
        return (
          <div
            key={label}
            data-action-row
            style={{
              display: "flex",
              alignItems: "center",
              gap: 9,
              padding: "6px 8px",
              borderRadius: 7,
              background: isSel ? "var(--panel2)" : "transparent",
              border: `1px solid ${isSel ? "var(--edge)" : "transparent"}`,
            }}
          >
            <span style={{ width: 40, font: "11px monospace", textTransform: "uppercase", color: isSel ? "var(--c-motor)" : "var(--text)" }}>
              {label}
            </span>
            <div style={{ flex: 1, height: 6, background: "var(--edge)", borderRadius: 3 }}>
              <div style={{ width: `${(util / maxUtil) * 100}%`, height: "100%", background: "var(--c-prefrontal)", borderRadius: 3 }} />
            </div>
            <span style={{ width: 26, textAlign: "right", font: "9px monospace", color: "var(--text-dim)" }}>{util.toFixed(2)}</span>
            <span
              data-gate
              style={{
                width: 48,
                textAlign: "center",
                font: "600 7.5px/1 monospace",
                letterSpacing: ".08em",
                padding: "3px 6px",
                borderRadius: 4,
                color: "#0a0a0c",
                background: open ? "var(--gate-open)" : "var(--gate-closed)",
              }}
            >
              {open ? "OPEN" : "CLOSED"}
            </span>
          </div>
        );
      })}
      <div style={{ font: "11px monospace", color: "var(--c-motor)", marginTop: 6 }}>
        selected action ▸ {selected != null ? labels[selected] : "—"}
      </div>
    </Panel>
  );
}
```

- [ ] **Step 4: Run test + typecheck**

Run: `npm test -- ThalamicRouter`
Expected: PASS (1 test).
Run: `npx tsc -b`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/panels/ThalamicRouter.tsx dashboard/src/panels/ThalamicRouter.test.tsx
git commit -m "feat(dashboard): Panel 04 Thalamic Router State"
```

---

## Task 6: Panel 02 — Inter-Region Communication Flow (new)

**Files:**
- Create: `dashboard/src/panels/CommunicationFlow.tsx`
- Create: `dashboard/src/panels/CommunicationFlow.test.tsx`

**Interfaces:**
- Consumes: `Panel` (Task 2), `regionHue` (Task 1), store selectors, contract types `Pathway`/`PathwayState`.
- Reads: `header.regions[]`, `header.pathways[]`, `frame.pathways[p].{intensity, gate_open}`.

- [ ] **Step 1: Write the failing test**

`dashboard/src/panels/CommunicationFlow.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { CommunicationFlow } from "./CommunicationFlow";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
  task: { type: "gridworld", grid_n: 5, action_labels: [] },
  regions: [
    { id: "sensory", label: "Sensory", n_neurons: 64, role: "input", render: "dots" },
    { id: "hippocampus", label: "Hippocampus", n_neurons: 150, role: "memory", render: "dots" },
    { id: "prefrontal", label: "Prefrontal", n_neurons: 4, role: "control", render: "dots" },
    { id: "motor", label: "Motor", n_neurons: 4, role: "output", render: "dots" },
  ],
  pathways: [
    { id: "sens_hippo", src: "sensory", dst: "hippocampus", gated: true, label: "store/recall" },
    { id: "sens_pfc", src: "sensory", dst: "prefrontal", gated: false, label: "perceive" },
    { id: "hippo_pfc", src: "hippocampus", dst: "prefrontal", gated: true, label: "recall" },
    { id: "pfc_motor", src: "prefrontal", dst: "motor", gated: true, label: "act" },
  ],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: {} as Frame["task"],
  regions: {},
  pathways: {
    sens_hippo: { intensity: 0.4, gate_open: 0 },
    sens_pfc: { intensity: 0.3 },
    hippo_pfc: { intensity: 0.2, gate_open: 0 },
    pfc_motor: { intensity: 0.1, gate_open: [0, 0, 0.7, 0] },
  },
  router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
} as unknown as Frame;

describe("CommunicationFlow", () => {
  it("renders a node per region, an edge per pathway, and gate tags", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<CommunicationFlow />);
    expect(container.querySelectorAll("[data-node]")).toHaveLength(4);
    expect(container.querySelectorAll("[data-edge]")).toHaveLength(4);
    // sens_hippo quiescent → STORE; sens_pfc ungated → OPEN; pfc_motor max 0.7 → OPEN; hippo_pfc 0 → CLOSED
    expect(screen.getByText("STORE")).toBeInTheDocument();
    expect(screen.getAllByText("OPEN")).toHaveLength(2);
    expect(screen.getByText("CLOSED")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- CommunicationFlow`
Expected: FAIL — cannot find module `./CommunicationFlow`.

- [ ] **Step 3: Write the implementation**

`dashboard/src/panels/CommunicationFlow.tsx`:
```tsx
import type { Pathway, PathwayState } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { regionHue } from "../theme/regionHue";
import { Panel } from "./Panel";

const W = 300;
const H = 120;

/** Aggregate a gate_open scalar/array into a single fraction. */
function aggGate(go: number | number[] | undefined): number {
  if (Array.isArray(go)) return go.length ? Math.max(...go) : 0;
  return typeof go === "number" ? go : 0;
}

function gateTag(p: Pathway, ps: PathwayState | undefined): { text: string; bg: string; fg: string } {
  const agg = aggGate(ps?.gate_open);
  if (p.id === "sens_hippo" && agg === 0) return { text: "STORE", bg: "var(--edge)", fg: "var(--text-faint)" };
  if (!p.gated) return { text: "OPEN", bg: "var(--gate-open)", fg: "#0a0a0c" };
  return agg > 0.5
    ? { text: "OPEN", bg: "var(--gate-open)", fg: "#0a0a0c" }
    : { text: "CLOSED", bg: "var(--gate-closed)", fg: "#0a0a0c" };
}

export function CommunicationFlow() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;

  const n = header.regions.length;
  const pos: Record<string, { x: number; y: number }> = {};
  header.regions.forEach((r, i) => {
    pos[r.id] = {
      x: n > 1 ? 24 + (i / (n - 1)) * (W - 48) : W / 2,
      y: H / 2 + (i % 2 === 0 ? -16 : 16),
    };
  });

  return (
    <Panel kicker="PANEL 02 · PATHWAY INTENSITY" title="Communication Flow">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
        {header.pathways.map((p) => {
          const a = pos[p.src];
          const b = pos[p.dst];
          if (!a || !b) return null;
          const ps = frame?.pathways[p.id];
          const intensity = ps?.intensity ?? 0;
          const closed = p.gated && aggGate(ps?.gate_open) <= 0.5;
          const mx = (a.x + b.x) / 2;
          return (
            <path
              key={p.id}
              data-edge
              d={`M ${a.x} ${a.y} C ${mx} ${a.y}, ${mx} ${b.y}, ${b.x} ${b.y}`}
              fill="none"
              stroke={regionHue(p.src)}
              strokeWidth={1 + intensity * 4}
              strokeOpacity={0.25 + intensity * 0.7}
              strokeDasharray={closed ? "3 4" : undefined}
            />
          );
        })}
        {header.regions.map((r) => (
          <circle key={r.id} data-node cx={pos[r.id].x} cy={pos[r.id].y} r={7} fill={regionHue(r.id)} fillOpacity={0.9} />
        ))}
      </svg>
      <div style={{ display: "flex", flexDirection: "column", gap: 5, marginTop: 8 }}>
        {header.pathways.map((p) => {
          const ps = frame?.pathways[p.id];
          const tag = gateTag(p, ps);
          return (
            <div key={p.id} style={{ display: "flex", alignItems: "center", gap: 8, font: "10px monospace" }}>
              <span style={{ flex: 1, color: "var(--text-dim)" }}>{p.label ?? p.id}</span>
              <span
                data-gate-tag
                style={{ font: "600 7.5px/1 monospace", letterSpacing: ".08em", padding: "3px 6px", borderRadius: 4, color: tag.fg, background: tag.bg }}
              >
                {tag.text}
              </span>
              <span style={{ width: 34, textAlign: "right", color: regionHue(p.src) }}>{(ps?.intensity ?? 0).toFixed(2)}</span>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}
```

- [ ] **Step 4: Run test + typecheck**

Run: `npm test -- CommunicationFlow`
Expected: PASS (1 test).
Run: `npx tsc -b`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/panels/CommunicationFlow.tsx dashboard/src/panels/CommunicationFlow.test.tsx
git commit -m "feat(dashboard): Panel 02 Inter-Region Communication Flow"
```

---

## Task 7: Panel 03 — Task State (upgrade)

**Files:**
- Modify: `dashboard/src/panels/TaskState.tsx`
- Modify: `dashboard/src/panels/TaskState.test.tsx`

**Interfaces:**
- Consumes: `Panel` (Task 2), store selectors.
- Reads: `header.task.grid_n`, `frame.task.{agent, goal, action_label, reward, return}`.

- [ ] **Step 1: Extend the test for the upgrade**

Replace `dashboard/src/panels/TaskState.test.tsx`:
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
  regions: [], pathways: [],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: { agent: [2, 3], goal: [4, 4], action: 1, action_label: "right", reward: -1, return: -5, terminated: false, truncated: false },
  regions: {}, pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] }, field: {},
} as unknown as Frame;

describe("TaskState", () => {
  it("renders the grid, action arrow, coords, and sign-colored reward/return", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<TaskState />);
    expect(container.querySelectorAll("[data-cell]")).toHaveLength(25);
    expect(screen.getByText(/▶ right/i)).toBeInTheDocument();
    expect(screen.getByText(/2,\s*3/)).toBeInTheDocument();
    const reward = container.querySelector("[data-reward]") as HTMLElement;
    const ret = container.querySelector("[data-return]") as HTMLElement;
    expect(reward.textContent).toBe("-1");
    expect(reward.style.color).toBe("var(--reward-neg)");
    expect(ret.textContent).toBe("-5");
    expect(ret.style.color).toBe("var(--return-neg)");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- TaskState`
Expected: FAIL — the current component has no arrow and no `[data-reward]`/`[data-return]` colored spans.

- [ ] **Step 3: Write the implementation**

`dashboard/src/panels/TaskState.tsx`:
```tsx
import { useTraceStore } from "../store/traceStore";
import { Panel } from "./Panel";

const ARROW: Record<string, string> = { up: "▲", right: "▶", down: "▼", left: "◀" };

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
          style={{
            aspectRatio: "1",
            border: "1px solid var(--edge)",
            borderRadius: 2,
            background: isAgent ? "var(--c-sensory)" : "transparent",
            boxShadow: isGoal ? "inset 0 0 0 2px var(--c-motor)" : "none",
          }}
        />,
      );
    }
  }

  const rewardColor = (v: number) => (v >= 0 ? "var(--reward-pos)" : "var(--reward-neg)");
  const returnColor = (v: number) => (v >= 0 ? "var(--reward-pos)" : "var(--return-neg)");

  return (
    <Panel kicker="PANEL 03 · TASK" title="Task State">
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${n}, 1fr)`, gap: 3, maxWidth: 260, margin: "0 auto" }}>
        {cells}
      </div>
      {task && (
        <div style={{ font: "11px monospace", color: "var(--text-dim)", marginTop: 10, display: "flex", flexDirection: "column", gap: 3 }}>
          <div>
            agent {task.agent[0]},{task.agent[1]} · goal {task.goal[0]},{task.goal[1]}
          </div>
          <div>
            action {ARROW[task.action_label] ?? ""} {task.action_label}
          </div>
          <div>
            reward <span data-reward style={{ color: rewardColor(task.reward) }}>{task.reward}</span>
            {" · return "}
            <span data-return style={{ color: returnColor(task.return) }}>{task.return}</span>
          </div>
        </div>
      )}
    </Panel>
  );
}
```

- [ ] **Step 4: Run test + typecheck**

Run: `npm test -- TaskState`
Expected: PASS (1 test).
Run: `npx tsc -b`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/panels/TaskState.tsx dashboard/src/panels/TaskState.test.tsx
git commit -m "feat(dashboard): Panel 03 Task State — arrow + sign-colored reward/return"
```

---

## Task 8: Panel 05 — Spike Raster (new)

**Files:**
- Create: `dashboard/src/panels/SpikeRaster.tsx`
- Create: `dashboard/src/panels/SpikeRaster.test.tsx`

**Interfaces:**
- Consumes: `Panel` (Task 2), store selectors `header`, `frames[envStep]`, `winTi`.
- Reads: `header.brain.T`, `frame.field.prefrontal.spikes` (`[T][N]`). Hardcoded to `prefrontal` in 1a (region selection deferred per spec §5).

- [ ] **Step 1: Write the failing test**

`dashboard/src/panels/SpikeRaster.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { SpikeRaster } from "./SpikeRaster";

const header = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 4 },
  task: { type: "gridworld", grid_n: 5, action_labels: [] },
  regions: [], pathways: [],
} as TraceHeader;
const frame = {
  episode: 0, step: 0, t: 0,
  task: {} as Frame["task"],
  regions: {}, pathways: {}, router: { gate_open: [], gate_open_t: [], utilities: [] },
  field: { prefrontal: { spikes: [[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]] } },
} as unknown as Frame;

describe("SpikeRaster", () => {
  it("renders one strip per prefrontal neuron, spike marks, a playhead, and the T label", () => {
    useTraceStore.getState().load(header, [frame]);
    const { container } = render(<SpikeRaster />);
    expect(container.querySelectorAll("[data-raster-row]")).toHaveLength(4);
    // two spikes total in the fixture
    expect(container.querySelectorAll("rect")).toHaveLength(2);
    // a playhead line per row
    expect(container.querySelectorAll("line")).toHaveLength(4);
    expect(screen.getByText("inference window · T=4")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- SpikeRaster`
Expected: FAIL — cannot find module `./SpikeRaster`.

- [ ] **Step 3: Write the implementation**

`dashboard/src/panels/SpikeRaster.tsx`:
```tsx
import { useTraceStore } from "../store/traceStore";
import { Panel } from "./Panel";

const REGION = "prefrontal"; // hardcoded in 1a; region selection deferred (spec §5)
const STRIP_W = 232;
const STRIP_H = 14;

export function SpikeRaster() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  const winTi = useTraceStore((s) => s.winTi);
  if (!header) return null;

  const T = header.brain.T;
  const spikes = frame?.field?.[REGION]?.spikes ?? []; // [T][N]
  const nNeurons = spikes[0]?.length ?? 0;
  const playheadX = T > 1 ? (winTi / (T - 1)) * STRIP_W : 0;

  const rows = [];
  for (let neuron = 0; neuron < nNeurons; neuron++) {
    const marks = [];
    for (let ti = 0; ti < spikes.length; ti++) {
      if (spikes[ti][neuron] === 1) {
        const x = T > 1 ? (ti / (T - 1)) * STRIP_W : 0;
        marks.push(<rect key={ti} x={x} y={2} width={3.2} height={10} rx={1} fill="var(--c-prefrontal)" />);
      }
    }
    rows.push(
      <div key={neuron} data-raster-row style={{ display: "flex", alignItems: "center", gap: 8, padding: "3px 0" }}>
        <span style={{ width: 34, font: "8px monospace", color: "var(--text-faint)" }}>n{neuron}</span>
        <svg
          viewBox={`0 0 ${STRIP_W} ${STRIP_H}`}
          width="100%"
          height={STRIP_H}
          style={{ background: "var(--panel2)", borderRadius: 4, display: "block" }}
        >
          {marks}
          <line x1={playheadX} x2={playheadX} y1={0} y2={STRIP_H} stroke="#fff" strokeWidth={1} strokeOpacity={0.6} />
        </svg>
      </div>,
    );
  }

  return (
    <Panel kicker="PANEL 05 · field" title="Spike Raster">
      {rows}
      <div style={{ display: "flex", justifyContent: "space-between", font: "500 8px/1 monospace", color: "var(--text-faint)", marginTop: 4 }}>
        <span>t₀</span>
        <span>inference window · T={T}</span>
        <span>t{T - 1}</span>
      </div>
    </Panel>
  );
}
```

- [ ] **Step 4: Run test + typecheck**

Run: `npm test -- SpikeRaster`
Expected: PASS (1 test).
Run: `npx tsc -b`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/panels/SpikeRaster.tsx dashboard/src/panels/SpikeRaster.test.tsx
git commit -m "feat(dashboard): Panel 05 Spike Raster (prefrontal, flash-only)"
```

---

## Task 9: Mount all five panels + extend the Playwright smoke

**Files:**
- Modify: `dashboard/src/shell/Shell.tsx`
- Modify: `dashboard/e2e/smoke.spec.ts`

**Interfaces:**
- Consumes: `RegionActivity`, `ThalamicRouter` (left rail); `CommunicationFlow`, `TaskState`, `SpikeRaster` (right rail).

- [ ] **Step 1: Mount the new panels in the Shell rails**

`dashboard/src/shell/Shell.tsx`:
```tsx
import { usePlayback } from "../playback/usePlayback";
import { Hero } from "../hero/Hero";
import { CommunicationFlow } from "../panels/CommunicationFlow";
import { RegionActivity } from "../panels/RegionActivity";
import { SpikeRaster } from "../panels/SpikeRaster";
import { TaskState } from "../panels/TaskState";
import { ThalamicRouter } from "../panels/ThalamicRouter";
import { Scrubber } from "./Scrubber";
import { TopBar } from "./TopBar";

export function Shell() {
  usePlayback();
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--bg)" }}>
      <TopBar />
      <div style={{ display: "grid", gridTemplateColumns: "316px minmax(0,1fr) 336px", flex: 1, minHeight: 0 }}>
        <aside style={{ overflow: "auto", borderRight: "1px solid var(--edge)" }}>
          <RegionActivity />
          <ThalamicRouter />
        </aside>
        <main style={{ position: "relative", minWidth: 0 }}>
          <Hero />
          <Scrubber />
        </main>
        <aside style={{ overflow: "auto", borderLeft: "1px solid var(--edge)" }}>
          <CommunicationFlow />
          <TaskState />
          <SpikeRaster />
        </aside>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Extend the Playwright smoke to assert the new panels render on the real trace**

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

  // Phase 1a panels render from the real trace
  await expect(page.getByText("Thalamic Router")).toBeVisible();
  await expect(page.getByText("Communication Flow")).toBeVisible();
  await expect(page.getByText("Spike Raster")).toBeVisible();
  // at least one router gate pill is present
  await expect(page.locator("[data-gate]").first()).toBeVisible();
});
```

- [ ] **Step 3: Run the full unit suite + typecheck + build**

Run: `npm test`
Expected: all tests PASS (Panel, regionHue, all five panels, store, parse, App, TopBar, playback).
Run: `npx tsc -b`
Expected: no errors.
Run: `npm run build`
Expected: `tsc -b` + `vite build` complete (chunk-size warning from three.js is fine).

- [ ] **Step 4: Run the Playwright smoke**

Run: `npm run e2e`
Expected: 1 passed. (The `webServer` block syncs the trace, builds, serves on :4173; the test asserts the full panel set rendered on the real trace.)

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/shell/Shell.tsx dashboard/e2e/smoke.spec.ts
git commit -m "feat(dashboard): mount all five panels + extend smoke to panels 02/04/05"
```

---

## Self-Review

**Spec coverage:**
- Theme token foundation (Observatory) → Task 1; chrome migration → Task 3. ✓
- Shared `Panel` primitive → Task 2. ✓
- Panel 01 upgrade (sparkline/active-frac/spikes/status dot) → Task 4. ✓
- Panel 04 Router (utility bars, gate pills, selected highlight) → Task 5. ✓
- Panel 02 Comm Flow (SVG graph + gate tags OPEN/CLOSED/STORE, dashed-closed edges) → Task 6. ✓
- Panel 03 upgrade (markers, arrow, sign-colored reward/return) → Task 7. ✓
- Panel 05 Spike Raster (prefrontal, playhead, T label, flash-only) → Task 8. ✓
- Placement (01+04 left, 02+03+05 right) + e2e → Task 9. ✓
- Deferred (Panel 05 selection, membrane) → documented in spec §5, not built. ✓
- Out of scope (hero, focus mode, theme toggle, scrubber polish, encoding key) → not in any task. ✓

**Placeholder scan:** none — every step has complete code and exact commands.

**Type consistency:** `regionHue(id): string`, `Panel({kicker,title,accent?,children})`, store selectors (`header`, `frames[s.envStep]`, `winTi`), and contract field names (`regions[r].{rate,rate_t,active_frac,spikes}`, `router.{utilities,gate_open}`, `pathways[p].{intensity,gate_open}`, `field[r].spikes`, `task.{agent,goal,action,action_label,reward,return}`) are used identically across tasks and match `contract.ts`. Gate-fraction threshold `> 0.5` and `pfc_motor` array aggregation (`max`) consistent in Tasks 5 and 6.

**Data-driven check:** all render loops iterate `header.regions`/`header.pathways`/`action_labels`; the only id→value map is `regionHue` (palette, with neutral fallback) — allowed by the spec.
