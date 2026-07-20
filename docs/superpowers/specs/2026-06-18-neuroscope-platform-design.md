# NEURO·SCOPE Platform — Long-Term Architecture & Phase Roadmap

**Date:** 2026-06-18
**Phase context:** Phase 2 (multi-region brain) tooling; the monitoring platform that outlives it
**Status:** Design locked — ready for Phase 0 implementation plan
**Builds on:** the Stage-2 data contract (`docs/superpowers/specs/2026-06-17-stage2-dashboard-design.md`) and the design handoff (`docs/handoffs/claude_design/`)

---

## 1. Purpose & scope

NEURO·SCOPE is the long-term **brain-monitoring platform** for this project — the production codebase that replaces the throwaway `.dc.html` prototype. It must:

- Give complete, presentable debug visibility into a running brain (no lapse in verbosity).
- Be **extensible to a set of brains** across many iterations and tasks — the UI renders whatever a trace *declares* (header topology), never hardcoding today's five regions.
- Grow from **local replay** → **live streaming** → **multi-brain management** → **scale/hosting** without rewrites, via stable seams.

This document is the **whole-platform architecture and the phase roadmap**. Phase 0 (§9) is the first concrete slice and gets its own implementation plan; later phases get their own spec → plan cycles as they come up.

## 2. Non-goals (now)

- Auth, multi-tenant hosting, cloud deploy — deferred to Phase 4 (the architecture stays hosting-ready; see §7).
- Redis — deferred to Phase 4; until multi-sim fan-out exists, the trace file is the queue (§4, Layer 2).
- Membrane voltage (`detail.membrane`) — still deferred per the Stage-2 reconciliation (flash-only); needs a region-level record hook first.
- Compare-multiple-runs analytics — a Phase 3 sub-phase, not Phase 0–2.

## 3. Monorepo layout

One repository — the whole brain + monitoring story lives together, and a contract change commits atomically with its consumer.

```
neuromorphic-nn-snn-research-project/
  src/neuromorphic/
    monitor/            # Layer 1 producer (BUILT): schema, frame, sink, runner
    server/             # Layer 2 live server (Phase 2): FastAPI file-tail → WS
  dashboard/            # Layer 3 frontend (Phase 0+): Vite + React + TS
    src/
      contract.ts       # TS types mirroring the Python Frame/Header schema
      source/           # TraceSource interface + File/WebSocket implementations
      store/            # Zustand TraceStore
      shell/            # top bar, stage, rails, scrubber
      hero/             # R3F WebGL hero (imperative rAF loop)
      panels/           # Panel 01–05 (D3 / Recharts), reactive
    index.html  vite.config.ts  package.json  ...
  docs/  experiments/  tests/        # existing python project
  outputs/week11_dashboard_trace.jsonl   # the real replay artifact (gitignored)
```

Python and JS toolchains coexist: `pytest` over `src/neuromorphic` + `tests/`; `vite`/`vitest` over `dashboard/`. They share docs and the data contract, nothing else.

## 4. Layered architecture

Three layers joined by **one contract** (the JSONL header + frames) and **two symmetrical seams** (`TraceSink` on the producer side, `TraceSource` on the consumer side).

### Layer 1 — Producer (BUILT)

`neuromorphic.monitor`: a `Brain` episode → `Frame` objects → a `TraceSink`. `FileSink` writes JSONL today; `WebSocketSink`/`RedisStreamSink` are later implementations of the same interface. Unchanged by this design except new sink implementations in later phases.

### Layer 2 — Transport / Server

Two paths carry the **identical Frame object**:

- **Batch / replay:** the frontend reads the `.jsonl` directly (a static fetch). No server needed.
- **Live (Phase 2):** a small FastAPI server **tails the active trace file** the `FileSink` is already writing and pushes each new line over a WebSocket. **The trace file is the queue** — replay and live read the *same artifact*, the sim stays decoupled from any viewer, and no message broker is required. Redis Streams enter only at Phase 4 when multiple concurrent sims fan out to multiple viewers.

### Layer 3 — Frontend (`dashboard/`)

A `TraceSource` yields `{ header, onFrame }` regardless of origin; a Zustand `TraceStore` holds replay/playback state; a data-driven Shell renders a WebGL Hero plus reactive Panels, all generated from `header.regions[]` / `header.pathways[]`.

## 5. The TraceSource / TraceSink symmetry (the load-bearing seam)

The producer abstracts its *destination* (`TraceSink`); the frontend abstracts its *origin* (`TraceSource`). They are mirror images of the same contract.

```ts
interface TraceSource {
  open(): Promise<TraceHeader>;            // resolve the run header
  subscribe(onFrame: (f: Frame) => void): void;  // stream frames (all-at-once for file, live for ws)
  close(): void;
}
```

- `FileTraceSource(url)` — fetch the JSONL, parse line 0 as header, emit the rest as frames (Phase 0).
- `WebSocketTraceSource(wsUrl)` — connect, receive header then a live frame stream (Phase 2).

**The render path never knows which source it has.** This is what makes live streaming a *new class, not a rewrite*, keeps endpoints config-driven (hosting-later), and means the prototype's already-decoupled loader becomes a first-class, tested boundary.

## 6. Frontend stack & conventions

- **Vite + React + TypeScript** — fast HMR; static build exports cleanly for the capstone and for future hosting.
- **WebGL hero: react-three-fiber (three.js).** The hero's headline is the orbiting 3D neuron cloud; three.js makes 3D orbit, depth fog, GPU-instanced points, and future click-to-inspect (raycaster) declarative inside React. The Flow Map treatment is an orthographic scene or a D3/SVG overlay. (Rejected: pixi.js — strong in 2D but hand-rolls 3D projection; raw regl — most code, least leverage.)
- **State: Zustand.** One `TraceStore` (header, `frames[]`, `envStep`, `winTi`, playback flags + actions). Selectors scope panel re-renders; rejected Redux (boilerplate) and Context (re-render storms at playback framerate).
- **Imperative hero / reactive panels** — the defining performance rule. The hero runs its **own `requestAnimationFrame` loop** reading the store imperatively (`store.getState()`), so the 60fps T-window animation never triggers a React render. React re-renders **panels** only when `envStep` changes (throttled). One store, two consumption styles, no fighting.
- **Panels: D3** for bespoke visuals (flow schematic, gridworld, router gates), **Recharts** for standard time-series (rate sparklines; membrane later). Recharts is swappable if it proves heavy.
- **Contract types: a hand-written `contract.ts`** mirroring the Python schema (Header, Frame, regions/pathways/router/field/encoding). The frontend's single source of truth for shapes; could be generated from a JSON schema later.
- **Testing: Vitest** for the data layer (TraceSource parsing, store transitions — where bugs hide) + a **Playwright** smoke test that loads the real trace and asserts the shell renders.
- **Data-driven, always:** clusters, gauges, flow nodes, edges, labels, and the hero representation (`render` hint) are generated from the header — never hardcoded to five regions. (Honors the four Stage-2 reconciliation rules: `pfc_motor` id, `gate_open` fraction, flash-only, `encoding.sensory_input` for the grid.)

## 7. Runtime model — local now, host later

Phase 0–3 run **local-only**: `python sim → trace/WS`, `vite dev` (or a static build) → browser, single user, no auth. The capstone is a screen-recording or a static-export replay.

To keep hosting a *later phase and not a rewrite*: endpoints come from **config** (Vite env vars — `VITE_TRACE_URL`, `VITE_WS_URL`), never hardcoded `localhost`; transport lives behind `TraceSource`; the build is static-export friendly. Phase 4 adds auth/CORS/deploy without touching the render path.

## 8. Phase roadmap

| Phase | Goal | Ships |
|---|---|---|
| **0 — Foundation** | A real React app that replays the real trace, every seam in place | Monorepo `dashboard/` scaffold; `contract.ts`; `TraceSource` + `FileTraceSource`; Zustand `TraceStore`; data-driven Shell; a minimal R3F hero proving the imperative-rAF↔store seam; 2 real panels (Region Activity + Task State); tests. Replaces the prototype as the foundation — not full fidelity. |
| **1 — Parity** | Match the design spec | All 5 panels; both hero treatments (3D cloud + flow map); focus mode; scrubber; Observatory/Clinical themes; encoding key. Port the prototype's visual logic into proper R3F/D3 components. |
| **2 — Live** | Watch a sim think live | FastAPI server tailing the active trace file → WS; `WebSocketTraceSource`; live indicator + reconnect. — **SHIPPED 2026-07-17** (file-tail server + WebSocketTraceSource + reconnect; `WebSocketSink` intentionally NOT built — the file-as-queue approach was chosen, so a sim-side sink stays a Phase-4 fan-out concern; see plan 2026-07-17-neuroscope-phase2-live). |
| **3 — Set of brains** | Manage runs | Trace registry (scan a traces dir + run metadata); trace picker; switch between brains/runs. Compare-runs a later sub-phase. |
| **4 — Scale + host** | Beyond one local sim | RedisStreamSink + server consumer for multi-sim fan-out; cloud/density representation ladder for large brains; optional hosting (config endpoints, auth/CORS, deploy). |

The seams carry the roadmap: `TraceSource` makes Phase 2 additive; data-driven-from-header makes Phase 3 need no UI change; the `render` hint pre-anticipates Phase 4.

## 9. Phase 0 — scope detail

**Goal:** the skeleton with real blood in it — a real Vite/React/TS app that loads `outputs/week11_dashboard_trace.jsonl` through the real `TraceSource`/`TraceStore` seams and renders a data-driven shell, a minimal WebGL hero, and two real panels. Lock the architecture against real data before scaling surface area.

**In scope:**
- `dashboard/` Vite + React + TS scaffold; Vitest + Playwright configured; npm scripts.
- `contract.ts` — Header/Frame types matching the Python schema.
- `TraceSource` interface + `FileTraceSource` (fetch + JSONL parse; header then frames).
- Zustand `TraceStore` — header, `frames[]`, `envStep`, `winTi`, `playing`; actions: `load`, `setEnvStep`, `play/pause`, window tick.
- A playback driver (the rAF loop advancing `winTi` across the T-window and `envStep` over the episode).
- Shell — top bar (run/topology readout), stage, side rails, bottom scrubber — laid out and **data-driven from the header** (region list, action labels). Panels can be placeholders except the two below.
- **Minimal R3F hero** — region clusters from `header.regions[]`, neurons rendered from `field[r].spikes` (flash-only), driven by the imperative rAF loop reading the store. Flow-map *or* basic cloud — whichever is simplest to prove the seam; full treatments are Phase 1.
- **Panel 01 Region Activity** + **Panel 03 Task State** — wired to real frames (the reactive path), honoring data-driven rendering.
- Tests: `FileTraceSource` parses the real trace into header + N frames; store transitions (load, setEnvStep, window wrap); a Playwright smoke that boots the app against the real trace and asserts the shell + a region label render.
- Trace file made reachable by the app (copy/symlink/served path — decided in the plan).

**Out of scope (Phase 1+):** Panels 02/04/05; both full hero treatments; focus mode; themes; scrubber polish; live/WebSocket; trace registry.

**Done when:** `npm run dev` shows the data-driven shell replaying the real trace, the hero animates real spikes across the T-window via the imperative loop, the two panels update with `envStep`, and `npm test` is green.

## 10. Risks & notes

- **Two toolchains in one repo.** Keep them isolated: Python tooling ignores `dashboard/` (it already would); add `dashboard/node_modules` and build output to `.gitignore`.
- **The hero imperative loop must not leak into React render.** If panels start stuttering, that's the symptom — the hero is re-rendering React. Tested by keeping hero subscriptions to `store.getState()` inside rAF only.
- **Trace file reachability.** The app fetches by config URL; the plan picks how the dev server serves `week11_dashboard_trace.jsonl` (Vite `public/` copy vs. a served path). The prototype's relative-path fetch only worked beside the HTML — this design fixes it via `VITE_TRACE_URL`.
- **Contract drift.** `contract.ts` is hand-mirrored from the Python schema; when the schema changes, both move in the same commit (the monorepo payoff). A later phase can generate the TS from a shared JSON schema.
