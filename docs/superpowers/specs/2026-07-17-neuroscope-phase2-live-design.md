# NEURO·SCOPE Phase 2 — Live Monitoring (design)

**Date:** 2026-07-17 · Week 15
**Phase context:** NEURO·SCOPE platform Phase 2 ("Live") from the roadmap in
`docs/superpowers/specs/2026-06-18-neuroscope-platform-design.md` §8
**Status:** Design approved — ready for an implementation plan
**Builds on:** the file-replay pipeline shipped in Phases 0/1 — `src/neuromorphic/monitor/` (producer +
JSONL contract) and `dashboard/` (React/R3F reader). The load-bearing seams `TraceSink` (producer) and
`TraceSource` (frontend) already exist; this phase adds a live path behind them **without touching the
render path**.

---

## 1. Goal & success criteria

Watch a **running** sim stream frame-by-frame into the existing NEURO·SCOPE UI, with reconnect and a
visible connection state. Live is a **new `TraceSource`**, not a rewrite — the platform design's promise
(§5, "the render path never knows which source it has") is the whole point.

**Done when:**
- With a sim writing an active JSONL trace, `npm run dev` (pointed at the live server) shows the shell
  replaying frames as they are written, the playhead **following the tail**, and a **LIVE** badge.
- Dropping/restarting the socket **auto-reconnects** (capped backoff); the UI shows
  `live / reconnecting / ended / error`.
- When the run finishes, the UI shows an **ended** state (not a false "reconnecting" loop).
- All new units are tested; the existing file-replay path and its tests are **unchanged and green**.

**Non-goals (explicit, deferred):** `WebSocketSink` (sim-pushes-directly), Redis / multi-sim fan-out,
multi-run registry & picker, backpressure tuning, a live Playwright e2e harness, and the unrelated
dashboard gaps (membrane detail, real measured pathway intensity, per-region raster selector).

## 2. Transport decision — file-as-queue

The sim keeps writing JSONL through the existing `FileSink`; a separate server **tails the active trace
file** and pushes new lines over a WebSocket. This is the platform design's Layer-2 "the trace file is
the queue" (§4): the sim stays fully decoupled from any viewer, replay and live read the **same
artifact**, and no `WebSocketSink` couples the training loop to a socket. `WebSocketSink`/Redis remain
future fan-out concerns (platform §8 Phase 4).

Data flow:

```
sim ──FileSink (flush per line)──▶ active.jsonl ──tail poll──▶ FastAPI ──WS──▶ WebSocketTraceSource ──▶ TraceStore ──▶ existing UI
                                   active.jsonl.done ────────────────────────▶ {type:"end"}
```

## 3. Components

Five independently testable units. Each states what it does, its interface, and its dependencies.

### 3.1 Producer change — `FileSink` flushes per line (tiny)

`FileSink.open()` and `FileSink.write()` (`src/neuromorphic/monitor/sink.py`) currently use default
buffering, so a tailing reader sees nothing until `close()`. Add a `flush()` after the header write and
after each frame write so the active file is tailable in real time.

- **Interface:** unchanged (`open/write/close`).
- **Effect:** output bytes are **identical** to today (flushing changes only *when* bytes hit disk, not
  their content), so every existing replay artifact and test is byte-for-byte unaffected.
- **Depends on:** nothing new.

### 3.2 Live server — `src/neuromorphic/server/` (new)

A small FastAPI + uvicorn app exposing one WebSocket endpoint.

- **Endpoint:** `WS /stream`. On client connect the server:
  1. Opens the configured trace file, reads **line 0** → sends `{"type":"header","data":<header>}`.
  2. Replays **all existing frames** → `{"type":"frame","data":<frame>}` each (catch-up-from-start, so
     a late viewer gets full context — same artifact as replay).
  3. **Tails** the file (poll every `--poll` seconds, default 0.15) and sends each newly appended line as
     a `frame`.
- **End-of-run:** the server watches a `<trace>.done` **sidecar file**; when it appears, sends
  `{"type":"end"}` and stops tailing. This keeps the JSONL file pure (header + frames only — no control
  lines to pollute the existing parsers) and gives a deterministic end signal. **Absent `.done` ⇒ keep
  tailing** (safe default; a slow/paused sim is never mistaken for "ended").
- **Message envelope (server → client):** `{type: "header"|"frame"|"end"|"error", data?}`. `error`
  carries a short reason (e.g. missing trace file).
- **Launch:** `python -m neuromorphic.server --trace <path> [--port 8000] [--poll 0.15]`.
- **Depends on:** `fastapi`, `uvicorn[standard]`; reads the JSONL the `FileSink` writes.

### 3.3 Frontend source — `WebSocketTraceSource.ts` (new)

Implements the existing `TraceSource` interface (`open(): Promise<Header>`, `subscribe(onFrame)`,
`close()`) in `dashboard/src/source/`.

- **open():** connect to the WS URL; resolve the promise on the first `header` message.
- **subscribe(onFrame):** each `frame` message → `onFrame(frame)`; `end` → mark ended; `error` → mark
  error.
- **Reconnect:** a socket close/error **before** an `end` triggers auto-reconnect with **exponential
  backoff** (capped). On reconnect the server re-sends header + all-frames-so-far; the client **resets**
  store frames and rebuilds from the fresh stream (catch-up replays fast; guarantees consistent state
  without diff/resume bookkeeping).
- **Connection state:** emits state transitions (see 3.4) via a callback/store setter:
  `connecting → live → (reconnecting ↔ live) → ended | error`.
- **Depends on:** native `WebSocket` (no new frontend deps); the shared `contract.ts` types.

### 3.4 Store + UI wiring (minimal)

- **`TraceStore`** (`dashboard/src/store/`): add `connectionState:
  "connecting"|"live"|"reconnecting"|"ended"|"error"` and **follow-live** behavior — when a new frame
  arrives, advance `envStep` to the newest frame (playhead tracks the tail). For MVP follow-live is
  **unconditional** — a new frame always advances the playhead; "scrubbing back detaches follow" is a
  deferred polish. Existing `frames[]`/append and scrubbing stay intact for the file path.
- **Source selection** (`App.tsx`): if `VITE_WS_URL` is set → `WebSocketTraceSource`, else the current
  `FileTraceSource`. **The render path is unchanged** — this is the seam.
- **`TopBar`**: a **LIVE badge** reflecting `connectionState` (live = solid; reconnecting = pulsing;
  ended = static "ended"; error = warning). Small, data-driven, no layout changes elsewhere.
- **Depends on:** 3.3.

### 3.5 Dev driver — `scripts/replay-into-file.py` (new)

Appends an existing trace's frames into a **fresh** JSONL one-at-a-time with a delay, then writes the
`.done` sidecar — simulating a running sim. Drives TDD, the reconnect/end-of-run UX, and demos.

- **Usage:** `python scripts/replay-into-file.py <source.jsonl> <dest.jsonl> [--delay 0.2]`.
- **Real runs:** a real REINFORCE run (`experiments/023_*`) already produces a live-tailable trace now
  that `FileSink` flushes; the run/experiment can `touch <trace>.done` after `close()` for the end
  signal. (Wiring that touch into the runner is optional and out of scope here.)
- **Depends on:** nothing new; reuses the JSONL format.

## 4. Error handling & edge cases

- **Socket drop before `end`** → reconnect with capped exponential backoff; state `reconnecting`.
- **`error` envelope** (e.g. trace file missing at connect) → state `error`, stop; one clean message, no
  reconnect storm.
- **Missing/empty trace at server start** → server waits for the file/first line rather than crashing;
  emits `error` only if explicitly unreadable.
- **Reconnect consistency** → full header+frames re-send + client store reset (idempotent); no partial
  resume logic.
- **`.done` never written** (e.g. killed sim) → stream stays `live`/tailing; the user can disconnect.
  Acceptable for MVP.
- **Poll latency** → frames appear within ~one poll interval (default 150 ms); tunable via `--poll`.

## 5. Testing strategy

- **Producer (Python):** `FileSink` flush test — after `write()`, the bytes are visible from a **second**
  file handle mid-run (proves tailability). Existing sink/runner tests stay green.
- **Server (Python):** FastAPI `TestClient` WebSocket tests — on connect receive `header` then existing
  `frame`s (catch-up); appending a line to the file yields a new `frame` (tail); creating `<trace>.done`
  yields `end`; a missing trace yields `error`.
- **Frontend (Vitest):** `WebSocketTraceSource` against a **mock WebSocket** — `open()` resolves on
  `header`; `frame`s call `onFrame`; a drop before `end` triggers backoff reconnect; `end` → ended;
  `error` → error. `TraceStore` transition tests for `connectionState` and follow-live (newest frame
  advances `envStep`).
- **Dev driver:** a unit test that it appends N frames and writes `.done`.
- **Unchanged paths:** all existing file-replay tests + the Playwright file smoke remain green. A **live**
  Playwright e2e (boot server + socket) is deferred and noted, not built.

## 6. Files touched / added

**Modified:** `src/neuromorphic/monitor/sink.py` (flush); `dashboard/src/store/traceStore.ts`
(connectionState + follow-live); `dashboard/src/App.tsx` (source selection);
`dashboard/src/shell/TopBar.tsx` (LIVE badge); Python deps (`pyproject.toml`/requirements — add
`fastapi`, `uvicorn[standard]`); `docs/superpowers/specs/2026-06-18-neuroscope-platform-design.md`
(mark Phase 2 status).

**Added:** `src/neuromorphic/server/__init__.py`, `.../__main__.py`, `.../app.py` (+ tail helper);
`dashboard/src/source/WebSocketTraceSource.ts` (+ `.test.ts`); `scripts/replay-into-file.py`;
`tests/server/…`; store test additions.

## 7. Verification (manual demo)

1. `python scripts/replay-into-file.py dashboard/public/week11_dashboard_trace.jsonl outputs/live.jsonl --delay 0.2`
2. `python -m neuromorphic.server --trace outputs/live.jsonl`
3. `VITE_WS_URL=ws://localhost:8000/stream npm run dev` → watch frames stream in, playhead follow the
   tail, the LIVE badge go solid, then flip to **ended** when `.done` lands. Kill/restart the server to
   see **reconnecting → live**.
