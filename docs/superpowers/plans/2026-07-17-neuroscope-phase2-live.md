# NEURO·SCOPE Phase 2 — Live Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream a running sim's JSONL trace into the existing NEURO·SCOPE UI live, via a file-tailing WebSocket server, with auto-reconnect and a connection-state indicator.

**Architecture:** The sim keeps writing JSONL through `FileSink` (now flushing per line). A new FastAPI server tails the active file and pushes `header`/`frame`/`end`/`error` envelopes over a WebSocket. A new `WebSocketTraceSource` implements the existing `TraceSource` interface and feeds the unchanged render path; the store gains follow-live + connection state. A dev "slow-writer" script simulates a running sim for TDD/demos.

**Tech Stack:** Python (FastAPI, uvicorn, httpx for TestClient), existing `neuromorphic.monitor`; TypeScript/React/Zustand/Vitest frontend (native `WebSocket`, no new deps).

**Spec:** `docs/superpowers/specs/2026-07-17-neuroscope-phase2-live-design.md`

## Global Constraints

- Python: src-layout package `neuromorphic` (`src/neuromorphic/…`), `requires-python >=3.10`. Run tests with `.venv\Scripts\python.exe -m pytest`. Ruff line-length 100, `E501` ignored.
- The **render path must stay agnostic to source origin** — do not change panels/hero/shell; live is a new `TraceSource` only.
- The JSONL trace file stays **pure** (line 0 header, every other line a Frame) — end-of-run is signaled by a `<trace>.done` sidecar file, never a control line inside the JSONL.
- Server→client WebSocket messages are typed envelopes: `{"type": "header"|"frame"|"end"|"error", "data"?: ...}`.
- Frontend adds **no new npm dependencies** (native `WebSocket`). Vitest run from `dashboard/`.
- Existing file-replay tests (`tests/monitor/*`, `tests/dashboard/*`, `dashboard/src/**/*.test.ts`, Playwright smoke) must stay green and unchanged.

---

### Task 1: `FileSink` flushes per line (tailable producer)

**Files:**
- Modify: `src/neuromorphic/monitor/sink.py:35-43`
- Test: `tests/monitor/test_sink.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `FileSink.open(header)` / `FileSink.write(frame)` guarantee bytes are on disk after each call (no API change).

- [ ] **Step 1: Write the failing test**

Add to `tests/monitor/test_sink.py`:

```python
def test_filesink_flushes_each_line_for_tailing(tmp_path):
    # A second reader must see header + frame BEFORE close() (proves the file is tailable live).
    path = tmp_path / "trace.jsonl"
    sink = FileSink(path)
    sink.open({"schema_version": "1.0"})
    with path.open("r", encoding="utf-8") as reader:
        assert reader.readline() != ""  # header visible before any write/close
        sink.write({"step": 0})
        assert json.loads(reader.readline())["step"] == 0  # frame visible before close
    sink.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/monitor/test_sink.py::test_filesink_flushes_each_line_for_tailing -v`
Expected: FAIL — the second reader gets `""` because the write is still buffered.

- [ ] **Step 3: Add flush calls**

In `src/neuromorphic/monitor/sink.py`, add `self._fh.flush()` at the end of both `open` and `write`:

```python
    def open(self, header: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(json.dumps(header) + "\n")
        self._fh.flush()

    def write(self, frame: dict) -> None:
        if self._fh is None:
            raise RuntimeError("FileSink.write called before open()")
        self._fh.write(json.dumps(frame) + "\n")
        self._fh.flush()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/monitor/test_sink.py -v`
Expected: PASS (new test + the two existing sink tests — output content is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/monitor/sink.py tests/monitor/test_sink.py
git commit -m "feat(monitor): flush FileSink per line so the active trace is tailable"
```

---

### Task 2: Live server — `create_app` + file-tail stream

**Files:**
- Create: `src/neuromorphic/server/__init__.py`
- Create: `src/neuromorphic/server/app.py`
- Modify: `pyproject.toml` (add a `server` optional-dependency group)
- Test: `tests/server/__init__.py` (empty), `tests/server/test_app.py`

**Interfaces:**
- Consumes: the JSONL written by `FileSink` (Task 1); a `<trace>.done` sidecar as the end signal.
- Produces: `create_app(trace_path, *, poll: float = 0.15) -> fastapi.FastAPI` exposing `WS /stream`; each client receives `{"type":"header","data":<header>}`, then `{"type":"frame","data":<frame>}` for every existing then newly-appended line, then `{"type":"end"}` when `<trace>.done` appears, or `{"type":"error","data":{"reason":...}}` if the trace file is absent at connect.

- [ ] **Step 1: Add the server dependencies**

In `pyproject.toml`, after the `[project]` table's `classifiers`, add:

```toml
[project.optional-dependencies]
server = ["fastapi>=0.110", "uvicorn[standard]>=0.29", "httpx>=0.27"]
```

Install: `.venv\Scripts\python.exe -m pip install -e ".[server]"`

- [ ] **Step 2: Write the failing test**

Create `tests/server/__init__.py` (empty) and `tests/server/test_app.py`:

```python
import json
from pathlib import Path

from fastapi.testclient import TestClient

from neuromorphic.server.app import create_app


def _write(path, *objs):
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o) + "\n")
        f.flush()


def test_streams_header_then_existing_frames(tmp_path):
    trace = tmp_path / "t.jsonl"
    _write(trace, {"schema_version": "1.0", "brain": {"T": 1}}, {"step": 0})
    client = TestClient(create_app(trace, poll=0.02))
    with client.websocket_connect("/stream") as ws:
        assert ws.receive_json() == {"type": "header", "data": {"schema_version": "1.0", "brain": {"T": 1}}}
        msg = ws.receive_json()
        assert msg["type"] == "frame" and msg["data"]["step"] == 0


def test_tails_newly_appended_frames_then_ends(tmp_path):
    trace = tmp_path / "t.jsonl"
    _write(trace, {"schema_version": "1.0"}, {"step": 0})
    client = TestClient(create_app(trace, poll=0.02))
    with client.websocket_connect("/stream") as ws:
        assert ws.receive_json()["type"] == "header"
        assert ws.receive_json()["data"]["step"] == 0
        with trace.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": 1}) + "\n")
            f.flush()
        assert ws.receive_json()["data"]["step"] == 1
        Path(str(trace) + ".done").write_text("", encoding="utf-8")
        assert ws.receive_json() == {"type": "end"}


def test_missing_trace_sends_error(tmp_path):
    client = TestClient(create_app(tmp_path / "nope.jsonl", poll=0.02))
    with client.websocket_connect("/stream") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error" and "nope.jsonl" in msg["data"]["reason"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/server/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: neuromorphic.server.app`.

- [ ] **Step 4: Write the server**

Create `src/neuromorphic/server/__init__.py`:

```python
"""NEURO·SCOPE live server: tail the active JSONL trace and push it over a WebSocket."""

from neuromorphic.server.app import create_app

__all__ = ["create_app"]
```

Create `src/neuromorphic/server/app.py`:

```python
"""FastAPI app that tails a JSONL trace file and streams it over a WebSocket.

The trace file is the queue (platform design §4): the sim writes JSONL via FileSink;
this server tails it and pushes typed envelopes. End-of-run is a `<trace>.done` sidecar,
so the JSONL itself stays pure (header + frames only).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


async def _read_full_line(fh, poll: float, alive) -> str | None:
    """Return the next newline-terminated line, waiting for partial writes to complete."""
    while alive():
        pos = fh.tell()
        line = fh.readline()
        if line.endswith("\n"):
            return line
        if line:
            fh.seek(pos)  # partial line mid-flush; retry from its start
        await asyncio.sleep(poll)
    return None


async def stream_trace(send, trace_path: Path, done_path: Path, poll: float, alive) -> None:
    """Send header + all frames (existing then tailed) until `done_path` appears or the client drops."""
    if not trace_path.exists():
        await send({"type": "error", "data": {"reason": f"trace not found: {trace_path}"}})
        return
    with trace_path.open("r", encoding="utf-8") as fh:
        header_line = await _read_full_line(fh, poll, alive)
        if header_line is None:
            return
        await send({"type": "header", "data": json.loads(header_line)})
        while alive():
            pos = fh.tell()
            line = fh.readline()
            if line.endswith("\n"):
                await send({"type": "frame", "data": json.loads(line)})
                continue
            if line:
                fh.seek(pos)
            if done_path.exists():
                await send({"type": "end"})
                return
            await asyncio.sleep(poll)


def create_app(trace_path, *, poll: float = 0.15) -> FastAPI:
    app = FastAPI()
    trace_path = Path(trace_path)
    done_path = Path(str(trace_path) + ".done")

    @app.websocket("/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        alive = {"v": True}
        try:
            await stream_trace(ws.send_json, trace_path, done_path, poll, lambda: alive["v"])
        except WebSocketDisconnect:
            alive["v"] = False

    return app
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/server/test_app.py -v`
Expected: PASS (all three).

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/server tests/server pyproject.toml
git commit -m "feat(server): FastAPI file-tail WebSocket stream (header/frame/end/error)"
```

---

### Task 3: Server CLI — `python -m neuromorphic.server`

**Files:**
- Create: `src/neuromorphic/server/__main__.py`
- Test: `tests/server/test_cli.py`

**Interfaces:**
- Consumes: `create_app` (Task 2).
- Produces: `build_parser() -> argparse.ArgumentParser` and `main(argv=None)`; CLI `python -m neuromorphic.server --trace <path> [--host 127.0.0.1] [--port 8000] [--poll 0.15]`.

- [ ] **Step 1: Write the failing test**

Create `tests/server/test_cli.py`:

```python
from neuromorphic.server.__main__ import build_parser


def test_parser_defaults_and_overrides():
    args = build_parser().parse_args(["--trace", "outputs/live.jsonl"])
    assert args.trace == "outputs/live.jsonl"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.poll == 0.15

    args = build_parser().parse_args(["--trace", "x.jsonl", "--port", "9001", "--poll", "0.05"])
    assert args.port == 9001 and args.poll == 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/server/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: neuromorphic.server.__main__`.

- [ ] **Step 3: Write the CLI**

Create `src/neuromorphic/server/__main__.py`:

```python
"""CLI entry point: python -m neuromorphic.server --trace <path>."""

from __future__ import annotations

import argparse

import uvicorn

from neuromorphic.server.app import create_app


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m neuromorphic.server")
    p.add_argument("--trace", required=True, help="Path to the active JSONL trace to tail.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--poll", type=float, default=0.15, help="Tail poll interval (seconds).")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    app = create_app(args.trace, poll=args.poll)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/server/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/server/__main__.py tests/server/test_cli.py
git commit -m "feat(server): CLI entry point (python -m neuromorphic.server)"
```

---

### Task 4: Dev driver — synthetic slow-writer

**Files:**
- Create: `scripts/replay_into_file.py`
- Test: `tests/scripts/__init__.py` (empty), `tests/scripts/test_replay_into_file.py`

**Interfaces:**
- Consumes: an existing JSONL trace.
- Produces: `replay_into_file(source, dest, delay, sleep=time.sleep) -> None` — appends source lines into a fresh `dest` one at a time (calling `sleep(delay)` between frames), then writes `<dest>.done`. CLI `python scripts/replay_into_file.py <source> <dest> [--delay 0.2]`.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/__init__.py` (empty) and `tests/scripts/test_replay_into_file.py`:

```python
import importlib.util
import json
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "replay_into_file.py"
_spec = importlib.util.spec_from_file_location("replay_into_file", _MOD_PATH)
replay_into_file_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_into_file_mod)


def test_replays_lines_and_writes_done(tmp_path):
    source = tmp_path / "src.jsonl"
    source.write_text(
        json.dumps({"schema_version": "1.0"}) + "\n" + json.dumps({"step": 0}) + "\n" + json.dumps({"step": 1}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out" / "live.jsonl"
    calls = []
    replay_into_file_mod.replay_into_file(source, dest, delay=0.2, sleep=calls.append)

    lines = dest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["schema_version"] == "1.0"
    assert json.loads(lines[2])["step"] == 1
    assert Path(str(dest) + ".done").exists()
    assert calls == [0.2, 0.2, 0.2]  # one sleep per line written
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/test_replay_into_file.py -v`
Expected: FAIL — `scripts/replay_into_file.py` does not exist.

- [ ] **Step 3: Write the script**

Create `scripts/replay_into_file.py`:

```python
"""Dev driver: replay an existing JSONL trace into a fresh file slowly, to simulate a live sim.

Usage: python scripts/replay_into_file.py <source.jsonl> <dest.jsonl> [--delay 0.2]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def replay_into_file(source, dest, delay: float, sleep=time.sleep) -> None:
    lines = [ln for ln in Path(source).read_text(encoding="utf-8").splitlines() if ln.strip()]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = Path(str(dest) + ".done")
    if done.exists():
        done.unlink()
    with dest.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
            f.flush()
            sleep(delay)
    done.write_text("", encoding="utf-8")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("dest")
    p.add_argument("--delay", type=float, default=0.2)
    args = p.parse_args(argv)
    replay_into_file(args.source, args.dest, args.delay)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/scripts/test_replay_into_file.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/replay_into_file.py tests/scripts
git commit -m "feat(scripts): synthetic slow-writer to drive live-stream TDD/demos"
```

---

### Task 5: Store — connection state + follow-live append

**Files:**
- Modify: `dashboard/src/store/traceStore.ts`
- Test: `dashboard/src/store/traceStore.live.test.ts`

**Interfaces:**
- Consumes: `TraceHeader`, `Frame` from `../contract`.
- Produces: on the store — `connectionState: ConnectionState` (initial `"idle"`), `setConnectionState(s)`, and `appendFrame(frame)` which pushes a frame and advances `envStep` to the newest (follow-live). Exports `type ConnectionState = "idle" | "connecting" | "live" | "reconnecting" | "ended" | "error"`.

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/store/traceStore.live.test.ts`:

```ts
import { beforeEach, describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "./traceStore";

const header = { schema_version: "1.0", brain: { id: "b", config_hash: "h", seed: 0, T: 1 } } as unknown as TraceHeader;
const frame = (step: number) => ({ episode: 0, step, t: step } as unknown as Frame);

describe("traceStore live", () => {
  beforeEach(() => useTraceStore.getState().reset());

  it("appendFrame appends and follows the tail", () => {
    const s = useTraceStore.getState();
    s.load(header, []);
    s.appendFrame(frame(0));
    expect(useTraceStore.getState().frames.length).toBe(1);
    expect(useTraceStore.getState().envStep).toBe(0);
    s.appendFrame(frame(1));
    expect(useTraceStore.getState().frames.length).toBe(2);
    expect(useTraceStore.getState().envStep).toBe(1);
  });

  it("connectionState defaults to idle and is settable", () => {
    expect(useTraceStore.getState().connectionState).toBe("idle");
    useTraceStore.getState().setConnectionState("live");
    expect(useTraceStore.getState().connectionState).toBe("live");
  });

  it("reset restores idle connectionState", () => {
    useTraceStore.getState().setConnectionState("ended");
    useTraceStore.getState().reset();
    expect(useTraceStore.getState().connectionState).toBe("idle");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/store/traceStore.live.test.ts`
Expected: FAIL — `appendFrame`/`connectionState`/`setConnectionState` do not exist.

- [ ] **Step 3: Extend the store**

In `dashboard/src/store/traceStore.ts`: add the exported type, the two fields/actions, and reset them. Full updated interface + relevant bodies:

```ts
export type ConnectionState = "idle" | "connecting" | "live" | "reconnecting" | "ended" | "error";

interface TraceStore {
  header?: TraceHeader;
  frames: Frame[];
  T: number;
  envStep: number;
  winTi: number;
  playing: boolean;
  heroLayout: "cloud" | "flow";
  connectionState: ConnectionState;

  load: (header: TraceHeader, frames: Frame[]) => void;
  appendFrame: (frame: Frame) => void;
  setConnectionState: (s: ConnectionState) => void;
  setEnvStep: (i: number) => void;
  setWinTi: (ti: number) => void;
  play: () => void;
  pause: () => void;
  tickWindow: () => void;
  reset: () => void;
  setHeroLayout: (v: "cloud" | "flow") => void;
}
```

Add to the `create(...)` object (initial value + actions), and update `reset`:

```ts
  connectionState: "idle",

  appendFrame: (frame) =>
    set((s) => {
      const frames = [...s.frames, frame];
      return { frames, envStep: frames.length - 1 }; // follow-live (unconditional for MVP)
    }),

  setConnectionState: (connectionState) => set({ connectionState }),

  reset: () =>
    set({ header: undefined, frames: [], T: 1, envStep: 0, winTi: 0, playing: false, connectionState: "idle" }),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && npx vitest run src/store/traceStore.live.test.ts src/store/traceStore.test.ts`
Expected: PASS (new live tests + the existing store tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/store/traceStore.ts dashboard/src/store/traceStore.live.test.ts
git commit -m "feat(dashboard): store connectionState + follow-live appendFrame"
```

---

### Task 6: `WebSocketTraceSource`

**Files:**
- Create: `dashboard/src/source/WebSocketTraceSource.ts`
- Test: `dashboard/src/source/WebSocketTraceSource.test.ts`

**Interfaces:**
- Consumes: the server envelopes (Task 2); the `TraceSource` interface; `ConnectionState` (Task 5).
- Produces: `class WebSocketTraceSource implements TraceSource`. Constructor `(url: string, opts?: WebSocketTraceSourceOpts)`; `open()` resolves on the first `header` (rejects on a pre-open `error`); `subscribe(onFrame)`; `close()`. `opts`: `onState?(s)`, `onReconnect?(header)` (fires on a header received after the first — consumer should reset store), `wsFactory?(url)`, `baseDelayMs?` (default 250), `maxDelayMs?` (default 5000). A drop before `end` triggers backoff reconnect; `end` → `ended` state + stop; `error` → `error` state + stop.

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/source/WebSocketTraceSource.test.ts`:

```ts
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WebSocketTraceSource } from "./WebSocketTraceSource";

class FakeWS {
  static instances: FakeWS[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  constructor(public url: string) {
    FakeWS.instances.push(this);
  }
  send(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  close() {
    this.closed = true;
    this.onclose?.();
  }
}

const factory = (url: string) => new FakeWS(url) as never;

beforeEach(() => {
  FakeWS.instances = [];
  vi.useFakeTimers();
});
afterEach(() => vi.useRealTimers());

describe("WebSocketTraceSource", () => {
  it("resolves open() on the header and delivers frames", async () => {
    const states: string[] = [];
    const src = new WebSocketTraceSource("ws://x/stream", { wsFactory: factory, onState: (s) => states.push(s) });
    const p = src.open();
    const frames: number[] = [];
    src.subscribe((f) => frames.push((f as { step: number }).step));
    FakeWS.instances[0].send({ type: "header", data: { schema_version: "1.0" } });
    await expect(p).resolves.toEqual({ schema_version: "1.0" });
    FakeWS.instances[0].send({ type: "frame", data: { step: 0 } });
    FakeWS.instances[0].send({ type: "frame", data: { step: 1 } });
    expect(frames).toEqual([0, 1]);
    expect(states).toContain("connecting");
    expect(states).toContain("live");
  });

  it("reconnects with backoff on a drop before end, firing onReconnect on the new header", async () => {
    const states: string[] = [];
    const reconns: unknown[] = [];
    const src = new WebSocketTraceSource("ws://x/stream", {
      wsFactory: factory,
      onState: (s) => states.push(s),
      onReconnect: (h) => reconns.push(h),
      baseDelayMs: 100,
    });
    const p = src.open();
    src.subscribe(() => {});
    FakeWS.instances[0].send({ type: "header", data: { schema_version: "1.0" } });
    await p;
    FakeWS.instances[0].close(); // drop before end
    expect(states).toContain("reconnecting");
    vi.advanceTimersByTime(100);
    expect(FakeWS.instances.length).toBe(2); // reconnected
    FakeWS.instances[1].send({ type: "header", data: { schema_version: "1.0" } });
    expect(reconns.length).toBe(1); // second header => reset signal
  });

  it("goes ended on end and does not reconnect", async () => {
    const states: string[] = [];
    const src = new WebSocketTraceSource("ws://x/stream", { wsFactory: factory, onState: (s) => states.push(s) });
    const p = src.open();
    src.subscribe(() => {});
    FakeWS.instances[0].send({ type: "header", data: {} });
    await p;
    FakeWS.instances[0].send({ type: "end" });
    expect(states.at(-1)).toBe("ended");
    vi.advanceTimersByTime(10000);
    expect(FakeWS.instances.length).toBe(1); // no reconnect after end
  });

  it("rejects open() and goes error on a pre-open error", async () => {
    const src = new WebSocketTraceSource("ws://x/stream", { wsFactory: factory });
    const p = src.open();
    FakeWS.instances[0].send({ type: "error", data: { reason: "trace not found" } });
    await expect(p).rejects.toThrow("trace not found");
    vi.advanceTimersByTime(10000);
    expect(FakeWS.instances.length).toBe(1); // no reconnect after error
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/source/WebSocketTraceSource.test.ts`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the source**

Create `dashboard/src/source/WebSocketTraceSource.ts`:

```ts
import type { Frame, TraceHeader } from "../contract";
import type { ConnectionState } from "../store/traceStore";
import type { TraceSource } from "./TraceSource";

interface WSLike {
  onopen: (() => void) | null;
  onmessage: ((ev: { data: string }) => void) | null;
  onclose: (() => void) | null;
  onerror: (() => void) | null;
  close(): void;
}

export interface WebSocketTraceSourceOpts {
  onState?: (s: ConnectionState) => void;
  /** Fires when a header arrives after the first one (post-reconnect). Consumer should reset the store. */
  onReconnect?: (header: TraceHeader) => void;
  wsFactory?: (url: string) => WSLike;
  baseDelayMs?: number;
  maxDelayMs?: number;
}

/** Live TraceSource: connects to the file-tail WebSocket server, with auto-reconnect. */
export class WebSocketTraceSource implements TraceSource {
  private ws?: WSLike;
  private onFrame?: (f: Frame) => void;
  private opened = false;
  private ended = false;
  private errored = false;
  private closed = false;
  private retry = 0;
  private resolveHeader?: (h: TraceHeader) => void;
  private rejectHeader?: (e: Error) => void;

  constructor(private readonly url: string, private readonly opts: WebSocketTraceSourceOpts = {}) {}

  open(): Promise<TraceHeader> {
    return new Promise<TraceHeader>((resolve, reject) => {
      this.resolveHeader = resolve;
      this.rejectHeader = reject;
      this.connect();
    });
  }

  subscribe(onFrame: (f: Frame) => void): void {
    this.onFrame = onFrame;
  }

  close(): void {
    this.closed = true;
    this.ws?.close();
  }

  private setState(s: ConnectionState): void {
    this.opts.onState?.(s);
  }

  private connect(): void {
    this.setState(this.opened ? "reconnecting" : "connecting");
    const make = this.opts.wsFactory ?? ((u: string) => new WebSocket(u) as unknown as WSLike);
    const ws = make(this.url);
    this.ws = ws;
    ws.onmessage = (ev) => this.handle(ev.data);
    ws.onclose = () => this.handleClose();
    ws.onerror = () => {}; // onclose drives reconnect
  }

  private handle(data: string): void {
    const msg = JSON.parse(data) as { type: string; data?: unknown };
    if (msg.type === "header") {
      this.retry = 0;
      this.setState("live");
      if (!this.opened) {
        this.opened = true;
        this.resolveHeader?.(msg.data as TraceHeader);
      } else {
        this.opts.onReconnect?.(msg.data as TraceHeader);
      }
    } else if (msg.type === "frame") {
      this.onFrame?.(msg.data as Frame);
    } else if (msg.type === "end") {
      this.ended = true;
      this.setState("ended");
      this.ws?.close();
    } else if (msg.type === "error") {
      this.errored = true;
      this.setState("error");
      if (!this.opened) this.rejectHeader?.(new Error(String((msg.data as { reason?: string })?.reason ?? "stream error")));
      this.ws?.close();
    }
  }

  private handleClose(): void {
    if (this.closed || this.ended || this.errored) return;
    const base = this.opts.baseDelayMs ?? 250;
    const max = this.opts.maxDelayMs ?? 5000;
    const delay = Math.min(max, base * 2 ** this.retry);
    this.retry += 1;
    this.setState("reconnecting");
    setTimeout(() => {
      if (!this.closed && !this.ended && !this.errored) this.connect();
    }, delay);
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && npx vitest run src/source/WebSocketTraceSource.test.ts`
Expected: PASS (all four).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/source/WebSocketTraceSource.ts dashboard/src/source/WebSocketTraceSource.test.ts
git commit -m "feat(dashboard): WebSocketTraceSource with auto-reconnect + connection state"
```

---

### Task 7: Wire live source into `App` + LIVE badge in `TopBar`

**Files:**
- Modify: `dashboard/src/App.tsx`
- Modify: `dashboard/src/shell/TopBar.tsx`
- Test: `dashboard/src/shell/TopBar.test.tsx`

**Interfaces:**
- Consumes: `WebSocketTraceSource` (Task 6), store `appendFrame`/`setConnectionState`/`connectionState` (Task 5).
- Produces: `App` selects `WebSocketTraceSource` when `import.meta.env.VITE_WS_URL` is set (else `FileTraceSource`, unchanged); `TopBar` renders a `data-testid="live-badge"` element whose text reflects `connectionState` when it is not `"idle"`.

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/shell/TopBar.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import type { TraceHeader } from "../contract";
import { useTraceStore } from "../store/traceStore";
import { TopBar } from "./TopBar";

const header = { schema_version: "1.0", brain: { id: "b", config_hash: "h", seed: 0, T: 1 } } as unknown as TraceHeader;

describe("TopBar live badge", () => {
  beforeEach(() => {
    useTraceStore.getState().reset();
    useTraceStore.getState().load(header, []);
  });

  it("hides the badge when idle", () => {
    expect(screen.queryByTestId("live-badge")).toBeNull();
    render(<TopBar />);
    expect(screen.queryByTestId("live-badge")).toBeNull();
  });

  it("shows connection state when live", () => {
    useTraceStore.getState().setConnectionState("live");
    render(<TopBar />);
    expect(screen.getByTestId("live-badge").textContent?.toLowerCase()).toContain("live");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/shell/TopBar.test.tsx`
Expected: FAIL — no `live-badge` element.

- [ ] **Step 3: Add the badge to `TopBar`**

In `dashboard/src/shell/TopBar.tsx`, read `connectionState` and render a badge when it isn't `idle`. Replace the component body:

```tsx
import { useTraceStore } from "../store/traceStore";

const BADGE_COLOR: Record<string, string> = {
  connecting: "#9aa3b6",
  live: "#4ade80",
  reconnecting: "#ffd24a",
  ended: "#9aa3b6",
  error: "#ff6b6b",
};

export function TopBar() {
  const header = useTraceStore((s) => s.header);
  const conn = useTraceStore((s) => s.connectionState);
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
      {conn !== "idle" && (
        <span
          data-testid="live-badge"
          style={{ marginLeft: "auto", color: BADGE_COLOR[conn], font: "700 12px monospace", textTransform: "uppercase" }}
        >
          ● {conn}
        </span>
      )}
    </header>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/shell/TopBar.test.tsx`
Expected: PASS.

- [ ] **Step 5: Wire source selection in `App`**

Replace `dashboard/src/App.tsx` with the source-selecting version (file path unchanged when `VITE_WS_URL` is unset):

```tsx
import { useEffect } from "react";
import { FileTraceSource } from "./source/FileTraceSource";
import { WebSocketTraceSource } from "./source/WebSocketTraceSource";
import { Shell } from "./shell/Shell";
import { useTraceStore } from "./store/traceStore";

const TRACE_URL = import.meta.env.VITE_TRACE_URL ?? "/week11_dashboard_trace.jsonl";
const WS_URL = import.meta.env.VITE_WS_URL as string | undefined;

export function App() {
  const header = useTraceStore((s) => s.header);
  const load = useTraceStore((s) => s.load);

  useEffect(() => {
    const store = useTraceStore.getState();
    let cancelled = false;

    if (WS_URL) {
      const source = new WebSocketTraceSource(WS_URL, {
        onState: store.setConnectionState,
        onReconnect: (hdr) => load(hdr, []), // reset on reconnect; frames re-stream from start
      });
      source.subscribe((f) => useTraceStore.getState().appendFrame(f));
      source
        .open()
        .then((hdr) => {
          if (!cancelled) load(hdr, []);
        })
        .catch((err) => console.error("live trace failed:", err));
      return () => {
        cancelled = true;
        source.close();
      };
    }

    const source = new FileTraceSource(TRACE_URL);
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

- [ ] **Step 6: Run the full frontend suite + typecheck**

Run: `cd dashboard && npx vitest run && npx tsc -b`
Expected: PASS — all unit tests green, no type errors. (The Playwright smoke on the file path is unaffected.)

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/App.tsx dashboard/src/shell/TopBar.tsx dashboard/src/shell/TopBar.test.tsx
git commit -m "feat(dashboard): live source selection (VITE_WS_URL) + LIVE badge"
```

---

### Task 8: Docs — roadmap status + run instructions

**Files:**
- Modify: `docs/superpowers/specs/2026-06-18-neuroscope-platform-design.md:110`
- Modify: `dashboard/README.md` (create if absent)

**Interfaces:** none (docs only).

- [ ] **Step 1: Mark Phase 2 status in the platform roadmap**

In `docs/superpowers/specs/2026-06-18-neuroscope-platform-design.md`, update the Phase 2 row of the §8 table to note it shipped, e.g. change the "**2 — Live**" row's "Ships" cell to end with: `— **SHIPPED 2026-07-17** (file-tail server + WebSocketTraceSource + reconnect; see plan 2026-07-17-neuroscope-phase2-live).`

- [ ] **Step 2: Add run instructions**

Create or append to `dashboard/README.md`:

```markdown
## Live monitoring (Phase 2)

Watch a running sim stream into the dashboard via the file-tail WebSocket server.

1. Simulate a live sim (or run a real experiment that writes JSONL via FileSink):
   `python scripts/replay_into_file.py dashboard/public/week11_dashboard_trace.jsonl outputs/live.jsonl --delay 0.2`
2. Start the server (needs `pip install -e ".[server]"`):
   `python -m neuromorphic.server --trace outputs/live.jsonl`
3. Start the dashboard pointed at the socket:
   `VITE_WS_URL=ws://localhost:8000/stream npm run dev`

Frames stream in, the playhead follows the tail, and the TopBar LIVE badge shows
`live / reconnecting / ended / error`. Without `VITE_WS_URL`, the dashboard loads a static
trace file exactly as before.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-06-18-neuroscope-platform-design.md dashboard/README.md
git commit -m "docs: mark NEURO-SCOPE Phase 2 live shipped + run instructions"
```

---

## Final verification (after all tasks)

- [ ] Python: `.venv\Scripts\python.exe -m pytest tests/monitor tests/server tests/scripts -v` — all green.
- [ ] Frontend: `cd dashboard && npx vitest run && npx tsc -b` — all green, no type errors.
- [ ] Manual demo (spec §7): run the slow-writer → server → `VITE_WS_URL=… npm run dev`; confirm frames stream, playhead follows, LIVE badge goes solid then flips to **ended** when `.done` lands; kill/restart the server to see **reconnecting → live**.

## Self-review notes (coverage against the spec)

- Spec §2 file-as-queue → Tasks 1–2. §3.1 FileSink flush → Task 1. §3.2 server + `.done` end + error → Task 2, CLI → Task 3. §3.3 WebSocketTraceSource + reconnect → Task 6. §3.4 store + App selection + LIVE badge → Tasks 5, 7. §3.5 slow-writer → Task 4. §4 error/edge cases → Tasks 2 (missing→error, tail/partial-line guard, `.done`) + 6 (drop→reconnect, end, error). §5 tests → every task is TDD; live Playwright e2e explicitly deferred. §6 files → all covered. §7 demo → final verification.
- Deferred per spec (not in this plan): `WebSocketSink`, Redis, multi-run registry, backpressure, live Playwright e2e, scrub-to-detach follow.
