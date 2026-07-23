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


async def _watch_disconnect(ws, alive) -> None:
    """Flip ``alive`` false as soon as the client disconnects.

    ``stream_trace``'s poll loops only notice a drop when the next ``send`` raises, so an
    idle client (no new frames, no ``.done``) would go unnoticed and leak a coroutine plus
    file handle. Awaiting ``ws.receive()`` surfaces the disconnect even with no send in flight.
    """
    try:
        while alive["v"]:
            # Low-level receive() returns the disconnect as a message (it does not raise);
            # a second call after that would raise RuntimeError, so return on it immediately.
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                alive["v"] = False
                return
    except WebSocketDisconnect:  # defensive: higher-level receive() variants raise instead
        alive["v"] = False


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
        watcher = asyncio.create_task(_watch_disconnect(ws, alive))
        try:
            await stream_trace(ws.send_json, trace_path, done_path, poll, lambda: alive["v"])
        except WebSocketDisconnect:
            alive["v"] = False
        finally:
            alive["v"] = False  # let the watcher's loop guard exit
            watcher.cancel()
            try:
                await watcher
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass

    return app
