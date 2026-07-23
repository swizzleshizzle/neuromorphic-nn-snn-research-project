import asyncio
import json
from pathlib import Path

from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from neuromorphic.server.app import _watch_disconnect, create_app


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


def test_watch_disconnect_flips_alive_on_disconnect_message():
    """Low-level receive() RETURNS a disconnect message (it does not raise); detect that."""
    alive = {"v": True}

    class FakeWS:
        async def receive(self):
            return {"type": "websocket.disconnect", "code": 1000}

    asyncio.run(_watch_disconnect(FakeWS(), alive))
    assert alive["v"] is False


def test_watch_disconnect_loops_past_messages_then_flips():
    """Stray client messages are ignored; the watcher returns on the disconnect message
    without making the fatal second receive() call."""
    alive = {"v": True}

    class FakeWS:
        def __init__(self):
            self.calls = 0

        async def receive(self):
            self.calls += 1
            if self.calls < 3:
                return {"type": "websocket.receive", "text": "ping"}
            return {"type": "websocket.disconnect", "code": 1000}

    ws = FakeWS()
    asyncio.run(_watch_disconnect(ws, alive))
    assert alive["v"] is False
    assert ws.calls == 3  # exactly the disconnect call, no fatal fourth call


def test_watch_disconnect_handles_raised_disconnect():
    """Defensive path: a higher-level receive() that raises WebSocketDisconnect still flips alive."""
    alive = {"v": True}

    class FakeWS:
        async def receive(self):
            raise WebSocketDisconnect()

    asyncio.run(_watch_disconnect(FakeWS(), alive))
    assert alive["v"] is False
