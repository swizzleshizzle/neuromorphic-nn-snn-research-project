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
