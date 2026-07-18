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
