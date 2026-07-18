import json

from neuromorphic.monitor.sink import FileSink


def test_filesink_writes_header_then_frames(tmp_path):
    path = tmp_path / "trace.jsonl"
    sink = FileSink(path)
    sink.open({"schema_version": "1.0", "regions": []})
    sink.write({"step": 0, "x": 1})
    sink.write({"step": 1, "x": 2})
    sink.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    header = json.loads(lines[0])
    assert header["schema_version"] == "1.0"
    assert json.loads(lines[1])["step"] == 0
    assert json.loads(lines[2])["step"] == 1


def test_filesink_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "trace.jsonl"
    sink = FileSink(path)
    sink.open({"schema_version": "1.0"})
    sink.close()
    assert path.exists()


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
