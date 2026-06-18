"""``TraceSink`` — interchangeable destinations for header + frames.

``FileSink`` writes JSONL (the system of record). ``WebSocketSink`` /
``RedisStreamSink`` will implement the same interface later — the same Frame
object, a different ``write``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path


class TraceSink(ABC):
    """Destination for one run: ``open(header)`` once, ``write(frame)`` per step, ``close()``."""

    @abstractmethod
    def open(self, header: dict) -> None: ...

    @abstractmethod
    def write(self, frame: dict) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class FileSink(TraceSink):
    """Append header + frames to a JSONL file (header = line 0)."""

    def __init__(self, path):
        self.path = Path(path)
        self._fh = None

    def open(self, header: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(json.dumps(header) + "\n")

    def write(self, frame: dict) -> None:
        if self._fh is None:
            raise RuntimeError("FileSink.write called before open()")
        self._fh.write(json.dumps(frame) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
