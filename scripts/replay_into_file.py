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
