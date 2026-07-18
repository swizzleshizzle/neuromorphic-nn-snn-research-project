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
