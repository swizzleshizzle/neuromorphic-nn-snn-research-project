"""NEURO·SCOPE live server: tail the active JSONL trace and push it over a WebSocket."""

from neuromorphic.server.app import create_app

__all__ = ["create_app"]
