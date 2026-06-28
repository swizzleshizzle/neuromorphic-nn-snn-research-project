"""``neuromorphic.monitor`` — server-side dashboard data contract.

Turns a running ``Brain`` episode into a versioned JSONL trace (header + per-step
Frames) via a ``TraceSink``. See docs/superpowers/specs/2026-06-17-stage2-dashboard-design.md.
"""

from neuromorphic.monitor.frame import build_frame
from neuromorphic.monitor.runner import record_episode, record_policy_episode
from neuromorphic.monitor.schema import REGION_OUTPUT_KEY, SCHEMA_VERSION, build_header, render_for_n
from neuromorphic.monitor.sink import FileSink, TraceSink

__all__ = [
    "SCHEMA_VERSION",
    "REGION_OUTPUT_KEY",
    "render_for_n",
    "build_header",
    "build_frame",
    "TraceSink",
    "FileSink",
    "record_episode",
    "record_policy_episode",
]
