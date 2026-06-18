"""EXP-022 — generate a dashboard trace (Week-11 S2, L17).

Runs one untrained five-region episode and writes a versioned JSONL trace to
``outputs/week11_dashboard_trace.jsonl`` (header line + one Frame per step). This
is the real-data artifact the Stage-2 dashboard / design system loads.

Run:
    python experiments/022_week11_dashboard_trace/run.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from neuromorphic.brain import Brain
from neuromorphic.envs import GridWorldEnv
from neuromorphic.monitor import FileSink, record_episode

OUT = Path("outputs/week11_dashboard_trace.jsonl")


def main() -> None:
    env = GridWorldEnv()
    brain = Brain(grid_n=env.size, seed=0)
    sink = FileSink(OUT)
    summary = record_episode(
        brain, env, sink, seed=0, generator=torch.Generator().manual_seed(0)
    )
    print("EXP-022 — dashboard trace written")
    print(f"  file         : {OUT}")
    print(f"  steps        : {summary['steps']}")
    print(f"  total reward : {summary['total_reward']:.0f}")
    print(f"  reached goal : {summary['reached_goal']}")


if __name__ == "__main__":
    main()
