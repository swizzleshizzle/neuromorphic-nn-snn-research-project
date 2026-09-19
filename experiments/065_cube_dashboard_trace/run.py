"""EXP-065 - the first CUBE trace NEURO-SCOPE has ever been given.

The cube path through the monitor and the dashboard was fully built and never used:
`CubeAdapter` emits `{"type": "cube", ...}` headers and facelet frames, `TaskState.tsx`
branches on that type at line 133, and `cubeNet.ts` lays out the unfolded net with tests. But
nothing ever CALLED it. `record_episode` defaults to `GridworldAdapter`, the only committed
traces are week-11 gridworld, and `dashboard/scripts/sync-trace.mjs` hardcodes the gridworld
file. So the dashboard has only ever rendered a 5x5 grid from July, through all of Phase 3.

> THE ACTION IN A TRACE IS THE BRAIN'S OWN, NOT THE POLICY'S. `record_episode` uses
> `out["action"]`, which comes from prefrontal -> router -> motor. Every cube experiment's
> policy instead used `head(concept)` and ignored that pathway entirely, and prefrontal and
> motor were frozen at random init. So this trace shows what the BRAIN would do, which until
> EXP-064 was not what the agent did. That is worth knowing while looking at it, and it is why
> a shallow scramble is used here: an untrained pathway is effectively a random walk.

Run:
    .venv/bin/python experiments/065_cube_dashboard_trace/run.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from neuromorphic.brain import Brain
from neuromorphic.encoders import cube_encoder
from neuromorphic.envs.cube import CubeEnv
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.monitor import CubeAdapter, FileSink, record_episode

OUT = Path("outputs/cube_dashboard_trace.jsonl")
CUBE_N_OBS = 144      # 24 facelets x 6 colors
CUBE_OBS_WIDTH = 24
DEPTH = 2             # shallow on purpose: the brain's own pathway is untrained
SEED = 0


def main() -> None:
    # A distance provider, so `info["distance"]` is populated and the panel can show it.
    # Without one CubeEnv leaves it None and the dashboard renders "distance -", which is how
    # the first trace came out. A BOUNDED build is near free (depth 6 is 11,913 states, ~0.04s);
    # only `max_depth=None` costs the 67 seconds. Distance stays an INSTRUMENT here: the agent
    # observes raw facelets and never sees it.
    # The table must cover every state the walk can REACH, not just the scramble depth. Built
    # at DEPTH first, distance came back None on 6 of 7 frames: the agent wanders AWAY from
    # solved, so a depth-2 table cannot score a state 3 moves out and the panel showed
    # "distance -". The reachable bound is the scramble depth plus the step budget.
    max_steps = 2 * DEPTH + 3
    provider = ExactBFSDistance(max_depth=DEPTH + max_steps)
    env = CubeEnv(scramble_depth=DEPTH, max_steps=max_steps, scramble_seed=SEED,
                  distance_provider=provider)
    brain = Brain(
        encoder=cube_encoder(), n_obs=CUBE_N_OBS, obs_width=CUBE_OBS_WIDTH,
        n_actions=env.action_space.n, content=64, seed=SEED,
    )
    sink = FileSink(OUT)
    summary = record_episode(
        brain, env, sink, seed=SEED, max_steps=max_steps, adapter=CubeAdapter(cube_n=2),
        generator=torch.Generator().manual_seed(SEED),
    )
    print("EXP-065 - cube dashboard trace written")
    print(f"  file         : {OUT}")
    print(f"  scramble     : depth {DEPTH}, {env.action_space.n} moves")
    print(f"  steps        : {summary['steps']}")
    print(f"  total reward : {summary['total_reward']:.0f}")
    print(f"  solved       : {summary['reached_goal']}")


if __name__ == "__main__":
    main()
