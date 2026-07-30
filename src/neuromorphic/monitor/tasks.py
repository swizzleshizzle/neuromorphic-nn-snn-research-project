"""Per-task trace blocks, so the monitor itself stays task agnostic.

A TaskAdapter owns the three places where the dashboard data contract is task
specific: the header's task block, each frame's task block, and the sensory
encoding block. Adding a task means adding an adapter, not adding a branch to
each of schema.py, frame.py and runner.py.
"""

from __future__ import annotations

from typing import Protocol

from neuromorphic.envs.cube import MOVE_LABELS

GRID_ACTION_LABELS: tuple[str, ...] = ("up", "right", "down", "left")


class TaskAdapter(Protocol):
    """The task-specific half of the trace contract."""

    action_labels: tuple[str, ...]

    def header_task(self) -> dict: ...

    def frame_task(self, obs, *, next_obs, action, reward, total, terminated, truncated, info) -> dict: ...

    def encoding(self, out: dict) -> dict | None: ...


class GridworldAdapter:
    """Reproduces the pre-adapter gridworld blocks exactly."""

    def __init__(self, grid_n: int, action_labels=GRID_ACTION_LABELS):
        self.grid_n = grid_n
        self.action_labels = tuple(action_labels)

    def header_task(self) -> dict:
        return {
            "type": "gridworld",
            "grid_n": self.grid_n,
            "action_labels": list(self.action_labels),
        }

    def frame_task(self, obs, *, next_obs, action, reward, total, terminated, truncated, info) -> dict:
        # Deliberately pre-move: gridworld's existing contract reads agent/goal from
        # ``obs``, and the digest fixture test pins that. ``next_obs`` is unused here.
        return {
            "agent": [int(obs[0]), int(obs[1])],
            "goal": [int(obs[2]), int(obs[3])],
            "action": action,
            "action_label": self.action_labels[action],
            "reward": float(reward),
            "return": total,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def encoding(self, out: dict) -> dict | None:
        if "obs_spikes" not in out:
            return None
        return {
            "sensory_input": {
                "spikes": out["obs_spikes"][:, 0, :].int().tolist(),
                "grid_n": self.grid_n,
                "planes": ["agent", "goal"],
                "index": "y*grid_n + x",
            }
        }


class CubeAdapter:
    """Cube blocks. Width comes from MOVE_LABELS, never a literal."""

    def __init__(self, cube_n: int = 2, n_colors: int = 6):
        self.cube_n = cube_n
        self.n_colors = n_colors
        self.action_labels = tuple(MOVE_LABELS)

    def header_task(self) -> dict:
        return {
            "type": "cube",
            "cube_n": self.cube_n,
            "action_labels": list(self.action_labels),
        }

    def frame_task(self, obs, *, next_obs, action, reward, total, terminated, truncated, info) -> dict:
        # A cube frame describes the state AFTER its move: facelets, solved and
        # distance must all agree, or the dashboard renders an unsolved cube
        # labelled solved (the defect this fixed). ``solved``/``distance`` already
        # come from the post-step ``info``, so ``facelets`` reads ``next_obs`` too.
        # Gridworld deliberately stays pre-move (see GridworldAdapter above) to
        # preserve its existing, digest-pinned contract.
        distance = info.get("distance")
        return {
            "facelets": [int(c) for c in next_obs],
            "solved": bool(info.get("solved", False)),
            "distance": None if distance is None else int(distance),
            "scramble_depth": int(info.get("scramble_depth", 0)),
            "move": action,
            "move_label": info.get("move_label"),
            "action": action,
            "action_label": self.action_labels[action],
            "reward": float(reward),
            "return": total,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def encoding(self, out: dict) -> dict | None:
        if "obs_spikes" not in out:
            return None
        return {
            "sensory_input": {
                "spikes": out["obs_spikes"][:, 0, :].int().tolist(),
                "cube_n": self.cube_n,
                "n_colors": self.n_colors,
                "index": "facelet*n_colors + color",
            }
        }
