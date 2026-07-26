"""Environments — Gymnasium-wrapped tasks for the brain to act in.

``GridWorldEnv`` wraps the Phase-0 5x5 grid world (``experiments/007_...``) as a
proper ``gymnasium.Env`` so the five-region brain (``neuromorphic.brain.Brain``)
can drive it through the standard reset/step API.

``CubeEnv`` wraps the 2x2 Pocket Cube (Phase 3) the same way, over a 6-move
quarter-turn action space with sparse reward and an optional distance provider.
"""

from neuromorphic.envs.cube import CubeEnv
from neuromorphic.envs.gridworld import GridWorldEnv

__all__ = ["CubeEnv", "GridWorldEnv"]
