"""Environments — Gymnasium-wrapped tasks for the brain to act in.

``GridWorldEnv`` wraps the Phase-0 5x5 grid world (``experiments/007_...``) as a
proper ``gymnasium.Env`` so the five-region brain (``neuromorphic.brain.Brain``)
can drive it through the standard reset/step API.
"""

from neuromorphic.envs.gridworld import GridWorldEnv

__all__ = ["GridWorldEnv"]
