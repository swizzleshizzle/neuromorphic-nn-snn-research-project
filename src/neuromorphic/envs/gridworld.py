"""``GridWorldEnv`` — the Phase-0 grid world as a proper Gymnasium environment.

Phase 0 (Module L4, `experiments/007_week4_2_qlearning_grid_agent/`) used a bare
`GridWorld` class with `reset()` / `step(action)`. This wraps the same dynamics in
the standard `gymnasium.Env` API (5-tuple `step`, `(obs, info)` `reset`, declared
`observation_space` / `action_space`) so the five-region brain can drive it.

Dynamics (unchanged from Phase 0):
- 5x5 grid, agent starts at (0, 0), goal at (4, 4).
- 4 discrete actions: 0=up, 1=right, 2=down, 3=left (Phase-0 ordering).
- Bumping a wall keeps the agent in place.
- Reward: `step_penalty` (-1) per move, `goal_reward` (+10) on reaching the goal.

Observation is `(agent_x, agent_y, goal_x, goal_y)` — exactly the 4-int layout that
`neuromorphic.regions.encode_gridworld` rate-encodes into Poisson spikes, so the env
plugs straight into the Sensory Cortex with no glue.
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def manhattan(a, b) -> int:
    """L1 distance between two (x, y) cells."""
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


class GridWorldEnv(gym.Env):
    """A minimal deterministic grid world (Gymnasium API).

    Args:
        size: grid side length ``N`` (grid is ``N x N``).
        start: agent start cell ``(x, y)``.
        goal: goal cell ``(x, y)``.
        step_penalty: reward per non-goal step (negative to encourage short paths).
        goal_reward: reward on reaching the goal.
        max_steps: episode truncates after this many steps.
    """

    metadata = {"render_modes": []}

    # Action deltas in (x, y): 0=up, 1=right, 2=down, 3=left (Phase-0 ordering).
    _DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

    def __init__(
        self,
        size: int = 5,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (4, 4),
        step_penalty: float = -1.0,
        goal_reward: float = 10.0,
        max_steps: int = 100,
        goals=None,
        goal_seed: int | None = None,
        reward_shaping: bool = False,
        shaping_gamma: float = 1.0,
    ):
        super().__init__()
        self.size = size
        self.start = start
        self.goal = goal
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        self._goals = list(goals) if goals is not None else None
        self._goal_rng = random.Random(goal_seed)
        self.reward_shaping = reward_shaping
        self.shaping_gamma = shaping_gamma
        self._prev_potential = 0.0

        self.action_space = spaces.Discrete(4)
        # obs = (agent_x, agent_y, goal_x, goal_y), each in [0, size).
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(4,), dtype=np.int64
        )

        self._agent = np.array(start, dtype=np.int64)
        self._steps = 0

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        return np.array(
            [self._agent[0], self._agent[1], self.goal[0], self.goal[1]],
            dtype=np.int64,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._goals is not None:
            self.goal = self._goal_rng.choice(self._goals)
        self._agent = np.array(self.start, dtype=np.int64)
        self._steps = 0
        self._prev_potential = -manhattan(self._agent, self.goal)
        return self._obs(), {}

    def step(self, action: int):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"invalid action {action!r}; expected 0..3")

        dx, dy = self._DELTAS[int(action)]
        x = int(np.clip(self._agent[0] + dx, 0, self.size - 1))
        y = int(np.clip(self._agent[1] + dy, 0, self.size - 1))
        self._agent = np.array([x, y], dtype=np.int64)
        self._steps += 1

        terminated = bool(x == self.goal[0] and y == self.goal[1])
        truncated = bool(self._steps >= self.max_steps)
        reward = self.goal_reward if terminated else self.step_penalty
        if self.reward_shaping:
            pot = -manhattan(self._agent, self.goal)
            reward += self.shaping_gamma * pot - self._prev_potential
            self._prev_potential = pot
        return self._obs(), float(reward), terminated, truncated, {}

    def render(self):
        return None
