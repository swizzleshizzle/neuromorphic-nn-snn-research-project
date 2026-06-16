"""End-to-end smoke test for the full five-region assembly (Week-11 S1).

Confirms the brain wires together and runs against the Gymnasium grid world:
signal flows through all five regions, an action comes out, the reward hook reaches
the dopamine bus, and an episode runs to termination/truncation. The brain is
untrained, so we assert *flow and shape*, not task success.
"""

from __future__ import annotations

import numpy as np
import torch

from neuromorphic.brain import Brain
from neuromorphic.envs import GridWorldEnv

T = 32
N_ACTIONS = 4


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(0)


# ----------------------------------------------------------------------- #
# Environment — Gymnasium API compliance
# ----------------------------------------------------------------------- #
def test_env_reset_returns_obs_info():
    env = GridWorldEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,)
    assert tuple(obs.tolist()) == (0, 0, 4, 4)  # agent start, goal
    assert isinstance(info, dict)


def test_env_step_is_five_tuple_and_bounds():
    env = GridWorldEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(1)  # right
    assert obs.shape == (4,)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False and truncated is False
    assert reward == -1.0  # step penalty


def test_env_reaches_goal_reward():
    env = GridWorldEnv()
    env.reset(seed=0)
    # walk right 4 then down 4 → goal (4,4)
    rewards = []
    for a in [1, 1, 1, 1, 2, 2, 2, 2]:
        _, r, term, trunc, _ = env.step(a)
        rewards.append(r)
        if term:
            break
    assert term is True
    assert rewards[-1] == 10.0


# ----------------------------------------------------------------------- #
# Brain — single step shapes + signal flow
# ----------------------------------------------------------------------- #
def test_brain_step_produces_valid_action():
    brain = Brain(seed=0)
    out = brain.step(np.array([0, 0, 4, 4]), generator=_gen())
    assert out["action"] in range(N_ACTIONS)
    assert out["utilities"].shape == (T, 1, N_ACTIONS)
    assert out["action_spikes"].shape == (T, 1, N_ACTIONS)
    # every stage emitted spikes (signal actually flowed through the assembly)
    assert out["concept"].sum() > 0
    assert out["utilities"].sum() > 0
    assert out["action_spikes"].sum() > 0


def test_brain_recall_shifts_utilities():
    """Storing a snapshot then recalling shifts PFC utilities vs sensory-only."""
    brain = Brain(seed=0)
    obs = np.array([0, 0, 4, 4])
    brain.remember(obs, generator=_gen())
    util_on = brain.step(obs, recall=True, generator=_gen())["utilities"].sum(dim=0)
    util_off = brain.step(obs, recall=False, generator=_gen())["utilities"].sum(dim=0)
    assert (util_on != util_off).any()  # the gated memory measurably moves the output


def test_brain_records_state_for_viz():
    brain = Brain(seed=0)
    out = brain.step(np.array([0, 0, 4, 4]), record=True, generator=_gen())
    recs = out["recordings"]
    assert set(recs) == {"sensory", "hippocampus", "prefrontal", "router", "motor"}
    # each region logged at least one [T, B, N] signal
    assert recs["prefrontal"]["utility"].shape[0] == T


# ----------------------------------------------------------------------- #
# Reward hook + full episode
# ----------------------------------------------------------------------- #
def test_learn_pushes_reward_to_dopamine_bus():
    brain = Brain(seed=0)
    assert brain.learn(10.0) is True  # goal reward → above threshold → learning enabled
    assert brain.bus.dopamine == 10.0
    assert brain.learn(-1.0) is False  # step penalty → below threshold


def test_run_episode_terminates_with_valid_actions():
    env = GridWorldEnv(max_steps=50)
    brain = Brain(grid_n=env.size, seed=0)
    summary = brain.run_episode(env, generator=_gen())
    assert summary["steps"] <= 50
    assert summary["steps"] == len(summary["actions"])
    assert all(a in range(N_ACTIONS) for a in summary["actions"])
    # episode ended one way or the other (untrained: almost certainly truncation)
    assert summary["reached_goal"] in (True, False)
