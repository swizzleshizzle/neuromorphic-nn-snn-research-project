"""EXP-074 seam: a critic that reads states directly. Each docstring names the bug it catches."""

from __future__ import annotations

import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.training import lookahead as la


class _NoStep:
    def step(self, *a, **k):
        raise AssertionError("a state-reading critic must not go through agent.step")


class _StateCritic:
    reads_states = True

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def __call__(self, states, generator):
        self.calls.append((list(states), generator))
        return torch.tensor(self.scores[: len(states)], dtype=torch.float32)


def test_state_critic_receives_the_states_and_generator_and_skips_agent_step():
    """Catches the seam routing through agent.step (a wide judge cannot be read from the
    concept) or passing a different generator than the search's imagination stream."""
    states = [apply_move(SOLVED, a) for a in range(N_ACTIONS)]
    critic = _StateCritic(list(range(N_ACTIONS)))
    g = torch.Generator().manual_seed(0)
    out = la.imagined_values(_NoStep(), critic, states, generator=g)
    assert critic.calls[0][0] == states
    assert critic.calls[0][1] is g
    assert out.tolist() == [float(i) for i in range(N_ACTIONS)]


def test_choose_move_c_with_a_state_critic_takes_the_best_leaf():
    """Catches the seam returning scores in a different order than levels[k]."""
    root = apply_move(apply_move(SOLVED, 0), 2)
    leaves = la.tree_levels(root, 1, N_ACTIONS)[1]
    scores = [0.0] * N_ACTIONS
    scores[4] = 9.0
    action, fired = la.choose_move("C", root, torch.zeros(N_ACTIONS), 1, N_ACTIONS,
                                   agent=_NoStep(), critic=_StateCritic(scores),
                                   imag_generator=torch.Generator().manual_seed(0))
    assert fired is False   # root is two moves out, so a 1-move goal check cannot fire
    assert action == 4
    assert len(leaves) == N_ACTIONS
