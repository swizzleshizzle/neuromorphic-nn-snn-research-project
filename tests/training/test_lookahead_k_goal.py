"""EXP-076 seam: a goal check deeper than the judge tree. Each docstring names the bug it catches."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import lookahead as la
from neuromorphic.training.cube_baseline import CubeConfig, make_agent, shell_states


def _agent_and_head(seed=0):
    agent = make_agent(CubeConfig(arm="regionalized", seed=seed))
    torch.manual_seed(seed)
    head = nn.Linear(agent.content, N_ACTIONS)
    return agent, head


def _shell(depth, n):
    return shell_states(ExactBFSDistance(max_depth=depth), depth)[:n]


class _HashCritic:
    """Deterministic, state-reading scores that ignore the generator: a judge stand-in."""

    reads_states = True

    def __call__(self, states, generator):
        return torch.tensor([float(hash(s) % 1009) for s in states])


class _DistanceCritic:
    """Steers toward solved: the negative exact distance, so the walk closes in and a deeper goal
    check is guaranteed to have something to fire on."""

    reads_states = True

    def __init__(self):
        self.bfs = ExactBFSDistance(max_depth=8)

    def __call__(self, states, generator):
        out = []
        for s in states:
            d = self.bfs.distance(s)
            out.append(-float(99 if d is None else d))
        return torch.tensor(out)


def _c(agent, head, states, depth, k, k_goal=None, trace=None, critic=None):
    return la.evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=k, k_goal=k_goal,
                                 generator=torch.Generator().manual_seed(3), rng_seed=3,
                                 imag_seed=3, critic=critic or _HashCritic(), no_revisit=True, trace=trace)


def test_k_goal_below_k_is_refused():
    """Catches a goal check shallower than the judge tree being accepted: the judge would score
    leaves past a solve the goal check cannot see, a regime the spec forbids."""
    s = apply_move(SOLVED, 0)
    with pytest.raises(ValueError, match="k_goal"):
        la.choose_move("C", s, torch.zeros(N_ACTIONS), 2, N_ACTIONS, critic=_HashCritic(),
                       k_goal=1)
    agent, head = _agent_and_head()
    with pytest.raises(ValueError, match="k_goal"):
        la.evaluate_lookahead(agent, head, [s], depth=1, mode="C", k=2, k_goal=1,
                              generator=torch.Generator().manual_seed(0),
                              critic=_HashCritic())


def test_choose_move_fires_at_k_goal_not_k():
    """Catches `solving_prefix` still being called with k: a 2-move solve must fire with k 1 and
    k_goal 2, must NOT fire with k 1 alone, and must report its prefix length."""
    s = apply_move(apply_move(SOLVED, 0), 2)  # two different faces: no 1-move solve
    info = {}
    a, fired = la.choose_move("C", s, torch.zeros(N_ACTIONS), 1, N_ACTIONS,
                              critic=_HashCritic(), info=info, k_goal=2)
    assert fired is True and info["fire_len"] == 2
    assert la.solving_prefix(s, 2, N_ACTIONS)[0] == a
    info = {}
    _, fired = la.choose_move("C", s, torch.zeros(N_ACTIONS), 1, N_ACTIONS,
                              critic=_HashCritic(), info=info)
    assert fired is False and "fire_len" not in info


def test_default_k_goal_is_byte_identical_to_explicit_k():
    """THE GATE 0(b) CODE PATH. Catches a default that is not k (e.g. 0 or k + 1): every field
    of a C rollout with k_goal omitted equals the same rollout with k_goal = k."""
    agent, head = _agent_and_head()
    states = _shell(3, 4)
    a = _c(agent, head, states, 3, 1)
    b = _c(agent, head, states, 3, 1, k_goal=1)
    assert a == b
    assert a["k_goal"] == 1


def test_first_fire_len_counts_each_episode_once_at_its_first_fire():
    """Catches counting every fired move instead of each episode's first fire, and a histogram
    keyed by step instead of prefix length. E with k 2 on depth-2 states fires at step 1 with a
    2-move prefix, then again with a 1-move prefix: only the first counts."""
    agent, head = _agent_and_head()
    states = _shell(2, 4)
    r = la.evaluate_lookahead(agent, head, states, depth=2, mode="E", k=2,
                              generator=torch.Generator().manual_seed(1), rng_seed=1)
    assert r["first_fire_len"] == {"2": 4}
    assert r["solved"] == 4 and r["k_goal"] == 2


def test_deeper_goal_check_plays_the_same_moves_until_it_first_fires_and_never_solves_less():
    """THE GATE D INVARIANT. Catches the goal-check arm consuming the evaluation or imagination
    stream differently from its k_goal = k twin: before its first fire it must play identical
    moves on every state, and it must solve at least as many. The critic steers toward solved so
    the deeper check is guaranteed to have something to fire on."""
    agent, head = _agent_and_head()
    states = _shell(4, 8)
    ta, tb = [], []
    crit = _DistanceCritic()
    a = _c(agent, head, states, 4, 1, trace=ta, critic=crit)
    b = _c(agent, head, states, 4, 1, k_goal=2, trace=tb, critic=crit)
    compared = 0
    for a_steps, b_steps in zip(ta, tb):
        for (aa, _, _), (ba, fired, _) in zip(a_steps, b_steps):
            if fired:
                break
            assert aa == ba
            compared += 1
    assert compared >= 8  # the test must actually compare something
    assert b["first_fire_len"].get("2", 0) >= 1  # and the deeper check must actually fire
    assert b["solved"] >= a["solved"]
