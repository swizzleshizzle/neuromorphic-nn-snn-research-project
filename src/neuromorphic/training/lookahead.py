"""Look-ahead move selection over the true simulator (EXP-070).

The PROCEDURE lives here: enumerate move sequences with the pure simulator, take a solve if one is
within reach, otherwise score the sequences and play the first move of the best. It is general:
nothing here knows the cube has 6 moves, and nothing here may read the BFS distance table, which
is an instrument and never an input (spec section 2.3). A test enforces the second rule at the
source level.
"""

from __future__ import annotations

import itertools

import torch

from neuromorphic.envs.cube import apply_move, is_solved


def solving_prefix(state, k, n_actions, *, apply_fn=apply_move, goal_fn=is_solved):
    """The shortest sequence of length 1..k that reaches the goal, or None.

    Lengths are tried in increasing order and sequences in `itertools.product` order, so the
    result is deterministic and ties go to the lowest action indices.
    """
    for length in range(1, k + 1):
        for seq in itertools.product(range(n_actions), repeat=length):
            s = state
            for a in seq:
                s = apply_fn(s, a)
            if goal_fn(s):
                return seq
    return None


def tree_levels(state, k, n_actions, *, apply_fn=apply_move):
    """levels[l] = the states every length-l sequence reaches, in product order."""
    levels = [[state]]
    for _ in range(k):
        levels.append([apply_fn(s, a) for s in levels[-1] for a in range(n_actions)])
    return levels


def sequence_scores(level_logp, k, n_actions):
    """Summed log-probability of every length-k sequence, indexed in product order.

    For sequence index j, the move at level l is (j // n**(k-l-1)) % n and the state it is taken
    from is level l's index j // n**(k-l).
    """
    j = torch.arange(n_actions ** k)
    scores = torch.zeros(n_actions ** k, dtype=level_logp[0].dtype)
    for l in range(k):
        parent = j // n_actions ** (k - l)
        move = (j // n_actions ** (k - l - 1)) % n_actions
        scores = scores + level_logp[l][parent, move]
    return scores
