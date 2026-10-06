"""Tests for the look-ahead procedure (EXP-070). Each docstring names the bug it catches."""

from __future__ import annotations

import inspect
import itertools

import pytest
import torch

from neuromorphic.envs.cube import SOLVED, apply_move, inverse_action, is_solved, N_ACTIONS
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import lookahead as la
from neuromorphic.training.cube_baseline import shell_states


def _state_from(moves):
    s = SOLVED
    for a in moves:
        s = apply_move(s, a)
    return s


def test_solving_prefix_finds_a_two_move_solve_at_k2_and_not_at_k1():
    """Catches an off-by-one in tree depth: a k=1 search must NOT see a 2-move solve."""
    s = _state_from([0, 2])  # two different faces, so no 1-move solve exists
    assert la.solving_prefix(s, 1, N_ACTIONS) is None
    seq = la.solving_prefix(s, 2, N_ACTIONS)
    assert seq is not None and len(seq) == 2
    assert is_solved(_state_from_on(s, seq))


def _state_from_on(state, moves):
    for a in moves:
        state = apply_move(state, a)
    return state


def test_solving_prefix_finds_exactly_depth_k_from_a_true_depth_3_state():
    """Catches a search that stops one level short (k-1) or reads one level too far (k+1)."""
    s = shell_states(ExactBFSDistance(max_depth=3), 3)[0]
    assert la.solving_prefix(s, 2, N_ACTIONS) is None
    seq = la.solving_prefix(s, 3, N_ACTIONS)
    assert seq is not None and len(seq) == 3 and is_solved(_state_from_on(s, seq))


def test_solving_prefix_prefers_the_shortest_solve():
    """Catches returning a length-k solve when a shorter one exists (e.g. U U U from U')."""
    s = _state_from([0])
    assert la.solving_prefix(s, 3, N_ACTIONS) == (inverse_action(0),)


def test_tree_levels_take_width_from_the_argument_not_a_literal():
    """Catches a hardcoded 6: a 12-action stub must give 1, 12, 144 states in product order."""
    def stub_apply(s, a):
        return s + (a,)
    levels = la.tree_levels((), 2, 12, apply_fn=stub_apply)
    assert [len(l) for l in levels] == [1, 12, 144]
    assert levels[2] == [tuple(p) for p in itertools.product(range(12), repeat=2)]


def test_solving_prefix_takes_width_from_the_argument():
    """Catches a hardcoded 6 in the goal test: action 11 only exists in a 12-action space."""
    def stub_apply(s, a):
        return s + (a,)
    seq = la.solving_prefix((), 2, 12, apply_fn=stub_apply, goal_fn=lambda s: s == (11,))
    assert seq == (11,)


def test_sequence_scores_sum_log_probs_along_each_path():
    """Catches wrong parent/child index arithmetic, which would score a sequence with a
    sibling's log-probabilities. Values are distinct powers of ten so any mix-up shows."""
    lp0 = torch.tensor([[1.0, 2.0]])
    lp1 = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    lp2 = torch.tensor([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0], [700.0, 800.0]])
    got = la.sequence_scores([lp0, lp1, lp2], 3, 2)
    want = []
    for a, b, c in itertools.product(range(2), repeat=3):
        want.append(lp0[0, a] + lp1[a, b] + lp2[a * 2 + b, c])
    assert torch.equal(got, torch.stack(want))


def test_lookahead_module_never_imports_the_bfs_provider():
    """Catches an oracle leak at the source level: the procedure must not reach distance."""
    src = inspect.getsource(la)
    assert "cube_distance" not in src
    assert "ExactBFSDistance" not in src
