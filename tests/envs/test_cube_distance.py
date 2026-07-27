# tests/envs/test_cube_distance.py
import random

import pytest

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, CubeEnv, apply_move, scramble
from neuromorphic.envs.cube_distance import ExactBFSDistance

# Published 2x2 quarter-turn BFS level counts.
PUBLISHED_LEVELS = [1, 6, 27, 120, 534, 2256, 8969]


@pytest.fixture(scope="module")
def shallow():
    """Depth-bounded table: ~0.04s to build, enough for every non-slow assertion."""
    return ExactBFSDistance(max_depth=6)


def test_solved_is_distance_zero(shallow):
    assert shallow.distance(SOLVED) == 0


def test_single_move_is_distance_one(shallow):
    for a in range(N_ACTIONS):
        assert shallow.distance(apply_move(SOLVED, a)) == 1


def test_level_counts_match_published(shallow):
    """The strongest single check on the move permutations."""
    assert shallow.level_counts() == PUBLISHED_LEVELS
    assert shallow.table_size == sum(PUBLISHED_LEVELS)


def test_beyond_the_bound_is_none(shallow):
    deep = scramble(12, random.Random(0))
    assert shallow.distance(deep) is None


def test_scramble_depth_matches_distance_for_shallow_depths(shallow):
    rng = random.Random(0)
    for depth in (1, 2):
        for _ in range(50):
            assert shallow.distance(scramble(depth, rng)) == depth


def test_exact_depth_scrambles_have_exact_distance(shallow):
    rng = random.Random(0)
    for depth in (3, 4, 5, 6):
        for _ in range(20):
            assert shallow.distance(scramble(depth, rng, provider=shallow)) == depth


def test_env_exact_depth_end_to_end(shallow):
    env = CubeEnv(scramble_depth=5, scramble_seed=0, distance_provider=shallow, exact_depth=True)
    for _ in range(10):
        _, info = env.reset()
        assert info["distance"] == 5


@pytest.mark.slow
def test_full_table_size_and_god_number():
    """Unbounded build is ~67s; the full-table facts live here alone."""
    d = ExactBFSDistance()
    assert d.table_size == 3674160
    assert d.max_distance == 14
    assert d.level_counts()[:7] == PUBLISHED_LEVELS
    assert all(c > 0 for c in d.level_counts())


def test_states_at_distance_matches_published_shells(shallow):
    for depth, expected in enumerate(PUBLISHED_LEVELS):
        states = shallow.states_at_distance(depth)
        assert len(states) == expected
        assert all(shallow.distance(s) == depth for s in states)


def test_states_at_distance_is_sorted_and_beyond_the_bound_is_empty(shallow):
    states = shallow.states_at_distance(3)
    assert states == sorted(states)
    assert shallow.states_at_distance(12) == []
