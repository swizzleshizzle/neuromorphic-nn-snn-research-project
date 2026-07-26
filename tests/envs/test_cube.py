import random

import numpy as np
import pytest

from neuromorphic.envs.cube import (
    FIXED_FACELETS,
    MOVE_LABELS,
    MOVES,
    N_ACTIONS,
    SOLVED,
    CubeEnv,
    apply_move,
    inverse_action,
    is_solved,
    scramble,
)


def test_solved_state_shape_and_uniform_faces():
    assert len(SOLVED) == 24
    for face in range(6):
        block = SOLVED[face * 4: face * 4 + 4]
        assert len(set(block)) == 1
    assert is_solved(SOLVED)


def test_six_moves_labelled():
    assert len(MOVES) == 6 == len(MOVE_LABELS) == N_ACTIONS
    assert MOVE_LABELS == ["U", "U'", "R", "R'", "F", "F'"]


def test_every_move_is_a_bijection():
    for P in MOVES:
        assert sorted(P) == list(range(24))


def test_each_move_has_order_4():
    for i in range(N_ACTIONS):
        f = SOLVED
        for _ in range(4):
            f = apply_move(f, i)
        assert f == SOLVED
        assert not is_solved(apply_move(SOLVED, i))


def test_cw_then_ccw_is_identity():
    for k in range(N_ACTIONS // 2):
        assert apply_move(apply_move(SOLVED, 2 * k), 2 * k + 1) == SOLVED
        assert inverse_action(2 * k) == 2 * k + 1
        assert inverse_action(2 * k + 1) == 2 * k


def test_no_two_moves_are_physically_identical():
    """The anti-redundancy property. A 12-move set fails this: U == D', R == L', F == B'."""
    for a in range(N_ACTIONS):
        for b in range(N_ACTIONS):
            if a == b:
                continue
            undone = apply_move(apply_move(SOLVED, a), inverse_action(b))
            assert not is_solved(undone), f"{MOVE_LABELS[a]} is redundant with {MOVE_LABELS[b]}"


def test_dlb_corner_is_held_still():
    assert FIXED_FACELETS == (12, 16, 21)
    for P in MOVES:
        for i in FIXED_FACELETS:
            assert P[i] == i


def test_scramble_then_reverse_returns_to_solved():
    rng = random.Random(0)
    f = SOLVED
    actions = []
    for _ in range(8):
        a = rng.randrange(N_ACTIONS)
        f = apply_move(f, a)
        actions.append(a)
    for a in reversed(actions):
        f = apply_move(f, inverse_action(a))
    assert f == SOLVED


def test_scramble_is_reproducible_and_never_solved():
    assert scramble(1, random.Random(3)) == scramble(1, random.Random(3))
    rng = random.Random(0)
    for depth in range(1, 9):
        for _ in range(200):
            s = scramble(depth, rng)
            assert len(s) == 24
            assert not is_solved(s)


def test_scramble_depth_zero_is_solved():
    assert scramble(0, random.Random(0)) == SOLVED


def test_reset_returns_scrambled_obs():
    env = CubeEnv(scramble_depth=1, scramble_seed=0)
    obs, info = env.reset()
    assert obs.shape == (24,) and obs.dtype == np.int64
    assert env.observation_space.contains(obs)
    assert info["scramble_depth"] == 1
    assert info["solved"] is False
    assert info["move"] is None and info["move_label"] is None


def test_depth_1_has_exactly_one_solving_action():
    """With the redundant 12-move set this would be 2, and chance would be 2/12 not 1/6."""
    for seed in range(25):
        env = CubeEnv(scramble_depth=1, scramble_seed=seed)
        env.reset()
        state = env._state
        solving = [a for a in range(N_ACTIONS) if is_solved(apply_move(state, a))]
        assert len(solving) == 1


def test_solving_move_terminates_with_reward():
    env = CubeEnv(scramble_depth=1, scramble_seed=0)
    env.reset()
    solving = [a for a in range(N_ACTIONS) if is_solved(apply_move(env._state, a))]
    obs, reward, terminated, truncated, info = env.step(solving[0])
    assert terminated is True and truncated is False
    assert info["solved"] is True
    assert reward == env.solve_reward
    assert info["move"] == solving[0]
    assert info["move_label"] == MOVE_LABELS[solving[0]]


def test_step_penalty_and_truncation_are_exclusive():
    env = CubeEnv(scramble_depth=4, scramble_seed=1, max_steps=3)
    env.reset()
    seen_trunc = False
    for _ in range(3):
        _, r, term, trunc, _ = env.step(0)
        assert not (term and trunc)
        if term:
            break
        assert r == env.step_penalty
        seen_trunc = trunc
    assert seen_trunc or term


def test_distance_is_none_without_provider():
    env = CubeEnv(scramble_depth=1, scramble_seed=0)
    _, info = env.reset()
    assert info["distance"] is None
    _, _, _, _, info = env.step(0)
    assert info["distance"] is None


def test_action_space_and_reproducibility():
    env = CubeEnv(scramble_depth=3, scramble_seed=7)
    assert env.action_space.n == N_ACTIONS
    o1, _ = env.reset()
    o2, _ = CubeEnv(scramble_depth=3, scramble_seed=7).reset()
    assert np.array_equal(o1, o2)


def test_invalid_action_raises():
    env = CubeEnv(scramble_seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(N_ACTIONS)


def test_optional_flags_require_a_provider():
    with pytest.raises(ValueError):
        CubeEnv(reward_shaping=True)
    with pytest.raises(ValueError):
        CubeEnv(exact_depth=True)
