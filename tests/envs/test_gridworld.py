from neuromorphic.envs.gridworld import GridWorldEnv, manhattan


def test_manhattan():
    assert manhattan((0, 0), (2, 3)) == 5
    assert manhattan((4, 4), (4, 4)) == 0


def test_fixed_goal_backward_compat():
    env = GridWorldEnv()
    obs, _ = env.reset()
    assert tuple(obs) == (0, 0, 4, 4)
    # action 1 = right; reward is the plain step penalty, goal unchanged
    obs, r, term, trunc, _ = env.step(1)
    assert tuple(obs) == (1, 0, 4, 4)
    assert r == -1.0 and term is False
    assert tuple(env.goal) == (4, 4)


def test_random_goal_sampled_from_set():
    goals = [(1, 2), (3, 4), (0, 1)]
    env = GridWorldEnv(goals=goals, goal_seed=0)
    seen = set()
    for _ in range(30):
        env.reset()
        g = tuple(env.goal)
        assert g in goals
        seen.add(g)
    assert len(seen) > 1  # actually samples more than one over 30 resets


def test_random_goal_deterministic_by_seed():
    goals = [(1, 2), (3, 4), (0, 1), (2, 0)]
    a = GridWorldEnv(goals=goals, goal_seed=7)
    b = GridWorldEnv(goals=goals, goal_seed=7)
    seq_a = [tuple((a.reset(), a.goal)[1]) for _ in range(10)]
    seq_b = [tuple((b.reset(), b.goal)[1]) for _ in range(10)]
    assert seq_a == seq_b
