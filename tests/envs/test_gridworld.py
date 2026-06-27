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


def test_shaping_off_is_unchanged():
    env = GridWorldEnv(goal=(0, 2))  # reward_shaping defaults False
    env.reset()
    _, r, _, _, _ = env.step(2)  # action 2 = down -> (0,1)
    assert r == -1.0


def test_shaping_rewards_progress():
    env = GridWorldEnv(goal=(0, 2), reward_shaping=True)
    env.reset()  # agent (0,0), manhattan 2, Phi=-2
    # down -> (0,1), manhattan 1, Phi=-1: reward = -1 + (-1 - -2) = 0
    _, r_closer, _, _, _ = env.step(2)
    assert r_closer == 0.0
    # up -> (0,0), manhattan 2, Phi=-2: reward = -1 + (-2 - -1) = -2
    _, r_farther, _, _, _ = env.step(0)
    assert r_farther == -2.0
    assert r_closer > r_farther


def test_shaping_telescopes_over_path():
    env = GridWorldEnv(goal=(0, 3), reward_shaping=True)
    obs, _ = env.reset()
    start_phi = -manhattan((obs[0], obs[1]), env.goal)
    total = 0.0
    for action in (2, 2):  # two steps down, no goal reached
        obs, r, term, trunc, _ = env.step(action)
        total += r
    end_phi = -manhattan((obs[0], obs[1]), env.goal)
    base = -1.0 * 2  # two non-goal steps
    # shaped total minus base equals telescoped potential difference
    assert abs((total - base) - (end_phi - start_phi)) < 1e-9
