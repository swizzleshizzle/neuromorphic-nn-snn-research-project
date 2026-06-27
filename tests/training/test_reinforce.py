import math

import torch

from neuromorphic.brain import Brain
from neuromorphic.envs import GridWorldEnv
from neuromorphic.training.reinforce import (
    action_distribution,
    discounted_returns,
    ema,
    greedy_action,
    make_policy_head,
    policy_parameters,
    train_episode,
)


def test_discounted_returns_to_go():
    r = discounted_returns([0.0, 0.0, 1.0], gamma=0.9)
    assert len(r) == 3
    assert math.isclose(r[2], 1.0, rel_tol=1e-6)
    assert math.isclose(r[1], 0.9, rel_tol=1e-6)
    assert math.isclose(r[0], 0.81, rel_tol=1e-6)


def test_discounted_returns_undiscounted_sum():
    r = discounted_returns([1.0, 1.0, 1.0], gamma=1.0)
    assert r == [3.0, 2.0, 1.0]


def test_discounted_returns_empty():
    assert discounted_returns([], gamma=0.99) == []


def test_ema_blends():
    assert math.isclose(ema(0.0, 10.0, 0.1), 1.0, rel_tol=1e-6)
    assert math.isclose(ema(10.0, 10.0, 0.5), 10.0, rel_tol=1e-6)


def test_policy_head_shapes():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    assert head.in_features == brain.content
    assert head.out_features == brain.n_actions


def test_action_distribution_is_a_valid_policy():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    dist, logits = action_distribution(
        brain, head, [0, 0, 4, 4], generator=torch.Generator().manual_seed(0)
    )
    assert logits.shape == (4,)
    assert logits.requires_grad
    assert torch.allclose(dist.probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert (dist.probs >= 0).all()


def test_policy_logits_are_state_dependent():
    """Root-cause #2 fix: different observations must produce different logits.

    The old motor/PFC readout was a degenerate structural favourite — near-identical
    across states. The head on the rich sensory concept must distinguish states.
    """
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    g = torch.Generator().manual_seed(0)
    _, la = action_distribution(brain, head, [0, 0, 4, 4], generator=g)
    _, lb = action_distribution(brain, head, [4, 4, 0, 0], generator=g)
    assert not torch.allclose(la, lb, atol=1e-3)


def test_greedy_action_is_valid():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    a = greedy_action(brain, head, [0, 0, 4, 4], generator=torch.Generator().manual_seed(0))
    assert isinstance(a, int)
    assert 0 <= a < 4


def test_train_episode_updates_head_but_not_the_frozen_brain():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    env = GridWorldEnv()
    opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
    head_before = head.weight.detach().clone()
    sensory_before = brain.sensory.fc1.weight.detach().clone()

    stats = train_episode(
        brain, head, env, opt, gamma=0.99, baseline=0.0,
        generator=torch.Generator().manual_seed(0), max_steps=10,
    )

    assert set(stats) == {"steps", "total_reward", "mean_return", "loss", "reached_goal", "mean_entropy"}
    assert stats["steps"] >= 1
    assert isinstance(stats["reached_goal"], bool)
    # the head learned; the brain stayed frozen (v1)
    assert not torch.equal(head_before, head.weight.detach())
    assert torch.equal(sensory_before, brain.sensory.fc1.weight.detach())


def test_train_episode_reports_mean_entropy():
    import torch
    from neuromorphic.brain import Brain
    from neuromorphic.envs import GridWorldEnv
    from neuromorphic.training.reinforce import make_policy_head, train_episode

    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    env = GridWorldEnv(max_steps=10)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    stats = train_episode(brain, head, env, opt, generator=gen, max_steps=10)
    assert "mean_entropy" in stats
    assert stats["mean_entropy"] >= 0.0
    assert stats["mean_entropy"] == stats["mean_entropy"]  # not NaN
