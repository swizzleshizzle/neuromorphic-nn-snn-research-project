import math

import pytest
import torch
import torch.nn as nn

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

    # EXP-053 adds "gate_open" unconditionally (True when no encoder_optimizer is passed,
    # as here); the critic-only keys stay absent because no critic was passed.
    assert set(stats) == {
        "steps", "total_reward", "mean_return", "loss", "reached_goal", "mean_entropy",
        "gate_open",
    }
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


def test_make_policy_head_linear_is_unchanged_default():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    assert isinstance(head, nn.Linear)
    assert head.in_features == brain.content
    assert head.out_features == brain.n_actions


def test_make_policy_head_mlp_shape_and_forward():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain, head_type="mlp", hidden=128)
    assert isinstance(head, nn.Sequential)
    x = torch.zeros(brain.content)
    out = head(x)
    assert out.shape == (brain.n_actions,)
    assert len(list(head.parameters())) > 0


def test_make_policy_head_rejects_unknown_type():
    brain = Brain(grid_n=5, seed=0)
    with pytest.raises(ValueError):
        make_policy_head(brain, head_type="transformer")


def test_train_episode_updates_mlp_head_but_not_the_frozen_brain():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain, head_type="mlp", hidden=128)
    env = GridWorldEnv()
    opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
    params_before = [p.detach().clone() for p in head.parameters()]
    sensory_before = brain.sensory.fc1.weight.detach().clone()

    train_episode(
        brain, head, env, opt, gamma=0.99, baseline=0.0,
        generator=torch.Generator().manual_seed(0), max_steps=10,
    )

    changed = any(
        not torch.equal(b, a.detach())
        for b, a in zip(params_before, head.parameters())
    )
    assert changed
    assert torch.equal(sensory_before, brain.sensory.fc1.weight.detach())


def test_train_episode_entropy_beta_default_matches_no_bonus():
    """entropy_beta=0.0 must reproduce the exact loss (EXP-023/024 byte-identity)."""
    def run(beta):
        brain = Brain(grid_n=5, seed=0)
        head = make_policy_head(brain)
        env = GridWorldEnv(max_steps=10)
        opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
        return train_episode(
            brain, head, env, opt, gamma=0.99, baseline=0.0,
            generator=torch.Generator().manual_seed(0), max_steps=10,
            entropy_beta=beta,
        )["loss"]
    assert run(0.0) == run(0.0)  # deterministic
    # explicit default path == explicitly passing 0.0
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    env = GridWorldEnv(max_steps=10)
    opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
    loss_default = train_episode(
        brain, head, env, opt, generator=torch.Generator().manual_seed(0), max_steps=10,
    )["loss"]
    assert loss_default == run(0.0)


def test_train_episode_entropy_bonus_lowers_loss():
    """With entropy > 0, a positive beta subtracts a positive term -> lower loss."""
    def loss_for(beta):
        brain = Brain(grid_n=5, seed=0)
        head = make_policy_head(brain)
        env = GridWorldEnv(max_steps=10)
        opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
        return train_episode(
            brain, head, env, opt, gamma=0.99, baseline=0.0,
            generator=torch.Generator().manual_seed(0), max_steps=10,
            entropy_beta=beta,
        )
    base = loss_for(0.0)
    bonus = loss_for(0.5)
    assert base["mean_entropy"] > 0.0
    assert bonus["loss"] < base["loss"]


def test_train_episode_with_bonus_updates_head():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain, head_type="mlp", hidden=128)
    env = GridWorldEnv(max_steps=10)
    opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
    before = [p.detach().clone() for p in head.parameters()]
    train_episode(
        brain, head, env, opt, generator=torch.Generator().manual_seed(0),
        max_steps=10, entropy_beta=0.01,
    )
    assert any(not torch.equal(b, a.detach()) for b, a in zip(before, head.parameters()))


def _loss_with(normalize):
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    env = GridWorldEnv(max_steps=10)
    opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
    return train_episode(
        brain, head, env, opt, gamma=0.99, baseline=0.0,
        generator=torch.Generator().manual_seed(0), max_steps=10,
        normalize_advantages=normalize,
    )


def test_normalize_advantages_default_matches_no_normalization():
    """Default (False) must reproduce the exact loss (EXP-023/024/025-b0 byte-identity)."""
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    env = GridWorldEnv(max_steps=10)
    opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
    loss_default = train_episode(
        brain, head, env, opt, generator=torch.Generator().manual_seed(0), max_steps=10,
    )["loss"]
    assert loss_default == _loss_with(False)["loss"]


def test_normalize_advantages_changes_loss_and_stays_finite():
    import math
    base = _loss_with(False)["loss"]
    norm = _loss_with(True)["loss"]
    assert math.isfinite(norm)
    assert norm != base


def test_normalize_advantages_updates_head_no_nan():
    import math
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain, head_type="mlp", hidden=128)
    env = GridWorldEnv(max_steps=10)
    opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
    before = [p.detach().clone() for p in head.parameters()]
    stats = train_episode(
        brain, head, env, opt, generator=torch.Generator().manual_seed(0),
        max_steps=10, entropy_beta=0.05, normalize_advantages=True,
    )
    assert math.isfinite(stats["loss"])
    assert all(torch.isfinite(p).all() for p in head.parameters())
    assert any(not torch.equal(b, a.detach()) for b, a in zip(before, head.parameters()))
