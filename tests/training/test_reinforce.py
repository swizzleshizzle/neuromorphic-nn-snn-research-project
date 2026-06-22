import math

import torch

from neuromorphic.training.reinforce import discounted_returns, ema


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


from neuromorphic.brain import Brain
from neuromorphic.training.reinforce import action_distribution, policy_logits


def test_policy_logits_shape_and_grad():
    brain = Brain(grid_n=5, seed=0)
    out = brain.step([0, 0, 4, 4], recall=False, generator=torch.Generator().manual_seed(0))
    logits = policy_logits(out)
    assert logits.shape == (4,)
    assert logits.requires_grad


def test_action_distribution_is_a_valid_policy():
    brain = Brain(grid_n=5, seed=0)
    dist, logits = action_distribution(brain, [0, 0, 4, 4], generator=torch.Generator().manual_seed(0))
    assert logits.shape == (4,)
    probs = dist.probs
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert (probs >= 0).all()


def test_log_prob_gradient_reaches_policy_but_not_memory():
    brain = Brain(grid_n=5, seed=0)
    dist, _ = action_distribution(brain, [0, 0, 4, 4], generator=torch.Generator().manual_seed(0))
    action = dist.sample()
    dist.log_prob(action).backward()
    # policy path received gradient
    assert brain.sensory.fc1.weight.grad is not None
    assert brain.motor.fc_in.weight.grad is not None
    # memory path bypassed (recall=False) → no gradient
    assert brain.hippo.fc_in.weight.grad is None
    assert brain.hippo.fc_out.weight.grad is None
