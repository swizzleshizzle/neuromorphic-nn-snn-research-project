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
