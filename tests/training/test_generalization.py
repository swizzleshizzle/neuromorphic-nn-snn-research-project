from neuromorphic.training.generalization import EvalResult, optimality, split_goals


def test_split_goals_disjoint_sized_and_excludes_start():
    train, held = split_goals(size=5, start=(0, 0), n_heldout=6, seed=0)
    assert len(train) == 18 and len(held) == 6
    s_train, s_held = set(train), set(held)
    assert s_train.isdisjoint(s_held)
    assert (0, 0) not in s_train and (0, 0) not in s_held
    all_candidates = {(x, y) for x in range(5) for y in range(5)} - {(0, 0)}
    assert s_train | s_held == all_candidates


def test_split_goals_deterministic():
    a = split_goals(5, (0, 0), 6, seed=3)
    b = split_goals(5, (0, 0), 6, seed=3)
    c = split_goals(5, (0, 0), 6, seed=4)
    assert a == b
    assert a != c


def test_optimality():
    assert optimality((0, 0), (2, 0), 2) == 1.0
    assert optimality((0, 0), (2, 0), 4) == 0.5
    assert optimality((0, 0), (2, 0), 0) == 0.0


def test_eval_result_fields():
    r = EvalResult(success_rate=0.5, mean_steps=8.0, optimality=0.9, n=4)
    assert r.success_rate == 0.5 and r.n == 4


import torch

from neuromorphic.brain import Brain
from neuromorphic.training.generalization import evaluate
from neuromorphic.training.reinforce import make_policy_head


def test_evaluate_smoke():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    gen = torch.Generator().manual_seed(0)
    res = evaluate(brain, head, [(1, 1), (2, 2)], size=5, start=(0, 0), max_steps=30, generator=gen)
    assert res.n == 2
    assert 0.0 <= res.success_rate <= 1.0
    assert res.mean_steps >= 0.0
    assert 0.0 <= res.optimality <= 1.0
