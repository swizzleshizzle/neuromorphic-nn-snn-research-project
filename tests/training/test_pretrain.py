import torch

from neuromorphic.training.pretrain import (
    displacement_target,
    enumerate_states,
    split_states,
)


def test_displacement_target_normalized_vectors():
    obs = torch.tensor([[0, 0, 4, 4], [4, 0, 0, 0], [2, 2, 2, 2]])
    d = displacement_target(obs, grid_n=5)
    assert d.shape == (3, 2)
    assert torch.allclose(d[0], torch.tensor([1.0, 1.0]))    # (4-0, 4-0)/4
    assert torch.allclose(d[1], torch.tensor([-1.0, 0.0]))   # (0-4, 0-0)/4
    assert torch.allclose(d[2], torch.tensor([0.0, 0.0]))    # same cell


def test_enumerate_states_count_and_no_self_pairs():
    states = enumerate_states(grid_n=5)
    assert states.shape == (5**2 * (5**2 - 1), 4)   # 600
    agent = states[:, :2]
    goal = states[:, 2:]
    assert not torch.any((agent == goal).all(dim=1))


def test_split_states_disjoint_sized_deterministic():
    states = enumerate_states(grid_n=5)
    tr_a, he_a = split_states(states, frac_heldout=0.2, seed=0)
    tr_b, he_b = split_states(states, frac_heldout=0.2, seed=0)
    assert he_a.shape[0] == round(states.shape[0] * 0.2)   # 120
    assert tr_a.shape[0] + he_a.shape[0] == states.shape[0]
    assert torch.equal(tr_a, tr_b) and torch.equal(he_a, he_b)   # deterministic
    # disjoint: no held-out row appears in train
    tr_set = {tuple(r.tolist()) for r in tr_a}
    assert all(tuple(r.tolist()) not in tr_set for r in he_a)
