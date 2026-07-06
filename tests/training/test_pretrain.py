import torch

from neuromorphic.regions.sensory_cortex import SensoryCortex
from neuromorphic.training.pretrain import (
    concept_rate_batch,
    displacement_target,
    enumerate_states,
    pretrain_sensory,
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


def _fresh_sensory(seed=0):
    # n_obs = 2 * grid_n**2 for a 5x5 grid = 50
    return SensoryCortex(n_obs=2 * 5 * 5, seed=seed)


def test_concept_rate_batch_shape_and_differentiable():
    sensory = _fresh_sensory()
    obs = torch.tensor([[0, 0, 4, 4], [1, 2, 3, 0]])
    gen = torch.Generator().manual_seed(0)
    rate = concept_rate_batch(sensory, obs, grid_n=5, T=8, generator=gen)
    assert rate.shape == (2, sensory.concept)
    # gradient must reach the encoder weights (the load-bearing requirement)
    rate.sum().backward()
    assert sensory.fc1.weight.grad is not None
    assert torch.any(sensory.fc1.weight.grad != 0)


def test_pretrain_sensory_learns_and_changes_encoder():
    sensory = _fresh_sensory(seed=0)
    w1_before = sensory.fc1.weight.detach().clone()
    info = pretrain_sensory(
        sensory, grid_n=5, epochs=30, lr=1e-3, frac_heldout=0.2, seed=0, T=8,
        generator=torch.Generator().manual_seed(0),
    )
    assert set(info) == {"train_disp_error", "heldout_disp_error", "epochs", "freeze_encoder"}
    assert info["epochs"] == 30
    assert info["freeze_encoder"] is False
    assert torch.isfinite(torch.tensor(info["heldout_disp_error"]))
    # encoder weights moved
    assert not torch.equal(w1_before, sensory.fc1.weight.detach())


def test_pretrain_sensory_freeze_encoder_leaves_weights_unchanged():
    sensory = _fresh_sensory(seed=0)
    w1_before = sensory.fc1.weight.detach().clone()
    info = pretrain_sensory(
        sensory, grid_n=5, epochs=20, lr=1e-3, seed=0, T=8,
        generator=torch.Generator().manual_seed(0), freeze_encoder=True,
    )
    assert info["freeze_encoder"] is True
    # only the scratch readout trained; the encoder is untouched
    assert torch.equal(w1_before, sensory.fc1.weight.detach())


def test_pretrain_sensory_is_deterministic():
    def run():
        s = _fresh_sensory(seed=1)
        return pretrain_sensory(
            s, grid_n=5, epochs=20, lr=1e-3, seed=1, T=8,
            generator=torch.Generator().manual_seed(1),
        )["heldout_disp_error"]
    assert run() == run()
