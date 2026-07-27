import numpy as np
import pytest
import torch

from neuromorphic.brain import Brain
from neuromorphic.encoders import cube_encoder
from neuromorphic.monolithic import MonolithicBrain
from neuromorphic.training.reinforce import action_distribution, make_policy_head


def _cube_pair(seed=0):
    brain = Brain(encoder=cube_encoder(), n_obs=144, obs_width=24, n_actions=6, seed=seed)
    mono = MonolithicBrain(
        n_obs=144, n_actions=6, total_neurons=brain.n_neurons,
        content=brain.content, obs_width=24, encoder=cube_encoder(), seed=seed,
    )
    return brain, mono


def test_neuron_count_matches_the_brain_exactly():
    brain, mono = _cube_pair()
    assert mono.n_neurons == brain.n_neurons == 510


def test_neuron_count_flows_from_total_neurons_parameter():
    """Verify n_neurons is computed from the parameter, not hardcoded to 510."""
    mono = MonolithicBrain(
        n_obs=144, n_actions=6, total_neurons=200, content=64, obs_width=24, encoder=cube_encoder()
    )
    assert mono.n_neurons == 200
    assert mono.stack.n_neurons == 200


def test_step_returns_a_concept_of_the_right_shape():
    _, mono = _cube_pair()
    out = mono.step(np.zeros(24, dtype=np.int64), generator=torch.Generator().manual_seed(0))
    assert out["concept"].shape == (mono.T, 1, mono.content)
    assert out["obs_spikes"].shape == (mono.T, 1, 144)


def test_is_a_drop_in_for_reinforce():
    """The whole point: reinforce.py must consume it with no changes."""
    _, mono = _cube_pair()
    head = make_policy_head(mono, "linear")
    dist, logits = action_distribution(
        mono, head, np.zeros(24, dtype=np.int64), generator=torch.Generator().manual_seed(0)
    )
    assert logits.shape == (6,)
    assert 0 <= int(dist.sample()) < 6


def test_head_is_parameter_identical_to_the_regionalized_arm():
    """Same content width means the two arms train the same number of head parameters."""
    brain, mono = _cube_pair()
    n_brain = sum(p.numel() for p in make_policy_head(brain, "linear").parameters())
    n_mono = sum(p.numel() for p in make_policy_head(mono, "linear").parameters())
    assert n_brain == n_mono


def test_rejects_a_budget_that_cannot_hold_the_concept():
    with pytest.raises(ValueError, match="total_neurons"):
        MonolithicBrain(n_obs=144, n_actions=6, total_neurons=64, content=64)
