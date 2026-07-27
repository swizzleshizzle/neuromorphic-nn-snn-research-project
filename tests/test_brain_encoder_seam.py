import numpy as np
import pytest
import torch

from neuromorphic.brain import Brain
from neuromorphic.encoders import cube_encoder, grid_encoder
from neuromorphic.regions.sensory_cortex import encode_gridworld


def test_grid_default_is_unchanged():
    """The seam must not perturb existing grid behavior at all."""
    brain = Brain(grid_n=5, seed=0)
    obs = [0, 0, 4, 4]
    out = brain.step(obs, store=False, recall=False, generator=torch.Generator().manual_seed(7))
    expected_spikes = encode_gridworld(
        torch.tensor([[0, 0, 4, 4]]), grid_n=5, T=brain.T,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(out["obs_spikes"], expected_spikes)
    assert out["obs_spikes"].shape == (brain.T, 1, 50)


def test_brain_reports_total_neuron_count():
    brain = Brain(grid_n=5, n_actions=6, seed=0)
    assert brain.n_neurons == 510  # sensory 192 + hippo 150 + pfc 150 + router 12 + motor 6
    assert brain.n_neurons == sum(r.n_neurons for r in brain._regions.values())


def test_cube_configured_brain_runs():
    brain = Brain(
        encoder=cube_encoder(), n_obs=144, obs_width=24, n_actions=6, seed=0,
    )
    obs = np.zeros(24, dtype=np.int64)
    out = brain.step(obs, store=False, recall=False, generator=torch.Generator().manual_seed(0))
    assert out["obs_spikes"].shape == (brain.T, 1, 144)
    assert out["concept"].shape == (brain.T, 1, brain.content)
    assert out["utilities"].shape == (brain.T, 1, 6)
    assert out["action"] in range(6)


def test_obs_width_is_validated():
    brain = Brain(encoder=cube_encoder(), n_obs=144, obs_width=24, n_actions=6, seed=0)
    with pytest.raises(ValueError, match="24"):
        brain.step([0, 0, 4, 4], store=False, recall=False)


def test_encoders_are_picklable():
    """The driver fans out over ProcessPoolExecutor; a lambda encoder would break it."""
    import pickle
    for enc in (grid_encoder(5), cube_encoder()):
        assert pickle.loads(pickle.dumps(enc)) is not None
