"""Brain assembly smoke: random input → spikes flow through all five regions."""

from __future__ import annotations

import torch

from neuromorphic.brain import Brain

# region id -> its primary output recording key (the spike train it emits)
PRIMARY = {
    "sensory": "concept",
    "hippocampus": "population",
    "prefrontal": "utility",
    "router": "select",
    "motor": "action",
}


def test_spikes_flow_through_all_regions():
    brain = Brain(grid_n=5, seed=0)
    gen = torch.Generator().manual_seed(0)
    # a random valid observation (coords in [0, grid_n))
    obs = torch.randint(0, 5, (4,), generator=gen).tolist()

    out = brain.step(obs, store=True, recall=True, record=True, generator=gen)
    recordings = out["recordings"]

    # every region produced recordings, and its primary output train has spikes
    for region, key in PRIMARY.items():
        assert region in recordings, f"{region} produced no recordings"
        train = recordings[region].get(key)
        assert train is not None, f"{region} has no '{key}' recording"
        assert train.sum().item() > 0, f"{region} ({key}) emitted zero spikes"


def test_step_produces_a_valid_action():
    brain = Brain(grid_n=5, seed=0)
    out = brain.step([0, 0, 4, 4], recall=False, generator=torch.Generator().manual_seed(0))
    assert isinstance(out["action"], int)
    assert 0 <= out["action"] < 4
