import pytest
import torch

from neuromorphic.regions.sensory_cortex import encode_cube


def test_shape_and_one_hot_rate():
    # 2x2 -> 24 facelets, 6 colors -> 144 inputs
    obs = torch.zeros(2, 24, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    spikes = encode_cube(obs, cube_n=2, n_colors=6, T=32, max_rate=0.5, generator=g)
    assert spikes.shape == (32, 2, 144)
    assert spikes[:, 0, 0].sum() > 0
    assert spikes[:, 0, 1:6].sum() == 0


def test_exactly_one_active_slot_per_facelet():
    obs = torch.randint(0, 6, (4, 24))
    spikes = encode_cube(obs, T=64, max_rate=1.0, generator=torch.Generator().manual_seed(0))
    active = spikes.sum(0).gt(0)
    assert active.sum(dim=1).tolist() == [24, 24, 24, 24]


def test_rate_respects_max_rate():
    obs = torch.zeros(1, 24, dtype=torch.long)
    spikes = encode_cube(obs, T=4000, max_rate=0.25, generator=torch.Generator().manual_seed(0))
    assert abs(spikes[:, 0, 0].mean().item() - 0.25) < 0.03


def test_size_generic_3x3_width():
    # 3x3 -> 54 facelets, 6 colors -> 324 inputs (no code change needed)
    obs = torch.zeros(1, 54, dtype=torch.long)
    spikes = encode_cube(obs, cube_n=3, n_colors=6, T=8)
    assert spikes.shape == (8, 1, 324)


def test_reproducible_under_generator():
    obs = torch.randint(0, 6, (3, 24))
    a = encode_cube(obs, generator=torch.Generator().manual_seed(1))
    b = encode_cube(obs, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_rejects_bad_shape_and_colors():
    with pytest.raises(ValueError):
        encode_cube(torch.zeros(1, 24, dtype=torch.long), cube_n=3)
    with pytest.raises(ValueError):
        encode_cube(torch.full((1, 24), 6, dtype=torch.long))
