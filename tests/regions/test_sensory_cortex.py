"""Tests for the Sensory Cortex region + grid-world encoder (Phase 2, Step 2.2)."""

from __future__ import annotations

import pytest
import torch

from neuromorphic.regions import BrainRegion
from neuromorphic.regions.sensory_cortex import SensoryCortex, encode_gridworld

GRID_N = 5
N_OBS = 2 * GRID_N * GRID_N  # agent one-hot + goal one-hot = 50
T = 32


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------------------- #
# encode_gridworld
# --------------------------------------------------------------------------- #
def test_encoder_output_shape():
    obs = torch.tensor([[0, 0, 4, 4], [2, 3, 1, 1]])  # [B=2, 4]
    spk = encode_gridworld(obs, grid_n=GRID_N, T=T, generator=_gen(0))
    assert spk.shape == (T, 2, N_OBS)


def test_encoder_output_is_binary():
    obs = torch.tensor([[1, 1, 3, 3]])
    spk = encode_gridworld(obs, grid_n=GRID_N, T=T, generator=_gen(0))
    assert torch.all((spk == 0) | (spk == 1))


def test_encoder_only_active_cells_fire():
    """Exactly the agent cell and the goal cell may spike; all others are silent."""
    obs = torch.tensor([[0, 0, 4, 4]])  # agent (0,0) -> idx 0; goal (4,4) -> idx 24
    spk = encode_gridworld(obs, grid_n=GRID_N, T=T, generator=_gen(0))
    fired = spk.sum(dim=0)[0] > 0  # [N_OBS] which columns ever fired
    active = torch.zeros(N_OBS, dtype=torch.bool)
    active[0] = True                       # agent half, cell (0,0)
    active[GRID_N * GRID_N + 24] = True     # goal half, cell (4,4)
    assert torch.equal(fired, active)


def test_encoder_is_deterministic_under_seed():
    obs = torch.tensor([[2, 2, 0, 4]])
    a = encode_gridworld(obs, grid_n=GRID_N, T=T, generator=_gen(123))
    b = encode_gridworld(obs, grid_n=GRID_N, T=T, generator=_gen(123))
    assert torch.equal(a, b)


def test_encoder_rejects_bad_coords():
    obs = torch.tensor([[0, 0, 5, 0]])  # x=5 out of range for 5x5
    with pytest.raises(ValueError):
        encode_gridworld(obs, grid_n=GRID_N, T=T, generator=_gen(0))


# --------------------------------------------------------------------------- #
# SensoryCortex
# --------------------------------------------------------------------------- #
def test_is_a_brain_region():
    region = SensoryCortex(n_obs=N_OBS, num_steps=T, seed=0)
    assert isinstance(region, BrainRegion)


def test_forward_output_shape():
    region = SensoryCortex(n_obs=N_OBS, concept=64, num_steps=T, seed=0)
    spk = encode_gridworld(torch.tensor([[0, 0, 4, 4]]), grid_n=GRID_N, T=T, generator=_gen(0))
    out = region(spk)
    assert out.shape == (T, 1, 64)


def test_concept_layer_is_not_dead():
    """Sparse grid input must drive measurable spikes in the concept layer."""
    region = SensoryCortex(n_obs=N_OBS, num_steps=T, seed=0)
    spk = encode_gridworld(torch.tensor([[2, 2, 4, 4]]), grid_n=GRID_N, T=T, generator=_gen(0))
    out = region(spk)
    assert out.sum() > 0


def test_distinct_positions_give_distinct_concept_codes():
    """Two different agent positions produce measurably different concept codes."""
    region = SensoryCortex(n_obs=N_OBS, num_steps=T, seed=0)
    spk_a = encode_gridworld(torch.tensor([[0, 0, 4, 4]]), grid_n=GRID_N, T=T, generator=_gen(0))
    spk_b = encode_gridworld(torch.tensor([[4, 0, 4, 4]]), grid_n=GRID_N, T=T, generator=_gen(0))
    code_a = region(spk_a).sum(dim=0)  # [B, 64] spike-count code
    code_b = region(spk_b).sum(dim=0)
    assert not torch.equal(code_a, code_b)


def test_forward_is_deterministic():
    spk = encode_gridworld(torch.tensor([[1, 1, 3, 3]]), grid_n=GRID_N, T=T, generator=_gen(0))
    a = SensoryCortex(n_obs=N_OBS, num_steps=T, seed=0)(spk)
    b = SensoryCortex(n_obs=N_OBS, num_steps=T, seed=0)(spk)
    assert torch.equal(a, b)


def test_recording_exposes_hidden_and_concept():
    region = SensoryCortex(n_obs=N_OBS, hidden=128, concept=64, num_steps=T, seed=0)
    region.enable_recording(True)
    spk = encode_gridworld(torch.tensor([[2, 2, 4, 4]]), grid_n=GRID_N, T=T, generator=_gen(0))
    region(spk)
    assert region.get_recording("hidden").shape == (T, 1, 128)
    assert region.get_recording("concept").shape == (T, 1, 64)
