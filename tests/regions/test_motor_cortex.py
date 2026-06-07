"""Tests for the Motor Cortex region + WTA (Phase 2, build-order step 2)."""

from __future__ import annotations

import pytest
import torch

from neuromorphic.regions import BrainRegion
from neuromorphic.regions.motor_cortex import MotorCortex

T = 32
N_ACTIONS = 4


def utility_spikes(rates: list[float], T: int = T, seed: int = 0) -> torch.Tensor:
    """Hand-made utility input: [T, 1, N_actions] Poisson spikes at the given per-step rates."""
    gen = torch.Generator().manual_seed(seed)
    rate = torch.tensor(rates).view(1, 1, -1).expand(T, 1, len(rates)).contiguous()
    return torch.bernoulli(rate, generator=gen)


def test_is_a_brain_region():
    assert isinstance(MotorCortex(n_actions=N_ACTIONS, num_steps=T), BrainRegion)


def test_forward_output_shape():
    region = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    out = region(utility_spikes([0.2, 0.9, 0.2, 0.2]))
    assert out.shape == (T, 1, N_ACTIONS)


def test_single_winner_matches_max_utility():
    """The action with the highest input utility wins the WTA."""
    region = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    out = region(utility_spikes([0.2, 0.2, 0.9, 0.2]))  # action 2 strongest
    counts = out.sum(dim=0)[0]  # [N_ACTIONS] spike count per action
    assert int(counts.argmax()) == 2


def test_winner_tracks_the_strong_action():
    """Moving the strong utility to a different action moves the winner."""
    region = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    for strong in range(N_ACTIONS):
        rates = [0.2] * N_ACTIONS
        rates[strong] = 0.9
        counts = region(utility_spikes(rates, seed=strong)).sum(dim=0)[0]
        assert int(counts.argmax()) == strong


def test_winner_dominates_losers():
    """Lateral inhibition suppresses the losers: winner outspikes all others combined."""
    region = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    counts = region(utility_spikes([0.2, 0.9, 0.2, 0.2])).sum(dim=0)[0]
    winner = int(counts.argmax())
    losers_total = counts.sum() - counts[winner]
    assert counts[winner] > losers_total


def test_inhibition_sharpens_selection():
    """Inhibition raises the winner's share of total spikes (sharper WTA)."""
    rates = [0.6, 0.9, 0.55, 0.5]  # close competitors
    spk = utility_spikes(rates)
    no_inh = MotorCortex(n_actions=N_ACTIONS, num_steps=T, inhibition=0.0)(spk).sum(0)[0]
    with_inh = MotorCortex(n_actions=N_ACTIONS, num_steps=T, inhibition=3.0)(spk).sum(0)[0]

    def winner_share(counts: torch.Tensor) -> float:
        counts = counts.float()
        return (counts.max() / counts.sum()).item()

    assert winner_share(with_inh) > winner_share(no_inh)


def test_forward_is_deterministic():
    spk = utility_spikes([0.2, 0.9, 0.2, 0.2])
    a = MotorCortex(n_actions=N_ACTIONS, num_steps=T)(spk)
    b = MotorCortex(n_actions=N_ACTIONS, num_steps=T)(spk)
    assert torch.equal(a, b)


def test_winner_helper():
    region = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    out = region(utility_spikes([0.2, 0.2, 0.2, 0.9]))
    assert int(region.winner(out)[0]) == 3


def test_recording_exposes_action():
    region = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    region.enable_recording(True)
    region(utility_spikes([0.2, 0.9, 0.2, 0.2]))
    assert region.get_recording("action").shape == (T, 1, N_ACTIONS)


def test_rejects_non_3d_input():
    region = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    with pytest.raises(ValueError):
        region(torch.rand(T, N_ACTIONS))


def test_rejects_wrong_action_dim():
    region = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    with pytest.raises(ValueError):
        region(torch.rand(T, 1, N_ACTIONS + 1))
