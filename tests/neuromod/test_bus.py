"""Tests for the NeuromodBus (Phase 2, build-order step 6)."""

from __future__ import annotations

import torch

from neuromorphic.neuromod import NeuromodBus
from neuromorphic.regions import MotorCortex

T = 32
N_ACTIONS = 4


def utility_spikes(rates, seed=0):
    gen = torch.Generator().manual_seed(seed)
    rate = torch.tensor(rates).view(1, 1, -1).expand(T, 1, len(rates)).contiguous()
    return torch.bernoulli(rate, generator=gen)


def winner_share(counts):
    counts = counts.float()
    return (counts.max() / counts.sum()).item()


def test_defaults():
    bus = NeuromodBus()
    assert bus.dopamine == 0.0
    assert bus.ach == 1.0


def test_set_updates_values():
    bus = NeuromodBus()
    bus.set(dopamine=1.0, ach=2.5)
    assert bus.dopamine == 1.0
    assert bus.ach == 2.5


def test_set_is_partial():
    bus = NeuromodBus(dopamine=0.3, ach=1.2)
    bus.set(ach=2.0)
    assert bus.dopamine == 0.3
    assert bus.ach == 2.0


def test_reset_restores_defaults():
    bus = NeuromodBus(dopamine=1.0, ach=3.0)
    bus.reset()
    assert bus.dopamine == 0.0
    assert bus.ach == 1.0


def test_learning_enabled_tracks_dopamine():
    bus = NeuromodBus(learning_threshold=0.5)
    assert not bus.learning_enabled
    bus.set(dopamine=0.8)
    assert bus.learning_enabled


def test_ach_sharpens_motor_wta():
    """ACh = gain/precision: higher ACh sharpens Motor's WTA (winner takes a larger share)."""
    spk = utility_spikes([0.6, 0.9, 0.55, 0.5])  # close competitors
    lo = MotorCortex(n_actions=N_ACTIONS, num_steps=T, bus=NeuromodBus(ach=0.3))
    hi = MotorCortex(n_actions=N_ACTIONS, num_steps=T, bus=NeuromodBus(ach=2.0))
    share_lo = winner_share(lo(spk).sum(dim=0)[0])
    share_hi = winner_share(hi(spk).sum(dim=0)[0])
    assert share_hi > share_lo


def test_motor_without_bus_unchanged():
    """A Motor with no bus behaves exactly as ach=1.0 (back-compatible)."""
    spk = utility_spikes([0.2, 0.9, 0.2, 0.2])
    no_bus = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    unit_bus = MotorCortex(n_actions=N_ACTIONS, num_steps=T, bus=NeuromodBus(ach=1.0))
    assert torch.equal(no_bus(spk), unit_bus(spk))
