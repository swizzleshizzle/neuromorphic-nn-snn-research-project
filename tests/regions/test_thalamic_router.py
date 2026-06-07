"""Tests for the Thalamic Router (Phase 2, build-order step 4)."""

from __future__ import annotations

import pytest
import torch

from neuromorphic.regions import BrainRegion
from neuromorphic.regions.thalamic_router import ThalamicRouter

T = 32
N_ACTIONS = 4


def utilities(rates, T: int = T, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    rate = torch.tensor(rates).view(1, 1, -1).expand(T, 1, len(rates)).contiguous()
    return torch.bernoulli(rate, generator=gen)


def test_is_a_brain_region():
    assert isinstance(ThalamicRouter(n_actions=N_ACTIONS, num_steps=T), BrainRegion)


def test_forward_output_shape():
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    gate = router(utilities([0.1, 0.1, 0.6, 0.1]))
    assert gate.shape == (T, 1, N_ACTIONS)


def test_selected_channel_opens():
    """The strongest-utility channel is the one disinhibited (opened)."""
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    gate = router(utilities([0.1, 0.1, 0.6, 0.1]))  # action 2 strongest
    open_mask = router.open_mask(gate)
    assert int(open_mask.sum(dim=0)[0].argmax()) == 2
    assert open_mask[:, :, 2].sum() > 0


def test_losers_stay_closed():
    """Non-selected channels remain tonically closed."""
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    open_mask = router.open_mask(router(utilities([0.1, 0.1, 0.6, 0.1])))
    opens = open_mask.sum(dim=0)[0]  # [N_ACTIONS]
    winner = int(opens.argmax())
    losers_total = opens.sum() - opens[winner]
    assert opens[winner] > losers_total


def test_do_nothing_floor():
    """Weak utilities (no clear winner) open no channel — the router withholds."""
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    open_mask = router.open_mask(router(utilities([0.08, 0.08, 0.08, 0.08])))
    assert open_mask.sum() == 0


def test_mode_off_opens_everything():
    """In 'off' mode there is no tonic inhibition — all channels stay open."""
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T, mode="off")
    gate = router(utilities([0.1, 0.1, 0.6, 0.1]))
    assert gate.sum() == 0  # gate-closed is never asserted


def test_forward_is_deterministic():
    x = utilities([0.1, 0.1, 0.6, 0.1])
    a = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)(x)
    b = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)(x)
    assert torch.equal(a, b)


def test_recording_exposes_select_and_gate():
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    router.enable_recording(True)
    router(utilities([0.1, 0.1, 0.6, 0.1]))
    assert router.get_recording("select").shape == (T, 1, N_ACTIONS)
    assert router.get_recording("gate").shape == (T, 1, N_ACTIONS)


def test_rejects_non_3d_input():
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    with pytest.raises(ValueError):
        router(torch.rand(T, N_ACTIONS))


def test_rejects_wrong_action_dim():
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    with pytest.raises(ValueError):
        router(torch.rand(T, 1, N_ACTIONS + 1))
