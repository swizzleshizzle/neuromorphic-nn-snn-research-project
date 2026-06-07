"""Gated-loop integration: Router gates pathway 5 (PFC→Motor) (build-order step 4).

The Thalamic Router reads PFC utilities, selects a winner, and emits gate-closed
control lines. ``apply_gate`` releases pathway 5 through them so only the selected
action's channel reaches Motor — and a below-floor utility vetoes the action.
"""

from __future__ import annotations

import torch

from neuromorphic.connections import apply_gate
from neuromorphic.regions import (
    MotorCortex,
    Prefrontal,
    SensoryCortex,
    ThalamicRouter,
    encode_gridworld,
)

GRID_N = 5
N_OBS = 2 * GRID_N * GRID_N
T = 32
N_ACTIONS = 4


def util(rates, seed=0):
    gen = torch.Generator().manual_seed(seed)
    rate = torch.tensor(rates).view(1, 1, -1).expand(T, 1, len(rates)).contiguous()
    return torch.bernoulli(rate, generator=gen)


def test_full_gated_pipeline_shapes():
    sensory = SensoryCortex(n_obs=N_OBS, concept=64, num_steps=T, seed=0)
    pfc = Prefrontal(concept_dim=64, n_actions=N_ACTIONS, num_steps=T, seed=0)
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    motor = MotorCortex(n_actions=N_ACTIONS, num_steps=T)

    spk = encode_gridworld(torch.tensor([[0, 0, 4, 4]]), grid_n=GRID_N, T=T,
                           generator=torch.Generator().manual_seed(0))
    utilities = pfc(sensory(spk))
    gate = router(utilities)
    action = motor(apply_gate(utilities, gate))
    assert gate.shape == (T, 1, N_ACTIONS)
    assert action.shape == (T, 1, N_ACTIONS)


def test_router_constrains_motor_to_selected_action():
    """Only the router-selected channel reaches Motor → Motor emits that action."""
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    motor = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    utilities = util([0.1, 0.1, 0.6, 0.1])  # action 2 strongest
    gate = router(utilities)
    selected = int(router.open_mask(gate).sum(dim=0)[0].argmax())
    action = motor(apply_gate(utilities, gate))
    assert int(motor.winner(action)[0]) == selected == 2


def test_gate_blocks_unselected_channels():
    """The gated pathway carries no spikes on closed (non-selected) channels."""
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    utilities = util([0.1, 0.1, 0.6, 0.1])
    gate = router(utilities)
    gated = apply_gate(utilities, gate)
    open_mask = router.open_mask(gate)
    # Wherever a channel is closed, the gated signal must be zero.
    assert torch.all(gated[open_mask == 0] == 0)


def test_do_nothing_vetoes_motor():
    """Below-floor utilities → router opens nothing → Motor stays silent (no action)."""
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    motor = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    weak = util([0.08, 0.08, 0.08, 0.08])
    gate = router(weak)
    action = motor(apply_gate(weak, gate))
    assert action.sum() == 0


def test_off_mode_passes_everything():
    """mode='off' opens all channels → gated signal equals the raw utilities."""
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T, mode="off")
    utilities = util([0.1, 0.3, 0.6, 0.2])
    gate = router(utilities)
    assert torch.equal(apply_gate(utilities, gate), utilities)
