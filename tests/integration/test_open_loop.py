"""Open-loop integration: Sensory → Prefrontal → Motor (build-order step 3).

Verifies the three regions chain end-to-end on a grid-world observation. No
memory, no gating. The pipeline is untrained, so the *winning action* is not
task-meaningful yet — the gate is end-to-end spike flow, correct shapes, every
stage alive, and a graded utility code that varies with the observation.
"""

from __future__ import annotations

import torch

from neuromorphic.regions import MotorCortex, Prefrontal, SensoryCortex, encode_gridworld

GRID_N = 5
N_OBS = 2 * GRID_N * GRID_N
T = 32
N_ACTIONS = 4


def build():
    sensory = SensoryCortex(n_obs=N_OBS, concept=64, num_steps=T, seed=0)
    pfc = Prefrontal(concept_dim=64, n_actions=N_ACTIONS, num_steps=T, seed=0)
    motor = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    return sensory, pfc, motor


def run(sensory, pfc, motor, obs, seed=0):
    spikes_in = encode_gridworld(obs, grid_n=GRID_N, T=T, generator=torch.Generator().manual_seed(seed))
    concept = sensory(spikes_in)
    utilities = pfc(concept)
    action = motor(utilities)
    return concept, utilities, action


def test_pipeline_shapes_flow_end_to_end():
    sensory, pfc, motor = build()
    concept, utilities, action = run(sensory, pfc, motor, torch.tensor([[0, 0, 4, 4]]))
    assert concept.shape == (T, 1, 64)
    assert utilities.shape == (T, 1, N_ACTIONS)
    assert action.shape == (T, 1, N_ACTIONS)


def test_every_stage_is_alive():
    sensory, pfc, motor = build()
    concept, utilities, action = run(sensory, pfc, motor, torch.tensor([[2, 2, 4, 4]]))
    assert concept.sum() > 0
    assert utilities.sum() > 0
    assert action.sum() > 0


def test_motor_produces_single_winner():
    sensory, pfc, motor = build()
    _, _, action = run(sensory, pfc, motor, torch.tensor([[0, 0, 4, 4]]))
    counts = action.sum(dim=0)[0]
    winner = int(counts.argmax())
    losers_total = counts.sum() - counts[winner]
    assert counts[winner] > losers_total  # one action dominates


def test_pipeline_is_deterministic():
    s1, p1, m1 = build()
    s2, p2, m2 = build()
    obs = torch.tensor([[1, 3, 4, 4]])
    a1 = run(s1, p1, m1, obs)[2]
    a2 = run(s2, p2, m2, obs)[2]
    assert torch.equal(a1, a2)


def test_utility_code_varies_with_observation():
    """Graded selectivity propagates: distinct positions → distinct utility codes."""
    sensory, pfc, motor = build()
    _, u_a, _ = run(sensory, pfc, motor, torch.tensor([[0, 0, 4, 4]]))
    _, u_b, _ = run(sensory, pfc, motor, torch.tensor([[4, 4, 0, 0]]))
    assert not torch.equal(u_a.sum(dim=0), u_b.sum(dim=0))
