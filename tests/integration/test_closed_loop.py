"""Closed-loop integration: Sensory → PFC + gated Hippocampus recall → PFC, with
the Sensory snapshot stored under router gating (EXP-020, Phase 2 step 2.1/2.2).

Wires the Week-10 upgrades end-to-end:
- pathway 2: Sensory concept → PFC (driver),
- pathway 3: Sensory concept → Hippocampus **store** content (router-gated),
- pathway 4: Hippocampus recall → PFC second afferent (router-gated).

The payoff verified here is that **recalling a memory shifts the PFC utility code**
versus the sensory-only case — the integration of Task 1 (PFC multi-source) and
Task 3 (sensory-snapshot store) together. Untrained, so the winning action is not
task-meaningful; the gate is spike flow, correct shapes, every stage alive, a
content-specific stored attractor, and a recall-driven shift.
"""

from __future__ import annotations

import torch

from neuromorphic.connections import apply_gate
from neuromorphic.regions import (
    Hippocampus,
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
CONTENT = 64
STORE_WINDOW = (0, 6)


def gate_closed_window(lo: int, hi: int, n: int) -> torch.Tensor:
    """Router store/recall command as a [T,1,n] gate: open (0) inside [lo,hi)."""
    g = torch.ones(T, 1, n)
    g[lo:hi] = 0.0
    return g


def build():
    sensory = SensoryCortex(n_obs=N_OBS, concept=CONTENT, num_steps=T, seed=0)
    pfc = Prefrontal(concept_dim=CONTENT, recall_dim=CONTENT, n_actions=N_ACTIONS, num_steps=T, seed=0)
    hippo = Hippocampus(content_dim=CONTENT, n_neurons=150, num_steps=T, seed=0)
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    motor = MotorCortex(n_actions=N_ACTIONS, num_steps=T)
    return sensory, pfc, hippo, router, motor


def concept_for(sensory, obs):
    spk = encode_gridworld(obs, grid_n=GRID_N, T=T, generator=torch.Generator().manual_seed(0))
    return sensory(spk)


def store_and_recall(sensory, hippo, obs):
    """Store the gated Sensory snapshot (pathway 3) and read the recall code back."""
    concept = concept_for(sensory, obs)
    store_gate = gate_closed_window(*STORE_WINDOW, CONTENT)
    snapshot = apply_gate(concept, store_gate).mean(dim=0)  # [B, CONTENT] sensory snapshot
    pattern = hippo.store(snapshot).bool()
    cue = apply_gate(concept, store_gate)                   # gated sensory cue into the attractor
    hippo.enable_recording(True)
    recall = hippo(cue)
    population = hippo.get_recording("population")
    hippo.enable_recording(False)
    return concept, recall, population, pattern


def test_closed_loop_shapes():
    sensory, pfc, hippo, _, _ = build()
    concept, recall, population, _ = store_and_recall(sensory, hippo, torch.tensor([[0, 0, 4, 4]]))
    util = pfc(concept, recall_spikes=recall)
    assert concept.shape == (T, 1, CONTENT)
    assert recall.shape == (T, 1, CONTENT)
    assert population.shape == (T, 1, 150)
    assert util.shape == (T, 1, N_ACTIONS)


def test_every_stage_is_alive():
    sensory, pfc, hippo, _, motor = build()
    concept, recall, population, _ = store_and_recall(sensory, hippo, torch.tensor([[0, 0, 4, 4]]))
    util = pfc(concept, recall_spikes=recall)
    assert concept.sum() > 0
    assert population.sum() > 0
    assert recall.sum() > 0
    assert util.sum() > 0
    assert motor(util).sum() > 0


def test_recall_shifts_pfc_utilities():
    """The payoff: recalling the stored memory changes the PFC utility code."""
    sensory, pfc, hippo, _, _ = build()
    concept, recall, _, _ = store_and_recall(sensory, hippo, torch.tensor([[0, 0, 4, 4]]))
    util_on = pfc(concept, recall_spikes=recall).sum(dim=0)
    util_off = pfc(concept, recall_spikes=None).sum(dim=0)
    assert not torch.equal(util_on, util_off)


def test_store_from_sensory_is_content_specific():
    """Task 3: a store driven by the Sensory snapshot holds a content-specific pattern."""
    sensory, _, hippo, _, _ = build()
    _, _, population, pattern = store_and_recall(sensory, hippo, torch.tensor([[0, 0, 4, 4]]))
    late = population[T // 2:].float().mean(dim=0)[0]
    held = late[pattern].mean().item()
    leak = late[~pattern].mean().item()
    assert held >= 0.9 and leak <= 0.1


def test_motor_selects_single_winner_through_router():
    """Router gates pathway 5; Motor still resolves a single winner."""
    sensory, pfc, hippo, router, motor = build()
    concept, recall, _, _ = store_and_recall(sensory, hippo, torch.tensor([[0, 0, 4, 4]]))
    util = pfc(concept, recall_spikes=recall)
    action = motor(apply_gate(util, router(util)))
    counts = action.sum(dim=0)[0]
    winner = int(counts.argmax())
    assert counts[winner] > counts.sum() - counts[winner]  # one action dominates


def test_closed_loop_is_deterministic():
    s1, p1, h1, _, _ = build()
    s2, p2, h2, _, _ = build()
    obs = torch.tensor([[1, 3, 4, 4]])
    c1, r1, _, _ = store_and_recall(s1, h1, obs)
    c2, r2, _, _ = store_and_recall(s2, h2, obs)
    assert torch.equal(p1(c1, recall_spikes=r1), p2(c2, recall_spikes=r2))
