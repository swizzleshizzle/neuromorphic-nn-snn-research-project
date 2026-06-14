"""Tests for the Prefrontal region (Phase 2, build-order step 3)."""

from __future__ import annotations

import pytest
import torch

from neuromorphic.regions import BrainRegion
from neuromorphic.regions.prefrontal import Prefrontal

T = 32
CONCEPT = 64
N_ACTIONS = 4


def concept_spikes(rate: float = 0.4, seed: int = 0, B: int = 1) -> torch.Tensor:
    """Stand-in sensory concept code: [T, B, CONCEPT] Poisson spikes."""
    gen = torch.Generator().manual_seed(seed)
    probs = torch.full((T, B, CONCEPT), rate)
    return torch.bernoulli(probs, generator=gen)


def recall_spikes(rate: float = 0.4, seed: int = 7, B: int = 1) -> torch.Tensor:
    """Stand-in hippocampal recall code: [T, B, CONCEPT] Poisson spikes (pathway 4)."""
    gen = torch.Generator().manual_seed(seed)
    probs = torch.full((T, B, CONCEPT), rate)
    return torch.bernoulli(probs, generator=gen)


def test_is_a_brain_region():
    assert isinstance(
        Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0), BrainRegion
    )


def test_forward_output_shape():
    pfc = Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0)
    out = pfc(concept_spikes())
    assert out.shape == (T, 1, N_ACTIONS)


def test_utility_output_is_not_dead():
    pfc = Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0)
    out = pfc(concept_spikes(rate=0.4))
    assert out.sum() > 0


def test_has_recurrent_state():
    """PFC holds recurrent state-hold state across the window (RLeaky)."""
    pfc = Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0)
    pfc(concept_spikes())
    state = pfc.get_state()
    assert "spk_state" in state and "mem_state" in state


def test_forward_is_deterministic():
    x = concept_spikes(seed=5)
    a = Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0)(x)
    b = Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0)(x)
    assert torch.equal(a, b)


def test_distinct_inputs_give_distinct_utilities():
    """Different concept codes produce different action-utility codes."""
    pfc = Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0)
    ua = pfc(concept_spikes(seed=1)).sum(dim=0)
    ub = pfc(concept_spikes(seed=2)).sum(dim=0)
    assert not torch.equal(ua, ub)


def test_recording_exposes_stages():
    pfc = Prefrontal(
        concept_dim=CONCEPT, n_state=100, n_transform=50, n_actions=N_ACTIONS,
        num_steps=T, seed=0,
    )
    pfc.enable_recording(True)
    pfc(concept_spikes())
    assert pfc.get_recording("state").shape == (T, 1, 100)
    assert pfc.get_recording("transform").shape == (T, 1, 50)
    assert pfc.get_recording("utility").shape == (T, 1, N_ACTIONS)


def test_afferent_delay_zeros_first_step():
    """Pathway-2 delay Δ=1: the afferent contributes nothing at t=0."""
    pfc = Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, delay=1, seed=0)
    afferent = pfc.afferent(concept_spikes())
    assert torch.count_nonzero(afferent[0]) == 0


def test_rejects_non_3d_input():
    pfc = Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0)
    with pytest.raises(ValueError):
        pfc(torch.rand(T, CONCEPT))


def test_rejects_wrong_concept_dim():
    pfc = Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0)
    with pytest.raises(ValueError):
        pfc(torch.rand(T, 1, CONCEPT + 1))


# --- Multi-source integration (spec §2.3, Week-10 S2) ---------------------- #

def test_recall_none_matches_single_source_golden():
    """recall=None reproduces the pre-multi-source utility code byte-for-byte.

    Locks the EXP-015 sensory-only open loop: adding the second afferent must
    not perturb the sensory path's weights or output. Golden = per-action spike
    counts of the single-source implementation (seed=0, concept_spikes(seed=0)).
    """
    pfc = Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0)
    out = pfc(concept_spikes(rate=0.4, seed=0))
    assert out.sum(dim=0)[0].tolist() == [0.0, 0.0, 3.0, 0.0]


def test_recall_none_equals_omitted():
    """Passing recall=None is identical to omitting the recall argument."""
    pfc = Prefrontal(concept_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0)
    x = concept_spikes(seed=3)
    assert torch.equal(pfc(x), pfc(x, recall_spikes=None))


def test_two_source_shape_contract():
    """[T,B,64] concept + [T,B,64] recall → [T,B,N_actions]."""
    pfc = Prefrontal(
        concept_dim=CONCEPT, recall_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0
    )
    out = pfc(concept_spikes(seed=1), recall_spikes=recall_spikes(seed=2))
    assert out.shape == (T, 1, N_ACTIONS)


def test_recall_shifts_utilities():
    """A non-zero recall changes the utility code vs the sensory-only case.

    This is the point of the second source: integrating memory actually moves
    the output (the payoff verified end-to-end in EXP-020).
    """
    pfc = Prefrontal(
        concept_dim=CONCEPT, recall_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0
    )
    x = concept_spikes(seed=1)
    u_solo = pfc(x).sum(dim=0)
    u_mem = pfc(x, recall_spikes=recall_spikes(seed=2)).sum(dim=0)
    assert not torch.equal(u_solo, u_mem)


def test_mem_afferent_recorded():
    """The memory afferent current is logged for viz, shape [T,B,n_state]."""
    pfc = Prefrontal(
        concept_dim=CONCEPT, recall_dim=CONCEPT, n_state=100, n_actions=N_ACTIONS,
        num_steps=T, seed=0,
    )
    pfc.enable_recording(True)
    pfc(concept_spikes(seed=1), recall_spikes=recall_spikes(seed=2))
    assert pfc.get_recording("mem_afferent").shape == (T, 1, 100)


def test_rejects_wrong_recall_dim():
    pfc = Prefrontal(
        concept_dim=CONCEPT, recall_dim=CONCEPT, n_actions=N_ACTIONS, num_steps=T, seed=0
    )
    with pytest.raises(ValueError):
        pfc(concept_spikes(seed=1), recall_spikes=torch.rand(T, 1, CONCEPT + 1))
