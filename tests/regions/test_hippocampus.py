"""Tests for the Hippocampus attractor memory (Phase 2, build-order step 5)."""

from __future__ import annotations

import pytest
import torch

from neuromorphic.regions import BrainRegion
from neuromorphic.regions.hippocampus import Hippocampus

T = 32
CONTENT = 64
N_NEURONS = 150
STORE_STEPS = 6


def content_code(seed: int = 0, B: int = 1) -> torch.Tensor:
    """A PFC-like content code [B, CONTENT] (continuous drive)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(B, CONTENT, generator=gen)


def store_then_delay(content: torch.Tensor, store_steps: int = STORE_STEPS) -> torch.Tensor:
    """Input sequence: content present for store_steps (pathway-3 store gate open), then
    zero (gate closed → delay)."""
    B = content.shape[0]
    seq = torch.zeros(T, B, CONTENT)
    seq[:store_steps] = content.unsqueeze(0)
    return seq


def test_is_a_brain_region():
    assert isinstance(Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0), BrainRegion)


def test_forward_output_shape():
    hippo = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    hippo.store(content_code())
    out = hippo(store_then_delay(content_code()))
    assert out.shape == (T, 1, CONTENT)


def test_store_imprints_recurrent_weights():
    """W_rec is zero before storing, non-zero after."""
    hippo = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    assert torch.count_nonzero(hippo.W_rec) == 0
    hippo.store(content_code())
    assert torch.count_nonzero(hippo.W_rec) > 0


def test_attractor_holds_pattern_through_delay():
    """The stored pattern persists during the no-input delay (attractor-persistence gate)."""
    hippo = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    p = hippo.store(content_code())
    hippo.enable_recording(True)
    hippo(store_then_delay(content_code()))
    pop = hippo.get_recording("population")  # [T, 1, N_NEURONS]
    late = pop[T // 2 :].float().mean(dim=0)[0]  # firing rate per neuron in deep delay
    pattern_rate = late[p.bool()].mean()
    nonpattern_rate = late[~p.bool()].mean()
    assert pop[T // 2 :].sum() > 0                      # not dead
    assert pattern_rate > 0.5                            # stored neurons stay active
    assert pattern_rate > 3 * nonpattern_rate            # clean separation


def test_recall_is_content_specific():
    """Different stored patterns yield different recall read-outs during the hold."""
    hippo = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    hippo.store(content_code(seed=1))
    recall_a = hippo(store_then_delay(content_code(seed=1)))[T // 2 :].sum(dim=0)
    hippo.store(content_code(seed=2))
    recall_b = hippo(store_then_delay(content_code(seed=2)))[T // 2 :].sum(dim=0)
    assert not torch.equal(recall_a, recall_b)


def test_forward_is_deterministic():
    hippo_a = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    hippo_b = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    hippo_a.store(content_code(seed=3))
    hippo_b.store(content_code(seed=3))
    seq = store_then_delay(content_code(seed=3))
    assert torch.equal(hippo_a(seq), hippo_b(seq))


def test_recording_exposes_population():
    hippo = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    hippo.store(content_code())
    hippo.enable_recording(True)
    hippo(store_then_delay(content_code()))
    assert hippo.get_recording("population").shape == (T, 1, N_NEURONS)


def test_rejects_non_3d_input():
    hippo = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    with pytest.raises(ValueError):
        hippo(torch.rand(T, CONTENT))


def test_rejects_wrong_content_dim():
    hippo = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    with pytest.raises(ValueError):
        hippo(torch.rand(T, 1, CONTENT + 1))
