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


def test_store_accumulates_rather_than_overwriting():
    """The defect this fixes: store() used to assign, so only one pattern survived."""
    h = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    g = torch.Generator().manual_seed(0)
    a = torch.rand(1, CONTENT, generator=g)
    b = torch.rand(1, CONTENT, generator=g)

    h.store(a)
    w_after_a = h.W_rec.clone()
    h.store(b)
    w_after_b = h.W_rec.clone()

    # B's contribution in isolation, from a fresh region with the same init
    h2 = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    h2.store(b)
    w_b_only = h2.W_rec.clone()

    assert not torch.allclose(w_after_b, w_b_only), "W_rec equals B alone: store still overwrites"
    assert torch.allclose(w_after_b, w_after_a + w_b_only, atol=1e-6)
    assert h.n_stored == 2


def test_clear_forgets_everything():
    h = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    h.store(torch.rand(1, CONTENT))
    assert torch.count_nonzero(h.W_rec) > 0
    h.clear()
    assert torch.count_nonzero(h.W_rec) == 0
    assert h.n_stored == 0


def test_recall_discriminates_between_stored_states():
    """Replaces the old not-equal assertion. Measured 0.912 at 4 patterns after the fix,
    versus 0.998 before it, so 0.95 is a threshold the broken code cannot pass."""
    import torch.nn.functional as F

    h = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    g = torch.Generator().manual_seed(0)
    contents = [torch.rand(1, CONTENT, generator=g) for _ in range(4)]
    for c in contents:
        h.store(c)

    recalls = torch.stack([h(c.unsqueeze(0).expand(T, 1, CONTENT)).mean(dim=0)[0] for c in contents])
    sims = [
        float(F.cosine_similarity(recalls[i], recalls[j], dim=0))
        for i in range(4) for j in range(i + 1, 4)
    ]
    assert sum(sims) / len(sims) < 0.95


def test_familiarity_separates_visited_from_novel():
    """Measured separation is +1.4 to +1.8 across loads; 0.3 is a conservative floor."""
    h = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    g = torch.Generator().manual_seed(0)
    visited = [torch.rand(1, CONTENT, generator=g) for _ in range(8)]
    novel = [torch.rand(1, CONTENT, generator=g) for _ in range(8)]
    for c in visited:
        h.store(c)

    fam_v = torch.cat([h.familiarity(c) for c in visited]).mean()
    fam_n = torch.cat([h.familiarity(c) for c in novel]).mean()
    assert float(fam_v - fam_n) > 0.3


def test_familiarity_is_zero_with_empty_memory():
    h = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=0)
    fam = h.familiarity(torch.rand(3, CONTENT))
    assert fam.shape == (3,)
    assert torch.allclose(fam, torch.zeros(3))


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
