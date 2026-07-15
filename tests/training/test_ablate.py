import torch
import torch.nn as nn
from neuromorphic.analysis.ablate import AblationSpec, AblatedConcept


def _head(width=8, actions=4):
    torch.manual_seed(0)
    return nn.Linear(width, actions)


def test_dose_zero_is_identity_gaussian():
    head = _head()
    x = torch.randn(8)
    wrapped = AblatedConcept(head, AblationSpec("gaussian", dose=0.0), width=8)
    assert torch.equal(wrapped(x), head(x))


def test_none_spec_is_identity():
    head = _head()
    x = torch.randn(8)
    wrapped = AblatedConcept(head, None, width=8)
    assert torch.equal(wrapped(x), head(x))


def test_gaussian_perturbs_input():
    head = _head()
    x = torch.zeros(8)
    wrapped = AblatedConcept(head, AblationSpec("gaussian", dose=0.5, seed=1), width=8)
    # With a nonzero dose the effective input differs from the clean one, so output moves.
    assert not torch.equal(wrapped(x), head(x))


def test_unitdrop_random_zeros_expected_count():
    head = nn.Identity()
    x = torch.ones(8)
    wrapped = AblatedConcept(head, AblationSpec("unitdrop", dose=0.25, mode="random", seed=3), width=8)
    out = wrapped(x)
    assert int((out == 0).sum()) == 2  # round(0.25 * 8)


def test_unitdrop_top_drops_most_important_units():
    head = nn.Identity()
    x = torch.ones(8)
    order = [7, 6, 5, 4, 3, 2, 1, 0]  # most-important-first
    wrapped = AblatedConcept(head, AblationSpec("unitdrop", dose=0.25, mode="top"), width=8, order=order)
    out = wrapped(x)
    zeroed = set(int(i) for i in torch.nonzero(out == 0).flatten())
    assert zeroed == {7, 6}


def test_gaussian_is_reproducible_for_fixed_seed():
    head = nn.Identity()
    x = torch.zeros(8)
    a = AblatedConcept(head, AblationSpec("gaussian", dose=0.5, seed=5), width=8)(x)
    b = AblatedConcept(head, AblationSpec("gaussian", dose=0.5, seed=5), width=8)(x)
    assert torch.equal(a, b)
