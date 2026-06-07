"""Tests for the BrainRegion abstract base class (Phase 2, Step 2.1)."""

from __future__ import annotations

import pytest
import torch

from neuromorphic.regions.base_region import BrainRegion


def test_cannot_instantiate_abstract_base():
    """BrainRegion is abstract — instantiating it directly must fail."""
    with pytest.raises(TypeError):
        BrainRegion(name="x", n_neurons=10)


def test_subclass_missing_method_cannot_instantiate():
    """A subclass that omits an abstract method is still abstract."""

    class Incomplete(BrainRegion):
        def forward(self, input_spikes):  # missing reset + get_state
            return input_spikes

    with pytest.raises(TypeError):
        Incomplete(name="x", n_neurons=10)


class _Echo(BrainRegion):
    """Minimal complete region: echoes input, records it per step."""

    def forward(self, input_spikes):
        for t in range(input_spikes.shape[0]):
            self._record("out", input_spikes[t])
        return input_spikes

    def reset(self, batch_size=None, device=None):
        self._state = {"dummy": torch.zeros(1)}

    def get_state(self):
        return dict(getattr(self, "_state", {}))


def test_concrete_subclass_instantiates_and_runs():
    r = _Echo(name="echo", n_neurons=4)
    out = r(torch.zeros(32, 2, 4))
    assert out.shape == (32, 2, 4)


def test_name_and_n_neurons_stored():
    r = _Echo(name="sensory", n_neurons=64)
    assert r.name == "sensory"
    assert r.n_neurons == 64


def test_recording_off_by_default_returns_none():
    """No overhead path: with recording disabled, nothing is captured."""
    r = _Echo(name="echo", n_neurons=4)
    r(torch.zeros(5, 2, 4))
    assert r.get_recording("out") is None


def test_recording_on_stacks_to_TBN():
    """Enabled recording yields the canonical [T, B, N] viz contract."""
    r = _Echo(name="echo", n_neurons=4)
    r.enable_recording(True)
    r(torch.ones(5, 2, 4))
    rec = r.get_recording("out")
    assert rec.shape == (5, 2, 4)
    assert torch.equal(rec, torch.ones(5, 2, 4))


def test_get_recording_all_returns_dict():
    r = _Echo(name="echo", n_neurons=4)
    r.enable_recording(True)
    r(torch.ones(3, 2, 4))
    allrec = r.get_recording()
    assert set(allrec.keys()) == {"out"}
    assert allrec["out"].shape == (3, 2, 4)


def test_clear_recording_empties_buffer():
    r = _Echo(name="echo", n_neurons=4)
    r.enable_recording(True)
    r(torch.ones(3, 2, 4))
    r.clear_recording()
    assert r.get_recording("out") is None


def test_recorded_tensors_are_detached():
    r = _Echo(name="echo", n_neurons=4)
    r.enable_recording(True)
    x = torch.ones(3, 2, 4, requires_grad=True)
    r(x)
    assert r.get_recording("out").requires_grad is False


def test_get_state_returns_dict():
    r = _Echo(name="echo", n_neurons=4)
    r.reset()
    assert isinstance(r.get_state(), dict)
