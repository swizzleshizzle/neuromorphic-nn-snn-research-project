"""EXP-074 wide readout. Each docstring names the bug it catches."""

from __future__ import annotations

import random

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.cube_baseline import CubeConfig, make_agent

GOLDEN_J = [0.807378, 0.805297, 0.761739, 0.811812, 0.689316, 0.73395]
GOLDEN_HEAD_SUM = 10.109715


def _judge(seed=0, readout="concept"):
    torch.manual_seed(seed)
    return vi.judge_from_brain(make_agent(CubeConfig(arm="regionalized", seed=seed)),
                               readout=readout)


def _states(n=6):
    return vi.random_walk_states(n, 5, random.Random(1), N_ACTIONS, exclude=set())


def test_concept_readout_is_byte_identical_to_exp073():
    """Catches any drift in the concept path or head construction order. Arms A and B here
    must BE EXP-073's arms (Gate 0(c)); a changed init or forward fails this before the laptop
    spends hours finding out."""
    j = _judge()
    with torch.no_grad():
        got = [round(v, 6) for v in j(_states(), torch.Generator().manual_seed(2)).tolist()]
    assert got == GOLDEN_J
    assert round(float(j.head[0].weight.sum()), 6) == GOLDEN_HEAD_SUM


def test_wide_features_are_concept_plus_hidden_from_the_same_pass():
    """Catches hidden rates taken from a SECOND encoding (a separate encoder_fn call): the
    first 64 columns would then not equal the concept under the same generator seed."""
    j = _judge(readout="wide")
    s = _states()
    with torch.no_grad():
        wide = j.features(s, torch.Generator().manual_seed(5))
        concept = j.concept(s, torch.Generator().manual_seed(5))
    assert wide.shape == (len(s), 64 + 128)
    assert torch.equal(wide[:, :64], concept)
    hidden = wide[:, 64:]
    # Independent reference: ONE encoding from the same seed, hidden rates accumulated by hand.
    # A hidden block drawn from a second encoder_fn call sees an advanced generator and differs.
    obs = torch.as_tensor(__import__("numpy").array(s), dtype=torch.long)
    with torch.no_grad():
        spikes = j.encoder_fn(obs, T=j.T, generator=torch.Generator().manual_seed(5))
        sc = j.sensory
        sc.reset()
        mem1 = sc.mem1
        rates = []
        for t in range(spikes.shape[0]):
            spk1, mem1 = sc.lif1(sc.fc1(spikes[t]), mem1)
            rates.append(spk1)
    assert torch.equal(hidden, torch.stack(rates).mean(dim=0))
    assert float(hidden.min()) >= 0.0 and float(hidden.max()) <= 1.0
    assert float(hidden.sum()) > 0.0


def test_wide_head_reads_192_inputs_and_concept_head_reads_64():
    assert _judge(readout="wide").head[0].in_features == 192
    assert _judge().head[0].in_features == 64


def test_unknown_readout_is_refused():
    with pytest.raises(ValueError, match="readout"):
        _judge(readout="everything")


def test_wide_hidden_path_carries_gradient_to_fc1():
    """Catches hidden rates detached from the graph. With the concept columns' head weights
    zeroed, the ONLY route from the loss to fc1 is through the hidden rates. Live graph: no
    no_grad anywhere in this test."""
    j = _judge(readout="wide")
    with torch.no_grad():
        j.head[0].weight[:, :64] = 0.0
    out = j(_states(16), torch.Generator().manual_seed(0))
    out.sum().backward()
    assert j.sensory.fc1.weight.grad is not None
    assert float(j.sensory.fc1.weight.grad.abs().sum()) > 0.0


def test_arm_w_trains_the_encoder_like_arm_a():
    """Catches make_optimizer rejecting W, or building W without the encoder group."""
    j = _judge(readout="wide")
    opt = vi.make_optimizer(j, "W")
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-4, 1e-3]
    jt = vi.sync_target(j)
    w0 = j.sensory.fc1.weight.detach().clone()
    vi.train_step(j, jt, opt, _states(16), N_ACTIONS, torch.Generator().manual_seed(0))
    assert not torch.equal(j.sensory.fc1.weight, w0)


def test_sync_target_copies_the_readout():
    """Catches a frozen target that reads the concept while the judge reads wide (targets would
    come from a different function than the one being trained)."""
    j = _judge(readout="wide")
    jt = vi.sync_target(j)
    assert jt.readout == "wide"
    s = _states()
    with torch.no_grad():
        assert torch.equal(jt(s, torch.Generator().manual_seed(3)),
                           j(s, torch.Generator().manual_seed(3)))
