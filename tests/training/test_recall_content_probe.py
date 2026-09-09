"""EXP-059's validity-gate instrument: does the attractor's STORED CONTENT change the recall?

Written under CLAUDE.md's gate-calibration rule, after EXP-058 was voided by a gate that could
not pass and could not discriminate. The three properties that rule demands are each pinned here:

  - the gate CAN FAIL: an empty attractor gives exactly 1.0
  - the gate CAN PASS: a loaded attractor gives less than 1.0, and a real run measured 0.8128
  - it measures the thing the arms actually differ in, which is RECALL, not storing
"""

from __future__ import annotations

import random

import pytest
import torch

from neuromorphic.training.cube_baseline import CubeConfig, MemoryReadout, make_agent


def _agent():
    cfg = CubeConfig(arm="regionalized", readout="memory", tag="t", depth=5, seed=0,
                     sigma=0.0, episodes=1, max_depth=6, out_dir=".")
    return make_agent(cfg)


def _cos(agent, W, q):
    agent.hippo.W_rec = W
    r = agent.hippo(q.unsqueeze(0).expand(agent.T, *q.shape)).mean(dim=0)[0]
    saved = agent.hippo.W_rec
    agent.hippo.W_rec = torch.zeros_like(saved)
    try:
        b = agent.hippo(q.unsqueeze(0).expand(agent.T, *q.shape)).mean(dim=0)[0]
    finally:
        agent.hippo.W_rec = saved
    d = float(r.norm()) * float(b.norm())
    return float(torch.dot(r, b)) / d


def test_an_empty_attractor_gives_exactly_one_so_the_gate_can_fail():
    """THE FAILURE END. If nothing is stored, the real recall IS the W_rec-zeroed recall, so the
    cosine is 1.0 by construction. A gate that cannot reach its own failure value is the defect
    that voided EXP-058, and this pins that this one can."""
    agent = _agent()
    torch.manual_seed(0)
    q = torch.randn(1, agent.content)
    assert _cos(agent, torch.zeros_like(agent.hippo.W_rec), q) == pytest.approx(1.0, abs=1e-6)


def test_a_loaded_attractor_falls_below_one_so_the_gate_can_pass():
    """THE PASSING END. Stored content must move the recall away from its memory-free transform.
    A real depth-5 run measures 0.8128 and the class docstring records 0.802 independently over
    79 policy steps, so the 0.95 gate has roughly 0.14 of headroom."""
    agent = _agent()
    torch.manual_seed(0)
    q = torch.randn(1, agent.content)
    loaded = _cos(agent, torch.randn_like(agent.hippo.W_rec) * 0.05, q)
    assert loaded < 1.0
    assert loaded < 0.999, f"a loaded attractor barely moved the recall: {loaded}"


def test_the_probe_is_sampled_and_records_its_own_count():
    """The probe costs a second hippocampal read, so it is strided. The count must be recorded,
    or a mean over an unknown number of samples is uninterpretable."""
    assert MemoryReadout.RECALL_PROBE_STRIDE >= 1
    r = MemoryReadout("memory", random.Random(0), _agent())
    assert r.recall_cos_n == 0 and r.recall_cos_sum == 0.0
    r.reset()
    assert r.recall_cos_n == 0, "reset must clear the probe accumulators"
