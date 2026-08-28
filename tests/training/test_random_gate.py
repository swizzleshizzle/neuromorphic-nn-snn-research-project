"""EXP-053 arm R: the rate-matched random gate, which is what makes arm G interpretable.

A gated arm also does FEWER encoder updates. Without a control that does the same NUMBER of
updates on different episodes, "G matches its always-on control" cannot distinguish "the
signal picked the right episodes" from "half the updates were redundant" - and EXP-049
already found the encoder gain does not compound, which makes the second reading likelier.

This is the EXP-030 lesson applied in advance: ask what a control holds fixed besides the
thing you named.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from neuromorphic.training.cube_baseline import (
    CubeConfig,
    RandomGate,
    run_cube_baseline,
)


def _cfg(out_dir: Path, **kw) -> CubeConfig:
    base = dict(arm="regionalized", readout="concept", tag="rgate", depth=3,
                seed=0, sigma=0.0, episodes=60, curriculum=(1, 2, 3),
                max_depth=4, out_dir=out_dir)
    base.update(kw)
    return CubeConfig(**base)


def test_random_gate_hits_its_rate():
    """Over 10,000 draws a rate of 0.37 must land within 0.02. Binomial sd here is 0.0048,
    so 0.02 is about four sd - tight enough to catch an off-by-one or an inverted comparison,
    loose enough not to flake."""
    gate = RandomGate(0.37, random.Random(0))
    opens = sum(gate(0.0, 0.0) for _ in range(10_000))
    assert abs(opens / 10_000 - 0.37) < 0.02


def test_random_gate_ignores_the_dopamine_value():
    """It must be blind to the signal - that is the entire point of the control."""
    high = RandomGate(0.5, random.Random(7))
    low = RandomGate(0.5, random.Random(7))
    assert [high(1000.0, 0.0) for _ in range(50)] == [low(-1000.0, 0.0) for _ in range(50)]


def test_random_gate_does_not_touch_the_torch_generator():
    """RandomGate must draw only from its own `random.Random` stream, never from torch's.

    The real risk is not a generator explicitly handed to the gate (nothing hands it one) -
    it is a gate that calls something like `torch.rand(...)` internally, which perturbs the
    GLOBAL torch RNG that `dist.sample()` draws on elsewhere in the same process. That would
    make arm R's action sampling and scrambles diverge from arm G's, so the two arms would
    differ in more than which episodes updated.

    Snapshots the GLOBAL torch RNG state (not an unused local Generator) before and after
    many gate calls and asserts bit-equality. Verified this fails if the gate is made to call
    `torch.rand(1)` internally (see the test-strength note in CLAUDE.md)."""
    torch.manual_seed(0)
    before = torch.get_rng_state().clone()
    gate = RandomGate(0.5, random.Random(0))
    for _ in range(100):
        gate(0.0, 0.0)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.slow
def test_run_uses_the_configured_rate_for_its_seed(tmp_path):
    """Spec section 4.2 item 6: the realized rate must be within a tolerance of the rate it
    was given. At 60 episodes and p=0.5 the binomial sd is 0.065 (sqrt(0.5*0.5/60)), so
    `abs(rate - 0.5) < 0.25` is about a 4sd threshold - tight enough to catch an off-by-one
    or an inverted comparison, loose enough not to flake at n=60. Do not tighten further."""
    rec = run_cube_baseline(_cfg(tmp_path, encoder_lr=1e-3, plasticity_gate="random",
                                 gate_rate_by_seed=((0, 0.5),)))
    assert abs(rec["gate_rate"] - 0.5) < 0.25, (
        f"gate_rate {rec['gate_rate']} is too far from the configured 0.5 over 60 episodes "
        "(binomial sd 0.065, so this is roughly a 4sd tolerance)."
    )


def test_random_gate_refuses_a_missing_seed(tmp_path):
    """Silently defaulting to 1.0 would turn arm R into arm G's always-on control and the
    attribution claim would compare the wrong things."""
    with pytest.raises(ValueError, match="no gate rate for seed 0"):
        run_cube_baseline(_cfg(tmp_path, encoder_lr=1e-3, plasticity_gate="random",
                               gate_rate_by_seed=((3, 0.5),)))
