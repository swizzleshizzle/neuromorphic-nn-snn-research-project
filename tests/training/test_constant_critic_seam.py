"""The EXP-057 constant-critic seam.

A state-blind baseline is only a control for CALIBRATION if it really cannot see the state and
really is fitted. Both halves are tested here, plus the Claim 4 validity gate that the
aggregator will refuse the experiment without.

The distinction this file exists to protect: a constant critic is NOT the per-episode batch
mean. That arm forms `G_t - mean(G)`, exactly zero on a one-step episode, and depth 1 averages
1.22 steps per episode; it was disqualified before running. A learned scalar is not the
episode's own mean.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from neuromorphic.training.cube_baseline import CubeConfig, run_cube_baseline
from neuromorphic.training.reinforce import (
    ConstantCritic,
    make_constant_critic,
    make_critic,
)

from .test_encoder_seam import BASELINE

CONTENT = 64


class _Brain:
    content = CONTENT


def _cfg(seed: int, out_dir: Path, **kw) -> CubeConfig:
    base = dict(arm="regionalized", readout="concept", tag="constcritic", depth=3,
                seed=seed, sigma=0.0, episodes=60, curriculum=(1, 2, 3),
                max_depth=4, out_dir=out_dir)
    base.update(kw)
    return CubeConfig(**base)


# --------------------------------------------------------------------------- fast unit tests

def test_it_has_exactly_one_parameter_against_the_state_critic_s_65():
    """The whole claim is 'differs from arm B in exactly one way'. A stray parameter, or a
    disguised Linear, would make that sentence false."""
    const = make_constant_critic(_Brain())
    state = make_critic(_Brain())
    assert sum(p.numel() for p in const.parameters()) == 1
    assert sum(p.numel() for p in state.parameters()) == CONTENT + 1 == 65


def test_it_cannot_see_the_state():
    """THE POINT OF THE ARM. Two very different inputs must produce the identical value. If
    this fails the arm is not state-blind and every claim in the spec is void."""
    c = ConstantCritic(CONTENT)
    a = c(torch.zeros(CONTENT))
    b = c(torch.randn(CONTENT) * 10.0)
    assert torch.equal(a, b), f"output moved with the input: {a} vs {b}"
    batch = c(torch.randn(7, CONTENT))
    assert torch.equal(batch, batch[0].expand_as(batch)), "batched outputs are not identical"


def test_it_mirrors_nn_linear_s_shape_convention():
    """The read site does `critic(features).squeeze(-1)` with UNBATCHED features. Getting this
    wrong raises a broadcast error deep in the advantage computation, which is how the first
    draft of EXP-056's test stub failed."""
    c, lin = ConstantCritic(CONTENT), make_critic(_Brain())
    for shape in ((CONTENT,), (5, CONTENT)):
        x = torch.randn(*shape)
        assert c(x).shape == lin(x).shape


def test_initialisation_matches_the_linear_bias_distribution():
    """Init must not be a SECOND difference against arm B. nn.Linear draws its bias from
    U(-1/sqrt(fan_in), +1/sqrt(fan_in)); so does this. Checked over many draws rather than one,
    and against the bound rather than a moment, because a single draw proves nothing."""
    bound = 1.0 / math.sqrt(CONTENT)
    vals = [float(ConstantCritic(CONTENT).value.detach()) for _ in range(400)]
    assert all(abs(v) <= bound + 1e-9 for v in vals), "a draw fell outside the Linear bias bound"
    assert max(vals) > 0.6 * bound and min(vals) < -0.6 * bound, (
        "draws are not spread across the bound; this does not look uniform"
    )


def test_the_switch_defaults_to_off():
    assert CubeConfig(arm="regionalized", readout="concept", tag="d", depth=3, seed=0,
                      sigma=0.0, episodes=1, out_dir=Path(".")).constant_critic is False


# --------------------------------------------------------------------------- run-level seam

@pytest.mark.slow
def test_constant_critic_is_inert_when_there_is_no_critic(tmp_path):
    """Without `critic_lr` there is no critic to replace, so the flag must not touch the EMA
    path. If this fails, every cube record from EXP-029 onward has stopped being comparable."""
    seed = sorted(BASELINE)[0]
    rec = run_cube_baseline(_cfg(seed, tmp_path, constant_critic=True))
    for field, expected in BASELINE[seed].items():
        assert rec[field] == pytest.approx(expected, abs=1e-6), (
            f"seed {seed} field {field}: {rec[field]} != pre-change {expected}. "
            "constant_critic is NOT inert on the no-critic path."
        )


@pytest.mark.slow
def test_the_claim_4_gate_holds_within_episode_spread_at_the_float_floor(tmp_path):
    """EXP-057's VALIDITY GATE, reusing EXP-056's instrumentation. Every timestep of an episode
    reads the same scalar, so the pooled within-episode RMS of V must be exactly 0.0 at every
    stage. A non-zero value means the arm is not state-blind, and the aggregator voids the
    experiment on it."""
    seed = sorted(BASELINE)[0]
    rec = run_cube_baseline(_cfg(seed, tmp_path, critic_lr=1e-2, constant_critic=True))
    stages = [s for s in rec["stage_trace"] if s.get("critic_n")]
    assert stages, "no stage recorded critic terms"
    for st in stages:
        # NOT exactly 0.0: `v.mean()` over identical floats reassociates and leaves ~1e-9. This
        # smoke run measures 6.8e-10. Arm B's real within-episode RMS is 0.65 to 2.20, so 1e-6
        # is three orders above the noise and six below any genuine variation. The spec's
        # Claim 4 was amended to this threshold before any experimental number existed.
        assert st["critic_within_rms"] < 1e-6, (
            f"depth {st['depth']}: critic_within_rms is {st['critic_within_rms']:.3e}, which is "
            "far above the float-reassociation floor. The constant critic is varying within an "
            "episode, so it is not state-blind."
        )
        # The denominator must still be alive, or the gate is trivially satisfied.
        assert st["return_within_rms"] > 0.0


@pytest.mark.slow
def test_a_constant_critic_run_differs_from_the_state_critic_run(tmp_path):
    """Otherwise every assertion above is satisfied by a switch that does nothing. Threshold is
    measured against EXP-056's prototyped scale: a real change to the advantage moved head
    weights by 1.80e-01, while float reassociation alone moved them by 1.34e-07."""
    seed = sorted(BASELINE)[0]
    a, b = tmp_path / "state", tmp_path / "const"
    run_cube_baseline(_cfg(seed, a, critic_lr=1e-2, tag="state"))
    run_cube_baseline(_cfg(seed, b, critic_lr=1e-2, tag="const", constant_critic=True))

    def heads(d):
        h = sorted(Path(d).glob("*_head.pt"))
        assert len(h) == 1, f"expected one head checkpoint, found {h}"
        st = torch.load(h[0], weights_only=True)
        return torch.cat([v.flatten() for _, v in sorted(st.items())])

    delta = (heads(a) - heads(b)).abs().max().item()
    assert delta > 1e-2, (
        f"removing the state input moved head weights by only {delta:.3e}, near the float-noise "
        "floor. The switch is inert and the arm would be vacuous."
    )
