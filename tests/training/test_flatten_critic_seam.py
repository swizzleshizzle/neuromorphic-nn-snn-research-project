"""The EXP-056 flatten-critic seam, and the proof it means what the spec says.

Same discipline as `test_critic_seam.py` (EXP-053) and `test_encoder_seam.py` (EXP-040): a new
switch must be a strict no-op when off, and must demonstrably change the run when on.

THE LOAD-BEARING TEST HERE IS `test_flattening_a_constant_critic_is_a_strict_no_op`. Flattening
subtracts the critic's OWN episode mean, `v.mean()`. The arm the 2026-08-31 handoff proposed
subtracted the RETURNS' episode mean, `returns.mean()`, which is disqualified because it is
exactly zero on a one-step episode and depth 1 averages 1.22 steps per episode. Those two
implementations are easy to confuse and produce different experiments. With a critic that is
constant within an episode, `v.mean() == v_t` for every t, so the correct implementation is
bit-for-bit inert and the wrong one is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from neuromorphic.training import cube_baseline as cb
from neuromorphic.training.cube_baseline import CubeConfig, run_cube_baseline
from neuromorphic.training.reinforce import critic_fit_terms

from .test_encoder_seam import BASELINE


def _cfg(seed: int, out_dir: Path, **kw) -> CubeConfig:
    """The config `test_critic_seam.py` measures its baseline with."""
    base = dict(arm="regionalized", readout="concept", tag="flatten", depth=3,
                seed=seed, sigma=0.0, episodes=60, curriculum=(1, 2, 3),
                max_depth=4, out_dir=out_dir)
    base.update(kw)
    return CubeConfig(**base)


class ConstantCritic(nn.Module):
    """A critic that ignores the state. Within one episode every `v_t` is the same value, so
    `v.mean()` equals each `v_t` and flattening cannot change a single advantage."""

    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # The critic is read ONE STATE AT A TIME, so `x` arrives unbatched with shape
        # (concept,). Mirroring `nn.Linear`'s trailing-dimension convention keeps the stub
        # shape-compatible with the real critic for both the (concept,) and (N, concept) cases.
        return self.b.expand(*x.shape[:-1], 1)


def _head_weights(out_dir: Path) -> torch.Tensor:
    heads = sorted(Path(out_dir).glob("*_head.pt"))
    assert len(heads) == 1, f"expected one head checkpoint, found {heads}"
    state = torch.load(heads[0], weights_only=True)
    return torch.cat([v.flatten() for _, v in sorted(state.items())])


# --------------------------------------------------------------------------- fast unit tests

def test_critic_fit_terms_centres_each_episode_on_its_own_mean():
    """The within-episode sums must centre on the EPISODE's mean, not on zero and not on a
    running mean. Centring on anything else would let between-episode variation leak into the
    validity gate, and the gate exists precisely to measure within-episode variation."""
    values = torch.tensor([1.0, 3.0, 5.0])       # mean 3, deviations -2, 0, +2 -> ss 8
    returns = torch.tensor([10.0, 20.0, 30.0])   # mean 20, deviations -10, 0, +10 -> ss 200
    fit = critic_fit_terms(values, returns)
    assert fit["critic_within_ss"] == pytest.approx(8.0)
    assert fit["return_within_ss"] == pytest.approx(200.0)
    assert fit["critic_n"] == 3


def test_a_constant_series_has_zero_within_episode_spread():
    """The degenerate case the validity gate is watching for. If `V` is constant within an
    episode there is nothing for flattening to remove, and Claim 1's null would be vacuous."""
    fit = critic_fit_terms(torch.tensor([4.0, 4.0, 4.0]), torch.tensor([1.0, 2.0, 3.0]))
    assert fit["critic_within_ss"] == pytest.approx(0.0)
    assert fit["return_within_ss"] > 0.0


def test_within_rms_pools_over_the_stage_and_is_zero_without_a_critic():
    assert cb.within_rms({"critic_n": 8, "critic_within_ss": 32.0}, "critic_within_ss") == 2.0
    assert cb.within_rms({"critic_n": 0, "critic_within_ss": 0.0}, "critic_within_ss") == 0.0


def test_the_switch_defaults_to_off():
    """A default of True would silently reinterpret every future critic run."""
    assert CubeConfig(arm="regionalized", readout="concept", tag="d", depth=3, seed=0,
                      sigma=0.0, episodes=1, out_dir=Path(".")).flatten_critic is False


# --------------------------------------------------------------------------- run-level seam

@pytest.mark.slow
def test_flatten_is_inert_when_there_is_no_critic(tmp_path):
    """Without a critic there is no `V` to flatten, so the flag must not touch the EMA path.
    If this fails, every cube record from EXP-029 onward has stopped being comparable."""
    seed = sorted(BASELINE)[0]
    rec = run_cube_baseline(_cfg(seed, tmp_path, flatten_critic=True))
    for field, expected in BASELINE[seed].items():
        assert rec[field] == pytest.approx(expected, abs=1e-6), (
            f"seed {seed} field {field}: {rec[field]} != pre-change {expected}. "
            "flatten_critic is NOT inert on the no-critic path."
        )


@pytest.mark.slow
def test_flattening_a_constant_critic_is_a_strict_no_op(tmp_path, monkeypatch):
    """THE DISCRIMINATING TEST. A critic that is constant within an episode has
    `v.mean() == v_t` for every t, so subtracting the mean and subtracting the prediction are
    the same arithmetic and the two runs must agree bit for bit.

    This FAILS against an implementation that subtracts `returns.mean()` instead, which is the
    disqualified batch-mean baseline and a genuinely different experiment.
    """
    monkeypatch.setattr(cb, "make_critic", lambda brain: ConstantCritic())
    seed = sorted(BASELINE)[0]
    a = tmp_path / "plain"
    b = tmp_path / "flat"
    rec_a = run_cube_baseline(_cfg(seed, a, critic_lr=1e-2, tag="const_plain"))
    rec_b = run_cube_baseline(_cfg(seed, b, critic_lr=1e-2, tag="const_flat",
                                   flatten_critic=True))
    for field in ("success_rate", "train_success_rate", "mean_steps", "optimality",
                  "revisit_rate", "generalisation_gap"):
        assert rec_a[field] == pytest.approx(rec_b[field], abs=1e-9), (
            f"{field} moved with a CONSTANT critic: {rec_a[field]} vs {rec_b[field]}. "
            "flatten_critic is subtracting something other than the critic's episode mean."
        )
    # NOT bitwise equality: `v.mean()` over identical floats is not bitwise the element value,
    # because the sum-then-divide reassociates. MEASURED thresholds, prototyped before being
    # written here: a constant critic differs by 1.34e-07 and a real one by 1.80e-01, six
    # orders apart. 1e-5 sits ~75x above the noise and ~4 orders below a real effect.
    drift = (_head_weights(a) - _head_weights(b)).abs().max().item()
    assert drift < 1e-5, (
        f"head weights moved by {drift:.3e} under a CONSTANT critic, far above the {1.34e-07:.2e} "
        "float-reassociation floor. The flattened path is subtracting something other than the "
        "critic's episode mean, most likely returns.mean()."
    )


@pytest.mark.slow
def test_flattening_a_real_critic_actually_changes_the_run(tmp_path, monkeypatch):
    """The complement. Without this, every assertion above is satisfied by a switch that does
    nothing, which is the defect class CLAUDE.md's test-strength rule exists for."""
    seed = sorted(BASELINE)[0]
    a = tmp_path / "plain"
    b = tmp_path / "flat"
    run_cube_baseline(_cfg(seed, a, critic_lr=1e-2, tag="real_plain"))
    run_cube_baseline(_cfg(seed, b, critic_lr=1e-2, tag="real_flat", flatten_critic=True))
    # A MEASURED threshold, not `not torch.equal`, which would be satisfied by the 1e-07 of
    # float noise that the constant-critic case produces and would therefore pass against an
    # inert switch. Prototyped at 1.80e-01; 1e-2 keeps ~18x margin below it and three orders
    # above the noise floor.
    delta = (_head_weights(a) - _head_weights(b)).abs().max().item()
    assert delta > 1e-2, (
        f"flattening a real state-dependent critic moved head weights by only {delta:.3e}, "
        "which is near the float-noise floor. The switch is effectively inert and the arm "
        "would be vacuous."
    )


@pytest.mark.slow
def test_the_validity_gate_is_recorded_and_v_really_does_vary_within_an_episode(tmp_path):
    """Claim 3. If `V` barely varies within an episode then flattening removes nothing and a
    null on Claim 1 is uninterpretable, so the numbers that decide that must exist in the
    record and must be non-degenerate for a real critic."""
    seed = sorted(BASELINE)[0]
    rec = run_cube_baseline(_cfg(seed, tmp_path, critic_lr=1e-2, tag="gate"))
    stages = [s for s in rec["stage_trace"] if s.get("critic_n")]
    assert stages, "no stage recorded critic terms"
    for st in stages:
        assert "critic_within_rms" in st and "return_within_rms" in st
        assert st["return_within_rms"] > 0.0, (
            f"depth {st['depth']}: returns have no within-episode spread, so the gate's "
            "denominator is degenerate"
        )
    assert max(s["critic_within_rms"] for s in stages) > 0.0, (
        "V is constant within every episode at every depth, so flattening removes nothing"
    )
