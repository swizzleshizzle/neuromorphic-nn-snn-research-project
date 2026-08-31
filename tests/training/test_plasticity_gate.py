"""EXP-053: gating encoder plasticity on NeuromodBus, and the Adam trap it has to avoid.

`NeuromodBus.learning_enabled` has existed since L11 and NOTHING has ever read it.
`Brain.learn()` has existed just as long and has no caller in the cube training loop. This
is the increment where both become load-bearing, so the tests have to prove the gate really
withholds updates rather than merely appearing to.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from neuromorphic.brain import Brain
from neuromorphic.encoders import encode_cube
from neuromorphic.envs.cube import scramble
from neuromorphic.training.cube_baseline import (
    CubeConfig,
    DopamineGate,
    ShellCubeEnv,
    run_cube_baseline,
)
from neuromorphic.training.reinforce import make_policy_head, train_episode


def _cfg(out_dir: Path, **kw) -> CubeConfig:
    base = dict(arm="regionalized", readout="concept", tag="gate", depth=3,
                seed=0, sigma=0.0, episodes=60, curriculum=(1, 2, 3),
                max_depth=4, out_dir=out_dir)
    base.update(kw)
    return CubeConfig(**base)


def test_zeroing_a_gradient_does_not_stop_adam():
    """The trap this whole design avoids, demonstrated on plain torch.

    If this ever FAILS, Adam's semantics changed and the separate-optimizer machinery below
    could be simplified. Until then it is the reason the gate is not implemented by zeroing
    the encoder parameter group's gradients.
    """
    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.Adam([p], lr=1e-2)

    p.grad = torch.ones(3)          # one real update, which builds momentum
    opt.step()
    after_first = p.detach().clone()

    p.grad = torch.zeros(3)         # "gated off" by zeroing the gradient
    opt.step()

    moved = (p.detach() - after_first).abs().max().item()
    assert moved > 1e-4, (
        "Adam left the parameter alone on a zero gradient, which would make gating by "
        "zero-grad viable. Re-read the design before simplifying anything."
    )


def test_the_gate_gates_exactly():
    """The encoder must move on exactly the gated episodes and be bit-identical otherwise.

    A `count_nonzero(delta) > 0` check would pass either way. That assertion is what hid
    `Hippocampus.store()` assigning instead of accumulating, and it is forbidden here.
    """
    torch.manual_seed(0)
    # `obs_width=24` (raw facelets) is required alongside `n_obs=144` (24 facelets x 6
    # colors, one-hot) for a cube encoder; the default `obs_width=4` is the gridworld shape
    # and raises on a 24-facelet observation. See `CUBE_OBS_WIDTH` in cube_baseline.py.
    brain = Brain(encoder=encode_cube, n_obs=144, n_actions=6, obs_width=24)
    head = make_policy_head(brain)
    # `ShellCubeEnv` draws its start state from a fixed pool (see cube_baseline.py); it needs
    # a real pool, not `None` - built here from `scramble` at the same depth so the test still
    # exercises a shallow-scrambled cube.
    pool = [scramble(2, random.Random(0))]
    env = ShellCubeEnv(pool, random.Random(0), scramble_depth=2, max_steps=4)

    head_opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    enc_opt = torch.optim.Adam(brain.sensory.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)

    scripted = [True, False, True, False, False]
    calls = iter(scripted)
    observed = []
    for expected_open in scripted:
        before = brain.sensory.fc1.weight.detach().clone()
        stats = train_episode(brain, head, env, head_opt, gamma=0.99, baseline=0.0,
                              generator=gen, max_steps=4, grad_brain=True,
                              encoder_optimizer=enc_opt,
                              gate_fn=lambda _m, _b: next(calls))
        drift = (brain.sensory.fc1.weight.detach() - before).abs().max().item()
        observed.append(drift)
        assert stats["gate_open"] is expected_open
        if expected_open:
            assert drift > 0.0, "the gate was open and the encoder did not move"
        else:
            assert drift == 0.0, (
                f"the gate was CLOSED and fc1.weight still moved by {drift:.3e}. The gate is "
                "almost certainly zeroing gradients inside a shared Adam instead of skipping "
                "a separate optimizer's step."
            )
    assert any(d > 0.0 for d in observed), "no episode moved the encoder at all"


def test_dopamine_gate_uses_the_bus_and_the_running_median():
    """The bus must be WRITTEN and READ, and the threshold must track the median."""
    torch.manual_seed(0)
    brain = Brain(encoder=encode_cube, n_obs=144, n_actions=6)
    gate = DopamineGate(brain, warmup=3)

    # Warmup: always open, whatever the value.
    assert gate(-100.0, 0.0) is True
    assert gate(-100.0, 0.0) is True
    assert gate(-100.0, 0.0) is True
    assert brain.bus.dopamine == pytest.approx(-100.0), (
        "Brain.learn was not called, so the bus never saw the reward-prediction error"
    )

    # History is now [-100, -100, -100]; a value above the median must open the gate and a
    # value below it must close it.
    assert gate(50.0, 0.0) is True
    assert gate(-200.0, 0.0) is False
    assert brain.bus.learning_threshold == pytest.approx(-100.0)


@pytest.mark.slow
def test_always_open_gate_reproduces_the_single_optimizer_run(tmp_path):
    """THE ADAM TRAP TEST. Splitting one two-group Adam must be mathematically neutral.

    Adam state is per-parameter, so a head optimizer plus an encoder optimizer is identical
    to one optimizer with two groups WHILE THE GATE IS ALWAYS OPEN. If this fails, arm G
    differs from its EXP-047 control in two ways at once and the paired delta measures
    something nobody chose.
    """
    ungated = run_cube_baseline(_cfg(tmp_path, encoder_lr=1e-3, tag="gate_off"))
    always = run_cube_baseline(
        _cfg(tmp_path, encoder_lr=1e-3, plasticity_gate="always", tag="gate_always"))

    for field in ("success_rate", "mean_train_entropy", "greedy_modal_action_frac",
                  "revisit_rate", "optimality"):
        assert always[field] == pytest.approx(ungated[field], abs=1e-6), (
            f"{field}: {always[field]} != {ungated[field]}. Splitting the optimizer is NOT "
            "neutral, so the gate arm carries a second uncontrolled difference."
        )
    assert always["gate_rate"] == pytest.approx(1.0)


def test_gate_requires_encoder_lr(tmp_path):
    """Gating plasticity that does not exist would silently do nothing. Refuse instead."""
    with pytest.raises(ValueError, match="plasticity_gate requires encoder_lr"):
        run_cube_baseline(_cfg(tmp_path, plasticity_gate="dopamine"))


def test_unknown_gate_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown plasticity_gate"):
        run_cube_baseline(_cfg(tmp_path, encoder_lr=1e-3, plasticity_gate="magic"))


@pytest.mark.slow
def test_a_gated_run_reports_the_finetuning_parameter_count(tmp_path):
    """27,206, not 27,271. Counting the critic twice, or counting the encoder not at all,
    both silently break the comparability of every fine-tuned record."""
    rec = run_cube_baseline(_cfg(tmp_path, encoder_lr=1e-3, plasticity_gate="dopamine",
                                 tag="gate_count"))
    assert rec["trainable_params"] == 27_206
