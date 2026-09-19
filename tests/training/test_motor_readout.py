"""EXP-064's `motor` readout: the brain's OWN pathway put on the policy path, and trained.

Before EXP-064, `Brain.step` computed `utilities = pfc(concept)` -> `router` -> `motor` -> an
action on **every step of every experiment ever run**, and that action was consumed only by
`monitor/runner.py` to draw the dashboard. Training and evaluation both ignored it and used
`head(concept)`. So **34,912 of the brain's 61,728 parameters (56.6%) were frozen at random
initialisation forever**, and the five-region topology was never on the policy path at all.

This readout makes the policy read the motor region's spike rates, so REINFORCE's gradient flows
back through motor -> router -> prefrontal via snnTorch's surrogate gradients.

The three tests that matter most:

  * `test_motor_gradient_reaches_prefrontal_and_motor` - the EXP-047 test. That experiment
    reported a 70x trainable surface while `fc1.weight` moved by exactly 0.0, and the run looked
    completely ordinary. Counting parameters in an optimizer is not evidence that they train.
  * `test_the_encoder_stays_frozen_under_the_motor_readout` - `region_lr` must move the regions
    and NOTHING else, or the arm silently becomes an encoder-finetuning arm too.
  * `test_preexisting_readouts_are_numerically_inert` - EXP-064 changed `use_memory` and the
    evaluation's `feature_fn` predicate, both of which every earlier readout flows through.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn as nn

from neuromorphic.training.cube_baseline import (
    MEMORY_MODES,
    MOTOR_MODES,
    CubeConfig,
    MemoryReadout,
    feature_width,
    make_agent,
    run_cube_baseline,
)
from neuromorphic.envs.cube import CubeEnv
from neuromorphic.training.reinforce import policy_parameters, train_episode


def _cfg(**kw):
    base = dict(arm="regionalized", readout="motor", tag="t", depth=1, seed=3, sigma=0.0,
                content=64)
    base.update(kw)
    return CubeConfig(**base)


def _readout(agent, seed=3):
    ro = MemoryReadout("motor", random.Random(seed), agent)
    ro.reset()
    return ro


# ------------------------------------------------------------------------------ shape and wiring

def test_feature_width_is_action_width_not_concept_width():
    """The motor region emits one spike train per action, so the head is a 6x6 affine: a learned
    temperature and bias on the brain's decision, not a second policy on top of it.

    Breaks returning `content`, which would silently reinstate a 64-wide readout.
    """
    assert feature_width(_cfg()) == 6
    assert feature_width(_cfg(readout="concept")) == 64


def test_motor_is_not_a_memory_mode():
    """Memory HURTS on this task (EXP-059/061/063). `use_memory` used to be `readout != "concept"`,
    which would have switched the hippocampus on for this arm and tested two changes at once.

    Breaks reverting that predicate.
    """
    assert "motor" in MOTOR_MODES
    assert "motor" not in MEMORY_MODES


def test_readout_returns_the_motor_spike_rates():
    """Breaks reading `utilities` (pre-motor) or `concept` instead of the motor output."""
    agent = make_agent(_cfg())
    obs = torch.tensor([random.Random(1).randrange(6) for _ in range(24)], dtype=torch.float32)
    out = agent.step(obs, store=False, recall=False)
    feats = _readout(agent)(out)
    assert feats.shape == (6,)
    assert torch.allclose(feats, out["action_spikes"].mean(dim=0)[0])


# -------------------------------------------------------------------------- the EXP-047 test

def test_motor_gradient_reaches_prefrontal_and_motor():
    """THE TEST THIS EXPERIMENT LIVES OR DIES ON. The regions must MOVE, not merely be registered
    in an optimizer.

    Breaks wrapping the motor branch in `no_grad`, breaks forgetting `grad_brain=True` (which
    would leave `brain.step` under `no_grad` so no gradient could reach the regions at all), and
    breaks detaching the rates before the head.
    """
    agent = make_agent(_cfg())
    torch.manual_seed(3)
    ro = _readout(agent)
    head = nn.Linear(6, 6)
    opt = torch.optim.Adam(list(policy_parameters(head)), lr=0.02)
    opt.add_param_group({"params": list(agent.pfc.parameters()) + list(agent.motor.parameters()),
                         "lr": 0.02})
    before = {n: p.detach().clone() for n, p in
              list(agent.pfc.named_parameters()) + list(agent.motor.named_parameters())}
    env = CubeEnv(scramble_depth=2, max_steps=6, scramble_seed=1)
    gen = torch.Generator().manual_seed(3)
    for _ in range(8):
        ro.reset()
        agent.hippo.clear()
        train_episode(agent, head, env, opt, generator=gen, max_steps=6,
                      store=False, recall=False, feature_fn=ro, grad_brain=True)
    for name, p in list(agent.pfc.named_parameters()) + list(agent.motor.named_parameters()):
        assert p.grad is not None, f"{name} received no gradient"
        assert float((p.detach() - before[name]).abs().max()) > 1e-9, f"{name} did not move"


def test_the_encoder_stays_frozen_under_the_motor_readout():
    """`region_lr` trains prefrontal and motor and NOTHING else. Breaks adding `sensory` to the
    region parameter group, which would make this an encoder-finetuning arm as well and change
    the trainable surface from 15,540 to over 42,000."""
    agent = make_agent(_cfg())
    torch.manual_seed(3)
    ro = _readout(agent)
    head = nn.Linear(6, 6)
    opt = torch.optim.Adam(list(policy_parameters(head)), lr=0.02)
    opt.add_param_group({"params": list(agent.pfc.parameters()) + list(agent.motor.parameters()),
                         "lr": 0.02})
    enc_before = {n: p.detach().clone() for n, p in agent.sensory.named_parameters()}
    env = CubeEnv(scramble_depth=2, max_steps=6, scramble_seed=1)
    gen = torch.Generator().manual_seed(3)
    for _ in range(6):
        ro.reset()
        agent.hippo.clear()
        train_episode(agent, head, env, opt, generator=gen, max_steps=6,
                      store=False, recall=False, feature_fn=ro, grad_brain=True)
    moved = max(float((p.detach() - enc_before[n]).abs().max())
                for n, p in agent.sensory.named_parameters())
    assert moved == 0.0, f"the encoder moved by {moved:.3e}; region_lr leaked into sensory"


def test_the_hippocampus_is_OFF_for_the_motor_arm(tmp_path):
    """Checking the mode tuples is not enough: `use_memory` is what actually gates store/recall.

    This asserts the OUTCOME - nothing was ever stored - rather than the switch. Breaks reverting
    `use_memory` to `readout != "concept"`, which would engage the hippocampus for this arm and
    test two changes at once, on a mechanism already measured to HURT (EXP-059/061/063).
    """
    r = run_cube_baseline(_cfg(region_lr=1e-3, tag="mem_off", episodes=4, max_depth=1,
                               out_dir=tmp_path))
    assert r["mean_n_stored"] == 0.0, (
        f"the motor arm stored {r['mean_n_stored']} patterns; the hippocampus is engaged"
    )
    assert r["recall_concept_norm_ratio"] is None


# ----------------------------------------------------------------------------------- the guards

def test_region_lr_refuses_a_readout_that_leaves_the_regions_off_the_path(tmp_path):
    """Training parameters no gradient reaches would report a trainable surface that is not
    actually trained. Refusing beats producing a plausible null."""
    with pytest.raises(ValueError, match="region_lr requires readout='motor'"):
        run_cube_baseline(_cfg(readout="concept", region_lr=1e-3, episodes=1, max_depth=1,
                               out_dir=tmp_path))


def test_region_lr_refuses_the_monolithic_arm(tmp_path):
    """The monolithic arm has no prefrontal or motor region at all."""
    with pytest.raises(ValueError, match="region_lr requires arm='regionalized'"):
        run_cube_baseline(_cfg(arm="monolithic", region_lr=1e-3, episodes=1, max_depth=1,
                               out_dir=tmp_path))


# ----------------------------------------------------------------- capacity match and instruments

def test_the_arms_are_capacity_matched(tmp_path):
    """The whole contrast rests on this. The motor arm trains 15,540 parameters (prefrontal
    15,456 + motor 42 + a 6x6 head 42); the control's MLP head at hidden=218 trains 15,484.

    Breaks any change to region sizes or the head that opens a capacity gap, which is the exact
    confound EXP-063 was caught by.
    """
    motor = run_cube_baseline(_cfg(region_lr=1e-3, tag="cap_m", episodes=2, max_depth=1,
                                   out_dir=tmp_path))
    ctrl = run_cube_baseline(_cfg(readout="concept", head_hidden=218, tag="cap_c", episodes=2,
                                  max_depth=1, out_dir=tmp_path))
    assert motor["trainable_params"] == 15_540
    assert ctrl["trainable_params"] == 15_484
    gap = abs(motor["trainable_params"] - ctrl["trainable_params"]) / motor["trainable_params"]
    assert gap < 0.01, f"capacity gap {gap:.1%} is too large for the contrast to be clean"


def test_region_drift_is_zero_for_an_untrained_pathway_and_positive_for_a_trained_one(tmp_path):
    """GATE 1 MUST BE ABLE TO BOTH PASS AND FAIL. `region_drift` is recorded only when the
    regions are trained, and reads exactly 0.0 if they never move.

    Breaks computing drift against the wrong snapshot (which would read ~0 always) and breaks
    recording it for arms that do not train the regions.
    """
    trained = run_cube_baseline(_cfg(region_lr=1e-2, tag="d_t", episodes=8, max_depth=1,
                                     out_dir=tmp_path))
    frozen = run_cube_baseline(_cfg(readout="concept", tag="d_f", episodes=2, max_depth=1,
                                    out_dir=tmp_path))
    assert trained["region_drift"] is not None and trained["region_drift"] > 0.0
    assert frozen["region_drift"] is None


def test_motor_firing_is_instrumented(tmp_path):
    """GATE 2. A silent motor region hands the head a constant zero vector and the arm degenerates
    to a bias-only policy that still trains and still reports an ordinary success rate.

    Breaks dropping the instrument, and breaks recording it for readouts that never touch motor.
    """
    motor = run_cube_baseline(_cfg(region_lr=1e-3, tag="f_m", episodes=4, max_depth=1,
                                   out_dir=tmp_path))
    assert motor["motor_rate_mean"] is not None and motor["motor_rate_mean"] >= 0.0
    assert 0.0 <= motor["motor_silent_frac"] <= 1.0
    ctrl = run_cube_baseline(_cfg(readout="concept", tag="f_c", episodes=2, max_depth=1,
                                  out_dir=tmp_path))
    assert ctrl["motor_rate_mean"] is None


# ------------------------------------------------------------------------------------ inertness

# Measured against the PRE-EXP-064 code in a clean worktree at `d2728fb`, depth 1, seed 3, 12
# episodes. `success_rate` is 0.3333 for all three and therefore discriminates nothing on its own,
# which is exactly why `revisit_rate` and `mean_train_entropy` are here: those DO differ per mode.
PRE_EXP064 = {
    "concept": (0.3333333333333333, 0.08333333333333333, 1.6683600048224132),
    "memory": (0.3333333333333333, 0.19230769230769232, 1.4800713956356049),
    "memory_attn": (0.3333333333333333, 0.10869565217391304, 1.505781392256419),
}


@pytest.mark.parametrize("mode", sorted(PRE_EXP064))
def test_preexisting_readouts_are_numerically_inert(mode, tmp_path):
    """EXP-064 changed `use_memory` and the evaluation `feature_fn` predicate, and EVERY earlier
    readout flows through both. Values measured against the pre-change code in a clean worktree.

    Breaks any change to either predicate that alters an existing arm.
    """
    cfg = CubeConfig(arm="regionalized", readout=mode, tag=f"i_{mode}", depth=1, seed=3,
                     sigma=0.0, episodes=12, entropy_beta=0.0, normalize_advantages=False,
                     max_depth=1, out_dir=tmp_path)
    r = run_cube_baseline(cfg)
    assert (r["success_rate"], r["revisit_rate"], r["mean_train_entropy"]) == PRE_EXP064[mode]
