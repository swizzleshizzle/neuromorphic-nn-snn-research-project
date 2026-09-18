"""The on-path-capacity-matched topology contrast cannot be run, and this proves why.

`docs/phase3-honest-assessment.md` section 2a established that the monolithic control in EXP-029
is neuron-matched on TOTAL neurons (510 each) while the arms differ enormously in what sits on the
policy path: the monolithic stack puts all 510 there, the regionalized brain puts only its
192-neuron sensory region there, because with `recall=False` the other four regions are
architecturally disconnected from the action.

The obvious fix looks like "match ON-PATH capacity instead" - a monolithic arm at `total_neurons =
192`. **That contrast is vacuous: the two arms are then the same network, bit for bit.**

`Brain` builds `SensoryCortex(n_obs=144, hidden=128, concept=64, seed=s)` and
`MonolithicBrain(total_neurons=192, content=64)` builds `SensoryCortex(n_obs=144, hidden=192-64,
concept=64, seed=s)`. Same module, same constructor arguments, same seed, so the same weights and
the same concept. The policy head reads `concept` in both cases and nothing else.

This is EXP-030's lesson recurring: **a path-matched control can turn out bit-identical to the arm
it is controlling for.** There the shuffle-null varied the query state and so also varied
"features of the current observation"; here the capacity-match removes the only thing that
differed. Both are the same error, which is why this project's habit is to ask what a control
holds fixed BESIDES the thing it names.

> WHAT A FAILURE OF THIS FILE MEANS, because it is the opposite of the usual reading.
>
> These assertions pass because the topology is NOT on the policy path. If they ever fail, that is
> **good news**: it means some region other than `sensory` now influences the action, and the
> topology question has become answerable for the first time. **Do not "fix" a failure here by
> deleting the test.** Read the failure, then go and design the contrast this file currently
> proves is impossible.
"""

from __future__ import annotations

import random

import pytest
import torch

from neuromorphic.encoders import cube_encoder
from neuromorphic.monolithic import MonolithicBrain
from neuromorphic.training.cube_baseline import CUBE_N_OBS, CUBE_OBS_WIDTH, CubeConfig, make_agent
from neuromorphic.training.reinforce import concept_rate

SEEDS = (0, 1, 3, 7, 11, 23)


def _regionalized(seed: int):
    return make_agent(CubeConfig(arm="regionalized", readout="concept", tag="t", depth=5,
                                 seed=seed, sigma=0.0, content=64))


def _monolithic(total_neurons: int, seed: int):
    return MonolithicBrain(n_obs=CUBE_N_OBS, n_actions=6, total_neurons=total_neurons,
                           content=64, obs_width=CUBE_OBS_WIDTH, encoder=cube_encoder(),
                           seed=seed)


def test_only_the_sensory_region_is_on_the_policy_path():
    """The premise everything below rests on. 318 of 510 neurons are disconnected from the action.

    Breaks any change that wires another region into `concept`, which is exactly the change that
    would make the topology contrast runnable.
    """
    brain = _regionalized(seed=3)
    assert brain.n_neurons == 510
    assert brain.sensory.n_neurons == 192
    off_path = brain.n_neurons - brain.sensory.n_neurons
    assert off_path == 318, (
        f"{off_path} neurons off the policy path, expected 318. If this fell, a region may now "
        "be on the path and the topology contrast may be worth designing."
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_on_path_matched_monolithic_has_identical_weights(seed):
    """The capacity-matched control IS the arm it controls for.

    Breaks the assumption behind "just match on-path capacity", and would break if either
    constructor stopped building the same `SensoryCortex` from the same seed.
    """
    reg = _regionalized(seed)
    matched = _monolithic(reg.sensory.n_neurons, seed)
    assert matched.n_neurons == 192
    rw, mw = dict(reg.sensory.state_dict()), dict(matched.stack.state_dict())
    assert set(rw) == set(mw), "the two arms no longer build the same module"
    for k in sorted(rw):
        assert torch.equal(rw[k], mw[k]), f"{k} differs; the contrast may have become non-vacuous"


@pytest.mark.parametrize("seed", SEEDS)
def test_on_path_matched_arms_produce_bit_identical_concepts(seed):
    """And they compute the same thing, which is what makes a success-rate contrast meaningless.

    Weight equality alone would not prove it: the two `step()` implementations differ, and only
    comparing their OUTPUT rules out a difference in how the spikes are driven. Exact equality,
    not a tolerance, because anything else would leave room for a real difference to hide.
    """
    reg = _regionalized(seed)
    matched = _monolithic(reg.sensory.n_neurons, seed)
    rng = random.Random(seed)
    for _ in range(4):
        obs = torch.tensor([rng.randrange(6) for _ in range(24)], dtype=torch.float32)
        g_reg = torch.Generator().manual_seed(5)
        g_mono = torch.Generator().manual_seed(5)
        with torch.no_grad():
            a = concept_rate(reg.step(obs, store=False, recall=False, generator=g_reg))
            b = concept_rate(matched.step(obs, generator=g_mono))
        assert torch.equal(a, b), (
            f"concepts differ by up to {float((a - b).abs().max()):.3e} at seed {seed}. "
            "If this is a real difference, the on-path-matched topology contrast has become "
            "runnable and should be designed properly rather than assumed vacuous."
        )


@pytest.mark.parametrize("seed", (0, 3))
def test_exp029s_actual_control_is_NOT_identical_and_favours_itself(seed):
    """The contrast EXP-029 really ran is not vacuous. It is mismatched, and in the control's favour.

    This is the other half of section 2a: total-matching gives the monolithic arm 2.66x the
    on-path capacity. Breaks a future "the monolithic control was matched, so the comparison was
    fair" claim, and breaks reading the vacuity above as applying to EXP-029 itself.
    """
    reg = _regionalized(seed)
    as_run = _monolithic(reg.n_neurons, seed)
    assert as_run.n_neurons == 510
    assert as_run.n_neurons / reg.sensory.n_neurons == pytest.approx(2.656, abs=0.01)
    rw, mw = dict(reg.sensory.state_dict()), dict(as_run.stack.state_dict())
    shapes_differ = any(rw[k].shape != mw[k].shape for k in set(rw) & set(mw))
    assert shapes_differ, "EXP-029's control should differ in width from the regionalized path"
