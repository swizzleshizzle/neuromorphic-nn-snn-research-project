import numpy as np
import pytest
import torch

from neuromorphic.brain import Brain
from neuromorphic.encoders import cube_encoder
from neuromorphic.monolithic import MonolithicBrain
from neuromorphic.training.reinforce import action_distribution, make_policy_head


def _cube_pair(seed=0):
    brain = Brain(encoder=cube_encoder(), n_obs=144, obs_width=24, n_actions=6, seed=seed)
    mono = MonolithicBrain(
        n_obs=144, n_actions=6, total_neurons=brain.n_neurons,
        content=brain.content, obs_width=24, encoder=cube_encoder(), seed=seed,
    )
    return brain, mono


def test_neuron_count_matches_the_brain_exactly():
    brain, mono = _cube_pair()
    assert mono.n_neurons == brain.n_neurons == 510


def test_neuron_count_flows_from_total_neurons_parameter():
    """Verify n_neurons is computed from the parameter, not hardcoded to 510."""
    mono = MonolithicBrain(
        n_obs=144, n_actions=6, total_neurons=200, content=64, obs_width=24, encoder=cube_encoder()
    )
    assert mono.n_neurons == 200
    assert mono.stack.n_neurons == 200


def test_step_returns_a_concept_of_the_right_shape():
    _, mono = _cube_pair()
    out = mono.step(np.zeros(24, dtype=np.int64), generator=torch.Generator().manual_seed(0))
    assert out["concept"].shape == (mono.T, 1, mono.content)
    assert out["obs_spikes"].shape == (mono.T, 1, 144)


def test_is_a_drop_in_for_reinforce():
    """The whole point: reinforce.py must consume it with no changes."""
    _, mono = _cube_pair()
    head = make_policy_head(mono, "linear")
    dist, logits = action_distribution(
        mono, head, np.zeros(24, dtype=np.int64), generator=torch.Generator().manual_seed(0)
    )
    assert logits.shape == (6,)
    # `0 <= sample() < 6` was a tautology: a Categorical over 6 logits cannot sample outside
    # [0, 6) under any implementation. It would have passed against NaN logits, a collapsed
    # distribution, or one that always returned action 0.
    #
    # These are the properties that can actually fail. Measured 2026-08-03 at init: logits
    # [0.0697, 0.2685, -0.5212, 0.6221, -0.249, 0.0756], probs spanning 0.089 to 0.278, and
    # all 6 actions appearing within 60 samples.
    assert torch.isfinite(logits).all(), "non-finite logits would break REINFORCE silently"
    assert dist.probs.sum().item() == pytest.approx(1.0, abs=1e-6)
    gen = torch.Generator().manual_seed(0)
    samples = {
        int(
            action_distribution(
                mono, head, np.zeros(24, dtype=np.int64), generator=gen
            )[0].sample()
        )
        for _ in range(60)
    }
    # An untrained head must not already prefer one action. This is the collapse EXP-031
    # found on the cube, and a bounds check is blind to it.
    assert len(samples) == 6, f"only {len(samples)} distinct actions in 60 samples: {samples}"


def test_head_is_parameter_identical_to_the_regionalized_arm():
    """Same content width means the two arms train the same number of head parameters."""
    brain, mono = _cube_pair()
    n_brain = sum(p.numel() for p in make_policy_head(brain, "linear").parameters())
    n_mono = sum(p.numel() for p in make_policy_head(mono, "linear").parameters())
    assert n_brain == n_mono


def test_rejects_a_budget_that_cannot_hold_the_concept():
    with pytest.raises(ValueError, match="total_neurons"):
        MonolithicBrain(n_obs=144, n_actions=6, total_neurons=64, content=64)
