"""EXP-053: the learned state-dependent baseline, and the proof it is actually wired in.

`CLAUDE.md` records that EXP-047's first fine-tuning implementation trained nothing -
`fc1.weight` moved by exactly 0.0 - and produced a perfectly ordinary success rate. A critic
that is built, optimized and never read would look exactly the same. These tests assert the
critic's value ENTERS the advantage and that gradient ARRIVES AT ITS PARAMETERS.
"""

from __future__ import annotations

import pytest
import torch

from neuromorphic.brain import Brain
from neuromorphic.envs.cube import scramble
from neuromorphic.encoders import encode_cube
from neuromorphic.training.cube_baseline import ShellCubeEnv
from neuromorphic.training.reinforce import (
    discounted_returns,
    make_critic,
    make_policy_head,
    train_episode,
)

import random


def _brain() -> Brain:
    torch.manual_seed(0)
    # `obs_width=24` (raw facelets) is required alongside `n_obs=144` (24 facelets x 6
    # colors, one-hot) for a cube encoder; the default `obs_width=4` is the gridworld shape
    # and raises on a 24-facelet observation. See `CUBE_OBS_WIDTH` in cube_baseline.py.
    return Brain(encoder=encode_cube, n_obs=144, n_actions=6, obs_width=24)


def _env(seed: int = 0) -> ShellCubeEnv:
    # `ShellCubeEnv` draws its start state from a fixed pool (see cube_baseline.py); it needs
    # a real pool, not `None` - built here from `scramble` at the same depth so the test still
    # exercises a shallow-scrambled cube.
    pool = [scramble(2, random.Random(seed))]
    return ShellCubeEnv(pool, random.Random(seed), scramble_depth=2, max_steps=4)


def test_critic_value_enters_the_advantage():
    """A constant critic must shift the advantage by exactly that constant.

    Pinned to a constant so the expected number is arithmetic, not a re-implementation of
    the code under test. `critic_sse` is the residual sum of squares against the realized
    returns, so with V == 5.0 it must equal sum((G_t - 5.0)**2).
    """
    brain, env = _brain(), _env()
    head = make_policy_head(brain)
    critic = make_critic(brain)
    with torch.no_grad():
        critic.weight.zero_()
        critic.bias.fill_(5.0)

    opt = torch.optim.Adam(head.parameters(), lr=0.0)
    copt = torch.optim.Adam(critic.parameters(), lr=0.0)
    gen = torch.Generator().manual_seed(0)

    stats = train_episode(brain, head, env, opt, gamma=0.99, baseline=0.0,
                          generator=gen, max_steps=4,
                          critic=critic, critic_optimizer=copt)

    # Rebuild the returns from the rewards the episode actually earned. A solved episode
    # ends early, so `steps` is the authority on length, not `max_steps`. Every non-terminal
    # step earns `step_penalty` (-1.0); the last step earns `solve_reward` (10.0) only if the
    # episode actually reached the goal, else another `step_penalty` (env default reward
    # rule: `solve_reward if terminated else step_penalty`). This reconstructs the reward
    # sequence independently of `critic_fit_terms`, so it can catch a broken residual
    # computation rather than just re-deriving it.
    assert stats["critic_n"] == stats["steps"]
    step_penalty, solve_reward = -1.0, 10.0
    rewards = [step_penalty] * (stats["steps"] - 1) + (
        [solve_reward] if stats["reached_goal"] else [step_penalty]
    )
    returns = discounted_returns(rewards, gamma=0.99)
    expected_sse = sum((g - 5.0) ** 2 for g in returns)
    assert stats["critic_sse"] == pytest.approx(expected_sse), (
        f"critic_sse {stats['critic_sse']} != sum((G_t - 5.0)**2) = {expected_sse} for the "
        "realized episode's returns"
    )


def test_gradient_arrives_at_the_critic_parameters():
    """The critic must MOVE. A silently frozen critic looks exactly like a null result.

    Threshold is measured, not qualitative: record the observed drift in the comment below
    when this first passes. If the observed drift is under 1e-3, do NOT lower this bar -
    investigate, because it means the critic is barely training.

    Observed 2026-08-28: drift = 1.000000536e-02 (Adam lr=1e-2, one step - a first Adam step
    moves by approximately lr regardless of gradient magnitude, so this is the expected size).
    """
    brain, env = _brain(), _env()
    head = make_policy_head(brain)
    critic = make_critic(brain)
    before = critic.weight.detach().clone()

    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    copt = torch.optim.Adam(critic.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)

    stats = train_episode(brain, head, env, opt, gamma=0.99, baseline=0.0,
                          generator=gen, max_steps=4,
                          critic=critic, critic_optimizer=copt)

    assert critic.weight.grad is not None, "the critic received no gradient at all"
    drift = (critic.weight.detach() - before).abs().max().item()
    assert drift > 1e-4, (
        f"critic.weight moved by only {drift:.3e}. The critic is not training - most likely "
        "its loss is never added to the backward, or its optimizer is never stepped."
    )
    assert "critic_sse" in stats


def test_no_critic_leaves_the_stats_and_the_advantage_alone():
    """Without a critic the executed path is the pre-change one: no critic keys at all."""
    brain, env = _brain(), _env()
    head = make_policy_head(brain)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)

    stats = train_episode(brain, head, env, opt, gamma=0.99, baseline=0.0,
                          generator=gen, max_steps=4)

    for k in ("critic_sse", "return_sum", "return_sq_sum", "critic_n"):
        assert k not in stats, f"{k} leaked into a run with no critic"
    assert stats["gate_open"] is True


def test_with_features_is_opt_in():
    """`action_distribution` keeps its 2-tuple contract unless asked for features."""
    from neuromorphic.training.reinforce import action_distribution
    brain, env = _brain(), _env()
    head = make_policy_head(brain)
    obs, _ = env.reset()
    gen = torch.Generator().manual_seed(0)

    two = action_distribution(brain, head, obs, generator=gen)
    assert len(two) == 2

    gen = torch.Generator().manual_seed(0)
    three = action_distribution(brain, head, obs, generator=gen, with_features=True)
    assert len(three) == 3
    assert three[2].shape == (brain.content,)


def test_the_critic_does_not_leak_gradient_into_the_encoder():
    """With grad_brain=True, the critic's own MSE loss must not reach the encoder.

    An earlier version of this test compared the encoder gradient of a critic-using run
    against a no-critic run and asserted exact equality. That comparison cannot work: ANY
    critic with a nonzero weight makes `v` (hence `advantages = returns - v.detach()`)
    depend on the state, which legitimately changes the encoder gradient through the POLICY
    loss alone (nothing to do with a leak) versus a constant `baseline=0.0`. And a critic
    with a zero weight can never leak in the first place, since `d(v)/d(features) == 0`
    regardless of whether `features` is detached. Measured directly: even with today's
    correct (detached) code, a default-init critic moved the encoder's max |grad| from 1.81
    (no critic) to 1.76 (critic) - already not equal, so `torch.equal` failed on CORRECT
    code, not just buggy code. That confound was in the test, not the fix.

    This version isolates the leak instead of comparing against a mismatched baseline: scale
    one critic's weight and bias by 1000x so its residual against the realized returns is
    large, then check the encoder gradient's raw magnitude stays bounded. Measured over this
    seeded episode: correctly detached, max |grad| on `fc1.weight` is ~891; with the detach
    removed (features fed to the critic un-detached), the SAME setup measures ~401,682 - a
    ~450x jump, because the critic's large-residual MSE gradient backprops through `features`
    into the encoder in addition to the (much smaller) policy-loss contribution. The
    threshold below sits with an 11x margin above the correct value and a 40x margin below
    the leaked one.
    """
    brain, env = _brain(), _env()
    head = make_policy_head(brain)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    critic = make_critic(brain)
    with torch.no_grad():
        # Deliberately large and off-target so a leaked gradient would be unmistakable; a
        # critic near its default init doesn't have enough residual to show the effect (see
        # docstring above - the confound this replaces failed even on correct code).
        critic.weight.mul_(1000.0)
        critic.bias.fill_(1000.0)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
    enc_opt = torch.optim.Adam(brain.sensory.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)

    train_episode(brain, head, env, opt, gamma=0.99, baseline=0.0,
                  generator=gen, max_steps=4, grad_brain=True,
                  critic=critic, critic_optimizer=critic_opt,
                  encoder_optimizer=enc_opt)

    max_grad = brain.sensory.fc1.weight.grad.detach().abs().max().item()
    assert max_grad < 10_000.0, (
        f"encoder fc1.weight.grad max magnitude was {max_grad:.1f}, expected roughly 900 "
        "(the policy-loss-only scale). A jump toward ~400,000 means the critic's MSE loss "
        "against a deliberately large residual is backpropagating through `features` into "
        "the encoder - `features` is not detached where the critic reads it."
    )
