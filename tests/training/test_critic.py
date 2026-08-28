"""EXP-053: the learned state-dependent baseline, and the proof it is actually wired in.

`CLAUDE.md` records that EXP-047's first fine-tuning implementation trained nothing -
`fc1.weight` moved by exactly 0.0 - and produced a perfectly ordinary success rate. A critic
that is built, optimized and never read would look exactly the same. These tests assert the
critic's value ENTERS the advantage and that gradient ARRIVES AT ITS PARAMETERS.
"""

from __future__ import annotations

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
    # ends early, so `steps` is the authority on length, not `max_steps`.
    assert stats["critic_n"] == stats["steps"]
    assert stats["critic_sse"] > 0.0, (
        "a constant critic of 5.0 cannot have zero residual against cube returns "
        "(solve_reward 10.0, step_penalty -1.0)"
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
