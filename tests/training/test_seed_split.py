"""Separate the three seeds `cfg.seed` currently conflates.

EXP-034 Finding 4 could only BOUND the seed variance ("over 90% is not encoder quality and
not test-set composition") because `cfg.seed` simultaneously drives the encoder init, the
head init, the action-sampling stream, the environment's scramble stream and the
train/held-out split. Nothing on this driver could attribute it.

Splitting them makes the decomposition measurable: hold `encoder_seed` fixed and vary
`train_seed` to isolate optimisation luck, and vice versa.

The load-bearing test is `test_defaults_are_byte_identical_to_a_single_seed_run`. Every cube
experiment from EXP-029 to EXP-035 shares this driver, so the default path must not move.
"""

from __future__ import annotations

import torch

from neuromorphic.envs.cube import CubeEnv
from neuromorphic.training.cube_baseline import (
    CubeConfig,
    make_agent,
    resolve_seed,
    run_cube_baseline,
    shell_states,
    split_shell,
)
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.reinforce import concept_rate


def _concept(encoder_seed: int, content: int = 64):
    """Behavioural fingerprint of an encoder: what concept it produces for a fixed state."""
    cfg = CubeConfig(seed=0, encoder_seed=encoder_seed, content=content)
    brain = make_agent(cfg)
    env = CubeEnv(scramble_depth=1, max_steps=5, scramble_seed=0)
    obs, _ = env.reset()
    gen = torch.Generator().manual_seed(0)   # fixed, so only the encoder differs
    with torch.no_grad():
        return concept_rate(brain.step(obs, store=False, recall=False, record=False, generator=gen))


def test_all_three_seeds_default_to_none_and_resolve_to_seed():
    cfg = CubeConfig(seed=7)
    assert cfg.encoder_seed is None and cfg.train_seed is None and cfg.split_seed is None
    for which in ("encoder", "train", "split"):
        assert resolve_seed(cfg, which) == 7


def test_an_explicit_seed_overrides_the_fallback():
    cfg = CubeConfig(seed=7, encoder_seed=1, train_seed=2, split_seed=3)
    assert resolve_seed(cfg, "encoder") == 1
    assert resolve_seed(cfg, "train") == 2
    assert resolve_seed(cfg, "split") == 3


def test_zero_is_respected_and_not_treated_as_unset():
    """`if not cfg.encoder_seed` would silently fall back to `seed` at 0, which is the most
    commonly used seed in this repo. None is the only sentinel."""
    cfg = CubeConfig(seed=9, encoder_seed=0, train_seed=0, split_seed=0)
    for which in ("encoder", "train", "split"):
        assert resolve_seed(cfg, which) == 0


def test_defaults_are_byte_identical_to_a_single_seed_run(tmp_path):
    """EXP-029 through EXP-035 all share this driver. The default path must not move."""
    common = dict(arm="regionalized", depth=2, seed=5, episodes=8, max_depth=2)
    a = run_cube_baseline(CubeConfig(out_dir=tmp_path / "a", **common))
    b = run_cube_baseline(CubeConfig(
        out_dir=tmp_path / "b", encoder_seed=5, train_seed=5, split_seed=5, **common
    ))
    for key in a:
        if key == "config":
            continue
        assert a[key] == b[key], f"{key} moved when the seeds were split"


def test_encoder_seed_alone_determines_the_encoder():
    same = torch.allclose(_concept(3), _concept(3))
    diff = torch.allclose(_concept(3), _concept(4))
    assert same, "same encoder_seed produced different encoders"
    assert not diff, "different encoder_seed produced the same encoder"


def test_train_seed_does_not_touch_the_encoder():
    """Otherwise 'hold the encoder fixed, vary training' would not actually hold it fixed."""
    a = make_agent(CubeConfig(seed=0, encoder_seed=2, train_seed=100))
    b = make_agent(CubeConfig(seed=0, encoder_seed=2, train_seed=999))
    env = CubeEnv(scramble_depth=1, max_steps=5, scramble_seed=0)
    obs, _ = env.reset()
    with torch.no_grad():
        ca = concept_rate(a.step(obs, store=False, recall=False, record=False,
                                 generator=torch.Generator().manual_seed(0)))
        cb = concept_rate(b.step(obs, store=False, recall=False, record=False,
                                 generator=torch.Generator().manual_seed(0)))
    assert torch.allclose(ca, cb)


def test_split_seed_alone_determines_the_heldout_set():
    provider = ExactBFSDistance(max_depth=3)
    states = shell_states(provider, 3)

    def held(split_seed):
        _, ev, _ = split_shell(states, 3, seed=split_seed, heldout_cap=200, heldout_frac=0.25)
        return set(ev)

    assert held(0) == held(0)
    assert held(0) != held(1)


def test_encoder_and_training_luck_are_separable_in_practice(tmp_path):
    """The point of the whole change.

    Same encoder, two different training streams: the outcome must be free to differ, or
    optimisation luck could never be measured apart from encoder quality.
    """
    common = dict(arm="regionalized", depth=2, seed=0, episodes=25, max_depth=2,
                  encoder_seed=1, split_seed=1)
    a = run_cube_baseline(CubeConfig(out_dir=tmp_path / "a", train_seed=10, **common))
    b = run_cube_baseline(CubeConfig(out_dir=tmp_path / "b", train_seed=11, **common))
    moved = [k for k in ("success_rate", "revisit_rate", "mean_train_entropy",
                         "greedy_modal_action_frac")
             if a[k] != b[k]]
    assert moved, "train_seed changed nothing; optimisation luck is not isolated"


def test_same_training_stream_across_different_encoders_still_differs(tmp_path):
    """The mirror case: encoder quality must remain measurable with training held fixed."""
    common = dict(arm="regionalized", depth=2, seed=0, episodes=25, max_depth=2,
                  train_seed=4, split_seed=4)
    a = run_cube_baseline(CubeConfig(out_dir=tmp_path / "a", encoder_seed=20, **common))
    b = run_cube_baseline(CubeConfig(out_dir=tmp_path / "b", encoder_seed=21, **common))
    moved = [k for k in ("success_rate", "revisit_rate", "greedy_modal_action_frac")
             if a[k] != b[k]]
    assert moved, "encoder_seed changed nothing with training held fixed"


def test_split_seed_does_not_leak_into_the_encoder():
    a = make_agent(CubeConfig(seed=0, encoder_seed=6, split_seed=100))
    b = make_agent(CubeConfig(seed=0, encoder_seed=6, split_seed=999))
    env = CubeEnv(scramble_depth=1, max_steps=5, scramble_seed=0)
    obs, _ = env.reset()
    with torch.no_grad():
        ca = concept_rate(a.step(obs, store=False, recall=False, record=False,
                                 generator=torch.Generator().manual_seed(0)))
        cb = concept_rate(b.step(obs, store=False, recall=False, record=False,
                                 generator=torch.Generator().manual_seed(0)))
    assert torch.allclose(ca, cb)
