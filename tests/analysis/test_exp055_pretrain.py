"""EXP-055: the pretraining driver must reuse EXP-052's machinery, not reimplement it.

`pretrain_one` applies EXP-040's `rl_heldout_union` exclusions and asserts that no RL held-out
state leaked in as either endpoint of a training pair. Without those, an arm could win by
leakage rather than by epochs, and the whole epoch series would be measuring the wrong thing.
A reimplementation would be where that protection quietly goes missing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUN_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "055_pretraining_left_edge" / "pretrain_left_edge.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp055_pretrain", RUN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_four_pre_registered_epoch_arms():
    assert _module().EPOCH_ARMS == (1, 2, 3, 5)


def test_twelve_seeds():
    assert _module().SEEDS == tuple(range(12))


def test_encoder_names_are_exp055_and_do_not_collide_with_exp052():
    """EXP-052's encoders live in a different directory but share the seed and epoch fields.
    A name collision would make an EXP-055 arm silently load an EXP-052 encoder."""
    m = _module()
    name = m.encoder_path(Path("/tmp/x"), 1, 0).name
    assert name == "exp055_encoder_e1_s0.pt"
    assert "exp052" not in name


def test_encoder_names_are_unique_across_arms_and_seeds():
    m = _module()
    names = [m.encoder_path(Path("/tmp/x"), e, s).name
             for e in m.EPOCH_ARMS for s in m.SEEDS]
    assert len(set(names)) == len(names)


def test_it_reuses_exp052_pretrain_one_rather_than_reimplementing():
    """The leakage exclusions and their assertions live inside `pretrain_one`. If this module
    grew its own training loop, those protections would have to be re-derived and could
    silently differ."""
    m = _module()
    import inspect
    src = inspect.getsource(m)
    assert "pretrain_one" in src, "EXP-052's pretrain_one is not referenced at all"
    assert "build_pairs" not in src, (
        "this module appears to build its own training pairs; the leakage exclusions live in "
        "pretrain_one and must not be re-derived here"
    )


def test_exp052_pretrain_one_still_carries_the_leakage_exclusions():
    """BEHAVIOURAL, not a grep. The test above only checks EXP-055's own source for the
    absence of `build_pairs`; it says nothing about whether the protection EXP-055 is relying
    on still exists in the function it calls. If EXP-052's `pretrain_one` were edited tomorrow
    to drop its `rl_heldout_union` exclusions or their assertions, that grep would stay green
    while EXP-055 silently started training on leaked RL held-out states. This inspects the
    live source of `exp052.pretrain_one` itself and asserts on both pair endpoints, so a
    regression in the function being reused - not just in EXP-055's own file - fails this
    test."""
    import inspect

    m = _module()
    src = inspect.getsource(m.exp052.pretrain_one)
    assert "rl_heldout_union" in src, (
        "exp052.pretrain_one no longer references rl_heldout_union; EXP-040's leakage "
        "exclusions may have been dropped"
    )
    assert "forbidden & {p[0] for p in pairs}" in src, (
        "exp052.pretrain_one no longer asserts on the source endpoint of each training pair"
    )
    assert "forbidden & {p[2] for p in pairs}" in src, (
        "exp052.pretrain_one no longer asserts on the successor endpoint of each training pair"
    )
