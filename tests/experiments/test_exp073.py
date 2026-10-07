"""EXP-073: cells, training driver, evaluation and aggregator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube_distance import ExactBFSDistance

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "073_learned_judge"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp073_cells", "cells.py")
train = _load("exp073_train", "train.py")


@pytest.fixture(scope="module")
def provider():
    return ExactBFSDistance(max_depth=None)


def test_depth_7_to_9_heldout_sets_match_exp070(provider):
    """Catches evaluating or excluding a different split than the one every paired reference
    was measured on."""
    for d in (7, 8, 9):
        states = cells.c70.load_cell(d, 0)[2]
        assert cells.heldout_states(d, 0, provider) == states


def test_probe_and_training_never_touch_an_evaluation_state(provider):
    """Catches an evaluated position leaking into the probe (and so into Gate T)."""
    excl = cells.exclusion_set(0, provider)
    probe = cells.probe_set(0, provider, per_distance=5)
    assert len(excl) == 800
    assert not ({s for s, _ in probe} & excl)
    assert sorted({d for _, d in probe}) == list(range(1, 12))


def test_probe_set_discards_every_excluded_state_even_when_shuffle_puts_it_first(monkeypatch):
    """Direct, deterministic companion to the test above. On the REAL BFS shells, the plan's
    own mutation ('probe built without the exclusion skip') turns out to be a near-miss: at
    per_distance=5 a 200-of-33058 exclusion set has only about a 3% chance of landing in the
    first 5 shuffled slots, and it empirically passed WITH the filter removed (confirmed by
    applying that exact mutation and re-running the test above: it still passed). This test
    removes the luck by making every candidate state excluded, so an unfiltered implementation
    returns a nonempty probe every time, deterministically."""
    fake_states = [(d, i) for d in range(1, 12) for i in range(20)]
    monkeypatch.setattr(cells, "shell_states", lambda provider, d: [s for s in fake_states if s[0] == d])
    monkeypatch.setattr(cells, "exclusion_set", lambda seed, provider: set(fake_states))
    probe = cells.probe_set(0, object(), per_distance=5)
    assert probe == []


def test_driver_refuses_seed_misuse():
    """Catches a pilot run on an evaluation seed, or an evaluation run on a pilot seed."""
    with pytest.raises(SystemExit):
        train.check_seeds([0], pilot=True)
    with pytest.raises(SystemExit):
        train.check_seeds([12], pilot=False)
    train.check_seeds([12, 13], pilot=True)
    train.check_seeds(list(range(12)), pilot=False)


def test_make_judge_starts_from_the_seeds_e1_encoder():
    """Catches a judge built on a random encoder (it would make arm A vs B meaningless)."""
    j = cells.make_judge(3)
    e1 = torch.load(cells.c70.published_config(7, 3).encoder_state_path, map_location="cpu")
    for k, v in j.sensory.state_dict().items():
        assert torch.equal(v, e1[k]), k
