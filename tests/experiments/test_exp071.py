"""EXP-071: cells, ranking check, runner and aggregator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.cube_baseline import shell_states

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "071_critic_and_no_revisit"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp071_cells", "cells.py")
rank = _load("exp071_rank", "rank.py")


def test_arms_are_exactly_the_spec_list():
    """Catches a missing or extra arm."""
    names = [cells.arm_name(*a) for a in cells.ARMS]
    assert names == ["G0", "G0V", "E3", "E3V", "P3", "P3V", "C1", "C1V", "C3", "C3V", "R3", "R3V"]


@pytest.mark.parametrize("text,expected", [("C3V", ("C", 3, True)), ("G0", ("G", 0, False))])
def test_parse_arm_round_trips(text, expected):
    assert cells.parse_arm(text) == expected


def test_parse_arm_rejects_arms_outside_the_spec():
    """Catches P1 or C2 sneaking in through the CLI."""
    for bad in ("P1", "C2", "G1V"):
        with pytest.raises(SystemExit):
            cells.parse_arm(bad)


@pytest.mark.parametrize("seed", cells.SEEDS)
def test_every_critic_is_tracked_and_loads(seed):
    """Catches a missing critic or a shape mismatch with the 64-wide concept."""
    critic = cells.load_critic(seed)
    assert critic.weight.shape == (1, 64)


class _Stub:
    """Concept [v, 0] per state; a critic reading coordinate 0; a head preferring move 0."""

    def __init__(self, values):
        self.values = {tuple(k): v for k, v in values.items()}

    def step(self, obs, *, recall=False, generator=None, **_):
        rows = np.asarray(obs)
        rows = rows[None, :] if rows.ndim == 1 else rows
        return {"concept": torch.tensor([[float(self.values.get(tuple(r), 0.0)), 0.0]
                                         for r in rows.tolist()]).unsqueeze(0)}


def _depth3_state_and_moves():
    # max_depth must cover the leaf level rank_state always computes (state depth 3 + 3 = 6),
    # not just the child level this helper's own good/bad selection needs (depth 4). At
    # max_depth=4 (the plan's original value) ExactBFSDistance.distance() returns None past
    # depth 4, so rank_state's leaf_closer_chance comparison crashes on NoneType < int. Input
    # fix, not an assertion change: see report.
    provider = ExactBFSDistance(max_depth=6)
    s = shell_states(provider, 3)[0]
    good = [a for a in range(N_ACTIONS) if provider.distance(apply_move(s, a)) == 2]
    bad = [a for a in range(N_ACTIONS) if provider.distance(apply_move(s, a)) == 4]
    return provider, s, good, bad


def test_rank_state_scores_a_critic_that_prefers_an_improving_child_as_a_hit():
    """Catches the hit being computed on the wrong side (distance + 1) or on the wrong argmax."""
    provider, s, good, bad = _depth3_state_and_moves()
    agent = _Stub({apply_move(s, good[0]): 5.0})
    critic = nn.Linear(2, 1)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[1.0, 0.0]]))
        critic.bias.zero_()
    head = nn.Linear(2, N_ACTIONS)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[bad[0]] = 5.0
    r = rank.rank_state(agent, head, critic, s, N_ACTIONS, provider, torch.Generator())
    assert r["critic_hit"] == 1 and r["policy_hit"] == 0
    assert r["chance"] == pytest.approx(len(good) / N_ACTIONS)


def test_rank_state_scores_a_critic_that_prefers_a_worsening_child_as_a_miss():
    """Catches a hit rate that cannot fail (e.g. always 1)."""
    provider, s, good, bad = _depth3_state_and_moves()
    agent = _Stub({apply_move(s, bad[0]): 5.0})
    critic = nn.Linear(2, 1)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[1.0, 0.0]]))
        critic.bias.zero_()
    head = nn.Linear(2, N_ACTIONS)
    r = rank.rank_state(agent, head, critic, s, N_ACTIONS, provider, torch.Generator())
    assert r["critic_hit"] == 0


def _leaf_case(want_closer):
    from neuromorphic.training.lookahead import tree_levels
    provider = ExactBFSDistance(max_depth=6)
    s = shell_states(provider, 3)[0]
    leaves = tree_levels(s, 3, N_ACTIONS)[3]
    target = next(l for l in leaves if (provider.distance(l) < 3) == want_closer)
    agent = _Stub({target: 5.0})
    critic = nn.Linear(2, 1)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[1.0, 0.0]]))
        critic.bias.zero_()
    head = nn.Linear(2, N_ACTIONS)
    return rank.rank_state(agent, head, critic, s, N_ACTIONS, provider, torch.Generator())


def test_leaf_check_scores_a_closer_top_leaf_as_a_hit():
    """Catches the leaf check reading children instead of leaves, or comparing to d + 3."""
    assert _leaf_case(True)["leaf_closer_hit"] == 1


def test_leaf_check_scores_a_farther_top_leaf_as_a_miss():
    """Catches a leaf hit that cannot fail."""
    r = _leaf_case(False)
    assert r["leaf_closer_hit"] == 0 and 0.0 < r["leaf_closer_chance"] < 1.0


run = _load("exp071_run", "run.py")


def test_run_cell_writes_a_complete_record(tmp_path):
    """Catches a record missing a field the aggregator or Gate V reads."""
    rec = run.run_cell("C", 1, True, 0, tmp_path, limit_states=2)
    on_disk = json.loads((tmp_path / cells.cell_record_name("C", 1, True, 0)).read_text())
    assert on_disk == rec
    for key in ("solved", "n", "success_rate", "eval_revisit_rate", "fallback_frac",
                "no_revisit", "goal_fired_frac", "mode", "k", "arm", "seed", "critic_file",
                "limit_states"):
        assert key in rec, key
    assert rec["arm"] == "C1V" and rec["no_revisit"] is True and rec["n"] == 2


GOLDEN = json.loads((REPO / "tests" / "fixtures" / "exp070_d7_s0_golden.json").read_text())


@pytest.mark.parametrize("arm", ["G0", "E3", "P3", "R3"])
def test_runner_plain_arms_match_the_pre_change_golden_fixture(arm, tmp_path):
    """THE GATE 0(b) CODE PATH, through the runner. Catches run_cell passing something that
    moves a plain arm (e.g. a critic changing a non-C arm, or no_revisit defaulting on). The
    fixture came from the UNMODIFIED code, so this can fail."""
    mode, k, v = cells.parse_arm(arm)
    rec = run.run_cell(mode, k, v, 0, tmp_path, limit_states=3)
    for field, value in GOLDEN[arm].items():
        assert rec[field] == value, (arm, field)
