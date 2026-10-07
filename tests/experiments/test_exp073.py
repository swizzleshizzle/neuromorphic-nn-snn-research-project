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


evaluate = _load("exp073_evaluate", "evaluate.py")


def test_negj_makes_the_critic_scorer_pick_the_lowest_j():
    """Catches J used with the wrong sign (the search would head AWAY from solved)."""
    head = torch.nn.Sequential(torch.nn.Linear(2, 1))
    with torch.no_grad():
        head[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        head[0].bias.zero_()
    neg = evaluate.NegJ(head)
    concepts = torch.tensor([[5.0, 0.0], [0.1, 0.0], [3.0, 0.0]])
    assert int(neg(concepts).squeeze(-1).argmax()) == 1


def test_eval_arms_are_exactly_the_spec_list():
    assert [a for a in evaluate.ARMS_EVAL] == ["J1V-A", "J1V-B", "J3V-A", "J3V-B", "P3V", "R3V"]


def test_p3v_cell_reproduces_exp072_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH: P3V through this harness on 3 states must equal a direct
    evaluate_lookahead of EXP-072's configuration on the same states."""
    from neuromorphic.training.lookahead import evaluate_lookahead
    rec = evaluate.run_cell("P3V", 8, 0, tmp_path, limit_states=3)
    agent, head, states, ts = cells.c70.load_cell(8, 0)
    ref = evaluate_lookahead(agent, head, states[:3], depth=8, mode="P", k=3,
                             generator=torch.Generator().manual_seed(ts), rng_seed=ts,
                             imag_seed=ts, no_revisit=True)
    for f in ("solved", "success_rate", "mean_steps", "eval_revisit_rate"):
        assert rec[f] == ref[f], f


def test_judge_cell_without_a_checkpoint_exits_naming_it(tmp_path, monkeypatch):
    """Catches a J cell silently evaluating an untrained judge when its checkpoint is missing."""
    monkeypatch.setattr(evaluate.cells, "ckpt_dir", lambda arm, seed: tmp_path / "nope" / f"{arm}{seed}")
    with pytest.raises(SystemExit, match="judge.pt"):
        evaluate.run_cell("J3V-A", 8, 0, tmp_path, limit_states=1)


agg = _load("exp073_aggregate", "aggregate.py")


def _gates(**over):
    g = {"gate0": True, "t": {"A": True, "B": True}, "e": True,
         "r": {("J-A", 9): True, ("J-B", 9): True}}
    g.update(over)
    return g


def test_gate_t_requires_monotone_means_not_just_correlation():
    """Catches a judge flat past distance 9 passing on correlation alone."""
    good = {"spearman_7_11": 0.6,
            "mean_j_by_distance": {"7": 6.0, "8": 7.0, "9": 8.0, "10": 9.0, "11": 9.5}}
    flat = {"spearman_7_11": 0.6,
            "mean_j_by_distance": {"7": 6.0, "8": 7.0, "9": 8.0, "10": 8.0, "11": 8.0}}
    assert agg.gate_t_verdict([good] * 12, 0.3) is True
    assert agg.gate_t_verdict([flat] * 12, 0.3) is False
    assert agg.gate_t_verdict([good] * 12, 0.7) is False


def test_gate_e_catches_a_frozen_arm_a_and_a_leaky_arm_b():
    """THE EXP-047 TRAP as a gate."""
    assert agg.gate_e_verdict([0.5] * 12, [0.0] * 12) is True
    assert agg.gate_e_verdict([0.5] * 11 + [0.0], [0.0] * 12) is False
    assert agg.gate_e_verdict([0.5] * 12, [0.0] * 11 + [1e-9]) is False


def test_claims_compare_the_registered_arms_at_depth_9():
    """Catches Claim 1 against J3V-B or Claim 2 against P3V, or the wrong depth."""
    r = lambda v: {s: v for s in range(12)}
    rates = {("J3V-A", 9): r(0.10), ("P3V", 9): r(0.06), ("J3V-B", 9): r(0.03)}
    out = agg.primary_verdicts(rates, _gates())
    assert out["claim1"][4] == pytest.approx(0.06)
    assert out["claim2"][4] == pytest.approx(0.03)


def test_claim_is_void_when_a_gate_in_its_contrast_fails():
    """Catches a verdict printed for an arm whose Gate T, E or R failed."""
    r = lambda v: {s: v + 0.001 * s for s in range(12)}
    rates = {("J3V-A", 9): r(0.20), ("P3V", 9): r(0.06), ("J3V-B", 9): r(0.03)}
    assert agg.primary_verdicts(rates, _gates())["claim1"][0] == "CONFIRMED"
    assert agg.primary_verdicts(rates, _gates())["claim2"][0] == "CONFIRMED"
    bad_b = agg.primary_verdicts(rates, _gates(t={"A": True, "B": False}))
    assert bad_b["claim1"][0] == "CONFIRMED" and bad_b["claim2"][0] == "VOID"
    bad_e = agg.primary_verdicts(rates, _gates(e=False))
    assert bad_e["claim1"][0] == "CONFIRMED" and bad_e["claim2"][0] == "VOID"
    bad_a = agg.primary_verdicts(rates, _gates(t={"A": False, "B": True}))
    assert bad_a["claim1"][0] == "VOID" and bad_a["claim2"][0] == "VOID"
    bad_r = agg.primary_verdicts(rates, _gates(r={("J-A", 9): False, ("J-B", 9): True}))
    assert bad_r["claim1"][0] == "VOID" and bad_r["claim2"][0] == "VOID"


def test_gate_r_needs_both_significance_and_a_positive_margin():
    """Catches a Gate R that passes on a hit rate at or below chance."""
    above = [{"hit": 0.4, "chance": 0.15 + 0.001 * i} for i in range(12)]
    below = [{"hit": 0.05, "chance": 0.15} for _ in range(12)]
    assert agg.gate_r_verdict(above) is True
    assert agg.gate_r_verdict(below) is False


def test_gate_t_threshold_is_unset_until_the_controller_amends():
    """Catches code (rather than a dated amendment) choosing the Gate T threshold."""
    assert agg.GATE_T_THRESHOLD is None
