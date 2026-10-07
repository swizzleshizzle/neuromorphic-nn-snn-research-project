"""EXP-074: cells, Gate L, training driver, evaluation and aggregator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "074_wide_judge"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp074_cells", "cells.py")
gate_l = _load("exp074_gate_l", "gate_l.py")


def test_arms_a_and_b_are_exactly_exp073s_judges():
    """THE GATE 0(c) PRECONDITION. Catches arm A or B built differently from EXP-073 (head init
    order, readout, encoder), which would make them a new arm rather than the control."""
    for arm in ("A", "B"):
        mine = cells.make_judge(3, arm).state_dict()
        ref = cells.c73.make_judge(3).state_dict()
        assert mine.keys() == ref.keys()
        for k in ref:
            assert torch.equal(mine[k], ref[k]), (arm, k)


def test_arm_w_is_wide_and_starts_from_the_seeds_e1_encoder():
    """Catches W built on a random encoder or with the concept readout."""
    j = cells.make_judge(3, "W")
    assert j.readout == "wide" and j.head[0].in_features == 192
    e1 = torch.load(cells.c70.published_config(7, 3).encoder_state_path, map_location="cpu")
    for k, v in j.sensory.state_dict().items():
        assert torch.equal(v, e1[k]), k


def test_make_judge_refuses_an_unknown_arm():
    with pytest.raises(ValueError, match="arm"):
        cells.make_judge(0, "C")


def test_sensitivity_drop_is_exactly_the_disclosed_seeds():
    assert cells.SENSITIVITY_DROP == (0, 3)


class _ByDistance(torch.nn.Module):
    """A stub judge: J(s) = sign * true distance."""

    def __init__(self, provider, sign):
        super().__init__()
        self.p, self.sign = provider, sign

    def forward(self, states, generator):
        return torch.tensor([self.sign * float(self.p.distance(s)) for s in states])


@pytest.fixture(scope="module")
def small():
    return ExactBFSDistance(max_depth=6)


def _probe(small):
    roots = [apply_move(apply_move(apply_move(SOLVED, 0), 2), 4),
             apply_move(apply_move(SOLVED, 1), 3)]
    return [(s, small.distance(s)) for s in roots]


def test_gate_l_perfect_judge_always_hits_and_anti_judge_never_does(small):
    """Catches argmax instead of argmin (the instrument would reward a judge for pointing AWAY
    from solved) and a hit defined on the wrong side of the root distance."""
    probe = _probe(small)
    good = gate_l.leaf_rank_margin(_ByDistance(small, 1.0), probe, small.distance, 0,
                                   N_ACTIONS, lo=1, hi=6)
    bad = gate_l.leaf_rank_margin(_ByDistance(small, -1.0), probe, small.distance, 0,
                                  N_ACTIONS, lo=1, hi=6)
    assert good["hit"] == 1.0 and bad["hit"] == 0.0
    assert good["n"] == 2
    assert good["chance"] == bad["chance"]
    assert 0.0 < good["chance"] < 1.0
    assert good["margin"] == pytest.approx(1.0 - good["chance"])


def test_gate_l_chance_is_the_fraction_of_closer_leaves(small):
    """Catches chance computed over children instead of the 3-move leaves the search scores."""
    from neuromorphic.training.lookahead import tree_levels
    s, d = _probe(small)[1]
    leaves = tree_levels(s, 3, N_ACTIONS)[3]
    want = sum(small.distance(x) < d for x in leaves) / len(leaves)
    got = gate_l.leaf_rank_margin(_ByDistance(small, 1.0), [(s, d)], small.distance, 0,
                                  N_ACTIONS, lo=1, hi=6)
    assert got["chance"] == pytest.approx(want)


def test_gate_l_restricts_to_distances_lo_to_hi(small):
    """Catches the gate pooling shallow probe states (which every judge ranks well) into the
    margin it reports for distances 7 to 11."""
    probe = _probe(small)
    got = gate_l.leaf_rank_margin(_ByDistance(small, 1.0), probe, small.distance, 0,
                                  N_ACTIONS, lo=3, hi=6)
    assert got["n"] == sum(1 for _, d in probe if 3 <= d <= 6)
    assert set(got["by_distance"]) == {str(d) for _, d in probe if 3 <= d <= 6}


train = _load("exp074_train", "train.py")


def test_driver_refuses_seed_misuse():
    """Catches a pilot run on an evaluation seed, or an evaluation run on a pilot seed."""
    with pytest.raises(SystemExit):
        train.check_seeds([0], pilot=True)
    with pytest.raises(SystemExit):
        train.check_seeds([12], pilot=False)
    train.check_seeds([12, 13], pilot=True)
    train.check_seeds(list(range(12)), pilot=False)


def test_driver_defaults_are_the_spec_settings():
    """Catches a driver whose defaults drift from spec section 4 (the launcher passes them, but a
    hand-run must not silently differ)."""
    ap = train.build_parser()
    a = ap.parse_args(["--n-updates", "1"])
    assert (a.batch, a.sync_every, a.probe_every, a.draws) == (1000, 100, 250, 1)
    assert a.arms == ["W", "A", "B"]


@pytest.mark.slow
def test_run_writes_gate_l_drift_and_readout(tmp_path):
    """Catches a record missing the Gate L block or the encoder drift Gate E reads. Slow: builds
    the full BFS table (about 65 s) as the real driver does."""
    rec = train.run("W", 12, 2, 8, 1, 1, 1, tmp_path)
    on_disk = json.loads((tmp_path / cells.record_name("W", 12)).read_text())
    assert on_disk["readout"] == "wide"
    assert on_disk["gate_l"]["n"] == 250
    assert 0.0 <= on_disk["gate_l"]["chance"] <= 1.0
    assert on_disk["encoder_drift"] > 0.0
    assert (tmp_path / "judge_W_s12" / "judge.pt").exists()
    assert rec["gate_l"] == on_disk["gate_l"]


def test_relaunch_skips_runs_whose_record_exists(tmp_path):
    """Catches a relaunch that resubmits finished cells and overwrites their records."""
    (tmp_path / cells.record_name("A", 12)).write_text("{}")
    assert train.pending_jobs(["W", "A"], [12, 13], tmp_path) == [("W", 12), ("W", 13), ("A", 13)]


evaluate = _load("exp074_evaluate", "evaluate.py")


class _FixedJ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def forward(self, states, generator):
        return torch.tensor(self.values[: len(states)])


def test_judge_critic_reads_states_and_prefers_the_lowest_j():
    """Catches J used with the wrong sign, or a critic the search would read via the concept."""
    c = evaluate.JudgeCritic(_FixedJ([5.0, 0.1, 3.0]))
    assert c.reads_states is True
    assert int(c(["a", "b", "c"], generator=None).argmax()) == 1


def test_eval_arms_and_rank_kinds_are_exactly_the_spec_lists():
    assert list(evaluate.ARMS_EVAL) == ["J3V-W", "J3V-A", "J3V-B", "P3V", "R3V"]
    assert list(evaluate.RANK_KINDS) == ["J-W", "J-A", "J-B", "P"]


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


def test_judge_cell_without_a_checkpoint_exits_naming_it(tmp_path):
    """Catches a J cell silently evaluating an untrained judge."""
    with pytest.raises(SystemExit, match="judge.pt"):
        evaluate.run_cell("J3V-A", 8, 0, tmp_path, limit_states=1, judge_dir=tmp_path / "nope")


def test_a_wide_judge_drives_a_real_j3v_cell(tmp_path):
    """Catches a W judge evaluated through the 64-unit concept (the EXP-073 NegJ path): the
    192-input head cannot read a 64-wide concept, so this cell would crash or need a wrong
    'fix'. Uses an untrained W checkpoint saved where the evaluator looks."""
    jdir = tmp_path / "judges"
    path = cells.ckpt_dir("W", 0, jdir)
    path.mkdir(parents=True)
    torch.save(cells.make_judge(0, "W").state_dict(), path / "judge.pt")
    rec = evaluate.run_cell("J3V-W", 7, 0, tmp_path / "out", limit_states=1, judge_dir=jdir)
    assert rec["arm"] == "J3V-W" and rec["n"] == 1 and rec["mode"] == "C"
    assert (tmp_path / "out" / "exp074_J3V-W_d7_s0.json").exists()
