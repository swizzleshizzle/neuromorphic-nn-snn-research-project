"""EXP-075: cells, pretraining, training driver, evaluation and aggregator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.encoder_pretrain import make_sensory, save_encoder

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "075_wide_region"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp075_cells", "cells.py")
pretrain = _load("exp075_pretrain", "pretrain.py")


def _fake_encoder(out_dir, arm, seed, hidden=None):
    path = cells.encoder_path(arm, seed, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder(make_sensory(seed, hidden=hidden or cells.HIDDEN[arm]), path)
    return path


def test_arms_widths_and_seeds_are_the_spec_values():
    """Catches a width, arm list or seed set drifting from spec sections 3 and 6."""
    assert cells.ARMS_TRAIN == ("X", "Y")
    assert cells.HIDDEN == {"X": 512, "Y": 128}
    assert (cells.CONTENT, cells.T) == (64, 32)
    assert tuple(cells.EVAL_SEEDS) == tuple(range(12)) and tuple(cells.PILOT_SEEDS) == (12, 13)
    assert tuple(cells.SENSITIVITY_DROP) == (0, 3)


def test_check_seeds_refuses_misuse():
    """Catches a pilot run on an evaluation seed, or an evaluation run on a pilot seed."""
    with pytest.raises(SystemExit):
        cells.check_seeds([0], pilot=True)
    with pytest.raises(SystemExit):
        cells.check_seeds([12], pilot=False)
    cells.check_seeds([12, 13], pilot=True)
    cells.check_seeds(list(range(12)), pilot=False)


def test_judges_have_the_wide_readout_at_their_width(tmp_path):
    """Catches X built at 128, a concept-only readout, or a head of the wrong input width.
    Head sizes from spec section 3: X 73,985 parameters (576 inputs), Y 24,833 (192)."""
    for arm, n_in, n_head in (("X", 576, 73_985), ("Y", 192, 24_833)):
        _fake_encoder(tmp_path, arm, 0)
        j = cells.make_judge(0, arm, tmp_path)
        assert j.readout == "wide"
        assert j.head[0].in_features == n_in
        assert sum(p.numel() for p in j.head.parameters()) == n_head


def test_make_judge_starts_from_the_pretrained_file_exactly(tmp_path):
    """Catches a judge that starts from a fresh random region instead of its pretrained one."""
    path = _fake_encoder(tmp_path, "X", 2)
    ref = torch.load(path)
    sens = cells.make_judge(2, "X", tmp_path).sensory.state_dict()
    for k, v in ref.items():
        assert torch.equal(sens[k], v), k
    fresh = cells.fresh_sensory("X", 3).state_dict()
    assert not torch.equal(fresh["fc1.weight"], ref["fc1.weight"])


def test_head_init_depends_only_on_the_seed(tmp_path):
    """Catches a head whose init depends on global RNG state left by earlier construction."""
    _fake_encoder(tmp_path, "Y", 5)
    torch.manual_seed(999)
    a = cells.make_judge(5, "Y", tmp_path).head.state_dict()
    torch.randn(100)
    b = cells.make_judge(5, "Y", tmp_path).head.state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_wrong_width_encoder_is_refused(tmp_path):
    """Catches a 128-wide file under X's name loading as a 128-wide 'X' (Review Focus 2)."""
    _fake_encoder(tmp_path, "X", 0, hidden=128)
    with pytest.raises(RuntimeError):
        cells.make_judge(0, "X", tmp_path)


def test_missing_encoder_exits_naming_the_file(tmp_path):
    """Catches a judge built on a random region because pretraining never ran."""
    with pytest.raises(SystemExit, match="enc_X_s0.pt"):
        cells.make_judge(0, "X", tmp_path)


def test_pretrain_states_and_pairs_without_exclusion_are_exp039s_recipe():
    """Catches the pretraining set drifting from EXP-039's (Y must be that recipe at 128).
    Checked against EXP-039's own recorded counts, not a re-implementation."""
    rec_path = REPO / "experiments" / "039_encoder_pretraining" / "outputs" / "exp039_s0.json"
    if not rec_path.exists():
        pytest.skip("EXP-039 records are untracked; present on the VPS")
    rec = json.loads(rec_path.read_text())
    prov7 = ExactBFSDistance(max_depth=7)
    states, depths = pretrain.pretrain_states(prov7)
    s39, _masks, d39 = pretrain.e39.build_dataset(prov7)
    assert states == s39 and depths == d39
    assert len(pretrain.probe_heldout(states, depths, 0)) == rec["n_heldout"]
    pairs, _forbidden = pretrain.pretrain_pairs(0, states, depths, set())
    assert len(pairs) == rec["n_pairs"]


def test_an_excluded_evaluation_state_removes_exactly_its_pairs():
    """Catches the evaluation exclusion being ignored, or applied to sources only."""
    prov7 = ExactBFSDistance(max_depth=7)
    states, depths = pretrain.pretrain_states(prov7)
    base, _ = pretrain.pretrain_pairs(0, states, depths, set())
    target = next(p[2] for p in base if prov7.distance(p[2]) == 7)
    pairs, forbidden = pretrain.pretrain_pairs(0, states, depths, {target})
    assert target in forbidden
    assert all(target not in (p[0], p[2]) for p in pairs)
    dropped = sum(1 for p in base if p[2] == target)
    assert dropped >= 1 and len(base) - len(pairs) == dropped


def test_pretrain_relaunch_skips_cells_whose_record_exists(tmp_path):
    """Catches a relaunch that re-pretrains a finished seed (Review Focus 3)."""
    (tmp_path / cells.pretrain_record_name("Y", 12)).write_text("{}")
    assert pretrain.pending_jobs(["X", "Y"], [12, 13], tmp_path) == [
        ("X", 12), ("X", 13), ("Y", 13)]


def test_pretrain_parser_defaults_are_the_spec_recipe():
    """Catches a hand-run pretraining at other than EXP-039's 40 epochs."""
    a = pretrain.build_parser().parse_args([])
    assert a.epochs == 40 and a.arms == ["X", "Y"]
    assert (pretrain.EPOCHS, pretrain.BATCH, pretrain.LR) == (40, 256, 3e-3)


@pytest.mark.slow
def test_pretrain_run_writes_an_encoder_at_the_arm_width(tmp_path):
    """Catches a record without its accuracy history, an encoder saved at the wrong width, or
    an encoder that training never moved. Slow: builds the full BFS table and runs 1 epoch."""
    rec = pretrain.run("X", 12, tmp_path, epochs=1)
    on_disk = json.loads((tmp_path / cells.pretrain_record_name("X", 12)).read_text())
    assert on_disk["hidden"] == 512 and on_disk["epochs"] == 1
    assert len(on_disk["history"]) == 1
    assert on_disk["final_move_accuracy"] == on_disk["history"][-1]["accuracy"]
    assert on_disk["n_pairs"] <= on_disk["n_pairs_exp039_recipe"]
    trained = cells.load_pretrained("X", 12, tmp_path)
    assert trained.fc1.out_features == 512
    untrained = cells.fresh_sensory("X", 12)
    assert not torch.equal(trained.fc1.weight, untrained.fc1.weight)
    assert rec["final_move_accuracy"] == on_disk["final_move_accuracy"]


train = _load("exp075_train", "train.py")


def test_train_parser_defaults_are_the_spec_settings():
    """Catches driver defaults drifting from spec section 5."""
    a = train.build_parser().parse_args(["--n-updates", "1"])
    assert (a.batch, a.sync_every, a.probe_every, a.draws) == (1000, 100, 250, 1)
    assert a.arms == ["X", "Y"] and train.MAX_LEN == 14


def test_one_update_moves_the_region_for_both_arms(tmp_path):
    """REVIEW FOCUS 1. Catches X or Y trained with a frozen region: the gradient must ARRIVE at
    fc1 (CLAUDE.md: verify the parameter moved, not that a switch is set)."""
    import random
    from neuromorphic.training import value_iteration as vi
    for arm in ("X", "Y"):
        _fake_encoder(tmp_path, arm, 0)
        judge = cells.make_judge(0, arm, tmp_path)
        before = judge.sensory.fc1.weight.detach().clone()
        opt = vi.make_optimizer(judge, train.VI_ARM)
        jt = vi.sync_target(judge)
        states = vi.random_walk_states(8, 14, random.Random(0), N_ACTIONS, set())
        vi.train_step(judge, jt, opt, states, N_ACTIONS, torch.Generator().manual_seed(0))
        assert not torch.equal(before, judge.sensory.fc1.weight.detach()), arm
        lrs = sorted(g["lr"] for g in opt.param_groups)
        assert lrs == [1e-4, 1e-3], arm


def test_train_refuses_a_missing_encoder_before_building_anything(tmp_path):
    """Catches training on a random region when pretraining was skipped, and catches the check
    running only after the 65 s BFS build."""
    import time
    t0 = time.time()
    with pytest.raises(SystemExit, match="enc_Y_s12.pt"):
        train.run("Y", 12, 1, 8, 1, 1, 1, tmp_path)
    assert time.time() - t0 < 20


def test_train_relaunch_skips_runs_whose_record_exists(tmp_path):
    """Catches a relaunch that resubmits finished runs (Review Focus 3)."""
    (tmp_path / cells.record_name("X", 13)).write_text("{}")
    assert train.pending_jobs(["X", "Y"], [12, 13], tmp_path) == [
        ("X", 12), ("Y", 12), ("Y", 13)]


@pytest.mark.slow
def test_run_writes_gate_l_drift_width_and_readout(tmp_path):
    """Catches a record missing Gate L, the drift Gate E reads, or the width. Slow: builds the
    full BFS table, as the real driver does."""
    _fake_encoder(tmp_path, "X", 12)
    rec = train.run("X", 12, 2, 8, 1, 1, 1, tmp_path)
    on_disk = json.loads((tmp_path / cells.record_name("X", 12)).read_text())
    assert on_disk["readout"] == "wide" and on_disk["hidden"] == 512
    assert on_disk["gate_l"]["n"] == 250
    assert 0.05 < on_disk["gate_l"]["chance"] < 0.6
    assert on_disk["encoder_drift"] > 0.0
    assert (tmp_path / "judge_X_s12" / "judge.pt").exists()
    assert rec["gate_l"] == on_disk["gate_l"]
