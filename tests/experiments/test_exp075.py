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


evaluate = _load("exp075_evaluate", "evaluate.py")


def _fake_judge_ckpt(jdir, arm, seed):
    path = cells.ckpt_dir(arm, seed, jdir)
    path.mkdir(parents=True)
    judge = cells.build_judge(seed, arm, cells.fresh_sensory(arm, seed))
    torch.save(judge.state_dict(), path / "judge.pt")
    return judge


def test_eval_arms_and_rank_kinds_are_the_spec_lists():
    assert list(evaluate.ARMS_EVAL) == ["J3V-X", "J3V-Y", "J3V-W"]
    assert list(evaluate.RANK_KINDS) == ["J-X", "J-Y"]


def test_judge_loads_its_checkpoint_not_its_pretrained_encoder(tmp_path):
    """Catches an evaluator that rebuilds the judge from the pretrained file and evaluates an
    untrained head, or needs the encoder file to exist at evaluation time."""
    saved = _fake_judge_ckpt(tmp_path, "X", 0).state_dict()
    loaded = evaluate.load_judge("X", 0, tmp_path).state_dict()
    for k, v in saved.items():
        assert torch.equal(loaded[k], v), k


def test_j3v_w_is_exp074s_committed_judge():
    """THE GATE 0(b) PRECONDITION. Catches W rebuilt or retrained instead of reused."""
    ref = torch.load(cells.E74_OUT / "judge_W_s0" / "judge.pt", map_location="cpu")
    mine = evaluate.load_judge("W", 0).state_dict()
    assert mine.keys() == ref.keys()
    for k in ref:
        assert torch.equal(mine[k], ref[k]), k


def test_missing_checkpoint_exits_naming_it(tmp_path):
    """Catches a J cell silently evaluating an untrained judge."""
    with pytest.raises(SystemExit, match="judge.pt"):
        evaluate.run_cell("J3V-Y", 8, 0, tmp_path, limit_states=1, judge_dir=tmp_path / "no")


def test_an_x_judge_drives_a_real_j3v_cell(tmp_path):
    """Catches X evaluated through the 64-unit concept: a 576-input head cannot read it."""
    jdir = tmp_path / "judges"
    _fake_judge_ckpt(jdir, "X", 0)
    rec = evaluate.run_cell("J3V-X", 7, 0, tmp_path / "out", limit_states=1, judge_dir=jdir)
    assert rec["arm"] == "J3V-X" and rec["n"] == 1 and rec["mode"] == "C"
    assert rec["limit_states"] == 1
    assert (tmp_path / "out" / "exp075_J3V-X_d7_s0.json").exists()


def test_j3v_w_cell_equals_exp074s_harness_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH: J3V-W through this harness equals EXP-074's own run_cell on
    the same states, in every outcome field."""
    mine = evaluate.run_cell("J3V-W", 9, 0, tmp_path / "a", limit_states=2)
    ref = evaluate.e74.run_cell("J3V-W", 9, 0, tmp_path / "b", limit_states=2,
                                judge_dir=cells.E74_OUT)
    for f in ("solved", "n", "success_rate", "mean_steps", "eval_revisit_rate"):
        assert mine[f] == ref[f], f


def test_rank_cell_records_hit_chance_and_its_limit(tmp_path):
    """Catches a rank record without `limit_states` (so a smoke row could enter Gate R), and a
    chance that is not the fraction of closer leaves."""
    jdir = tmp_path / "judges"
    _fake_judge_ckpt(jdir, "Y", 0)
    rec = evaluate.rank_cell("J-Y", 7, 0, tmp_path / "out", judge_dir=jdir, limit_states=2)
    assert rec["n"] == 2 and rec["limit_states"] == 2
    assert 0.0 < rec["chance"] < 0.5 and rec["hit"] in (0.0, 0.5, 1.0)
    assert (tmp_path / "out" / "exp075_rank_J-Y_d7_s0.json").exists()


agg = _load("exp075_aggregate", "aggregate.py")

_SETTINGS = {"n_updates": 4000, "batch": 1000, "sync_every": 100, "probe_every": 250, "draws": 1}


def _rates(x9, w9, y9, x11, w11, seeds=range(12)):
    return {("J3V-X", 9): {s: x9 for s in seeds}, ("J3V-W", 9): {s: w9 for s in seeds},
            ("J3V-Y", 9): {s: y9 for s in seeds}, ("J3V-X", 11): {s: x11 for s in seeds},
            ("J3V-W", 11): {s: w11 for s in seeds}}


def _gates(ok=True, **over):
    g = {"gate0": ok, "p": {"X": True, "Y": True}, "l": {"X": True, "Y": True},
         "e": {"X": True, "Y": True},
         "r": {(k, d): True for k in ("J-X", "J-Y") for d in (7, 8, 9, 11)}}
    g.update(over)
    return g


def _noisy(base, seeds=range(12)):
    return {s: base + 0.01 * ((s % 3) - 1) for s in seeds}


def test_alpha_is_a_third_of_005_and_the_verdict_uses_it(monkeypatch):
    """Catches the claims read at EXP-071's 0.025 instead of this spec's 0.05 / 3."""
    assert agg.ALPHA == pytest.approx(0.05 / 3)
    diffs = [0.05] * 12
    assert agg.claim_verdict(diffs, 0.3, 0.25, True)[0] == "CONFIRMED"
    monkeypatch.setattr(agg, "ALPHA", 0.0)
    assert agg.claim_verdict(diffs, 0.3, 0.25, True)[0] == "NOT SIGNIFICANT"


def test_claims_compare_the_registered_arms_and_depths():
    """Catches a claim wired to the wrong arm or depth: each claim's two means are distinct."""
    r = _rates(0.30, 0.20, 0.25, 0.15, 0.10)
    r[("J3V-X", 9)] = _noisy(0.30)
    r[("J3V-X", 11)] = _noisy(0.15)
    v = agg.primary_verdicts(r, _gates(), range(12))
    assert v["claim1"][3:] == pytest.approx((0.30, 0.20))
    assert v["claim2"][3:] == pytest.approx((0.30, 0.25))
    assert v["claim3"][3:] == pytest.approx((0.15, 0.10))
    assert all(v[c][0] == "CONFIRMED" for c in ("claim1", "claim2", "claim3"))


@pytest.mark.parametrize("over,void", [
    ({"gate0": False}, {"claim1", "claim2", "claim3"}),
    ({"p": {"X": False, "Y": True}}, {"claim1", "claim2", "claim3"}),
    ({"l": {"X": True, "Y": False}}, {"claim2"}),
    ({"e": {"X": True, "Y": False}}, {"claim2"}),
    ({"r": {**{(k, d): True for k in ("J-X", "J-Y") for d in (7, 8, 9, 11)},
            ("J-X", 11): False}}, {"claim3"}),
])
def test_a_failed_gate_voids_exactly_the_claims_it_guards(over, void):
    """Catches a gate that voids too little (a claim read on a broken arm) or too much."""
    r = _rates(0.30, 0.20, 0.25, 0.15, 0.10)
    v = agg.primary_verdicts(r, _gates(**over), range(12))
    assert {c for c in ("claim1", "claim2", "claim3") if v[c][0] == "VOID"} == void


def test_a_contrast_on_the_floor_is_unresolved():
    """Catches the Gate 1 band living only in prose (CLAUDE.md, EXP-068)."""
    r = _rates(0.012, 0.010, 0.011, 0.012, 0.010)
    assert agg.primary_verdicts(r, _gates(), range(12))["claim1"][0] == "UNRESOLVED"


def test_sensitivity_drops_exactly_seeds_0_and_3():
    r = agg.sensitivity_rates(_rates(0.3, 0.2, 0.25, 0.15, 0.1))
    assert sorted(r[("J3V-X", 9)]) == [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]


def test_gate_p_requires_every_seed():
    """Catches Gate P read as a mean (one failed pretraining hidden by eleven good ones)."""
    assert agg.gate_p_verdict([0.45] * 12, 0.40) is True
    assert agg.gate_p_verdict([0.45] * 11 + [0.39], 0.40) is False


def test_gate_e_requires_every_drift_positive():
    assert agg.gate_e_verdict([0.1] * 12) is True
    assert agg.gate_e_verdict([0.1] * 11 + [0.0]) is False


def test_thresholds_unset_block_every_verdict(monkeypatch):
    """REVIEW FOCUS 5. Catches a verdict read before the dated amendment."""
    monkeypatch.setattr(agg, "GATE_P_THRESHOLD", None)
    monkeypatch.setattr(agg, "GATE_L_THRESHOLD", {"X": 0.1, "Y": 0.1})
    with pytest.raises(SystemExit, match="GATE_P_THRESHOLD"):
        agg.require_thresholds()
    monkeypatch.setattr(agg, "GATE_P_THRESHOLD", {"X": 0.4, "Y": 0.4})
    monkeypatch.setattr(agg, "GATE_L_THRESHOLD", None)
    with pytest.raises(SystemExit, match="GATE_L_THRESHOLD"):
        agg.require_thresholds()


def _write_pilot(d, acc, margin):
    for arm in ("X", "Y"):
        for i, s in enumerate((12, 13)):
            (d / cells.pretrain_record_name(arm, s)).write_text(json.dumps(
                {"arm": arm, "seed": s, "final_move_accuracy": acc[arm][i]}))
            (d / cells.record_name(arm, s)).write_text(json.dumps(
                {"arm": arm, "seed": s, "gate_l": {"margin": margin[arm][i]}}))


def test_pilot_amendment_numbers_follow_the_spec_forms(tmp_path):
    """Catches a threshold form other than spec section 7's (0.9 x mean pilot accuracy, half
    the mean pilot margin), and a missing Y working-control check (spec section 6)."""
    _write_pilot(tmp_path, {"X": [0.50, 0.52], "Y": [0.45, 0.46]},
                 {"X": [0.30, 0.26], "Y": [0.20, 0.18]})
    a = agg.pilot_amendment(tmp_path)
    assert a["gate_p"]["X"] == pytest.approx(0.9 * 0.51)
    assert a["gate_l"]["Y"] == pytest.approx(0.19 / 2)
    assert a["pilot_failed"] is False and a["y_working_control"] is True
    _write_pilot(tmp_path, {"X": [0.25, 0.28], "Y": [0.45, 0.46]},
                 {"X": [0.30, 0.26], "Y": [0.05, 0.06]})
    a = agg.pilot_amendment(tmp_path)
    assert a["pilot_failed"] is True and a["y_working_control"] is False


def test_records_at_other_settings_or_smoke_records_are_refused(tmp_path):
    """REVIEW FOCUS 4. Catches a smoke cell, a record at other settings, or the wrong width
    entering a verdict."""
    good = {**_SETTINGS, "arm": "X", "seed": 0, "readout": "wide", "hidden": 512}
    agg.check_train_record(good)
    for bad in ({"hidden": 128}, {"readout": "concept"}, {"n_updates": 2}):
        with pytest.raises(SystemExit):
            agg.check_train_record({**good, **bad})
    pre = {"arm": "Y", "seed": 0, "epochs": 40, "batch_size": 256, "lr": 3e-3, "hidden": 128}
    agg.check_pretrain_record(pre)
    with pytest.raises(SystemExit):
        agg.check_pretrain_record({**pre, "epochs": 1})
    p = tmp_path / "exp075_J3V-X_d9_s0.json"
    p.write_text(json.dumps({"success_rate": 0.2, "limit_states": 1}))
    with pytest.raises(SystemExit, match="smoke"):
        agg._read(p)


def test_launcher_carries_the_exit_code_fix_and_the_registered_cells():
    """Catches a launcher copied without EXP-074's .Handle fix (every cell counted as failed)
    or with an eval list missing an arm or depth."""
    text = (EXP / "launch075.ps1").read_text()
    assert "$null = $proc.Handle" in text
    assert '"J3V-X", "J3V-Y"' in text and "7, 8, 9, 11" in text
    assert "exp075-det" in text and "GATE_P_THRESHOLD" in text
