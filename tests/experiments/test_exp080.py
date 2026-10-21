"""EXP-080: oracle judges trained on the true distance, as instruments for X's far ordering."""

from __future__ import annotations

import importlib.util
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.encoder_pretrain import save_encoder
from neuromorphic.training.lookahead import tree_levels

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "080_oracle_ordering"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp080_cells", EXP / "cells.py")
c75 = cells.c75


def _fake_x(d, seed):
    """EXP-075's layout for one seed under `d`: the pretrained encoder, and an X judge at 4000
    updates whose region has moved away from it (so 'X's trained region' and 'the pretrained
    encoder' are different tensors)."""
    enc = c75.fresh_sensory("X", seed)
    path = c75.encoder_path("X", seed, d)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder(enc, path)
    judge = c75.build_judge(seed, "X", c75.fresh_sensory("X", seed))
    with torch.no_grad():
        judge.sensory.fc1.weight.add_(0.01)
        judge.head[0].weight.add_(0.5)
    ck = c75.ckpt_dir("X", seed, d)
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(judge.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": 4000}))
    return judge


def test_constants_are_the_spec_values():
    """Catches an arm, budget, walk length or seed set drifting from spec section 3."""
    assert cells.ARMS == ("O-raw", "O-Xf", "O-Xt")
    assert cells.TRAIN_ORDER == ("O-Xt", "O-Xf", "O-raw")
    assert (cells.MAX_LEN, cells.N_UPDATES, cells.BATCH, cells.PROBE_EVERY) == (40, 4000, 1000, 250)
    assert (cells.RAW_STATES, cells.RAW_EPOCHS, cells.RAW_BATCH, cells.RAW_HIDDEN) == (
        600_000, 12, 512, 512)
    assert cells.EVAL_SEEDS == tuple(range(12)) and cells.PILOT_SEEDS == (12, 13)
    assert cells.x_judge_root(False) == cells.E75_OUT and cells.x_judge_root(True) == cells.E75_PILOT
    assert cells.encoder_root(False) == cells.E75_OUT
    assert cells.record_name("O-Xt", 3) == "exp080_train_O-Xt_s3.json"
    assert cells.ckpt_dir("O-Xf", 3, Path("/x")) == Path("/x/judge_O-Xf_s3")


def test_leaf_exclusion_is_every_3_move_leaf_of_probe_states_7_to_11():
    """Catches leaves taken at the wrong depth, from probe states outside 7 to 11, or not at all
    (the label leak spec section 3 closes)."""
    a = apply_move(SOLVED, 0)
    b = apply_move(SOLVED, 1)
    c = apply_move(SOLVED, 2)
    probe = [(a, 7), (b, 11), (c, 6)]
    got = cells.leaf_exclusion(probe)
    want = set(tree_levels(a, 3, N_ACTIONS)[3]) | set(tree_levels(b, 3, N_ACTIONS)[3])
    assert got == want
    assert not set(tree_levels(c, 3, N_ACTIONS)[3]) <= got  # distance 6 contributes nothing


def test_oracle_exclusion_is_heldout_plus_probe_plus_leaves(monkeypatch):
    """Catches an exclusion that drops the held-out states, the probe states or their leaves."""
    a = apply_move(SOLVED, 0)
    held = {apply_move(SOLVED, 5)}
    monkeypatch.setattr(cells, "exclusion_set", lambda seed, provider: set(held))
    monkeypatch.setattr(cells, "probe_set", lambda seed, provider: [(a, 9)])
    got = cells.oracle_exclusion(0, None)
    assert held <= got and a in got
    assert set(tree_levels(a, 3, N_ACTIONS)[3]) <= got
    assert got == held | {a} | set(tree_levels(a, 3, N_ACTIONS)[3])


def test_raw_judge_shape_seed_and_generator():
    """Catches an O-raw net of the wrong shape, an init not fixed by the seed, or a forward that
    cannot be called the way leaf_rank_margin and probe_judge call a judge."""
    j0, j0b, j1 = cells.make_oraw(0), cells.make_oraw(0), cells.make_oraw(1)
    shapes = [tuple(p.shape) for p in j0.parameters()]
    assert shapes == [(512, 144), (512,), (512, 512), (512,), (1, 512), (1,)]
    assert all(torch.equal(p, q) for p, q in zip(j0.parameters(), j0b.parameters()))
    assert not torch.equal(j0.net[0].weight, j1.net[0].weight)
    states = [SOLVED, apply_move(SOLVED, 3)]
    out = j0(states, torch.Generator().manual_seed(0))
    assert out.shape == (2,)
    assert torch.equal(out, j0(np.array(states, dtype=np.uint8)))
    oh = cells.RawJudge.one_hot([SOLVED])
    assert oh.shape == (1, 144) and float(oh.sum()) == 24.0


def test_oxf_is_xs_trained_region_under_oxts_fresh_head(tmp_path):
    """Catches O-Xf built on X's trained head, on the pretrained encoder instead of X's trained
    region, or with a region that is not frozen; and O-Xt not built from the encoder."""
    x = _fake_x(tmp_path, 12)
    oxf = cells.make_oxf(12, tmp_path)
    oxt = cells.make_oxt(12, tmp_path)
    for k, v in oxf.head.state_dict().items():
        assert torch.equal(v, oxt.head.state_dict()[k])
    assert not torch.equal(oxf.head[0].weight, x.head[0].weight)
    assert torch.equal(oxf.sensory.fc1.weight, x.sensory.fc1.weight)
    assert not torch.equal(oxt.sensory.fc1.weight, x.sensory.fc1.weight)
    enc = c75.load_pretrained("X", 12, tmp_path)
    assert torch.equal(oxt.sensory.fc1.weight, enc.fc1.weight)
    assert all(not p.requires_grad for p in oxf.sensory.parameters())
    assert all(p.requires_grad for p in oxt.sensory.parameters())


def test_optimizers_are_the_spec_ones(tmp_path):
    """Catches O-Xf's optimizer touching the region, O-Xt's missing it, or a wrong lr."""
    _fake_x(tmp_path, 12)
    oxf, oxt, raw = (cells.make_judge(a, 12, tmp_path, tmp_path) for a in ("O-Xf", "O-Xt", "O-raw"))
    f, t, r = (cells.optimizer_for(a, j) for a, j in (("O-Xf", oxf), ("O-Xt", oxt), ("O-raw", raw)))
    f_params = {id(p) for g in f.param_groups for p in g["params"]}
    assert f_params == {id(p) for p in oxf.head.parameters()}
    assert [g["lr"] for g in t.param_groups] == [1e-3, 1e-4]
    assert {id(p) for p in t.param_groups[1]["params"]} == {id(p) for p in oxt.sensory.parameters()}
    assert [g["lr"] for g in r.param_groups] == [1e-3]
    with pytest.raises(ValueError):
        cells.optimizer_for("S", raw)


def test_reference_sensory_is_xs_region_for_oxf_and_the_encoder_for_oxt(tmp_path):
    """Catches a drift measured from the wrong starting point, which would make O-Xf's 0.0 or
    O-Xt's > 0 meaningless."""
    x = _fake_x(tmp_path, 12)
    rf = cells.reference_sensory("O-Xf", 12, tmp_path, tmp_path)
    rt = cells.reference_sensory("O-Xt", 12, tmp_path, tmp_path)
    assert torch.equal(rf["fc1.weight"], x.sensory.fc1.weight)
    assert torch.equal(rt["fc1.weight"], c75.load_pretrained("X", 12, tmp_path).fc1.weight)
    assert cells.drift(cells.make_oxf(12, tmp_path).sensory.state_dict(), rf) == 0.0
    assert cells.drift(cells.make_oxt(12, tmp_path).sensory.state_dict(), rt) == 0.0


def test_load_trained_refuses_a_missing_or_unfinished_judge(tmp_path):
    """Catches evaluating a judge that is absent or short of its budget (judge.pt exists from the
    first checkpoint, so its existence proves nothing)."""
    with pytest.raises(SystemExit, match="missing"):
        cells.load_trained("O-Xt", 12, tmp_path)
    j = cells.make_oraw(12)
    ck = cells.ckpt_dir("O-raw", 12, tmp_path)
    ck.mkdir(parents=True)
    torch.save(j.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"epochs": 3}))
    with pytest.raises(SystemExit, match="epochs 3"):
        cells.load_trained("O-raw", 12, tmp_path)
    got = cells.load_trained("O-raw", 12, tmp_path, required=3)
    assert torch.equal(got.net[0].weight, j.net[0].weight) and not got.training
    x = _fake_x(tmp_path / "x", 12)
    ck = cells.ckpt_dir("O-Xt", 12, tmp_path)
    ck.mkdir(parents=True)
    torch.save(x.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": 250}))
    with pytest.raises(SystemExit, match="update 250"):
        cells.load_trained("O-Xt", 12, tmp_path)
    (ck / "state.json").write_text(json.dumps({"update": 4000}))
    got = cells.load_trained("O-Xt", 12, tmp_path)
    assert torch.equal(got.sensory.fc1.weight, x.sensory.fc1.weight)
