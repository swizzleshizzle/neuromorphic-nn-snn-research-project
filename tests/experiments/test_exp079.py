"""EXP-079: judge S, EXP-075's X recipe trained on walks of length 1 to 40."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.encoder_pretrain import save_encoder

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "079_judge_trained_deep"
E75 = REPO / "experiments" / "075_wide_region"
E76 = REPO / "experiments" / "076_deeper_tree"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp079_cells", EXP / "cells.py")


def _fake_x_pilot(d, seed, perturb=0.0):
    """EXP-075's pilot layout for one seed under `d`: encoder, X judge whose region has moved,
    and a training record whose encoder_drift is the true drift (plus `perturb`)."""
    enc = cells.c75.fresh_sensory("X", seed)
    path = cells.c75.encoder_path("X", seed, d)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder(enc, path)
    judge = cells.c75.build_judge(seed, "X", cells.c75.fresh_sensory("X", seed))
    with torch.no_grad():
        judge.sensory.fc1.weight.add_(0.01)
    ck = cells.c75.ckpt_dir("X", seed, d)
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(judge.state_dict(), ck / "judge.pt")
    e0 = cells.c75.load_pretrained("X", seed, d).state_dict()
    true = cells.drift(judge.sensory.state_dict(), e0)
    (d / cells.c75.record_name("X", seed)).write_text(
        json.dumps({"encoder_drift": true + perturb}))
    return true


def test_constants_are_the_spec_values():
    """Catches the walk length, update count, seeds or source folders drifting from spec
    sections 3 and 6."""
    assert cells.MAX_LEN == 40 and cells.N_UPDATES == 4000
    assert cells.ARM == "S" and cells.BASE_ARM == "X"
    assert cells.EVAL_SEEDS == tuple(range(12)) and cells.PILOT_SEEDS == (12, 13)
    assert cells.E75_OUT.name == "outputs" and cells.E75_OUT.parent.name == "075_wide_region"
    assert cells.E76_OUT.parent.name == "076_deeper_tree"
    assert cells.encoder_root(False) == cells.E75_OUT
    assert cells.encoder_root(True) == cells.PILOT_OUT
    assert cells.record_name(3) == "exp079_train_S_s3.json"
    assert cells.ckpt_dir(3, Path("x")) == Path("x") / "judge_S_s3"


def test_make_judge_is_x_built_from_the_committed_encoder():
    """Catches S built at another width, with another readout, from another encoder, or with a
    head not seeded like X's (spec section 3: the only difference is max_len)."""
    s = cells.make_judge(0, cells.E75_OUT)
    x = cells.c75.make_judge(0, "X", cells.E75_OUT)
    assert s.readout == "wide" and s.sensory.fc1.out_features == 512
    sd_s, sd_x = s.state_dict(), x.state_dict()
    assert sd_s.keys() == sd_x.keys()
    for k in sd_s:
        assert torch.equal(sd_s[k], sd_x[k]), k


def test_load_trained_refuses_a_partly_trained_or_missing_judge(tmp_path):
    """Catches an S judge evaluated before training finished (judge.pt exists from update 0)."""
    with pytest.raises(SystemExit, match="judge_S_s0"):
        cells.load_trained(0, tmp_path)
    ck = cells.ckpt_dir(0, tmp_path)
    ck.mkdir(parents=True)
    judge = cells.c75.build_judge(0, "X", cells.c75.fresh_sensory("X", 0))
    torch.save(judge.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": 3999}))
    with pytest.raises(SystemExit, match="3999"):
        cells.load_trained(0, tmp_path)
    (ck / "state.json").write_text(json.dumps({"update": 4000}))
    loaded = cells.load_trained(0, tmp_path)
    assert torch.equal(loaded.head[0].weight, judge.head[0].weight)
    assert cells.judge_updates(0, tmp_path) == 4000


def test_drift_is_exp075s_l2_over_the_encoder_tensors():
    """Catches a drift that is not EXP-075's (so the pilot proof would compare different
    quantities): one tensor moved by 0.01 everywhere gives 0.01 * sqrt(numel)."""
    a = {"w": torch.zeros(4, 25), "b": torch.zeros(3)}
    b = {"w": torch.full((4, 25), 0.01), "b": torch.zeros(3)}
    assert cells.drift(b, a) == pytest.approx(0.01 * 10.0, rel=1e-6)


def test_pilot_encoder_is_proved_only_when_the_drift_reproduces(tmp_path):
    """Catches a pilot encoder accepted without checking it is the one EXP-075's pilot record
    describes (spec section 6), and a missing file read as a match."""
    _fake_x_pilot(tmp_path, 12)
    ok, why = cells.pilot_encoder_matches(12, tmp_path)
    assert ok, why
    _fake_x_pilot(tmp_path, 13, perturb=1e-3)
    ok, why = cells.pilot_encoder_matches(13, tmp_path)
    assert not ok and "seed 13" in why
    ok, why = cells.pilot_encoder_matches(12, tmp_path / "nowhere")
    assert not ok and "missing" in why


def test_pilot_encoders_copy_all_or_nothing(tmp_path):
    """REVIEW FOCUS 3. Catches one proved encoder copied while the other failed, which would mix
    a copied encoder with a re-pretrained one in the same pilot."""
    src, dest = tmp_path / "e75", tmp_path / "e79"
    _fake_x_pilot(src, 12)
    _fake_x_pilot(src, 13, perturb=1e-3)
    with pytest.raises(SystemExit, match="pilot-pretrain"):
        cells.prepare_pilot_encoders(src, dest)
    assert not cells.c75.encoder_path("X", 12, dest).exists()
    _fake_x_pilot(src, 13)
    lines = cells.prepare_pilot_encoders(src, dest)
    assert len(lines) == 2
    for s in (12, 13):
        a = cells.c75.encoder_path("X", s, src).read_bytes()
        assert cells.c75.encoder_path("X", s, dest).read_bytes() == a
