"""EXP-084: the GPU port's gates (C, P, E, M, T), aggregator and drivers. The device code itself is
tested in tests/training/test_device_port.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "084_gpu_port"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp084_cells", EXP / "cells.py")
gatec = _load("exp084_gatec", EXP / "gatec.py")


def test_names():
    assert cells.GATEC_NAME == "exp084_gatec.json"
    assert cells.p_name("J4V-X16k", 3) == "exp084_P_J4V-X16k_s3.json"
    assert cells.e_name("J4V-X", 11) == "exp084_E_J4V-X_d9_s11.json"
    assert cells.E_GATER_NAME == "exp084_E_gateR_s0.json"
    assert cells.M_NAME == "exp084_M_s0.json"
    assert cells.train_name(5) == "exp084_train_X_s5.json"
    assert cells.ckpt_dir(5) == cells.OUT / "judge_X_s5"
    assert (cells.TOL, cells.P_LIMIT, cells.N_UPDATES) == (2, 9, 4000)
    assert cells.ARMS == ("J4V-X16k", "J4V-X") and list(cells.SEEDS) == list(range(12))


def _fp():
    return {"encode_sha": "a", "judge_vals": ["0.5", "0.25"], "step_loss": "0.7",
            "step_param_sha": "b", "probe": {"spearman_7_11": 0.1, "mean_j_by_distance": {"7": 1.0}},
            "search": {"solved": 12, "optimality": 0.9}}


def test_compare_equal_is_empty():
    assert gatec.compare(_fp(), _fp()) == []


def test_compare_flags_one_float():
    b = _fp()
    b["probe"]["mean_j_by_distance"]["7"] = 1.0000000000000002
    assert gatec.compare(_fp(), b) == ["probe"]


def test_files_under_root(tmp_path):
    inside = (tmp_path / "experiments" / "x.py").as_posix()
    assert gatec.files_under({"module_files": [inside]}, tmp_path)
    assert not gatec.files_under({"module_files": [inside, "/elsewhere/y.py"]}, tmp_path)
