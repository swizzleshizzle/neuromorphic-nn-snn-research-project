"""EXP-082: EXP-075's X judge continued by value iteration from 4000 to 16,000 updates."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "082_x_long_budget"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp082_cells", EXP / "cells.py")
c75 = cells.c75
t75 = _load("exp075_train", REPO / "experiments" / "075_wide_region" / "train.py")


def test_constants_are_the_spec_values():
    """Catches a setting drifting from EXP-075's X: a continuation on other settings is not the
    same run as a straight 16,000-update X."""
    a = t75.build_parser().parse_args(["--n-updates", "1"])
    assert (cells.BATCH, cells.SYNC_EVERY, cells.PROBE_EVERY, cells.DRAWS) == (
        a.batch, a.sync_every, a.probe_every, a.draws) == (1000, 100, 250, 1)
    assert t75.MAX_LEN == 14 and t75.VI_ARM == "W"
    assert cells.START == c75.N_UPDATES == 4000 and cells.LEGS == (8000, 16000)
    assert cells.DEPTHS == (9, 11) and cells.ARM == "X16k" and cells.SRC_ARM == "X"
    assert list(cells.EVAL_SEEDS) == list(range(12))


def test_names_and_layout():
    """The checkpoint folder is EXP-075's layout under EXP-082's outputs, so EXP-075's
    load_judge reads an X16k judge unchanged; every record name carries the experiment."""
    assert cells.ckpt_dir(3) == c75.ckpt_dir("X", 3, cells.OUT)
    assert cells.ckpt_dir(3).parent == EXP / "outputs"
    assert cells.record_name(8000, 3) == "exp082_X16k_u8000_s3.json"
    assert cells.gatec_name(3) == "exp082_gatec_s3.json"
    assert cells.snapshot_name(8000) == "judge_u8000.pt"
    assert cells.search_name(11, 3) == "exp082_J4V-X16k_d11_s3.json"
    assert cells.E75_OUT != cells.OUT
