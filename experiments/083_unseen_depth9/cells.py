"""EXP-083 cells: J4V-X16k against J4V-X at depths 9 and 10 on states neither judge was trained
on. Distance-to-solved builds the evaluation sets and is the search's success test only.

Spec: docs/superpowers/specs/2026-10-30-exp083-unseen-depth9-design.md, sections 2 to 4.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E75 = REPO / "experiments" / "075_wide_region"
E82 = REPO / "experiments" / "082_x_long_budget"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c75 = _load("exp075_cells", E75 / "cells.py")

DEPTHS = (9, 10)
CAP = 200
SHUFFLE_BASE = 83_000
X_END = 4000
X16K_END = 16000
# EXP-075's X walk settings (EXP-082 continued them unchanged).
BATCH = 1000
MAX_LEN = 14
ARMS = ("J4V-X16k", "J4V-X")
EVAL_SEEDS = c75.EVAL_SEEDS
OUT = HERE / "outputs"
E75_OUT = E75 / "outputs"
E82_OUT = E82 / "outputs"
E76_OUT = REPO / "experiments" / "076_deeper_tree" / "outputs"
GATE_R_NAME = "exp083_gateR_s0.json"

check_seeds = c75.check_seeds


def sets_name(seed: int) -> str:
    return f"exp083_sets_s{seed}.json"


def search_name(arm: str, depth: int, seed: int) -> str:
    return f"exp083_{arm}_d{depth}_s{seed}.json"


def floor_name(depth: int, seed: int) -> str:
    return f"exp083_floor_d{depth}_s{seed}.json"
