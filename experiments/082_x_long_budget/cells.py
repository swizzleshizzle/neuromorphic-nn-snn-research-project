"""EXP-082 cells: EXP-075's X judge continued by value iteration from update 4000 to 16,000, arm
X16k. Distance-to-solved is the probe's yardstick and the search's success test only.

Spec: docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md, sections 2 to 5.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E75 = REPO / "experiments" / "075_wide_region"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c75 = _load("exp075_cells", E75 / "cells.py")

ARM = "X16k"
# EXP-075's arm. Also the checkpoint folder's name, so EXP-075's load_judge reads ours.
SRC_ARM = "X"
START = c75.N_UPDATES
LEGS = (8000, 16000)
# EXP-075's X settings (its train.py defaults), unchanged.
BATCH = 1000
SYNC_EVERY = 100
PROBE_EVERY = 250
DRAWS = 1
DEPTHS = (9, 11)
EVAL_SEEDS = c75.EVAL_SEEDS
OUT = HERE / "outputs"
E75_OUT = E75 / "outputs"
E76_OUT = REPO / "experiments" / "076_deeper_tree" / "outputs"
E81_OUT = REPO / "experiments" / "081_oxt_long_budget" / "outputs"

check_seeds = c75.check_seeds


def ckpt_dir(seed: int, root=None) -> Path:
    return c75.ckpt_dir(SRC_ARM, seed, OUT if root is None else root)


def record_name(update: int, seed: int) -> str:
    return f"exp082_{ARM}_u{update}_s{seed}.json"


def gatec_name(seed: int) -> str:
    return f"exp082_gatec_s{seed}.json"


def snapshot_name(update: int) -> str:
    return f"judge_u{update}.pt"


def search_name(depth: int, seed: int) -> str:
    return f"exp082_J4V-{ARM}_d{depth}_s{seed}.json"
