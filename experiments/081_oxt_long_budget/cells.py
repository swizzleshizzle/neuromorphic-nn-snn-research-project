"""EXP-081 cells: EXP-080's O-Xt continued from update 4000 to 16,000 on a mint four times as
long, whose first 4000 blocks are EXP-080's. Distance-to-solved is a diagnostic target only.

Spec: docs/superpowers/specs/2026-10-23-exp081-oxt-long-budget-design.md, sections 2 to 5.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E80 = REPO / "experiments" / "080_oracle_ordering"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c80 = _load("exp080_cells", E80 / "cells.py")

ARM = "O-Xt"
N_BLOCKS = 16000
BATCH = c80.BATCH
MAX_LEN = c80.MAX_LEN
PROBE_EVERY = c80.PROBE_EVERY
START = c80.N_UPDATES
LEGS = (8000, 16000)
EVAL_SEEDS = c80.EVAL_SEEDS
OUT = HERE / "outputs"
E80_OUT = c80.OUT

check_seeds = c80.check_seeds


def ckpt_dir(seed: int, root=None) -> Path:
    """EXP-080's layout, so EXP-080's load_trained reads an EXP-081 judge unchanged."""
    return c80.ckpt_dir(ARM, seed, OUT if root is None else root)


def record_name(update: int, seed: int) -> str:
    return f"exp081_{ARM}_u{update}_s{seed}.json"


def gatec_name(seed: int) -> str:
    return f"exp081_gatec_s{seed}.json"


def snapshot_name(update: int) -> str:
    return f"judge_u{update}.pt"
