"""EXP-085 cells: X16k (EXP-082) continued from update 16,000 to 24,000 on the device in two arms,
C on X's walks (max length 14) and O on outward walks (max length 14 rising to 30 by update
20,000). Distance-to-solved is the probe's yardstick and the search's success test only; the
lever is walk length, never true distance.

Spec: docs/superpowers/specs/2026-11-03-exp085-outward-walks-design.md, sections 2 to 6.
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

ARMS = ("C", "O")
# EXP-075's arm and folder name, so EXP-075's load_judge reads our checkpoints unchanged.
SRC_ARM = "X"
START = 16000
RAMP_END = 20000
END = 24000
# Legs: records at 20,000 and 24,000; the 23,751 leg only snapshots the checkpoint for Gate H (b)
# (train_judge's save after update 23,750 records next_update 23,751).
LEGS = (20000, 23751, 24000)
RECORD_LEGS = (20000, 24000)
SNAP_UPDATE = 23751
MAX_LEN_C = 14
MAX_LEN_O = 30
# EXP-082's X settings, unchanged.
BATCH = 1000
SYNC_EVERY = 100
PROBE_EVERY = 250
DRAWS = 1
DEPTHS = (9, 11)
SEEDS = c75.EVAL_SEEDS
H_SEED = 85
OUT = HERE / "outputs"
E75_OUT = E75 / "outputs"
E82_OUT = REPO / "experiments" / "082_x_long_budget" / "outputs"
E84_OUT = REPO / "experiments" / "084_gpu_port" / "outputs"
PREFLIGHT_NAME = "exp085_preflight.json"

check_seeds = c75.check_seeds


def max_len_schedule(arm: str):
    """The arm's max walk length as a function of the absolute update (spec section 2)."""
    if arm == "C":
        return lambda u: MAX_LEN_C
    if arm == "O":
        span = MAX_LEN_O - MAX_LEN_C
        return lambda u: (MAX_LEN_C + (span * (u - START)) // (RAMP_END - START)
                          if u < RAMP_END else MAX_LEN_O)
    raise ValueError(f"unknown arm {arm!r}")


def schedule_label(arm: str) -> str:
    return {"C": "14", "O": "14->30 by 20000"}[arm]


def _out(out):
    return OUT if out is None else Path(out)


def start_dir(seed: int, out=None) -> Path:
    return _out(out) / "start" / f"judge_{SRC_ARM}_s{seed}"


def arm_root(arm: str, out=None) -> Path:
    return _out(out) / f"arm_{arm}"


def ckpt_dir(arm: str, seed: int, out=None) -> Path:
    return c75.ckpt_dir(SRC_ARM, seed, arm_root(arm, out))


def snap_dir(arm: str, seed: int, out=None) -> Path:
    return ckpt_dir(arm, seed, out) / f"snap_u{SNAP_UPDATE}"


def gatec_name(seed: int) -> str:
    return f"exp085_gatec_s{seed}.json"


def rebase_name(arm: str, seed: int) -> str:
    return f"exp085_rebase_{arm}_s{seed}.json"


def record_name(arm: str, update: int, seed: int) -> str:
    return f"exp085_{arm}_u{update}_s{seed}.json"


def search_name(arm: str, depth: int, seed: int) -> str:
    return f"exp085_J4V-{arm}_d{depth}_s{seed}.json"


def h_search_name(arm: str, depth: int, seed: int) -> str:
    return f"exp085_H_J4V-{arm}_d{depth}_s{seed}.json"


def h_train_name(arm: str, seed: int) -> str:
    return f"exp085_Hb_{arm}_s{seed}.json"
