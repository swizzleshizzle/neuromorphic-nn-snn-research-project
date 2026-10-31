"""EXP-084 cells: the GPU port's record names and pre-registered constants.

Spec: docs/superpowers/specs/2026-10-31-exp084-gpu-port-design.md, section 5.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E75 = REPO / "experiments" / "075_wide_region"
E82 = REPO / "experiments" / "082_x_long_budget"
E83 = REPO / "experiments" / "083_unseen_depth9"
OUT = HERE / "outputs"
E75_OUT = E75 / "outputs"
E82_OUT = E82 / "outputs"
E83_OUT = E83 / "outputs"

SEEDS = range(12)
ARMS = ("J4V-X16k", "J4V-X")
REQUIRED_UPDATES = {"J4V-X16k": 16000, "J4V-X": 4000}
TOL = 2  # Gate E and Gate M: within 2 solves of the published count (Michael, 2026-10-31)
P_LIMIT = 9  # Gate P stop rule: more than 9 of 4,800 first moves disagree
N_UPDATES = 4000  # Gate T: X retrained from scratch to EXP-075's budget
BASE_COMMIT = "9ffe730"  # EXP-083's commit, the pre-port code Gate C compares against
DEPTH = 9

GATEC_NAME = "exp084_gatec.json"
E_GATER_NAME = "exp084_E_gateR_s0.json"
M_NAME = "exp084_M_s0.json"


def p_name(arm: str, seed: int) -> str:
    return f"exp084_P_{arm}_s{seed}.json"


def e_name(arm: str, seed: int) -> str:
    return f"exp084_E_{arm}_d{DEPTH}_s{seed}.json"


def train_name(seed: int) -> str:
    return f"exp084_train_X_s{seed}.json"


def ckpt_dir(seed: int, out=None) -> Path:
    return Path(out or OUT) / f"judge_X_s{seed}"
