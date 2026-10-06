"""EXP-071 cells: EXP-070's depth-7 cells plus each seed's EXP-053 arm B critic.

Spec: docs/superpowers/specs/2026-10-06-exp071-critic-and-no-revisit-design.md
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location(
    "exp070_cells", REPO / "experiments" / "070_lookahead_existing" / "cells.py")
c70 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c70)

DEPTH = 7
SEEDS = tuple(range(12))
ARMS = [(m, k, v) for (m, k) in (("G", 0), ("E", 3), ("P", 3), ("C", 1), ("C", 3), ("R", 3))
        for v in (False, True)]
CRITIC_DIR = REPO / "experiments" / "053_neuromod_stage3" / "outputs"


def arm_name(mode: str, k: int, v: bool) -> str:
    return f"{mode}{k}{'V' if v else ''}"


def parse_arm(text: str):
    for arm in ARMS:
        if arm_name(*arm) == text:
            return arm
    raise SystemExit(f"unknown arm {text!r}; valid: {[arm_name(*a) for a in ARMS]}")


def critic_path(seed: int) -> Path:
    return CRITIC_DIR / f"exp053_critic_d7_regionalized_d7_s{seed}_sig0.0_critic.pt"


def load_critic(seed: int) -> nn.Linear:
    critic = nn.Linear(64, 1)
    critic.load_state_dict(torch.load(critic_path(seed), map_location="cpu"))
    critic.eval()
    return critic


def cell_record_name(mode: str, k: int, v: bool, seed: int) -> str:
    return f"exp071_{arm_name(mode, k, v)}_d{DEPTH}_s{seed}.json"


def load_cell(seed: int):
    """(agent, head, held-out states, train seed), exactly EXP-070's depth-7 cell."""
    return c70.load_cell(DEPTH, seed)
