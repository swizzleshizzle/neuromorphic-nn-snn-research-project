"""EXP-074 cells: arms W, A and B over EXP-073's per-seed held-out, exclusion and probe sets.

Arms A and B are built by EXP-073's own `make_judge`, so they ARE EXP-073's judges (Gate 0(c)).
Arm W is the same construction with the wide readout.

Spec: docs/superpowers/specs/2026-10-07-exp074-wide-judge-design.md, sections 3 and 6.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

_spec = importlib.util.spec_from_file_location(
    "exp073_cells", REPO / "experiments" / "073_learned_judge" / "cells.py")
c73 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c73)
c70 = c73.c70

from neuromorphic.training.cube_baseline import make_agent  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

EVAL_SEEDS = c73.EVAL_SEEDS
PILOT_SEEDS = c73.PILOT_SEEDS
EVAL_DEPTHS = c73.EVAL_DEPTHS
ARMS_TRAIN = ("W", "A", "B")
# Spec section 2: seeds whose depth-9 cells were looked at before this spec was written.
SENSITIVITY_DROP = (0, 3)

heldout_states = c73.heldout_states
exclusion_set = c73.exclusion_set
probe_set = c73.probe_set


def make_judge(seed: int, arm: str) -> vi.Judge:
    """A and B: EXP-073's judge exactly. W: the same construction (same agent, same seeding
    point before the head) with the wide readout."""
    if arm in ("A", "B"):
        return c73.make_judge(seed)
    if arm == "W":
        brain = make_agent(c70.published_config(7, seed))
        torch.manual_seed(seed)
        return vi.judge_from_brain(brain, readout="wide")
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS_TRAIN}")


def ckpt_dir(arm: str, seed: int, out_dir: Path | None = None) -> Path:
    return Path(out_dir or HERE / "outputs") / f"judge_{arm}_s{seed}"


def record_name(arm: str, seed: int) -> str:
    return f"exp074_train_{arm}_s{seed}.json"
