"""EXP-075 cells: arms X (hidden 512) and Y (hidden 128). Each starts from its own region,
pretrained by EXP-039's inverse-model recipe at that width (pretrain.py), and becomes a judge with
EXP-074's W readout (concept plus hidden mean rates). EXP-074's held-out, exclusion and probe sets
are reused unchanged, and W is EXP-074's committed judge.

Spec: docs/superpowers/specs/2026-10-09-exp075-wide-sensory-region-design.md, sections 3 and 4.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

_spec = importlib.util.spec_from_file_location(
    "exp074_cells", REPO / "experiments" / "074_wide_judge" / "cells.py")
c74 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c74)
c70 = c74.c70

from neuromorphic.encoders import cube_encoder  # noqa: E402
from neuromorphic.regions.sensory_cortex import SensoryCortex  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402
from neuromorphic.training.encoder_pretrain import load_encoder, make_sensory  # noqa: E402

EVAL_SEEDS = c74.EVAL_SEEDS
PILOT_SEEDS = c74.PILOT_SEEDS
EVAL_DEPTHS = c74.EVAL_DEPTHS
SENSITIVITY_DROP = c74.SENSITIVITY_DROP
ARMS_TRAIN = ("X", "Y")
HIDDEN = {"X": 512, "Y": 128}
CONTENT = 64
T = 32
E74_OUT = REPO / "experiments" / "074_wide_judge" / "outputs"

heldout_states = c74.heldout_states
exclusion_set = c74.exclusion_set
probe_set = c74.probe_set


def _check_arm(arm: str) -> None:
    if arm not in ARMS_TRAIN:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS_TRAIN}")


def check_seeds(seeds, pilot: bool) -> None:
    allowed = PILOT_SEEDS if pilot else EVAL_SEEDS
    bad = [s for s in seeds if s not in allowed]
    if bad:
        raise SystemExit(f"seeds {bad} not allowed with pilot={pilot}; allowed seeds are {allowed}")


def _out(out_dir) -> Path:
    return Path(out_dir) if out_dir is not None else HERE / "outputs"


def encoder_path(arm: str, seed: int, out_dir=None) -> Path:
    return _out(out_dir) / "encoders" / f"enc_{arm}_s{seed}.pt"


def pretrain_record_name(arm: str, seed: int) -> str:
    return f"exp075_pretrain_{arm}_s{seed}.json"


def ckpt_dir(arm: str, seed: int, out_dir=None) -> Path:
    return _out(out_dir) / f"judge_{arm}_s{seed}"


def record_name(arm: str, seed: int) -> str:
    return f"exp075_train_{arm}_s{seed}.json"


def fresh_sensory(arm: str, seed: int) -> SensoryCortex:
    """An untrained region at the arm's width: the shape a saved judge or encoder loads into."""
    _check_arm(arm)
    return make_sensory(seed, content=CONTENT, num_steps=T, hidden=HIDDEN[arm])


def load_pretrained(arm: str, seed: int, out_dir=None) -> SensoryCortex:
    """The arm's pretrained region. Strict load, so a file of another width raises."""
    _check_arm(arm)
    path = encoder_path(arm, seed, out_dir)
    if not path.exists():
        raise SystemExit(f"missing pretrained encoder {path} for arm {arm} seed {seed}; "
                         f"run pretrain.py first")
    return load_encoder(path, seed=seed, content=CONTENT, num_steps=T, hidden=HIDDEN[arm])


def build_judge(seed: int, arm: str, sensory: SensoryCortex) -> vi.Judge:
    """A W-readout judge over `sensory`. The head is seeded right before it is built (as in
    EXP-073 and EXP-074), so its init depends only on `seed`."""
    _check_arm(arm)
    torch.manual_seed(seed)
    return vi.Judge(cube_encoder(), sensory, T, CONTENT, readout="wide")


def make_judge(seed: int, arm: str, out_dir=None) -> vi.Judge:
    return build_judge(seed, arm, load_pretrained(arm, seed, out_dir))
