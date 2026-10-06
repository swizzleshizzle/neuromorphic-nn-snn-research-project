"""Continuity with EXP-070: the C and V changes must not move any existing arm by one bit.

The fixture was produced by the UNMODIFIED lookahead.py (EXP-071 plan, Task 1 Step 0). Compare,
never regenerate: regenerating from modified code would make this test unable to fail.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neuromorphic.training.lookahead import evaluate_lookahead

REPO = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((REPO / "tests" / "fixtures" / "exp070_d7_s0_golden.json").read_text())
_s = importlib.util.spec_from_file_location(
    "c70", REPO / "experiments" / "070_lookahead_existing" / "cells.py")
c70 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(c70)


@pytest.mark.parametrize("arm,mode,k", [("G0", "G", 0), ("E3", "E", 3), ("P3", "P", 3),
                                        ("R3", "R", 3)])
def test_existing_arms_match_the_pre_change_golden_fixture(arm, mode, k):
    """Catches any change to the shared code path (stream draws, scoring, budget, metrics)."""
    agent, head, states, ts = c70.load_cell(7, 0)
    r = evaluate_lookahead(agent, head, states[:3], depth=7, mode=mode, k=k,
                           generator=torch.Generator().manual_seed(ts), rng_seed=ts, imag_seed=ts)
    for field, value in GOLDEN[arm].items():
        assert r[field] == value, (arm, field)
