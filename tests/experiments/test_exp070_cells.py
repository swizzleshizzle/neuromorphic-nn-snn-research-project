"""EXP-070 cells: each re-evaluated cell must be the published cell, not a lookalike."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "070_lookahead_existing"
_spec = importlib.util.spec_from_file_location("exp070_cells", EXP / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)

COMPARED = ("seed", "depth", "tag", "arm", "sigma", "content", "n_actions", "max_depth",
            "heldout_cap", "heldout_frac", "encoder_seed", "train_seed", "split_seed")


@pytest.mark.parametrize("depth", cells.DEPTHS)
@pytest.mark.parametrize("seed", cells.SEEDS)
def test_every_head_and_encoder_is_tracked_and_present(depth, seed):
    """Catches a config that names a checkpoint nobody committed."""
    cfg = cells.published_config(depth, seed)
    assert cells.head_path(depth, seed).exists()
    assert (REPO / cfg.encoder_state_path).exists()


@pytest.mark.parametrize("depth", cells.DEPTHS)
@pytest.mark.parametrize("seed", (0, 11))
def test_config_matches_the_published_record_field_for_field(depth, seed):
    """Catches a re-evaluation of a different split or encoder than the one published. Needs
    the local records; skips cleanly on a checkout that lacks them."""
    path = cells.published_record_path(depth, seed)
    if not path.exists():
        pytest.skip(f"untracked published record not present: {path.name}")
    published = json.loads(path.read_text())["config"]
    cfg = cells.published_config(depth, seed)
    for key in COMPARED:
        assert getattr(cfg, key) == published[key], key
    assert Path(cfg.encoder_state_path).name == Path(published["encoder_state_path"].replace("\\", "/")).name


def test_published_counts_table_is_complete_and_matches_results_md():
    """Catches a truncated or mis-keyed table, and pins depths 8 and 9 to the per-seed lines
    EXP-062's RESULTS.md published (lines 75 and 87)."""
    table = json.loads((EXP / "published_counts.json").read_text())
    assert sorted(table) == sorted(f"d{d}_s{s}" for d in cells.DEPTHS for s in cells.SEEDS)
    d8 = [0.115, 0.085, 0.015, 0.110, 0.110, 0.060, 0.035, 0.135, 0.130, 0.055, 0.055, 0.035]
    d9 = [0.000, 0.000, 0.000, 0.080, 0.015, 0.005, 0.020, 0.025, 0.045, 0.000, 0.000, 0.005]
    for s in cells.SEEDS:
        assert table[f"d8_s{s}"] == {"solved": round(d8[s] * 200), "n": 200}
        assert table[f"d9_s{s}"] == {"solved": round(d9[s] * 200), "n": 200}


def test_cell_record_names_are_unique():
    """Catches records silently overwriting each other."""
    names = [cells.cell_record_name(m, k, d, s)
             for (m, k) in cells.ARMS for d in cells.DEPTHS for s in cells.SEEDS]
    assert len(names) == len(set(names)) == 9 * 3 * 12


def test_there_is_no_p1_arm():
    """Spec amendment 2026-10-06: P1 is E1 by construction and must not run."""
    assert ("P", 1) not in cells.ARMS


_rspec = importlib.util.spec_from_file_location("exp070_run", EXP / "run.py")
run = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(run)


def test_run_cell_writes_a_complete_record(tmp_path):
    """Catches a record missing the fields the aggregator and Gate 0 read."""
    rec = run.run_cell("E", 1, 8, 0, tmp_path, limit_states=3)
    on_disk = json.loads((tmp_path / cells.cell_record_name("E", 1, 8, 0)).read_text())
    assert on_disk == rec
    for key in ("solved", "n", "success_rate", "goal_fired_frac", "eval_revisit_rate",
                "mode", "k", "depth", "seed", "wall_s", "git_commit", "limit_states"):
        assert key in rec, key
    assert rec["n"] == 3 and rec["limit_states"] == 3


def test_parse_arm_rejects_p1():
    """Catches P1 sneaking back in through the CLI."""
    with pytest.raises(SystemExit):
        run.parse_arm("P1")
    assert run.parse_arm("P3") == ("P", 3)
