"""EXP-032 sweep driver: the entropy_beta x advantage-normalization grid.

The load-bearing test here is filename uniqueness. `run_cube_baseline` names records
`{tag}_{arm}_d{depth}_s{seed}_sig{sigma}.json`, which encodes NEITHER `entropy_beta` NOR
`normalize_advantages`. A sweep that varies only those two at fixed depth and seed
therefore collides silently: 192 runs would land in 24 files, each holding whichever cell
finished last, and every downstream number would be wrong without anything raising.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from neuromorphic.training.cube_baseline import CubeConfig, record_filename, run_cube_baseline

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "exp032_run", ROOT / "experiments" / "032_collapse_sweep" / "run.py"
)
run_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_mod)

BETAS = run_mod.BETAS
DEPTHS = run_mod.DEPTHS
NORMALIZE = run_mod.NORMALIZE
sweep_configs = run_mod.sweep_configs

SEEDS = list(range(12))


def test_sweep_grid_is_betas_times_normalize_times_depths_times_seeds(tmp_path):
    cfgs = sweep_configs(SEEDS, episodes=600, out_dir=tmp_path)
    assert len(cfgs) == len(BETAS) * len(NORMALIZE) * len(DEPTHS) * len(SEEDS) == 192


def test_every_sweep_cell_writes_a_distinct_file(tmp_path):
    """The collision guard. Fails against any tag omitting entropy_beta or normalization."""
    cfgs = sweep_configs(SEEDS, episodes=600, out_dir=tmp_path)
    names = [record_filename(c) for c in cfgs]
    collisions = len(cfgs) - len(set(names))
    assert collisions == 0, f"{collisions} records would silently overwrite each other"


def test_record_filename_matches_what_the_trainer_actually_writes(tmp_path):
    """Pins the helper against real output so the collision guard cannot drift from reality.

    Without this, the guard above could pass against a helper that no longer describes the
    file `run_cube_baseline` writes, which is the "test re-implements the fix" failure the
    repo's test-strength rule names.
    """
    cfg = CubeConfig(
        arm="regionalized", depth=1, seed=0, episodes=1, max_depth=1, out_dir=tmp_path,
        tag="exp032_probe", entropy_beta=0.05, normalize_advantages=True,
    )
    run_cube_baseline(cfg)
    assert [p.name for p in tmp_path.glob("*.json")] == [record_filename(cfg)]


def test_sweep_holds_everything_fixed_except_the_two_knobs_and_depth(tmp_path):
    cfgs = sweep_configs(SEEDS, episodes=600, out_dir=tmp_path)
    assert {c.readout for c in cfgs} == {"concept"}
    assert {c.arm for c in cfgs} == {"regionalized"}
    assert {c.sigma for c in cfgs} == {0.0}
    assert {c.depth for c in cfgs} == set(DEPTHS)
    assert sorted({c.entropy_beta for c in cfgs}) == sorted(BETAS)
    assert {c.normalize_advantages for c in cfgs} == set(NORMALIZE)


def test_normalization_is_crossed_not_pinned(tmp_path):
    """ADR 0001's claim is that beta ALONE fails and only beta+normalization works.

    Pinning normalize_advantages=True would assume that transfers from the grid world to
    the cube instead of testing it, so both levels must be present at every beta.
    """
    cfgs = sweep_configs(SEEDS, episodes=600, out_dir=tmp_path)
    for beta in BETAS:
        levels = {c.normalize_advantages for c in cfgs if c.entropy_beta == beta}
        assert levels == {False, True}, f"beta={beta} is not crossed with normalization"


def test_baseline_cell_reproduces_the_exp031_configuration(tmp_path):
    """(beta=0, normalize=False) is EXP-030/031's exact config, so those 24 runs are a
    free byte-identity check against records that already exist."""
    cfgs = sweep_configs(SEEDS, episodes=600, out_dir=tmp_path)
    baseline = [c for c in cfgs if c.entropy_beta == 0.0 and not c.normalize_advantages]
    assert len(baseline) == len(DEPTHS) * len(SEEDS) == 24
    assert {c.depth for c in baseline} == set(DEPTHS)


def test_depth_2_is_included_because_the_exp030_headline_lives_there(tmp_path):
    cfgs = sweep_configs(SEEDS, episodes=600, out_dir=tmp_path)
    assert 2 in {c.depth for c in cfgs}
    assert 3 in {c.depth for c in cfgs}


@pytest.mark.parametrize("beta", BETAS)
def test_each_beta_is_a_nonnegative_float(beta):
    assert isinstance(beta, float) and beta >= 0.0
