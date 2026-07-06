import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "exp025_aggregate", ROOT / "experiments" / "025_head_capacity" / "aggregate.py"
)
agg_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg_mod)

import importlib.util as _ilu
from pathlib import Path as _P
_run_spec = _ilu.spec_from_file_location(
    "exp025_run", _P(__file__).resolve().parents[2] / "experiments" / "025_head_capacity" / "run.py"
)
run_mod = _ilu.module_from_spec(_run_spec)
_run_spec.loader.exec_module(run_mod)


def _summary(head_type, shaping, heldout_success, train_success):
    return {
        "config": {"head_type": head_type, "shaping": shaping},
        "eval": {
            "train": {"success_rate": train_success},
            "heldout": {"success_rate": heldout_success},
        },
    }


def test_aggregate_groups_by_head_and_regime():
    summaries = [
        _summary("linear", True, 0.4, 0.5),
        _summary("linear", True, 0.6, 0.7),
        _summary("mlp", True, 0.8, 0.9),
        _summary("mlp", True, 0.9, 1.0),
    ]
    agg = agg_mod.aggregate(summaries)
    assert agg[("linear", "shaped")]["heldout_mean"] == 0.5
    assert agg[("linear", "shaped")]["heldout_spread"] == 0.2
    assert agg[("mlp", "shaped")]["heldout_mean"] == 0.85


def test_format_table_is_markdown_with_both_heads():
    summaries = [
        _summary("linear", False, 0.3, 0.4),
        _summary("mlp", False, 0.5, 0.6),
    ]
    table = agg_mod.format_table(agg_mod.aggregate(summaries))
    assert "linear" in table and "mlp" in table
    assert "sparse" in table
    assert table.count("|") >= 6  # at least a header row of cells


def test_build_configs_sets_entropy_beta_and_tag_suffix(tmp_path):
    cfgs = run_mod.build_configs([0], episodes=5, out_dir=tmp_path, entropy_beta=0.01)
    assert len(cfgs) == 4  # 2 heads x 2 regimes x 1 seed
    assert all(c.entropy_beta == 0.01 for c in cfgs)
    assert all(c.tag.endswith("_b01") for c in cfgs)


def test_build_configs_zero_beta_has_no_suffix(tmp_path):
    cfgs = run_mod.build_configs([0], episodes=5, out_dir=tmp_path, entropy_beta=0.0)
    assert all(not c.tag.endswith("_b01") for c in cfgs)
    assert all(c.entropy_beta == 0.0 for c in cfgs)


def test_build_configs_normalize_advantages_flag_and_suffix(tmp_path):
    cfgs = run_mod.build_configs(
        [0], episodes=5, out_dir=tmp_path, entropy_beta=0.05, normalize_advantages=True
    )
    assert len(cfgs) == 4
    assert all(c.normalize_advantages is True for c in cfgs)
    assert all(c.entropy_beta == 0.05 for c in cfgs)
    # suffix encodes both the beta value and the advantage-norm flag
    assert all(c.tag.endswith("_b05_an") for c in cfgs)


def test_build_configs_no_normalize_has_no_an_suffix(tmp_path):
    cfgs = run_mod.build_configs([0], episodes=5, out_dir=tmp_path, entropy_beta=0.01)
    assert all(c.normalize_advantages is False for c in cfgs)
    assert all(not c.tag.endswith("_an") for c in cfgs)
