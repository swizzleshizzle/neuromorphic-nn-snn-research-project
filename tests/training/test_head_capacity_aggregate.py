import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "exp025_aggregate", ROOT / "experiments" / "025_head_capacity" / "aggregate.py"
)
agg_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg_mod)


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
