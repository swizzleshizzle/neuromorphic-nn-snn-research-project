import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "experiments" / "028_sensory_ablation"
spec = importlib.util.spec_from_file_location("exp028_agg", HERE / "aggregate.py")
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)


def test_aggregate_curve_means_across_seeds():
    cells = [
        {"operator": "gaussian", "dose": 0.0, "seed": 0, "heldout_success": 0.4},
        {"operator": "gaussian", "dose": 0.0, "seed": 1, "heldout_success": 0.6},
        {"operator": "gaussian", "dose": 0.2, "seed": 0, "heldout_success": 0.2},
    ]
    curve = agg.aggregate_curve(cells)
    assert curve["gaussian"][0.0] == 0.5
    assert curve["gaussian"][0.2] == 0.2


def test_format_curve_has_row_per_dose_and_operator_columns():
    curve = {"gaussian": {0.0: 0.5, 0.2: 0.2}, "unitdrop_top": {0.0: 0.5}}
    md = agg.format_curve(curve)
    assert "gaussian" in md and "unitdrop_top" in md
    assert "| 0.0 |" in md and "| 0.2 |" in md
