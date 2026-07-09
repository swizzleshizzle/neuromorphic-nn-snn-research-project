import importlib.util as ilu
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = ilu.spec_from_file_location(
    "exp027_probe", ROOT / "experiments" / "027_encoder_characterization" / "probe.py")
probe_mod = ilu.module_from_spec(_spec)
_spec.loader.exec_module(probe_mod)


def test_characterize_seed_smoke_and_specialization_direction():
    res = probe_mod.characterize_seed(0, grid_n=5, pretrain_epochs=30, T=8)
    # every region present, hippocampus zero-filled -> displacement R2 ~ 0
    assert "sensory" in res["regions"] and "hippocampus" in res["regions"]
    assert res["regions"]["hippocampus"]["displacement_r2"] < 0.1
    # trained sensory concept beats its own shuffle-null band on displacement
    s = res["regions"]["sensory"]
    assert s["displacement_r2"] > s["displacement_null_hi"]
    # geometry present
    assert 1.0 <= res["geometry"]["participation_ratio"] <= 64.0


import importlib.util as ilu2
_aspec = ilu2.spec_from_file_location(
    "exp027_agg", ROOT / "experiments" / "027_encoder_characterization" / "aggregate.py")
agg_mod = ilu2.module_from_spec(_aspec)
_aspec.loader.exec_module(agg_mod)


def test_aggregate_regions_paired_winfraction():
    per_seed = [
        {"regions": {"sensory": {"displacement_r2": 0.8, "optimal_action_acc": 0.9},
                     "motor": {"displacement_r2": 0.2, "optimal_action_acc": 0.5}}},
        {"regions": {"sensory": {"displacement_r2": 0.7, "optimal_action_acc": 0.85},
                     "motor": {"displacement_r2": 0.3, "optimal_action_acc": 0.55}}},
    ]
    agg = agg_mod.aggregate_regions(per_seed)
    assert agg["motor"]["displacement_win_fraction"] == 1.0   # sensory beats motor both seeds
    assert agg["motor"]["n"] == 2
