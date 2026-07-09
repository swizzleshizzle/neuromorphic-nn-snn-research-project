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
