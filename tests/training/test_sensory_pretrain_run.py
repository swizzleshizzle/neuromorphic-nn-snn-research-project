import importlib.util as ilu
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = ilu.spec_from_file_location(
    "exp026_run", ROOT / "experiments" / "026_sensory_pretrain" / "run.py"
)
run_mod = ilu.module_from_spec(_spec)
_spec.loader.exec_module(run_mod)


def test_build_configs_pretrain_linear_both_regimes(tmp_path):
    cfgs = run_mod.build_configs([0, 1], episodes=5, out_dir=tmp_path)
    assert len(cfgs) == 4   # 2 regimes x 2 seeds, linear only
    assert all(c.pretrain_sensory is True for c in cfgs)
    assert all(c.head_type == "linear" for c in cfgs)
    assert all(c.tag.endswith("_pt") for c in cfgs)
    assert {c.shaping for c in cfgs} == {True, False}
