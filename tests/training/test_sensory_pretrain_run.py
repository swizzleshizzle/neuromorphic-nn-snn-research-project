import importlib.util as ilu
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = ilu.spec_from_file_location(
    "exp026_run", ROOT / "experiments" / "026_sensory_pretrain" / "run.py"
)
run_mod = ilu.module_from_spec(_spec)
_spec.loader.exec_module(run_mod)


def test_build_configs_both_arms_linear(tmp_path):
    cfgs = run_mod.build_configs([0, 1], episodes=5, out_dir=tmp_path, n_heldout=10)
    assert len(cfgs) == 8   # 2 arms (pretrain on/off) x 2 regimes x 2 seeds
    assert all(c.head_type == "linear" for c in cfgs)
    assert all(c.n_heldout == 10 for c in cfgs)
    pt = [c for c in cfgs if c.pretrain_sensory]
    rand = [c for c in cfgs if not c.pretrain_sensory]
    assert len(pt) == 4 and len(rand) == 4
    assert all(c.tag.endswith("_pt") for c in pt)
    assert all(c.tag.endswith("_rand") for c in rand)
    assert {c.shaping for c in cfgs} == {True, False}
