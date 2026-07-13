import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "experiments" / "028_sensory_ablation"
spec = importlib.util.spec_from_file_location("exp028_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_mint_and_cell_smoke(tmp_path):
    # tiny config: mint one encoder, take its importance order, run one gaussian cell
    ck = run.mint_encoder(0, grid_n=5, out_dir=tmp_path, episodes=3, pretrain_epochs=2)
    assert Path(ck).exists()
    order = run.importance_order(ck, grid_n=5)
    assert len(order) == 64 and sorted(order) == list(range(64))
    cell = run.run_cell(0, "gaussian", 0.0, ckpt_path=ck, order=order, grid_n=5, episodes=3)
    assert cell["operator"] == "gaussian" and cell["dose"] == 0.0
    assert 0.0 <= cell["heldout_success"] <= 1.0
