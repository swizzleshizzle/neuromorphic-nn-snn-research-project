import json
from dataclasses import asdict

import torch

from neuromorphic.analysis.ablate import AblationSpec
from neuromorphic.training.generalization import GenConfig, run_generalization


def _cfg(tmp_path, **kw):
    return GenConfig(seed=0, episodes=5, n_heldout=4, size=5,
                     out_dir=tmp_path, tag="t", **kw)


def test_default_off_is_json_serializable_and_unwrapped(tmp_path):
    summary = run_generalization(_cfg(tmp_path))
    # summary must round-trip through JSON (no tensors/callables leaked into config)
    json.loads(json.dumps(summary["config"]))
    assert summary["config"]["ablation"] is None


def test_ablation_config_serializes(tmp_path):
    spec = AblationSpec("gaussian", dose=0.2, seed=0)
    summary = run_generalization(_cfg(tmp_path, ablation=spec))
    dumped = json.loads(json.dumps(summary["config"]))
    assert dumped["ablation"]["kind"] == "gaussian"
    assert dumped["ablation"]["dose"] == 0.2


def test_load_encoder_path_skips_pretrain(tmp_path):
    # mint an encoder checkpoint via the existing pretrain+save path
    ck = str(tmp_path / "enc.pt")
    run_generalization(_cfg(tmp_path, pretrain_sensory=True, checkpoint_path=ck))
    # loading it must skip pretraining (pretrain info stays None) and still run
    summary = run_generalization(_cfg(tmp_path, load_encoder_path=ck))
    assert summary["pretrain"] is None
    assert "heldout" in summary["eval"]
