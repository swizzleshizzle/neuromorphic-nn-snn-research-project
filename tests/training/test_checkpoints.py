import torch

from neuromorphic.brain import Brain
from neuromorphic.training.reinforce import make_policy_head, greedy_action
from neuromorphic.training.checkpoints import save_trained, load_trained
from neuromorphic.training.generalization import GenConfig, run_generalization


def test_checkpoint_roundtrip_reproduces_eval(tmp_path):
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    path = tmp_path / "ckpt.pt"
    save_trained(path, brain, head, {"grid_n": 5, "seed": 0})
    b2, h2 = load_trained(path, grid_n=5, seed=0)
    gen1 = torch.Generator().manual_seed(3)
    gen2 = torch.Generator().manual_seed(3)
    a1 = greedy_action(brain, head, [0, 0, 4, 4], generator=gen1)
    a2 = greedy_action(b2, h2, [0, 0, 4, 4], generator=gen2)
    assert a1 == a2
    # sensory + head weights round-trip exactly
    assert torch.equal(brain.sensory.fc1.weight, b2.sensory.fc1.weight)
    assert torch.equal(head.weight, h2.weight)


def test_genconfig_default_no_checkpoint(tmp_path):
    cfg = GenConfig()
    assert cfg.checkpoint_path is None


def test_run_generalization_writes_checkpoint_when_set(tmp_path):
    ckpt = tmp_path / "run.pt"
    cfg = GenConfig(seed=0, episodes=2, n_heldout=2, max_steps=8,
                    pretrain_sensory=True, pretrain_epochs=5,
                    checkpoint_path=str(ckpt), tag="ck", out_dir=tmp_path)
    run_generalization(cfg)
    assert ckpt.exists()
    b2, h2 = load_trained(ckpt, grid_n=5, seed=0)
    # the saved encoder was pretrained, so it must differ from a fresh random Brain(seed=0)
    fresh = Brain(grid_n=5, seed=0)
    assert not torch.equal(b2.sensory.fc1.weight, fresh.sensory.fc1.weight)
    assert h2.in_features == b2.content   # head reads the 64-d concept
