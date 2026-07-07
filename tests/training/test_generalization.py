from neuromorphic.training.generalization import EvalResult, optimality, split_goals


def test_split_goals_disjoint_sized_and_excludes_start():
    train, held = split_goals(size=5, start=(0, 0), n_heldout=6, seed=0)
    assert len(train) == 18 and len(held) == 6
    s_train, s_held = set(train), set(held)
    assert s_train.isdisjoint(s_held)
    assert (0, 0) not in s_train and (0, 0) not in s_held
    all_candidates = {(x, y) for x in range(5) for y in range(5)} - {(0, 0)}
    assert s_train | s_held == all_candidates


def test_split_goals_deterministic():
    a = split_goals(5, (0, 0), 6, seed=3)
    b = split_goals(5, (0, 0), 6, seed=3)
    c = split_goals(5, (0, 0), 6, seed=4)
    assert a == b
    assert a != c


def test_optimality():
    assert optimality((0, 0), (2, 0), 2) == 1.0
    assert optimality((0, 0), (2, 0), 4) == 0.5
    assert optimality((0, 0), (2, 0), 0) == 0.0


def test_eval_result_fields():
    r = EvalResult(success_rate=0.5, mean_steps=8.0, optimality=0.9, n=4)
    assert r.success_rate == 0.5 and r.n == 4


import torch

from neuromorphic.brain import Brain
from neuromorphic.training.generalization import evaluate
from neuromorphic.training.reinforce import make_policy_head


def test_evaluate_smoke():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    gen = torch.Generator().manual_seed(0)
    res = evaluate(brain, head, [(1, 1), (2, 2)], size=5, start=(0, 0), max_steps=30, generator=gen)
    assert res.n == 2
    assert 0.0 <= res.success_rate <= 1.0
    assert res.mean_steps >= 0.0
    assert 0.0 <= res.optimality <= 1.0


def test_run_generalization_smoke(tmp_path):
    from neuromorphic.training.generalization import GenConfig, run_generalization

    cfg = GenConfig(seed=0, episodes=2, max_steps=20, n_heldout=6, tag="smoke", out_dir=tmp_path)
    summary = run_generalization(cfg)

    assert (tmp_path / "024_grid_generalization_smoke_metrics.csv").exists()
    assert (tmp_path / "024_grid_generalization_smoke_summary.json").exists()
    assert "generalization_gap" in summary
    assert "train" in summary["eval"] and "heldout" in summary["eval"]
    assert 0.0 <= summary["eval"]["train"]["success_rate"] <= 1.0

    # CSV has a header plus one row per episode
    lines = (tmp_path / "024_grid_generalization_smoke_metrics.csv").read_text().strip().splitlines()
    assert lines[0] == "episode,goal_x,goal_y,total_reward,steps,goal_reached,entropy"
    assert len(lines) == 1 + 2  # header + 2 episodes


from neuromorphic.training.generalization import GenConfig, run_generalization


def test_genconfig_defaults_to_linear_head():
    cfg = GenConfig()
    assert cfg.head_type == "linear"
    assert cfg.hidden == 128


def test_run_generalization_mlp_head_records_head_type(tmp_path):
    cfg = GenConfig(
        seed=0, episodes=3, n_heldout=2, max_steps=8,
        head_type="mlp", hidden=32, tag="smoke_mlp", out_dir=tmp_path,
    )
    summary = run_generalization(cfg)
    assert summary["config"]["head_type"] == "mlp"
    assert summary["config"]["hidden"] == 32
    assert "train" in summary["eval"] and "heldout" in summary["eval"]


def test_run_generalization_is_deterministic_for_fixed_seed_and_head(tmp_path):
    base = dict(seed=1, episodes=3, n_heldout=2, max_steps=8, head_type="linear")
    a = run_generalization(GenConfig(**base, tag="det_a", out_dir=tmp_path))
    b = run_generalization(GenConfig(**base, tag="det_b", out_dir=tmp_path))
    assert a["eval"] == b["eval"]
    assert a["train_goals"] == b["train_goals"]
    assert a["heldout_goals"] == b["heldout_goals"]


def test_genconfig_defaults_entropy_beta_zero():
    assert GenConfig().entropy_beta == 0.0


def test_run_generalization_records_entropy_beta(tmp_path):
    cfg = GenConfig(
        seed=0, episodes=3, n_heldout=2, max_steps=8,
        entropy_beta=0.01, tag="smoke_beta", out_dir=tmp_path,
    )
    summary = run_generalization(cfg)
    assert summary["config"]["entropy_beta"] == 0.01


def test_genconfig_defaults_normalize_advantages_false():
    assert GenConfig().normalize_advantages is False


def test_run_generalization_records_normalize_advantages(tmp_path):
    cfg = GenConfig(
        seed=0, episodes=3, n_heldout=2, max_steps=8,
        normalize_advantages=True, tag="smoke_an", out_dir=tmp_path,
    )
    summary = run_generalization(cfg)
    assert summary["config"]["normalize_advantages"] is True


def test_genconfig_defaults_no_pretrain():
    cfg = GenConfig()
    assert cfg.pretrain_sensory is False
    assert cfg.pretrain_epochs == 200
    assert cfg.pretrain_lr == 1e-3


def test_run_generalization_without_pretrain_has_null_pretrain(tmp_path):
    cfg = GenConfig(seed=0, episodes=2, n_heldout=2, max_steps=8, tag="no_pt", out_dir=tmp_path)
    summary = run_generalization(cfg)
    assert summary["pretrain"] is None


def test_run_generalization_with_pretrain_records_gate_and_changes_encoder(tmp_path):
    import torch
    from neuromorphic.brain import Brain

    random_w1 = Brain(grid_n=5, seed=0).sensory.fc1.weight.detach().clone()
    cfg = GenConfig(
        seed=0, episodes=2, n_heldout=2, max_steps=8,
        pretrain_sensory=True, pretrain_epochs=10, tag="pt", out_dir=tmp_path,
    )
    summary = run_generalization(cfg)
    assert summary["pretrain"] is not None
    assert "heldout_disp_error" in summary["pretrain"]
    assert summary["config"]["pretrain_sensory"] is True
