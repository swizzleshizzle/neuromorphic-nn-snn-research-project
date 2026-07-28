import torch

from neuromorphic.brain import Brain
from neuromorphic.training.reinforce import (
    action_distribution,
    concept_rate,
    make_policy_head,
)


def test_default_readout_is_bit_identical_to_concept_rate():
    """The compatibility gate: every 024-029 driver must be unaffected."""
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain, "linear")
    obs = [0, 0, 4, 4]

    _, logits_default = action_distribution(
        brain, head, obs, generator=torch.Generator().manual_seed(3)
    )
    _, logits_explicit = action_distribution(
        brain, head, obs, generator=torch.Generator().manual_seed(3), feature_fn=concept_rate
    )
    assert torch.equal(logits_default, logits_explicit)


def test_feature_fn_receives_the_step_output_and_its_result_drives_the_head():
    brain = Brain(grid_n=5, seed=0)
    seen = {}

    def fake_features(out):
        seen["keys"] = set(out.keys())
        return torch.zeros(brain.content)

    head = make_policy_head(brain, "linear")
    _, logits = action_distribution(
        brain, head, [0, 0, 4, 4],
        generator=torch.Generator().manual_seed(0), feature_fn=fake_features,
    )
    assert "concept" in seen["keys"]
    # zero features -> logits are exactly the head bias
    assert torch.allclose(logits, head.bias)


def test_recall_flag_reaches_the_brain():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain, "linear")
    captured = {}
    original = brain.step

    def spy(obs, **kwargs):
        captured.update(kwargs)
        return original(obs, **kwargs)

    brain.step = spy
    action_distribution(
        brain, head, [0, 0, 4, 4],
        generator=torch.Generator().manual_seed(0), store=True, recall=True,
    )
    assert captured["store"] is True
    assert captured["recall"] is True
