import importlib.util as ilu
from pathlib import Path
import torch

from neuromorphic.brain import Brain
from neuromorphic.training.reinforce import make_policy_head, greedy_action

ROOT = Path(__file__).resolve().parents[2]
_spec = ilu.spec_from_file_location(
    "exp027_dropout", ROOT / "experiments" / "027_encoder_characterization" / "dropout_eval.py")
de = ilu.module_from_spec(_spec)
_spec.loader.exec_module(de)


def test_masked_head_k0_matches_unmasked():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    mh = de.MaskedHead(head)
    mh.set_mask(torch.ones(64))
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(1)
    assert greedy_action(brain, head, [0, 0, 4, 4], generator=g1) == \
           greedy_action(brain, mh, [0, 0, 4, 4], generator=g2)


def test_masked_head_masking_zeros_units_and_changes_output():
    torch.manual_seed(0)
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    mh = de.MaskedHead(head)
    x = torch.ones(head.in_features)
    # a fully-zero mask makes the head read all zeros -> the output is exactly its bias
    mh.set_mask(torch.zeros(head.in_features))
    assert torch.allclose(mh(x), head.bias)
    # masking a single unit changes the output vs the unmasked head (mask is load-bearing for k>0)
    mask = torch.ones(head.in_features)
    mask[0] = 0.0
    mh.set_mask(mask)
    assert not torch.allclose(mh(x), head(x))
    # the change is exactly the dropped unit's contribution
    assert torch.allclose(head(x) - mh(x), head.weight[:, 0] * x[0])


def test_random_mask_zeros_k_units():
    m = de.random_mask(64, 10, seed=0)
    assert m.shape == (64,)
    assert int((m == 0).sum()) == 10


def test_importance_mask_top_vs_bottom_disjoint():
    order = torch.arange(64)   # 0 most important
    top = de.importance_mask(order, 8, mode="top")
    bot = de.importance_mask(order, 8, mode="bottom")
    assert int((top == 0).sum()) == 8 and int((bot == 0).sum()) == 8
    assert int(((top == 0) & (bot == 0)).sum()) == 0   # disjoint masked sets
