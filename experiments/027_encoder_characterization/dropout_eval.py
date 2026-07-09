"""EXP-027 Component B: causal dropout-on-navigation via a MaskedHead on the concept."""

from __future__ import annotations

import torch
import torch.nn as nn

from neuromorphic.training.generalization import evaluate
from neuromorphic.training.pretrain import enumerate_states, split_states
from neuromorphic.analysis.probes import region_rate_matrix, task_targets, unit_importance


class MaskedHead(nn.Module):
    """Wrap a trained head; zero a subset of concept units before it reads them."""

    def __init__(self, head, mask=None):
        super().__init__()
        self.head = head
        self.mask = torch.ones(head.in_features) if mask is None else mask

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return self.head(x * self.mask.to(x.dtype))


def random_mask(width, k, seed) -> torch.Tensor:
    m = torch.ones(width)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(width, generator=g)[:k]
    m[idx] = 0.0
    return m


def importance_mask(order, k, mode) -> torch.Tensor:
    m = torch.ones(len(order))
    idx = order[:k] if mode == "top" else order[len(order) - k:]
    m[idx] = 0.0
    return m


def dropout_curve(brain, head, goals, *, grid_n, ks, n_random=5, size=5, max_steps=100) -> dict:
    """Held-out nav success under random-k / top-k / bottom-k concept masking (importance from THIS brain)."""
    tr, _ = split_states(enumerate_states(grid_n), frac_heldout=0.2, seed=0)
    Xtr = region_rate_matrix(brain, tr, region_key="sensory", signal_key="concept",
                             width=head.in_features, recall=False, T=brain.T,
                             generator=torch.Generator().manual_seed(0))
    order = unit_importance(Xtr, task_targets(tr, grid_n)["displacement"])
    mh = MaskedHead(head)

    def succ():
        return evaluate(brain, mh, goals, size=size, start=(0, 0), max_steps=max_steps,
                        generator=torch.Generator().manual_seed(0)).success_rate

    out = {"random": {}, "top": {}, "bottom": {}}
    for k in ks:
        rs = []
        for j in range(n_random):
            mh.set_mask(random_mask(head.in_features, k, seed=j))
            rs.append(succ())
        out["random"][k] = sum(rs) / len(rs)
        mh.set_mask(importance_mask(order, k, "top")); out["top"][k] = succ()
        mh.set_mask(importance_mask(order, k, "bottom")); out["bottom"][k] = succ()
    return out
