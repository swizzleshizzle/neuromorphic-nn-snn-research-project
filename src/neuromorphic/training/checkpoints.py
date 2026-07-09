"""Save/load a trained (sensory encoder, policy head) so downstream analysis reloads them."""

from __future__ import annotations

from pathlib import Path

import torch

from neuromorphic.brain import Brain
from neuromorphic.training.reinforce import make_policy_head


def save_trained(path, brain, head, cfg_dict: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sensory_state": brain.sensory.state_dict(),
        "head_state": head.state_dict(),
        "config": cfg_dict,
    }, path)


def load_trained(path, *, grid_n, seed):
    ckpt = torch.load(path, weights_only=True)
    brain = Brain(grid_n=grid_n, seed=seed)
    brain.sensory.load_state_dict(ckpt["sensory_state"])
    head_type = ckpt["config"].get("head_type", "linear")
    hidden = ckpt["config"].get("hidden", 128)
    head = make_policy_head(brain, head_type=head_type, hidden=hidden)
    head.load_state_dict(ckpt["head_state"])
    return brain, head
