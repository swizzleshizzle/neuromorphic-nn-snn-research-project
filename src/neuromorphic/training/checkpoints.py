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


def load_trained(path, *, grid_n=None, seed=None):
    ckpt = torch.load(path, weights_only=True)
    cfg = ckpt.get("config", {})
    # Prefer the values the checkpoint was trained with; caller args override but must
    # agree (a mismatched grid_n fails loudly on load_state_dict, but a mismatched seed
    # would silently rebuild a differently-initialized brain).
    if grid_n is None:
        grid_n = cfg.get("grid_n")
    elif "grid_n" in cfg and cfg["grid_n"] != grid_n:
        raise ValueError(f"grid_n {grid_n} != checkpoint grid_n {cfg['grid_n']}")
    if seed is None:
        seed = cfg.get("seed")
    elif "seed" in cfg and cfg["seed"] != seed:
        raise ValueError(f"seed {seed} != checkpoint seed {cfg['seed']}")
    if grid_n is None or seed is None:
        raise ValueError("grid_n/seed not supplied and not present in checkpoint config")
    brain = Brain(grid_n=grid_n, seed=seed)
    brain.sensory.load_state_dict(ckpt["sensory_state"])
    head_type = ckpt["config"].get("head_type", "linear")
    hidden = ckpt["config"].get("hidden", 128)
    head = make_policy_head(brain, head_type=head_type, hidden=hidden)
    head.load_state_dict(ckpt["head_state"])
    return brain, head
