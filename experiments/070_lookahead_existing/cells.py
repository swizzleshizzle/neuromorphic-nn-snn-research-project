"""EXP-070 cells: the published EXP-053 arm B (depth 7) and EXP-062 (depths 8, 9) checkpoints.

Spec: docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md, section 2.2.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from neuromorphic.analysis.ablate import AblatedConcept
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.cube_baseline import (
    CubeConfig, feature_width, make_agent, record_filename, resolve_seed, shell_states,
    split_shell,
)

REPO = Path(__file__).resolve().parents[2]
E1_DIR = Path("experiments/047_encoder_finetuning/outputs")
PUBLISHED_DIR = {
    7: Path("experiments/053_neuromod_stage3/outputs"),
    8: Path("experiments/062_depth_frontier/outputs"),
    9: Path("experiments/062_depth_frontier/outputs"),
}
TAGS = {7: "exp053_critic_d7", 8: "exp062_frontier_d8", 9: "exp062_frontier_d9"}

DEPTHS = (7, 8, 9)
SEEDS = tuple(range(12))
# P1 is absent by design: at k=1 it is E1 by construction (spec amendment 2026-10-06).
ARMS = (("G", 0), ("E", 1), ("E", 2), ("E", 3), ("P", 2), ("P", 3),
        ("R", 1), ("R", 2), ("R", 3))
EPISODES = 10_000
CAP = ((1, 2),)
CRITIC_LR = 0.01


def published_config(depth: int, seed: int) -> CubeConfig:
    """EXP-053 arm B's config, which EXP-062 copied field for field, varying only depth."""
    return CubeConfig(
        arm="regionalized", readout="concept", tag=TAGS[depth],
        depth=depth, seed=seed, sigma=0.0, episodes=EPISODES,
        curriculum=tuple(range(1, depth + 1)), max_steps_by_depth=CAP,
        entropy_beta=0.0, normalize_advantages=False,
        encoder_state_path=str(E1_DIR / f"exp047_ft_d6_lr0.0001_regionalized_d6_s{seed}_sig0.0_encoder.pt"),
        critic_lr=CRITIC_LR, max_depth=depth, out_dir=REPO / PUBLISHED_DIR[depth],
    )


def published_record_path(depth: int, seed: int) -> Path:
    """The UNTRACKED published record. `NN_PUBLISHED_ROOT` points at a checkout that holds
    them (the main checkout, when working in a worktree); defaults to this repo."""
    root = Path(os.environ.get("NN_PUBLISHED_ROOT", REPO))
    return root / PUBLISHED_DIR[depth] / record_filename(published_config(depth, seed))


def head_path(depth: int, seed: int) -> Path:
    return REPO / PUBLISHED_DIR[depth] / record_filename(
        published_config(depth, seed)).replace(".json", "_head.pt")


def cell_record_name(mode: str, k: int, depth: int, seed: int) -> str:
    return f"exp070_{mode}{k}_d{depth}_s{seed}.json"


def load_cell(depth: int, seed: int):
    """(agent, head, held-out states, train seed) exactly as the published evaluation saw them.

    The BFS provider is used HERE, to rebuild the held-out shell, and nowhere in the procedure.
    """
    cfg = published_config(depth, seed)
    agent = make_agent(cfg)
    width = feature_width(cfg)
    head = AblatedConcept(nn.Linear(width, cfg.n_actions), None, width=width)
    head.load_state_dict(torch.load(head_path(depth, seed), map_location="cpu"))
    _train, eval_states, _ = split_shell(
        shell_states(ExactBFSDistance(max_depth=depth), depth), depth,
        seed=resolve_seed(cfg, "split"),
        heldout_cap=cfg.heldout_cap, heldout_frac=cfg.heldout_frac,
    )
    return agent, head, eval_states, resolve_seed(cfg, "train")
