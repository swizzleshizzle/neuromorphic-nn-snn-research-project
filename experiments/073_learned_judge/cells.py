"""EXP-073 cells: per-seed exclusion and probe sets, and judge construction from the seed's E1
encoder. The BFS provider is used HERE, at setup time, to build the probe and exclusion lists;
`value_iteration.py` itself never imports or reads it (see its module docstring and the
module-level test in tests/training/test_value_iteration.py).

Spec: docs/superpowers/specs/2026-10-07-exp073-learned-judge-design.md, sections 2 and 3.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

_spec = importlib.util.spec_from_file_location(
    "exp070_cells", REPO / "experiments" / "070_lookahead_existing" / "cells.py")
c70 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c70)

from neuromorphic.training.cube_baseline import (  # noqa: E402
    make_agent, resolve_seed, shell_states, split_shell,
)
from neuromorphic.training import value_iteration as vi  # noqa: E402

EVAL_SEEDS = tuple(range(12))
PILOT_SEEDS = (12, 13)
EVAL_DEPTHS = (7, 8, 9, 11)
ARMS_TRAIN = ("A", "B")

HELDOUT_CAP = 200
HELDOUT_FRAC = 0.25


def heldout_states(depth: int, seed: int, provider) -> list:
    """The held-out side of `depth`'s shell, for the given seed.

    Depths 7 to 9 reuse EXP-070's exact split (its own `split_seed`, resolved from the published
    config EXP-070 trained under), so an excluded or evaluated state here is the identical set
    every paired P3V/R3V reference was measured on. Depth 11 has no published EXP-070 config (it
    is exploratory-only, never published at depth 11), so it uses the eval seed itself as the
    split seed.
    """
    if depth in (7, 8, 9):
        split_seed = resolve_seed(c70.published_config(depth, seed), "split")
    else:
        split_seed = seed
    return split_shell(shell_states(provider, depth), depth, seed=split_seed,
                       heldout_cap=HELDOUT_CAP, heldout_frac=HELDOUT_FRAC)[1]


def exclusion_set(seed: int, provider) -> set:
    """Union of every evaluation depth's held-out states, for one seed."""
    out: set = set()
    for d in EVAL_DEPTHS:
        out.update(heldout_states(d, seed, provider))
    return out


def probe_set(seed: int, provider, per_distance: int = 50) -> list:
    """`per_distance` states per true distance 1 to 11, drawn from the BFS shells, disjoint from
    every held-out set at this seed. Each shell is shuffled independently (keyed by distance
    through the 10_000 offset having no effect on which RNG stream is used, only its seed), then
    filtered against the exclusion set, THEN truncated: filtering before truncating is what keeps
    an excluded state from silently displacing a kept one further down the shuffle.
    """
    excl = exclusion_set(seed, provider)
    out = []
    for d in range(1, 12):
        shuffled = list(shell_states(provider, d))
        random.Random(10_000 + seed).shuffle(shuffled)
        kept = [s for s in shuffled if s not in excl][:per_distance]
        out.extend((s, d) for s in kept)
    return out


def make_judge(seed: int) -> vi.Judge:
    """A judge whose sensory region starts at this seed's E1 encoder, with the head freshly
    initialised under a seeded global RNG (so two calls with the same seed produce byte-identical
    heads). `torch.manual_seed` is deliberately called AFTER the agent is built, not before:
    `make_agent` builds a whole five-region `Brain` internally, and each region's `nn.Linear`
    construction (including `SensoryCortex.fc1`/`fc2`) draws its throwaway default init from
    the GLOBAL RNG before `_init_weights` overwrites it with a local, explicitly-seeded
    generator. So `make_agent` DOES advance the global RNG, by an amount that depends on
    region sizes elsewhere in the brain and is not part of this function's contract. Seeding
    right before `judge_from_brain` constructs the head overrides whatever state that left
    behind, so the head's init depends only on `seed`, never on what `make_agent` did first.
    """
    brain = make_agent(c70.published_config(7, seed))
    torch.manual_seed(seed)
    return vi.judge_from_brain(brain)


def ckpt_dir(arm: str, seed: int) -> Path:
    return HERE / "outputs" / f"judge_{arm}_s{seed}"


def record_name(arm: str, seed: int) -> str:
    return f"exp073_train_{arm}_s{seed}.json"
