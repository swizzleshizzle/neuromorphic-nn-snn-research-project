# experiments/033_concept_decodability/probe.py
"""EXP-033 Part 1: how much of the optimal move is linearly readable from the concept?

EXP-032 concluded that policy collapse is a symptom rather than the binding constraint:
de-collapsing the policy made it worse at solving cubes. The remaining suspect is the
representation. The cube brain is **frozen at random initialisation** and only the
`Linear(64 -> 6)` head trains, which is 390 parameters, so nothing in the system can learn
what a move does to a cube.

This measures the wall directly and cheaply, with no reinforcement learning involved: fit a
linear probe from features to "which moves reduce distance-to-solved" and read its held-out
top-1 accuracy. Three feature sets are compared at matched states and splits:

    facelets   raw 144-d one-hot observation      what a linear model could ever extract
    concept    frozen concept at width W          what the policy head actually sees
    chance     measured, not assumed              the floor

Distance-to-solved is used ONLY to build labels for this offline probe. It is never a model
input, and nothing here changes the policy path. That is the same instrument/input line
`CLAUDE.md` draws.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from neuromorphic.envs.cube import MOVES, apply_move

N_ACTIONS = len(MOVES)  # never a literal: a 3x3 cube is 12 or 18


def optimal_move_mask(provider, state) -> list[bool]:
    """Which moves strictly reduce distance-to-solved.

    Multi-label on purpose. Several moves can be optimal from the same state, so forcing a
    single canonical label would score a correct alternative as an error and understate
    every feature set equally but unpredictably.
    """
    d = provider.distance(state)
    if d is None:
        raise ValueError("state is outside the provider's table; widen max_depth")
    mask = []
    for a in range(N_ACTIONS):
        nd = provider.distance(apply_move(state, a))
        if nd is None:
            # A neighbour one step DEEPER than the table bound. Silently treating this as
            # "not distance-reducing" would usually be right and occasionally be a wrong
            # label, so refuse instead: the provider must be built one depth beyond the
            # deepest state being labelled.
            raise ValueError(
                f"neighbour of a distance-{d} state is outside the table; "
                f"build the provider with max_depth >= {d + 1}"
            )
        mask.append(nd < d)
    return mask


def chance_floor(masks) -> float:
    """Accuracy of a uniformly random guess, measured from the label set.

    Equal to the mean fraction of moves that are optimal. This is NOT 1/6: states differ in
    how many optimal moves they have, which is exactly why `CLAUDE.md` says to measure the
    floor rather than assume it.
    """
    if not masks:
        return 0.0
    return sum(sum(m) / len(m) for m in masks) / len(masks)


def top1_accuracy(logits: torch.Tensor, masks) -> float:
    """Fraction of states whose argmax move is one of the optimal moves."""
    if len(masks) == 0:
        return 0.0
    picks = logits.argmax(dim=1)
    return sum(1 for i, m in enumerate(masks) if m[int(picks[i])]) / len(masks)


def split_states(states, frac_heldout: float, seed: int):
    """Deterministic disjoint (train, heldout) split."""
    shuffled = list(states)
    random.Random(seed).shuffle(shuffled)
    n_held = int(round(frac_heldout * len(shuffled)))
    return shuffled[n_held:], shuffled[:n_held]


def fit_linear_probe(x: torch.Tensor, masks, *, epochs: int = 250, lr: float = 0.1,
                     seed: int = 0, weight_decay: float = 1e-4) -> nn.Linear:
    """Multi-label linear probe trained with BCE.

    BCE rather than cross-entropy because the labels are multi-label: with several optimal
    moves per state, cross-entropy would force the probe to pick a winner among equally
    correct answers and penalise it for the others.
    """
    torch.manual_seed(seed)
    y = torch.tensor([[1.0 if v else 0.0 for v in m] for m in masks])
    model = nn.Linear(x.shape[1], N_ACTIONS)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return model


def facelet_features(states) -> torch.Tensor:
    """Raw observation as the encoder receives it: 24 facelets one-hot over 6 colours."""
    n_facelets = len(states[0])
    n_colours = max(max(s) for s in states) + 1
    x = torch.zeros(len(states), n_facelets * n_colours)
    for i, s in enumerate(states):
        for j, c in enumerate(s):
            x[i, j * n_colours + int(c)] = 1.0
    return x
