"""Approximate value iteration for a "moves to solved" judge (EXP-073).

The judge learns from its OWN look-ahead through the simulator: a state's target is one move plus
the best child's value under a frozen copy of the judge. No distance table is read here; probe and
exclusion lists arrive as plain arguments. Training builds a graph through the spiking encoder in
arm A, so nothing in this module may route through Brain.step or the look-ahead helpers, which run
under no_grad.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromorphic.envs.cube import SOLVED, apply_move, is_solved


class Judge(nn.Module):
    def __init__(self, encoder_fn, sensory: nn.Module, T: int, content: int, hidden: int = 128):
        super().__init__()
        self.encoder_fn = encoder_fn
        self.sensory = sensory
        self.T = T
        self.head = nn.Sequential(nn.Linear(content, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def concept(self, states, generator):
        obs = torch.as_tensor(np.array(states), dtype=torch.long)
        spikes = self.encoder_fn(obs, T=self.T, generator=generator)
        return self.sensory(spikes).mean(dim=0)

    def forward(self, states, generator):
        return F.softplus(self.head(self.concept(states, generator))).squeeze(-1)


def judge_from_brain(brain, hidden: int = 128) -> Judge:
    return Judge(brain._encoder, copy.deepcopy(brain.sensory), brain.T, brain.content, hidden)


def random_walk_states(n, max_len, rng, n_actions, exclude, *, start=SOLVED,
                       apply_fn=apply_move):
    out = []
    while len(out) < n:
        s = start
        for _ in range(rng.randint(1, max_len)):
            s = apply_fn(s, rng.randrange(n_actions))
        if s not in exclude:
            out.append(s)
    return out


def bellman_targets(states, jt, n_actions, generator, draws=1, *, apply_fn=apply_move,
                    goal_fn=is_solved):
    with torch.no_grad():
        children = [apply_fn(s, a) for s in states for a in range(n_actions)]
        vals = torch.stack([jt(children, generator) for _ in range(draws)]).mean(dim=0)
        solved_child = torch.tensor([goal_fn(c) for c in children])
        vals = torch.where(solved_child, torch.zeros_like(vals), vals)
        best = vals.view(len(states), n_actions).min(dim=1).values
        y = 1.0 + best
        solved = torch.tensor([goal_fn(s) for s in states])
        return torch.where(solved, torch.zeros_like(y), y)


def sync_target(judge):
    # `judge.sensory` may already have run a live-graph forward (train_step in arm A), which
    # leaves `mem1`/`mem2` as non-leaf tensors attached to that graph. torch's deepcopy protocol
    # raises on a non-leaf tensor, so reset the membrane state to fresh leaf zeros first. This is
    # a strict no-op on the next real forward, which calls `reset()` itself regardless.
    judge.sensory.reset()
    jt = copy.deepcopy(judge)
    for p in jt.parameters():
        p.requires_grad_(False)
    return jt


def make_optimizer(judge, arm):
    if arm == "A":
        return torch.optim.Adam([{"params": judge.head.parameters(), "lr": 1e-3},
                                 {"params": judge.sensory.parameters(), "lr": 1e-4}])
    if arm == "B":
        for p in judge.sensory.parameters():
            p.requires_grad_(False)
        return torch.optim.Adam(judge.head.parameters(), lr=1e-3)
    raise ValueError(f"unknown arm {arm!r}")


def train_step(judge, jt, optimizer, states, n_actions, generator, draws=1):
    y = bellman_targets(states, jt, n_actions, generator, draws)
    loss = F.mse_loss(judge(states, generator), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss)
