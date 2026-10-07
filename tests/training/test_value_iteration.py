"""Tests for value iteration (EXP-073). Each docstring names the bug it catches."""

from __future__ import annotations

import inspect
import random

import pytest
import torch
import torch.nn as nn

from neuromorphic.envs.cube import SOLVED, N_ACTIONS, apply_move, inverse_action, is_solved
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.cube_baseline import CubeConfig, make_agent


def _judge(seed=0):
    torch.manual_seed(seed)
    return vi.judge_from_brain(make_agent(CubeConfig(arm="regionalized", seed=seed)))


def test_module_never_imports_the_bfs_provider():
    """Catches an answer leak at the source level."""
    src = inspect.getsource(vi)
    assert "cube_distance" not in src and "ExactBFSDistance" not in src


def test_random_walks_respect_length_and_exclusions():
    """Catches walks longer than max_len, length-0 walks, or an ignored exclusion list."""
    rng = random.Random(0)
    one_move = {apply_move(SOLVED, a) for a in range(N_ACTIONS)}
    states = vi.random_walk_states(200, 1, rng, N_ACTIONS, exclude=set())
    assert set(states) <= one_move
    banned = {apply_move(SOLVED, 0)}
    states = vi.random_walk_states(200, 1, random.Random(1), N_ACTIONS, exclude=banned)
    assert not (set(states) & banned) and len(states) == 200


class _ConstJudge(nn.Module):
    """Jt stub: value looked up per state (default 5.0)."""

    def __init__(self, values):
        super().__init__()
        self.values = {tuple(k): v for k, v in values.items()}

    def forward(self, states, generator):
        return torch.tensor([float(self.values.get(tuple(s), 5.0)) for s in states])


def test_bellman_target_is_zero_at_solved_and_one_next_to_it():
    """Catches a solved state getting 1 + min, or a solved child not counted as 0."""
    s1 = apply_move(SOLVED, 2)
    y = vi.bellman_targets([SOLVED, s1], _ConstJudge({}), N_ACTIONS, torch.Generator())
    assert y.tolist() == [0.0, 1.0]


def test_bellman_target_takes_one_plus_the_minimum_child():
    """Catches max instead of min, or a missing +1."""
    s = apply_move(apply_move(SOLVED, 0), 2)
    child = apply_move(s, 4)
    y = vi.bellman_targets([s], _ConstJudge({child: 2.0}), N_ACTIONS, torch.Generator())
    assert y.item() == pytest.approx(3.0)


def test_jt_is_a_frozen_copy_of_encoder_and_head():
    """Catches Jt sharing the live encoder (targets would move every update) or keeping grads."""
    judge = _judge()
    jt = vi.sync_target(judge)
    assert all(not p.requires_grad for p in jt.parameters())
    before = [p.detach().clone() for p in jt.parameters()]
    opt = vi.make_optimizer(judge, "A")
    states = vi.random_walk_states(16, 4, random.Random(0), N_ACTIONS, exclude=set())
    vi.train_step(judge, jt, opt, states, N_ACTIONS, torch.Generator().manual_seed(0))
    after = list(jt.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    moved = sum(float((p - q).abs().sum()) for p, q in
                zip(judge.sensory.parameters(), jt.sensory.parameters()))
    assert moved > 0.0


def test_arm_a_gradient_reaches_the_encoder_and_arm_b_leaves_it_untouched():
    """THE GATE E MECHANISM. Catches the EXP-047 failure (encoder silently frozen in arm A) and a
    leaky control (encoder moving in arm B). Inputs carry a live graph: no no_grad fixture."""
    states = vi.random_walk_states(16, 4, random.Random(0), N_ACTIONS, exclude=set())
    a = _judge()
    jt_a = vi.sync_target(a)
    w0 = a.sensory.fc1.weight.detach().clone()
    vi.train_step(a, jt_a, vi.make_optimizer(a, "A"), states, N_ACTIONS,
                  torch.Generator().manual_seed(0))
    assert not torch.equal(a.sensory.fc1.weight, w0)

    b = _judge()
    jt_b = vi.sync_target(b)
    w0 = b.sensory.fc1.weight.detach().clone()
    h0 = b.head[0].weight.detach().clone()
    vi.train_step(b, jt_b, vi.make_optimizer(b, "B"), states, N_ACTIONS,
                  torch.Generator().manual_seed(0))
    assert torch.equal(b.sensory.fc1.weight, w0)
    assert not torch.equal(b.head[0].weight, h0)


def test_judge_output_is_nonnegative_and_batched():
    """Catches a missing softplus or a shape bug in the batched forward."""
    states = vi.random_walk_states(8, 6, random.Random(2), N_ACTIONS, exclude=set())
    v = _judge()(states, torch.Generator().manual_seed(0))
    assert v.shape == (8,) and bool((v >= 0).all())


def test_targets_use_one_batched_call_per_draw():
    """Catches per-child encoding (n times slower) by counting judge calls."""
    calls = []

    class Spy(_ConstJudge):
        def forward(self, states, generator):
            calls.append(len(states))
            return super().forward(states, generator)

    states = vi.random_walk_states(5, 4, random.Random(3), N_ACTIONS, exclude=set())
    vi.bellman_targets(states, Spy({}), N_ACTIONS, torch.Generator(), draws=2)
    assert calls == [5 * N_ACTIONS, 5 * N_ACTIONS]


def test_sync_target_survives_a_judge_that_already_ran_a_live_graph_forward():
    """Catches sync_target deepcopying graph-attached membrane state. SensoryCortex.forward
    stores self.mem1/self.mem2 as plain attributes; after a grad-enabled forward (exactly what
    train_step does in arm A) those are non-leaf tensors, and torch's deepcopy protocol raises
    on a non-leaf tensor. Task 2's train_judge calls sync_target mid-training, after the judge has
    already run forward passes with gradients on, so this is not a hypothetical: the literal
    plan code raises RuntimeError here before any fix. A sync_target that resets the membrane
    state (or otherwise avoids deepcopying a graph-attached tensor) first must pass this."""
    judge = _judge()
    jt = vi.sync_target(judge)
    opt = vi.make_optimizer(judge, "A")
    states = vi.random_walk_states(8, 3, random.Random(0), N_ACTIONS, exclude=set())
    vi.train_step(judge, jt, opt, states, N_ACTIONS, torch.Generator().manual_seed(0))
    # judge.sensory now holds graph-attached mem1/mem2 from the training forward above.
    jt2 = vi.sync_target(judge)
    assert all(not p.requires_grad for p in jt2.parameters())
    assert all(torch.equal(p, q) for p, q in zip(judge.parameters(), jt2.parameters()))
