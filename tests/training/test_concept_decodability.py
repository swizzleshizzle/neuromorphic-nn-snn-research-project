"""EXP-033 Part 1: can a linear probe read the optimal move out of the frozen concept?

The probe is the instrument, so it has to be shown to discriminate before any result from
it means anything. `test_probe_learns_a_separable_mapping` and
`test_probe_stays_near_chance_on_shuffled_labels` are a matched pair: the first shows the
probe can find structure, the second shows it does not manufacture structure that is not
there. A probe that passed only the first could be reporting its own capacity.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import MOVES, SOLVED, apply_move, inverse_action
from neuromorphic.envs.cube_distance import ExactBFSDistance

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "exp033_probe", ROOT / "experiments" / "033_concept_decodability" / "probe.py"
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

N_ACTIONS = len(MOVES)  # never a literal: a 3x3 cube is 12 or 18


@pytest.fixture(scope="module")
def provider():
    return ExactBFSDistance(max_depth=4)


def test_optimal_mask_marks_exactly_the_distance_reducing_moves(provider):
    """Ground truth by construction: recompute the distance after every move."""
    for state in provider.states_at_distance(3)[:20]:
        mask = probe.optimal_move_mask(provider, state)
        assert len(mask) == N_ACTIONS
        d = provider.distance(state)
        for a in range(N_ACTIONS):
            reduces = provider.distance(apply_move(state, a)) < d
            assert mask[a] == reduces, f"move {a} mislabelled at distance {d}"


def test_every_state_off_solved_has_at_least_one_optimal_move(provider):
    """Otherwise the label set would be empty and top-1 accuracy undefined."""
    for depth in (1, 2, 3):
        for state in provider.states_at_distance(depth)[:15]:
            assert any(probe.optimal_move_mask(provider, state))


def test_undoing_a_single_move_is_optimal_at_depth_one(provider):
    """Independent of the distance table's own bookkeeping: construct the state by hand.

    One move from SOLVED, so undoing it must be optimal. If `optimal_move_mask` disagreed
    with this, the labels would be wrong and every number in the experiment with them.
    """
    for a in range(N_ACTIONS):
        state = apply_move(SOLVED, a)
        mask = probe.optimal_move_mask(provider, state)
        assert mask[inverse_action(a)] is True


def test_chance_floor_is_measured_from_the_masks_not_assumed(provider):
    """CLAUDE.md: measure the chance floor, do not assume it.

    A uniformly random guess is correct with probability equal to the mean fraction of
    moves that are optimal, which is NOT 1/6 in general.
    """
    states = provider.states_at_distance(3)[:60]
    masks = [probe.optimal_move_mask(provider, s) for s in states]
    floor = probe.chance_floor(masks)
    hand = sum(sum(m) / len(m) for m in masks) / len(masks)
    assert floor == pytest.approx(hand)
    assert 0.0 < floor < 1.0


def test_top1_accuracy_counts_a_hit_only_when_the_argmax_move_is_optimal():
    masks = [[False, True, False, False, False, False],
             [True, False, False, False, False, False]]
    logits = torch.tensor([[0.0, 9.0, 0.0, 0.0, 0.0, 0.0],   # picks 1 -> optimal
                           [0.0, 9.0, 0.0, 0.0, 0.0, 0.0]])  # picks 1 -> not optimal
    assert probe.top1_accuracy(logits, masks) == pytest.approx(0.5)


def test_probe_learns_a_separable_mapping():
    """The instrument can find structure when structure exists."""
    g = torch.Generator().manual_seed(0)
    n, dim = 400, 16
    y = torch.randint(0, N_ACTIONS, (n,), generator=g)
    x = torch.zeros(n, dim)
    x[torch.arange(n), y % dim] = 1.0           # trivially separable
    masks = [[i == int(t) for i in range(N_ACTIONS)] for t in y]
    model = probe.fit_linear_probe(x, masks, epochs=250, lr=0.1, seed=0)
    acc = probe.top1_accuracy(model(x), masks)
    assert acc > 0.90, f"probe failed to fit separable data: {acc:.3f}"


def test_probe_stays_near_chance_on_shuffled_labels():
    """The instrument does not manufacture structure that is not there.

    Random features against random labels: anything much above the chance floor would mean
    the probe is memorising, and every decodability number it reports would be inflated.
    """
    g = torch.Generator().manual_seed(1)
    n, dim = 400, 16
    x = torch.randn(n, dim, generator=g)
    y = torch.randint(0, N_ACTIONS, (n,), generator=g)
    masks = [[i == int(t) for i in range(N_ACTIONS)] for t in y]
    split = n // 2
    model = probe.fit_linear_probe(x[:split], masks[:split], epochs=250, lr=0.1, seed=0)
    heldout = probe.top1_accuracy(model(x[split:]), masks[split:])
    floor = probe.chance_floor(masks[split:])
    assert heldout < floor + 0.15, f"probe memorised noise: {heldout:.3f} vs floor {floor:.3f}"


def test_split_is_disjoint_and_covers_every_state(provider):
    states = provider.states_at_distance(3)
    train, held = probe.split_states(states, frac_heldout=0.25, seed=0)
    assert set(train).isdisjoint(set(held))
    assert set(train) | set(held) == set(states)
    assert len(held) == pytest.approx(0.25 * len(states), abs=1)


def test_split_is_deterministic_per_seed_and_differs_across_seeds(provider):
    states = provider.states_at_distance(4)
    a1, _ = probe.split_states(states, frac_heldout=0.25, seed=0)
    a2, _ = probe.split_states(states, frac_heldout=0.25, seed=0)
    b1, _ = probe.split_states(states, frac_heldout=0.25, seed=1)
    assert a1 == a2
    assert a1 != b1


def test_mask_refuses_to_label_at_the_table_boundary():
    """A neighbour one step deeper than the table would otherwise be silently mislabelled.

    A depth-4 state has depth-5 neighbours. With a depth-4 provider those return None, and
    treating None as "not distance-reducing" is right most of the time and wrong some of
    the time, which is the worst kind of bug. It must raise instead.
    """
    shallow = ExactBFSDistance(max_depth=4)
    state = shallow.states_at_distance(4)[0]
    with pytest.raises(ValueError, match="outside the table"):
        probe.optimal_move_mask(shallow, state)
