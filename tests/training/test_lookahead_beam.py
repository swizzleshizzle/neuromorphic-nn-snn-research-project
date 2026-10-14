"""EXP-077 seam: a beam search over judge values. Each docstring names the bug it catches."""

from __future__ import annotations

from collections import deque

import pytest
import torch
import torch.nn as nn

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import lookahead as la
from neuromorphic.training.cube_baseline import CubeConfig, make_agent, shell_states


def _agent_and_head(seed=0):
    agent = make_agent(CubeConfig(arm="regionalized", seed=seed))
    torch.manual_seed(seed)
    head = nn.Linear(agent.content, N_ACTIONS)
    return agent, head


def _shell(depth, n):
    return shell_states(ExactBFSDistance(max_depth=depth), depth)[:n]


class _HashCritic:
    """Deterministic, nearly tie-free state scores that ignore the generator."""

    reads_states = True

    def __init__(self):
        self.calls = []

    def __call__(self, states, generator):
        self.calls.append(len(states))
        return torch.tensor([float(hash(s) % 1_000_003) for s in states])


class _ConstCritic:
    reads_states = True

    def __call__(self, states, generator):
        return torch.zeros(len(states))


def _bfs_shell(root, depth, visited):
    """States at exactly `depth` moves from root in the move graph with `visited` removed
    (root itself is the start and is never re-entered). An independent reference."""
    dist = {root: 0}
    q = deque([root])
    while q:
        s = q.popleft()
        if dist[s] == depth:
            continue
        for a in range(N_ACTIONS):
            c = apply_move(s, a)
            if c in dist or c in visited:
                continue
            dist[c] = dist[s] + 1
            q.append(c)
    return {s for s, d in dist.items() if d == depth}


def test_full_width_beam_final_frontier_is_the_bfs_shell():
    """Catches the shallowest-level rule being wrong (a state scored twice, or at a deeper level
    than it is first reached, or a visited state kept): with width larger than any level, the
    final frontier must be exactly the BFS shell at the beam depth, and the chosen leaf its
    best-scoring member."""
    critic = _HashCritic()
    for i, root in enumerate(_shell(6, 25)):
        visited = {root, apply_move(root, i % N_ACTIONS)}
        info = {}
        move = la.beam_move(root, 10_000, 4, N_ACTIONS, agent=None, critic=critic,
                            imag_generator=torch.Generator().manual_seed(0), visited=visited,
                            info=info)
        shell = _bfs_shell(root, 4, visited)
        assert set(info["beam_frontier"]) == shell
        best = max(shell, key=lambda s: float(hash(s) % 1_000_003))
        assert info["beam_leaf"] == best
        assert move is not None


def test_depth_one_beam_equals_mode_c_k1():
    """Catches the beam diverging from the full tree where they must agree: at depth 1 any width
    plays mode C's k 1 move, with no visited set, with one, and when every neighbour is visited
    (the fallback, which must also be flagged)."""
    critic = _HashCritic()
    logits = torch.zeros(N_ACTIONS)
    for root in _shell(5, 30):
        nbrs = {apply_move(root, a) for a in range(N_ACTIONS)}
        for visited in (None, {root, apply_move(root, 2)}, {root} | nbrs):
            ic, ib = {}, {}
            c, _ = la.choose_move("C", root, logits, 1, N_ACTIONS, critic=critic,
                                  imag_generator=torch.Generator().manual_seed(1),
                                  visited=visited, info=ic, k_goal=1)
            b, _ = la.choose_move("B", root, logits, 1, N_ACTIONS, critic=critic,
                                  imag_generator=torch.Generator().manual_seed(1),
                                  visited=visited, info=ib, k_goal=1, beam_width=3)
            assert b == c
            assert ib.get("fallback", False) == ic.get("fallback", False)


def test_width_one_is_greedy_hill_climbing():
    """Catches width being ignored: width 1 must follow the single best child at every level,
    computed here by hand from the same scores."""
    critic = _HashCritic()
    score = lambda s: float(hash(s) % 1_000_003)
    for root in _shell(6, 20):
        seen, cur, first = {root}, root, None
        for _ in range(3):
            kids = [(apply_move(cur, a), a) for a in range(N_ACTIONS)]
            kids = [(c, a) for c, a in kids if c not in seen]
            seen.update(c for c, _ in kids)
            c, a = max(kids, key=lambda t: score(t[0]))
            first = a if first is None else first
            cur = c
        info = {}
        move = la.beam_move(root, 1, 3, N_ACTIONS, agent=None, critic=critic,
                            imag_generator=torch.Generator(), visited={root}, info=info)
        assert (move, info["beam_leaf"]) == (first, cur)


def test_ties_go_to_the_earlier_child():
    """Catches an unstable sort or a reversed order: with every score equal, width 1 depth 1
    plays move 0."""
    root = _shell(5, 1)[0]
    assert la.beam_move(root, 1, 1, N_ACTIONS, agent=None, critic=_ConstCritic(),
                        imag_generator=torch.Generator(), visited={root}) == 0


def test_one_batched_judge_call_per_level_and_the_eval_count():
    """Catches per-node judge calls (5 calls is the spec's cost model, not 1000) and a wrong
    judge_evals count: depth 3 makes exactly 3 calls whose sizes sum to judge_evals, the first
    being 6 and the second at most 6 * 6."""
    critic = _HashCritic()
    root = _shell(6, 1)[0]
    info = {}
    la.beam_move(root, 4, 3, N_ACTIONS, agent=None, critic=critic,
                 imag_generator=torch.Generator(), visited={root}, info=info)
    assert len(critic.calls) == 3
    assert critic.calls[0] == N_ACTIONS
    assert critic.calls[1] <= 4 * N_ACTIONS and critic.calls[2] <= 4 * N_ACTIONS
    assert info["judge_evals"] == sum(critic.calls)


def test_beam_width_is_refused_outside_mode_b_and_required_in_it():
    """Catches a silent beam_width on mode C (an arm mislabelled) and mode B without a width."""
    root = _shell(5, 1)[0]
    logits = torch.zeros(N_ACTIONS)
    with pytest.raises(ValueError):
        la.choose_move("C", root, logits, 2, N_ACTIONS, critic=_HashCritic(), beam_width=4)
    with pytest.raises(ValueError):
        la.choose_move("B", root, logits, 2, N_ACTIONS, critic=_HashCritic())
    with pytest.raises(ValueError):
        la.choose_move("B", root, logits, 2, N_ACTIONS, beam_width=4)


def test_goal_check_runs_before_the_beam():
    """Catches the beam overriding a solve: a state two moves from solved with k_goal 2 plays the
    solving move and fires, whatever the critic says."""
    root = apply_move(apply_move(SOLVED, 0), 2)
    logits = torch.zeros(N_ACTIONS)
    info = {}
    move, fired = la.choose_move("B", root, logits, 3, N_ACTIONS, critic=_ConstCritic(),
                                 imag_generator=torch.Generator(), visited={root}, info=info,
                                 k_goal=3, beam_width=2)
    assert fired is True and info["fire_len"] == 2
    assert move == la.solving_prefix(root, 3, N_ACTIONS)[0]
    assert "judge_evals" not in info


def test_evaluate_records_judge_evals_and_beam_width():
    """Catches the instrument the spec gates on (Gate B) being missing or wrong: mode C's
    judge_evals is exactly 6^k, and a mode B record carries its width and a mean no larger than
    the per-move bound."""
    agent, head = _agent_and_head()
    states = _shell(6, 3)
    common = dict(depth=6, generator=torch.Generator().manual_seed(3), rng_seed=3, imag_seed=3,
                  critic=_HashCritic(), no_revisit=True)
    rc = la.evaluate_lookahead(agent, head, states, mode="C", k=2, **common)
    assert rc["judge_evals"] == N_ACTIONS ** 2
    assert "beam_width" not in rc
    common["generator"] = torch.Generator().manual_seed(3)
    rb = la.evaluate_lookahead(agent, head, states, mode="B", k=3, k_goal=3, beam_width=4,
                               **common)
    assert rb["beam_width"] == 4 and rb["mode"] == "B" and rb["k"] == 3
    assert 0 < rb["judge_evals"] <= N_ACTIONS + 2 * 4 * N_ACTIONS
