"""Tests for the look-ahead procedure (EXP-070). Each docstring names the bug it catches."""

from __future__ import annotations

import inspect
import itertools

import pytest
import torch

from neuromorphic.envs.cube import SOLVED, apply_move, inverse_action, is_solved, N_ACTIONS
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import lookahead as la
from neuromorphic.training.cube_baseline import shell_states


def _state_from(moves):
    s = SOLVED
    for a in moves:
        s = apply_move(s, a)
    return s


def test_solving_prefix_finds_a_two_move_solve_at_k2_and_not_at_k1():
    """Catches an off-by-one in tree depth: a k=1 search must NOT see a 2-move solve."""
    s = _state_from([0, 2])  # two different faces, so no 1-move solve exists
    assert la.solving_prefix(s, 1, N_ACTIONS) is None
    seq = la.solving_prefix(s, 2, N_ACTIONS)
    assert seq is not None and len(seq) == 2
    assert is_solved(_state_from_on(s, seq))


def _state_from_on(state, moves):
    for a in moves:
        state = apply_move(state, a)
    return state


def test_solving_prefix_finds_exactly_depth_k_from_a_true_depth_3_state():
    """Catches a search that stops one level short (k-1) or reads one level too far (k+1)."""
    s = shell_states(ExactBFSDistance(max_depth=3), 3)[0]
    assert la.solving_prefix(s, 2, N_ACTIONS) is None
    seq = la.solving_prefix(s, 3, N_ACTIONS)
    assert seq is not None and len(seq) == 3 and is_solved(_state_from_on(s, seq))


def test_solving_prefix_prefers_the_shortest_solve():
    """Catches returning a length-k solve when a shorter one exists (e.g. U U U from U')."""
    s = _state_from([0])
    assert la.solving_prefix(s, 3, N_ACTIONS) == (inverse_action(0),)


def test_tree_levels_take_width_from_the_argument_not_a_literal():
    """Catches a hardcoded 6: a 12-action stub must give 1, 12, 144 states in product order."""
    def stub_apply(s, a):
        return s + (a,)
    levels = la.tree_levels((), 2, 12, apply_fn=stub_apply)
    assert [len(l) for l in levels] == [1, 12, 144]
    assert levels[2] == [tuple(p) for p in itertools.product(range(12), repeat=2)]


def test_solving_prefix_takes_width_from_the_argument():
    """Catches a hardcoded 6 in the goal test: action 11 only exists in a 12-action space."""
    def stub_apply(s, a):
        return s + (a,)
    seq = la.solving_prefix((), 2, 12, apply_fn=stub_apply, goal_fn=lambda s: s == (11,))
    assert seq == (11,)


def test_sequence_scores_sum_log_probs_along_each_path():
    """Catches wrong parent/child index arithmetic, which would score a sequence with a
    sibling's log-probabilities. Values are distinct powers of ten so any mix-up shows."""
    lp0 = torch.tensor([[1.0, 2.0]])
    lp1 = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    lp2 = torch.tensor([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0], [700.0, 800.0]])
    got = la.sequence_scores([lp0, lp1, lp2], 3, 2)
    want = []
    for a, b, c in itertools.product(range(2), repeat=3):
        want.append(lp0[0, a] + lp1[a, b] + lp2[a * 2 + b, c])
    assert torch.equal(got, torch.stack(want))


def test_lookahead_module_never_imports_the_bfs_provider():
    """Catches an oracle leak at the source level: the procedure must not reach distance."""
    src = inspect.getsource(la)
    assert "cube_distance" not in src
    assert "ExactBFSDistance" not in src


import numpy as np
import torch.nn as nn

from neuromorphic.training.cube_baseline import CubeConfig, make_agent
from neuromorphic.training.reinforce import action_distribution, concept_rate


def _agent_and_head(seed=0):
    agent = make_agent(CubeConfig(arm="regionalized", seed=seed))
    torch.manual_seed(seed)
    head = nn.Linear(agent.content, N_ACTIONS)
    return agent, head


def _far_state():
    """A state at exact depth 6, proven by BFS (tests may use it): no solve within k <= 3."""
    return shell_states(ExactBFSDistance(max_depth=6), 6)[0]


def test_imagined_logp_for_one_state_matches_the_evaluation_path():
    """Catches the batched path and the evaluation path disagreeing (a different readout
    or a different concept reduction), AND catches recall being left on.

    recall cannot be caught by the returned log-probs alone: in `Brain.step`, `concept` is
    computed BEFORE the hippocampal store/recall branch runs, so `out["concept"]` (and
    therefore these log-probs) is bit-identical whether `recall` is True or False. A spy on
    the `recall` kwarg itself is required to catch a `recall=True` regression, matching the
    CLAUDE.md invariant that `recall=False` keeps only the sensory region on the policy path.
    """
    agent, head = _agent_and_head()
    s = _far_state()
    seen_recall = []
    real_step = agent.step

    def spying_step(obs, **kwargs):
        seen_recall.append(kwargs.get("recall"))
        return real_step(obs, **kwargs)

    agent.step = spying_step
    got = la.imagined_logp(agent, head, [s], generator=torch.Generator().manual_seed(7))
    assert seen_recall == [False]
    with torch.no_grad():
        _, logits = action_distribution(agent, head, np.array(s),
                                        generator=torch.Generator().manual_seed(7))
    # 1e-6, not 0: a [1, 64] matmul and a [64] matvec may differ in the last bit. A different
    # reduction moves these by orders of magnitude more; recall is caught by the spy above.
    assert torch.allclose(got[0], torch.log_softmax(logits, dim=-1), atol=1e-6, rtol=0)


def test_sensory_batch_rows_are_independent():
    """Catches cross-row mixing in a batched forward: row i of a batch must equal the same
    input run alone. Uses fixed spikes, so Poisson noise cannot mask a difference."""
    agent, _ = _agent_and_head()
    g = torch.Generator().manual_seed(3)
    spikes = (torch.rand(agent.T, 3, 144, generator=g) < 0.3).float()
    with torch.no_grad():
        batched = agent.sensory(spikes)
        alone = agent.sensory(spikes[:, 1:2, :])
    assert torch.equal(batched[:, 1:2, :], alone)


def test_r_scores_do_not_depend_on_the_head():
    """Catches the random scorer accidentally reading the network: two different heads must
    give the same R choice under the same imagined seed."""
    a0, h0 = _agent_and_head(0)
    _, h1 = _agent_and_head(1)
    s = _far_state()
    root = torch.zeros(N_ACTIONS)
    m0 = la.choose_move("R", s, root, 3, N_ACTIONS, agent=a0, head=h0,
                        imag_generator=torch.Generator().manual_seed(5))
    m1 = la.choose_move("R", s, root, 3, N_ACTIONS, agent=a0, head=h1,
                        imag_generator=torch.Generator().manual_seed(5))
    assert m0 == m1


def test_r_choice_varies_with_the_imagined_seed():
    """Catches a constant 'random' scorer (always index 0), which would make R a fixed policy.
    Over 20 seeds a uniform scorer picks at least 3 distinct first moves."""
    s = _far_state()
    root = torch.zeros(N_ACTIONS)
    picks = {la.choose_move("R", s, root, 2, N_ACTIONS,
                            imag_generator=torch.Generator().manual_seed(i))[0]
             for i in range(20)}
    assert len(picks) >= 3


def test_e_plays_greedy_when_the_goal_test_does_not_fire():
    """Catches E silently scoring or reordering: with no solve in reach E must equal argmax."""
    s = _far_state()
    root = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.0, 0.5])
    assert la.choose_move("E", s, root, 3, N_ACTIONS) == (1, False)


def test_every_lookahead_mode_takes_a_solve_in_reach():
    """Catches a mode that skips the goal test: one move from solved, E, P and R must all take
    the solving move whatever the logits say."""
    agent, head = _agent_and_head()
    s = _state_from([2])
    root = torch.zeros(N_ACTIONS)
    root[inverse_action(2)] = -50.0  # the network hates the solving move
    for mode in ("E", "P", "R"):
        k = 2
        got = la.choose_move(mode, s, root, k, N_ACTIONS, agent=agent, head=head,
                             imag_generator=torch.Generator().manual_seed(0))
        assert got == (inverse_action(2), True), mode


def test_g_ignores_the_goal_test():
    """Catches G picking up the goal test, which would break Gate 0."""
    s = _state_from([2])
    root = torch.zeros(N_ACTIONS)
    root[0] = 1.0
    assert la.choose_move("G", s, root, 0, N_ACTIONS) == (0, False)


def test_g_ignores_a_solve_in_reach_even_when_given_a_lookahead_depth():
    """Catches G consulting the goal test. At k=0 the goal test is structurally empty, so the
    test above only catches this via the k<1 guard; here k=2 and a 1-move solve IS in reach,
    and G must still play the greedy move."""
    s = _state_from([2])
    root = torch.zeros(N_ACTIONS)
    root[0] = 1.0
    assert inverse_action(2) != 0
    assert la.choose_move("G", s, root, 2, N_ACTIONS) == (0, False)


class _StubAgent:
    """Returns a 2-wide concept: [1, 0] for one marked state, [0, 1] for every other."""

    def __init__(self, marked):
        self.marked = tuple(marked)

    def step(self, obs, *, recall=False, generator=None, **_):
        rows = np.asarray(obs)
        rows = rows[None, :] if rows.ndim == 1 else rows
        feats = torch.tensor([[1.0, 0.0] if tuple(r) == self.marked else [0.0, 1.0]
                              for r in rows.tolist()])
        return {"concept": feats.unsqueeze(0)}  # [T=1, B, 2]


def test_p_picks_the_first_move_of_the_best_scored_sequence():
    """Catches P returning the best sequence's LAST move (or greedy). The root mildly prefers
    move 1 (greedy would play it), but only the child reached by move 4 is one where the head is
    confident (log-prob ~0 for move 2); every other child is uniform (log-prob -log 6). So the
    best length-2 sequence is (4, 2) by a margin of ~1.7, and P must play 4: not 1 (greedy) and
    not 2 (the last move)."""
    s = _far_state()
    agent = _StubAgent(apply_move(s, 4))
    head = nn.Linear(2, N_ACTIONS)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[2, 0] = 50.0
    root = torch.tensor([0.0, 0.5, 0.0, 0.0, 0.4, 0.0])
    action, fired = la.choose_move("P", s, root, 2, N_ACTIONS, agent=agent, head=head,
                                   imag_generator=torch.Generator().manual_seed(0))
    assert (action, fired) == (4, False)


from neuromorphic.training.cube_baseline import evaluate_states


def _shell(depth, n):
    return shell_states(ExactBFSDistance(max_depth=depth), depth)[:n]


def test_mode_g_reproduces_evaluate_states_exactly():
    """THE GATE 0 CODE PATH. Catches any drift between G and the published evaluator: a
    different generator draw, budget, termination rule or metric formula. Every field must
    match at full float repr."""
    agent, head = _agent_and_head()
    states = _shell(2, 8)
    ref = evaluate_states(agent, head, states, depth=2,
                          generator=torch.Generator().manual_seed(4), rng_seed=4)
    got = la.evaluate_lookahead(agent, head, states, depth=2, mode="G", k=0,
                                generator=torch.Generator().manual_seed(4), rng_seed=4)
    for key, value in ref.items():
        assert got[key] == value, key
    assert got["solved"] == round(ref["success_rate"] * ref["n"])


def test_e_matches_g_move_for_move_until_the_goal_test_first_fires():
    """Catches E consuming the evaluation stream differently from G: before E's first fired
    step, the two must play identical moves on every state."""
    agent, head = _agent_and_head()
    states = _shell(4, 6)
    tg, te = [], []
    la.evaluate_lookahead(agent, head, states, depth=4, mode="G", k=0,
                          generator=torch.Generator().manual_seed(2), rng_seed=2, trace=tg)
    la.evaluate_lookahead(agent, head, states, depth=4, mode="E", k=1,
                          generator=torch.Generator().manual_seed(2), rng_seed=2, trace=te)
    compared = 0
    for g_steps, e_steps in zip(tg, te):
        for (ga, _, _), (ea, fired, _) in zip(g_steps, e_steps):
            if fired:
                break
            assert ga == ea
            compared += 1
    assert compared >= 10  # the test must actually compare something


def test_p_real_logits_match_g_while_their_moves_agree():
    """Catches imagined states drawing from the EVALUATION stream inside evaluate_lookahead:
    while P has played exactly G's moves, its real-state logits must be bit-identical to G's.
    Any imagined draw on the real stream shifts every later real encoding."""
    agent, head = _agent_and_head()
    with torch.no_grad():
        head.bias.zero_()
        head.bias[3] = 20.0  # P and G then agree on every move; logits still read the concept
    states = _shell(5, 8)
    tg, tp = [], []
    la.evaluate_lookahead(agent, head, states, depth=5, mode="G", k=0,
                          generator=torch.Generator().manual_seed(6), rng_seed=6, trace=tg)
    la.evaluate_lookahead(agent, head, states, depth=5, mode="P", k=2,
                          generator=torch.Generator().manual_seed(6), rng_seed=6, trace=tp)
    compared_after_first = 0
    for g_steps, p_steps in zip(tg, tp):
        for t, ((ga, _, gl), (pa, _, pl)) in enumerate(zip(g_steps, p_steps)):
            assert gl == pl, f"real logits diverged at step {t} while moves agreed"
            if t > 0:
                compared_after_first += 1
            if ga != pa:
                break
    assert compared_after_first >= 20  # the test must reach well past the first move


def test_the_bfs_provider_is_never_consulted_during_a_rollout(monkeypatch):
    """Catches an oracle leak at run time: if any code path in a P rollout reads distance,
    this raises."""
    agent, head = _agent_and_head()
    states = _shell(3, 2)  # built BEFORE the provider is poisoned

    def boom(*_a, **_k):
        raise AssertionError("BFS distance read inside the procedure")

    monkeypatch.setattr(ExactBFSDistance, "distance", boom)
    monkeypatch.setattr(ExactBFSDistance, "__init__", boom)
    out = la.evaluate_lookahead(agent, head, states, depth=3, mode="P", k=2,
                                generator=torch.Generator().manual_seed(0), rng_seed=0)
    assert out["n"] == 2


def test_rollouts_are_deterministic():
    """Gate 0(a). Catches any unseeded randomness in the procedure (e.g. torch's global RNG)."""
    agent, head = _agent_and_head()
    states = _shell(3, 4)
    runs = [la.evaluate_lookahead(agent, head, states, depth=3, mode="R", k=2,
                                  generator=torch.Generator().manual_seed(1), rng_seed=1,
                                  imag_seed=1)
            for _ in range(2)]
    assert runs[0] == runs[1]


class _StubCritic(nn.Module):
    """Value = the concept's first coordinate, so a stub agent can dictate the ranking."""

    def forward(self, x):
        return x[..., :1]


class _ValueAgent:
    """Concept [v, 0] where v is looked up per state (default 0); T=1."""

    def __init__(self, values):
        self.values = {tuple(k): v for k, v in values.items()}

    def step(self, obs, *, recall=False, generator=None, **_):
        rows = np.asarray(obs)
        rows = rows[None, :] if rows.ndim == 1 else rows
        feats = torch.tensor([[float(self.values.get(tuple(r), 0.0)), 0.0]
                              for r in rows.tolist()])
        return {"concept": feats.unsqueeze(0)}


def test_c1_plays_the_child_the_critic_rates_highest():
    """Catches C reading the policy instead of the critic: the root prefers move 0, the
    critic prefers the child reached by move 3."""
    s = _far_state()
    agent = _ValueAgent({apply_move(s, 3): 5.0})
    root = torch.tensor([9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    got = la.choose_move("C", s, root, 1, N_ACTIONS, agent=agent, critic=_StubCritic(),
                         imag_generator=torch.Generator().manual_seed(0))
    assert got == (3, False)


def test_c3_scores_sequences_by_their_leaf_not_their_first_child():
    """Catches C scoring level 1 instead of the leaves: the only high-value state is a leaf
    reached by (2, 4, 1), whose first child is unremarkable. C3 must play 2."""
    s = _far_state()
    leaf = apply_move(apply_move(apply_move(s, 2), 4), 1)
    agent = _ValueAgent({leaf: 5.0, apply_move(s, 0): 1.0})
    got = la.choose_move("C", s, torch.zeros(N_ACTIONS), 3, N_ACTIONS, agent=agent,
                         critic=_StubCritic(), imag_generator=torch.Generator().manual_seed(0))
    assert got == (2, False)


def test_c_takes_a_solve_in_reach_before_consulting_the_critic():
    """Catches C skipping the goal test."""
    s = _state_from([2])
    agent = _ValueAgent({apply_move(s, 0): 99.0})
    got = la.choose_move("C", s, torch.zeros(N_ACTIONS), 1, N_ACTIONS, agent=agent,
                         critic=_StubCritic(), imag_generator=torch.Generator().manual_seed(0))
    assert got == (inverse_action(2), True)


def test_sequence_eligibility_excludes_any_sequence_through_a_visited_state():
    """Catches checking only the leaf (or only the first move): a visited state at level 2
    must exclude exactly the n sequences through it, and nothing else."""
    def stub_apply(s, a):
        return s + (a,)
    levels = la.tree_levels((), 3, 3, apply_fn=stub_apply)
    mask = la.sequence_eligibility(levels, 3, 3, {(1, 2)})
    excluded = [j for j in range(27) if not mask[j]]
    assert excluded == [1 * 9 + 2 * 3 + c for c in range(3)]


def test_g_with_v_plays_the_best_unvisited_move():
    """Catches V being ignored by G: the greedy move leads to a visited state, so G0V must
    take the next-best logit."""
    s = _far_state()
    root = torch.tensor([0.0, 9.0, 5.0, 0.0, 0.0, 0.0])
    info = {}
    got = la.choose_move("G", s, root, 0, N_ACTIONS, visited={apply_move(s, 1)}, info=info)
    assert got == (2, False) and not info.get("fallback", False)


def test_v_falls_back_and_says_so_when_every_candidate_is_visited():
    """Catches a crash or a silent masked argmax over all -inf when nothing is eligible."""
    s = _far_state()
    root = torch.tensor([0.0, 9.0, 5.0, 0.0, 0.0, 0.0])
    everything = {apply_move(s, a) for a in range(N_ACTIONS)}
    info = {}
    got = la.choose_move("G", s, root, 0, N_ACTIONS, visited=everything, info=info)
    assert got == (1, False) and info["fallback"] is True


def test_r_draws_exactly_n_actions_to_the_k_random_numbers_and_nothing_else():
    """Catches ANY extra draw in R's path, conditional or not, against an independent
    from-scratch reference: scores must be torch.rand(n_actions**k, generator=g) from a fresh
    generator seeded identically, with nothing drawn before it (interface contract, Task 1).

    Added because the plan's own mutation table entry for an UNCONDITIONAL extra draw ('in the
    R branch, add torch.rand(1, generator=imag_generator) before the real draw') does not fail
    the committed golden fixture test: empirically the bucket-collapse rate for this exact
    perturbation is about 2.75% per real move (measured directly), so the golden fixture's 3
    states x 17 steps has about a 24% chance of missing it by pure luck, and in fact does miss
    it. The golden fixture is pinned and must not be touched (plan constraint), so this test
    checks the same invariant directly and with much higher power (200 seeds here; the mutation
    above was confirmed to produce 5/200 mismatches against 0/200 on correct code)."""
    s = _far_state()
    root = torch.zeros(N_ACTIONS)
    for seed in range(200):
        ref = torch.rand(N_ACTIONS ** 3, generator=torch.Generator().manual_seed(seed))
        got = la.choose_move("R", s, root, 3, N_ACTIONS,
                             imag_generator=torch.Generator().manual_seed(seed))
        want = int(ref.argmax()) // N_ACTIONS ** 2
        assert got == (want, False), seed


def test_r_draws_the_same_random_scores_with_and_without_v():
    """Catches V changing R's stream: with nothing visited, R3 and R3V must choose alike.

    Checked over many seeds, not just one: an extra draw consumed only on the visited branch
    does shift torch.rand's output (verified directly), but a single badly-chosen seed can still
    collapse to the same first-move bucket by coincidence (216 leaves map onto 6 first moves).
    Seed 4 alone does not discriminate this mutation; seed 43 does, confirmed empirically against
    both the correct code (always equal here) and the mutation (differs at seed 43). Plan note:
    the plan's one-seed version of this test did not catch its own listed mutation; the input was
    widened rather than the threshold weakened."""
    s = _far_state()
    root = torch.zeros(N_ACTIONS)
    for seed in range(60):
        a = la.choose_move("R", s, root, 3, N_ACTIONS,
                           imag_generator=torch.Generator().manual_seed(seed))
        b = la.choose_move("R", s, root, 3, N_ACTIONS,
                           imag_generator=torch.Generator().manual_seed(seed), visited=set())
        assert a == b, seed


def test_no_revisit_cuts_the_revisit_rate_of_a_looping_reflex():
    """Catches V not reaching the rollout: a head with a dominant bias turns one face over
    and over (revisit rate high); with no_revisit=True the same head must revisit far less."""
    agent, head = _agent_and_head()
    with torch.no_grad():
        head.bias.zero_()
        head.bias[3] = 20.0
    states = _shell(5, 6)
    plain = la.evaluate_lookahead(agent, head, states, depth=5, mode="G", k=0,
                                  generator=torch.Generator().manual_seed(1), rng_seed=1)
    v = la.evaluate_lookahead(agent, head, states, depth=5, mode="G", k=0,
                              generator=torch.Generator().manual_seed(1), rng_seed=1,
                              no_revisit=True)
    assert plain["eval_revisit_rate"] > 0.5
    assert v["eval_revisit_rate"] < 0.5 * plain["eval_revisit_rate"]
    assert v["no_revisit"] is True and plain["no_revisit"] is False


def test_imagined_values_reads_the_concept_with_recall_off():
    """Catches the critic path reading something other than what the critic was trained on."""
    agent, _ = _agent_and_head()
    critic = nn.Linear(agent.content, 1)
    s = _far_state()
    seen = []
    real_step = agent.step

    def spy(obs, **kw):
        seen.append(kw.get("recall"))
        return real_step(obs, **kw)

    agent.step = spy
    v = la.imagined_values(agent, critic, [s, s], generator=torch.Generator().manual_seed(0))
    assert seen == [False] and v.shape == (2,)


@pytest.mark.parametrize("mode", ["P", "C", "R"])
def test_scored_modes_fall_back_and_say_so_when_every_sequence_is_visited(mode):
    """Catches the scored-mode fallback (a separate branch from G's) dropping its flag or
    taking an argmax over all -inf: with every leaf-path through a visited child, the move
    must equal the unmasked choice and info must record the fallback."""
    s = _far_state()
    agent = _ValueAgent({apply_move(s, 3): 5.0})
    _, head = _agent_and_head()
    everything = {apply_move(s, a) for a in range(N_ACTIONS)}
    root = torch.zeros(N_ACTIONS)
    kw = dict(agent=agent if mode == "C" else _agent_and_head()[0], head=head,
              critic=_StubCritic())
    plain = la.choose_move(mode, s, root, 1, N_ACTIONS,
                           imag_generator=torch.Generator().manual_seed(2), **kw)
    info = {}
    masked = la.choose_move(mode, s, root, 1, N_ACTIONS,
                            imag_generator=torch.Generator().manual_seed(2),
                            visited=everything, info=info, **kw)
    assert masked == plain and info.get("fallback") is True
