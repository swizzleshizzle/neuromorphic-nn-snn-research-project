# EXP-070 Look-ahead on Existing Networks: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a look-ahead move selector over the true cube simulator, a harness that re-evaluates the tracked EXP-053/062 checkpoints under it, and an aggregator that encodes EXP-070's gates and verdicts, so the experiment can be pre-flighted and launched on the laptop.

**Architecture:** A reusable module `src/neuromorphic/training/lookahead.py` holds the procedure (goal test, tree, scorers, an evaluator that mirrors `evaluate_states`). It never touches the BFS provider. An experiment folder `experiments/070_lookahead_existing/` holds the cell configs, the runner, a committed table of published solved counts, and the aggregator. Tasks 1 to 6 are implementer tasks on the VPS; Task 7 is a controller checklist for the laptop.

**Tech Stack:** Python 3.10 venv, torch (CPU), snnTorch via the existing `Brain`, pytest. No scipy.

**Spec:** `docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md` (section 2 and 3, including the 2026-10-06 amendment dropping P1). Read it before starting any task.

## Global Constraints

- Run Python only via `.venv/bin/python`, never a bare `python`.
- **Always pass an explicit Bash `timeout` (the default is 120 s). Run tests in the FOREGROUND and let them block; 600000 ms covers every command in this plan. Silence mid-run is buffered output, not a hang. Never run the whole suite from a task; run only the files named in the task.**
- No em-dashes anywhere in code, docs, or commit messages.
- Commit messages are plain. No `Co-Authored-By` trailer, no "Generated with" line.
- Stage explicit paths only. Never `git add -A` or `git add .`.
- Action-space width comes from `env.action_space.n` or an `n_actions` argument, never a literal 6.
- The BFS distance provider (`neuromorphic.envs.cube_distance`) may be used to build held-out shells and in tests, and must NEVER be imported or read by `lookahead.py`.
- Never write an assertion that cannot fail. Each test in this plan names the bug it catches; keep that in its docstring.
- Arms are exactly: `G0, E1, E2, E3, P2, P3, R1, R2, R3`. Depths `7, 8, 9`. Seeds `0..11`. P1 does not exist (spec amendment 2026-10-06).

---

## File Structure

| file | responsibility |
|---|---|
| `src/neuromorphic/training/lookahead.py` (create) | goal test, tree levels, sequence scoring, move choice, `evaluate_lookahead` |
| `tests/training/test_lookahead.py` (create) | unit tests for the procedure, including the Gate 0 code-path test |
| `experiments/070_lookahead_existing/cells.py` (create) | per-(depth, seed) published config, checkpoint paths, held-out states, agent and head loading |
| `experiments/070_lookahead_existing/published_counts.json` (create, committed) | the 36 published solved counts, extracted once from the local untracked records |
| `experiments/070_lookahead_existing/extract_published.py` (create) | writes `published_counts.json` from the local records |
| `experiments/070_lookahead_existing/run.py` (create) | runs cells in a process pool, one JSON record per cell, `--skip-existing` |
| `experiments/070_lookahead_existing/aggregate.py` (create) | Gate 0, Gate 1, Claim 1 to 3, verdict functions |
| `tests/experiments/test_exp070_cells.py` (create) | configs match the published records; checkpoints exist |
| `tests/experiments/test_exp070_aggregate.py` (create) | verdict bands, gates, permutation test |

---

### Task 1: The pure procedure: goal test, tree, and sequence scores

**Files:**
- Create: `src/neuromorphic/training/lookahead.py`
- Test: `tests/training/test_lookahead.py`

**Interfaces:**
- Produces:
  - `solving_prefix(state, k, n_actions, *, apply_fn=apply_move, goal_fn=is_solved) -> tuple[int, ...] | None`: the shortest move sequence of length 1..k that reaches the goal, earliest in `itertools.product` order among equals; `None` if none.
  - `tree_levels(state, k, n_actions, *, apply_fn=apply_move) -> list[list]`: `levels[0] == [state]`, `levels[l]` holds `n_actions**l` states in `itertools.product(range(n_actions), repeat=l)` order.
  - `sequence_scores(level_logp: list[torch.Tensor], k: int, n_actions: int) -> torch.Tensor`: `level_logp[l]` has shape `[n_actions**l, n_actions]`; returns shape `[n_actions**k]`, where entry `j` is the summed log-probability of the length-k sequence with product index `j`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_lookahead.py -q` (Bash timeout 300000)
Expected: FAIL / ERROR, `ModuleNotFoundError` or `AttributeError` for `neuromorphic.training.lookahead`.

- [ ] **Step 3: Write the minimal implementation**

```python
"""Look-ahead move selection over the true simulator (EXP-070).

The PROCEDURE lives here: enumerate move sequences with the pure simulator, take a solve if one is
within reach, otherwise score the sequences and play the first move of the best. It is general:
nothing here knows the cube has 6 moves, and nothing here may read the BFS distance table, which
is an instrument and never an input (spec section 2.3). A test enforces the second rule at the
source level.
"""

from __future__ import annotations

import itertools

import torch

from neuromorphic.envs.cube import apply_move, is_solved


def solving_prefix(state, k, n_actions, *, apply_fn=apply_move, goal_fn=is_solved):
    """The shortest sequence of length 1..k that reaches the goal, or None.

    Lengths are tried in increasing order and sequences in `itertools.product` order, so the
    result is deterministic and ties go to the lowest action indices.
    """
    for length in range(1, k + 1):
        for seq in itertools.product(range(n_actions), repeat=length):
            s = state
            for a in seq:
                s = apply_fn(s, a)
            if goal_fn(s):
                return seq
    return None


def tree_levels(state, k, n_actions, *, apply_fn=apply_move):
    """levels[l] = the states every length-l sequence reaches, in product order."""
    levels = [[state]]
    for _ in range(k):
        levels.append([apply_fn(s, a) for s in levels[-1] for a in range(n_actions)])
    return levels


def sequence_scores(level_logp, k, n_actions):
    """Summed log-probability of every length-k sequence, indexed in product order.

    For sequence index j, the move at level l is (j // n**(k-l-1)) % n and the state it is taken
    from is level l's index j // n**(k-l).
    """
    j = torch.arange(n_actions ** k)
    scores = torch.zeros(n_actions ** k, dtype=level_logp[0].dtype)
    for l in range(k):
        parent = j // n_actions ** (k - l)
        move = (j // n_actions ** (k - l - 1)) % n_actions
        scores = scores + level_logp[l][parent, move]
    return scores
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/training/test_lookahead.py -q` (Bash timeout 300000)
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/lookahead.py tests/training/test_lookahead.py
git commit -m "EXP-070: the look-ahead procedure's pure core, goal test, tree and sequence scores"
```

---

### Task 2: Move choice with the network, and the two-stream rule

**Files:**
- Modify: `src/neuromorphic/training/lookahead.py`
- Test: `tests/training/test_lookahead.py`

**Interfaces:**
- Consumes: Task 1's `solving_prefix`, `tree_levels`, `sequence_scores`.
- Produces:
  - `MODES = ("G", "E", "P", "R")`
  - `imagined_logp(agent, head, states) -> torch.Tensor`, shape `[len(states), n_actions]`, called with a generator via keyword `generator`. Encodes the states in ONE batched `agent.step(..., recall=False)` call under `torch.no_grad()`.
  - `choose_move(mode, state, root_logits, k, n_actions, *, agent=None, head=None, imag_generator=None) -> tuple[int, bool]`: returns `(action, goal_fired)`.
  - `imag_seed_for(base: int, state_index: int, step: int) -> int`

**The two-stream rule (spec 2.3):** the REAL state's logits come from the evaluation generator, drawn by the caller exactly once per real move in every mode. Imagined states (P) and random scores (R) draw only from `imag_generator`, which the caller seeds fresh per (state, step). `choose_move` never receives the evaluation generator.

- [ ] **Step 1: Write the failing tests** (append to `tests/training/test_lookahead.py`)

```python
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
    """A depth-6 state: no solve within k <= 3, so the goal test never fires."""
    return _state_from([0, 2, 4, 1, 3, 5])


def test_imagined_logp_for_one_state_matches_the_evaluation_path():
    """Catches the batched path and the evaluation path disagreeing (a different readout,
    recall left on, or a different concept reduction). Same seed, same state, same numbers."""
    agent, head = _agent_and_head()
    s = _far_state()
    got = la.imagined_logp(agent, head, [s], generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        _, logits = action_distribution(agent, head, np.array(s),
                                        generator=torch.Generator().manual_seed(7))
    # 1e-6, not 0: a [1, 64] matmul and a [64] matvec may differ in the last bit. recall=True
    # or a different reduction moves these by orders of magnitude more.
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


def test_p_with_k2_does_not_advance_the_evaluation_generator():
    """Catches imagined states drawing from the real stream, which would shift every later
    real encoding and confound P against E."""
    agent, head = _agent_and_head()
    eval_gen = torch.Generator().manual_seed(11)
    s = _far_state()
    with torch.no_grad():
        _, root = action_distribution(agent, head, np.array(s), generator=eval_gen)
    before = eval_gen.get_state().clone()
    la.choose_move("P", s, root, 2, N_ACTIONS, agent=agent, head=head,
                   imag_generator=torch.Generator().manual_seed(99))
    assert torch.equal(eval_gen.get_state(), before)


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_lookahead.py -q` (Bash timeout 600000)
Expected: the Task 2 tests FAIL with `AttributeError` (`imagined_logp`, `choose_move`); Task 1 tests still pass.

- [ ] **Step 3: Write the implementation** (append to `lookahead.py`)

```python
import numpy as np

MODES = ("G", "E", "P", "R")


def imag_seed_for(base: int, state_index: int, step: int) -> int:
    """A distinct, deterministic seed per (cell, held-out state, real step)."""
    return (base * 1_000_003 + state_index) * 1_009 + step


def imagined_logp(agent, head, states, *, generator):
    """Log-probabilities over moves for many states, in ONE batched brain call.

    Must read exactly what `action_distribution` reads for a single state: the sensory concept's
    mean rate over the window, with recall off. A test pins this against the evaluation path.
    """
    with torch.no_grad():
        out = agent.step(np.array(states), recall=False, generator=generator)
        features = out["concept"].mean(dim=0)  # [B, content]
        return torch.log_softmax(head(features), dim=-1)


def choose_move(mode, state, root_logits, k, n_actions, *, agent=None, head=None,
                imag_generator=None):
    """One real move. Returns (action, goal_fired). Never sees the evaluation generator."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    greedy = int(root_logits.argmax())
    if mode == "G":
        return greedy, False
    if k < 1:
        raise ValueError(f"mode {mode} needs k >= 1, got {k}")
    prefix = solving_prefix(state, k, n_actions)
    if prefix is not None:
        return prefix[0], True
    if mode == "E":
        return greedy, False
    if mode == "R":
        scores = torch.rand(n_actions ** k, generator=imag_generator)
    else:  # "P"
        levels = tree_levels(state, k, n_actions)
        level_logp = [torch.log_softmax(root_logits.reshape(1, -1), dim=-1)]
        for l in range(1, k):
            level_logp.append(imagined_logp(agent, head, levels[l], generator=imag_generator))
        scores = sequence_scores(level_logp, k, n_actions)
    best = int(scores.argmax())  # first maximum, so ties go to the lowest product index
    return best // n_actions ** (k - 1), False
```

Check before running: `agent.T` (used in the batch test) and `agent.content` are the attribute names on `Brain`; confirm with `rg -n "self\.T\b|self\.content\b" src/neuromorphic/brain.py`. If either differs, use the real name in the tests.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/training/test_lookahead.py -q` (Bash timeout 600000)
Expected: PASS, all tests.

- [ ] **Step 5: Mutation check (required, the CLAUDE.md test-strength rule)**

Apply each mutation, run the file, confirm the named test FAILS, then revert with `git checkout -- src/neuromorphic/training/lookahead.py` and confirm `git diff --stat` is empty before the next one:

| mutation | test that must fail |
|---|---|
| `return best // n_actions ** (k - 1)` becomes `return best % n_actions` | `test_p_picks_the_first_move_of_the_best_scored_sequence` |
| in `imagined_logp`, `recall=False` becomes `recall=True` | `test_imagined_logp_for_one_state_matches_the_evaluation_path` |
| in `choose_move` R branch, `torch.rand(...)` becomes `torch.zeros(n_actions ** k)` | `test_r_choice_varies_with_the_imagined_seed` |
| `if mode == "G": return greedy, False` moved below the goal test | `test_g_ignores_the_goal_test` |
| `range(1, k + 1)` in `solving_prefix` becomes `range(1, k)` | both `solving_prefix` depth tests |

If any mutation survives, strengthen that test until it fails, then re-run the table. Record the table with outcomes in the commit message body.

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/training/lookahead.py tests/training/test_lookahead.py
git commit -m "EXP-070: move choice for G, E, P and R, with imagined states on their own stream"
```

---

### Task 3: `evaluate_lookahead`, mirroring `evaluate_states` exactly at G

**Files:**
- Modify: `src/neuromorphic/training/lookahead.py`
- Test: `tests/training/test_lookahead.py`

**Interfaces:**
- Consumes: Task 2's `choose_move`, `imag_seed_for`, `MODES`.
- Produces: `evaluate_lookahead(agent, head, states, *, depth, mode, k, generator, rng_seed=0, imag_seed=0, trace=None) -> dict`. Returns every key `evaluate_states` returns (`success_rate, mean_steps, optimality, n, eval_revisit_rate, greedy_modal_action_frac`) plus `solved` (int), `goal_fired_frac` (float), `mode` (str), `k` (int). If `trace` is a list, appends one list per state of `(action, goal_fired, root_logits)` tuples, where `root_logits` is a Python list of floats.

- [ ] **Step 1: Write the failing tests** (append)

```python
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
    assert compared_after_first >= 3  # the test must reach past the first move somewhere


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
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_lookahead.py -q -k "mode_g or matches_g or real_logits or bfs_provider or deterministic"` (Bash timeout 600000)
Expected: FAIL with `AttributeError: ... evaluate_lookahead`.

- [ ] **Step 3: Write the implementation** (append)

```python
from neuromorphic.envs.cube import CubeEnv
from neuromorphic.training.cube_baseline import max_steps_for, modal_action_fraction
from neuromorphic.training.reinforce import action_distribution


def evaluate_lookahead(agent, head, states, *, depth, mode, k, generator, rng_seed=0,
                       imag_seed=0, trace=None):
    """`evaluate_states` with a look-ahead move rule. At mode G, k 0 it IS `evaluate_states`.

    Every mode draws the real state's logits from `generator` exactly once per real move, the
    same call `greedy_action` makes. That is what keeps G's stream, and therefore Gate 0, intact,
    and what makes E and P see bit-identical real encodings while they play the same moves.
    """
    if mode == "G" and k != 0:
        raise ValueError("mode G is the published evaluator and takes k=0")
    limit = max_steps_for(depth)
    env = CubeEnv(scramble_depth=depth, max_steps=limit, scramble_seed=rng_seed)
    n_actions = env.action_space.n
    solved = 0
    steps_solved: list[int] = []
    eval_revisits = 0
    eval_steps = 0
    fired_moves = 0
    modal_fracs: list[float] = []
    for i, state in enumerate(states):
        obs, _ = env.reset(options={"state": state})
        visited = [env._state]
        actions: list[int] = []
        state_trace: list[tuple[int, bool, list[float]]] = []
        for t in range(1, limit + 1):
            with torch.no_grad():
                _, logits = action_distribution(agent, head, obs, generator=generator)
            imag = torch.Generator().manual_seed(imag_seed_for(imag_seed, i, t))
            action, fired = choose_move(mode, env._state, logits, k, n_actions,
                                        agent=agent, head=head, imag_generator=imag)
            fired_moves += int(fired)
            actions.append(int(action))
            state_trace.append((int(action), bool(fired), logits.reshape(-1).tolist()))
            obs, _, terminated, truncated, _ = env.step(action)
            visited.append(env._state)
            eval_steps += 1
            if terminated:
                solved += 1
                steps_solved.append(t)
                break
            if truncated:
                break
        eval_revisits += len(visited) - len(set(visited))
        modal_fracs.append(modal_action_fraction(actions))
        if trace is not None:
            trace.append(state_trace)
    n = len(states)
    total_steps = sum(steps_solved)
    return {
        "success_rate": solved / n if n else 0.0,
        "mean_steps": total_steps / len(steps_solved) if steps_solved else 0.0,
        "optimality": (depth * len(steps_solved) / total_steps) if total_steps else 0.0,
        "n": n,
        "eval_revisit_rate": (eval_revisits / eval_steps) if eval_steps else 0.0,
        "greedy_modal_action_frac": (sum(modal_fracs) / len(modal_fracs)) if modal_fracs else 0.0,
        "solved": solved,
        "goal_fired_frac": (fired_moves / eval_steps) if eval_steps else 0.0,
        "mode": mode,
        "k": k,
    }
```

Before running, open `evaluate_states` (`src/neuromorphic/training/cube_baseline.py`, around line 914) and confirm the body above matches it line for line in every place that affects a returned value. Any difference you find in `evaluate_states` wins; copy it.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/training/test_lookahead.py -q` (Bash timeout 600000)
Expected: PASS, all tests in the file.

- [ ] **Step 5: Mutation check**

| mutation | test that must fail |
|---|---|
| in `evaluate_lookahead`, draw the root logits only when `mode == "G"` (else use zeros) | `test_e_matches_g_move_for_move_until_the_goal_test_first_fires` |
| `limit = max_steps_for(depth)` becomes `limit = max_steps_for(depth) + 1` | `test_mode_g_reproduces_evaluate_states_exactly` |
| `imag = torch.Generator().manual_seed(...)` becomes `imag = generator` | `test_p_real_logits_match_g_while_their_moves_agree` |
| `imag_seed_for` returns `torch.seed()` | `test_rollouts_are_deterministic` |

Revert after each with `git checkout -- src/neuromorphic/training/lookahead.py` and confirm `git diff --stat` is clean. Record outcomes in the commit body.

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/training/lookahead.py tests/training/test_lookahead.py
git commit -m "EXP-070: evaluate_lookahead, identical to evaluate_states at G"
```

---

### Task 4: Cells: published configs, checkpoints, and the published counts table

**Files:**
- Create: `experiments/070_lookahead_existing/cells.py`
- Create: `experiments/070_lookahead_existing/extract_published.py`
- Create: `experiments/070_lookahead_existing/published_counts.json` (generated, committed)
- Create: `experiments/070_lookahead_existing/outputs/.gitkeep`
- Test: `tests/experiments/test_exp070_cells.py`

**Interfaces:**
- Produces (in `cells.py`):
  - `DEPTHS = (7, 8, 9)`, `SEEDS = tuple(range(12))`
  - `ARMS = (("G", 0), ("E", 1), ("E", 2), ("E", 3), ("P", 2), ("P", 3), ("R", 1), ("R", 2), ("R", 3))`
  - `published_config(depth: int, seed: int) -> CubeConfig`
  - `published_record_path(depth: int, seed: int) -> Path`
  - `head_path(depth: int, seed: int) -> Path`
  - `load_cell(depth: int, seed: int) -> tuple[agent, head, list[state], int]` returning `(agent, head, eval_states, train_seed)`
  - `cell_record_name(mode: str, k: int, depth: int, seed: int) -> str` = `f"exp070_{mode}{k}_d{depth}_s{seed}.json"`

**Background:** the depth-7 checkpoints are EXP-053 arm B (`tag="exp053_critic_d7"`, in `experiments/053_neuromod_stage3/outputs/`); depths 8 and 9 are EXP-062 (`tag="exp062_frontier_d{depth}"`, in `experiments/062_depth_frontier/outputs/`). Both are EXP-053 arm B's config field for field; see `sweep_configs` in `experiments/062_depth_frontier/run.py`. Heads are tracked. The published JSON records are NOT tracked (`.gitignore` excludes `experiments/*/outputs/*`) but exist on this VPS; `extract_published.py` copies the 36 solved counts into a committed table so Gate 0(b) has a tracked reference.

- [ ] **Step 1: Write the failing tests**

```python
"""EXP-070 cells: each re-evaluated cell must be the published cell, not a lookalike."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "070_lookahead_existing"
_spec = importlib.util.spec_from_file_location("exp070_cells", EXP / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)

COMPARED = ("seed", "depth", "tag", "arm", "sigma", "content", "n_actions", "max_depth",
            "heldout_cap", "heldout_frac", "encoder_seed", "train_seed", "split_seed")


@pytest.mark.parametrize("depth", cells.DEPTHS)
@pytest.mark.parametrize("seed", cells.SEEDS)
def test_every_head_and_encoder_is_tracked_and_present(depth, seed):
    """Catches a config that names a checkpoint nobody committed."""
    cfg = cells.published_config(depth, seed)
    assert cells.head_path(depth, seed).exists()
    assert (REPO / cfg.encoder_state_path).exists()


@pytest.mark.parametrize("depth", cells.DEPTHS)
@pytest.mark.parametrize("seed", (0, 11))
def test_config_matches_the_published_record_field_for_field(depth, seed):
    """Catches a re-evaluation of a different split or encoder than the one published. Needs
    the local records; skips cleanly on a checkout that lacks them."""
    path = cells.published_record_path(depth, seed)
    if not path.exists():
        pytest.skip(f"untracked published record not present: {path.name}")
    published = json.loads(path.read_text())["config"]
    cfg = cells.published_config(depth, seed)
    for key in COMPARED:
        assert getattr(cfg, key) == published[key], key
    assert Path(cfg.encoder_state_path).name == Path(published["encoder_state_path"].replace("\\", "/")).name


def test_published_counts_table_is_complete_and_matches_results_md():
    """Catches a truncated or mis-keyed table, and pins depths 8 and 9 to the per-seed lines
    EXP-062's RESULTS.md published (lines 75 and 87)."""
    table = json.loads((EXP / "published_counts.json").read_text())
    assert sorted(table) == sorted(f"d{d}_s{s}" for d in cells.DEPTHS for s in cells.SEEDS)
    d8 = [0.115, 0.085, 0.015, 0.110, 0.110, 0.060, 0.035, 0.135, 0.130, 0.055, 0.055, 0.035]
    d9 = [0.000, 0.000, 0.000, 0.080, 0.015, 0.005, 0.020, 0.025, 0.045, 0.000, 0.000, 0.005]
    for s in cells.SEEDS:
        assert table[f"d8_s{s}"] == {"solved": round(d8[s] * 200), "n": 200}
        assert table[f"d9_s{s}"] == {"solved": round(d9[s] * 200), "n": 200}


def test_cell_record_names_are_unique():
    """Catches records silently overwriting each other."""
    names = [cells.cell_record_name(m, k, d, s)
             for (m, k) in cells.ARMS for d in cells.DEPTHS for s in cells.SEEDS]
    assert len(names) == len(set(names)) == 9 * 3 * 12


def test_there_is_no_p1_arm():
    """Spec amendment 2026-10-06: P1 is E1 by construction and must not run."""
    assert ("P", 1) not in cells.ARMS
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/experiments/test_exp070_cells.py -q` (Bash timeout 300000)
Expected: ERROR, `cells.py` not found.

- [ ] **Step 3: Write `cells.py`**

```python
"""EXP-070 cells: the published EXP-053 arm B (depth 7) and EXP-062 (depths 8, 9) checkpoints.

Spec: docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md, section 2.2.
"""

from __future__ import annotations

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
    return REPO / PUBLISHED_DIR[depth] / record_filename(published_config(depth, seed))


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
```

`make_agent` loads `cfg.encoder_state_path` relative to the working directory, so `run.py` and the tests must be run from the repo root (they are, in every command here).

- [ ] **Step 4: Write `extract_published.py` and generate the table**

```python
"""Copy the 36 published solved counts into a COMMITTED table.

The published JSON records are gitignored, so without this Gate 0(b) would depend on files that
exist only on the machines that ran EXP-053 and EXP-062. Run once, from the repo root.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp070_cells", HERE / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)


def main() -> None:
    table = {}
    for d in cells.DEPTHS:
        for s in cells.SEEDS:
            rec = json.loads(cells.published_record_path(d, s).read_text())
            n = int(rec["n"])
            table[f"d{d}_s{s}"] = {"solved": round(float(rec["success_rate"]) * n), "n": n}
    (HERE / "published_counts.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(table)} cells")


if __name__ == "__main__":
    main()
```

Run: `.venv/bin/python experiments/070_lookahead_existing/extract_published.py` (Bash timeout 120000)
Expected: `wrote 36 cells`. Then `mkdir -p experiments/070_lookahead_existing/outputs && touch experiments/070_lookahead_existing/outputs/.gitkeep`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/experiments/test_exp070_cells.py -q` (Bash timeout 300000)
Expected: PASS (no skips on this VPS, where the records exist).

- [ ] **Step 6: Smoke one real cell (read the real output, do not trust green tests)**

Run from the repo root (Bash timeout 600000):

```bash
.venv/bin/python - <<'EOF'
import importlib.util, json, torch
from pathlib import Path
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("c", "experiments/070_lookahead_existing/cells.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
from neuromorphic.training.lookahead import evaluate_lookahead
agent, head, states, ts = c.load_cell(8, 0)
r = evaluate_lookahead(agent, head, states[:20], depth=8, mode="G", k=0,
                       generator=torch.Generator().manual_seed(ts), rng_seed=ts)
print(len(states), r)
EOF
```

Expected: `200` held-out states and a sane record (success between 0 and 1, `n` 20). Paste the printed line into the task report. This is NOT Gate 0; that is the laptop pre-flight over all 200 states.

- [ ] **Step 7: Commit**

```bash
git add experiments/070_lookahead_existing/cells.py experiments/070_lookahead_existing/extract_published.py experiments/070_lookahead_existing/published_counts.json experiments/070_lookahead_existing/outputs/.gitkeep tests/experiments/test_exp070_cells.py
git commit -m "EXP-070: cells, and a committed table of the 36 published solved counts"
```

---

### Task 5: The runner

**Files:**
- Create: `experiments/070_lookahead_existing/run.py`
- Test: `tests/experiments/test_exp070_cells.py` (append)

**Interfaces:**
- Consumes: `cells.ARMS, DEPTHS, SEEDS, load_cell, cell_record_name`; `lookahead.evaluate_lookahead`.
- Produces: `run_cell(mode, k, depth, seed, out_dir: Path) -> dict` writing one JSON record per cell with every `evaluate_lookahead` key plus `depth, seed, wall_s, git_commit, head_file, encoder_file`; CLI flags `--arms` (e.g. `G0 E1 P3`), `--depths`, `--seeds`, `--workers`, `--out-dir`, `--skip-existing`, `--limit-states N` (smoke only; records it in the record as `limit_states`).

- [ ] **Step 1: Write the failing test** (append)

```python
_rspec = importlib.util.spec_from_file_location("exp070_run", EXP / "run.py")
run = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(run)


def test_run_cell_writes_a_complete_record(tmp_path):
    """Catches a record missing the fields the aggregator and Gate 0 read."""
    rec = run.run_cell("E", 1, 8, 0, tmp_path, limit_states=3)
    on_disk = json.loads((tmp_path / cells.cell_record_name("E", 1, 8, 0)).read_text())
    assert on_disk == rec
    for key in ("solved", "n", "success_rate", "goal_fired_frac", "eval_revisit_rate",
                "mode", "k", "depth", "seed", "wall_s", "git_commit", "limit_states"):
        assert key in rec, key
    assert rec["n"] == 3 and rec["limit_states"] == 3


def test_parse_arm_rejects_p1():
    """Catches P1 sneaking back in through the CLI."""
    with pytest.raises(SystemExit):
        run.parse_arm("P1")
    assert run.parse_arm("P3") == ("P", 3)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/experiments/test_exp070_cells.py -q -k "run_cell or parse_arm"` (Bash timeout 300000)
Expected: ERROR, `run.py` not found.

- [ ] **Step 3: Write `run.py`**

```python
"""EXP-070: look-ahead on the networks we already have. Re-evaluation only, nothing trains.

Spec: docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md

Usage (from the repo root):
    .venv/bin/python -u experiments/070_lookahead_existing/run.py --arms G0 E1 E2 E3 R1 R2 R3 --workers 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp070_cells", HERE / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)

from neuromorphic.training.lookahead import evaluate_lookahead  # noqa: E402

torch.set_num_threads(1)


def parse_arm(text: str) -> tuple[str, int]:
    arm = (text[0], int(text[1:]))
    if arm not in cells.ARMS:
        raise SystemExit(f"unknown arm {text!r}; valid: {[m + str(k) for m, k in cells.ARMS]}")
    return arm


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_cell(mode, k, depth, seed, out_dir: Path, limit_states=None) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    agent, head, states, train_seed = cells.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode=mode, k=k,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed)
    cfg = cells.published_config(depth, seed)
    rec = {**res, "depth": depth, "seed": seed, "wall_s": round(time.time() - t0, 1),
           "git_commit": _git_commit(), "limit_states": limit_states,
           "head_file": str(cells.head_path(depth, seed).relative_to(cells.REPO)),
           "encoder_file": cfg.encoder_state_path}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cells.cell_record_name(mode, k, depth, seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=[m + str(k) for m, k in cells.ARMS])
    ap.add_argument("--depths", type=int, nargs="+", default=list(cells.DEPTHS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit-states", type=int, default=None)
    args = ap.parse_args()

    arms = [parse_arm(a) for a in args.arms]
    jobs = [(m, k, d, s) for (m, k) in arms for d in args.depths for s in args.seeds]
    if args.skip_existing:
        jobs = [j for j in jobs if not (args.out_dir / cells.cell_record_name(*j)).exists()]
    print(f"EXP-070: {len(jobs)} cells, {args.workers} workers, arms {args.arms}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_cell, *j, args.out_dir, args.limit_states): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  {r['mode']}{r['k']} d{r['depth']} s{r['seed']}  "
                  f"solved {r['solved']}/{r['n']}  {r['wall_s']} s", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/experiments/test_exp070_cells.py -q` (Bash timeout 600000)
Expected: PASS.

- [ ] **Step 5: Measure one P3 move cost on the VPS** (informs the laptop budget; not the calibration wave)

Run: `.venv/bin/python -u experiments/070_lookahead_existing/run.py --arms P3 --depths 8 --seeds 0 --limit-states 5 --workers 1 --out-dir /root/scratch/exp070-smoke` (Bash timeout 600000)
Expected: one line with `wall_s`. Report it with the number of real moves (5 states x up to 19 moves). The scratch directory is outside the repo on purpose.

- [ ] **Step 6: Commit**

```bash
git add experiments/070_lookahead_existing/run.py tests/experiments/test_exp070_cells.py
git commit -m "EXP-070: runner, one record per cell"
```

---

### Task 6: The aggregator: gates and verdicts encoded as code

**Files:**
- Create: `experiments/070_lookahead_existing/aggregate.py`
- Test: `tests/experiments/test_exp070_aggregate.py`

**Interfaces:**
- Produces (pure functions, unit-tested):
  - `GATE0_FORM: str | None = None` (set to `"exact"` or `"wilson"` ONLY in the dated spec-amendment commit after the pre-flight)
  - `ALPHA = 0.05`, `FLOOR = 0.02`, `CEILING = 0.98`, `CLAIM2_BAR = 0.10`
  - `one_sided_p(diffs: list[float]) -> float`: exact over all `2**n` sign flips, P(sum >= observed).
  - `gate1_verdict(e_mean: float, p_mean: float) -> str`: `"UNRESOLVED"` if both < FLOOR or both > CEILING, else `"RESOLVED"`.
  - `claim1_verdict(diffs: list[float], e_mean: float, p_mean: float) -> tuple[str, float]`: `UNRESOLVED` (gate 1 first), `REFUTED` (mean <= 0), `CONFIRMED` (p < ALPHA), else `NOT SIGNIFICANT`; returns `(verdict, p)`.
  - `wilson95(successes: int, n: int) -> tuple[float, float]`
  - `gate0_verdict(form, g_counts: dict, published: dict, determinism_ok: bool) -> str`: `"PASS"`/`"FAIL"`; `form=None` raises `SystemExit` naming the pre-flight.
  - `main()` prints per-depth tables (G, E1-3, P2-3, R1-3 means), Gate 0, Gate 1 per (depth, k), Claim 1 primary (d8 k3) and secondaries, Claim 2, revisit rates. Refuses to print any claim verdict if Gate 0 is not PASS.

- [ ] **Step 1: Write the failing tests**

```python
"""EXP-070 aggregator. The gates and the unresolved band are verdicts, not warnings (EXP-068)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "exp070_aggregate",
    Path(__file__).resolve().parents[2] / "experiments" / "070_lookahead_existing" / "aggregate.py",
)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def test_one_sided_p_all_positive_twelve_is_one_over_4096():
    """Catches a two-sided test or a sampled approximation: 12 equal positive diffs reach the
    observed sum only under the identity flip."""
    assert agg.one_sided_p([0.1] * 12) == pytest.approx(1 / 4096)


def test_one_sided_p_of_zeros_is_one():
    """Catches a strict > comparison, which would call a null effect significant."""
    assert agg.one_sided_p([0.0] * 12) == 1.0


@pytest.mark.parametrize("e,p,expected", [
    (0.0, 0.0, "UNRESOLVED"), (0.019, 0.0199, "UNRESOLVED"),
    (0.019, 0.02, "RESOLVED"), (0.99, 0.985, "UNRESOLVED"), (0.98, 0.99, "RESOLVED"),
    (0.0, 0.3, "RESOLVED"),
])
def test_gate1_bands_including_edges(e, p, expected):
    """Catches floor-on-floor being read as a result (EXP-064) and pins edge inclusivity."""
    assert agg.gate1_verdict(e, p) == expected


def test_claim1_floor_on_floor_is_unresolved_even_with_a_positive_mean():
    """Catches the gate being applied after the significance test instead of before."""
    v, _ = agg.claim1_verdict([0.005] * 12, e_mean=0.001, p_mean=0.006)
    assert v == "UNRESOLVED"


def test_claim1_verdicts():
    """Catches a mis-ordered verdict ladder."""
    assert agg.claim1_verdict([0.05] * 12, 0.1, 0.15)[0] == "CONFIRMED"
    assert agg.claim1_verdict([-0.01] * 12, 0.1, 0.09)[0] == "REFUTED"
    assert agg.claim1_verdict([0.0] * 12, 0.1, 0.1)[0] == "REFUTED"
    mixed = [0.02, -0.02] * 6
    mixed[0] = 0.03
    assert agg.claim1_verdict(mixed, 0.1, 0.1008)[0] == "NOT SIGNIFICANT"


def test_wilson_interval_contains_the_point_and_is_inside_unit():
    """Catches a swapped or unclamped interval."""
    lo, hi = agg.wilson95(10, 200)
    assert 0.0 <= lo < 0.05 < hi <= 1.0


def test_gate0_refuses_to_run_before_the_preflight_fixes_its_form():
    """Catches the aggregator choosing Gate 0's form after seeing P numbers."""
    with pytest.raises(SystemExit):
        agg.gate0_verdict(None, {}, {}, True)


def test_gate0_exact_fails_on_one_count_off():
    """Catches a tolerance sneaking into the exact form."""
    pub = {"d8_s0": {"solved": 23, "n": 200}}
    assert agg.gate0_verdict("exact", {"d8_s0": 23}, pub, True) == "PASS"
    assert agg.gate0_verdict("exact", {"d8_s0": 22}, pub, True) == "FAIL"


def test_gate0_fails_without_determinism_in_either_form():
    """Gate 0(a) is required in both forms."""
    pub = {"d8_s0": {"solved": 23, "n": 200}}
    assert agg.gate0_verdict("exact", {"d8_s0": 23}, pub, False) == "FAIL"
    assert agg.gate0_verdict("wilson", {"d8_s0": 23}, pub, False) == "FAIL"


def test_gate0_wilson_is_per_depth_on_pooled_counts():
    """Catches the Wilson form being applied per seed (too strict) or across depths (too loose).
    Published pooled 120/2400 at depth 8; G pooled 118/2400 passes, 60/2400 fails."""
    pub = {f"d8_s{s}": {"solved": 10, "n": 200} for s in range(12)}
    near = {f"d8_s{s}": (10 if s else 8) for s in range(12)}
    far = {f"d8_s{s}": 5 for s in range(12)}
    assert agg.gate0_verdict("wilson", near, pub, True) == "PASS"
    assert agg.gate0_verdict("wilson", far, pub, True) == "FAIL"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/experiments/test_exp070_aggregate.py -q` (Bash timeout 300000)
Expected: ERROR, `aggregate.py` not found.

- [ ] **Step 3: Write `aggregate.py`**

```python
"""EXP-070 aggregator. Gates and the unresolved band are VERDICTS (the EXP-068 rule).

Spec: docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md, section 2.5.

Usage: .venv/bin/python experiments/070_lookahead_existing/aggregate.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Set ONLY in the dated spec-amendment commit after pre-flight step 1. None blocks every verdict.
GATE0_FORM: str | None = None
ALPHA = 0.05
FLOOR = 0.02
CEILING = 0.98
CLAIM2_BAR = 0.10
PRIMARY = (8, 3)
SECONDARY = ((8, 2), (9, 2), (9, 3))


def one_sided_p(diffs) -> float:
    """Exact one-sided paired sign-flip test: P(flipped sum >= observed sum)."""
    n, obs = len(diffs), sum(diffs)
    hits = sum(1 for s in itertools.product((1, -1), repeat=n)
               if sum(x * y for x, y in zip(s, diffs)) >= obs - 1e-12)
    return hits / 2 ** n


def gate1_verdict(e_mean: float, p_mean: float) -> str:
    if (e_mean < FLOOR and p_mean < FLOOR) or (e_mean > CEILING and p_mean > CEILING):
        return "UNRESOLVED"
    return "RESOLVED"


def claim1_verdict(diffs, e_mean: float, p_mean: float) -> tuple[str, float]:
    p = one_sided_p(diffs)
    if gate1_verdict(e_mean, p_mean) == "UNRESOLVED":
        return "UNRESOLVED", p
    if st.mean(diffs) <= 0:
        return "REFUTED", p
    if p < ALPHA:
        return "CONFIRMED", p
    return "NOT SIGNIFICANT", p


def wilson95(successes: int, n: int) -> tuple[float, float]:
    z = 1.959964
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def gate0_verdict(form, g_counts: dict, published: dict, determinism_ok: bool) -> str:
    if form is None:
        raise SystemExit("GATE0_FORM is unset. Run pre-flight step 1 and amend the spec first.")
    if not determinism_ok:
        return "FAIL"
    if form == "exact":
        return "PASS" if all(g_counts[c] == published[c]["solved"] for c in published) else "FAIL"
    if form == "wilson":
        by_depth = defaultdict(lambda: [0, 0, 0])
        for cell, pub in published.items():
            d = cell.split("_")[0]
            by_depth[d][0] += pub["solved"]
            by_depth[d][1] += pub["n"]
            by_depth[d][2] += g_counts[cell]
        for pub_solved, n, g_solved in by_depth.values():
            lo, hi = wilson95(pub_solved, n)
            if not lo <= g_solved / n <= hi:
                return "FAIL"
        return "PASS"
    raise SystemExit(f"unknown GATE0_FORM {form!r}")


def load(out_dir: Path) -> dict:
    recs = {}
    for p in sorted(out_dir.glob("exp070_*.json")):
        r = json.loads(p.read_text())
        if r.get("limit_states") is not None:
            continue  # smoke records never enter a verdict
        recs[(r["mode"], r["k"], r["depth"], r["seed"])] = r
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="set only after re-running a sample of cells and diffing records")
    args = ap.parse_args()
    recs = load(args.out_dir)
    published = json.loads((HERE / "published_counts.json").read_text())
    seeds = range(12)

    def rate(mode, k, d, s):
        return recs[(mode, k, d, s)]["success_rate"]

    print("EXP-070 mean held-out success")
    arms = [("G", 0), ("E", 1), ("E", 2), ("E", 3), ("P", 2), ("P", 3), ("R", 1), ("R", 2), ("R", 3)]
    for d in (7, 8, 9):
        row = []
        for m, k in arms:
            vals = [rate(m, k, d, s) for s in seeds if (m, k, d, s) in recs]
            row.append(f"{m}{k} {st.mean(vals):.4f} (n={len(vals)})" if vals else f"{m}{k} -")
        print(f"  d{d}: " + "  ".join(row))

    g_counts = {f"d{d}_s{s}": recs[("G", 0, d, s)]["solved"] for d in (7, 8, 9) for s in seeds}
    g0 = gate0_verdict(GATE0_FORM, g_counts, published, args.determinism_ok)
    print(f"\nGATE 0 ({GATE0_FORM}): {g0}")
    if g0 != "PASS":
        print("Gate 0 did not pass. No claim may be read.")
        return

    for label, cellset in (("PRIMARY", (PRIMARY,)), ("secondary", SECONDARY)):
        for d, k in cellset:
            diffs = [rate("P", k, d, s) - rate("E", k, d, s) for s in seeds]
            e_mean = st.mean(rate("E", k, d, s) for s in seeds)
            p_mean = st.mean(rate("P", k, d, s) for s in seeds)
            v, p = claim1_verdict(diffs, e_mean, p_mean)
            print(f"CLAIM 1 {label} d{d} k{k}: P {p_mean:.4f} vs E {e_mean:.4f}, "
                  f"diff {st.mean(diffs):+.4f}, p {p:.4f} -> {v}")

    p3_d9 = st.mean(rate("P", 3, 9, s) for s in seeds)
    print(f"CLAIM 2 d9 P3 {p3_d9:.4f} vs bar {CLAIM2_BAR}: "
          f"{'MET' if p3_d9 >= CLAIM2_BAR else 'NOT MET'} (practical only; read with Claim 1)")
    print("CLAIM 3 revisit rate (no threshold):")
    for d in (7, 8, 9):
        print(f"  d{d}: " + "  ".join(
            f"{m}{k} {st.mean(recs[(m, k, d, s)]['eval_revisit_rate'] for s in seeds):.3f}"
            for m, k in arms if all((m, k, d, s) in recs for s in seeds)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/experiments/test_exp070_aggregate.py -q` (Bash timeout 300000)
Expected: PASS.

- [ ] **Step 5: Mutation check**

| mutation | test that must fail |
|---|---|
| `>= obs - 1e-12` becomes `> obs + 1e-12` | `test_one_sided_p_of_zeros_is_one` |
| `claim1_verdict` checks `p < ALPHA` before the Gate 1 check | `test_claim1_floor_on_floor_is_unresolved_even_with_a_positive_mean` |
| `and` in the FLOOR branch of `gate1_verdict` becomes `or` | `test_gate1_bands_including_edges` |
| Wilson form compares per cell instead of pooled per depth | `test_gate0_wilson_is_per_depth_on_pooled_counts` |
| `if form is None: raise` deleted | `test_gate0_refuses_to_run_before_the_preflight_fixes_its_form` |

Revert after each (`git checkout -- experiments/070_lookahead_existing/aggregate.py`), confirm `git diff --stat` is clean, and record outcomes in the commit body.

- [ ] **Step 6: Commit**

```bash
git add experiments/070_lookahead_existing/aggregate.py tests/experiments/test_exp070_aggregate.py
git commit -m "EXP-070: aggregator, gates and the unresolved band as verdicts"
```

---

### Task 7 (CONTROLLER, not an implementer task): pre-flight and launch on the laptop

Follow `docs/playbooks/remote-experiment-runs.md` for every laptop step (ssh via the `ssh laptop` alias only; drive PowerShell from a `.ps1` file; `python -u`; `Tee-Object` to a log; progress from the record count, not the log). Use `experiments/062_depth_frontier/launch062_wt.ps1` as the launcher template.

- [ ] **7.1** Merge the implementation branch, push, and update the laptop worktree with `scripts/wt-switch-branch.ps1` (never `git checkout -f`).
- [ ] **7.2** Confirm the laptop's tracked heads and E1 encoders match this VPS: `git ls-files -s` hashes for the 36 heads and 12 E1 encoders.
- [ ] **7.3 Pre-flight step 1 (Gate 0 form):** `run.py --arms G0 --workers 20`. Compare each cell's `solved` to `published_counts.json`. Re-run 3 cells into a second directory and diff the records byte for byte (Gate 0(a)).
- [ ] **7.4 Pre-flight step 2 (Gate 1 calibration):** `run.py --arms E1 E2 E3 R1 R2 R3 --workers 20`. Print E and R means per (depth, k).
- [ ] **7.5 Pre-flight step 3 (calibration wave):** `run.py --arms P2 P3 --depths 8 --seeds 0 --workers 2`. Record wall-clock per cell.
- [ ] **7.6 Amend the spec, dated, BEFORE any further P cell:** Gate 0 form (`exact` if all 36 counts matched, else `wilson`, with the differences listed); E and R levels per (depth, k); measured cost replacing the estimate. Set `GATE0_FORM` in `aggregate.py` in the same commit. Commit message names it as a pre-launch amendment.
- [ ] **7.7 Full launch:** `run.py --arms P2 P3 --workers 20 --skip-existing`. Bank per cell; a lost wave is re-run with `--skip-existing`.
- [ ] **7.8** Confirm completion from COUNTS (324 records, excluding smoke records), copy records back, run `aggregate.py --determinism-ok`, write `experiments/070_lookahead_existing/RESULTS.md` (provenance: seeds, date, machine, commit, regeneration command), and decide stage 2 per the roadmap.

**Note for stage 2, found while writing this plan:** EXP-053 arm B's critics (`exp053_critic_d7_*_critic.pt`, 12 seeds plus pilots) exist untracked on this VPS. Stage 2 at depth 7 may need no retraining. Track them (`git add -f` with a `.gitignore` negation, per the existing head/encoder precedent) before they are lost.
