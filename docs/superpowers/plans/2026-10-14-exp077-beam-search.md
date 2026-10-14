# EXP-077 Beam Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A width-64, depth-5 beam search (mode "B") driven by EXP-075's committed X judges, with its evaluation driver, aggregator and laptop launcher, so EXP-077 can pilot and run.

**Architecture:** `beam_move` and mode "B" go into `src/neuromorphic/training/lookahead.py` beside mode C; every existing mode stays byte-identical. `experiments/077_beam_search/` copies EXP-076's driver, aggregator and launcher pattern, importing EXP-075's judge loader and EXP-070/071/072's statistics through importlib exactly as EXP-076 does.

**Tech Stack:** Python 3.10 (VPS) / 3.13 (laptop), PyTorch, pytest; PowerShell 5.1 on the laptop.

**Spec:** `docs/superpowers/specs/2026-10-14-exp077-beam-search-design.md` (read sections 2 to 9 before any task).

## Global Constraints

- Run Python only via the venv: `.venv/bin/python` on the VPS. In a git worktree prefix every command with `PYTHONPATH=src` (the venv's editable install points at the main checkout).
- **Always pass an explicit Bash `timeout`; the default is 120 s.** Call test commands directly and let them block; a 600000 ms timeout covers every file named here. Buffered output means silence mid-run is normal, not a hang. Never background a test run.
- No em-dashes anywhere (code, comments, docs, commit messages).
- Plain commit messages: no `Co-Authored-By` trailer, no "Generated with" line. Stage explicit paths; never `git add -A`.
- Never write an assertion that cannot fail; a new test must fail against the pre-fix code.
- Distance-to-solved (BFS) is an instrument, never a model input: allowed in tests and in Gate R style yardsticks, never inside `beam_move`.
- Action count comes from `n_actions` / `head.head.out_features` / `env.action_space.n`, never a literal 6 in library code.
- Beam width 64, beam depth 5, goal check 5 for the new arms (spec sections 2 and 3). These are constants, not CLI choices.

## Review Focus

1. **A `visited` set that contains the root** (it always does in `evaluate_lookahead`): the beam must still expand the root's children. Pinned in Task 1's depth-1 equivalence test, which passes the real path including the root.
2. **Every neighbour of the root visited**: the fallback must re-run the beam unmasked and flag it, matching mode C at depth 1. Pinned in Task 1.
3. **Ties in judge value**: the earlier child (frontier order, then move order) must win, deterministically. Pinned in Task 1 with a constant critic.
4. **A smoke or pilot record reaching a verdict**, or a J4V-X record read from the wrong experiment: the aggregator must refuse. Pinned in Task 3.
5. **The launcher measuring the redirector instead of the worker** (EXP-076's 2 MB peak): pinned by Task 4's launcher text test, and checked for real in the pilot.

---

### Task 1: `beam_move` and mode "B" in the look-ahead library

**Files:**
- Modify: `src/neuromorphic/training/lookahead.py` (MODES, new `beam_move`, `choose_move`, `evaluate_lookahead`)
- Test: `tests/training/test_lookahead_beam.py` (new)

**Interfaces:**
- Produces: `beam_move(state, width, depth, n_actions, *, agent, critic, imag_generator, visited=None, info=None) -> int | None`
- Produces: `choose_move(..., beam_width=None)`; mode `"B"` uses `k` as the beam depth.
- Produces: `evaluate_lookahead(..., beam_width=None)`; records gain `"judge_evals"` (mode C and B only: mean judge-scored states per real move that reached the search; goal fires and fallbacks-without-search excluded) and, for mode B only, `"beam_width"`.

- [ ] **Step 1: Write the failing tests** in `tests/training/test_lookahead_beam.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_lookahead_beam.py -q` (timeout 600000)
Expected: FAIL (`AttributeError: ... has no attribute 'beam_move'` and the like).

- [ ] **Step 3: Implement** in `src/neuromorphic/training/lookahead.py`:

```python
MODES = ("G", "E", "P", "R", "C", "B")


def beam_move(state, width, depth, n_actions, *, agent, critic, imag_generator, visited=None,
              info=None):
    """EXP-077's beam: keep the `width` best states by judge value at each level, `depth` levels
    deep, and return the first move of the best final state (None if level 1 is empty).

    Each state is scored at most once, at the shallowest level the beam reaches it (`seen`), so
    no path returns to its own ancestors and a full-width beam's level l is the BFS shell at
    distance l. One batched judge call per level; ties go to the earlier child."""
    seen = {state}
    frontier = [(state, None)]
    evals = 0
    for _ in range(depth):
        kids = []
        for s, first in frontier:
            for a in range(n_actions):
                c = apply_move(s, a)
                if c in seen or (visited is not None and c in visited):
                    continue
                seen.add(c)
                kids.append((c, a if first is None else first))
        if not kids:
            break
        scores = imagined_values(agent, critic, [c for c, _ in kids],
                                 generator=imag_generator).reshape(-1).tolist()
        evals += len(kids)
        order = sorted(range(len(kids)), key=lambda i: -scores[i])
        frontier = [kids[i] for i in order[:width]]
    if info is not None:
        info["judge_evals"] = info.get("judge_evals", 0) + evals
    if frontier[0][1] is None:
        return None
    if info is not None:
        info["beam_leaf"] = frontier[0][0]
        info["beam_frontier"] = [s for s, _ in frontier]
    return frontier[0][1]
```

In `choose_move`, add the keyword `beam_width=None` and, right after `k_goal = _check_k_goal(k, k_goal)`:

```python
    if mode == "B":
        if beam_width is None or beam_width < 1:
            raise ValueError("mode B needs beam_width >= 1")
    elif beam_width is not None:
        raise ValueError(f"beam_width is only for mode B, got mode {mode!r}")
```

After the goal check (`prefix` block), before `if mode == "E":`:

```python
    if mode == "B":
        if critic is None:
            raise ValueError("mode B needs a critic")
        move = beam_move(state, beam_width, k, n_actions, agent=agent, critic=critic,
                         imag_generator=imag_generator, visited=visited, info=info)
        if move is None:
            # Every neighbour is on the real path: mode C's fallback, its unmasked best.
            if info is not None:
                info["fallback"] = True
            move = beam_move(state, beam_width, k, n_actions, agent=agent, critic=critic,
                             imag_generator=imag_generator, visited=None, info=info)
        return move, False
```

In the mode C branch, after `scores = imagined_values(...)`, add `if info is not None: info["judge_evals"] = n_actions ** k`.

Note the beam's `visited` excludes children only; the root is already in `seen`. With `width >= 6` at depth 1 the beam scores the same 6 children as mode C k 1, so the depth-1 test pins the fallback.

In `evaluate_lookahead`: add the keyword `beam_width=None`, pass `beam_width=beam_width` to `choose_move`, and accumulate per move:

```python
            if "judge_evals" in info:
                judge_evals_total += info["judge_evals"]
                searched_moves += 1
```

(initialise both to 0 beside `fallback_moves`). In the returned dict, add after building it:

```python
    if mode in ("C", "B"):
        out["judge_evals"] = judge_evals_total / searched_moves if searched_moves else 0.0
    if mode == "B":
        out["beam_width"] = beam_width
```

(rename the existing literal `return {...}` to `out = {...}` then `return out`).

- [ ] **Step 4: Run the new tests and the regression set**

Run, each with timeout 600000:
`.venv/bin/python -m pytest tests/training/test_lookahead_beam.py tests/training/test_lookahead_k_goal.py -q`
`.venv/bin/python -m pytest tests/training -q -m "not slow" -k "lookahead"`
`.venv/bin/python -m pytest tests/experiments/test_exp076.py tests/experiments/test_exp075.py -q -m "not slow"`
Expected: all PASS. If `test_full_width_beam_final_frontier_is_the_bfs_shell` fails, do not weaken it: report the failing root and the set difference.

- [ ] **Step 5: Mutation check.** Temporarily (a) drop the `c in seen` test, (b) reverse the sort, (c) remove the fallback re-run, one at a time; confirm at least one new test fails for each; restore the file and confirm `git diff` shows only the intended change. Record results in the report.

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/training/lookahead.py tests/training/test_lookahead_beam.py
git commit -m "Look-ahead: mode B, a beam over judge values that scores each state once at its shallowest level, and judge_evals per move (EXP-077)"
```

---

### Task 2: EXP-077 evaluation driver

**Files:**
- Create: `experiments/077_beam_search/evaluate.py`
- Test: `tests/experiments/test_exp077.py` (new; Task 3 appends to it)

**Interfaces:**
- Consumes: `evaluate_lookahead(..., mode, k, k_goal, beam_width)` from Task 1; EXP-075's `load_judge`, `JudgeCritic`, `load_cell`, `judge_updates` through importlib (as `experiments/076_deeper_tree/evaluate.py` does).
- Produces: `ARMS = {"J4V-X": ("C", 4, 4, None), "J4VG5-X": ("C", 4, 5, None), "B5-X": ("B", 5, 5, 64)}` (arm -> mode, k, k_goal, beam_width); `DEPTHS = (9, 11)`; `E76_OUT` (EXP-076's `outputs/`); `record_name(arm, depth, seed) -> "exp077_{arm}_d{depth}_s{seed}.json"`; `run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict` writing a record with every `evaluate_lookahead` key plus `depth, seed, arm, fired5, wall_s, git_commit, limit_states, judge_updates`.

- [ ] **Step 1: Write the failing tests** in `tests/experiments/test_exp077.py`:

```python
"""EXP-077: driver, aggregator and launcher. Each docstring names the bug it catches."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "experiments" / "077_beam_search"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("exp077_evaluate", HERE / "evaluate.py")


def test_arms_are_the_spec_table():
    """Catches an arm whose mode, depth, goal check or width drifted from spec section 3."""
    assert ev.ARMS == {"J4V-X": ("C", 4, 4, None), "J4VG5-X": ("C", 4, 5, None),
                       "B5-X": ("B", 5, 5, 64)}
    assert ev.DEPTHS == (9, 11)


def test_j4v_x_here_equals_exp076s_record_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH. Catches mode B's seam changing mode C: J4V-X here, on the first
    3 depth-9 states of seed 0, equals EXP-076's driver on the same slice in every outcome
    field and in fallback_frac."""
    e76 = _load("exp076_evaluate_t", REPO / "experiments" / "076_deeper_tree" / "evaluate.py")
    mine = ev.run_cell("J4V-X", 9, 0, tmp_path / "a", limit_states=3)
    ref = e76.run_cell("J4V-X", 9, 0, tmp_path / "b", limit_states=3)
    for f in ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
              "goal_fired_frac", "first_fire_len", "fallback_frac"):
        assert mine[f] == ref[f], f


def test_new_arms_record_mode_depths_width_and_fired5(tmp_path):
    """Catches a record that does not say what ran: B5-X carries mode B, k 5, k_goal 5,
    beam_width 64 and fired5; J4VG5-X carries mode C, k 4, k_goal 5."""
    b = ev.run_cell("B5-X", 9, 0, tmp_path, limit_states=2)
    assert (b["mode"], b["k"], b["k_goal"], b["beam_width"]) == ("B", 5, 5, 64)
    assert b["fired5"] == b["first_fire_len"].get("5", 0)
    assert 0 < b["judge_evals"] <= 1026
    g = ev.run_cell("J4VG5-X", 9, 0, tmp_path, limit_states=2)
    assert (g["mode"], g["k"], g["k_goal"]) == ("C", 4, 5) and "beam_width" not in g
    assert g["judge_evals"] == 1296
    assert json.loads((tmp_path / ev.record_name("B5-X", 9, 0)).read_text())["arm"] == "B5-X"


def test_unknown_arm_exits_cleanly(tmp_path):
    """Catches an arm typo running something unregistered."""
    with pytest.raises(SystemExit):
        ev.run_cell("B6-X", 9, 0, tmp_path, limit_states=1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/experiments/test_exp077.py -q` (timeout 600000)
Expected: FAIL, `evaluate.py` missing.

- [ ] **Step 3: Implement** `experiments/077_beam_search/evaluate.py` by copying `experiments/076_deeper_tree/evaluate.py` and changing:
  - module docstring: EXP-077, the three arms, spec path `docs/superpowers/specs/2026-10-14-exp077-beam-search-design.md`;
  - remove `rank_cell`, `rank_record_name`, `RANK_K`, `--rank` and the BFS / `tree_levels` imports (there is no Gate R);
  - `E76_OUT = REPO / "experiments" / "076_deeper_tree" / "outputs"`; keep `E75_OUT` only if used;
  - `ARMS` as in Interfaces; `record_name` with prefix `exp077_`;
  - in `run_cell`: `mode, k, k_goal, beam_width = ARMS[arm]` and call
    `evaluate_lookahead(agent, head, states, depth=depth, mode=mode, k=k, k_goal=k_goal, beam_width=beam_width, generator=torch.Generator().manual_seed(train_seed), rng_seed=train_seed, imag_seed=train_seed, critic=critic, no_revisit=True)`;
    record `"fired5": res["first_fire_len"].get("5", 0)` instead of `fired4`;
  - `main`: `--arm` required; print `fired5` and `judge_evals`.

- [ ] **Step 4: Run tests** (same command, timeout 600000). Expected: PASS. The slice test runs a real judge and takes about a minute.

- [ ] **Step 5: Commit**

```bash
git add experiments/077_beam_search/evaluate.py tests/experiments/test_exp077.py
git commit -m "EXP-077: evaluation driver for J4V-X, J4VG5-X and B5-X over EXP-075's committed X judges"
```

---

### Task 3: EXP-077 aggregator

**Files:**
- Create: `experiments/077_beam_search/aggregate.py`
- Test: `tests/experiments/test_exp077.py` (append)

**Interfaces:**
- Consumes: Task 2's `ARMS`, `DEPTHS`, `E76_OUT`, `record_name`; EXP-070's `one_sided_p`, `gate1_verdict`; EXP-071's `gate0b_verdict`; EXP-072's `pair_diffs`; EXP-075's `cells.SENSITIVITY_DROP` (all as `experiments/076_deeper_tree/aggregate.py` imports them).
- Produces: `ALPHA = 0.05 / 3`, `SMALL = 0.01`, `MAX_EVALS = 1026`, `NEW_ARMS = ("J4VG5-X", "B5-X")`, `AMENDMENT: dict | None = None`, `claim_verdict`, `contrast`, `_read`, `check_eval_record`, `_read_cell`, `load_records(out_dir)`, `rates_of`, `gate_d(recs) -> list[(depth, seed)]`, `gate_b(recs) -> dict` with keys `ok` (bool), `judge_evals` ({depth: mean}), `fallback` ({depth: mean}), `stage_cont(out) -> str`, `primary_verdicts(rates, gate_ok, seeds)`, `pilot_report(pilot_dir) -> dict`, CLI flags `--pilot-report`, `--stage cont`, `--determinism-ok`.

- [ ] **Step 1: Write the failing tests** (append to `tests/experiments/test_exp077.py`):

```python
ag = _load("exp077_aggregate", HERE / "aggregate.py")


def _rec(arm, d, s, solved=50, **kw):
    mode, k, kg, w = ev.ARMS[arm]
    r = {"arm": arm, "depth": d, "seed": s, "mode": mode, "k": k, "k_goal": kg,
         "no_revisit": True, "judge_updates": 4000, "limit_states": None, "n": 200,
         "solved": solved, "success_rate": solved / 200, "first_fire_len": {"5": solved},
         "fired5": solved, "judge_evals": 1296 if mode == "C" else 900, "fallback_frac": 0.0}
    if arm == "J4V-X":
        r["first_fire_len"] = {"4": solved}
    if w is not None:
        r["beam_width"] = w
    r.update(kw)
    return r


def _recs(**over):
    recs = {}
    for d in ev.DEPTHS:
        for arm in ag.ALL_ARMS:
            recs[(arm, d)] = {s: _rec(arm, d, s) for s in ag.SEEDS}
    for (arm, d, s), r in over.items():
        recs[(arm, d)][s] = r
    return recs


def test_constants_and_the_amendment_starts_unset():
    """Catches a wrong alpha, a drifted eval bound, or an amendment set before the pilot."""
    assert ag.ALPHA == pytest.approx(0.05 / 3) and ag.SMALL == 0.01 and ag.MAX_EVALS == 1026
    assert ag.ALL_ARMS == ("J4V-X", "J4VG5-X", "B5-X")


def test_claims_compare_the_registered_arms_and_depths():
    """Catches a claim wired to the wrong pair or depth (spec section 8)."""
    rates = {(a, d): {s: 0.1 for s in ag.SEEDS} for a in ag.ALL_ARMS for d in ev.DEPTHS}
    for s in ag.SEEDS:
        rates[("B5-X", 11)][s] = 0.2 + 0.001 * s
        rates[("B5-X", 9)][s] = 0.3 + 0.001 * s
    v = ag.primary_verdicts(rates, True, ag.SEEDS)
    assert v["claim1"][0] == "CONFIRMED" and v["claim1"][2] == pytest.approx(0.1055, abs=1e-4)
    assert v["claim2"][2] == pytest.approx(0.2055, abs=1e-4)
    assert v["claim3"][0] == "CONFIRMED"
    rates[("J4VG5-X", 11)] = dict(rates[("B5-X", 11)])
    assert ag.primary_verdicts(rates, True, ag.SEEDS)["claim3"][0] == "REFUTED"


def test_gate_d_clauses_each_name_their_cell():
    """Catches any of the three Gate D clauses missing: dominance, the first-fire key, and
    fired5 below J4V-X's solves, each on its own cell."""
    assert ag.gate_d(_recs()) == []
    bad = _recs(**{("J4VG5-X", 9, 2): _rec("J4VG5-X", 9, 2, solved=40, fired5=60,
                                             first_fire_len={"5": 60})})
    assert ag.gate_d(bad) == [(9, 2)]
    bad = _recs(**{("B5-X", 11, 5): _rec("B5-X", 11, 5, first_fire_len={"4": 1, "5": 49})})
    assert ag.gate_d(bad) == [(11, 5)]
    bad = _recs(**{("J4VG5-X", 11, 7): _rec("J4VG5-X", 11, 7, fired5=10)})
    assert ag.gate_d(bad) == [(11, 7)]


def test_gate_b_voids_above_the_eval_bound():
    """Catches the beam instrument check being reported but never enforced."""
    assert ag.gate_b(_recs())["ok"] is True
    assert ag.gate_b(_recs(**{("B5-X", 9, 0): _rec("B5-X", 9, 0, judge_evals=1100)}))["ok"] is False


def test_records_are_refused_unless_they_are_the_registered_cell(tmp_path):
    """Catches a smoke record, a wrong width, a partly trained judge or a renamed file entering
    a verdict."""
    p = tmp_path / ev.record_name("B5-X", 9, 0)
    for bad in (_rec("B5-X", 9, 0, limit_states=20), _rec("B5-X", 9, 0, beam_width=32),
                _rec("B5-X", 9, 0, judge_updates=3000), _rec("B5-X", 9, 1)):
        p.write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            ag._read_cell(p, "B5-X", 9, 0)


def test_load_records_reads_j4v_from_exp076(tmp_path):
    """Catches J4V-X read from this experiment's outputs (re-run data) instead of EXP-076's
    committed records, which Gate 0(b) licenses."""
    for d in ev.DEPTHS:
        for arm in ag.NEW_ARMS:
            for s in ag.SEEDS:
                (tmp_path / ev.record_name(arm, d, s)).write_text(json.dumps(_rec(arm, d, s)))
    recs = ag.load_records(tmp_path)
    ref = json.loads((ev.E76_OUT / "exp076_J4V-X_d11_s0.json").read_text())
    assert recs[("J4V-X", 11)][0]["solved"] == ref["solved"] == 36
```

In `test_constants_and_the_amendment_starts_unset`, the amendment assertion is deliberately omitted: the pilot amendment sets it in a later commit, and a test asserting `None` would then have to be deleted.

- [ ] **Step 2: Run to verify they fail** (timeout 600000). Expected: FAIL, `aggregate.py` missing.

- [ ] **Step 3: Implement** `experiments/077_beam_search/aggregate.py` by copying `experiments/076_deeper_tree/aggregate.py` and changing:
  - docstring to EXP-077, spec path, Gates 0, D, B and 1;
  - `ev = _load("exp077_evaluate", HERE / "evaluate.py")`; `NEW_ARMS = ("J4VG5-X", "B5-X")`; `ALL_ARMS = ("J4V-X",) + NEW_ARMS`; `MAX_EVALS = 1026`; `AMENDMENT: dict | None = None`;
  - `check_eval_record`: arm in `ev.ARMS`; `mode, k, kg, w = ev.ARMS[arm]`; check `mode`, `k`, `k_goal`, `no_revisit is True`, `judge_updates == 4000`, and `rec.get("beam_width") == w` (None for mode C: `.get` returns None when absent);
  - `load_records`: J4V-X from `ev.E76_OUT / f"exp076_J4V-X_d{d}_s{s}.json"` with `_read_cell(..., "J4V-X", d, s)`; new arms from `out_dir`;
  - `gate_d` per spec section 7: on each cell, `J4VG5-X solved < J4V-X solved`, or a `first_fire_len` key other than "5" for either new arm, or `J4VG5-X fired5 < J4V-X solved`, adds the cell;
  - `gate_b(recs)`: `judge_evals` and `fallback_frac` means per depth for B5-X; `ok` is False if any B5-X cell's `judge_evals > MAX_EVALS`;
  - delete `gate_g` and Gate R;
  - `stage_cont(out)`: `a71.gate0b_verdict` of this run's `J4V-X` seed 0 records at depths 9 and 11 (read through `_read_cell`) against EXP-076's;
  - `primary_verdicts`: claim1 `("B5-X", 11)` vs `("J4V-X", 11)`, claim2 the same at 9, claim3 `("B5-X", 11)` vs `("J4VG5-X", 11)`;
  - `pilot_report(pilot_dir)`: seconds per state per new arm from `outputs_pilot/` (20-state records, depth 11, seed 0), projected depth-11 cell hours (`* 200 / 3600`), and total evaluation cell hours `12 * (h + h * ratio)` per arm with `ratio` the EXP-076 J4V-X depth-9 over depth-11 mean `wall_s`;
  - `main`: print Gates 0(a), 0(b), D, B, then means (success, mean steps, optimality, `judge_evals`) per arm and depth, then claims with their 10-seed lines, then secondaries (claim 3 at depth 9; J4VG5-X minus J4V-X with summed `fired5`). `gate_ok = g0b == "PASS" and determinism_ok and not gate_d and gate_b["ok"]`.

- [ ] **Step 4: Run tests** (`tests/experiments/test_exp077.py`, timeout 600000). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/077_beam_search/aggregate.py tests/experiments/test_exp077.py
git commit -m "EXP-077: aggregator with Gates 0, D, B and 1 as verdicts, claims against J4V-X and J4VG5-X"
```

---

### Task 4: EXP-077 laptop launcher and chain

**Files:**
- Create: `experiments/077_beam_search/launch077.ps1`, `experiments/077_beam_search/chain077.ps1`
- Test: `tests/experiments/test_exp077.py` (append)

**Interfaces:**
- Consumes: Task 2's CLI (`--arm --depth --seed --out-dir --limit-states`), Task 3's `AMENDMENT` and `--stage cont`.
- Produces: phases `check, pilot, cont, eval, det`; logs `experiments\077_beam_search\phase_<phase>.log` with `peak commit MB <label> <n>` lines measured on the `python3.13` child.

- [ ] **Step 1: Write the failing tests** (append):

```python
def _ps(name):
    return (HERE / name).read_text()


def test_launcher_carries_the_fixes_and_the_registered_cells():
    """Catches a launcher missing the EXP-074 handle fix, the amendment guard, the worker cap,
    the beam seam check, or a registered cell."""
    t = _ps("launch077.ps1")
    assert "$null = $proc.Handle" in t
    assert "AMENDMENT is unset" in t
    assert "Workers -gt 6" in t
    assert "beam_width" in t
    for arm in ("J4VG5-X", "B5-X"):
        assert f'"{arm}"' in t
    assert '"J4V-X d9 s0"' in t and '"J4V-X d11 s0"' in t      # cont
    assert '"B5-X d9 s0"' in t                                   # det


def test_launcher_measures_the_worker_not_the_redirector():
    """Catches EXP-076's 2 MB peak: the venv's python.exe is a redirector whose CHILD is the
    python3.13 worker, so memory must be read from the child process."""
    t = _ps("launch077.ps1")
    assert "ParentProcessId" in t and "Win32_Process" in t
    assert "PrivateMemorySize64" in t


def test_chain_stops_unless_gate_0b_passes():
    """Catches the chain spending the evaluation's hours after a failed continuity check."""
    t = _ps("chain077.ps1")
    assert "--stage cont" in t and "STOP" in t
    assert t.index('"cont"') < t.index('"gate"') < t.index('"eval"') < t.index('"det"')
```

- [ ] **Step 2: Run to verify they fail** (timeout 600000).

- [ ] **Step 3: Implement.**
  - `launch077.ps1`: copy `experiments/076_deeper_tree/launch076.ps1` and change: `$exp = "experiments\077_beam_search"`; the seam check tests `'beam_width' in inspect.signature(l.evaluate_lookahead).parameters`; the presence check counts 12 trained X judges (unchanged) and 24 `experiments/076_deeper_tree/outputs/exp076_J4V-X_d{9,11}_s{0..11}.json` records, expecting `"12 24"`; `ValidateSet("check","pilot","cont","eval","det")`; jobs:
    - pilot: `J4VG5-X` and `B5-X` at d11 s0 with `--limit-states 20`;
    - cont: `J4V-X d9 s0` and `J4V-X d11 s0`;
    - eval: for arm in `"B5-X", "J4VG5-X"`, depth in 11, 9, seed 0..11;
    - det: `B5-X d9 s0`.
    Write the cont and det jobs with literal labels (`"J4V-X d9 s0"`, `"J4V-X d11 s0"`, `"B5-X d9 s0"`), not a loop over `$d`: the launcher test greps for them.
    Replace the `$sample` block so it reads the child:

```powershell
$sample = {
    foreach ($p in $running) {
        try {
            if (-not $p.HasExited) {
                $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" -ErrorAction SilentlyContinue)
                $ids = @($p.Id) + @($kids | ForEach-Object { $_.ProcessId })
                foreach ($id in $ids) {
                    $q = Get-Process -Id $id -ErrorAction SilentlyContinue
                    if ($q) {
                        $m = [int]($q.PrivateMemorySize64 / 1MB)
                        if ($m -gt $peak[$p.Id]) { $peak[$p.Id] = $m }
                    }
                }
            }
        } catch { }
    }
}
```

    and add a header comment: the venv's `python.exe` is a redirector; the worker is its `python3.13` child (EXP-076 section 12).
  - `chain077.ps1`: copy `experiments/076_deeper_tree/chain076.ps1`; launcher path `experiments\077_beam_search\launch077.ps1`, log `C:\Users\mlgbr\chain077.log`, order `@("cont","gate","eval","det")` (no rank), the gate runs `experiments\077_beam_search\aggregate.py --stage cont`.

- [ ] **Step 4: Run tests** (timeout 600000). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/077_beam_search/launch077.ps1 experiments/077_beam_search/chain077.ps1 tests/experiments/test_exp077.py
git commit -m "EXP-077: laptop launcher reading the python3.13 worker's committed memory, and the cont-gate-eval-det chain"
```
