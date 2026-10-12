# EXP-076 Deeper Tree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate EXP-075's 12 committed X judges with a 4-move look-ahead tree, plus an arm that extends only the goal check, and aggregate the three pre-registered claims.

**Architecture:** One library change: `k_goal` in `src/neuromorphic/training/lookahead.py`, which defaults to `k` so every existing caller is byte-identical, and a `first_fire_len` histogram. A new experiment folder `experiments/076_deeper_tree/` reuses EXP-075's judge loading and cells through importlib (never copies them) and adds an evaluation driver, an aggregator whose gates are verdicts, and a laptop launcher.

**Tech Stack:** Python 3.10 (VPS) / 3.13 (laptop), PyTorch, pytest, PowerShell 5.1 on the laptop.

**Spec:** `docs/superpowers/specs/2026-10-11-exp076-deeper-tree-design.md`

## Global Constraints

- Arms (spec section 3): `J3V-X` = k 3, goal check 3 (default); `J3VG4-X` = k 3, goal check 4; `J4V-X` = k 4, goal check 4. Every arm uses mode `"C"`, `no_revisit=True`, and EXP-075's committed X judge for that seed.
- Depths 9 and 11. Seeds 0 to 11. Sensitivity line drops seeds 0 and 3.
- Alpha 0.05 / 3. One-sided exact sign-flip (EXP-070's `one_sided_p`). Gate 1 floor 0.02, ceiling 0.98 (EXP-070's `gate1_verdict`).
- `CONFIRMED (SMALL)`: a confirmed claim whose mean difference is under **0.01**.
- Judges must be at **4000** completed updates (EXP-075's `N_UPDATES`).
- Records are named `exp076_<arm>_d<depth>_s<seed>.json`; rank records `exp076_rank_J-X4_d<depth>_s<seed>.json`.
- Plain commit messages. **No `Co-Authored-By` trailer, no "Generated with" line. No em-dashes** in code, docs or commit messages.
- Stage explicit paths. Never `git add -A`.
- Work happens in a worktree. The shared `.venv` lives in the main checkout, so every Python command from the worktree is `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python ...`. Without `PYTHONPATH=src` the tests import main's `src` and test the wrong code.
- **Always pass an explicit Bash `timeout` (the default is 120 s).** Run each test command in the FOREGROUND and let it block; a 600000 ms timeout covers every command in this plan. Silence mid-run is buffered output, not a hang. Never background a test run.
- Every test's docstring names the bug it catches, and every test must fail against the pre-change code (CLAUDE.md test-strength rule).
- `distance-to-solved` is an instrument only: nothing in `lookahead.py` may import the BFS provider (an existing test enforces this).

## Review Focus

1. A smoke record (`limit_states` set) or a record at the wrong `k`/`k_goal` reaching a verdict: the aggregator must refuse it by name (Task 3 tests `test_smoke_and_wrong_k_records_are_refused`).
2. A J3V-X row from EXP-075 evaluated from a partly trained judge entering Claims 1 and 2: refused on `judge_updates != 4000` (Task 3 `test_partly_trained_records_are_refused`).
3. Gate D comparing the wrong pairing (another seed or depth): a violation on one specific cell must be reported as exactly that cell (Task 3 `test_gate_d_names_the_violating_cell`).
4. A relaunch after a Windows Update restart re-running finished cells: the launcher's `-SkipExisting` must check the phase's own output directory, including the pilot and det directories (Task 4 `test_launcher_skips_by_the_phase_directory`).
5. `k_goal < k` silently accepted, letting the judge see past the goal check: refused in both `choose_move` and `evaluate_lookahead` (Task 1 `test_k_goal_below_k_is_refused`).

---

### Task 1: `k_goal` and the first-fire histogram in the look-ahead library

**Files:**
- Modify: `src/neuromorphic/training/lookahead.py` (`choose_move` at about line 116, `evaluate_lookahead` at about line 161)
- Test: `tests/training/test_lookahead_k_goal.py` (create)

**Interfaces:**
- Produces: `choose_move(mode, state, root_logits, k, n_actions, *, agent=None, head=None, imag_generator=None, critic=None, visited=None, info=None, k_goal=None)`. When the goal check fires and `info` is a dict, sets `info["fire_len"] = len(prefix)`.
- Produces: `evaluate_lookahead(agent, head, states, *, depth, mode, k, generator, rng_seed=0, imag_seed=0, trace=None, critic=None, no_revisit=False, k_goal=None)`. Its returned dict gains `"k_goal"` (int: `k` when `k_goal` is None) and `"first_fire_len"` (dict, str(length) -> number of episodes whose FIRST goal fire had that prefix length, keys sorted ascending by length).
- Both raise `ValueError` when `k_goal is not None and k_goal < k`.

- [ ] **Step 1: Write the failing tests**

Create `tests/training/test_lookahead_k_goal.py`:

```python
"""EXP-076 seam: a goal check deeper than the judge tree. Each docstring names the bug it catches."""

from __future__ import annotations

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
    """Deterministic, state-reading scores that ignore the generator: a judge stand-in."""

    reads_states = True

    def __call__(self, states, generator):
        return torch.tensor([float(hash(s) % 1009) for s in states])


def _c(agent, head, states, depth, k, k_goal=None, trace=None):
    return la.evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=k, k_goal=k_goal,
                                 generator=torch.Generator().manual_seed(3), rng_seed=3,
                                 imag_seed=3, critic=_HashCritic(), no_revisit=True, trace=trace)


def test_k_goal_below_k_is_refused():
    """Catches a goal check shallower than the judge tree being accepted: the judge would score
    leaves past a solve the goal check cannot see, a regime the spec forbids."""
    s = apply_move(SOLVED, 0)
    with pytest.raises(ValueError, match="k_goal"):
        la.choose_move("C", s, torch.zeros(N_ACTIONS), 2, N_ACTIONS, critic=_HashCritic(),
                       k_goal=1)
    agent, head = _agent_and_head()
    with pytest.raises(ValueError, match="k_goal"):
        la.evaluate_lookahead(agent, head, [s], depth=1, mode="C", k=2, k_goal=1,
                              generator=torch.Generator().manual_seed(0),
                              critic=_HashCritic())


def test_choose_move_fires_at_k_goal_not_k():
    """Catches `solving_prefix` still being called with k: a 2-move solve must fire with k 1 and
    k_goal 2, must NOT fire with k 1 alone, and must report its prefix length."""
    s = apply_move(apply_move(SOLVED, 0), 2)  # two different faces: no 1-move solve
    info = {}
    a, fired = la.choose_move("C", s, torch.zeros(N_ACTIONS), 1, N_ACTIONS,
                              critic=_HashCritic(), info=info, k_goal=2)
    assert fired is True and info["fire_len"] == 2
    assert la.solving_prefix(s, 2, N_ACTIONS)[0] == a
    info = {}
    _, fired = la.choose_move("C", s, torch.zeros(N_ACTIONS), 1, N_ACTIONS,
                              critic=_HashCritic(), info=info)
    assert fired is False and "fire_len" not in info


def test_default_k_goal_is_byte_identical_to_explicit_k():
    """THE GATE 0(b) CODE PATH. Catches a default that is not k (e.g. 0 or k + 1): every field
    of a C rollout with k_goal omitted equals the same rollout with k_goal = k."""
    agent, head = _agent_and_head()
    states = _shell(3, 4)
    a = _c(agent, head, states, 3, 1)
    b = _c(agent, head, states, 3, 1, k_goal=1)
    assert a == b
    assert a["k_goal"] == 1


def test_first_fire_len_counts_each_episode_once_at_its_first_fire():
    """Catches counting every fired move instead of each episode's first fire, and a histogram
    keyed by step instead of prefix length. E with k 2 on depth-2 states fires at step 1 with a
    2-move prefix, then again with a 1-move prefix: only the first counts."""
    agent, head = _agent_and_head()
    states = _shell(2, 4)
    r = la.evaluate_lookahead(agent, head, states, depth=2, mode="E", k=2,
                              generator=torch.Generator().manual_seed(1), rng_seed=1)
    assert r["first_fire_len"] == {"2": 4}
    assert r["solved"] == 4 and r["k_goal"] == 2


def test_deeper_goal_check_plays_the_same_moves_until_it_first_fires_and_never_solves_less():
    """THE GATE D INVARIANT. Catches the goal-check arm consuming the evaluation or imagination
    stream differently from its k_goal = k twin: before its first fire it must play identical
    moves on every state, and it must solve at least as many."""
    agent, head = _agent_and_head()
    states = _shell(3, 8)
    ta, tb = [], []
    a = _c(agent, head, states, 3, 1, trace=ta)
    b = _c(agent, head, states, 3, 1, k_goal=2, trace=tb)
    compared = 0
    for a_steps, b_steps in zip(ta, tb):
        for (aa, _, _), (ba, fired, _) in zip(a_steps, b_steps):
            if fired:
                break
            assert aa == ba
            compared += 1
    assert compared >= 8  # the test must actually compare something
    assert b["first_fire_len"].get("2", 0) >= 1  # and the deeper check must actually fire
    assert b["solved"] >= a["solved"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (from the worktree root, Bash timeout 600000):
`PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/training/test_lookahead_k_goal.py -q`
Expected: 5 FAIL, each with `TypeError: ... unexpected keyword argument 'k_goal'` (or a missing `k_goal`/`first_fire_len` key).

- [ ] **Step 3: Implement**

In `src/neuromorphic/training/lookahead.py`, add a validator above `choose_move`:

```python
def _check_k_goal(k, k_goal) -> int:
    """k_goal defaults to k. A goal check shallower than the judge tree is refused (EXP-076)."""
    if k_goal is None:
        return k
    if k_goal < k:
        raise ValueError(f"k_goal {k_goal} is below k {k}: the judge tree would score leaves "
                         f"past the goal check")
    return k_goal
```

Change `choose_move`'s signature to end `..., visited=None, info=None, k_goal=None):`. Directly after the `if mode not in MODES:` check, add `k_goal = _check_k_goal(k, k_goal)`. Replace

```python
    prefix = solving_prefix(state, k, n_actions)
    if prefix is not None:
        return prefix[0], True
```

with

```python
    prefix = solving_prefix(state, k_goal, n_actions)
    if prefix is not None:
        if info is not None:
            info["fire_len"] = len(prefix)
        return prefix[0], True
```

Nothing else in `choose_move` changes.

In `evaluate_lookahead`:
- Add `k_goal=None` as the last keyword parameter (after `no_revisit=False`).
- After the `if mode == "G" and k != 0:` check, add `k_goal = _check_k_goal(k, k_goal)`.
- Before the `for i, state in enumerate(states):` loop, add `fire_hist: dict[int, int] = {}`.
- Inside that loop, next to `state_trace: list[...] = []`, add `first_len = None`.
- Pass `k_goal=k_goal` to the `choose_move(...)` call.
- Directly after `fired_moves += int(fired)`, add:

```python
            if fired and first_len is None:
                first_len = info["fire_len"]
```

- After the inner `for t in ...` loop ends (next to `eval_revisits += ...`), add:

```python
        if first_len is not None:
            fire_hist[first_len] = fire_hist.get(first_len, 0) + 1
```

- In the returned dict, add after `"k": k,`:

```python
        "k_goal": k_goal,
        "first_fire_len": {str(n): fire_hist[n] for n in sorted(fire_hist)},
```

Mode G with `k=0` returns `"k_goal": 0` and `{}`. G never fires.

- [ ] **Step 4: Run the new tests and every existing look-ahead test**

Run (Bash timeout 600000):
`PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/training/test_lookahead_k_goal.py tests/training/test_lookahead.py tests/training/test_lookahead_state_critic.py tests/training/test_lookahead_golden.py -q -m "not slow"`
Expected: all PASS. The golden test proves EXP-070's arms are unchanged. If `test_deeper_goal_check_...` fails only on its `>= 1` resolution line, report it as DONE_WITH_CONCERNS with the histogram you saw. Do not weaken the assertion.

- [ ] **Step 5: Mutation check (record the results in your report)**

Apply each mutation, run `tests/training/test_lookahead_k_goal.py`, confirm at least one test FAILS, then revert. Run `git diff --stat` after each revert to prove the tree is clean.
1. `solving_prefix(state, k, n_actions)` instead of `k_goal`.
2. `_check_k_goal` returns `k + 1` when `k_goal is None`.
3. Count every fired move: drop the `first_len is None` condition.
4. Drop the `if k_goal < k: raise`.

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/training/lookahead.py tests/training/test_lookahead_k_goal.py
git commit -m "Look-ahead: k_goal, a goal check deeper than the judge tree (default k), and the first-fire histogram (EXP-076)"
```

---

### Task 2: EXP-076 evaluation and rank driver

**Files:**
- Create: `experiments/076_deeper_tree/evaluate.py`
- Test: `tests/experiments/test_exp076.py` (create; Task 3 appends to it)

**Interfaces:**
- Consumes: Task 1's `evaluate_lookahead(..., k_goal=...)`, which returns `"k_goal"` and `"first_fire_len"`. From `experiments/075_wide_region/evaluate.py` (loaded by importlib as `e75`): `e75.load_judge(train_arm, seed, judge_dir=None, required_updates=4000)`, `e75.judge_updates(train_arm, seed, judge_dir=None)`, `e75.JudgeCritic`, and `e75.load_cell(depth, seed) -> (agent, head, states, train_seed)`.
- Produces:
  - `ARMS = {"J3V-X": (3, None), "J3VG4-X": (3, 4), "J4V-X": (4, 4)}`, `DEPTHS = (9, 11)`, `RANK_K = 4`, `E75_OUT`;
  - `record_name(arm, depth, seed) -> str` and `rank_record_name(depth, seed) -> str`;
  - `run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict`. The record holds every `evaluate_lookahead` key plus `depth, seed, arm, fired4, wall_s, git_commit, limit_states, judge_updates`;
  - `rank_cell(depth, seed, out_dir, judge_dir=None, limit_states=None, provider=None) -> dict`, with keys `kind ("J-X4"), k (4), depth, seed, n, hit, chance, limit_states, judge_updates`.

- [ ] **Step 1: Write the failing tests**

Create `tests/experiments/test_exp076.py`:

```python
"""EXP-076: a deeper look-ahead tree over EXP-075's committed X judges."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from neuromorphic.envs.cube_distance import ExactBFSDistance

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "076_deeper_tree"
E75 = REPO / "experiments" / "075_wide_region"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate = _load("exp076_evaluate", EXP / "evaluate.py")


def test_arms_are_the_spec_table():
    """Catches an arm wired to the wrong tree or goal depth (spec section 3), or J3V-X given an
    explicit goal depth instead of the default that Gate 0(b) checks."""
    assert evaluate.ARMS == {"J3V-X": (3, None), "J3VG4-X": (3, 4), "J4V-X": (4, 4)}
    assert evaluate.DEPTHS == (9, 11) and evaluate.RANK_K == 4


def test_j3v_x_here_equals_exp075s_harness_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH. Catches this harness drifting from EXP-075's: J3V-X through it
    equals EXP-075's own run_cell on the same states, committed judge, in every outcome field."""
    mine = evaluate.run_cell("J3V-X", 9, 0, tmp_path / "a", limit_states=2)
    ref = evaluate.e75.run_cell("J3V-X", 9, 0, tmp_path / "b", limit_states=2)
    for f in ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
              "greedy_modal_action_frac", "goal_fired_frac", "fallback_frac"):
        assert mine[f] == ref[f], f
    assert mine["k"] == 3 and mine["k_goal"] == 3


def test_new_arms_write_their_tree_and_goal_depths_and_fired4(tmp_path):
    """Catches a record that does not say which tree it ran (so the aggregator could not refuse
    a mislabelled one), and fired4 not taken from the first-fire histogram."""
    for arm, (k, kg) in (("J3VG4-X", (3, 4)), ("J4V-X", (4, 4))):
        rec = evaluate.run_cell(arm, 7, 0, tmp_path, limit_states=1)
        assert rec["arm"] == arm and rec["k"] == k and rec["k_goal"] == kg
        assert rec["mode"] == "C" and rec["no_revisit"] is True
        assert rec["fired4"] == rec["first_fire_len"].get("4", 0)
        assert rec["limit_states"] == 1 and rec["judge_updates"] == 4000
        on_disk = json.loads((tmp_path / f"exp076_{arm}_d7_s0.json").read_text())
        assert on_disk["k_goal"] == kg


def test_goal_check_arm_never_solves_less_than_j3v_on_a_slice(tmp_path):
    """THE GATE D INVARIANT ON THE REAL JUDGE. Catches the goal-check arm using a different
    stream from J3V-X: on the same states it must solve at least as many."""
    a = evaluate.run_cell("J3V-X", 9, 0, tmp_path, limit_states=6)
    b = evaluate.run_cell("J3VG4-X", 9, 0, tmp_path, limit_states=6)
    assert b["solved"] >= a["solved"]


def test_unknown_arm_and_missing_judge_exit_cleanly(tmp_path):
    """Catches a typo'd arm running some default, and a missing judge evaluating an untrained
    one."""
    with pytest.raises(SystemExit, match="unknown arm"):
        evaluate.run_cell("J5V-X", 9, 0, tmp_path, limit_states=1)
    with pytest.raises(SystemExit, match="judge.pt"):
        evaluate.run_cell("J4V-X", 9, 0, tmp_path, limit_states=1, judge_dir=tmp_path / "no")


def test_rank_cell_ranks_four_move_leaves(tmp_path, monkeypatch):
    """Catches Gate R measured on EXP-075's 3-move leaves instead of 4, and a rank record
    without its limit (so a smoke row could enter Gate R)."""
    seen = []
    real = evaluate.tree_levels

    def spy(state, k, n):
        seen.append(k)
        return real(state, k, n)

    monkeypatch.setattr(evaluate, "tree_levels", spy)
    rec = evaluate.rank_cell(7, 0, tmp_path, limit_states=2,
                             provider=ExactBFSDistance(max_depth=11))
    assert set(seen) == {4}
    assert rec["kind"] == "J-X4" and rec["k"] == 4 and rec["n"] == 2
    assert rec["limit_states"] == 2 and rec["judge_updates"] == 4000
    assert 0.0 < rec["chance"] < 1.0 and rec["hit"] in (0.0, 0.5, 1.0)
    assert (tmp_path / "exp076_rank_J-X4_d7_s0.json").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (Bash timeout 600000):
`PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp076.py -q`
Expected: collection ERROR, because `experiments/076_deeper_tree/evaluate.py` does not exist.

- [ ] **Step 3: Implement**

Create `experiments/076_deeper_tree/evaluate.py`:

```python
"""EXP-076 evaluation: EXP-075's committed X judges drive a deeper look-ahead tree.

J3V-X is EXP-075's arm, re-run here only for Gate 0(b) (default goal depth). J3VG4-X extends the
goal check to 4 moves with the 3-move judge tree; J4V-X searches 4 moves with both. Judges,
held-out sets, budgets, streams and the no-revisit rule are EXP-075's (its `load_judge`,
`JudgeCritic` and `load_cell` are used directly). Spec:
docs/superpowers/specs/2026-10-11-exp076-deeper-tree-design.md, sections 3 and 7.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/076_deeper_tree/evaluate.py --arm J4V-X --depth 11 --seed 0
    .venv/bin/python -u experiments/076_deeper_tree/evaluate.py --rank --depth 11 --seed 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import subprocess
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e75 = _load("exp075_evaluate", REPO / "experiments" / "075_wide_region" / "evaluate.py")

from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training.lookahead import (  # noqa: E402
    evaluate_lookahead, imag_seed_for, imagined_values, tree_levels,
)

torch.set_num_threads(1)

E75_OUT = REPO / "experiments" / "075_wide_region" / "outputs"
# arm -> (judge tree depth k, goal check depth k_goal); None means the default, k (spec section 3)
ARMS = {"J3V-X": (3, None), "J3VG4-X": (3, 4), "J4V-X": (4, 4)}
DEPTHS = (9, 11)
RANK_K = 4


def record_name(arm: str, depth: int, seed: int) -> str:
    return f"exp076_{arm}_d{depth}_s{seed}.json"


def rank_record_name(depth: int, seed: int) -> str:
    return f"exp076_rank_J-X4_d{depth}_s{seed}.json"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _critic(seed, judge_dir):
    return e75.JudgeCritic(e75.load_judge("X", seed, judge_dir))


def _write(out_dir, name, rec) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")


def run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict:
    torch.set_num_threads(1)
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; valid: {list(ARMS)}")
    k, k_goal = ARMS[arm]
    t0 = time.time()
    critic = _critic(seed, judge_dir)
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=k, k_goal=k_goal,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    rec = {**res, "depth": depth, "seed": seed, "arm": arm,
           "fired4": res["first_fire_len"].get("4", 0),
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "limit_states": limit_states,
           "judge_updates": e75.judge_updates("X", seed, judge_dir)}
    _write(out_dir, record_name(arm, depth, seed), rec)
    return rec


def rank_cell(depth, seed, out_dir, judge_dir=None, limit_states=None, provider=None) -> dict:
    """Gate R at k 4 (reported, not voiding; spec section 7): per-seed mean hit of 'the lowest-J
    4-move leaf is closer to solved than the root', on evaluation held-out states. BFS is the
    yardstick only; unbounded by default because depth + 4 passes the 2x2's diameter."""
    torch.set_num_threads(1)
    k = RANK_K
    critic = _critic(seed, judge_dir)
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    if provider is None:
        provider = ExactBFSDistance(max_depth=None)
    n_actions = head.head.out_features
    hits, chances = [], []
    for i, s in enumerate(states):
        g = torch.Generator().manual_seed(imag_seed_for(train_seed, i, 0))
        levels = tree_levels(s, k, n_actions)
        scores = imagined_values(agent, critic, levels[k], generator=g)
        d = provider.distance(s)
        leaf_d = [provider.distance(x) for x in levels[k]]
        hits.append(int(leaf_d[int(scores.argmax())] < d))
        chances.append(sum(x < d for x in leaf_d) / len(leaf_d))
    rec = {"kind": "J-X4", "k": k, "depth": depth, "seed": seed, "n": len(states),
           "hit": st.mean(hits), "chance": st.mean(chances), "limit_states": limit_states,
           "judge_updates": e75.judge_updates("X", seed, judge_dir)}
    _write(out_dir, rank_record_name(depth, seed), rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--judge-dir", type=Path, default=None,
                    help="EXP-075's outputs (where judge_X_s<seed>/ live) when omitted")
    ap.add_argument("--limit-states", type=int, default=None)
    args = ap.parse_args()
    if (args.arm is None) == (not args.rank):
        raise SystemExit("pass exactly one of --arm or --rank")
    if args.arm:
        r = run_cell(args.arm, args.depth, args.seed, args.out_dir, args.limit_states,
                     args.judge_dir)
        print(f"{r['arm']} d{r['depth']} s{r['seed']} solved {r['solved']}/{r['n']} "
              f"fired4 {r['fired4']} wall {r['wall_s']}s")
    else:
        r = rank_cell(args.depth, args.seed, args.out_dir, args.judge_dir, args.limit_states)
        print(f"J-X4 d{r['depth']} s{r['seed']} hit {r['hit']:.4f} chance {r['chance']:.4f}")


if __name__ == "__main__":
    main()
```

`e75.load_judge("X", seed, None)` resolves to EXP-075's `outputs/judge_X_s<seed>/`, the committed judges, and refuses a judge that is not at 4000 updates. Check that `e75.load_judge` and `e75.judge_updates` accept `judge_dir=None` this way (read `experiments/075_wide_region/evaluate.py` lines 60 to 102). If they do not, pass `E75_OUT` explicitly instead, and say so in your report.

- [ ] **Step 4: Run the tests**

Run (Bash timeout 600000):
`PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp076.py -q`
Expected: 6 PASS. The slice tests run the real committed judges and take a few minutes; that is expected.

- [ ] **Step 5: Mutation check (record in your report)**

Apply each mutation, run the file, confirm at least one test FAILS, then revert and run `git diff --stat`.
1. `"J4V-X": (4, 3)` in `ARMS`.
2. `k=3` instead of `k = RANK_K` in `rank_cell`.
3. Drop `k_goal=k_goal` from the `evaluate_lookahead` call.

- [ ] **Step 6: Commit**

```bash
git add experiments/076_deeper_tree/evaluate.py tests/experiments/test_exp076.py
git commit -m "EXP-076: evaluation and rank driver over EXP-075's committed X judges"
```

---

### Task 3: EXP-076 aggregator, gates as verdicts

**Files:**
- Create: `experiments/076_deeper_tree/aggregate.py`
- Test: `tests/experiments/test_exp076.py` (append)

**Interfaces:**
- Consumes: from Task 2, `ARMS`, `DEPTHS`, `E75_OUT`, `record_name`, `rank_record_name`. Loaded by importlib: EXP-070's `one_sided_p` and `gate1_verdict`, EXP-071's `gate0b_verdict` and `gate_r_verdict`, EXP-072's `pair_diffs`, and EXP-075's `cells.SENSITIVITY_DROP` (= (0, 3)).
- Produces:
  - constants `ALPHA`, `SMALL = 0.01`, `N_UPDATES = 4000`, `PILOT_LIMIT = 20`, `AMENDMENT: dict | None = None`;
  - `claim_verdict(diffs, a_mean, b_mean, gate_ok) -> (str, float)`;
  - `contrast(rates, a, b, gate_ok) -> (verdict, p, mean_diff, a_mean, b_mean)`;
  - `check_eval_record(rec, path)` and `load_records(out_dir) -> {(arm, depth): {seed: rec}}`;
  - `rates_of(recs)`, `gate_d(recs) -> list[(depth, seed)]` and `gate_g(recs) -> {depth: "RESOLVED" | "UNRESOLVED"}`;
  - `stage_cont(out) -> "PASS" | "FAIL"`, `primary_verdicts(rates, gate_ok, seeds) -> dict`, `sensitivity_rates(rates)` and `pilot_report(pilot_dir) -> dict`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp076.py`:

```python
agg = _load("exp076_aggregate", EXP / "aggregate.py")


def _rec(arm, depth, seed, solved, **over):
    k, kg = evaluate.ARMS[arm]
    r = {"arm": arm, "depth": depth, "seed": seed, "solved": solved, "n": 200,
         "success_rate": solved / 200, "k": k, "k_goal": kg if kg is not None else k,
         "mode": "C", "no_revisit": True, "judge_updates": 4000, "limit_states": None,
         "fired4": 3, "first_fire_len": {"4": 3}, "mean_steps": 10.0, "optimality": 0.9}
    r.update(over)
    return r


def _recs(j3, g4, j4, seeds=range(12)):
    return {(arm, d): {s: _rec(arm, d, s, v) for s in seeds}
            for arm, v in (("J3V-X", j3), ("J3VG4-X", g4), ("J4V-X", j4)) for d in (9, 11)}


def test_alpha_small_and_the_small_verdict():
    """Catches the claims read at another alpha, and a confirmed gain under 0.01 printed as a
    plain CONFIRMED (spec section 8's band must be a verdict, not prose)."""
    assert agg.ALPHA == pytest.approx(0.05 / 3) and agg.SMALL == 0.01
    assert agg.claim_verdict([0.05] * 12, 0.30, 0.25, True)[0] == "CONFIRMED"
    assert agg.claim_verdict([0.005] * 12, 0.255, 0.25, True)[0] == "CONFIRMED (SMALL)"
    assert agg.claim_verdict([0.05] * 12, 0.30, 0.25, False)[0] == "VOID"
    assert agg.claim_verdict([-0.01] * 12, 0.24, 0.25, True)[0] == "REFUTED"
    assert agg.claim_verdict([0.0] * 8 + [0.01] * 4, 0.255, 0.25, True)[0] == "NOT SIGNIFICANT"
    assert agg.claim_verdict([0.005] * 12, 0.015, 0.01, True)[0] == "UNRESOLVED"


def test_claims_compare_the_registered_arms_and_depths():
    """Catches a claim wired to the wrong arm or depth: every arm and depth gets a distinct rate,
    so each claim's two means identify exactly which cells it read."""
    rates = {("J3V-X", 9): 0.25, ("J3VG4-X", 9): 0.26, ("J4V-X", 9): 0.30,
             ("J3V-X", 11): 0.12, ("J3VG4-X", 11): 0.13, ("J4V-X", 11): 0.17}
    rates = {k: {s: v for s in range(12)} for k, v in rates.items()}
    v = agg.primary_verdicts(rates, True, range(12))
    assert v["claim1"][3:] == (pytest.approx(0.17), pytest.approx(0.12))
    assert v["claim2"][3:] == (pytest.approx(0.30), pytest.approx(0.25))
    assert v["claim3"][3:] == (pytest.approx(0.17), pytest.approx(0.13))


def test_sensitivity_drops_exactly_seeds_0_and_3():
    """Catches the 10-seed line dropping the wrong seeds or none."""
    rates = {("J4V-X", 11): {s: float(s) for s in range(12)}}
    assert sorted(agg.sensitivity_rates(rates)[("J4V-X", 11)]) == [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]


def test_gate_d_names_the_violating_cell():
    """Catches Gate D comparing a different pairing (another seed or depth) or passing with one
    violation: exactly the broken cell must be reported."""
    recs = _recs(30, 32, 40)
    assert agg.gate_d(recs) == []
    recs[("J3VG4-X", 11)][7] = _rec("J3VG4-X", 11, 7, 29)
    assert agg.gate_d(recs) == [(11, 7)]


def test_gate_g_is_unresolved_only_when_the_deeper_check_never_fired():
    """Catches Gate G reading fired4 from the wrong arm, or summing over both depths."""
    recs = _recs(30, 32, 40)
    assert agg.gate_g(recs) == {9: "RESOLVED", 11: "RESOLVED"}
    for s in range(12):
        recs[("J3VG4-X", 9)][s] = _rec("J3VG4-X", 9, s, 32, fired4=0)
        recs[("J4V-X", 9)][s] = _rec("J4V-X", 9, s, 40, fired4=9)
    assert agg.gate_g(recs) == {9: "UNRESOLVED", 11: "RESOLVED"}


def test_smoke_and_wrong_k_records_are_refused(tmp_path):
    """REVIEW FOCUS 1. Catches a smoke record or a record from another tree entering a verdict."""
    p = tmp_path / "exp076_J4V-X_d9_s0.json"
    p.write_text(json.dumps(_rec("J4V-X", 9, 0, 40, limit_states=20)))
    with pytest.raises(SystemExit, match="smoke"):
        agg._read(p)
    with pytest.raises(SystemExit, match="k_goal"):
        agg.check_eval_record(_rec("J4V-X", 9, 0, 40, k_goal=3), p)
    with pytest.raises(SystemExit, match="k is"):
        agg.check_eval_record(_rec("J4V-X", 9, 0, 40, k=3), p)
    with pytest.raises(SystemExit, match="arm"):
        agg.check_eval_record(_rec("J4V-X", 9, 0, 40, arm="J3V-W"), p)


def test_partly_trained_records_are_refused(tmp_path):
    """REVIEW FOCUS 2. Catches a row evaluated from a judge short of 4000 updates entering a
    claim, including the EXP-075 J3V-X rows this experiment reuses."""
    p = tmp_path / "x.json"
    with pytest.raises(SystemExit, match="judge_updates"):
        agg.check_eval_record(_rec("J3V-X", 11, 0, 30, judge_updates=3750), p)


def test_load_records_reads_j3v_from_exp075_and_new_arms_from_out(tmp_path):
    """Catches J3V-X read from this experiment's outputs (only Gate 0(b)'s seed 0 is re-run
    here) and refuses a missing new-arm record by name."""
    for arm in ("J3VG4-X", "J4V-X"):
        for d in (9, 11):
            for s in range(12):
                (tmp_path / evaluate.record_name(arm, d, s)).write_text(
                    json.dumps(_rec(arm, d, s, 50)))
    recs = agg.load_records(tmp_path)
    committed = json.loads((evaluate.E75_OUT / "exp075_J3V-X_d9_s0.json").read_text())
    assert recs[("J3V-X", 9)][0]["solved"] == committed["solved"]
    assert recs[("J4V-X", 11)][5]["solved"] == 50
    (tmp_path / evaluate.record_name("J4V-X", 11, 5)).unlink()
    with pytest.raises(SystemExit, match="exp076_J4V-X_d11_s5"):
        agg.load_records(tmp_path)


def test_stage_cont_compares_against_exp075s_committed_records(tmp_path):
    """THE GATE 0(b) VERDICT. Catches the continuity check comparing a record with itself, or
    against anything but EXP-075's committed J3V-X seed-0 records."""
    for d in (9, 11):
        ref = json.loads((evaluate.E75_OUT / f"exp075_J3V-X_d{d}_s0.json").read_text())
        (tmp_path / evaluate.record_name("J3V-X", d, 0)).write_text(json.dumps(ref))
    assert agg.stage_cont(tmp_path) == "PASS"
    bad = json.loads((tmp_path / evaluate.record_name("J3V-X", 11, 0)).read_text())
    bad["solved"] += 1
    (tmp_path / evaluate.record_name("J3V-X", 11, 0)).write_text(json.dumps(bad))
    assert agg.stage_cont(tmp_path) == "FAIL"


def test_pilot_report_projects_cell_hours_from_seconds_per_state(tmp_path):
    """Catches the amendment's schedule computed from wall_s without dividing by n, or from a
    pilot at another size."""
    for arm, wall in (("J3VG4-X", 200.0), ("J4V-X", 600.0)):
        (tmp_path / evaluate.record_name(arm, 11, 0)).write_text(json.dumps(
            _rec(arm, 11, 0, 3, n=20, limit_states=20, wall_s=wall)))
    (tmp_path / evaluate.rank_record_name(11, 0)).write_text(json.dumps(
        {"kind": "J-X4", "k": 4, "n": 20, "limit_states": 20, "hit": 0.6, "chance": 0.5,
         "wall_s": 40.0}))
    r = agg.pilot_report(tmp_path)
    assert r["s_per_state"]["J4V-X"] == pytest.approx(30.0)
    assert r["d11_cell_hours"]["J4V-X"] == pytest.approx(30.0 * 200 / 3600)
    bad = _rec("J4V-X", 11, 0, 3, n=5, limit_states=5, wall_s=1.0)
    (tmp_path / evaluate.record_name("J4V-X", 11, 0)).write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="limit_states"):
        agg.pilot_report(tmp_path)
```

`test_pilot_report_...` writes the rank pilot record with `wall_s`. Task 2's `rank_cell` does not write `wall_s` yet, so in this task add `"wall_s": round(time.time() - t0, 1)` to `rank_cell`'s record, with `t0 = time.time()` as its first line, and add `"wall_s"` to the keys asserted in Task 2's `test_rank_cell_ranks_four_move_leaves`.

- [ ] **Step 2: Run the tests to verify they fail**

Run (Bash timeout 600000):
`PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp076.py -q -k "not slice and not real"`
Expected: collection ERROR, because `aggregate.py` does not exist.

- [ ] **Step 3: Implement**

Create `experiments/076_deeper_tree/aggregate.py`:

```python
"""EXP-076 aggregator. Gates 0, D and 1 are VERDICTS: a claim whose gate failed prints VOID, a
contrast on the floor or ceiling prints UNRESOLVED, and a confirmed gain under 0.01 prints
CONFIRMED (SMALL). Gate G returns UNRESOLVED for the goal-check decomposition when the 4-move
goal check never fired. Gate R is reported, never voiding.

Spec: docs/superpowers/specs/2026-10-11-exp076-deeper-tree-design.md, sections 6 to 8.

The sign-flip test and Gate 1 are EXP-070's, Gate 0(b)'s field comparison and Gate R are
EXP-071's, seed pairing is EXP-072's, all reused through importlib. Alpha is 0.05 / 3.

Usage:
    .venv/bin/python experiments/076_deeper_tree/aggregate.py --pilot-report
    .venv/bin/python experiments/076_deeper_tree/aggregate.py --stage cont
    .venv/bin/python experiments/076_deeper_tree/aggregate.py --determinism-ok
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a70 = _load("exp070_aggregate", REPO / "experiments" / "070_lookahead_existing" / "aggregate.py")
a71 = _load("exp071_aggregate", REPO / "experiments" / "071_critic_and_no_revisit" / "aggregate.py")
a72 = _load("exp072_aggregate", REPO / "experiments" / "072_p3v_frontier" / "aggregate.py")
cells75 = _load("exp075_cells", REPO / "experiments" / "075_wide_region" / "cells.py")
ev = _load("exp076_evaluate", HERE / "evaluate.py")
one_sided_p = a70.one_sided_p

SEEDS = range(12)
DEPTHS = ev.DEPTHS
ALPHA = 0.05 / 3
SMALL = 0.01
N_UPDATES = 4000
PILOT_LIMIT = 20
NEW_ARMS = ("J3VG4-X", "J4V-X")
ALL_ARMS = ("J3V-X",) + NEW_ARMS

# Set by the dated post-pilot amendment (spec section 6): worker count and schedule. The
# launcher refuses every phase but `check` and `pilot` while this is None.
AMENDMENT: dict | None = None


def claim_verdict(diffs, a_mean: float, b_mean: float, gate_ok: bool) -> tuple[str, float]:
    p = one_sided_p(diffs)
    if not gate_ok:
        return "VOID", p
    if a70.gate1_verdict(a_mean, b_mean) == "UNRESOLVED":
        return "UNRESOLVED", p
    m = st.mean(diffs)
    if m <= 0:
        return "REFUTED", p
    if p < ALPHA:
        return ("CONFIRMED (SMALL)" if m < SMALL else "CONFIRMED"), p
    return "NOT SIGNIFICANT", p


def contrast(rates, a, b, gate_ok):
    diffs = a72.pair_diffs(rates[a], rates[b])
    a_mean, b_mean = st.mean(rates[a].values()), st.mean(rates[b].values())
    verdict, p = claim_verdict(diffs, a_mean, b_mean, gate_ok)
    return verdict, p, st.mean(diffs), a_mean, b_mean


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path.name}: the run is incomplete")
    r = json.loads(path.read_text())
    if r.get("limit_states") is not None:
        raise SystemExit(f"{path.name} is a smoke record and may not enter a verdict")
    return r


def check_eval_record(rec, path) -> None:
    name = Path(path).name
    arm = rec.get("arm")
    if arm not in ev.ARMS:
        raise SystemExit(f"{name}: arm is {arm!r}, expected one of {list(ev.ARMS)}")
    k, kg = ev.ARMS[arm]
    kg = k if kg is None else kg
    if rec.get("k") != k:
        raise SystemExit(f"{name}: k is {rec.get('k')!r}, spec says {k} for {arm}")
    # EXP-075's J3V-X records predate k_goal; their goal check was k by construction.
    if rec.get("k_goal", k) != kg:
        raise SystemExit(f"{name}: k_goal is {rec.get('k_goal')!r}, spec says {kg} for {arm}")
    if rec.get("mode") != "C" or rec.get("no_revisit") is not True:
        raise SystemExit(f"{name}: mode {rec.get('mode')!r}, no_revisit "
                         f"{rec.get('no_revisit')!r}; spec says C with no-revisit")
    if rec.get("judge_updates") != N_UPDATES:
        raise SystemExit(f"{name}: judge_updates is {rec.get('judge_updates')!r}, spec says "
                         f"{N_UPDATES}: it was evaluated from a partly trained judge")


def _read_checked(path: Path) -> dict:
    r = _read(path)
    check_eval_record(r, path)
    return r


def load_records(out_dir) -> dict:
    """(arm, depth) -> {seed: record}. J3V-X from EXP-075's committed records (licensed by
    Gate 0(b)); the two new arms from this experiment's outputs."""
    out_dir = Path(out_dir)
    recs = {}
    for d in DEPTHS:
        recs[("J3V-X", d)] = {s: _read_checked(ev.E75_OUT / f"exp075_J3V-X_d{d}_s{s}.json")
                              for s in SEEDS}
        for arm in NEW_ARMS:
            recs[(arm, d)] = {s: _read_checked(out_dir / ev.record_name(arm, d, s))
                              for s in SEEDS}
    return recs


def rates_of(recs) -> dict:
    return {key: {s: r["success_rate"] for s, r in by_seed.items()}
            for key, by_seed in recs.items()}


def gate_d(recs) -> list:
    """(depth, seed) cells where the goal-check arm solved fewer than J3V-X. Empty is PASS."""
    return [(d, s) for d in DEPTHS for s in SEEDS
            if recs[("J3VG4-X", d)][s]["solved"] < recs[("J3V-X", d)][s]["solved"]]


def gate_g(recs) -> dict:
    return {d: "RESOLVED" if sum(recs[("J3VG4-X", d)][s]["fired4"] for s in SEEDS) > 0
            else "UNRESOLVED" for d in DEPTHS}


def stage_cont(out) -> str:
    """Gate 0(b): J3V-X at seed 0, depths 9 and 11, re-run here, against EXP-075's records."""
    out = Path(out)
    return a71.gate0b_verdict(
        {("J3V-X", d): _read(out / ev.record_name("J3V-X", d, 0)) for d in DEPTHS},
        {("J3V-X", d): _read(ev.E75_OUT / f"exp075_J3V-X_d{d}_s0.json") for d in DEPTHS})


def primary_verdicts(rates: dict, gate_ok: bool, seeds) -> dict:
    """Claim 1: d11 J4V-X vs J3V-X. Claim 2: d9 J4V-X vs J3V-X. Claim 3: d11 J4V-X vs J3VG4-X."""
    r = {k: {s: v[s] for s in seeds} for k, v in rates.items()}
    return {"claim1": contrast(r, ("J4V-X", 11), ("J3V-X", 11), gate_ok),
            "claim2": contrast(r, ("J4V-X", 9), ("J3V-X", 9), gate_ok),
            "claim3": contrast(r, ("J4V-X", 11), ("J3VG4-X", 11), gate_ok)}


def sensitivity_rates(rates: dict) -> dict:
    return {k: {s: v for s, v in d.items() if s not in cells75.SENSITIVITY_DROP}
            for k, d in rates.items()}


def pilot_report(pilot_dir) -> dict:
    """Seconds per state and projected depth-11 cell hours per new arm, from the pilot's
    20-state records (spec section 6). Depth 9 is projected by EXP-075's own d9/d11 wall
    ratio for J3V-X. Memory is read on the laptop, not here."""
    pilot_dir = Path(pilot_dir)

    def pilot(path):
        if not path.exists():
            raise SystemExit(f"missing pilot record {path.name}")
        r = json.loads(path.read_text())
        if r.get("limit_states") != PILOT_LIMIT:
            raise SystemExit(f"{path.name}: limit_states is {r.get('limit_states')!r}, the "
                             f"pilot is {PILOT_LIMIT} states")
        return r

    walls = {d: st.mean(json.loads((ev.E75_OUT / f"exp075_J3V-X_d{d}_s{s}.json").read_text())
                        ["wall_s"] for s in SEEDS) for d in DEPTHS}
    ratio = walls[9] / walls[11]
    sps = {arm: pilot(pilot_dir / ev.record_name(arm, 11, 0))["wall_s"] / PILOT_LIMIT
           for arm in NEW_ARMS}
    rank = pilot(pilot_dir / ev.rank_record_name(11, 0))
    d11 = {arm: v * 200 / 3600 for arm, v in sps.items()}
    total = sum(12 * (h + h * ratio) for h in d11.values())
    return {"s_per_state": sps, "d11_cell_hours": d11, "d9_over_d11_wall_ratio": ratio,
            "rank_s_per_state": rank["wall_s"] / PILOT_LIMIT,
            "eval_cell_hours_total": total}


def _mean(recs, key, field):
    return st.mean(r[field] for r in recs[key].values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--pilot-dir", type=Path, default=HERE / "outputs_pilot")
    ap.add_argument("--pilot-report", action="store_true",
                    help="print the amendment's numbers from the pilot records and stop")
    ap.add_argument("--stage", choices=("cont",), help="read only Gate 0(b) and stop")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="set only after the det re-run matches its full-run copy")
    args = ap.parse_args()
    if args.pilot_report:
        for key, val in pilot_report(args.pilot_dir).items():
            print(f"{key}: {val}")
        return
    out = args.out_dir
    g0b = stage_cont(out)
    if args.stage == "cont":
        print(f"GATE 0(b) J3V-X continuity with EXP-075 (d9, d11, seed 0): {g0b}")
        return
    recs = load_records(out)
    rates = rates_of(recs)
    viol = gate_d(recs)
    gg = gate_g(recs)
    rank = {d: [_read(out / ev.rank_record_name(d, s)) for s in SEEDS] for d in DEPTHS}
    rank75 = {d: [json.loads((ev.E75_OUT / f"exp075_rank_J-X_d{d}_s{s}.json").read_text())
                  for s in SEEDS] for d in DEPTHS}

    print(f"GATE 0(a) determinism: {'PASS' if args.determinism_ok else 'NOT CHECKED'}")
    print(f"GATE 0(b) J3V-X continuity with EXP-075 (d9, d11, seed 0): {g0b}")
    print(f"GATE D dominance (J3VG4-X >= J3V-X on every cell): "
          f"{'PASS' if not viol else 'FAIL at (depth, seed) ' + str(viol)}")
    print(f"GATE G (4-move goal check fired): {gg}")
    print("GATE R at k 4 (reported, not voiding): lowest-J 4-move leaf closer than root")
    for d in DEPTHS:
        ok, p = a71.gate_r_verdict(rank[d], "hit", "chance")
        print(f"  d{d}: hit {st.mean(r['hit'] for r in rank[d]):.4f} chance "
              f"{st.mean(r['chance'] for r in rank[d]):.4f} p {p:.4f} -> {ok}   "
              f"(EXP-075 k 3: hit {st.mean(r['hit'] for r in rank75[d]):.4f} chance "
              f"{st.mean(r['chance'] for r in rank75[d]):.4f})")

    print("\nMean held-out success (mean steps to solve, optimality)")
    for d in DEPTHS:
        print(f"  d{d}: " + "  ".join(
            f"{a} {_mean(recs, (a, d), 'success_rate'):.4f} "
            f"({_mean(recs, (a, d), 'mean_steps'):.2f}, {_mean(recs, (a, d), 'optimality'):.3f})"
            for a in ALL_ARMS))

    gate_ok = g0b == "PASS" and args.determinism_ok and not viol
    if not gate_ok:
        print("Gate 0 or Gate D did not pass. Every claim is VOID.")
    full = primary_verdicts(rates, gate_ok, SEEDS)
    sens_rates = sensitivity_rates(rates)
    sens = primary_verdicts(sens_rates, gate_ok, sorted(sens_rates[("J4V-X", 9)]))
    for name, label in (("claim1", "CLAIM 1 (primary) d11 J4V-X - J3V-X"),
                        ("claim2", "CLAIM 2 (primary) d9 J4V-X - J3V-X"),
                        ("claim3", "CLAIM 3 (primary) d11 J4V-X - J3VG4-X")):
        v, p, diff, am, bm = full[name]
        sv, sp, sdiff, _, _ = sens[name]
        print(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} -> {v}")
        print(f"    sensitivity, seeds 0 and 3 dropped: diff {sdiff:+.4f}, p {sp:.4f} -> {sv}")
        if sv != v:
            print("    SENSITIVITY DISAGREES WITH THE VERDICT: RESULTS.md must lead with this.")
    if gg[11] == "UNRESOLVED":
        print("    Claim 3 reads against a 4-move goal check that never fired at depth 11.")

    print("\nSecondary (patterns, never confirmations):")
    v, p, diff, am, bm = contrast(rates, ("J4V-X", 9), ("J3VG4-X", 9), gate_ok)
    print(f"  d9 J4V-X - J3VG4-X: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f}")
    for d in DEPTHS:
        _, p, diff, am, bm = contrast(rates, ("J3VG4-X", d), ("J3V-X", d), gate_ok)
        f4 = sum(recs[("J3VG4-X", d)][s]["fired4"] for s in SEEDS)
        print(f"  d{d} J3VG4-X - J3V-X (goal reach alone, >= 0 by construction): "
              f"diff {diff:+.4f}, summed fired4 {f4} -> Gate G {gg[d]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the whole file**

Run (Bash timeout 600000):
`PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp076.py -q`
Expected: 16 PASS.

- [ ] **Step 5: Mutation check (record in your report)**

Apply each mutation, run the file, confirm at least one test FAILS, then revert and run `git diff --stat`.
1. Remove the `CONFIRMED (SMALL)` branch.
2. In `gate_d`, compare against `recs[("J3V-X", d)][0]`, i.e. seed 0 every time.
3. In `gate_g`, read `fired4` from `J4V-X`.
4. In `primary_verdicts`, make Claim 3's baseline `("J3V-X", 11)`.
5. In `check_eval_record`, delete the `judge_updates` check.

- [ ] **Step 6: Commit**

```bash
git add experiments/076_deeper_tree/aggregate.py experiments/076_deeper_tree/evaluate.py tests/experiments/test_exp076.py
git commit -m "EXP-076: aggregator with Gates 0, D and 1 and CONFIRMED (SMALL) as verdicts, Gate G for the goal-check decomposition"
```

---

### Task 4: Laptop launcher

**Files:**
- Create: `experiments/076_deeper_tree/launch076.ps1`
- Test: `tests/experiments/test_exp076.py` (append)

**Interfaces:**
- Consumes: Task 2's `evaluate.py` CLI (`--arm A | --rank`, `--depth`, `--seed`, `--out-dir`, `--limit-states`), and Task 3's `aggregate.AMENDMENT`.
- Produces: `launch076.ps1 -Phase check|pilot|cont|rank|eval|det -Workers N [-SkipExisting]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/experiments/test_exp076.py`:

```python
def test_launcher_carries_the_fixes_and_the_registered_cells():
    """Catches a launcher without EXP-074's .Handle fix (every cell counted as failed), without
    the amendment guard, or with an eval list missing an arm or depth."""
    text = (EXP / "launch076.ps1").read_text()
    assert "$null = $proc.Handle" in text
    assert '"J4V-X", "J3VG4-X"' in text and "11, 9" in text
    assert "a.AMENDMENT" in text and "exp076-det" in text
    assert "--limit-states" in text and "outputs_pilot" in text
    assert "k_goal" in text  # the seam check refuses a stale checkout


def test_launcher_skips_by_the_phase_directory():
    """REVIEW FOCUS 4. Catches -SkipExisting checking outputs/ for pilot or det cells, so a
    relaunch after a Windows Update restart re-runs (or wrongly skips) them."""
    text = (EXP / "launch076.ps1").read_text()
    assert '$target = Join-Path $phaseDir $j[1]' in text
    assert '"pilot" { $pilotDir }' in text and '"det" { $detDir }' in text
```

- [ ] **Step 2: Run to verify they fail**

Run (Bash timeout 600000):
`PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp076.py -q -k launcher`
Expected: 2 FAIL with FileNotFoundError.

- [ ] **Step 3: Implement**

Create `experiments/076_deeper_tree/launch076.ps1`:

```powershell
# EXP-076 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase pilot -Workers 3
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase cont  -Workers 2
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase rank  -Workers N -SkipExisting
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase eval  -Workers N -SkipExisting
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase det   -Workers 1
#
# Order (spec section 9): pilot -> dated amendment (worker count, schedule; sets
# aggregate.AMENDMENT) -> cont (Gate 0(b)) -> rank -> eval -> det (Gate 0(a), separate dir).
# Every phase but check and pilot refuses to run while aggregate.AMENDMENT is None.
# Evaluation cells take -SkipExisting: re-run the same phase after a Windows Update restart.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Arm and seed lists therefore live in this file.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","pilot","cont","rank","eval","det")][string]$Phase,
    [int]$Workers = 0,
    [switch]$SkipExisting
)

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\076_deeper_tree"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# The k_goal seam must be present: a stale checkout fails here first.
$hasSeam = & $py -c "import sys, inspect; from neuromorphic.training import lookahead as l; sys.stdout.write(str('k_goal' in inspect.signature(l.evaluate_lookahead).parameters))"
if ($LASTEXITCODE -ne 0 -or $hasSeam -ne "True") { Write-Error "k_goal seam missing (got '$hasSeam'). Sync the repo."; exit 1 }

# EXP-075's 12 X judges at 4000 updates, and its 24 J3V-X records at depths 9 and 11.
$present = & $py -c "import json, sys; from pathlib import Path; o=Path('experiments/075_wide_region/outputs'); j=sum(json.loads((o/('judge_X_s'+str(s))/'state.json').read_text())['update']==4000 and (o/('judge_X_s'+str(s))/'judge.pt').exists() for s in range(12)); r=sum((o/('exp075_J3V-X_d'+str(d)+'_s'+str(s)+'.json')).exists() for d in (9, 11) for s in range(12)); sys.stdout.write(str(j)+' '+str(r))"
if ($LASTEXITCODE -ne 0 -or $present -ne "12 24") { Write-Error "expected 12 trained X judges and 24 J3V-X records, found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                 = $where"
"checkout                = $head"
"X judges, J3V-X records = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs                 = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

# Every phase but the pilot produces numbers that decide a claim, and none may run before the
# dated amendment. Check the exit code too: an import failure prints nothing.
if ($Phase -ne "pilot") {
    $amend = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','$($exp.Replace('\','/'))/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.AMENDMENT))"
    if ($LASTEXITCODE -ne 0 -or $amend -eq "") { Write-Error "could not read AMENDMENT (got '$amend'); refusing to run."; exit 1 }
    if ($amend -eq "None") { Write-Error "AMENDMENT is unset: commit the dated post-pilot amendment before this phase."; exit 1 }
    "amendment               = $amend"
}

if ($Workers -eq 0) { $Workers = 4 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$outDir   = Join-Path $repo "$exp\outputs"
$pilotDir = Join-Path $repo "$exp\outputs_pilot"
$detDir   = "C:\Users\mlgbr\exp076-det"
$phaseDir = switch ($Phase) { "pilot" { $pilotDir } "det" { $detDir } default { $outDir } }
$seeds = 0..11

# Each job is @(label, expected-output-file, args...). The longest cells go first.
$jobs = New-Object System.Collections.ArrayList
switch ($Phase) {
    "pilot" {
        foreach ($a in "J4V-X", "J3VG4-X") {
            [void]$jobs.Add(@("pilot $a d11 s0", "exp076_${a}_d11_s0.json", "--arm", $a, "--depth", 11, "--seed", 0, "--limit-states", 20)) }
        [void]$jobs.Add(@("pilot rank d11 s0", "exp076_rank_J-X4_d11_s0.json", "--rank", "--depth", 11, "--seed", 0, "--limit-states", 20))
    }
    "cont" {
        foreach ($d in 9, 11) {
            [void]$jobs.Add(@("J3V-X d$d s0", "exp076_J3V-X_d${d}_s0.json", "--arm", "J3V-X", "--depth", $d, "--seed", 0)) }
    }
    "rank" {
        foreach ($d in 11, 9) { foreach ($s in $seeds) {
            [void]$jobs.Add(@("rank d$d s$s", "exp076_rank_J-X4_d${d}_s${s}.json", "--rank", "--depth", $d, "--seed", $s)) } }
    }
    "eval" {
        foreach ($a in "J4V-X", "J3VG4-X") { foreach ($d in 11, 9) { foreach ($s in $seeds) {
            [void]$jobs.Add(@("$a d$d s$s", "exp076_${a}_d${d}_s${s}.json", "--arm", $a, "--depth", $d, "--seed", $s)) } } }
    }
    "det" {
        [void]$jobs.Add(@("J4V-X d9 s0", "exp076_J4V-X_d9_s0.json", "--arm", "J4V-X", "--depth", 9, "--seed", 0))
    }
}

"phase                   = $Phase, $($jobs.Count) cells, $Workers workers, into $phaseDir"
"log                     = $log"
""
$running = @()
$failed = 0
$stamp = { param($m) Add-Content -Path $log -Value $m; $m }
foreach ($j in $jobs) {
    $target = Join-Path $phaseDir $j[1]
    if ($SkipExisting -and (Test-Path $target)) { continue }
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Workers) { Start-Sleep -Seconds 5 }
    foreach ($p in @($running | Where-Object { $_.HasExited })) { if ($p.ExitCode -ne 0) { $failed++ } }
    $running = @($running | Where-Object { -not $_.HasExited })
    & $stamp "start $($j[0])" | Out-Null
    $argList = @("-u", "$exp\evaluate.py") + @($j[2..($j.Count - 1)] | ForEach-Object { "$_" }) + @("--out-dir", $phaseDir)
    # Read .Handle at once: without it Windows PowerShell 5.1 reports ExitCode as $null after the
    # process exits, and every cell counts as failed whatever it did (EXP-074, 2026-10-08).
    $proc = Start-Process -FilePath $py -ArgumentList $argList -PassThru -NoNewWindow -WorkingDirectory $repo
    $null = $proc.Handle
    $running += $proc
}
while (@($running | Where-Object { -not $_.HasExited }).Count -gt 0) { Start-Sleep -Seconds 5 }
foreach ($p in $running) { if ($p.ExitCode -ne 0) { $failed++ } }
if ($failed -gt 0) { Write-Error "$failed cells exited non-zero; do not aggregate."; exit 1 }
"PHASE OK"
```

`--out-dir` is appended to every job from `$phaseDir`, so the pilot's records land in `outputs_pilot`, det's in `C:\Users\mlgbr\exp076-det`, and the rest in `outputs`. A `$phaseDir` that contains spaces would need quoting. None of these paths do.

- [ ] **Step 4: Run the file**

Run (Bash timeout 600000):
`PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp076.py -q`
Expected: 18 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/076_deeper_tree/launch076.ps1 tests/experiments/test_exp076.py
git commit -m "EXP-076: laptop launcher with the amendment guard, k_goal seam check and per-phase output directories"
```

---

## After the tasks (controller, not an implementer)

1. Whole-branch review, then merge with `--no-ff` (`Merge exp076: EXP-076 deeper tree (code)`), push, and update the CLAUDE.md test count from `--collect-only`.
2. Sync the laptop, then run `-Phase check` and `-Phase pilot -Workers 3`. Sample `PrivateMemorySize64` of the `^python` processes during the J4V-X pilot cell.
3. Run `aggregate.py --pilot-report`. Write the dated amendment: spec section 11, plus `AMENDMENT = {"date": ..., "workers": N}` in `aggregate.py`. Show it to Michael before any full cell.
4. After approval: run cont (read with `--stage cont`), then rank, then eval, then det. Diff det against its full-run copy, then aggregate with `--determinism-ok`. Then write RESULTS.md, commit the records, and update the handoff and roadmap.
