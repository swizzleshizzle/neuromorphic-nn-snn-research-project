# EXP-071 Critic as Judge and No-Revisit Rule: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a critic scorer (C) and a no-revisit modifier (V) to the look-ahead procedure, a step-0 ranking check, and an EXP-071 harness and aggregator, so the experiment can be pre-flighted and launched on the laptop.

**Architecture:** `src/neuromorphic/training/lookahead.py` gains mode `"C"`, a visited-set mask, and a fallback counter, without changing any existing code path (EXP-070's tests pin that, and Gate 0(b) checks it against EXP-070's committed records). `experiments/071_critic_and_no_revisit/` reuses EXP-070's `cells.py` for loading and adds critic loading, the ranking check, a runner and an aggregator. Tasks 1 to 4 are implementer tasks on the VPS; Task 5 is the controller's laptop checklist.

**Tech Stack:** Python 3.10 venv, torch CPU, the existing `Brain`, pytest. No scipy.

**Spec:** `docs/superpowers/specs/2026-10-06-exp071-critic-and-no-revisit-design.md`. Read it, and `experiments/070_lookahead_existing/RESULTS.md`, before starting any task.

## Global Constraints

- Work ONLY in the worktree `/root/projects/.wt/exp-071` on branch `exp-071-critic`. Prefix every python/pytest command with `PYTHONPATH=src` (the venv's editable install points at main's `src`).
- Always pass an explicit Bash `timeout` (default is 120 s). Run tests in the FOREGROUND; 600000 ms covers every command here. Never run the whole suite.
- No em-dashes anywhere. Plain commit messages: no `Co-Authored-By`, no "Generated with". Stage explicit paths only.
- Action width from `env.action_space.n` or an `n_actions` argument, never a literal 6.
- `lookahead.py` must NEVER import or read `neuromorphic.envs.cube_distance`. The ranking check (Task 2) uses BFS as a yardstick and therefore lives in the experiment folder, not in `lookahead.py`.
- **Existing behaviour must not change.** All of `tests/training/test_lookahead.py` must still pass after every task, unmodified except for additions.
- Never write an assertion that cannot fail; each test's docstring names the bug it catches. Never weaken a threshold to make something land.
- Arms are exactly `G0 G0V E3 E3V P3 P3V C1 C1V C3 C3V R3 R3V`, depth 7, seeds 0..11.

---

## Step 0 (controller, before Task 1)

```bash
cd /root/projects/neuromorphic-nn-snn-research-project
git worktree add /root/projects/.wt/exp-071 -b exp-071-critic
ln -s /root/projects/neuromorphic-nn-snn-research-project/.venv /root/projects/.wt/exp-071/.venv
```

`.venv` is already in `.git/info/exclude`.

## File Structure

| file | responsibility |
|---|---|
| `src/neuromorphic/training/lookahead.py` (modify) | mode C, `imagined_values`, `sequence_eligibility`, V in `choose_move` and `evaluate_lookahead` |
| `tests/training/test_lookahead.py` (modify, additions only) | C and V unit tests |
| `experiments/071_critic_and_no_revisit/cells.py` (create) | arm list, names, critic loading; reuses EXP-070 `cells.py` |
| `experiments/071_critic_and_no_revisit/rank.py` (create) | step-0 ranking check, one record per seed |
| `experiments/071_critic_and_no_revisit/run.py` (create) | runner, one record per cell |
| `experiments/071_critic_and_no_revisit/aggregate.py` (create) | Gates 0(b), R, V, 1; Claims 1 and 2 at alpha 0.025; secondaries |
| `experiments/071_critic_and_no_revisit/launch071.ps1` (create) | laptop launcher with phases |
| `tests/experiments/test_exp071.py` (create) | cells, rank, runner and aggregator tests |

---

### Task 1: Mode C and the V modifier in the procedure

**Files:**
- Modify: `src/neuromorphic/training/lookahead.py`
- Test: `tests/training/test_lookahead.py` (append only)

**Interfaces:**
- Produces:
  - `MODES = ("G", "E", "P", "R", "C")`
  - `imagined_values(agent, critic, states, *, generator) -> torch.Tensor` of shape `[len(states)]`: one batched `agent.step(np.array(states), recall=False, generator=generator)` under `torch.no_grad()`, then `critic(out["concept"].mean(dim=0)).squeeze(-1)`.
  - `sequence_eligibility(levels, k, n_actions, visited) -> torch.Tensor` (bool, `[n_actions**k]`): sequence `j` is eligible iff for every `l` in `1..k`, `levels[l][j // n_actions**(k-l)]` is not in `visited`.
  - `choose_move(mode, state, root_logits, k, n_actions, *, agent=None, head=None, imag_generator=None, critic=None, visited=None, info=None) -> tuple[int, bool]`. Unchanged when `critic` and `visited` are None. If `info` is a dict, sets `info["fallback"] = True` when V excluded every candidate and the unmasked choice was used.
  - `evaluate_lookahead(..., critic=None, no_revisit=False)` adds record keys `no_revisit` (bool) and `fallback_frac` (fallback moves / real moves). Every existing key keeps its exact computation.

**V semantics (spec section 3):**
- G with `visited`: eligible moves are those whose resulting state is not in `visited`; play the highest-logit eligible move (first maximum); if none, plain greedy and fallback.
- E with `visited`: goal test first (unchanged); otherwise exactly G's V rule.
- P, C, R with `visited`: compute scores as without V, then set ineligible entries to `-inf` if at least one is eligible; otherwise leave scores unmasked and fallback.
- R must draw `torch.rand(n_actions ** k, generator=imag_generator)` exactly as before, BEFORE any masking, so R3 and R3V see the same random numbers.

- [ ] **Step 0: Capture the golden fixture BEFORE touching `lookahead.py`**

This is what makes the continuity check able to fail: every later comparison is against numbers
produced by the UNMODIFIED code. Run from the worktree root (timeout 600000):

```bash
cd /root/projects/.wt/exp-071 && git diff --quiet src/ && mkdir -p tests/fixtures && PYTHONPATH=src .venv/bin/python - <<'PY'
import importlib.util, json, torch
torch.set_num_threads(1)
s = importlib.util.spec_from_file_location("c", "experiments/070_lookahead_existing/cells.py")
c = importlib.util.module_from_spec(s); s.loader.exec_module(c)
from neuromorphic.training.lookahead import evaluate_lookahead
agent, head, states, ts = c.load_cell(7, 0)
FIELDS = ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
          "greedy_modal_action_frac", "goal_fired_frac")
out = {}
for mode, k in (("G", 0), ("E", 3), ("P", 3), ("R", 3)):
    r = evaluate_lookahead(agent, head, states[:3], depth=7, mode=mode, k=k,
                           generator=torch.Generator().manual_seed(ts), rng_seed=ts, imag_seed=ts)
    out[f"{mode}{k}"] = {f: r[f] for f in FIELDS}
open("tests/fixtures/exp070_d7_s0_golden.json", "w").write(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(out)
PY
```

Then add `tests/training/test_lookahead_golden.py`:

```python
"""Continuity with EXP-070: the C and V changes must not move any existing arm by one bit.

The fixture was produced by the UNMODIFIED lookahead.py (EXP-071 plan, Task 1 Step 0). Compare,
never regenerate: regenerating from modified code would make this test unable to fail.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neuromorphic.training.lookahead import evaluate_lookahead

REPO = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((REPO / "tests" / "fixtures" / "exp070_d7_s0_golden.json").read_text())
_s = importlib.util.spec_from_file_location(
    "c70", REPO / "experiments" / "070_lookahead_existing" / "cells.py")
c70 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(c70)


@pytest.mark.parametrize("arm,mode,k", [("G0", "G", 0), ("E3", "E", 3), ("P3", "P", 3),
                                        ("R3", "R", 3)])
def test_existing_arms_match_the_pre_change_golden_fixture(arm, mode, k):
    """Catches any change to the shared code path (stream draws, scoring, budget, metrics)."""
    agent, head, states, ts = c70.load_cell(7, 0)
    r = evaluate_lookahead(agent, head, states[:3], depth=7, mode=mode, k=k,
                           generator=torch.Generator().manual_seed(ts), rng_seed=ts, imag_seed=ts)
    for field, value in GOLDEN[arm].items():
        assert r[field] == value, (arm, field)
```

Run it (timeout 600000); it must PASS on the unmodified code. Commit the fixture and the test on
their own, before Step 1:

```bash
git add tests/fixtures/exp070_d7_s0_golden.json tests/training/test_lookahead_golden.py
git commit -m "EXP-071: golden fixture of EXP-070's arms, captured before the procedure changes"
```

- [ ] **Step 1: Write the failing tests** (append to `tests/training/test_lookahead.py`)

```python
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


def test_r_draws_the_same_random_scores_with_and_without_v():
    """Catches V changing R's stream: with nothing visited, R3 and R3V must choose alike."""
    s = _far_state()
    root = torch.zeros(N_ACTIONS)
    a = la.choose_move("R", s, root, 3, N_ACTIONS,
                       imag_generator=torch.Generator().manual_seed(4))
    b = la.choose_move("R", s, root, 3, N_ACTIONS,
                       imag_generator=torch.Generator().manual_seed(4), visited=set())
    assert a == b


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
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /root/projects/.wt/exp-071 && PYTHONPATH=src .venv/bin/python -m pytest tests/training/test_lookahead.py -q` (timeout 600000)
Expected: the new tests FAIL (unknown mode `C`, missing `sequence_eligibility`, unexpected kwargs); every pre-existing test still PASSES.

- [ ] **Step 3: Implement**

Replace `MODES`, add the two helpers, and replace `choose_move` and `evaluate_lookahead` with:

```python
MODES = ("G", "E", "P", "R", "C")


def imagined_values(agent, critic, states, *, generator):
    """The critic's value for many states, in ONE batched brain call, recall off."""
    with torch.no_grad():
        out = agent.step(np.array(states), recall=False, generator=generator)
        return critic(out["concept"].mean(dim=0)).squeeze(-1)


def sequence_eligibility(levels, k, n_actions, visited):
    """True where a length-k sequence passes through no state in `visited`."""
    j = torch.arange(n_actions ** k)
    ok = torch.ones(n_actions ** k, dtype=torch.bool)
    for l in range(1, k + 1):
        bad = torch.tensor([s in visited for s in levels[l]], dtype=torch.bool)
        ok &= ~bad[j // n_actions ** (k - l)]
    return ok


def _greedy_unvisited(state, root_logits, n_actions, visited, info):
    eligible = [a for a in range(n_actions) if apply_move(state, a) not in visited]
    if not eligible:
        if info is not None:
            info["fallback"] = True
        return int(root_logits.argmax())
    return max(eligible, key=lambda a: (float(root_logits[a]), -a))


def choose_move(mode, state, root_logits, k, n_actions, *, agent=None, head=None,
                imag_generator=None, critic=None, visited=None, info=None):
    """One real move. Returns (action, goal_fired). Never sees the evaluation generator."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    greedy = int(root_logits.argmax())
    if mode == "G":
        if visited is not None:
            return _greedy_unvisited(state, root_logits, n_actions, visited, info), False
        return greedy, False
    if k < 1:
        raise ValueError(f"mode {mode} needs k >= 1, got {k}")
    prefix = solving_prefix(state, k, n_actions)
    if prefix is not None:
        return prefix[0], True
    if mode == "E":
        if visited is not None:
            return _greedy_unvisited(state, root_logits, n_actions, visited, info), False
        return greedy, False
    levels = None
    if mode == "R":
        scores = torch.rand(n_actions ** k, generator=imag_generator)
    elif mode == "C":
        if critic is None:
            raise ValueError("mode C needs a critic")
        levels = tree_levels(state, k, n_actions)
        scores = imagined_values(agent, critic, levels[k], generator=imag_generator)
    else:  # "P"
        levels = tree_levels(state, k, n_actions)
        level_logp = [torch.log_softmax(root_logits.reshape(1, -1), dim=-1)]
        for l in range(1, k):
            level_logp.append(imagined_logp(agent, head, levels[l], generator=imag_generator))
        scores = sequence_scores(level_logp, k, n_actions)
    if visited is not None:
        if levels is None:
            levels = tree_levels(state, k, n_actions)
        mask = sequence_eligibility(levels, k, n_actions, visited)
        if bool(mask.any()):
            scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
        elif info is not None:
            info["fallback"] = True
    best = int(scores.argmax())  # first maximum, so ties go to the lowest product index
    return best // n_actions ** (k - 1), False
```

In `evaluate_lookahead`, add `critic=None, no_revisit=False` to the signature; keep a `visited_set = set(visited)` alongside the existing `visited` list (add each new state to both); count `fallback_moves`; and change the `choose_move` call to:

```python
            info = {}
            action, fired = choose_move(mode, env._state, logits, k, n_actions,
                                        agent=agent, head=head, imag_generator=imag,
                                        critic=critic,
                                        visited=visited_set if no_revisit else None,
                                        info=info)
            fallback_moves += int(info.get("fallback", False))
```

and add to the returned dict:

```python
        "no_revisit": bool(no_revisit),
        "fallback_frac": (fallback_moves / eval_steps) if eval_steps else 0.0,
```

Initialise `fallback_moves = 0` with the other counters. Nothing else in the function changes.

- [ ] **Step 4: Run to verify pass**

Same command, plus `tests/training/test_lookahead_golden.py`. Expected: every test passes, the
golden test included (EXP-070's tests unchanged).

- [ ] **Step 5: Mutation check**

| mutation | test that must fail |
|---|---|
| C scores `levels[1]` instead of `levels[k]` | `test_c3_scores_sequences_by_their_leaf_not_their_first_child` |
| `sequence_eligibility` checks only `l == k` | `test_sequence_eligibility_excludes_any_sequence_through_a_visited_state` |
| in `evaluate_lookahead`, pass `visited=None` always | `test_no_revisit_cuts_the_revisit_rate_of_a_looping_reflex` |
| `_greedy_unvisited` returns `int(root_logits.argmax())` unconditionally | `test_g_with_v_plays_the_best_unvisited_move` |
| R branch draws `torch.rand` AFTER building `levels` with an extra `torch.rand(1, generator=imag_generator)` | `test_r_draws_the_same_random_scores_with_and_without_v` (only if that extra draw is under `visited is not None`; place it there) |
| fallback branch no longer sets `info["fallback"]` | `test_v_falls_back_and_says_so_when_every_candidate_is_visited` |
| in the R branch, add `torch.rand(1, generator=imag_generator)` before the real draw (unconditionally) | `test_existing_arms_match_the_pre_change_golden_fixture[R3]` |

Revert after each with `git checkout -- src/neuromorphic/training/lookahead.py`; confirm `git diff --stat` shows only the test file. If a mutation survives, strengthen the test until it fails, then re-run the table. Record outcomes in the commit body.

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/training/lookahead.py tests/training/test_lookahead.py
git commit -m "EXP-071: critic scorer C and the no-revisit modifier V in the look-ahead procedure"
```

---

### Task 2: EXP-071 cells and the step-0 ranking check

**Files:**
- Create: `experiments/071_critic_and_no_revisit/cells.py`
- Create: `experiments/071_critic_and_no_revisit/rank.py`
- Create: `experiments/071_critic_and_no_revisit/outputs/.gitkeep`
- Test: `tests/experiments/test_exp071.py`

**Interfaces:**
- Consumes: EXP-070 `cells.py` (`load_cell`, `published_config`, `REPO`); Task 1's `imagined_values`, `imagined_logp`, `imag_seed_for`.
- Produces (`cells.py`): `DEPTH = 7`, `SEEDS = tuple(range(12))`, `ARMS` = list of `(mode, k, v)` in the order `G0 G0V E3 E3V P3 P3V C1 C1V C3 C3V R3 R3V`; `arm_name(mode, k, v) -> str` (e.g. `"C3V"`); `parse_arm(text) -> (mode, k, v)` raising `SystemExit` on anything not in `ARMS`; `critic_path(seed) -> Path`; `load_critic(seed) -> nn.Linear`; `cell_record_name(mode, k, v, seed) -> str` = `f"exp071_{arm_name(mode, k, v)}_d7_s{seed}.json"`.
- Produces (`rank.py`): `rank_state(agent, head, critic, state, n_actions, provider, generator) -> dict` with keys `critic_hit`, `policy_hit`, `chance` (child level, spec R1) and `leaf_closer_hit`, `leaf_closer_chance`, `leaf_d3_hit`, `leaf_d3_chance` (leaf level, spec R3); `rank_seed(seed, out_dir) -> dict` writing `exp071_rank_d7_s{seed}.json` with the per-seed mean of each of those seven keys plus `seed` and `n`.

`rank_state`, in this draw order on the one generator: `logp = imagined_logp(agent, head, [state])[0]`, then `values = imagined_values(agent, critic, children)` for `children = [apply_move(state, a) for a in range(n_actions)]`, then `leaf_values = imagined_values(agent, critic, leaves)` for `leaves = tree_levels(state, 3, n_actions)[3]`. With `d = provider.distance(state)`: child level as before (`improving = [dist(c) == d - 1]`, `critic_hit`, `policy_hit`, `chance = mean(improving)`); leaf level `leaf_d = [dist(l) for l in leaves]`, `top = int(leaf_values.argmax())`, `leaf_closer_hit = int(leaf_d[top] < d)`, `leaf_closer_chance = mean(x < d for x in leaf_d)`, `leaf_d3_hit = int(leaf_d[top] == d - 3)`, `leaf_d3_chance = mean(x == d - 3 for x in leaf_d)`. `rank_seed` uses `ExactBFSDistance(max_depth=DEPTH + 3)` (leaves reach depth 10) and a generator seeded `imag_seed_for(train_seed, i, 0)` per held-out state `i` (step 0 is never used by rollouts, whose steps start at 1). Spec pre-flight chance values at depth 7: child 0.2203, leaf-closer 0.1344, leaf-at-d-3 0.0082.

- [ ] **Step 1: Write the failing tests**

```python
"""EXP-071: cells, ranking check, runner and aggregator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.cube_baseline import shell_states

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "071_critic_and_no_revisit"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp071_cells", "cells.py")
rank = _load("exp071_rank", "rank.py")


def test_arms_are_exactly_the_spec_list():
    """Catches a missing or extra arm."""
    names = [cells.arm_name(*a) for a in cells.ARMS]
    assert names == ["G0", "G0V", "E3", "E3V", "P3", "P3V", "C1", "C1V", "C3", "C3V", "R3", "R3V"]


@pytest.mark.parametrize("text,expected", [("C3V", ("C", 3, True)), ("G0", ("G", 0, False))])
def test_parse_arm_round_trips(text, expected):
    assert cells.parse_arm(text) == expected


def test_parse_arm_rejects_arms_outside_the_spec():
    """Catches P1 or C2 sneaking in through the CLI."""
    for bad in ("P1", "C2", "G1V"):
        with pytest.raises(SystemExit):
            cells.parse_arm(bad)


@pytest.mark.parametrize("seed", cells.SEEDS)
def test_every_critic_is_tracked_and_loads(seed):
    """Catches a missing critic or a shape mismatch with the 64-wide concept."""
    critic = cells.load_critic(seed)
    assert critic.weight.shape == (1, 64)


class _Stub:
    """Concept [v, 0] per state; a critic reading coordinate 0; a head preferring move 0."""

    def __init__(self, values):
        self.values = {tuple(k): v for k, v in values.items()}

    def step(self, obs, *, recall=False, generator=None, **_):
        rows = np.asarray(obs)
        rows = rows[None, :] if rows.ndim == 1 else rows
        return {"concept": torch.tensor([[float(self.values.get(tuple(r), 0.0)), 0.0]
                                         for r in rows.tolist()]).unsqueeze(0)}


def _depth3_state_and_moves():
    provider = ExactBFSDistance(max_depth=4)
    s = shell_states(provider, 3)[0]
    good = [a for a in range(N_ACTIONS) if provider.distance(apply_move(s, a)) == 2]
    bad = [a for a in range(N_ACTIONS) if provider.distance(apply_move(s, a)) == 4]
    return provider, s, good, bad


def test_rank_state_scores_a_critic_that_prefers_an_improving_child_as_a_hit():
    """Catches the hit being computed on the wrong side (distance + 1) or on the wrong argmax."""
    provider, s, good, bad = _depth3_state_and_moves()
    agent = _Stub({apply_move(s, good[0]): 5.0})
    critic = nn.Linear(2, 1)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[1.0, 0.0]]))
        critic.bias.zero_()
    head = nn.Linear(2, N_ACTIONS)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[bad[0]] = 5.0
    r = rank.rank_state(agent, head, critic, s, N_ACTIONS, provider, torch.Generator())
    assert r["critic_hit"] == 1 and r["policy_hit"] == 0
    assert r["chance"] == pytest.approx(len(good) / N_ACTIONS)


def test_rank_state_scores_a_critic_that_prefers_a_worsening_child_as_a_miss():
    """Catches a hit rate that cannot fail (e.g. always 1)."""
    provider, s, good, bad = _depth3_state_and_moves()
    agent = _Stub({apply_move(s, bad[0]): 5.0})
    critic = nn.Linear(2, 1)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[1.0, 0.0]]))
        critic.bias.zero_()
    head = nn.Linear(2, N_ACTIONS)
    r = rank.rank_state(agent, head, critic, s, N_ACTIONS, provider, torch.Generator())
    assert r["critic_hit"] == 0


def _leaf_case(want_closer):
    from neuromorphic.training.lookahead import tree_levels
    provider = ExactBFSDistance(max_depth=6)
    s = shell_states(provider, 3)[0]
    leaves = tree_levels(s, 3, N_ACTIONS)[3]
    target = next(l for l in leaves if (provider.distance(l) < 3) == want_closer)
    agent = _Stub({target: 5.0})
    critic = nn.Linear(2, 1)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[1.0, 0.0]]))
        critic.bias.zero_()
    head = nn.Linear(2, N_ACTIONS)
    return rank.rank_state(agent, head, critic, s, N_ACTIONS, provider, torch.Generator())


def test_leaf_check_scores_a_closer_top_leaf_as_a_hit():
    """Catches the leaf check reading children instead of leaves, or comparing to d + 3."""
    assert _leaf_case(True)["leaf_closer_hit"] == 1


def test_leaf_check_scores_a_farther_top_leaf_as_a_miss():
    """Catches a leaf hit that cannot fail."""
    r = _leaf_case(False)
    assert r["leaf_closer_hit"] == 0 and 0.0 < r["leaf_closer_chance"] < 1.0
```

- [ ] **Step 2: Run to verify failure**

`cd /root/projects/.wt/exp-071 && PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp071.py -q` (timeout 300000). Expected: ERROR, files missing.

- [ ] **Step 3: Implement `cells.py`**

```python
"""EXP-071 cells: EXP-070's depth-7 cells plus each seed's EXP-053 arm B critic.

Spec: docs/superpowers/specs/2026-10-06-exp071-critic-and-no-revisit-design.md
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location(
    "exp070_cells", REPO / "experiments" / "070_lookahead_existing" / "cells.py")
c70 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c70)

DEPTH = 7
SEEDS = tuple(range(12))
ARMS = [(m, k, v) for (m, k) in (("G", 0), ("E", 3), ("P", 3), ("C", 1), ("C", 3), ("R", 3))
        for v in (False, True)]
CRITIC_DIR = REPO / "experiments" / "053_neuromod_stage3" / "outputs"


def arm_name(mode: str, k: int, v: bool) -> str:
    return f"{mode}{k}{'V' if v else ''}"


def parse_arm(text: str):
    for arm in ARMS:
        if arm_name(*arm) == text:
            return arm
    raise SystemExit(f"unknown arm {text!r}; valid: {[arm_name(*a) for a in ARMS]}")


def critic_path(seed: int) -> Path:
    return CRITIC_DIR / f"exp053_critic_d7_regionalized_d7_s{seed}_sig0.0_critic.pt"


def load_critic(seed: int) -> nn.Linear:
    critic = nn.Linear(64, 1)
    critic.load_state_dict(torch.load(critic_path(seed), map_location="cpu"))
    critic.eval()
    return critic


def cell_record_name(mode: str, k: int, v: bool, seed: int) -> str:
    return f"exp071_{arm_name(mode, k, v)}_d{DEPTH}_s{seed}.json"


def load_cell(seed: int):
    """(agent, head, held-out states, train seed), exactly EXP-070's depth-7 cell."""
    return c70.load_cell(DEPTH, seed)
```

Note: `nn.Linear(64, 1)` hardcodes the concept width EXP-053 trained with; the test pins it against every tracked critic.

- [ ] **Step 4: Implement `rank.py`**

```python
"""EXP-071 step 0: can the critic rank a state's children? (spec section 4)

The BFS table is the YARDSTICK here and nowhere in the procedure, which is why this lives in the
experiment folder and not in lookahead.py.

Usage: .venv/bin/python -u experiments/071_critic_and_no_revisit/rank.py --workers 12
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from neuromorphic.envs.cube import apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.lookahead import (
    imag_seed_for, imagined_logp, imagined_values, tree_levels,
)

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp071_cells", HERE / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)

torch.set_num_threads(1)


def rank_state(agent, head, critic, state, n_actions, provider, generator) -> dict:
    children = [apply_move(state, a) for a in range(n_actions)]
    leaves = tree_levels(state, 3, n_actions)[3]
    logp = imagined_logp(agent, head, [state], generator=generator)[0]
    values = imagined_values(agent, critic, children, generator=generator)
    leaf_values = imagined_values(agent, critic, leaves, generator=generator)
    d = provider.distance(state)
    improving = [provider.distance(c) == d - 1 for c in children]
    leaf_d = [provider.distance(l) for l in leaves]
    top = int(leaf_values.argmax())
    return {
        "critic_hit": int(improving[int(values.argmax())]),
        "policy_hit": int(improving[int(logp.argmax())]),
        "chance": sum(improving) / n_actions,
        "leaf_closer_hit": int(leaf_d[top] < d),
        "leaf_closer_chance": sum(x < d for x in leaf_d) / len(leaf_d),
        "leaf_d3_hit": int(leaf_d[top] == d - 3),
        "leaf_d3_chance": sum(x == d - 3 for x in leaf_d) / len(leaf_d),
    }


def rank_seed(seed: int, out_dir: Path) -> dict:
    torch.set_num_threads(1)
    agent, head, states, train_seed = cells.load_cell(seed)
    critic = cells.load_critic(seed)
    provider = ExactBFSDistance(max_depth=cells.DEPTH + 3)
    n_actions = head.head.out_features
    rows = [rank_state(agent, head, critic, s, n_actions, provider,
                       torch.Generator().manual_seed(imag_seed_for(train_seed, i, 0)))
            for i, s in enumerate(states)]
    rec = {"seed": seed, "n": len(rows)}
    for key in rows[0]:
        rec[key] = st.mean(r[key] for r in rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp071_rank_d{cells.DEPTH}_s{seed}.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(rank_seed, args.seeds, [args.out_dir] * len(args.seeds)):
            print(f"  s{r['seed']}: child critic {r['critic_hit']:.4f} policy {r['policy_hit']:.4f} "
                  f"chance {r['chance']:.4f} | leaf-closer {r['leaf_closer_hit']:.4f} "
                  f"chance {r['leaf_closer_chance']:.4f}", flush=True)


if __name__ == "__main__":
    main()
```

`head.head.out_features` reads the action width from the loaded `AblatedConcept`'s inner `nn.Linear`; verify the attribute name in `src/neuromorphic/analysis/ablate.py` (it stores the wrapped head as `self.head`).

- [ ] **Step 5: Run to verify pass**, then `mkdir -p experiments/071_critic_and_no_revisit/outputs && touch experiments/071_critic_and_no_revisit/outputs/.gitkeep`.

- [ ] **Step 6: Smoke one seed on 10 states (read the real output)**

```bash
cd /root/projects/.wt/exp-071 && PYTHONPATH=src .venv/bin/python - <<'EOF'
import importlib.util, torch
s = importlib.util.spec_from_file_location("r", "experiments/071_critic_and_no_revisit/rank.py")
r = importlib.util.module_from_spec(s); s.loader.exec_module(r)
from neuromorphic.envs.cube_distance import ExactBFSDistance
agent, head, states, ts = r.cells.load_cell(0)
critic = r.cells.load_critic(0)
p = ExactBFSDistance(max_depth=10)
print([r.rank_state(agent, head, critic, x, 6, p, torch.Generator().manual_seed(i)) for i, x in enumerate(states[:10])])
EOF
```

(timeout 600000). Paste the output in the report. Do NOT run `rank_seed` over all 200 states or all seeds: that is the laptop's step 0, and its numbers decide a gate.

- [ ] **Step 7: Mutation check**

| mutation | test that must fail |
|---|---|
| `improving = [... == d + 1 ...]` | `test_rank_state_scores_a_critic_that_prefers_an_improving_child_as_a_hit` |
| `critic_hit` computed from `logp.argmax()` | both child-level rank tests |
| `top = int(values.argmax())` (children instead of leaves) | `test_leaf_check_scores_a_closer_top_leaf_as_a_hit` |
| `parse_arm` accepts any `mode+k` | `test_parse_arm_rejects_arms_outside_the_spec` |

Revert after each; confirm clean. Strengthen any survivor. Record outcomes in the commit body.

- [ ] **Step 8: Commit**

```bash
git add experiments/071_critic_and_no_revisit/cells.py experiments/071_critic_and_no_revisit/rank.py experiments/071_critic_and_no_revisit/outputs/.gitkeep tests/experiments/test_exp071.py
git commit -m "EXP-071: cells, critic loading, and the step-0 ranking check"
```

---

### Task 3: The runner and launcher

**Files:**
- Create: `experiments/071_critic_and_no_revisit/run.py`
- Create: `experiments/071_critic_and_no_revisit/launch071.ps1`
- Test: `tests/experiments/test_exp071.py` (append)

**Interfaces:**
- Produces: `run_cell(mode, k, v, seed, out_dir, limit_states=None) -> dict`, writing `cells.cell_record_name(...)` with every `evaluate_lookahead` key plus `depth, seed, arm, wall_s, git_commit, limit_states, head_file, encoder_file, critic_file`. Passes `critic=cells.load_critic(seed)` for every arm (harmless where unused) and `no_revisit=v`. CLI: `--arms`, `--seeds`, `--workers`, `--out-dir`, `--skip-existing`, `--limit-states`.

- [ ] **Step 1: Failing test** (append)

```python
run = _load("exp071_run", "run.py")


def test_run_cell_writes_a_complete_record(tmp_path):
    """Catches a record missing a field the aggregator or Gate V reads."""
    rec = run.run_cell("C", 1, True, 0, tmp_path, limit_states=2)
    on_disk = json.loads((tmp_path / cells.cell_record_name("C", 1, True, 0)).read_text())
    assert on_disk == rec
    for key in ("solved", "n", "success_rate", "eval_revisit_rate", "fallback_frac",
                "no_revisit", "goal_fired_frac", "mode", "k", "arm", "seed", "critic_file",
                "limit_states"):
        assert key in rec, key
    assert rec["arm"] == "C1V" and rec["no_revisit"] is True and rec["n"] == 2


GOLDEN = json.loads((REPO / "tests" / "fixtures" / "exp070_d7_s0_golden.json").read_text())


@pytest.mark.parametrize("arm", ["G0", "E3", "P3", "R3"])
def test_runner_plain_arms_match_the_pre_change_golden_fixture(arm, tmp_path):
    """THE GATE 0(b) CODE PATH, through the runner. Catches run_cell passing something that
    moves a plain arm (e.g. a critic changing a non-C arm, or no_revisit defaulting on). The
    fixture came from the UNMODIFIED code, so this can fail."""
    mode, k, v = cells.parse_arm(arm)
    rec = run.run_cell(mode, k, v, 0, tmp_path, limit_states=3)
    for field, value in GOLDEN[arm].items():
        assert rec[field] == value, (arm, field)
```

- [ ] **Step 2: Verify failure.** Same test command (timeout 600000).

- [ ] **Step 3: Implement `run.py`** as EXP-070's `run.py` (read `experiments/070_lookahead_existing/run.py` and mirror it), with these differences: import `cells` from this folder; `run_cell(mode, k, v, seed, out_dir, limit_states=None)` calls `cells.load_cell(seed)` and `cells.load_critic(seed)`, then

```python
    res = evaluate_lookahead(agent, head, states, depth=cells.DEPTH, mode=mode, k=k,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=v)
```

and adds `"arm": cells.arm_name(mode, k, v)` and `"critic_file": str(cells.critic_path(seed).relative_to(cells.REPO))` to the record. `main()` parses `--arms` with `cells.parse_arm`, defaults to every arm, and builds jobs `(mode, k, v, seed)`.

- [ ] **Step 4: Implement `launch071.ps1`** by copying `experiments/070_lookahead_existing/launch070.ps1` and changing: the module check expects `G,E,P,R,C`; the presence check counts 12 heads (depth 7), 12 E1 encoders and 12 critics via this folder's `cells.py`; phases are `check`, `rank` (runs `rank.py --workers 12`), `base` (`G0 E3 P3 R3`), `det` (re-runs `G0V C3 R3V` seed 0 into `C:\Users\mlgbr\exp071-det`), `calib` (`C3 C3V`, seed 0, workers 2), and `full` (`G0V E3V P3V C1 C1V C3 C3V R3V`). BOTH `calib` and `full` are gated on `GATE_R1_PASSED` and `GATE_R3_PASSED` in `aggregate.py` not being `None`, exactly as launch070 gates `p` on `GATE0_FORM`: no C cell may run before the gate amendment is committed. Keep every warning comment from launch070 (no backticks, no commas across ssh, PYTHONPATH check).

- [ ] **Step 5: Verify pass**; then measure one C3 cell's cost on the VPS: `run.py --arms C3 --seeds 0 --limit-states 3 --workers 1 --out-dir /root/scratch/exp071-smoke` (timeout 600000) and report `wall_s` and the real-move count.

- [ ] **Step 6: Commit**

```bash
git add experiments/071_critic_and_no_revisit/run.py experiments/071_critic_and_no_revisit/launch071.ps1 tests/experiments/test_exp071.py
git commit -m "EXP-071: runner and laptop launcher"
```

---

### Task 4: The aggregator

**Files:**
- Create: `experiments/071_critic_and_no_revisit/aggregate.py`
- Test: `tests/experiments/test_exp071.py` (append)

**Interfaces:**
- Consumes: EXP-070 `aggregate.py` (import `one_sided_p`, `gate1_verdict` via importlib; do not copy them).
- Produces: `ALPHA = 0.025`; `GATE_R1_PASSED: bool | None = None` and `GATE_R3_PASSED: bool | None = None` (set by the controller in the dated amendment after step 0, never by code); `V_RATIO = 0.5`; `OUTCOME_FIELDS = ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate", "greedy_modal_action_frac", "goal_fired_frac")`;
  - `gate0b_verdict(recs071: dict, recs070: dict) -> str` ("PASS"/"FAIL"): for arms G0, E3, P3, R3 and every seed, every `OUTCOME_FIELDS` value equal to EXP-070's depth-7 record of the same arm and seed.
  - `gate_r_verdict(rank_rows: list[dict], hit_key: str = "critic_hit", chance_key: str = "chance") -> tuple[bool, float]`: `(p < 0.05 and mean > 0, p)` on per-seed `row[hit_key] - row[chance_key]`. R1 uses the defaults; R3 uses `("leaf_closer_hit", "leaf_closer_chance")`.
  - `gate_v_verdict(g0v_revisit: float, g0_revisit: float) -> bool`: `g0v_revisit <= V_RATIO * g0_revisit`.
  - `claim_verdict(diffs, a_mean, b_mean, gate_ok) -> tuple[str, float]`: `VOID` if not `gate_ok`; then `UNRESOLVED` per `gate1_verdict`; `REFUTED` if mean <= 0; `CONFIRMED` if p < ALPHA; else `NOT SIGNIFICANT`.
  - `main()` prints the success and revisit tables (12 arms), every gate, Claim 1 (C3 - E3, gated on R3), Claim 2 (G0V - G0, gated on V), secondaries (C3 - P3 and C3V - C3 gated on R3; C1 - G0 gated on R1; E3V - E3, P3V - P3 ungated), and fallback fractions. Refuses every claim if Gate 0(b) fails or either `GATE_R1_PASSED` or `GATE_R3_PASSED` is None. It recomputes R1 and R3 from the rank records and refuses to run if they disagree with the committed constants.

- [ ] **Step 1: Failing tests** (append)

```python
agg = _load("exp071_aggregate", "aggregate.py")


def test_claim_verdict_ladder_including_void():
    """Catches VOID being checked after significance, and the 0.025 alpha being 0.05."""
    assert agg.claim_verdict([0.05] * 12, 0.2, 0.25, gate_ok=False)[0] == "VOID"
    assert agg.claim_verdict([0.05] * 12, 0.2, 0.25, gate_ok=True)[0] == "CONFIRMED"
    assert agg.claim_verdict([-0.01] * 12, 0.2, 0.19, gate_ok=True)[0] == "REFUTED"
    # p for 9 positive of 12 equal-magnitude diffs is 299/4096 = 0.073: NOT SIGNIFICANT at
    # either alpha. 10 of 12 is 79/4096 = 0.0193: CONFIRMED at 0.025.
    nine = [0.01] * 9 + [-0.01] * 3
    ten = [0.01] * 10 + [-0.01] * 2
    assert agg.claim_verdict(nine, 0.2, 0.205, gate_ok=True)[0] == "NOT SIGNIFICANT"
    assert agg.claim_verdict(ten, 0.2, 0.207, gate_ok=True)[0] == "CONFIRMED"


def test_alpha_is_the_split_family_value():
    """Catches the two primaries each being tested at 0.05."""
    assert agg.ALPHA == 0.025
    # Measured when the plan was written: p = 0.0486, inside (0.025, 0.05).
    between = [0.02] * 10 + [-0.03] * 2
    p = agg.one_sided_p(between)
    assert 0.025 < p < 0.05
    assert agg.claim_verdict(between, 0.2, 0.211, gate_ok=True)[0] == "NOT SIGNIFICANT"


def test_gate_v_is_a_ratio():
    """Catches an absolute bar: 0.19 against a reflex at 0.40 passes; 0.21 fails."""
    assert agg.gate_v_verdict(0.19, 0.40) is True
    assert agg.gate_v_verdict(0.21, 0.40) is False


def test_gate_r_fails_at_chance_and_passes_well_above_it():
    """Catches a gate that cannot fail (e.g. comparing the critic to 0 instead of chance)."""
    at_chance = [{"critic_hit": 0.22, "chance": 0.22}] * 12
    above = [{"critic_hit": 0.40, "chance": 0.22}] * 12
    assert agg.gate_r_verdict(at_chance)[0] is False
    assert agg.gate_r_verdict(above)[0] is True


def test_gate_r3_reads_the_leaf_keys():
    """Catches R3 silently reusing the child-level keys: here the children are at chance and
    the leaves are well above it, so R1 fails and R3 passes."""
    rows = [{"critic_hit": 0.22, "chance": 0.22,
             "leaf_closer_hit": 0.40, "leaf_closer_chance": 0.13}] * 12
    assert agg.gate_r_verdict(rows)[0] is False
    assert agg.gate_r_verdict(rows, "leaf_closer_hit", "leaf_closer_chance")[0] is True


def test_gate0b_fails_on_one_field_off():
    """Catches a tolerance or a skipped field in the continuity gate."""
    base = {f: 1 for f in agg.OUTCOME_FIELDS}
    r70 = {("G0", s): dict(base) for s in range(2)}
    r71 = {("G0", s): dict(base) for s in range(2)}
    assert agg.gate0b_verdict(r71, r70) == "PASS"
    r71[("G0", 1)]["optimality"] = 0.999
    assert agg.gate0b_verdict(r71, r70) == "FAIL"
```

The p-values in these comments were computed with EXP-070's `one_sided_p` when the plan was written (ten of twelve 0.0193, nine of twelve 0.0730, the `between` vector 0.0486). If any differs in your run, fix the test's inputs (never the alpha) and say so in the report.

Note on `gate0b_verdict`'s keys: in the test it compares whatever `(arm, seed)` keys `r71` holds; in `main()`, pass only the G0, E3, P3, R3 records. Implement it as "every key in `recs071` must exist in `recs070` with equal `OUTCOME_FIELDS`", which the test pins.

- [ ] **Step 2: Verify failure.** **Step 3: Implement** (mirror EXP-070's `aggregate.py` structure; load EXP-070's committed depth-7 records from `experiments/070_lookahead_existing/outputs/exp070_{G0,E3,P3,R3}_d7_s*.json` and rank records from this folder's `outputs/`). **Step 4: Verify pass.**

- [ ] **Step 5: Mutation check**

| mutation | test that must fail |
|---|---|
| `ALPHA = 0.05` | `test_alpha_is_the_split_family_value` |
| VOID check moved after the significance check | `test_claim_verdict_ladder_including_void` |
| `gate_v_verdict` uses `g0v_revisit <= 0.2` | `test_gate_v_is_a_ratio` (0.19 vs 0.40 still passes; add a case 0.15 vs 0.25 that must FAIL under the ratio, if the mutation survives) |
| `gate_r_verdict` compares `critic_hit` to 0 | `test_gate_r_fails_at_chance_and_passes_well_above_it` |
| `gate_r_verdict` ignores `hit_key`/`chance_key` and always reads the child keys | `test_gate_r3_reads_the_leaf_keys` |
| `gate0b_verdict` skips `optimality` | `test_gate0b_fails_on_one_field_off` |

Revert, confirm clean, strengthen survivors, record outcomes in the commit body.

- [ ] **Step 6: Commit**

```bash
git add experiments/071_critic_and_no_revisit/aggregate.py tests/experiments/test_exp071.py
git commit -m "EXP-071: aggregator, four gates and two primaries at alpha 0.025"
```

---

### Task 5 (CONTROLLER): verify, merge, pre-flight, launch

- [ ] **5.1** Chunked suite per CLAUDE.md from the worktree with `PYTHONPATH=src` (the 2 EXP-054 tests that need untracked records are known to fail in a worktree), `--collect-only` count, update CLAUDE.md's count line, merge `--no-ff` as `Merge exp-071-critic: <summary>`, push, remove the worktree and branch, sync the laptop with `scripts/laptop/sync_repo.ps1`, scp `launch071.ps1`, run `-Phase check`.
- [ ] **5.2 Step 0, ranking:** `-Phase rank`. Fetch the 12 rank records. Compute R1 and R3. **Do not run any C arm yet.**
- [ ] **5.3 Base and determinism:** `-Phase base` (G0, E3, P3, R3: 48 cells; Gate 0(b) against EXP-070's records), then `-Phase det` (Gate 0(a)).
- [ ] **5.4 Dated gate amendment to the spec, BEFORE any C cell:** R1 and R3 with per-seed hit and chance rates (and the policy head's child-level rate); Gate 0(a) and 0(b). Set `GATE_R1_PASSED` and `GATE_R3_PASSED` in `aggregate.py` in the same commit; push; sync the laptop. If R3 failed, the V track still runs but the amendment records Claim 1 as VOID before any C number exists.
- [ ] **5.5 Calibration:** `-Phase calib` (C3, C3V seed 0) for wall-clock only; append the cost to the spec, dated, before the full launch. Do not read its success numbers.
- [ ] **5.5b** Hash-check the critics: after `sync_repo`, compare any EXP-053 `*_critic.pt` moved into the laptop's `repo-attic` against the committed files. A mismatch means that seed's critic did not come from the run that produced its head; stop and report.
- [ ] **5.6 Full launch:** `-Phase full -Workers 20 -SkipExisting`. Poll the record count (144 + 12 rank) rather than trusting the ssh callback.
- [ ] **5.7** Fetch, force-add records, run `aggregate.py --determinism-ok`, write `RESULTS.md` with provenance, commit, push, handoff, `secretary log`.
