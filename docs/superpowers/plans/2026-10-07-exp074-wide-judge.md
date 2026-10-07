# EXP-074 Wide Judge: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a judge readout that sees all 192 sensory neurons (arm W), a training gate (Gate L) that measures 3-move leaf ranking, and an EXP-074 experiment folder that trains W, A and B and evaluates J3V against P3V.

**Architecture:** `value_iteration.Judge` gains a `readout` argument ("concept", byte-identical to EXP-073, or "wide": concept mean rate concatenated with hidden-layer mean rate from the same pass). `lookahead.imagined_values` gains one seam: a critic that declares `reads_states = True` is called on the leaf states directly, so a wide judge can drive search. `experiments/074_wide_judge/` reuses EXP-073's cells for held-out, exclusion and probe sets, adds the Gate L instrument, a training driver, an evaluator, an aggregator and a laptop launcher. Tasks 1 to 6 are implementer tasks on the VPS; Task 7 is the controller's laptop checklist.

**Tech Stack:** Python 3.10 venv, torch CPU, snnTorch via `SensoryCortex`, pytest. No scipy.

**Spec:** `docs/superpowers/specs/2026-10-07-exp074-wide-judge-design.md`. Read it fully before any task, especially sections 3, 6 and 7. Background on the judge and training loop: `docs/superpowers/specs/2026-10-07-exp073-learned-judge-design.md` section 3.

## Global Constraints

- Work ONLY in the worktree `/root/projects/.wt/exp-074` on branch `exp-074-wide-judge`. Prefix every python/pytest command with `PYTHONPATH=src` (the venv's editable install points at main's `src`; without the prefix your tests silently import main's code).
- **Always pass an explicit Bash `timeout` (the default is 120 s).** Call the test command directly and let it block; 600000 ms covers every command in this plan. Buffered output means silence mid-run is normal, not a hang. Never background a test run. Never run the whole suite; run only the files named in your task.
- No em-dashes anywhere. Plain commit messages: no `Co-Authored-By`, no "Generated with". Stage explicit paths only (never `git add -A`).
- Action width from `n_actions` arguments, `N_ACTIONS` or `env.action_space.n`, never a literal 6.
- `value_iteration.py` must NEVER import or read `neuromorphic.envs.cube_distance` (an existing test enforces it). Distances reach the Gate L instrument as a plain callable.
- **Training builds gradients through the encoder in arms A and W.** The wide readout must not run under `torch.no_grad()` and must not route through `Brain.step`, `imagined_values` or `action_distribution`.
- Tests whose subject is gradient flow build their inputs WITH a live graph (EXP-063: a `no_grad` fixture disarms the test).
- **Every test must be able to fail.** After writing one, break the code it guards and confirm it fails, then restore. The task text names the mutation for each key test.
- Existing tests keep passing unmodified, including `tests/training/test_lookahead_golden.py`, `tests/training/test_value_iteration.py` and `tests/experiments/test_exp073.py`.
- Seeds 12 and 13 are pilot-only; seeds 0 to 11 are evaluation seeds.
- Training settings (spec section 4): `batch 1000`, `sync_every 100`, `probe_every 250`, `draws 1`, `n_updates 4000`.

## Review Focus

1. **The concept readout drifting from EXP-073.** Arms A and B here must BE EXP-073's arms; any change to head construction order or the concept forward breaks Gate 0(c) on the laptop after hours of training. Task 1 pins it with a golden value captured from unmodified code.
2. **Hidden spikes from a second encoding pass** (a separate `encoder_fn` call) instead of the same pass as the concept. W would then see two noisy views, not more of one. Task 1 pins it: the wide features' first 64 columns must equal `concept()` exactly under the same generator seed.
3. **A W judge silently evaluated through the 64-unit concept** (the old `NegJ` path). It would crash on a shape mismatch at best and, if someone "fixed" the shape, evaluate the wrong thing. Task 5 pins it with an end-to-end J3V-W cell on a real W checkpoint.
4. **The 10-seed sensitivity line using the wrong seeds or the wrong p-value** (e.g. 4096 flips over 10 seeds). Task 6 pins it: the dropped seeds are exactly 0 and 3 and an all-positive 10-seed contrast has p exactly 1/1024.
5. **A claim read with Gate L or Gate E failing for an arm in its contrast.** Task 6 pins every VOID rule.

---

## Step 0 (controller)

```bash
cd /root/projects/neuromorphic-nn-snn-research-project
git worktree add /root/projects/.wt/exp-074 -b exp-074-wide-judge
ln -s /root/projects/neuromorphic-nn-snn-research-project/.venv /root/projects/.wt/exp-074/.venv
```

## File Structure

| file | responsibility |
|---|---|
| `src/neuromorphic/training/value_iteration.py` (modify) | `Judge(readout=...)`, `Judge.features`, `judge_from_brain(readout=...)`, `make_optimizer` accepts arm "W" |
| `src/neuromorphic/training/lookahead.py` (modify) | `imagined_values` calls a `reads_states` critic on the states |
| `tests/training/test_value_iteration_readout.py` (create) | readout tests |
| `tests/training/test_lookahead_state_critic.py` (create) | seam tests |
| `experiments/074_wide_judge/cells.py` (create) | arms, seeds, sensitivity drop list, `make_judge(seed, arm)`, paths; reuses EXP-073 cells |
| `experiments/074_wide_judge/gate_l.py` (create) | `leaf_rank_margin`, the Gate L instrument |
| `experiments/074_wide_judge/train.py` (create) | training driver; writes Gate L and encoder drift into each record |
| `experiments/074_wide_judge/evaluate.py` (create) | `JudgeCritic`, `run_cell`, `rank_cell`, CLI |
| `experiments/074_wide_judge/aggregate.py` (create) | Gates L, E, R, 0(a/b/c), 1, claims, sensitivity line |
| `experiments/074_wide_judge/launch074.ps1` (create) | laptop phases |
| `experiments/074_wide_judge/outputs/.gitkeep`, `outputs_pilot/.gitkeep` (create) | record folders |
| `tests/experiments/test_exp074.py` (create) | experiment-folder tests |

---

### Task 1: Wide readout in the judge

**Files:**
- Modify: `src/neuromorphic/training/value_iteration.py` (class `Judge`, `judge_from_brain`, `make_optimizer`)
- Create: `tests/training/test_value_iteration_readout.py`

**Interfaces:**
- Produces: `vi.READOUTS = ("concept", "wide")`; `vi.Judge(encoder_fn, sensory, T, content, hidden=128, readout="concept")`; `Judge.readout: str`; `Judge.features(states, generator) -> Tensor[B, 64 or 192]`; `Judge.forward` unchanged in signature; `vi.judge_from_brain(brain, hidden=128, readout="concept")`; `vi.make_optimizer(judge, arm)` accepts `"A"`, `"W"` (encoder trains, lr 1e-4, head lr 1e-3) and `"B"` (encoder frozen).

- [ ] **Step 1: Capture the golden value from UNMODIFIED code.** Before touching `value_iteration.py`, run:

```bash
cd /root/projects/.wt/exp-074 && PYTHONPATH=src .venv/bin/python -c "
import torch, random
from neuromorphic.envs.cube import N_ACTIONS
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.cube_baseline import CubeConfig, make_agent
torch.manual_seed(0)
j = vi.judge_from_brain(make_agent(CubeConfig(arm='regionalized', seed=0)))
states = vi.random_walk_states(6, 5, random.Random(1), N_ACTIONS, exclude=set())
with torch.no_grad():
    print([round(v, 6) for v in j(states, torch.Generator().manual_seed(2)).tolist()])
    print(round(float(j.head[0].weight.sum()), 6))
"
```

Record both printed lines; they go into the golden test below as `GOLDEN_J` and `GOLDEN_HEAD_SUM`.

- [ ] **Step 2: Write the failing tests** in `tests/training/test_value_iteration_readout.py`:

```python
"""EXP-074 wide readout. Each docstring names the bug it catches."""

from __future__ import annotations

import random

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.cube_baseline import CubeConfig, make_agent

GOLDEN_J = [...]          # paste Step 1's first line
GOLDEN_HEAD_SUM = ...     # paste Step 1's second line


def _judge(seed=0, readout="concept"):
    torch.manual_seed(seed)
    return vi.judge_from_brain(make_agent(CubeConfig(arm="regionalized", seed=seed)),
                               readout=readout)


def _states(n=6):
    return vi.random_walk_states(n, 5, random.Random(1), N_ACTIONS, exclude=set())


def test_concept_readout_is_byte_identical_to_exp073():
    """Catches any drift in the concept path or head construction order. Arms A and B here
    must BE EXP-073's arms (Gate 0(c)); a changed init or forward fails this before the laptop
    spends hours finding out."""
    j = _judge()
    with torch.no_grad():
        got = [round(v, 6) for v in j(_states(), torch.Generator().manual_seed(2)).tolist()]
    assert got == GOLDEN_J
    assert round(float(j.head[0].weight.sum()), 6) == GOLDEN_HEAD_SUM


def test_wide_features_are_concept_plus_hidden_from_the_same_pass():
    """Catches hidden rates taken from a SECOND encoding (a separate encoder_fn call): the
    first 64 columns would then not equal the concept under the same generator seed."""
    j = _judge(readout="wide")
    s = _states()
    with torch.no_grad():
        wide = j.features(s, torch.Generator().manual_seed(5))
        concept = j.concept(s, torch.Generator().manual_seed(5))
    assert wide.shape == (len(s), 64 + 128)
    assert torch.equal(wide[:, :64], concept)
    hidden = wide[:, 64:]
    assert float(hidden.min()) >= 0.0 and float(hidden.max()) <= 1.0
    assert float(hidden.sum()) > 0.0


def test_wide_head_reads_192_inputs_and_concept_head_reads_64():
    assert _judge(readout="wide").head[0].in_features == 192
    assert _judge().head[0].in_features == 64


def test_unknown_readout_is_refused():
    with pytest.raises(ValueError, match="readout"):
        _judge(readout="everything")


def test_wide_hidden_path_carries_gradient_to_fc1():
    """Catches hidden rates detached from the graph. With the concept columns' head weights
    zeroed, the ONLY route from the loss to fc1 is through the hidden rates. Live graph: no
    no_grad anywhere in this test."""
    j = _judge(readout="wide")
    with torch.no_grad():
        j.head[0].weight[:, :64] = 0.0
    out = j(_states(16), torch.Generator().manual_seed(0))
    out.sum().backward()
    assert j.sensory.fc1.weight.grad is not None
    assert float(j.sensory.fc1.weight.grad.abs().sum()) > 0.0


def test_arm_w_trains_the_encoder_like_arm_a():
    """Catches make_optimizer rejecting W, or building W without the encoder group."""
    j = _judge(readout="wide")
    opt = vi.make_optimizer(j, "W")
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-4, 1e-3]
    jt = vi.sync_target(j)
    w0 = j.sensory.fc1.weight.detach().clone()
    vi.train_step(j, jt, opt, _states(16), N_ACTIONS, torch.Generator().manual_seed(0))
    assert not torch.equal(j.sensory.fc1.weight, w0)


def test_sync_target_copies_the_readout():
    """Catches a frozen target that reads the concept while the judge reads wide (targets would
    come from a different function than the one being trained)."""
    j = _judge(readout="wide")
    jt = vi.sync_target(j)
    assert jt.readout == "wide"
    s = _states()
    with torch.no_grad():
        assert torch.equal(jt(s, torch.Generator().manual_seed(3)),
                           j(s, torch.Generator().manual_seed(3)))
```

- [ ] **Step 3: Run them to verify they fail.**

Run: `cd /root/projects/.wt/exp-074 && PYTHONPATH=src timeout 590 .venv/bin/python -m pytest tests/training/test_value_iteration_readout.py -q` (Bash timeout 600000)
Expected: the golden test PASSES (unmodified code); every other test FAILS (`readout` is an unexpected keyword).

- [ ] **Step 4: Implement.** In `value_iteration.py` replace `Judge`, `judge_from_brain` and `make_optimizer` with:

```python
READOUTS = ("concept", "wide")


class Judge(nn.Module):
    def __init__(self, encoder_fn, sensory: nn.Module, T: int, content: int, hidden: int = 128,
                 readout: str = "concept"):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"unknown readout {readout!r}; expected one of {READOUTS}")
        self.encoder_fn = encoder_fn
        self.sensory = sensory
        self.T = T
        self.readout = readout
        n_in = content if readout == "concept" else content + sensory.fc1.out_features
        self.head = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def concept(self, states, generator):
        obs = torch.as_tensor(np.array(states), dtype=torch.long)
        spikes = self.encoder_fn(obs, T=self.T, generator=generator)
        return self.sensory(spikes).mean(dim=0)

    def _wide(self, states, generator):
        """Concept mean rate then hidden mean rate, from ONE encoding and ONE pass. Mirrors
        SensoryCortex.forward step for step (reset, then fc1 -> lif1 -> fc2 -> lif2 per t)."""
        obs = torch.as_tensor(np.array(states), dtype=torch.long)
        spikes = self.encoder_fn(obs, T=self.T, generator=generator)
        s = self.sensory
        s.reset()
        mem1, mem2 = s.mem1, s.mem2
        hidden, concept = [], []
        for t in range(spikes.shape[0]):
            spk1, mem1 = s.lif1(s.fc1(spikes[t]), mem1)
            spk2, mem2 = s.lif2(s.fc2(spk1), mem2)
            hidden.append(spk1)
            concept.append(spk2)
        s.mem1, s.mem2 = mem1, mem2
        return torch.cat([torch.stack(concept).mean(dim=0), torch.stack(hidden).mean(dim=0)],
                         dim=1)

    def features(self, states, generator):
        if self.readout == "concept":
            return self.concept(states, generator)
        return self._wide(states, generator)

    def forward(self, states, generator):
        return F.softplus(self.head(self.features(states, generator))).squeeze(-1)


def judge_from_brain(brain, hidden: int = 128, readout: str = "concept") -> Judge:
    return Judge(brain._encoder, copy.deepcopy(brain.sensory), brain.T, brain.content, hidden,
                 readout)
```

and in `make_optimizer` change `if arm == "A":` to `if arm in ("A", "W"):`. Leave everything else in the module unchanged.

- [ ] **Step 5: Run the new tests and the existing value-iteration tests.**

Run: `cd /root/projects/.wt/exp-074 && PYTHONPATH=src timeout 590 .venv/bin/python -m pytest tests/training/test_value_iteration_readout.py tests/training/test_value_iteration.py -q` (Bash timeout 600000)
Expected: all PASS.

- [ ] **Step 6: Mutation checks** (restore after each): (a) in `_wide`, replace the hidden list with rates from a second `self.encoder_fn(...)` call: `test_wide_features_are_concept_plus_hidden_from_the_same_pass` must fail. (b) append `.detach()` to `torch.stack(hidden).mean(dim=0)`: `test_wide_hidden_path_carries_gradient_to_fc1` must fail. (c) add a stray `torch.randn(1)` in `__init__` just before `self.head` is built (an RNG draw that shifts the head's init): the golden test must fail. Report each result.

- [ ] **Step 7: Commit.**

```bash
git add src/neuromorphic/training/value_iteration.py tests/training/test_value_iteration_readout.py
git commit -m "EXP-074: judge readout option, wide reads concept plus hidden rates from one pass"
```

---

### Task 2: State-reading critic seam in look-ahead

**Files:**
- Modify: `src/neuromorphic/training/lookahead.py` (`imagined_values`)
- Create: `tests/training/test_lookahead_state_critic.py`

**Interfaces:**
- Produces: a critic object with attribute `reads_states = True` is called as `critic(states, generator=generator)` by `imagined_values`, under `torch.no_grad()`, and must return a 1-D tensor of scores (higher is better), one per state. Every other critic is called exactly as before.

- [ ] **Step 1: Write the failing tests** in `tests/training/test_lookahead_state_critic.py`:

```python
"""EXP-074 seam: a critic that reads states directly. Each docstring names the bug it catches."""

from __future__ import annotations

import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.training import lookahead as la


class _NoStep:
    def step(self, *a, **k):
        raise AssertionError("a state-reading critic must not go through agent.step")


class _StateCritic:
    reads_states = True

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def __call__(self, states, generator):
        self.calls.append((list(states), generator))
        return torch.tensor(self.scores[: len(states)], dtype=torch.float32)


def test_state_critic_receives_the_states_and_generator_and_skips_agent_step():
    """Catches the seam routing through agent.step (a wide judge cannot be read from the
    concept) or passing a different generator than the search's imagination stream."""
    states = [apply_move(SOLVED, a) for a in range(N_ACTIONS)]
    critic = _StateCritic(list(range(N_ACTIONS)))
    g = torch.Generator().manual_seed(0)
    out = la.imagined_values(_NoStep(), critic, states, generator=g)
    assert critic.calls[0][0] == states
    assert critic.calls[0][1] is g
    assert out.tolist() == [float(i) for i in range(N_ACTIONS)]


def test_choose_move_c_with_a_state_critic_takes_the_best_leaf():
    """Catches the seam returning scores in a different order than levels[k]."""
    root = apply_move(apply_move(SOLVED, 0), 2)
    leaves = la.tree_levels(root, 1, N_ACTIONS)[1]
    scores = [0.0] * N_ACTIONS
    scores[4] = 9.0
    action, fired = la.choose_move("C", root, torch.zeros(N_ACTIONS), 1, N_ACTIONS,
                                   agent=_NoStep(), critic=_StateCritic(scores),
                                   imag_generator=torch.Generator().manual_seed(0))
    assert fired is False   # root is two moves out, so a 1-move goal check cannot fire
    assert action == 4
    assert len(leaves) == N_ACTIONS
```

- [ ] **Step 2: Run to verify they fail.**

Run: `cd /root/projects/.wt/exp-074 && PYTHONPATH=src timeout 300 .venv/bin/python -m pytest tests/training/test_lookahead_state_critic.py -q` (Bash timeout 330000)
Expected: FAIL with the AssertionError from `_NoStep.step`.

- [ ] **Step 3: Implement.** Replace `imagined_values` in `lookahead.py` with:

```python
def imagined_values(agent, critic, states, *, generator):
    """The critic's value for many states, in ONE batched call, recall off.

    A critic with `reads_states = True` (EXP-074's judge) is called on the states themselves; it
    owns its own encoder and readout. Any other critic reads `agent.step`'s concept as before.
    """
    with torch.no_grad():
        if getattr(critic, "reads_states", False):
            return critic(states, generator=generator)
        out = agent.step(np.array(states), recall=False, generator=generator)
        return critic(out["concept"].mean(dim=0)).squeeze(-1)
```

- [ ] **Step 4: Run the new tests and the look-ahead regression tests.**

Run: `cd /root/projects/.wt/exp-074 && PYTHONPATH=src timeout 590 .venv/bin/python -m pytest tests/training/test_lookahead_state_critic.py tests/training/test_lookahead.py tests/training/test_lookahead_golden.py -q` (Bash timeout 600000)
Expected: all PASS (the golden file is unmodified).

- [ ] **Step 5: Mutation check:** make the seam call `critic(states, generator=torch.Generator())`: the first test must fail. Restore.

- [ ] **Step 6: Commit.**

```bash
git add src/neuromorphic/training/lookahead.py tests/training/test_lookahead_state_critic.py
git commit -m "Look-ahead: a critic that reads states is called on them directly (EXP-074 seam)"
```

---

### Task 3: EXP-074 cells and the Gate L instrument

**Files:**
- Create: `experiments/074_wide_judge/cells.py`, `experiments/074_wide_judge/gate_l.py`, `experiments/074_wide_judge/outputs/.gitkeep`, `experiments/074_wide_judge/outputs_pilot/.gitkeep`
- Create: `tests/experiments/test_exp074.py`

**Interfaces:**
- Consumes: Task 1's `vi.judge_from_brain(brain, readout=...)`; EXP-073 `cells.py` (`c70`, `make_judge`, `exclusion_set`, `probe_set`, `heldout_states`, `EVAL_SEEDS`, `PILOT_SEEDS`, `EVAL_DEPTHS`).
- Produces (cells): `c73`, `c70`, `EVAL_SEEDS`, `PILOT_SEEDS`, `EVAL_DEPTHS`, `ARMS_TRAIN = ("W", "A", "B")`, `SENSITIVITY_DROP = (0, 3)`, `heldout_states`, `exclusion_set`, `probe_set` (all re-exported from EXP-073), `make_judge(seed: int, arm: str) -> vi.Judge`, `ckpt_dir(arm, seed, out_dir=None) -> Path` (default `HERE / "outputs"`), `record_name(arm, seed) -> str` = `f"exp074_train_{arm}_s{seed}.json"`.
- Produces (gate_l): `leaf_rank_margin(judge, probe, distance, seed, n_actions, k=3, lo=7, hi=11) -> dict` with keys `hit`, `chance`, `margin` (= hit - chance), `n`, `by_distance` (`{str(d): {"hit": float, "chance": float, "n": int}}`).

- [ ] **Step 1: Write the failing tests** in `tests/experiments/test_exp074.py`:

```python
"""EXP-074: cells, Gate L, training driver, evaluation and aggregator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "074_wide_judge"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp074_cells", "cells.py")
gate_l = _load("exp074_gate_l", "gate_l.py")


def test_arms_a_and_b_are_exactly_exp073s_judges():
    """THE GATE 0(c) PRECONDITION. Catches arm A or B built differently from EXP-073 (head init
    order, readout, encoder), which would make them a new arm rather than the control."""
    for arm in ("A", "B"):
        mine = cells.make_judge(3, arm).state_dict()
        ref = cells.c73.make_judge(3).state_dict()
        assert mine.keys() == ref.keys()
        for k in ref:
            assert torch.equal(mine[k], ref[k]), (arm, k)


def test_arm_w_is_wide_and_starts_from_the_seeds_e1_encoder():
    """Catches W built on a random encoder or with the concept readout."""
    j = cells.make_judge(3, "W")
    assert j.readout == "wide" and j.head[0].in_features == 192
    e1 = torch.load(cells.c70.published_config(7, 3).encoder_state_path, map_location="cpu")
    for k, v in j.sensory.state_dict().items():
        assert torch.equal(v, e1[k]), k


def test_make_judge_refuses_an_unknown_arm():
    with pytest.raises(ValueError, match="arm"):
        cells.make_judge(0, "C")


def test_sensitivity_drop_is_exactly_the_disclosed_seeds():
    assert cells.SENSITIVITY_DROP == (0, 3)


class _ByDistance(torch.nn.Module):
    """A stub judge: J(s) = sign * true distance."""

    def __init__(self, provider, sign):
        super().__init__()
        self.p, self.sign = provider, sign

    def forward(self, states, generator):
        return torch.tensor([self.sign * float(self.p.distance(s)) for s in states])


@pytest.fixture(scope="module")
def small():
    return ExactBFSDistance(max_depth=6)


def _probe(small):
    roots = [apply_move(apply_move(apply_move(SOLVED, 0), 2), 4),
             apply_move(apply_move(SOLVED, 1), 3)]
    return [(s, small.distance(s)) for s in roots]


def test_gate_l_perfect_judge_always_hits_and_anti_judge_never_does(small):
    """Catches argmax instead of argmin (the instrument would reward a judge for pointing AWAY
    from solved) and a hit defined on the wrong side of the root distance."""
    probe = _probe(small)
    good = gate_l.leaf_rank_margin(_ByDistance(small, 1.0), probe, small.distance, 0,
                                   N_ACTIONS, lo=1, hi=6)
    bad = gate_l.leaf_rank_margin(_ByDistance(small, -1.0), probe, small.distance, 0,
                                  N_ACTIONS, lo=1, hi=6)
    assert good["hit"] == 1.0 and bad["hit"] == 0.0
    assert good["n"] == 2
    assert good["chance"] == bad["chance"]
    assert 0.0 < good["chance"] < 1.0
    assert good["margin"] == pytest.approx(1.0 - good["chance"])


def test_gate_l_chance_is_the_fraction_of_closer_leaves(small):
    """Catches chance computed over children instead of the 3-move leaves the search scores."""
    from neuromorphic.training.lookahead import tree_levels
    s, d = _probe(small)[1]
    leaves = tree_levels(s, 3, N_ACTIONS)[3]
    want = sum(small.distance(x) < d for x in leaves) / len(leaves)
    got = gate_l.leaf_rank_margin(_ByDistance(small, 1.0), [(s, d)], small.distance, 0,
                                  N_ACTIONS, lo=1, hi=6)
    assert got["chance"] == pytest.approx(want)


def test_gate_l_restricts_to_distances_lo_to_hi(small):
    """Catches the gate pooling shallow probe states (which every judge ranks well) into the
    margin it reports for distances 7 to 11."""
    probe = _probe(small)
    got = gate_l.leaf_rank_margin(_ByDistance(small, 1.0), probe, small.distance, 0,
                                  N_ACTIONS, lo=3, hi=6)
    assert got["n"] == sum(1 for _, d in probe if 3 <= d <= 6)
    assert set(got["by_distance"]) == {str(d) for _, d in probe if 3 <= d <= 6}
```

- [ ] **Step 2: Run to verify they fail.**

Run: `cd /root/projects/.wt/exp-074 && PYTHONPATH=src timeout 590 .venv/bin/python -m pytest tests/experiments/test_exp074.py -q` (Bash timeout 600000)
Expected: collection error (`cells.py` does not exist).

- [ ] **Step 3: Implement `cells.py`:**

```python
"""EXP-074 cells: arms W, A and B over EXP-073's per-seed held-out, exclusion and probe sets.

Arms A and B are built by EXP-073's own `make_judge`, so they ARE EXP-073's judges (Gate 0(c)).
Arm W is the same construction with the wide readout.

Spec: docs/superpowers/specs/2026-10-07-exp074-wide-judge-design.md, sections 3 and 6.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

_spec = importlib.util.spec_from_file_location(
    "exp073_cells", REPO / "experiments" / "073_learned_judge" / "cells.py")
c73 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c73)
c70 = c73.c70

from neuromorphic.training.cube_baseline import make_agent  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

EVAL_SEEDS = c73.EVAL_SEEDS
PILOT_SEEDS = c73.PILOT_SEEDS
EVAL_DEPTHS = c73.EVAL_DEPTHS
ARMS_TRAIN = ("W", "A", "B")
# Spec section 2: seeds whose depth-9 cells were looked at before this spec was written.
SENSITIVITY_DROP = (0, 3)

heldout_states = c73.heldout_states
exclusion_set = c73.exclusion_set
probe_set = c73.probe_set


def make_judge(seed: int, arm: str) -> vi.Judge:
    """A and B: EXP-073's judge exactly. W: the same construction (same agent, same seeding
    point before the head) with the wide readout."""
    if arm in ("A", "B"):
        return c73.make_judge(seed)
    if arm == "W":
        brain = make_agent(c70.published_config(7, seed))
        torch.manual_seed(seed)
        return vi.judge_from_brain(brain, readout="wide")
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS_TRAIN}")


def ckpt_dir(arm: str, seed: int, out_dir: Path | None = None) -> Path:
    return Path(out_dir or HERE / "outputs") / f"judge_{arm}_s{seed}"


def record_name(arm: str, seed: int) -> str:
    return f"exp074_train_{arm}_s{seed}.json"
```

- [ ] **Step 4: Implement `gate_l.py`:**

```python
"""Gate L (spec section 6): does the judge's lowest-J leaf of the 3-move tree lie closer to
solved than the root, more often than chance? Measured on probe states only. `distance` is the
BFS yardstick, passed in as a callable; the judge never sees it."""

from __future__ import annotations

import statistics as st

import torch

from neuromorphic.training.lookahead import tree_levels


def leaf_rank_margin(judge, probe, distance, seed, n_actions, k=3, lo=7, hi=11) -> dict:
    g = torch.Generator().manual_seed(seed)
    by: dict[int, tuple[list, list]] = {}
    for s, d in probe:
        if not lo <= d <= hi:
            continue
        leaves = tree_levels(s, k, n_actions)[k]
        with torch.no_grad():
            j = judge(leaves, g)
        leaf_d = [distance(x) for x in leaves]
        hits, chances = by.setdefault(d, ([], []))
        hits.append(int(leaf_d[int(j.argmin())] < d))  # first minimum
        chances.append(sum(x < d for x in leaf_d) / len(leaf_d))
    all_h = [h for hs, _ in by.values() for h in hs]
    all_c = [c for _, cs in by.values() for c in cs]
    hit = st.mean(all_h) if all_h else float("nan")
    chance = st.mean(all_c) if all_c else float("nan")
    return {
        "hit": hit, "chance": chance, "margin": hit - chance, "n": len(all_h),
        "by_distance": {str(d): {"hit": st.mean(hs), "chance": st.mean(cs), "n": len(hs)}
                        for d, (hs, cs) in sorted(by.items())},
    }
```

Create the two empty `.gitkeep` files.

- [ ] **Step 5: Run the tests.** Same command as Step 2. Expected: all PASS.

- [ ] **Step 6: Mutation checks** (restore after each): (a) `j.argmin()` to `j.argmax()`: the perfect/anti test must fail. (b) chance over `tree_levels(s, 1, n_actions)[1]` instead of the k-leaves: the chance test must fail. (c) in `make_judge`, call `torch.manual_seed(seed)` BEFORE `make_agent` for W and route A through the W code with `readout="concept"`: the arms-A-and-B test must fail. Report each.

- [ ] **Step 7: Commit.**

```bash
git add experiments/074_wide_judge/cells.py experiments/074_wide_judge/gate_l.py experiments/074_wide_judge/outputs/.gitkeep experiments/074_wide_judge/outputs_pilot/.gitkeep tests/experiments/test_exp074.py
git commit -m "EXP-074: cells for arms W, A and B, and the Gate L leaf-ranking instrument"
```

---

### Task 4: Training driver

**Files:**
- Create: `experiments/074_wide_judge/train.py`
- Modify: `tests/experiments/test_exp074.py` (append)

**Interfaces:**
- Consumes: Task 3 `cells` and `gate_l.leaf_rank_margin`; `vi.train_judge` (unchanged; accepts arm "W" after Task 1).
- Produces: `train.run(arm, seed, n_updates, batch, sync_every, probe_every, draws, out_dir) -> dict`, writing `out_dir / cells.record_name(arm, seed)` with every EXP-073 record field plus `"gate_l"` (the `leaf_rank_margin` dict) and `"readout"`; `train.check_seeds(seeds, pilot)`; CLI `--arms --seeds --n-updates --batch 1000 --sync-every 100 --probe-every 250 --draws 1 --workers 4 --pilot --out-dir`. Checkpoints go to `cells.ckpt_dir(arm, seed, out_dir)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/experiments/test_exp074.py`):

```python
train = _load("exp074_train", "train.py")


def test_driver_refuses_seed_misuse():
    """Catches a pilot run on an evaluation seed, or an evaluation run on a pilot seed."""
    with pytest.raises(SystemExit):
        train.check_seeds([0], pilot=True)
    with pytest.raises(SystemExit):
        train.check_seeds([12], pilot=False)
    train.check_seeds([12, 13], pilot=True)
    train.check_seeds(list(range(12)), pilot=False)


def test_driver_defaults_are_the_spec_settings():
    """Catches a driver whose defaults drift from spec section 4 (the launcher passes them, but a
    hand-run must not silently differ)."""
    ap = train.build_parser()
    a = ap.parse_args(["--n-updates", "1"])
    assert (a.batch, a.sync_every, a.probe_every, a.draws) == (1000, 100, 250, 1)
    assert a.arms == ["W", "A", "B"]


@pytest.mark.slow
def test_run_writes_gate_l_drift_and_readout(tmp_path):
    """Catches a record missing the Gate L block or the encoder drift Gate E reads. Slow: builds
    the full BFS table (about 65 s) as the real driver does."""
    rec = train.run("W", 12, 2, 8, 1, 1, 1, tmp_path)
    on_disk = json.loads((tmp_path / cells.record_name("W", 12)).read_text())
    assert on_disk["readout"] == "wide"
    assert on_disk["gate_l"]["n"] == 250
    assert 0.0 <= on_disk["gate_l"]["chance"] <= 1.0
    assert on_disk["encoder_drift"] > 0.0
    assert (tmp_path / "judge_W_s12" / "judge.pt").exists()
    assert rec["gate_l"] == on_disk["gate_l"]
```

- [ ] **Step 2: Run to verify they fail.**

Run: `cd /root/projects/.wt/exp-074 && PYTHONPATH=src timeout 590 .venv/bin/python -m pytest tests/experiments/test_exp074.py -q -k "driver or gate_l_drift"` (Bash timeout 600000)
Expected: collection error (`train.py` does not exist).

- [ ] **Step 3: Implement `train.py`** (modelled on `experiments/073_learned_judge/train.py`; read it first):

```python
"""EXP-074 training driver: arms W, A and B by value iteration, then Gate L on the final judge.

Builds the full BFS table once per process (about 65 s) for the exclusion and probe sets and as
Gate L's yardstick. Training itself never reads it.

Spec: docs/superpowers/specs/2026-10-07-exp074-wide-judge-design.md, sections 4 to 6.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/074_wide_judge/train.py --pilot --arms W A B --seeds 12 13 \
        --n-updates 4000 --out-dir experiments/074_wide_judge/outputs_pilot --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp074_cells", "cells.py")
gate_l = _load("exp074_gate_l", "gate_l.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

torch.set_num_threads(1)

MAX_LEN = 14  # 2x2 quarter-turn diameter (spec EXP-073 section 3.1)
OUT_DIR_DEFAULT = HERE / "outputs"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def check_seeds(seeds, pilot: bool) -> None:
    allowed = cells.PILOT_SEEDS if pilot else cells.EVAL_SEEDS
    bad = [s for s in seeds if s not in allowed]
    if bad:
        raise SystemExit(f"seeds {bad} not allowed with pilot={pilot}; allowed seeds are {allowed}")


def _encoder_drift(judge, seed: int) -> float:
    e1 = torch.load(cells.c70.published_config(7, seed).encoder_state_path, map_location="cpu")
    cur = judge.sensory.state_dict()
    return sum(float(((cur[k] - e1[k]) ** 2).sum()) for k in e1) ** 0.5


def run(arm, seed, n_updates, batch, sync_every, probe_every, draws, out_dir) -> dict:
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    provider = ExactBFSDistance(max_depth=None)
    excl = cells.exclusion_set(seed, provider)
    probe = cells.probe_set(seed, provider)
    judge = cells.make_judge(seed, arm)
    result = vi.train_judge(
        judge, arm, n_updates=n_updates, batch=batch, max_len=MAX_LEN, n_actions=N_ACTIONS,
        exclude=excl, probe=probe, seed=seed, sync_every=sync_every, probe_every=probe_every,
        draws=draws, ckpt_dir=cells.ckpt_dir(arm, seed, out_dir),
    )
    judge.eval()
    rec = {
        **result,
        "arm": arm, "seed": seed, "readout": judge.readout,
        "n_updates": n_updates, "batch": batch, "max_len": MAX_LEN, "n_actions": N_ACTIONS,
        "sync_every": sync_every, "probe_every": probe_every, "draws": draws,
        "n_exclude": len(excl), "n_probe": len(probe),
        "encoder_drift": _encoder_drift(judge, seed),
        "gate_l": gate_l.leaf_rank_margin(judge, probe, provider.distance, seed, N_ACTIONS),
        "git_commit": _git_commit(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cells.record_name(arm, seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(cells.ARMS_TRAIN))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--n-updates", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--sync-every", type=int, default=100)
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    bad_arms = [a for a in args.arms if a not in cells.ARMS_TRAIN]
    if bad_arms:
        raise SystemExit(f"arms {bad_arms} outside {cells.ARMS_TRAIN}")
    check_seeds(args.seeds, args.pilot)
    jobs = [(arm, seed) for arm in args.arms for seed in args.seeds]
    print(f"EXP-074: {len(jobs)} training runs, {args.workers} workers, "
          f"pilot={args.pilot}, n_updates={args.n_updates}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, arm, seed, args.n_updates, args.batch, args.sync_every,
                            args.probe_every, args.draws, args.out_dir): (arm, seed)
                for arm, seed in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  arm {r['arm']} seed {r['seed']}  updates {r['updates']}  "
                  f"loss_last {r['loss_last']:.6f}  wall_s {r['wall_s']:.1f}  "
                  f"encoder_drift {r['encoder_drift']:.6f}  "
                  f"gate_l margin {r['gate_l']['margin']:.4f}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the fast tests** (Step 2's command): PASS. **Then run the slow one** on its own: `cd /root/projects/.wt/exp-074 && PYTHONPATH=src timeout 590 .venv/bin/python -m pytest tests/experiments/test_exp074.py -q -k gate_l_drift` (Bash timeout 600000). Expected: PASS.

- [ ] **Step 5: Mutation check:** drop `"gate_l"` from the record: the slow test must fail. Restore.

- [ ] **Step 6: Commit.**

```bash
git add experiments/074_wide_judge/train.py tests/experiments/test_exp074.py
git commit -m "EXP-074: training driver writes Gate L, encoder drift and readout into each record"
```

---

### Task 5: Evaluation

**Files:**
- Create: `experiments/074_wide_judge/evaluate.py`
- Modify: `tests/experiments/test_exp074.py` (append)

**Interfaces:**
- Consumes: Task 2's seam; Task 3 `cells.make_judge`, `cells.ckpt_dir`; EXP-070 `cells.c70.load_cell(depth, seed) -> (agent, head, states, train_seed)`; `lookahead.evaluate_lookahead`, `imag_seed_for`, `imagined_logp`, `imagined_values`, `sequence_scores`, `tree_levels`.
- Produces: `ARMS_EVAL = ("J3V-W", "J3V-A", "J3V-B", "P3V", "R3V")`; `RANK_KINDS = ("J-W", "J-A", "J-B", "P")`; `JudgeCritic(judge)` with `reads_states = True` and `forward(states, generator) -> -judge(states, generator)`; `load_judge(train_arm, seed, out_dir=None) -> vi.Judge`; `run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict` writing `exp074_{arm}_d{depth}_s{seed}.json`; `rank_cell(kind, depth, seed, out_dir, judge_dir=None) -> dict` writing `exp074_rank_{kind}_d{depth}_s{seed}.json` with keys `kind, depth, seed, n, hit, chance`. `judge_dir` is where trained checkpoints live (default `HERE / "outputs"`).

- [ ] **Step 1: Write the failing tests** (append):

```python
evaluate = _load("exp074_evaluate", "evaluate.py")


class _FixedJ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def forward(self, states, generator):
        return torch.tensor(self.values[: len(states)])


def test_judge_critic_reads_states_and_prefers_the_lowest_j():
    """Catches J used with the wrong sign, or a critic the search would read via the concept."""
    c = evaluate.JudgeCritic(_FixedJ([5.0, 0.1, 3.0]))
    assert c.reads_states is True
    assert int(c(["a", "b", "c"], generator=None).argmax()) == 1


def test_eval_arms_and_rank_kinds_are_exactly_the_spec_lists():
    assert list(evaluate.ARMS_EVAL) == ["J3V-W", "J3V-A", "J3V-B", "P3V", "R3V"]
    assert list(evaluate.RANK_KINDS) == ["J-W", "J-A", "J-B", "P"]


def test_p3v_cell_reproduces_exp072_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH: P3V through this harness on 3 states must equal a direct
    evaluate_lookahead of EXP-072's configuration on the same states."""
    from neuromorphic.training.lookahead import evaluate_lookahead
    rec = evaluate.run_cell("P3V", 8, 0, tmp_path, limit_states=3)
    agent, head, states, ts = cells.c70.load_cell(8, 0)
    ref = evaluate_lookahead(agent, head, states[:3], depth=8, mode="P", k=3,
                             generator=torch.Generator().manual_seed(ts), rng_seed=ts,
                             imag_seed=ts, no_revisit=True)
    for f in ("solved", "success_rate", "mean_steps", "eval_revisit_rate"):
        assert rec[f] == ref[f], f


def test_judge_cell_without_a_checkpoint_exits_naming_it(tmp_path):
    """Catches a J cell silently evaluating an untrained judge."""
    with pytest.raises(SystemExit, match="judge.pt"):
        evaluate.run_cell("J3V-A", 8, 0, tmp_path, limit_states=1, judge_dir=tmp_path / "nope")


def test_a_wide_judge_drives_a_real_j3v_cell(tmp_path):
    """Catches a W judge evaluated through the 64-unit concept (the EXP-073 NegJ path): the
    192-input head cannot read a 64-wide concept, so this cell would crash or need a wrong
    'fix'. Uses an untrained W checkpoint saved where the evaluator looks."""
    jdir = tmp_path / "judges"
    path = cells.ckpt_dir("W", 0, jdir)
    path.mkdir(parents=True)
    torch.save(cells.make_judge(0, "W").state_dict(), path / "judge.pt")
    rec = evaluate.run_cell("J3V-W", 7, 0, tmp_path / "out", limit_states=1, judge_dir=jdir)
    assert rec["arm"] == "J3V-W" and rec["n"] == 1 and rec["mode"] == "C"
    assert (tmp_path / "out" / "exp074_J3V-W_d7_s0.json").exists()
```

- [ ] **Step 2: Run to verify they fail.**

Run: `cd /root/projects/.wt/exp-074 && PYTHONPATH=src timeout 590 .venv/bin/python -m pytest tests/experiments/test_exp074.py -q -k "critic or arms_and_rank or p3v_cell or checkpoint or wide_judge_drives"` (Bash timeout 600000)
Expected: collection error (`evaluate.py` does not exist).

- [ ] **Step 3: Implement `evaluate.py`** (modelled on `experiments/073_learned_judge/evaluate.py`; read it first):

```python
"""EXP-074 evaluation: trained judges drive the look-ahead search through a state-reading critic.

Spec: docs/superpowers/specs/2026-10-07-exp074-wide-judge-design.md, section 7.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/074_wide_judge/evaluate.py --arm J3V-W --depth 9 --seed 0
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
import torch.nn as nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp074_cells", HERE / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)

from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training.lookahead import (  # noqa: E402
    evaluate_lookahead, imag_seed_for, imagined_logp, imagined_values, sequence_scores,
    tree_levels,
)

torch.set_num_threads(1)

# arm -> (mode, k, trained-judge arm or None)
_ARMS = {
    "J3V-W": ("C", 3, "W"),
    "J3V-A": ("C", 3, "A"),
    "J3V-B": ("C", 3, "B"),
    "P3V": ("P", 3, None),
    "R3V": ("R", 3, None),
}
ARMS_EVAL = tuple(_ARMS)
RANK_KINDS = ("J-W", "J-A", "J-B", "P")
_RANK_ARM = {"J-W": "J3V-W", "J-A": "J3V-A", "J-B": "J3V-B", "P": "P3V"}


class JudgeCritic(nn.Module):
    """-J over leaf STATES: the search's argmax then picks the lowest J. `reads_states` makes
    `imagined_values` call this on the states, so the judge's own encoder and readout run."""

    reads_states = True

    def __init__(self, judge):
        super().__init__()
        self.judge = judge

    def forward(self, states, generator):
        return -self.judge(states, generator)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_judge(train_arm: str, seed: int, judge_dir=None):
    path = cells.ckpt_dir(train_arm, seed, judge_dir) / "judge.pt"
    if not path.exists():
        raise SystemExit(f"missing judge checkpoint {path} for arm {train_arm} seed {seed}; "
                         f"train it with train.py first")
    judge = cells.make_judge(seed, train_arm)
    judge.load_state_dict(torch.load(path, map_location="cpu"))
    judge.eval()
    return judge


def _critic(arm, seed, judge_dir):
    train_arm = _ARMS[arm][2]
    return None if train_arm is None else JudgeCritic(load_judge(train_arm, seed, judge_dir))


def run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict:
    torch.set_num_threads(1)
    if arm not in _ARMS:
        raise SystemExit(f"unknown arm {arm!r}; valid: {list(ARMS_EVAL)}")
    mode, k, _ = _ARMS[arm]
    t0 = time.time()
    critic = _critic(arm, seed, judge_dir)
    agent, head, states, train_seed = cells.c70.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode=mode, k=k,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    rec = {**res, "depth": depth, "seed": seed, "arm": arm,
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "limit_states": limit_states}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp074_{arm}_d{depth}_s{seed}.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def rank_cell(kind, depth, seed, out_dir, judge_dir=None) -> dict:
    """Per-seed mean hit of 'the top leaf at k=3 is closer to solved than the root' (Gate R),
    on evaluation held-out states. BFS (max_depth = depth + 3) is the yardstick only."""
    torch.set_num_threads(1)
    if kind not in RANK_KINDS:
        raise SystemExit(f"unknown kind {kind!r}; valid: {list(RANK_KINDS)}")
    k = 3
    arm = _RANK_ARM[kind]
    critic = _critic(arm, seed, judge_dir)
    agent, head, states, train_seed = cells.c70.load_cell(depth, seed)
    provider = ExactBFSDistance(max_depth=depth + 3)
    n_actions = head.head.out_features
    hits, chances = [], []
    for i, s in enumerate(states):
        g = torch.Generator().manual_seed(imag_seed_for(train_seed, i, 0))
        levels = tree_levels(s, k, n_actions)
        if critic is not None:
            scores = imagined_values(agent, critic, levels[k], generator=g)
        else:
            level_logp = [imagined_logp(agent, head, [s], generator=g)]
            for l in range(1, k):
                level_logp.append(imagined_logp(agent, head, levels[l], generator=g))
            scores = sequence_scores(level_logp, k, n_actions)
        d = provider.distance(s)
        leaf_d = [provider.distance(x) for x in levels[k]]
        hits.append(int(leaf_d[int(scores.argmax())] < d))
        chances.append(sum(x < d for x in leaf_d) / len(leaf_d))
    rec = {"kind": kind, "depth": depth, "seed": seed, "n": len(states),
           "hit": st.mean(hits), "chance": st.mean(chances)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp074_rank_{kind}_d{depth}_s{seed}.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS_EVAL)
    ap.add_argument("--rank", choices=RANK_KINDS)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--judge-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--limit-states", type=int, default=None)
    args = ap.parse_args()
    if (args.arm is None) == (args.rank is None):
        raise SystemExit("pass exactly one of --arm or --rank")
    if args.arm:
        r = run_cell(args.arm, args.depth, args.seed, args.out_dir, args.limit_states,
                     args.judge_dir)
        print(f"{r['arm']} d{r['depth']} s{r['seed']} solved {r['solved']}/{r['n']}")
    else:
        r = rank_cell(args.rank, args.depth, args.seed, args.out_dir, args.judge_dir)
        print(f"{r['kind']} d{r['depth']} s{r['seed']} hit {r['hit']:.4f} chance {r['chance']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests** (Step 2's command). Expected: all PASS.

- [ ] **Step 5: Mutation check:** remove `reads_states = True` from `JudgeCritic`: `test_a_wide_judge_drives_a_real_j3v_cell` must fail (shape error through the concept path). Restore.

- [ ] **Step 6: Commit.**

```bash
git add experiments/074_wide_judge/evaluate.py tests/experiments/test_exp074.py
git commit -m "EXP-074: evaluation through a state-reading judge critic, and Gate R rank cells"
```

---

### Task 6: Aggregator and laptop launcher

**Files:**
- Create: `experiments/074_wide_judge/aggregate.py`, `experiments/074_wide_judge/launch074.ps1`
- Modify: `tests/experiments/test_exp074.py` (append)

**Interfaces:**
- Consumes: EXP-071 aggregate (`one_sided_p`, `gate0b_verdict`, `gate_r_verdict`, `claim_verdict`), EXP-072 aggregate (`contrast`, `pair_diffs`); Task 3 `cells.SENSITIVITY_DROP`; record names from Tasks 4 and 5.
- Produces: `GATE_L_THRESHOLD: dict | None = None`; `GATE_L_ALPHA = 0.05`; `gate_l_verdict(margins: list[float], threshold: float) -> bool`; `gate_e_verdicts(drift: dict[str, list[float]]) -> dict[str, bool]`; `gate0c_verdict(mine: dict, ref: dict) -> str`; `primary_verdicts(rates, gates, seeds) -> dict`; `sensitivity_rates(rates) -> dict` (every arm's seed dict restricted to seeds not in `SENSITIVITY_DROP`); `main()`.

- [ ] **Step 1: Write the failing tests** (append):

```python
agg = _load("exp074_aggregate", "aggregate.py")


def _gates(**over):
    g = {"gate0": True, "l": {"W": True, "A": True, "B": True},
         "e": {"W": True, "A": True, "B": True},
         "r": {("J-W", 9): True, ("J-A", 9): True, ("J-B", 9): True}}
    g.update(over)
    return g


def _rates(w=0.2, a=0.1, p=0.05, seeds=range(12)):
    return {("J3V-W", 9): {s: w + 0.001 * s for s in seeds},
            ("J3V-A", 9): {s: a for s in seeds},
            ("P3V", 9): {s: p for s in seeds}}


def test_gate_l_needs_significance_and_the_threshold():
    """Catches a gate that passes on a positive mean alone, or on significance alone."""
    assert agg.gate_l_verdict([0.10] * 12, 0.073) is True
    assert agg.gate_l_verdict([0.05] * 12, 0.073) is False           # significant, below bar
    assert agg.gate_l_verdict([0.5, -0.4] * 6, 0.01) is False        # above bar, not significant


def test_gate_e_per_arm():
    """Catches a frozen W or A, and a leaky B."""
    ok = agg.gate_e_verdicts({"W": [1.0] * 12, "A": [1.0] * 12, "B": [0.0] * 12})
    assert ok == {"W": True, "A": True, "B": True}
    bad = agg.gate_e_verdicts({"W": [1.0] * 11 + [0.0], "A": [1.0] * 12, "B": [0.0] * 11 + [1e-9]})
    assert bad == {"W": False, "A": True, "B": False}


def test_claims_compare_the_registered_arms_at_depth_9():
    out = agg.primary_verdicts(_rates(), _gates(), range(12))
    assert out["claim1"][0] == "CONFIRMED" and out["claim1"][3] > out["claim1"][4]
    assert out["claim2"][0] == "CONFIRMED"
    assert out["claim1"][4] == pytest.approx(0.05)    # P3V mean
    assert out["claim2"][4] == pytest.approx(0.1)     # J3V-A mean


@pytest.mark.parametrize("over,c1,c2", [
    ({"gate0": False}, "VOID", "VOID"),
    ({"l": {"W": False, "A": True, "B": True}}, "VOID", "VOID"),
    ({"l": {"W": True, "A": False, "B": True}}, "CONFIRMED", "VOID"),
    ({"e": {"W": False, "A": True, "B": True}}, "VOID", "VOID"),
    ({"e": {"W": True, "A": False, "B": True}}, "CONFIRMED", "VOID"),
    ({"r": {("J-W", 9): False, ("J-A", 9): True, ("J-B", 9): True}}, "VOID", "VOID"),
    ({"r": {("J-W", 9): True, ("J-A", 9): False, ("J-B", 9): True}}, "CONFIRMED", "VOID"),
])
def test_claim_is_void_when_a_gate_in_its_contrast_fails(over, c1, c2):
    out = agg.primary_verdicts(_rates(), _gates(**over), range(12))
    assert (out["claim1"][0], out["claim2"][0]) == (c1, c2)


def test_sensitivity_drops_exactly_seeds_0_and_3_and_uses_exact_flips():
    """Catches the sensitivity line dropping the wrong seeds, or reusing a 12-seed null."""
    r = agg.sensitivity_rates(_rates())
    assert sorted(r[("P3V", 9)]) == [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
    out = agg.primary_verdicts(r, _gates(), sorted(r[("P3V", 9)]))
    assert out["claim1"][1] == pytest.approx(1 / 1024)


def test_gate0c_compares_the_full_probe_history():
    """Catches Gate 0(c) passing on the final probe alone when an earlier probe differs."""
    rec = {"probes": [{"spearman_7_11": 0.1}, {"spearman_7_11": 0.2}],
           "encoder_drift": 1.0}
    same = json.loads(json.dumps(rec))
    diff = json.loads(json.dumps(rec))
    diff["probes"][0]["spearman_7_11"] = 0.1000001
    assert agg.gate0c_verdict({("A", 12): same}, {("A", 12): rec}) == "PASS"
    assert agg.gate0c_verdict({("A", 12): diff}, {("A", 12): rec}) == "FAIL"
    assert agg.gate0c_verdict({}, {("A", 12): rec}) == "FAIL"


def test_gate_l_threshold_is_unset_until_the_controller_amends():
    assert agg.GATE_L_THRESHOLD is None
```

- [ ] **Step 2: Run to verify they fail.**

Run: `cd /root/projects/.wt/exp-074 && PYTHONPATH=src timeout 590 .venv/bin/python -m pytest tests/experiments/test_exp074.py -q -k "gate_l_needs or gate_e_per or claim or sensitivity or gate0c or threshold_is_unset"` (Bash timeout 600000)
Expected: collection error (`aggregate.py` does not exist).

- [ ] **Step 3: Implement `aggregate.py`:**

```python
"""EXP-074 aggregator. Gates 0, L, E, R and 1 are VERDICTS: a claim whose gate failed prints VOID.

Spec: docs/superpowers/specs/2026-10-07-exp074-wide-judge-design.md, sections 6 to 8.

The ladder (alpha 0.025, Gate 1, one-sided exact sign-flip, continuity comparison) is EXP-071's
and EXP-072's, reused through importlib so the experiments cannot drift apart.

Usage: .venv/bin/python experiments/074_wide_judge/aggregate.py --determinism-ok
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


a71 = _load("exp071_aggregate", REPO / "experiments" / "071_critic_and_no_revisit" / "aggregate.py")
a72 = _load("exp072_aggregate", REPO / "experiments" / "072_p3v_frontier" / "aggregate.py")
cells = _load("exp074_cells", HERE / "cells.py")
one_sided_p = a71.one_sided_p
contrast = a72.contrast
SEEDS = range(12)
EXP071_OUT = REPO / "experiments" / "071_critic_and_no_revisit" / "outputs"
EXP072_OUT = REPO / "experiments" / "072_p3v_frontier" / "outputs"
EXP073_OUT = REPO / "experiments" / "073_learned_judge" / "outputs"

# Set ONLY by the controller's dated amendment, after the pilot and before any seed 0-11 trains:
# {"W": float, "A": float, "B": float}, each half that arm's mean pilot Gate L margin (spec
# section 6). None blocks every verdict. Code never chooses it.
GATE_L_THRESHOLD: dict | None = None
GATE_L_ALPHA = 0.05

J_ARMS = ("J3V-W", "J3V-A", "J3V-B")
RANK_DEPTHS = (7, 8, 9)
_KIND = {"J3V-W": "J-W", "J3V-A": "J-A", "J3V-B": "J-B"}
_TRAIN_ARM = {"J3V-W": "W", "J3V-A": "A", "J3V-B": "B"}


def gate_l_verdict(margins: list[float], threshold: float) -> bool:
    """Per-seed margin > 0 by exact one-sided sign-flip (p < 0.05) AND mean margin >= threshold."""
    return bool(one_sided_p(margins) < GATE_L_ALPHA and st.mean(margins) > 0
                and st.mean(margins) >= threshold)


def gate_e_verdicts(drift: dict) -> dict:
    """W and A: every drift > 0 (the encoder trained). B: every drift exactly 0.0."""
    return {"W": all(d > 0 for d in drift["W"]), "A": all(d > 0 for d in drift["A"]),
            "B": all(d == 0.0 for d in drift["B"])}


def gate0c_verdict(mine: dict, ref: dict) -> str:
    """Every key in `ref` (EXP-073 pilot 1 records) must exist in `mine` with an identical
    probe history (every probe, not only the last)."""
    for key, r in ref.items():
        m = mine.get(key)
        if m is None or m["probes"] != r["probes"]:
            return "FAIL"
    return "PASS"


def _arm_ok(gates, j_arm, depth) -> bool:
    a = _TRAIN_ARM[j_arm]
    return gates["l"][a] and gates["e"][a] and gates["r"][(_KIND[j_arm], depth)]


def primary_verdicts(rates: dict, gates: dict, seeds) -> dict:
    """Claim 1: J3V-W vs P3V at depth 9. Claim 2: J3V-W vs J3V-A at depth 9. VOID on Gate 0, or
    Gate L, E or R for any J arm in the contrast. `seeds` restricts both arms' seed sets."""
    r = {k: {s: v[s] for s in seeds} for k, v in rates.items()}
    c1_ok = gates["gate0"] and _arm_ok(gates, "J3V-W", 9)
    c2_ok = c1_ok and _arm_ok(gates, "J3V-A", 9)
    return {"claim1": contrast(r, ("J3V-W", 9), ("P3V", 9), c1_ok),
            "claim2": contrast(r, ("J3V-W", 9), ("J3V-A", 9), c2_ok)}


def sensitivity_rates(rates: dict) -> dict:
    return {k: {s: v for s, v in d.items() if s not in cells.SENSITIVITY_DROP}
            for k, d in rates.items()}


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path.name}: the run is incomplete")
    r = json.loads(path.read_text())
    if r.get("limit_states") is not None:
        raise SystemExit(f"{path.name} is a smoke record and may not enter a verdict")
    return r


def load_rates(out_dir: Path) -> dict:
    """(arm, depth) -> {seed: success_rate}. J arms from this experiment at depths 7, 8, 9, 11;
    P3V and R3V at depth 7 from EXP-071, at 8 and 9 from EXP-072, at 11 from this experiment."""
    rates = {}
    for d in (*RANK_DEPTHS, 11):
        for arm in J_ARMS:
            rates[(arm, d)] = {s: _read(out_dir / f"exp074_{arm}_d{d}_s{s}.json")["success_rate"]
                               for s in SEEDS}
        for arm in ("P3V", "R3V"):
            if d == 11:
                path = lambda s: out_dir / f"exp074_{arm}_d11_s{s}.json"  # noqa: E731
            elif d == 7:
                path = lambda s: EXP071_OUT / f"exp071_{arm}_d7_s{s}.json"  # noqa: E731
            else:
                path = lambda s: EXP072_OUT / f"exp072_{arm}_d{d}_s{s}.json"  # noqa: E731
            rates[(arm, d)] = {s: _read(path(s))["success_rate"] for s in SEEDS}
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--pilot-dir", type=Path, default=HERE / "outputs_pilot")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="set only after the det re-run matches its full-run copies")
    args = ap.parse_args()
    if GATE_L_THRESHOLD is None:
        raise SystemExit("GATE_L_THRESHOLD is unset. The controller sets it in the dated "
                         "post-pilot amendment; no verdict may be read before then.")

    train = {(a, s): _read(args.out_dir / cells.record_name(a, s))
             for a in cells.ARMS_TRAIN for s in SEEDS}
    rank = {(k, d): [_read(args.out_dir / f"exp074_rank_{k}_d{d}_s{s}.json") for s in SEEDS]
            for k in ("J-W", "J-A", "J-B", "P") for d in RANK_DEPTHS}
    rates = load_rates(args.out_dir)

    gate_l = {a: gate_l_verdict([train[(a, s)]["gate_l"]["margin"] for s in SEEDS],
                                GATE_L_THRESHOLD[a]) for a in cells.ARMS_TRAIN}
    gate_e = gate_e_verdicts({a: [train[(a, s)]["encoder_drift"] for s in SEEDS]
                              for a in cells.ARMS_TRAIN})
    gate_r = {(k, d): bool(a71.gate_r_verdict(rank[(k, d)], "hit", "chance")[0])
              for k in ("J-W", "J-A", "J-B") for d in RANK_DEPTHS}
    mine0c = {(a, s): _read(args.pilot_dir / cells.record_name(a, s))
              for a in ("A", "B") for s in cells.PILOT_SEEDS}
    ref0c = {(a, s): _read(EXP073_OUT / f"exp073_train_{a}_s{s}.json")
             for a in ("A", "B") for s in cells.PILOT_SEEDS}
    g0c = gate0c_verdict(mine0c, ref0c)
    g0b = a71.gate0b_verdict(
        {("P3V", d): _read(args.out_dir / f"exp074_P3V_d{d}_s0.json") for d in (8, 9)},
        {("P3V", d): _read(EXP072_OUT / f"exp072_P3V_d{d}_s0.json") for d in (8, 9)})

    print("GATE L (3-move leaf ranking on probe distances 7-11, margin over chance)")
    for a in cells.ARMS_TRAIN:
        m = [train[(a, s)]["gate_l"]["margin"] for s in SEEDS]
        print(f"  arm {a}: mean margin {st.mean(m):.4f} vs threshold {GATE_L_THRESHOLD[a]:.4f}, "
              f"p {one_sided_p(m):.4f} -> {gate_l[a]}")
    print(f"GATE E: {gate_e}")
    print("GATE R (top 3-move leaf closer than root on held-out states; policy alongside)")
    for d in RANK_DEPTHS:
        row = []
        for k in ("J-W", "J-A", "J-B", "P"):
            rows = rank[(k, d)]
            row.append(f"{k} {st.mean(r['hit'] for r in rows):.4f}/"
                       f"{st.mean(r['chance'] for r in rows):.4f}"
                       + (f" -> {gate_r[(k, d)]}" if k != "P" else " (reference)"))
        print(f"  d{d}: " + "  |  ".join(row))
    print(f"GATE 0(a) determinism: {'PASS' if args.determinism_ok else 'NOT CHECKED'}")
    print(f"GATE 0(b) continuity with EXP-072: {g0b}")
    print(f"GATE 0(c) concept path equals EXP-073 pilot 1: {g0c}")

    print("\nMean held-out success")
    for d in (*RANK_DEPTHS, 11):
        print(f"  d{d}: " + "  ".join(f"{a} {st.mean(rates[(a, d)].values()):.4f}"
                                      for a in (*J_ARMS, "P3V", "R3V")))

    gate0_ok = g0b == "PASS" and g0c == "PASS" and args.determinism_ok
    if not gate0_ok:
        print("Gate 0 did not pass. No claim may be read.")
        return
    gates = {"gate0": gate0_ok, "l": gate_l, "e": gate_e, "r": gate_r}
    full = primary_verdicts(rates, gates, SEEDS)
    sens_rates = sensitivity_rates(rates)
    sens = primary_verdicts(sens_rates, gates, sorted(sens_rates[("P3V", 9)]))
    for name, label in (("claim1", "CLAIM 1 (primary) d9 J3V-W - P3V"),
                        ("claim2", "CLAIM 2 (primary) d9 J3V-W - J3V-A")):
        v, p, diff, am, bm = full[name]
        sv, sp, sdiff, _, _ = sens[name]
        print(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} -> {v}")
        print(f"    sensitivity, seeds 0 and 3 dropped: diff {sdiff:+.4f}, p {sp:.4f} -> {sv}")
        if sv != v:
            print("    SENSITIVITY DISAGREES WITH THE VERDICT: RESULTS.md must lead with this.")

    print("\nSecondary (a pattern, never confirmations; labels for reference only):")
    for d in RANK_DEPTHS:
        pairs = [("J3V-A", "P3V"), ("J3V-A", "J3V-B")] if d == 9 else \
                [("J3V-W", "P3V"), ("J3V-W", "J3V-A")]
        for a, b in pairs:
            ok = gate0_ok and _arm_ok(gates, a, d) and (b not in _TRAIN_ARM or _arm_ok(gates, b, d))
            v, p, diff, am, bm = contrast(rates, (a, d), (b, d), ok)
            print(f"  d{d} {a} - {b}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} ({v})")

    print("\nExploratory, depth 11 (no claims):")
    print("  " + "  ".join(f"{a} {st.mean(rates[(a, 11)].values()):.4f}"
                           for a in (*J_ARMS, "P3V", "R3V")))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `launch074.ps1`.** Copy `experiments/073_learned_judge/launch073.ps1` and change exactly these things:
  1. `$exp = "experiments\074_wide_judge"`; every `exp073` in file names and the det folder becomes `exp074` (`C:\Users\mlgbr\exp074-det`).
  2. Replace the pilot/train settings block with:
     ```powershell
     # Spec section 4, fixed: the EXP-073 pilots ran exactly these for A and B.
     $Updates = 4000
     $SyncEvery = 100
     $ProbeEvery = 250
     $PilotOutDir = "experiments\074_wide_judge\outputs_pilot"
     ```
  3. The `$hasVi` probe checks the new seams instead: `sys.stdout.write(str('readout' in inspect.signature(v.Judge).parameters and 'reads_states' in inspect.getsource(l.imagined_values)))` with `from neuromorphic.training import value_iteration as v, lookahead as l`.
  4. The gate check reads `GATE_L_THRESHOLD` (not `GATE_T_THRESHOLD`) and still exempts only `pilot` and `cont`.
  5. Pilot args: `@("--pilot", "--arms", "W", "A", "B", "--seeds", "12", "13", "--n-updates", $Updates, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--out-dir", (Join-Path $repo $PilotOutDir), "--workers", $Workers)` and the pilot log is `phase_pilot.log`. Train args: `@("--arms", "W", "A", "B", "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "--n-updates", $Updates, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--workers", $Workers)`; remove the `$TrainUpdates` check.
  6. Phases: `rank` loops kinds `J-W, J-A, J-B, P` over depths 7, 8, 9; `cont` is P3V d8 and d9 s0 (unchanged); `eval` is `J3V-W, J3V-A, J3V-B` over depths 7, 8, 9, 11 plus `P3V, R3V` at depth 11; `det` is `J3V-W` and `J3V-A` at d9 s0 into the det folder.
  
  Read the result end to end once and confirm no `073` string remains except in comments.

- [ ] **Step 5: Run the tests** (Step 2's command), then the whole new file: `cd /root/projects/.wt/exp-074 && PYTHONPATH=src timeout 590 .venv/bin/python -m pytest tests/experiments/test_exp074.py -q -m "not slow"` (Bash timeout 600000). Expected: all PASS.

- [ ] **Step 6: Mutation checks** (restore after each): (a) `SENSITIVITY_DROP = (0, 4)` in cells: the sensitivity test must fail. (b) compare only `probes[-1]` in `gate0c_verdict`: the gate0c test must fail. (c) drop `_arm_ok(gates, "J3V-A", 9)` from `c2_ok`: the parametrized VOID test must fail. Report each.

- [ ] **Step 7: Commit.**

```bash
git add experiments/074_wide_judge/aggregate.py experiments/074_wide_judge/launch074.ps1 tests/experiments/test_exp074.py
git commit -m "EXP-074: aggregator with Gate L, per-arm Gate E, Gate 0(c), sensitivity line; laptop launcher"
```

---

### Task 7 (controller): merge, pilot, amendment, then the run

Not for a subagent.

- [ ] Whole-branch review; run, in separate foreground calls with explicit timeouts, `tests/training/test_value_iteration*.py tests/training/test_lookahead*.py`, `tests/experiments/test_exp073.py tests/experiments/test_exp074.py -m "not slow"`, then the slow test alone. Merge `--no-ff` as `Merge exp-074-wide-judge: ...`; update CLAUDE.md's test count from `--collect-only`; push.
- [ ] Laptop: `sync_repo.ps1`; scp `launch074.ps1` to `C:\Users\mlgbr\`; `-Phase check`; then `-Phase pilot -Workers 6` (6 runs: W, A, B x seeds 12, 13). Point `pilotstatus.ps1` at `outputs_pilot` with 6 expected records and the `exp074_train_*` glob; arm the guard.
- [ ] When the pilot completes: fetch the 6 records. **Gate 0(c)**: A and B probe histories equal EXP-073 pilot 1's exactly (run `gate0c_verdict` by hand). If it FAILS, stop: arms A and B are not EXP-073's arms; report to Michael before anything trains. Gate E on W (drift > 0). W's throughput.
- [ ] **Dated amendment** in the spec (section 11): every arm's pilot Gate L margin from the PRODUCTION instrument (these supersede section 6's diagnostic calibration for A and B if they differ), `GATE_L_THRESHOLD = {arm: half the mean}`, and the 20-worker time estimate. Set the constant in `aggregate.py`, commit, show Michael before `-Phase train`.
- [ ] Then the spec's order, steps 4 to 7: train (36 runs, guard re-armed per window), Gates L and E plus `-Phase rank` committed before any J eval cell, `-Phase cont`, `-Phase eval`, `-Phase det` (diff against full-run copies ignoring `wall_s` and `git_commit`), `aggregate.py --determinism-ok`, RESULTS.md (EXP-074, plus a short superseded note for EXP-073), roadmap entry, handoff, docs-session message, `secretary log`.
