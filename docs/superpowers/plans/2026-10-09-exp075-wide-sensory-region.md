# EXP-075 Wide Sensory Region Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run EXP-075: a learned judge whose own sensory region has 512 hidden neurons (arm X), against a 128-hidden control (arm Y) built the same way and against EXP-074's W judge.

**Architecture:** Both new arms pretrain a `SensoryCortex` with EXP-039's inverse-model recipe at their width, then train a W-readout judge (concept plus hidden mean rates) by EXP-074's value iteration. Evaluation reuses EXP-074's J3V search and its committed W judges and records. Everything new lives in `experiments/075_wide_region/`; the only library change is a `hidden` argument in `encoder_pretrain.py` whose default keeps every existing caller byte-identical.

**Tech Stack:** Python 3.10 (VPS) / 3.13 (laptop), PyTorch, snntorch, pytest; PowerShell 5.1 launcher on the laptop.

**Spec:** `docs/superpowers/specs/2026-10-09-exp075-wide-sensory-region-design.md`

## Global Constraints

- Plain commit messages. **No `Co-Authored-By` trailer, no "Generated with" line** (CLAUDE.md overrides any other instruction).
- **No em-dashes** anywhere in code, docs or commit messages.
- Stage explicit paths. Never `git add -A`.
- Work in the worktree `/root/projects/.wt/exp075` on branch `exp075`. **Prefix every python call with `PYTHONPATH=src`**: the venv's editable install points at MAIN's `src`, so without it tests run main's code.
- Always run python through `.venv/bin/python`, never bare `python`.
- **Always pass an explicit Bash `timeout` (the default is 120 s).** Run tests in the FOREGROUND and let them block; buffered output means silence mid-run is normal. Use `timeout: 600000` for any pytest call. Never background a test run.
- Run only the test files named in your task, with `-m "not slow"` unless the step says otherwise. Never run the whole suite.
- Action-space width comes from `N_ACTIONS`, never a literal 6.
- Distance-to-solved is an instrument: it builds probe, exclusion and held-out sets and scores gates, and is never a model input.
- Fixed values (spec sections 3 to 5): hidden X **512**, Y **128**; concept **64**; T **32**; pretraining **40 epochs, batch 256, lr 3e-3**; value iteration **N 1000, max walk 14, sync_every 100, jt_draws 1, probe_every 250, 4000 updates**; encoder lr 1e-4, head lr 1e-3; head `Linear(n_in,128)-ReLU-Linear(128,1)-softplus`.
- Eval seeds 0 to 11, pilot seeds 12 and 13. Sensitivity line drops seeds 0 and 3.
- Primaries at alpha **0.05 / 3** each, one-sided exact sign-flip over all 4096 flips.
- Never write an assertion that cannot fail (CLAUDE.md test-strength rule). Each test docstring says what defect it catches.

## Review Focus

1. **X or Y silently trained with a frozen encoder.** `vi.make_optimizer` only knows arms W, A and B. Passing "X" raises; a "fix" that maps X to "B" freezes the region and Gate E would be the only thing to notice. Task 3 tests that one update moves `fc1.weight` for both arms.
2. **An encoder of the wrong width loaded into an arm.** A 128-wide file under X's name must fail loudly, not produce a 128-wide "X". Task 2 tests the strict load refuses it.
3. **A relaunch after the 03:31 Windows Update reboot redoing finished work.** Pretraining and training must both skip cells whose record exists. Tasks 2 and 3 test `pending_jobs`.
4. **A smoke record (`limit_states`) or a record made at other settings entering a verdict.** Task 5 tests the aggregator refuses both, for evaluation, rank, training and pretraining records.
5. **A verdict read before the dated amendment sets the thresholds.** Task 5 tests the aggregator refuses while `GATE_P_THRESHOLD` or `GATE_L_THRESHOLD` is `None`; the launcher checks the same before every non-pilot phase.

---

### Task 1: `hidden` width in the pretraining library

**Files:**
- Modify: `src/neuromorphic/training/encoder_pretrain.py` (`make_sensory`, `load_encoder`, `PretrainConfig`, `train_inverse_model`)
- Test: `tests/training/test_encoder_pretrain.py`

**Interfaces:**
- Produces: `DEFAULT_HIDDEN = 128`; `make_sensory(seed, *, content=64, num_steps=32, hidden=DEFAULT_HIDDEN) -> SensoryCortex`; `load_encoder(path, *, seed=0, content=64, num_steps=32, hidden=DEFAULT_HIDDEN) -> SensoryCortex`; `PretrainConfig.hidden: int = DEFAULT_HIDDEN`; `train_inverse_model` builds its region at `cfg.hidden` when no `sensory` is passed.

- [ ] **Step 1: Write the failing tests** (append to `tests/training/test_encoder_pretrain.py`; reuse its existing imports and add any missing ones at the top of the file)

```python
def test_make_sensory_hidden_defaults_to_the_shipped_128_and_widens_on_request():
    """Catches a default that drifted from 128 (every existing pretrained encoder and the
    shipped brain are 128 wide) and a `hidden` argument that is accepted but ignored."""
    from neuromorphic.training.encoder_pretrain import DEFAULT_HIDDEN, make_sensory
    assert DEFAULT_HIDDEN == 128
    a, b = make_sensory(4), make_sensory(4, hidden=128)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    w = make_sensory(4, hidden=512)
    assert (w.fc1.out_features, w.fc2.in_features, w.fc2.out_features) == (512, 512, 64)


def test_load_encoder_refuses_a_file_of_another_width(tmp_path):
    """Catches a width mismatch loaded silently: strict loading must raise."""
    from neuromorphic.training.encoder_pretrain import load_encoder, make_sensory, save_encoder
    path = tmp_path / "enc.pt"
    save_encoder(make_sensory(1, hidden=128), path)
    with pytest.raises(RuntimeError):
        load_encoder(path, hidden=512)
    save_encoder(make_sensory(1, hidden=512), path)
    assert load_encoder(path, hidden=512).fc1.out_features == 512


def test_train_inverse_model_builds_its_region_at_cfg_hidden():
    """Catches `train_inverse_model` ignoring `cfg.hidden` (X would pretrain a 128 region)."""
    from neuromorphic.envs.cube import SOLVED
    from neuromorphic.training.encoder_pretrain import (PretrainConfig, build_pairs,
                                                        train_inverse_model)
    pairs = build_pairs([SOLVED])
    r512 = train_inverse_model(pairs, PretrainConfig(seed=0, epochs=1, batch_size=6, hidden=512))
    r128 = train_inverse_model(pairs, PretrainConfig(seed=0, epochs=1, batch_size=6))
    assert r512.sensory.fc1.out_features == 512
    assert r128.sensory.fc1.out_features == 128
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/training/test_encoder_pretrain.py -q -m "not slow" -k "hidden or width"`
Expected: 3 FAIL (ImportError on `DEFAULT_HIDDEN`, TypeError on `hidden=`).

- [ ] **Step 3: Implement**

In `encoder_pretrain.py`, next to `DEFAULT_CONTENT = 64` / `DEFAULT_T = 32`:

```python
DEFAULT_HIDDEN = 128   # the shipped region; every existing pretrained encoder is this wide
```

Replace `make_sensory`'s signature and return (keep its docstring, adding one sentence: "`hidden` (EXP-075) widens the first layer; its default is the shipped 128, so every existing caller is unchanged."):

```python
def make_sensory(seed: int, *, content: int = DEFAULT_CONTENT, num_steps: int = DEFAULT_T,
                 hidden: int = DEFAULT_HIDDEN) -> SensoryCortex:
    ...
    return SensoryCortex(n_obs=CUBE_N_OBS, hidden=hidden, concept=content, num_steps=num_steps,
                         seed=seed)
```

`load_encoder` gains `hidden: int = DEFAULT_HIDDEN` and passes it: `sensory = make_sensory(seed, content=content, num_steps=num_steps, hidden=hidden)`.

`PretrainConfig` gains a field after `num_steps`: `hidden: int = DEFAULT_HIDDEN`.

In `train_inverse_model`, the default construction becomes:

```python
    sensory = sensory if sensory is not None else make_sensory(cfg.seed, content=cfg.content,
                                                              num_steps=cfg.num_steps,
                                                              hidden=cfg.hidden)
```

- [ ] **Step 4: Run the whole file to verify the new tests pass and nothing else changed**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/training/test_encoder_pretrain.py tests/training/test_pretrain.py -q -m "not slow"`
Expected: all PASS, including the existing `test_make_sensory_matches_the_shipped_brain_encoder`.

- [ ] **Step 5: Mutation check.** Temporarily change `hidden=cfg.hidden` to `hidden=DEFAULT_HIDDEN` in `train_inverse_model`, re-run Step 2's command, confirm `test_train_inverse_model_builds_its_region_at_cfg_hidden` FAILS, then restore and confirm `git diff` shows only the intended change.

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/training/encoder_pretrain.py tests/training/test_encoder_pretrain.py
git commit -m "encoder_pretrain: hidden width argument, default 128 unchanged (EXP-075)"
```

---

### Task 2: EXP-075 cells and the pretraining driver

**Files:**
- Create: `experiments/075_wide_region/cells.py`
- Create: `experiments/075_wide_region/pretrain.py`
- Test: `tests/experiments/test_exp075.py` (new)

**Interfaces:**
- Consumes (Task 1): `make_sensory(..., hidden=)`, `load_encoder(..., hidden=)`, `save_encoder`, `PretrainConfig(hidden=)`, `train_inverse_model`, `build_pairs`.
- Produces, in `cells.py`: `EVAL_SEEDS`, `PILOT_SEEDS`, `EVAL_DEPTHS`, `SENSITIVITY_DROP`, `ARMS_TRAIN = ("X", "Y")`, `HIDDEN = {"X": 512, "Y": 128}`, `CONTENT = 64`, `T = 32`, `E74_OUT: Path`, `c70`, `c74`, `heldout_states`, `exclusion_set`, `probe_set`, `check_seeds(seeds, pilot) -> None`, `encoder_path(arm, seed, out_dir=None) -> Path`, `pretrain_record_name(arm, seed) -> str`, `ckpt_dir(arm, seed, out_dir=None) -> Path`, `record_name(arm, seed) -> str`, `fresh_sensory(arm, seed) -> SensoryCortex`, `load_pretrained(arm, seed, out_dir=None) -> SensoryCortex`, `build_judge(seed, arm, sensory) -> vi.Judge`, `make_judge(seed, arm, out_dir=None) -> vi.Judge`.
- Produces, in `pretrain.py`: `EPOCHS = 40`, `BATCH = 256`, `LR = 3e-3`, `e39` (EXP-039's run module), `pretrain_states(provider7) -> (list, list)`, `probe_heldout(states, depths, seed) -> set`, `pretrain_pairs(seed, states, depths, excl) -> (pairs, forbidden)`, `run(arm, seed, out_dir, epochs=EPOCHS) -> dict`, `pending_jobs(arms, seeds, out_dir) -> list[tuple[str, int]]`, `build_parser()`, `main()`. Pretraining record JSON keys: `arm, seed, hidden, epochs, batch_size, lr, n_pairs, n_pairs_exp039_recipe, n_forbidden, history, final_move_accuracy, wall_s, git_commit`.

- [ ] **Step 1: Write the failing tests** (`tests/experiments/test_exp075.py`)

```python
"""EXP-075: cells, pretraining, training driver, evaluation and aggregator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.encoder_pretrain import make_sensory, save_encoder

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "075_wide_region"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp075_cells", "cells.py")
pretrain = _load("exp075_pretrain", "pretrain.py")


def _fake_encoder(out_dir, arm, seed, hidden=None):
    path = cells.encoder_path(arm, seed, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder(make_sensory(seed, hidden=hidden or cells.HIDDEN[arm]), path)
    return path


def test_arms_widths_and_seeds_are_the_spec_values():
    """Catches a width, arm list or seed set drifting from spec sections 3 and 6."""
    assert cells.ARMS_TRAIN == ("X", "Y")
    assert cells.HIDDEN == {"X": 512, "Y": 128}
    assert (cells.CONTENT, cells.T) == (64, 32)
    assert tuple(cells.EVAL_SEEDS) == tuple(range(12)) and tuple(cells.PILOT_SEEDS) == (12, 13)
    assert tuple(cells.SENSITIVITY_DROP) == (0, 3)


def test_check_seeds_refuses_misuse():
    """Catches a pilot run on an evaluation seed, or an evaluation run on a pilot seed."""
    with pytest.raises(SystemExit):
        cells.check_seeds([0], pilot=True)
    with pytest.raises(SystemExit):
        cells.check_seeds([12], pilot=False)
    cells.check_seeds([12, 13], pilot=True)
    cells.check_seeds(list(range(12)), pilot=False)


def test_judges_have_the_wide_readout_at_their_width(tmp_path):
    """Catches X built at 128, a concept-only readout, or a head of the wrong input width.
    Head sizes from spec section 3: X 73,985 parameters (576 inputs), Y 24,833 (192)."""
    for arm, n_in, n_head in (("X", 576, 73_985), ("Y", 192, 24_833)):
        _fake_encoder(tmp_path, arm, 0)
        j = cells.make_judge(0, arm, tmp_path)
        assert j.readout == "wide"
        assert j.head[0].in_features == n_in
        assert sum(p.numel() for p in j.head.parameters()) == n_head


def test_make_judge_starts_from_the_pretrained_file_exactly(tmp_path):
    """Catches a judge that starts from a fresh random region instead of its pretrained one."""
    path = _fake_encoder(tmp_path, "X", 2)
    ref = torch.load(path)
    sens = cells.make_judge(2, "X", tmp_path).sensory.state_dict()
    for k, v in ref.items():
        assert torch.equal(sens[k], v), k
    fresh = cells.fresh_sensory("X", 3).state_dict()
    assert not torch.equal(fresh["fc1.weight"], ref["fc1.weight"])


def test_head_init_depends_only_on_the_seed(tmp_path):
    """Catches a head whose init depends on global RNG state left by earlier construction."""
    _fake_encoder(tmp_path, "Y", 5)
    torch.manual_seed(999)
    a = cells.make_judge(5, "Y", tmp_path).head.state_dict()
    torch.randn(100)
    b = cells.make_judge(5, "Y", tmp_path).head.state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_wrong_width_encoder_is_refused(tmp_path):
    """Catches a 128-wide file under X's name loading as a 128-wide 'X' (Review Focus 2)."""
    _fake_encoder(tmp_path, "X", 0, hidden=128)
    with pytest.raises(RuntimeError):
        cells.make_judge(0, "X", tmp_path)


def test_missing_encoder_exits_naming_the_file(tmp_path):
    """Catches a judge built on a random region because pretraining never ran."""
    with pytest.raises(SystemExit, match="enc_X_s0.pt"):
        cells.make_judge(0, "X", tmp_path)


def test_pretrain_states_and_pairs_without_exclusion_are_exp039s_recipe():
    """Catches the pretraining set drifting from EXP-039's (Y must be that recipe at 128).
    Checked against EXP-039's own recorded counts, not a re-implementation."""
    rec_path = REPO / "experiments" / "039_encoder_pretraining" / "outputs" / "exp039_s0.json"
    if not rec_path.exists():
        pytest.skip("EXP-039 records are untracked; present on the VPS")
    rec = json.loads(rec_path.read_text())
    prov7 = ExactBFSDistance(max_depth=7)
    states, depths = pretrain.pretrain_states(prov7)
    s39, _masks, d39 = pretrain.e39.build_dataset(prov7)
    assert states == s39 and depths == d39
    assert len(pretrain.probe_heldout(states, depths, 0)) == rec["n_heldout"]
    pairs, _forbidden = pretrain.pretrain_pairs(0, states, depths, set())
    assert len(pairs) == rec["n_pairs"]


def test_an_excluded_evaluation_state_removes_exactly_its_pairs():
    """Catches the evaluation exclusion being ignored, or applied to sources only."""
    prov7 = ExactBFSDistance(max_depth=7)
    states, depths = pretrain.pretrain_states(prov7)
    base, _ = pretrain.pretrain_pairs(0, states, depths, set())
    target = next(p[2] for p in base if prov7.distance(p[2]) == 7)
    pairs, forbidden = pretrain.pretrain_pairs(0, states, depths, {target})
    assert target in forbidden
    assert all(target not in (p[0], p[2]) for p in pairs)
    dropped = sum(1 for p in base if p[2] == target)
    assert dropped >= 1 and len(base) - len(pairs) == dropped


def test_pretrain_relaunch_skips_cells_whose_record_exists(tmp_path):
    """Catches a relaunch that re-pretrains a finished seed (Review Focus 3)."""
    (tmp_path / cells.pretrain_record_name("Y", 12)).write_text("{}")
    assert pretrain.pending_jobs(["X", "Y"], [12, 13], tmp_path) == [
        ("X", 12), ("X", 13), ("Y", 13)]


def test_pretrain_parser_defaults_are_the_spec_recipe():
    """Catches a hand-run pretraining at other than EXP-039's 40 epochs."""
    a = pretrain.build_parser().parse_args([])
    assert a.epochs == 40 and a.arms == ["X", "Y"]
    assert (pretrain.EPOCHS, pretrain.BATCH, pretrain.LR) == (40, 256, 3e-3)


@pytest.mark.slow
def test_pretrain_run_writes_an_encoder_at_the_arm_width(tmp_path):
    """Catches a record without its accuracy history, an encoder saved at the wrong width, or
    an encoder that training never moved. Slow: builds the full BFS table and runs 1 epoch."""
    rec = pretrain.run("X", 12, tmp_path, epochs=1)
    on_disk = json.loads((tmp_path / cells.pretrain_record_name("X", 12)).read_text())
    assert on_disk["hidden"] == 512 and on_disk["epochs"] == 1
    assert len(on_disk["history"]) == 1
    assert on_disk["final_move_accuracy"] == on_disk["history"][-1]["accuracy"]
    assert on_disk["n_pairs"] <= on_disk["n_pairs_exp039_recipe"]
    trained = cells.load_pretrained("X", 12, tmp_path)
    assert trained.fc1.out_features == 512
    untrained = cells.fresh_sensory("X", 12)
    assert not torch.equal(trained.fc1.weight, untrained.fc1.weight)
    assert rec["final_move_accuracy"] == on_disk["final_move_accuracy"]
```

- [ ] **Step 2: Run them to verify they fail**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp075.py -q -m "not slow"`
Expected: collection error (no `experiments/075_wide_region/cells.py`).

- [ ] **Step 3: Write `experiments/075_wide_region/cells.py`**

```python
"""EXP-075 cells: arms X (hidden 512) and Y (hidden 128). Each starts from its own region,
pretrained by EXP-039's inverse-model recipe at that width (pretrain.py), and becomes a judge with
EXP-074's W readout (concept plus hidden mean rates). EXP-074's held-out, exclusion and probe sets
are reused unchanged, and W is EXP-074's committed judge.

Spec: docs/superpowers/specs/2026-10-09-exp075-wide-sensory-region-design.md, sections 3 and 4.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

_spec = importlib.util.spec_from_file_location(
    "exp074_cells", REPO / "experiments" / "074_wide_judge" / "cells.py")
c74 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c74)
c70 = c74.c70

from neuromorphic.encoders import cube_encoder  # noqa: E402
from neuromorphic.regions.sensory_cortex import SensoryCortex  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402
from neuromorphic.training.encoder_pretrain import load_encoder, make_sensory  # noqa: E402

EVAL_SEEDS = c74.EVAL_SEEDS
PILOT_SEEDS = c74.PILOT_SEEDS
EVAL_DEPTHS = c74.EVAL_DEPTHS
SENSITIVITY_DROP = c74.SENSITIVITY_DROP
ARMS_TRAIN = ("X", "Y")
HIDDEN = {"X": 512, "Y": 128}
CONTENT = 64
T = 32
E74_OUT = REPO / "experiments" / "074_wide_judge" / "outputs"

heldout_states = c74.heldout_states
exclusion_set = c74.exclusion_set
probe_set = c74.probe_set


def _check_arm(arm: str) -> None:
    if arm not in ARMS_TRAIN:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS_TRAIN}")


def check_seeds(seeds, pilot: bool) -> None:
    allowed = PILOT_SEEDS if pilot else EVAL_SEEDS
    bad = [s for s in seeds if s not in allowed]
    if bad:
        raise SystemExit(f"seeds {bad} not allowed with pilot={pilot}; allowed seeds are {allowed}")


def _out(out_dir) -> Path:
    return Path(out_dir) if out_dir is not None else HERE / "outputs"


def encoder_path(arm: str, seed: int, out_dir=None) -> Path:
    return _out(out_dir) / "encoders" / f"enc_{arm}_s{seed}.pt"


def pretrain_record_name(arm: str, seed: int) -> str:
    return f"exp075_pretrain_{arm}_s{seed}.json"


def ckpt_dir(arm: str, seed: int, out_dir=None) -> Path:
    return _out(out_dir) / f"judge_{arm}_s{seed}"


def record_name(arm: str, seed: int) -> str:
    return f"exp075_train_{arm}_s{seed}.json"


def fresh_sensory(arm: str, seed: int) -> SensoryCortex:
    """An untrained region at the arm's width: the shape a saved judge or encoder loads into."""
    _check_arm(arm)
    return make_sensory(seed, content=CONTENT, num_steps=T, hidden=HIDDEN[arm])


def load_pretrained(arm: str, seed: int, out_dir=None) -> SensoryCortex:
    """The arm's pretrained region. Strict load, so a file of another width raises."""
    _check_arm(arm)
    path = encoder_path(arm, seed, out_dir)
    if not path.exists():
        raise SystemExit(f"missing pretrained encoder {path} for arm {arm} seed {seed}; "
                         f"run pretrain.py first")
    return load_encoder(path, seed=seed, content=CONTENT, num_steps=T, hidden=HIDDEN[arm])


def build_judge(seed: int, arm: str, sensory: SensoryCortex) -> vi.Judge:
    """A W-readout judge over `sensory`. The head is seeded right before it is built (as in
    EXP-073 and EXP-074), so its init depends only on `seed`."""
    _check_arm(arm)
    torch.manual_seed(seed)
    return vi.Judge(cube_encoder(), sensory, T, CONTENT, readout="wide")


def make_judge(seed: int, arm: str, out_dir=None) -> vi.Judge:
    return build_judge(seed, arm, load_pretrained(arm, seed, out_dir))
```

- [ ] **Step 4: Write `experiments/075_wide_region/pretrain.py`**

```python
"""EXP-075 pretraining driver: EXP-039's inverse-model recipe at each arm's width.

Unchanged from EXP-039 except (spec section 4): the region's hidden width, and a forbidden set
that adds every state in the seed's evaluation held-out sets (`cells.exclusion_set`) to EXP-039's
probe held-out states. Self-supervised: the move is known because it was applied; no distance
label enters. The BFS table only builds the state and forbidden sets.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/075_wide_region/pretrain.py --pilot --seeds 12 13 \
        --out-dir experiments/075_wide_region/outputs_pilot --workers 4
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
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp075_cells", HERE / "cells.py")
e39 = _load("exp039_run", REPO / "experiments" / "039_encoder_pretraining" / "run.py")

from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training.encoder_pretrain import (  # noqa: E402
    PretrainConfig, build_pairs, save_encoder, train_inverse_model,
)

torch.set_num_threads(1)

EPOCHS = 40
BATCH = 256
LR = 3e-3
DEPTHS = e39.DEPTHS
FRAC_HELDOUT = e39.FRAC_HELDOUT
OUT_DIR_DEFAULT = HERE / "outputs"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def pretrain_states(provider7):
    """Every state at EXP-039's depths with its depth, in `e39.build_dataset`'s order (without
    the optimal-move masks, which only EXP-039's probe needs)."""
    states, depths = [], []
    for d in DEPTHS:
        for s in provider7.states_at_distance(d):
            states.append(s)
            depths.append(d)
    return states, depths


def probe_heldout(states, depths, seed) -> set:
    """EXP-039's probe held-out states for `seed`, stratified by depth exactly as its run_one."""
    held = set()
    for d in DEPTHS:
        pool = [i for i, dd in enumerate(depths) if dd == d]
        _train, he = e39.probe.split_states(pool, FRAC_HELDOUT, seed)
        held.update(states[i] for i in he)
    return held


def pretrain_pairs(seed, states, depths, excl):
    """(pairs, forbidden): EXP-039's pairs with `excl` also forbidden at both endpoints."""
    forbidden = probe_heldout(states, depths, seed) | set(excl)
    pairs = build_pairs(states, forbidden=forbidden)
    assert not (forbidden & {p[0] for p in pairs}), "forbidden state leaked in as a source"
    assert not (forbidden & {p[2] for p in pairs}), "forbidden state leaked in as a successor"
    return pairs, forbidden


def run(arm, seed, out_dir, epochs=EPOCHS) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    out_dir = Path(out_dir)
    # One depth beyond the deepest source state, as EXP-039: successors of depth-6 states are
    # depth 7. The full table builds the evaluation exclusion set (depths 7, 8, 9, 11).
    prov7 = ExactBFSDistance(max_depth=max(DEPTHS) + 1)
    excl = cells.exclusion_set(seed, ExactBFSDistance(max_depth=None))
    states, depths = pretrain_states(prov7)
    pairs, forbidden = pretrain_pairs(seed, states, depths, excl)
    n39 = len(build_pairs(states, forbidden=probe_heldout(states, depths, seed)))
    cfg = PretrainConfig(seed=seed, epochs=epochs, batch_size=BATCH, lr=LR,
                         hidden=cells.HIDDEN[arm])
    res = train_inverse_model(pairs, cfg)
    path = cells.encoder_path(arm, seed, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder(res.sensory, path)
    rec = {
        "arm": arm, "seed": seed, "hidden": cells.HIDDEN[arm], "epochs": epochs,
        "batch_size": BATCH, "lr": LR, "n_pairs": len(pairs), "n_pairs_exp039_recipe": n39,
        "n_forbidden": len(forbidden), "history": res.history,
        "final_move_accuracy": res.final_accuracy, "wall_s": round(time.time() - t0, 1),
        "git_commit": _git_commit(),
    }
    (out_dir / cells.pretrain_record_name(arm, seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def pending_jobs(arms, seeds, out_dir) -> list[tuple[str, int]]:
    """(arm, seed) pairs, arms then seeds, whose pretraining record does not exist yet."""
    return [(a, s) for a in arms for s in seeds
            if not (Path(out_dir) / cells.pretrain_record_name(a, s)).exists()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(cells.ARMS_TRAIN))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    bad = [a for a in args.arms if a not in cells.ARMS_TRAIN]
    if bad:
        raise SystemExit(f"arms {bad} outside {cells.ARMS_TRAIN}")
    cells.check_seeds(args.seeds, args.pilot)
    jobs = pending_jobs(args.arms, args.seeds, args.out_dir)
    print(f"EXP-075 pretraining: {len(jobs)} runs, {args.workers} workers, pilot={args.pilot}, "
          f"epochs={args.epochs}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, a, s, args.out_dir, args.epochs): (a, s) for a, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  arm {r['arm']} seed {r['seed']}  hidden {r['hidden']}  "
                  f"pairs {r['n_pairs']}/{r['n_pairs_exp039_recipe']}  "
                  f"move-acc {r['final_move_accuracy']:.4f}  wall_s {r['wall_s']:.0f}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp075.py -q -m "not slow"`
Expected: all PASS (the EXP-039 recipe test passes on the VPS, where the records exist).

- [ ] **Step 6: Run the slow pretraining test once, in the foreground**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp075.py -q -m slow -k pretrain_run`
Expected: PASS. Record its wall time in your report (it prices the pilot).

- [ ] **Step 7: Mutation checks.** One at a time, apply, run Step 5's command, confirm the named test FAILS, restore:
  - `forbidden = probe_heldout(states, depths, seed)` (drop `| set(excl)`) -> `test_an_excluded_evaluation_state_removes_exactly_its_pairs`.
  - `HIDDEN = {"X": 128, "Y": 128}` -> `test_judges_have_the_wide_readout_at_their_width`.
  - In `build_judge`, move `torch.manual_seed(seed)` to after the `vi.Judge(...)` line -> `test_head_init_depends_only_on_the_seed`.
  Confirm `git diff` is clean of mutations afterwards.

- [ ] **Step 8: Commit**

```bash
git add experiments/075_wide_region/cells.py experiments/075_wide_region/pretrain.py tests/experiments/test_exp075.py
git commit -m "EXP-075: cells and the inverse-model pretraining driver at each arm's width"
```

---

### Task 3: Value-iteration training driver

**Files:**
- Create: `experiments/075_wide_region/train.py`
- Test: `tests/experiments/test_exp075.py` (append)

**Interfaces:**
- Consumes (Task 2): `cells.make_judge(seed, arm, out_dir)`, `cells.load_pretrained`, `cells.exclusion_set`, `cells.probe_set`, `cells.ckpt_dir`, `cells.record_name`, `cells.check_seeds`, `cells.HIDDEN`. EXP-074's `gate_l.leaf_rank_margin(judge, probe, distance, seed, n_actions) -> dict` (keys include `margin`, `n`, `chance`).
- Produces: `VI_ARM = "W"`, `MAX_LEN = 14`, `run(arm, seed, n_updates, batch, sync_every, probe_every, draws, out_dir) -> dict`, `pending_jobs(arms, seeds, out_dir)`, `build_parser()`, `main()`. Training record JSON: every key `vi.train_judge` returns, plus `arm, seed, hidden, readout, n_updates, batch, max_len, n_actions, sync_every, probe_every, draws, n_exclude, n_probe, encoder_drift, gate_l, git_commit`. Encoders are read from `out_dir/encoders/`; judges are banked at `out_dir/judge_{arm}_s{seed}/`.

- [ ] **Step 1: Write the failing tests** (append to `tests/experiments/test_exp075.py`)

```python
train = _load("exp075_train", "train.py")


def test_train_parser_defaults_are_the_spec_settings():
    """Catches driver defaults drifting from spec section 5."""
    a = train.build_parser().parse_args(["--n-updates", "1"])
    assert (a.batch, a.sync_every, a.probe_every, a.draws) == (1000, 100, 250, 1)
    assert a.arms == ["X", "Y"] and train.MAX_LEN == 14


def test_one_update_moves_the_region_for_both_arms(tmp_path):
    """REVIEW FOCUS 1. Catches X or Y trained with a frozen region: the gradient must ARRIVE at
    fc1 (CLAUDE.md: verify the parameter moved, not that a switch is set)."""
    import random
    from neuromorphic.training import value_iteration as vi
    for arm in ("X", "Y"):
        _fake_encoder(tmp_path, arm, 0)
        judge = cells.make_judge(0, arm, tmp_path)
        before = judge.sensory.fc1.weight.detach().clone()
        opt = vi.make_optimizer(judge, train.VI_ARM)
        jt = vi.sync_target(judge)
        states = vi.random_walk_states(8, 14, random.Random(0), N_ACTIONS, set())
        vi.train_step(judge, jt, opt, states, N_ACTIONS, torch.Generator().manual_seed(0))
        assert not torch.equal(before, judge.sensory.fc1.weight.detach()), arm
        lrs = sorted(g["lr"] for g in opt.param_groups)
        assert lrs == [1e-4, 1e-3], arm


def test_train_refuses_a_missing_encoder_before_building_anything(tmp_path):
    """Catches training on a random region when pretraining was skipped, and catches the check
    running only after the 65 s BFS build."""
    import time
    t0 = time.time()
    with pytest.raises(SystemExit, match="enc_Y_s12.pt"):
        train.run("Y", 12, 1, 8, 1, 1, 1, tmp_path)
    assert time.time() - t0 < 20


def test_train_relaunch_skips_runs_whose_record_exists(tmp_path):
    """Catches a relaunch that resubmits finished runs (Review Focus 3)."""
    (tmp_path / cells.record_name("X", 13)).write_text("{}")
    assert train.pending_jobs(["X", "Y"], [12, 13], tmp_path) == [
        ("X", 12), ("Y", 12), ("Y", 13)]


@pytest.mark.slow
def test_run_writes_gate_l_drift_width_and_readout(tmp_path):
    """Catches a record missing Gate L, the drift Gate E reads, or the width. Slow: builds the
    full BFS table, as the real driver does."""
    _fake_encoder(tmp_path, "X", 12)
    rec = train.run("X", 12, 2, 8, 1, 1, 1, tmp_path)
    on_disk = json.loads((tmp_path / cells.record_name("X", 12)).read_text())
    assert on_disk["readout"] == "wide" and on_disk["hidden"] == 512
    assert on_disk["gate_l"]["n"] == 250
    assert 0.05 < on_disk["gate_l"]["chance"] < 0.6
    assert on_disk["encoder_drift"] > 0.0
    assert (tmp_path / "judge_X_s12" / "judge.pt").exists()
    assert rec["gate_l"] == on_disk["gate_l"]
```

- [ ] **Step 2: Run them to verify they fail**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp075.py -q -m "not slow" -k "train or update or refuses_a_missing"`
Expected: collection error (no `train.py`).

- [ ] **Step 3: Write `experiments/075_wide_region/train.py`**

```python
"""EXP-075 training driver: arms X and Y by EXP-074's value iteration, then Gate L.

Each run starts from its arm's pretrained encoder in `out_dir/encoders/` (pretrain.py). Builds
the full BFS table once per process for the exclusion and probe sets and as Gate L's yardstick;
training never reads it.

Spec: docs/superpowers/specs/2026-10-09-exp075-wide-sensory-region-design.md, sections 5 to 7.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/075_wide_region/train.py --pilot --seeds 12 13 \
        --n-updates 4000 --out-dir experiments/075_wide_region/outputs_pilot --workers 4
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
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp075_cells", HERE / "cells.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

torch.set_num_threads(1)

MAX_LEN = 14  # 2x2 quarter-turn diameter
# `vi.make_optimizer` names arms by how their encoder is treated. X and Y train the encoder
# exactly as EXP-074's W did (encoder lr 1e-4, head lr 1e-3), so they use W's optimizer.
VI_ARM = "W"
OUT_DIR_DEFAULT = HERE / "outputs"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _encoder_drift(judge, arm, seed, out_dir) -> float:
    e0 = cells.load_pretrained(arm, seed, out_dir).state_dict()
    cur = judge.sensory.state_dict()
    return sum(float(((cur[k] - e0[k]) ** 2).sum()) for k in e0) ** 0.5


def run(arm, seed, n_updates, batch, sync_every, probe_every, draws, out_dir) -> dict:
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    judge = cells.make_judge(seed, arm, out_dir)  # first: a missing encoder fails before the BFS
    provider = ExactBFSDistance(max_depth=None)
    excl = cells.exclusion_set(seed, provider)
    probe = cells.probe_set(seed, provider)
    result = vi.train_judge(
        judge, VI_ARM, n_updates=n_updates, batch=batch, max_len=MAX_LEN, n_actions=N_ACTIONS,
        exclude=excl, probe=probe, seed=seed, sync_every=sync_every, probe_every=probe_every,
        draws=draws, ckpt_dir=cells.ckpt_dir(arm, seed, out_dir),
    )
    judge.eval()
    rec = {
        **result,
        "arm": arm, "seed": seed, "hidden": cells.HIDDEN[arm], "readout": judge.readout,
        "n_updates": n_updates, "batch": batch, "max_len": MAX_LEN, "n_actions": N_ACTIONS,
        "sync_every": sync_every, "probe_every": probe_every, "draws": draws,
        "n_exclude": len(excl), "n_probe": len(probe),
        "encoder_drift": _encoder_drift(judge, arm, seed, out_dir),
        "gate_l": gate_l.leaf_rank_margin(judge, probe, provider.distance, seed, N_ACTIONS),
        "git_commit": _git_commit(),
    }
    (out_dir / cells.record_name(arm, seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def pending_jobs(arms, seeds, out_dir) -> list[tuple[str, int]]:
    """(arm, seed) pairs, SEEDS then arms, whose record does not exist yet. Seed-major so the
    first wave mixes X (heavy) and Y (light) workers instead of starting with every X run."""
    return [(a, s) for s in seeds for a in arms
            if not (Path(out_dir) / cells.record_name(a, s)).exists()]


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
    bad = [a for a in args.arms if a not in cells.ARMS_TRAIN]
    if bad:
        raise SystemExit(f"arms {bad} outside {cells.ARMS_TRAIN}")
    cells.check_seeds(args.seeds, args.pilot)
    jobs = pending_jobs(args.arms, args.seeds, args.out_dir)
    print(f"EXP-075: {len(jobs)} training runs, {args.workers} workers, pilot={args.pilot}, "
          f"n_updates={args.n_updates}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, a, s, args.n_updates, args.batch, args.sync_every,
                            args.probe_every, args.draws, args.out_dir): (a, s) for a, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            loss = "None" if r["loss_last"] is None else format(r["loss_last"], ".6f")
            print(f"  {i}/{len(jobs)}  arm {r['arm']} seed {r['seed']}  updates {r['updates']}  "
                  f"loss_last {loss}  wall_s {r['wall_s']:.1f}  "
                  f"encoder_drift {r['encoder_drift']:.6f}  "
                  f"gate_l margin {r['gate_l']['margin']:.4f}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
```

Note the test in Step 1 expects seed-major order: `[("X", 12), ("Y", 12), ("Y", 13)]` after `("X", 13)` is done. That ordering is deliberate (see the docstring).

- [ ] **Step 4: Run the tests to verify they pass**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp075.py -q -m "not slow"`
Expected: all PASS.

- [ ] **Step 5: Run the slow training test once, in the foreground**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp075.py -q -m slow -k gate_l_drift`
Expected: PASS.

- [ ] **Step 6: Mutation checks.** One at a time, apply, re-run Step 4's command, confirm the named test FAILS, restore:
  - `VI_ARM = "B"` -> `test_one_update_moves_the_region_for_both_arms`.
  - Move `judge = cells.make_judge(...)` below the `ExactBFSDistance(...)` line -> `test_train_refuses_a_missing_encoder_before_building_anything`.
  Confirm `git diff` is clean of mutations afterwards.

- [ ] **Step 7: Commit**

```bash
git add experiments/075_wide_region/train.py tests/experiments/test_exp075.py
git commit -m "EXP-075: value-iteration training driver for arms X and Y"
```

---

### Task 4: Evaluation

**Files:**
- Create: `experiments/075_wide_region/evaluate.py`
- Test: `tests/experiments/test_exp075.py` (append)

**Interfaces:**
- Consumes (Task 2): `cells.build_judge`, `cells.fresh_sensory`, `cells.ckpt_dir`, `cells.E74_OUT`. EXP-074's `evaluate.py`: `JudgeCritic(judge)`, `load_cell(depth, seed) -> (agent, head, states, train_seed)` (depths 7, 8, 9, 11), `load_judge("W", seed, judge_dir)`, `run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None)`. Library: `evaluate_lookahead`, `imag_seed_for`, `imagined_values`, `tree_levels`.
- Produces: `ARMS_EVAL = ("J3V-X", "J3V-Y", "J3V-W")`, `RANK_KINDS = ("J-X", "J-Y")`, `load_judge(train_arm, seed, judge_dir=None)`, `run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict` writing `exp075_{arm}_d{depth}_s{seed}.json`, `rank_cell(kind, depth, seed, out_dir, judge_dir=None, limit_states=None) -> dict` writing `exp075_rank_{kind}_d{depth}_s{seed}.json` with keys `kind, depth, seed, n, hit, chance, limit_states`. J3V-W always loads EXP-074's committed judge from `cells.E74_OUT`.

- [ ] **Step 1: Write the failing tests** (append)

```python
evaluate = _load("exp075_evaluate", "evaluate.py")


def _fake_judge_ckpt(jdir, arm, seed):
    path = cells.ckpt_dir(arm, seed, jdir)
    path.mkdir(parents=True)
    judge = cells.build_judge(seed, arm, cells.fresh_sensory(arm, seed))
    torch.save(judge.state_dict(), path / "judge.pt")
    return judge


def test_eval_arms_and_rank_kinds_are_the_spec_lists():
    assert list(evaluate.ARMS_EVAL) == ["J3V-X", "J3V-Y", "J3V-W"]
    assert list(evaluate.RANK_KINDS) == ["J-X", "J-Y"]


def test_judge_loads_its_checkpoint_not_its_pretrained_encoder(tmp_path):
    """Catches an evaluator that rebuilds the judge from the pretrained file and evaluates an
    untrained head, or needs the encoder file to exist at evaluation time."""
    saved = _fake_judge_ckpt(tmp_path, "X", 0).state_dict()
    loaded = evaluate.load_judge("X", 0, tmp_path).state_dict()
    for k, v in saved.items():
        assert torch.equal(loaded[k], v), k


def test_j3v_w_is_exp074s_committed_judge():
    """THE GATE 0(b) PRECONDITION. Catches W rebuilt or retrained instead of reused."""
    ref = torch.load(cells.E74_OUT / "judge_W_s0" / "judge.pt", map_location="cpu")
    mine = evaluate.load_judge("W", 0).state_dict()
    assert mine.keys() == ref.keys()
    for k in ref:
        assert torch.equal(mine[k], ref[k]), k


def test_missing_checkpoint_exits_naming_it(tmp_path):
    """Catches a J cell silently evaluating an untrained judge."""
    with pytest.raises(SystemExit, match="judge.pt"):
        evaluate.run_cell("J3V-Y", 8, 0, tmp_path, limit_states=1, judge_dir=tmp_path / "no")


def test_an_x_judge_drives_a_real_j3v_cell(tmp_path):
    """Catches X evaluated through the 64-unit concept: a 576-input head cannot read it."""
    jdir = tmp_path / "judges"
    _fake_judge_ckpt(jdir, "X", 0)
    rec = evaluate.run_cell("J3V-X", 7, 0, tmp_path / "out", limit_states=1, judge_dir=jdir)
    assert rec["arm"] == "J3V-X" and rec["n"] == 1 and rec["mode"] == "C"
    assert rec["limit_states"] == 1
    assert (tmp_path / "out" / "exp075_J3V-X_d7_s0.json").exists()


def test_j3v_w_cell_equals_exp074s_harness_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH: J3V-W through this harness equals EXP-074's own run_cell on
    the same states, in every outcome field."""
    mine = evaluate.run_cell("J3V-W", 9, 0, tmp_path / "a", limit_states=2)
    ref = evaluate.e74.run_cell("J3V-W", 9, 0, tmp_path / "b", limit_states=2,
                                judge_dir=cells.E74_OUT)
    for f in ("solved", "n", "success_rate", "mean_steps", "eval_revisit_rate"):
        assert mine[f] == ref[f], f


def test_rank_cell_records_hit_chance_and_its_limit(tmp_path):
    """Catches a rank record without `limit_states` (so a smoke row could enter Gate R), and a
    chance that is not the fraction of closer leaves."""
    jdir = tmp_path / "judges"
    _fake_judge_ckpt(jdir, "Y", 0)
    rec = evaluate.rank_cell("J-Y", 7, 0, tmp_path / "out", judge_dir=jdir, limit_states=2)
    assert rec["n"] == 2 and rec["limit_states"] == 2
    assert 0.0 < rec["chance"] < 0.5 and rec["hit"] in (0.0, 0.5, 1.0)
    assert (tmp_path / "out" / "exp075_rank_J-Y_d7_s0.json").exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp075.py -q -m "not slow" -k "eval or judge or rank or checkpoint or j3v"`
Expected: collection error (no `evaluate.py`).

- [ ] **Step 3: Write `experiments/075_wide_region/evaluate.py`**

```python
"""EXP-075 evaluation: trained judges drive EXP-074's J3V search through its state-reading critic.

J3V-X and J3V-Y load this experiment's judges; J3V-W loads EXP-074's COMMITTED judges, re-run
only for Gate 0(b). The search, held-out sets, budgets, stream discipline and no-revisit rule are
EXP-074's (its `load_cell` and `JudgeCritic` are used directly). `run_cell` and `rank_cell` follow
EXP-074's, with this experiment's arms and record names.

Spec: docs/superpowers/specs/2026-10-09-exp075-wide-sensory-region-design.md, section 8.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/075_wide_region/evaluate.py --arm J3V-X --depth 9 --seed 0
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


cells = _load("exp075_cells", HERE / "cells.py")
e74 = _load("exp074_evaluate", REPO / "experiments" / "074_wide_judge" / "evaluate.py")

from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training.lookahead import (  # noqa: E402
    evaluate_lookahead, imag_seed_for, imagined_values, tree_levels,
)

torch.set_num_threads(1)

# arm -> trained-judge arm
_ARMS = {"J3V-X": "X", "J3V-Y": "Y", "J3V-W": "W"}
ARMS_EVAL = tuple(_ARMS)
RANK_KINDS = ("J-X", "J-Y")
_RANK_ARM = {"J-X": "J3V-X", "J-Y": "J3V-Y"}
JudgeCritic = e74.JudgeCritic
load_cell = e74.load_cell


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_judge(train_arm: str, seed: int, judge_dir=None):
    """X and Y: this experiment's checkpoint, loaded into an untrained region of the arm's width
    (the checkpoint holds the region's trained weights, so the pretrained file is not needed).
    W: EXP-074's committed judge, always from EXP-074's outputs."""
    if train_arm == "W":
        return e74.load_judge("W", seed, cells.E74_OUT)
    path = cells.ckpt_dir(train_arm, seed, judge_dir) / "judge.pt"
    if not path.exists():
        raise SystemExit(f"missing judge checkpoint {path} for arm {train_arm} seed {seed}; "
                         f"train it with train.py first")
    judge = cells.build_judge(seed, train_arm, cells.fresh_sensory(train_arm, seed))
    judge.load_state_dict(torch.load(path, map_location="cpu"))
    judge.eval()
    return judge


def _critic(arm, seed, judge_dir):
    return JudgeCritic(load_judge(_ARMS[arm], seed, judge_dir))


def run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict:
    torch.set_num_threads(1)
    if arm not in _ARMS:
        raise SystemExit(f"unknown arm {arm!r}; valid: {list(ARMS_EVAL)}")
    t0 = time.time()
    critic = _critic(arm, seed, judge_dir)
    agent, head, states, train_seed = load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=3,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    rec = {**res, "depth": depth, "seed": seed, "arm": arm,
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "limit_states": limit_states}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp075_{arm}_d{depth}_s{seed}.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def rank_cell(kind, depth, seed, out_dir, judge_dir=None, limit_states=None) -> dict:
    """Gate R: per-seed mean hit of 'the lowest-J leaf at k=3 is closer to solved than the root',
    on evaluation held-out states. BFS (max_depth = depth + 3) is the yardstick only."""
    torch.set_num_threads(1)
    if kind not in RANK_KINDS:
        raise SystemExit(f"unknown kind {kind!r}; valid: {list(RANK_KINDS)}")
    k = 3
    critic = _critic(_RANK_ARM[kind], seed, judge_dir)
    agent, head, states, train_seed = load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    provider = ExactBFSDistance(max_depth=depth + 3)
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
    rec = {"kind": kind, "depth": depth, "seed": seed, "n": len(states),
           "hit": st.mean(hits), "chance": st.mean(chances), "limit_states": limit_states}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp075_rank_{kind}_d{depth}_s{seed}.json").write_text(
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
        r = rank_cell(args.rank, args.depth, args.seed, args.out_dir, args.judge_dir,
                      args.limit_states)
        print(f"{r['kind']} d{r['depth']} s{r['seed']} hit {r['hit']:.4f} chance {r['chance']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp075.py -q -m "not slow"`
Expected: all PASS. If the run approaches the ceiling, split it with `-k` and report the per-part times.

- [ ] **Step 5: Mutation checks.** One at a time, apply, re-run, confirm the named test FAILS, restore:
  - In `load_judge`, replace the W branch with the generic checkpoint path (so W reads `judge_dir`) -> `test_j3v_w_is_exp074s_committed_judge`.
  - Remove `"limit_states": limit_states` from `rank_cell`'s record -> `test_rank_cell_records_hit_chance_and_its_limit`.
  Confirm `git diff` is clean of mutations afterwards.

- [ ] **Step 6: Commit**

```bash
git add experiments/075_wide_region/evaluate.py tests/experiments/test_exp075.py
git commit -m "EXP-075: evaluation over X, Y and EXP-074's committed W judges"
```

---

### Task 5: Aggregator and laptop launcher

**Files:**
- Create: `experiments/075_wide_region/aggregate.py`
- Create: `experiments/075_wide_region/launch075.ps1`
- Test: `tests/experiments/test_exp075.py` (append)

**Interfaces:**
- Consumes: Tasks 2 to 4 record names and keys. EXP-070 aggregate: `one_sided_p(diffs)`, `gate1_verdict(a_mean, b_mean)`. EXP-071 aggregate: `gate0b_verdict(mine: dict, ref: dict) -> "PASS"|"FAIL"` (compares `OUTCOME_FIELDS`), `gate_r_verdict(rows, "hit", "chance") -> (bool, p)`. EXP-072 aggregate: `pair_diffs(a: dict, b: dict) -> list`. EXP-074 aggregate: `load_rates(out_dir) -> {(arm, depth): {seed: rate}}`, `gate_l_verdict(margins, threshold) -> bool`, `GATE_L_THRESHOLD["W"]`.
- Produces: `ALPHA = 0.05 / 3`, `GATE_P_THRESHOLD: dict | None = None`, `GATE_L_THRESHOLD: dict | None = None`, `PILOT_FAIL_ACCURACY = 0.30`, `W_GATE_L_THRESHOLD`, `claim_verdict`, `contrast`, `check_train_record`, `check_pretrain_record`, `gate_p_verdict`, `gate_e_verdict`, `require_thresholds`, `pilot_amendment(pilot_dir) -> dict`, `primary_verdicts(rates, gates, seeds) -> dict` with keys `claim1`, `claim2`, `claim3`, `sensitivity_rates`, `load_rates(out_dir)`, `main()`.

- [ ] **Step 1: Write the failing tests** (append)

```python
agg = _load("exp075_aggregate", "aggregate.py")

_SETTINGS = {"n_updates": 4000, "batch": 1000, "sync_every": 100, "probe_every": 250, "draws": 1}


def _rates(x9, w9, y9, x11, w11, seeds=range(12)):
    return {("J3V-X", 9): {s: x9 for s in seeds}, ("J3V-W", 9): {s: w9 for s in seeds},
            ("J3V-Y", 9): {s: y9 for s in seeds}, ("J3V-X", 11): {s: x11 for s in seeds},
            ("J3V-W", 11): {s: w11 for s in seeds}}


def _gates(ok=True, **over):
    g = {"gate0": ok, "p": {"X": True, "Y": True}, "l": {"X": True, "Y": True},
         "e": {"X": True, "Y": True},
         "r": {(k, d): True for k in ("J-X", "J-Y") for d in (7, 8, 9, 11)}}
    g.update(over)
    return g


def _noisy(base, seeds=range(12)):
    return {s: base + 0.01 * ((s % 3) - 1) for s in seeds}


def test_alpha_is_a_third_of_005_and_the_verdict_uses_it(monkeypatch):
    """Catches the claims read at EXP-071's 0.025 instead of this spec's 0.05 / 3."""
    assert agg.ALPHA == pytest.approx(0.05 / 3)
    diffs = [0.05] * 12
    assert agg.claim_verdict(diffs, 0.3, 0.25, True)[0] == "CONFIRMED"
    monkeypatch.setattr(agg, "ALPHA", 0.0)
    assert agg.claim_verdict(diffs, 0.3, 0.25, True)[0] == "NOT SIGNIFICANT"


def test_claims_compare_the_registered_arms_and_depths():
    """Catches a claim wired to the wrong arm or depth: each claim's two means are distinct."""
    r = _rates(0.30, 0.20, 0.25, 0.15, 0.10)
    r[("J3V-X", 9)] = _noisy(0.30)
    r[("J3V-X", 11)] = _noisy(0.15)
    v = agg.primary_verdicts(r, _gates(), range(12))
    assert v["claim1"][3:] == pytest.approx((0.30, 0.20))
    assert v["claim2"][3:] == pytest.approx((0.30, 0.25))
    assert v["claim3"][3:] == pytest.approx((0.15, 0.10))
    assert all(v[c][0] == "CONFIRMED" for c in ("claim1", "claim2", "claim3"))


@pytest.mark.parametrize("over,void", [
    ({"gate0": False}, {"claim1", "claim2", "claim3"}),
    ({"p": {"X": False, "Y": True}}, {"claim1", "claim2", "claim3"}),
    ({"l": {"X": True, "Y": False}}, {"claim2"}),
    ({"e": {"X": True, "Y": False}}, {"claim2"}),
    ({"r": {**{(k, d): True for k in ("J-X", "J-Y") for d in (7, 8, 9, 11)},
            ("J-X", 11): False}}, {"claim3"}),
])
def test_a_failed_gate_voids_exactly_the_claims_it_guards(over, void):
    """Catches a gate that voids too little (a claim read on a broken arm) or too much."""
    r = _rates(0.30, 0.20, 0.25, 0.15, 0.10)
    v = agg.primary_verdicts(r, _gates(**over), range(12))
    assert {c for c in ("claim1", "claim2", "claim3") if v[c][0] == "VOID"} == void


def test_a_contrast_on_the_floor_is_unresolved():
    """Catches the Gate 1 band living only in prose (CLAUDE.md, EXP-068)."""
    r = _rates(0.012, 0.010, 0.011, 0.012, 0.010)
    assert agg.primary_verdicts(r, _gates(), range(12))["claim1"][0] == "UNRESOLVED"


def test_sensitivity_drops_exactly_seeds_0_and_3():
    r = agg.sensitivity_rates(_rates(0.3, 0.2, 0.25, 0.15, 0.1))
    assert sorted(r[("J3V-X", 9)]) == [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]


def test_gate_p_requires_every_seed():
    """Catches Gate P read as a mean (one failed pretraining hidden by eleven good ones)."""
    assert agg.gate_p_verdict([0.45] * 12, 0.40) is True
    assert agg.gate_p_verdict([0.45] * 11 + [0.39], 0.40) is False


def test_gate_e_requires_every_drift_positive():
    assert agg.gate_e_verdict([0.1] * 12) is True
    assert agg.gate_e_verdict([0.1] * 11 + [0.0]) is False


def test_thresholds_unset_block_every_verdict(monkeypatch):
    """REVIEW FOCUS 5. Catches a verdict read before the dated amendment."""
    monkeypatch.setattr(agg, "GATE_P_THRESHOLD", None)
    monkeypatch.setattr(agg, "GATE_L_THRESHOLD", {"X": 0.1, "Y": 0.1})
    with pytest.raises(SystemExit, match="GATE_P_THRESHOLD"):
        agg.require_thresholds()
    monkeypatch.setattr(agg, "GATE_P_THRESHOLD", {"X": 0.4, "Y": 0.4})
    monkeypatch.setattr(agg, "GATE_L_THRESHOLD", None)
    with pytest.raises(SystemExit, match="GATE_L_THRESHOLD"):
        agg.require_thresholds()


def _write_pilot(d, acc, margin):
    for arm in ("X", "Y"):
        for i, s in enumerate((12, 13)):
            (d / cells.pretrain_record_name(arm, s)).write_text(json.dumps(
                {"arm": arm, "seed": s, "final_move_accuracy": acc[arm][i]}))
            (d / cells.record_name(arm, s)).write_text(json.dumps(
                {"arm": arm, "seed": s, "gate_l": {"margin": margin[arm][i]}}))


def test_pilot_amendment_numbers_follow_the_spec_forms(tmp_path):
    """Catches a threshold form other than spec section 7's (0.9 x mean pilot accuracy, half
    the mean pilot margin), and a missing Y working-control check (spec section 6)."""
    _write_pilot(tmp_path, {"X": [0.50, 0.52], "Y": [0.45, 0.46]},
                 {"X": [0.30, 0.26], "Y": [0.20, 0.18]})
    a = agg.pilot_amendment(tmp_path)
    assert a["gate_p"]["X"] == pytest.approx(0.9 * 0.51)
    assert a["gate_l"]["Y"] == pytest.approx(0.19 / 2)
    assert a["pilot_failed"] is False and a["y_working_control"] is True
    _write_pilot(tmp_path, {"X": [0.25, 0.28], "Y": [0.45, 0.46]},
                 {"X": [0.30, 0.26], "Y": [0.05, 0.06]})
    a = agg.pilot_amendment(tmp_path)
    assert a["pilot_failed"] is True and a["y_working_control"] is False


def test_records_at_other_settings_or_smoke_records_are_refused(tmp_path):
    """REVIEW FOCUS 4. Catches a smoke cell, a record at other settings, or the wrong width
    entering a verdict."""
    good = {**_SETTINGS, "arm": "X", "seed": 0, "readout": "wide", "hidden": 512}
    agg.check_train_record(good)
    for bad in ({"hidden": 128}, {"readout": "concept"}, {"n_updates": 2}):
        with pytest.raises(SystemExit):
            agg.check_train_record({**good, **bad})
    pre = {"arm": "Y", "seed": 0, "epochs": 40, "batch_size": 256, "lr": 3e-3, "hidden": 128}
    agg.check_pretrain_record(pre)
    with pytest.raises(SystemExit):
        agg.check_pretrain_record({**pre, "epochs": 1})
    p = tmp_path / "exp075_J3V-X_d9_s0.json"
    p.write_text(json.dumps({"success_rate": 0.2, "limit_states": 1}))
    with pytest.raises(SystemExit, match="smoke"):
        agg._read(p)


def test_launcher_carries_the_exit_code_fix_and_the_registered_cells():
    """Catches a launcher copied without EXP-074's .Handle fix (every cell counted as failed)
    or with an eval list missing an arm or depth."""
    text = (EXP / "launch075.ps1").read_text()
    assert "$null = $proc.Handle" in text
    assert '"J3V-X", "J3V-Y"' in text and "7, 8, 9, 11" in text
    assert "exp075-det" in text and "GATE_P_THRESHOLD" in text
```

- [ ] **Step 2: Run them to verify they fail**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp075.py -q -m "not slow" -k "alpha or claim or gate or unresolved or sensitivity or threshold or pilot or refused or launcher"`
Expected: collection error (no `aggregate.py`).

- [ ] **Step 3: Write `experiments/075_wide_region/aggregate.py`**

```python
"""EXP-075 aggregator. Gates 0, P, L, E, R and 1 are VERDICTS: a claim whose gate failed prints
VOID, and a contrast on the floor or ceiling prints UNRESOLVED.

Spec: docs/superpowers/specs/2026-10-09-exp075-wide-sensory-region-design.md, sections 6 to 9.

The sign-flip test and Gate 1 are EXP-070's, Gate 0(b)'s field comparison and Gate R are EXP-071's,
seed pairing is EXP-072's, and Gate L and EXP-074's rates are EXP-074's, all reused through
importlib so the experiments cannot drift apart. Alpha is this spec's own: 0.05 / 3.

Usage:
    .venv/bin/python experiments/075_wide_region/aggregate.py --pilot-report
    .venv/bin/python experiments/075_wide_region/aggregate.py --determinism-ok
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
a74 = _load("exp074_aggregate", REPO / "experiments" / "074_wide_judge" / "aggregate.py")
cells = _load("exp075_cells", HERE / "cells.py")
one_sided_p = a70.one_sided_p
SEEDS = range(12)

ALPHA = 0.05 / 3

# Set ONLY by the controller's dated amendment, after the pilot and before any seed 0-11
# pretrains: {"X": float, "Y": float}. Gate P: 0.9 x that arm's mean pilot move accuracy.
# Gate L: half that arm's mean pilot margin (spec section 7). None blocks every verdict.
GATE_P_THRESHOLD: dict | None = None
GATE_L_THRESHOLD: dict | None = None
PILOT_FAIL_ACCURACY = 0.30
# Spec section 6: Y must clear W's own Gate L threshold to count as a working control.
W_GATE_L_THRESHOLD = a74.GATE_L_THRESHOLD["W"]

J_ARMS = ("J3V-X", "J3V-Y")
DEPTHS = (7, 8, 9, 11)
_KIND = {"J3V-X": "J-X", "J3V-Y": "J-Y"}
_TRAIN_ARM = {"J3V-X": "X", "J3V-Y": "Y"}
TRAIN_SETTINGS = {"n_updates": 4000, "batch": 1000, "sync_every": 100, "probe_every": 250,
                  "draws": 1}
PRETRAIN_SETTINGS = {"epochs": 40, "batch_size": 256, "lr": 3e-3}


def claim_verdict(diffs, a_mean: float, b_mean: float, gate_ok: bool) -> tuple[str, float]:
    p = one_sided_p(diffs)
    if not gate_ok:
        return "VOID", p
    if a70.gate1_verdict(a_mean, b_mean) == "UNRESOLVED":
        return "UNRESOLVED", p
    if st.mean(diffs) <= 0:
        return "REFUTED", p
    if p < ALPHA:
        return "CONFIRMED", p
    return "NOT SIGNIFICANT", p


def contrast(rates, a, b, gate_ok):
    diffs = a72.pair_diffs(rates[a], rates[b])
    a_mean, b_mean = st.mean(rates[a].values()), st.mean(rates[b].values())
    verdict, p = claim_verdict(diffs, a_mean, b_mean, gate_ok)
    return verdict, p, st.mean(diffs), a_mean, b_mean


def _check(rec, settings, kind) -> None:
    who = f"{kind} record for arm {rec.get('arm')} seed {rec.get('seed')}"
    for k, v in settings.items():
        if rec.get(k) != v:
            raise SystemExit(f"{who}: {k} is {rec.get(k)!r}, spec says {v!r}")
    if rec.get("hidden") != cells.HIDDEN[rec["arm"]]:
        raise SystemExit(f"{who}: hidden is {rec.get('hidden')!r}, "
                         f"spec says {cells.HIDDEN[rec['arm']]!r}")


def check_train_record(rec) -> None:
    _check(rec, TRAIN_SETTINGS, "training")
    if rec.get("readout") != "wide":
        raise SystemExit(f"training record for arm {rec.get('arm')} seed {rec.get('seed')}: "
                         f"readout is {rec.get('readout')!r}, spec says 'wide'")


def check_pretrain_record(rec) -> None:
    _check(rec, PRETRAIN_SETTINGS, "pretraining")


def gate_p_verdict(accuracies, threshold: float) -> bool:
    return all(a >= threshold for a in accuracies)


def gate_e_verdict(drifts) -> bool:
    return all(d > 0 for d in drifts)


def require_thresholds() -> None:
    for name in ("GATE_P_THRESHOLD", "GATE_L_THRESHOLD"):
        if globals()[name] is None:
            raise SystemExit(f"{name} is unset. The controller sets it in the dated post-pilot "
                             f"amendment; no verdict may be read before then.")


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path.name}: the run is incomplete")
    r = json.loads(path.read_text())
    if r.get("limit_states") is not None:
        raise SystemExit(f"{path.name} is a smoke record and may not enter a verdict")
    return r


def pilot_amendment(pilot_dir) -> dict:
    """The numbers the dated amendment states (spec sections 6 and 7), from the pilot records."""
    pilot_dir = Path(pilot_dir)
    acc = {a: [_read(pilot_dir / cells.pretrain_record_name(a, s))["final_move_accuracy"]
               for s in cells.PILOT_SEEDS] for a in cells.ARMS_TRAIN}
    mar = {a: [_read(pilot_dir / cells.record_name(a, s))["gate_l"]["margin"]
               for s in cells.PILOT_SEEDS] for a in cells.ARMS_TRAIN}
    return {
        "mean_accuracy": {a: st.mean(v) for a, v in acc.items()},
        "mean_margin": {a: st.mean(v) for a, v in mar.items()},
        "gate_p": {a: 0.9 * st.mean(v) for a, v in acc.items()},
        "gate_l": {a: st.mean(v) / 2 for a, v in mar.items()},
        "pilot_failed": any(st.mean(v) < PILOT_FAIL_ACCURACY for v in acc.values()),
        "y_working_control": st.mean(mar["Y"]) >= W_GATE_L_THRESHOLD,
    }


def _arm_ok(gates, j_arm, depth) -> bool:
    a = _TRAIN_ARM[j_arm]
    return gates["p"][a] and gates["l"][a] and gates["e"][a] and gates["r"][(_KIND[j_arm], depth)]


def primary_verdicts(rates: dict, gates: dict, seeds) -> dict:
    """Claim 1: d9 J3V-X vs J3V-W. Claim 2: d9 J3V-X vs J3V-Y. Claim 3: d11 J3V-X vs J3V-W.
    W is EXP-074's reused reference, whose gates passed there; X and Y are gated here."""
    r = {k: {s: v[s] for s in seeds} for k, v in rates.items()}
    c1_ok = gates["gate0"] and _arm_ok(gates, "J3V-X", 9)
    c2_ok = c1_ok and _arm_ok(gates, "J3V-Y", 9)
    c3_ok = gates["gate0"] and _arm_ok(gates, "J3V-X", 11)
    return {"claim1": contrast(r, ("J3V-X", 9), ("J3V-W", 9), c1_ok),
            "claim2": contrast(r, ("J3V-X", 9), ("J3V-Y", 9), c2_ok),
            "claim3": contrast(r, ("J3V-X", 11), ("J3V-W", 11), c3_ok)}


def sensitivity_rates(rates: dict) -> dict:
    return {k: {s: v for s, v in d.items() if s not in cells.SENSITIVITY_DROP}
            for k, d in rates.items()}


def load_rates(out_dir: Path) -> dict:
    """(arm, depth) -> {seed: success_rate}. X and Y from this experiment; W, P3V and R3V from
    EXP-074's aggregator (which reads EXP-071 and EXP-072 for P3V and R3V at depths 7 to 9)."""
    rates = {k: v for k, v in a74.load_rates(cells.E74_OUT).items()
             if k[0] in ("J3V-W", "P3V", "R3V")}
    for d in DEPTHS:
        for arm in J_ARMS:
            rates[(arm, d)] = {s: _read(out_dir / f"exp075_{arm}_d{d}_s{s}.json")["success_rate"]
                               for s in SEEDS}
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--pilot-dir", type=Path, default=HERE / "outputs_pilot")
    ap.add_argument("--pilot-report", action="store_true",
                    help="print the amendment's numbers from the pilot records and stop")
    ap.add_argument("--determinism-ok", action="store_true",
                    help="set only after the det re-run matches its full-run copy")
    args = ap.parse_args()
    if args.pilot_report:
        a = pilot_amendment(args.pilot_dir)
        for key, val in a.items():
            print(f"{key}: {val}")
        if a["pilot_failed"]:
            print(f"PILOT FAILED: an arm's mean pilot move accuracy is below "
                  f"{PILOT_FAIL_ACCURACY}. Stop and report; no amendment (spec section 7).")
        if not a["y_working_control"]:
            print(f"Y IS NOT A WORKING CONTROL: mean pilot margin below W's Gate L threshold "
                  f"{W_GATE_L_THRESHOLD:.4f}. Stop and report; no amendment (spec section 6).")
        return
    require_thresholds()
    out = args.out_dir
    pre = {(a, s): _read(out / cells.pretrain_record_name(a, s))
           for a in cells.ARMS_TRAIN for s in SEEDS}
    for r in pre.values():
        check_pretrain_record(r)
    train = {(a, s): _read(out / cells.record_name(a, s)) for a in cells.ARMS_TRAIN for s in SEEDS}
    for r in train.values():
        check_train_record(r)
    rank = {(k, d): [_read(out / f"exp075_rank_{k}_d{d}_s{s}.json") for s in SEEDS]
            for k in ("J-X", "J-Y") for d in DEPTHS}
    rates = load_rates(out)

    gate_p = {a: gate_p_verdict([pre[(a, s)]["final_move_accuracy"] for s in SEEDS],
                                GATE_P_THRESHOLD[a]) for a in cells.ARMS_TRAIN}
    gate_l = {a: a74.gate_l_verdict([train[(a, s)]["gate_l"]["margin"] for s in SEEDS],
                                    GATE_L_THRESHOLD[a]) for a in cells.ARMS_TRAIN}
    gate_e = {a: gate_e_verdict([train[(a, s)]["encoder_drift"] for s in SEEDS])
              for a in cells.ARMS_TRAIN}
    gate_r = {(k, d): bool(a71.gate_r_verdict(rank[(k, d)], "hit", "chance")[0])
              for k in ("J-X", "J-Y") for d in DEPTHS}
    g0b = a71.gate0b_verdict(
        {("J3V-W", d): _read(out / f"exp075_J3V-W_d{d}_s0.json") for d in (9, 11)},
        {("J3V-W", d): _read(cells.E74_OUT / f"exp074_J3V-W_d{d}_s0.json") for d in (9, 11)})

    print("GATE P (pretraining move accuracy, every seed)")
    for a in cells.ARMS_TRAIN:
        v = [pre[(a, s)]["final_move_accuracy"] for s in SEEDS]
        print(f"  arm {a}: min {min(v):.4f} mean {st.mean(v):.4f} vs threshold "
              f"{GATE_P_THRESHOLD[a]:.4f} -> {gate_p[a]}")
    print("GATE L (3-move leaf ranking on probe distances 7-11, margin over chance)")
    for a in cells.ARMS_TRAIN:
        m = [train[(a, s)]["gate_l"]["margin"] for s in SEEDS]
        print(f"  arm {a}: mean margin {st.mean(m):.4f} vs threshold {GATE_L_THRESHOLD[a]:.4f}, "
              f"p {one_sided_p(m):.4f} -> {gate_l[a]}")
    print(f"GATE E: {gate_e}")
    print("GATE R (lowest-J leaf closer than root on held-out states)")
    for d in DEPTHS:
        print(f"  d{d}: " + "  |  ".join(
            f"{k} {st.mean(r['hit'] for r in rank[(k, d)]):.4f}/"
            f"{st.mean(r['chance'] for r in rank[(k, d)]):.4f} -> {gate_r[(k, d)]}"
            for k in ("J-X", "J-Y")))
    print(f"GATE 0(a) determinism: {'PASS' if args.determinism_ok else 'NOT CHECKED'}")
    print(f"GATE 0(b) J3V-W continuity with EXP-074 (d9, d11, seed 0): {g0b}")

    print("\nMean held-out success")
    for d in DEPTHS:
        print(f"  d{d}: " + "  ".join(f"{a} {st.mean(rates[(a, d)].values()):.4f}"
                                      for a in ("J3V-X", "J3V-Y", "J3V-W", "P3V", "R3V")))

    gate0_ok = g0b == "PASS" and args.determinism_ok
    if not gate0_ok:
        print("Gate 0 did not pass. No claim may be read.")
        return
    gates = {"gate0": gate0_ok, "p": gate_p, "l": gate_l, "e": gate_e, "r": gate_r}
    full = primary_verdicts(rates, gates, SEEDS)
    sens_rates = sensitivity_rates(rates)
    sens = primary_verdicts(sens_rates, gates, sorted(sens_rates[("J3V-X", 9)]))
    for name, label in (("claim1", "CLAIM 1 (primary) d9 J3V-X - J3V-W"),
                        ("claim2", "CLAIM 2 (primary) d9 J3V-X - J3V-Y"),
                        ("claim3", "CLAIM 3 (primary) d11 J3V-X - J3V-W")):
        v, p, diff, am, bm = full[name]
        sv, sp, sdiff, _, _ = sens[name]
        print(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} -> {v}")
        print(f"    sensitivity, seeds 0 and 3 dropped: diff {sdiff:+.4f}, p {sp:.4f} -> {sv}")
        if sv != v:
            print("    SENSITIVITY DISAGREES WITH THE VERDICT: RESULTS.md must lead with this.")

    print("\nSecondary (patterns, never confirmations):")
    sec = [(d, "J3V-X", "J3V-W") for d in (7, 8)] + [(d, "J3V-X", "J3V-Y") for d in (7, 8, 11)] \
        + [(9, "J3V-Y", "J3V-W")]
    for d, a, b in sec:
        ok = gate0_ok and _arm_ok(gates, a, d) and (b not in _TRAIN_ARM or _arm_ok(gates, b, d))
        v, p, diff, am, bm = contrast(rates, (a, d), (b, d), ok)
        tag = "VOID (gate)" if v == "VOID" else (
            "positive pattern" if diff > 0 else "non-positive pattern")
        print(f"  d{d} {a} - {b}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, p {p:.4f} ({tag})")
    print("  mechanism line, leaf hit X - Y: " + "  ".join(
        f"d{d} {st.mean(r['hit'] for r in rank[('J-X', d)]) - st.mean(r['hit'] for r in rank[('J-Y', d)]):+.4f}"
        for d in DEPTHS))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `experiments/075_wide_region/launch075.ps1`**

Model it on `experiments/074_wide_judge/launch074.ps1` (read it first). Exact content:

```powershell
# EXP-075 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase pilot-pretrain -Workers 4
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase pilot-train    -Workers 4
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase pretrain -Workers 12
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase train    -Workers N   (N from the amendment)
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase cont     -Workers 2
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase rank     -Workers 12
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase eval     -Workers 12 -SkipExisting
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase det      -Workers 1
#
# Order (spec section 10): pilot-pretrain -> pilot-train -> dated amendment (Gate P and L
# thresholds, worker count) -> pretrain (Gate P committed) -> train (Gates L and E committed) ->
# cont (Gate 0(b): J3V-W at seed 0, depths 9 and 11, must equal EXP-074's records) -> rank ->
# eval -> det (J3V-X d9 s0 into a SEPARATE directory, Gate 0(a)).
#
# Pretraining and training both resume by themselves: their drivers skip cells whose record
# exists, and train_judge resumes from its banked checkpoint. Re-run the same phase after a
# Windows Update restart. Evaluation phases take -SkipExisting.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Arm and seed lists therefore live in this file.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","pilot-pretrain","pilot-train","pretrain","train","cont","rank","eval","det")][string]$Phase,
    [int]$Workers = 0,
    [switch]$SkipExisting
)

# Spec section 5, fixed.
$Updates = 4000
$SyncEvery = 100
$ProbeEvery = 250
$PilotOutDir = "experiments\075_wide_region\outputs_pilot"

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\075_wide_region"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# The hidden-width seam must be present: a stale checkout fails here first.
$hasHidden = & $py -c "import sys, inspect; from neuromorphic.training import encoder_pretrain as e; sys.stdout.write(str('hidden' in inspect.signature(e.make_sensory).parameters))"
if ($LASTEXITCODE -ne 0 -or $hasHidden -ne "True") { Write-Error "hidden seam missing (got '$hasHidden'). Sync the repo."; exit 1 }

# 36 policy heads and 12 E1 encoders (EXP-070's path functions; W's judges rebuild through E1),
# and EXP-074's 12 committed W judges.
$present = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('c','experiments/070_lookahead_existing/cells.py'); c=importlib.util.module_from_spec(s); s.loader.exec_module(c); from pathlib import Path; h=sum(c.head_path(d,x).exists() for d in (7,8,9) for x in c.SEEDS); e=sum(Path(c.published_config(7,x).encoder_state_path).exists() for x in c.SEEDS); w=sum(Path('experiments/074_wide_judge/outputs/judge_W_s'+str(x)+'/judge.pt').exists() for x in c.SEEDS); sys.stdout.write(str(h)+' '+str(e)+' '+str(w))"
if ($LASTEXITCODE -ne 0 -or $present -ne "36 12 12") { Write-Error "expected 36 heads, 12 E1 encoders and 12 W judges, found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                 = $where"
"checkout                = $head"
"heads, E1, W judges     = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs                 = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

# Every phase except the pilot and cont reads or produces numbers that decide a claim, and none
# may run before the dated amendment sets BOTH thresholds. Check the exit code too: an import
# failure prints nothing, and "" would not match "None".
if ($Phase -ne "pilot-pretrain" -and $Phase -ne "pilot-train" -and $Phase -ne "cont") {
    $gate = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','$($exp.Replace('\','/'))/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.GATE_P_THRESHOLD)+' | '+str(a.GATE_L_THRESHOLD))"
    if ($LASTEXITCODE -ne 0 -or $gate -eq "") { Write-Error "could not read GATE_P_THRESHOLD and GATE_L_THRESHOLD (got '$gate'); refusing to run."; exit 1 }
    if ($gate -match "None") { Write-Error "a threshold is unset ($gate): commit the dated post-pilot amendment before this phase."; exit 1 }
    "thresholds P | L        = $gate"
}

if ($Workers -eq 0) { $Workers = 4 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$evalSeeds = @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")

# Pretraining and training phases: one driver call, it manages its own worker pool.
$driver = $null
switch ($Phase) {
    "pilot-pretrain" { $driver = "pretrain.py"; $cliArgs = @("--pilot", "--arms", "X", "Y", "--seeds", "12", "13", "--out-dir", (Join-Path $repo $PilotOutDir), "--workers", $Workers) }
    "pilot-train"    { $driver = "train.py";    $cliArgs = @("--pilot", "--arms", "X", "Y", "--seeds", "12", "13", "--n-updates", $Updates, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--out-dir", (Join-Path $repo $PilotOutDir), "--workers", $Workers) }
    "pretrain"       { $driver = "pretrain.py"; $cliArgs = @("--arms", "X", "Y", "--seeds") + $evalSeeds + @("--workers", $Workers) }
    "train"          { $driver = "train.py";    $cliArgs = @("--arms", "X", "Y", "--seeds") + $evalSeeds + @("--n-updates", $Updates, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--workers", $Workers) }
}
if ($driver) {
    "script                  = $exp\$driver $cliArgs"
    "log                     = $log"
    ""
    & $py -u "$exp\$driver" @cliArgs 2>&1 | Tee-Object -FilePath $log
    exit $LASTEXITCODE
}

# Evaluation phases: evaluate.py runs one cell per call, so this file throttles a pool of
# processes itself. Each job is @(label, expected-output-file, args...).
$seeds = 0..11
$outDir = Join-Path $repo "$exp\outputs"
$detDir = "C:\Users\mlgbr\exp075-det"
$jobs = New-Object System.Collections.ArrayList
switch ($Phase) {
    "cont" {
        foreach ($d in 9, 11) {
            [void]$jobs.Add(@("J3V-W d$d s0", "exp075_J3V-W_d${d}_s0.json", "--arm", "J3V-W", "--depth", $d, "--seed", 0)) }
    }
    "rank" {
        foreach ($d in 7, 8, 9, 11) { foreach ($k in "J-X", "J-Y") { foreach ($s in $seeds) {
            [void]$jobs.Add(@("rank $k d$d s$s", "exp075_rank_${k}_d${d}_s${s}.json", "--rank", $k, "--depth", $d, "--seed", $s)) } } }
    }
    "eval" {
        foreach ($d in 7, 8, 9, 11) { foreach ($a in "J3V-X", "J3V-Y") { foreach ($s in $seeds) {
            [void]$jobs.Add(@("$a d$d s$s", "exp075_${a}_d${d}_s${s}.json", "--arm", $a, "--depth", $d, "--seed", $s)) } } }
    }
    "det" {
        [void]$jobs.Add(@("J3V-X d9 s0", "exp075_J3V-X_d9_s0.json", "--arm", "J3V-X", "--depth", 9, "--seed", 0, "--out-dir", $detDir))
    }
}

"phase                   = $Phase, $($jobs.Count) cells, $Workers workers"
"log                     = $log"
""
$running = @()
$failed = 0
$stamp = { param($m) Add-Content -Path $log -Value $m; $m }
foreach ($j in $jobs) {
    $target = if ($Phase -eq "det") { Join-Path $detDir $j[1] } else { Join-Path $outDir $j[1] }
    if ($SkipExisting -and (Test-Path $target)) { continue }
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Workers) { Start-Sleep -Seconds 5 }
    foreach ($p in @($running | Where-Object { $_.HasExited })) { if ($p.ExitCode -ne 0) { $failed++ } }
    $running = @($running | Where-Object { -not $_.HasExited })
    & $stamp "start $($j[0])" | Out-Null
    $argList = @("-u", "$exp\evaluate.py") + @($j[2..($j.Count - 1)] | ForEach-Object { "$_" })
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

- [ ] **Step 5: Run the tests to verify they pass**

Run (timeout 600000): `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp075.py -q -m "not slow"`
Expected: all PASS.

- [ ] **Step 6: Mutation checks.** One at a time, apply, re-run Step 5's command, confirm the named test FAILS, restore:
  - `if p < a71.ALPHA:` in `claim_verdict` -> `test_alpha_is_a_third_of_005_and_the_verdict_uses_it`.
  - Drop `and _arm_ok(gates, "J3V-Y", 9)` from `c2_ok` -> `test_a_failed_gate_voids_exactly_the_claims_it_guards`.
  - `c3_ok = gates["gate0"] and _arm_ok(gates, "J3V-X", 9)` -> same test.
  - `"gate_l": {a: st.mean(v) for ...}` (no `/ 2`) -> `test_pilot_amendment_numbers_follow_the_spec_forms`.
  - Delete the line `$null = $proc.Handle` -> `test_launcher_carries_the_exit_code_fix_and_the_registered_cells`.
  Confirm `git diff` is clean of mutations afterwards.

- [ ] **Step 7: Commit**

```bash
git add experiments/075_wide_region/aggregate.py experiments/075_wide_region/launch075.ps1 tests/experiments/test_exp075.py
git commit -m "EXP-075: aggregator with Gates P, L, E, R, 0 and 1 as verdicts, and the laptop launcher"
```

---

### Task 6 (controller): merge, pilot, amendment, then the run

Not dispatched to an implementer. The controller runs these in order.

- [ ] **Step 1: Whole-branch checks on the VPS.** Run `tests/experiments/test_exp075.py` and `tests/training/test_encoder_pretrain.py tests/training/test_pretrain.py` (not slow) in the foreground, then their slow tests in a separate call. Run `tests/experiments/test_exp074.py -m "not slow"` to confirm EXP-074 is untouched. Update the test count in CLAUDE.md from `--collect-only`.
- [ ] **Step 2: Merge** `exp075` into `main` with `--no-ff`, subject `Merge exp075: EXP-075 wide sensory region (code)`, push, delete the branch.
- [ ] **Step 3: Sync the laptop** (`git pull` in its main checkout), copy `launch075.ps1` to `C:\Users\mlgbr\`, run `-Phase check`.
- [ ] **Step 4: Pilot.** `-Phase pilot-pretrain -Workers 4`, then `-Phase pilot-train -Workers 4`, each under a guard script like EXP-074's (100-minute windows, re-arm on exit 3). Measure s per update and per-worker working set (`WorkingSet64` on `^python` processes) for X and Y.
- [ ] **Step 5: Amendment.** Copy the pilot records to the VPS, run `aggregate.py --pilot-report`. If it prints PILOT FAILED or Y IS NOT A WORKING CONTROL, stop and report to Michael. Otherwise write spec section 12, "Amendment <date>: pilot results, Gate P and L thresholds, worker count" (thresholds, worker count with total peak working set under 24 GB, schedule). Set `GATE_P_THRESHOLD` and `GATE_L_THRESHOLD` in `aggregate.py` with a comment citing it, plus a test that reads `outputs_pilot` and checks both against the spec forms (as EXP-074's `test_gate_l_threshold_is_half_the_mean_pilot_margin`). Commit the pilot records and amendment, **show Michael**, then push.
- [ ] **Step 6: Pretrain** seeds 0 to 11 (`-Phase pretrain`). Record Gate P in spec section 13 and commit it before any VI run.
- [ ] **Step 7: Train** (`-Phase train -Workers N`). Record Gates L and E in the spec and commit before any evaluation cell.
- [ ] **Step 8: Evaluate.** `-Phase cont` first; if Gate 0(b) fails, stop. Then `rank`, `eval -SkipExisting`, `det`. Confirm the det record equals its full-run copy except `wall_s` and `git_commit`.
- [ ] **Step 9: Aggregate** with `--determinism-ok`. Force-add the small JSON records, the 24 encoders and the 24 `judge.pt` files (not `jt.pt` or `opt.pt`). Write `experiments/075_wide_region/RESULTS.md` with the disclosure, the verdicts with sensitivity lines, gates, means by depth, per-seed depth 9 and 11, secondaries, caveats and provenance.
- [ ] **Step 10: Close out.** Roadmap entry, handoff and next-session prompt, a message to the docs session, and `secretary log`.
