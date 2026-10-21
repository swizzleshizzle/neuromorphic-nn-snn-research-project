# EXP-080 Oracle Ordering Ceiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build EXP-080's experiment folder: mint true-distance labels once per seed, train three oracle judges on them (O-raw, O-Xf, O-Xt), measure each one's far-band ordering margin against EXP-075's X, run the J4V-O-Xt depth-11 search arm when Claim 1 confirms, and aggregate the pre-registered gates, claims and the one-line verdict.

**Architecture:** One folder, `experiments/080_oracle_ordering/`, reusing earlier experiments through `importlib` exactly as EXP-075 to EXP-079 do. `cells.py` (constants, paths, the three judges, the probe-leaf exclusion), `mint.py` (the only file that builds the unbounded BFS table: walk states, their labels, the probe set and every probe leaf's distance, written to one sha-checked file per seed), `train.py` (MSE on minted labels, resumable, never touches BFS; the Gate L margin is measured at the end from the minted leaf distances), `evaluate.py` (Gate 0's X re-measurement through the mint, the J4V-O-Xt search cells, the pilot timing cell), `aggregate.py` (Gates 0, V, T, 1, Claims 1 and 2, the search reading, the verdict, the pilot report), `launch080.ps1` and `chain080.ps1` for the laptop. No library change.

**Tech Stack:** Python 3 via `.venv/bin/python`, PyTorch (CPU), NumPy, pytest, Windows PowerShell 5.1 on the laptop.

**Spec:** `docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md` (read it before starting; it carries a clarification dated 2026-10-21, commit `6241536`, made before any code: one shared draw per seed, minted probe-leaf distances, the pilot timing cell, and verdict rows for Gate T and Gate 1).

## Global Constraints

- Training walks: `random_walk_states(1000, 40, rng, N_ACTIONS, exclude)` called once per block, **4000 blocks of 1000**, one `random.Random(seed)` per seed, in update order. O-Xf and O-Xt train on block `u` at update `u`; O-raw trains on the first **600,000** states.
- Exclusion: the seed's held-out states (`exclusion_set`), its 550 probe states (`probe_set`), and every 3-move leaf (`tree_levels(s, 3, N_ACTIONS)[3]`) of each probe state at true distance 7 to 11.
- O-raw: MLP 144-512-512-1 on one-hot facelets (24 facelets x 6 colours), no output nonlinearity, `torch.manual_seed(seed)` before construction, 12 epochs, batch 512, Adam lr 1e-3, epoch order from `torch.randperm(n, generator=torch.Generator().manual_seed(seed))`.
- O-Xf: EXP-075's committed X judge's region, frozen, under a fresh wide head built by `c75.build_judge(seed, "X", ...)`; `vi.make_optimizer(judge, "B")` (head lr 1e-3). 4000 updates, batch 1000.
- O-Xt: EXP-079's `make_judge(seed, enc_root)` (EXP-075's pretrained `enc_X_s<seed>.pt` under a fresh wide head); `vi.make_optimizer(judge, "W")` (head 1e-3, region 1e-4). 4000 updates, batch 1000.
- Loss: `F.mse_loss(judge(states, generator), labels.float())`. One `torch.Generator().manual_seed(seed)` drives the encoding across updates; every probe uses its own fresh `torch.Generator().manual_seed(seed)`.
- Probe and checkpoint every **250** updates and at the last update.
- Far band: `a79.margin_band(gate_l)` over probe distances 9 to 11. Overall: `gate_l["margin"]` over 7 to 11. Gate L instrument: `gate_l.leaf_rank_margin(judge, probe, distance, seed, N_ACTIONS)` with `k=3, lo=7, hi=11` defaults.
- Search arm: EXP-076's J4V (`mode="C"`, `k=4`, `k_goal=4`, `no_revisit=True`), depth 11, 200 held-out states, seeds 0 to 11, against EXP-076's committed `exp076_J4V-X_d11_s<seed>.json` (mean 0.1950).
- Gate 0: X seed 0's `gate_l`, re-measured through seed 0's mint, must equal EXP-075's committed `exp075_train_X_s0.json` `gate_l` exactly (dict equality).
- Gate V: O-raw far minus X far per seed, one-sided exact sign-flip `p < 0.05` AND mean `>= 0.05`.
- Gate T: `a74.gate_l_verdict(overall_margins, 0.1356358024691358)` per arm (half X's 12-seed mean overall margin 0.2712716049382716).
- Claims 1 (O-Xt) and 2 (O-Xf): far minus X far per seed, alpha **0.025**; CONFIRMED needs mean `>= 0.05`, CONFIRMED (SMALL) below it; REFUTED otherwise; VOID if Gate 0 or Gate V fails or that arm's Gate T fails.
- Search reading: alpha 0.05, PAYS needs mean `>= 0.03`; UNRESOLVED when EXP-070's `gate1_verdict` says so. Search cells run only when Claim 1 is CONFIRMED (not SMALL).
- Pilot stop bar: O-raw's mean pilot far-band margin below **0.14532407407407405** (X's mean pilot far-band margin on seeds 12 and 13, 0.09532407407407405, plus 0.05) stops the experiment.
- Pilot seeds 12 and 13 only. Worker cap 6 on the laptop.
- Run python through the venv, never bare `python`. In a git worktree prefix every command with `PYTHONPATH=src` (the editable install points at the main checkout), and call the main checkout's interpreter: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python`.
- **Always pass an explicit Bash tool timeout; the default is 120 s.** Every test command in this plan fits in one call with `timeout: 600000`. Call it directly and let it block; piped output buffers, so silence mid-run is normal and is not a hang. Never run the whole suite from a task.
- No test may build the unbounded BFS table (`ExactBFSDistance(max_depth=None)`, 67 s). Tests use `FakeDistance` or monkeypatched `probe_set` / `exclusion_set`.
- No em-dashes anywhere (code, docs, commit messages). Plain commit messages, no `Co-Authored-By`, no "Generated with". Stage explicit paths, never `git add -A`.
- Test-strength rule (CLAUDE.md): every assertion must be able to fail against the defect it names. Never weaken a threshold to make a test pass. Each test's docstring names the defect it catches.

## Review Focus

1. **A mint killed mid-write by the 03:31 Windows Update restart** leaves a partial `.npz`. Expected: the seed counts as unminted and is redrawn; training never reads a partial file. Pinned in Task 2 (`test_mint_is_pending_until_its_sidecar_exists` and `test_read_mint_refuses_a_tampered_file`).
2. **A training run resumed against a re-minted file** (a seed re-minted after a crash, with a different draw) would splice two training sets into one judge. Expected: resume refuses with a message naming both shas. Pinned in Task 3 (`test_resume_refuses_a_checkpoint_from_a_different_mint`).
3. **O-Xf's region silently trains** (a freeze that does not hold) or **O-Xt's region silently does not** (a gradient that never arrives, CLAUDE.md's `MemoryReadout` lesson). Expected: the training records show it, and the aggregator refuses either record. Pinned in Task 3 (`test_oxt_trains_its_region_and_oxf_does_not`) and Task 5 (`test_train_record_refuses_a_moved_oxf_or_a_still_oxt`).
4. **O-Xf built on X's trained head instead of a fresh one.** Expected: O-Xf's head before training is bit-identical to O-Xt's (same seed, same construction) and its region is X's trained region. Pinned in Task 1 (`test_oxf_is_xs_trained_region_under_oxts_fresh_head`).
5. **A smoke or timing record (`limit_states` set), or a record whose seed differs from its file name, entering a verdict.** Expected: refused by the aggregator. Pinned in Task 5 (`test_smoke_and_misnamed_records_are_refused`).

---

## File Structure

| file | responsibility |
|---|---|
| `experiments/080_oracle_ordering/cells.py` | constants, paths, record names, the probe-leaf exclusion, the three judges, optimizers, loading a trained judge |
| `experiments/080_oracle_ordering/mint.py` | the only BFS user: draws, labels, probe, probe-leaf distances; atomic, sha-checked mint files |
| `experiments/080_oracle_ordering/train.py` | MSE training of one (arm, seed), resumable, the training record with Gate L from the minted leaf table |
| `experiments/080_oracle_ordering/evaluate.py` | Gate 0's X cell, J4V-O-Xt search cells, the pilot timing cell |
| `experiments/080_oracle_ordering/aggregate.py` | record checks, Gates 0, V, T, 1, Claims 1 and 2, search reading, verdict, reported lines, pilot report |
| `experiments/080_oracle_ordering/launch080.ps1` | laptop launcher, one phase per call, amendment guard, memory sampling of every python process |
| `experiments/080_oracle_ordering/chain080.ps1` | the full run after the amendment: mint, gate0, Gate 0 check, train, gates, search |
| `tests/experiments/test_exp080.py` | every test in this plan |

Mint files live at `<out>/mint/mint_s<seed>.npz` plus `mint_s<seed>.json`. `experiments/*/outputs/*` and `experiments/*/outputs_*/` are gitignored, so mint files are untracked by default; records and judges are added with `git add -f` at results time, never the mint.

---

### Task 1: cells.py (constants, the probe-leaf exclusion, the three judges)

**Files:**
- Create: `experiments/080_oracle_ordering/cells.py`
- Create: `tests/experiments/test_exp080.py`

**Interfaces:**
- Consumes: `experiments/075_wide_region/cells.py` (`build_judge`, `fresh_sensory`, `load_pretrained`, `encoder_path`, `ckpt_dir`, `record_name`, `check_seeds`, `exclusion_set`, `probe_set`, `EVAL_SEEDS`, `PILOT_SEEDS`); `experiments/075_wide_region/evaluate.py` (`load_judge(train_arm, seed, judge_dir=None, required_updates=4000)`); `experiments/079_judge_trained_deep/cells.py` (`make_judge(seed, enc_root)`, `encoder_root(pilot)`, `drift(sensory_state, e0_state)`).
- Produces:
  - constants `ARMS = ("O-raw", "O-Xf", "O-Xt")`, `TRAIN_ORDER = ("O-Xt", "O-Xf", "O-raw")`, `EVAL_SEEDS`, `PILOT_SEEDS`, `MAX_LEN = 40`, `N_UPDATES = 4000`, `BATCH = 1000`, `PROBE_EVERY = 250`, `RAW_STATES = 600_000`, `RAW_EPOCHS = 12`, `RAW_BATCH = 512`, `RAW_HIDDEN = 512`, paths `OUT`, `PILOT_OUT`, `E75_OUT`, `E75_PILOT`, `E76_OUT`, `E79_OUT`.
  - `check_seeds(seeds, pilot)`, `exclusion_set(seed, provider)`, `probe_set(seed, provider)`, `drift(a, b)` (re-exports).
  - `x_judge_root(pilot) -> Path`, `encoder_root(pilot) -> Path`.
  - `ckpt_dir(arm, seed, out_dir=None) -> Path`, `record_name(arm, seed) -> str`.
  - `leaf_exclusion(probe, n_actions=N_ACTIONS, k=3, lo=7, hi=11) -> set`.
  - `oracle_exclusion(seed, provider, probe=None) -> set`.
  - `class RawJudge(nn.Module)` with `one_hot(states) -> Tensor` (static) and `forward(states, generator=None) -> Tensor` of shape `(n,)`.
  - `make_oraw(seed) -> RawJudge`, `make_oxf(seed, x_root) -> vi.Judge`, `make_oxt(seed, enc_root) -> vi.Judge`, `make_judge(arm, seed, x_root, enc_root)`.
  - `optimizer_for(arm, judge) -> torch.optim.Optimizer`.
  - `reference_sensory(arm, seed, x_root, enc_root) -> dict` (the state dict the region's drift is measured from).
  - `load_trained(arm, seed, judge_dir=None, required=None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/experiments/test_exp080.py`:

```python
"""EXP-080: oracle judges trained on the true distance, as instruments for X's far ordering."""

from __future__ import annotations

import importlib.util
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.encoder_pretrain import save_encoder
from neuromorphic.training.lookahead import tree_levels

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "080_oracle_ordering"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp080_cells", EXP / "cells.py")
c75 = cells.c75


def _fake_x(d, seed):
    """EXP-075's layout for one seed under `d`: the pretrained encoder, and an X judge at 4000
    updates whose region has moved away from it (so 'X's trained region' and 'the pretrained
    encoder' are different tensors)."""
    enc = c75.fresh_sensory("X", seed)
    path = c75.encoder_path("X", seed, d)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder(enc, path)
    judge = c75.build_judge(seed, "X", c75.fresh_sensory("X", seed))
    with torch.no_grad():
        judge.sensory.fc1.weight.add_(0.01)
        judge.head[0].weight.add_(0.5)
    ck = c75.ckpt_dir("X", seed, d)
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(judge.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": 4000}))
    return judge


def test_constants_are_the_spec_values():
    """Catches an arm, budget, walk length or seed set drifting from spec section 3."""
    assert cells.ARMS == ("O-raw", "O-Xf", "O-Xt")
    assert cells.TRAIN_ORDER == ("O-Xt", "O-Xf", "O-raw")
    assert (cells.MAX_LEN, cells.N_UPDATES, cells.BATCH, cells.PROBE_EVERY) == (40, 4000, 1000, 250)
    assert (cells.RAW_STATES, cells.RAW_EPOCHS, cells.RAW_BATCH, cells.RAW_HIDDEN) == (
        600_000, 12, 512, 512)
    assert cells.EVAL_SEEDS == tuple(range(12)) and cells.PILOT_SEEDS == (12, 13)
    assert cells.x_judge_root(False) == cells.E75_OUT and cells.x_judge_root(True) == cells.E75_PILOT
    assert cells.encoder_root(False) == cells.E75_OUT
    assert cells.record_name("O-Xt", 3) == "exp080_train_O-Xt_s3.json"
    assert cells.ckpt_dir("O-Xf", 3, Path("/x")) == Path("/x/judge_O-Xf_s3")


def test_leaf_exclusion_is_every_3_move_leaf_of_probe_states_7_to_11():
    """Catches leaves taken at the wrong depth, from probe states outside 7 to 11, or not at all
    (the label leak spec section 3 closes)."""
    a = apply_move(SOLVED, 0)
    b = apply_move(SOLVED, 1)
    c = apply_move(SOLVED, 2)
    probe = [(a, 7), (b, 11), (c, 6)]
    got = cells.leaf_exclusion(probe)
    want = set(tree_levels(a, 3, N_ACTIONS)[3]) | set(tree_levels(b, 3, N_ACTIONS)[3])
    assert got == want
    assert not set(tree_levels(c, 3, N_ACTIONS)[3]) <= got  # distance 6 contributes nothing


def test_oracle_exclusion_is_heldout_plus_probe_plus_leaves(monkeypatch):
    """Catches an exclusion that drops the held-out states, the probe states or their leaves."""
    a = apply_move(SOLVED, 0)
    held = {apply_move(SOLVED, 5)}
    monkeypatch.setattr(cells, "exclusion_set", lambda seed, provider: set(held))
    monkeypatch.setattr(cells, "probe_set", lambda seed, provider: [(a, 9)])
    got = cells.oracle_exclusion(0, None)
    assert held <= got and a in got
    assert set(tree_levels(a, 3, N_ACTIONS)[3]) <= got
    assert got == held | {a} | set(tree_levels(a, 3, N_ACTIONS)[3])


def test_raw_judge_shape_seed_and_generator():
    """Catches an O-raw net of the wrong shape, an init not fixed by the seed, or a forward that
    cannot be called the way leaf_rank_margin and probe_judge call a judge."""
    j0, j0b, j1 = cells.make_oraw(0), cells.make_oraw(0), cells.make_oraw(1)
    shapes = [tuple(p.shape) for p in j0.parameters()]
    assert shapes == [(512, 144), (512,), (512, 512), (512,), (1, 512), (1,)]
    assert all(torch.equal(p, q) for p, q in zip(j0.parameters(), j0b.parameters()))
    assert not torch.equal(j0.net[0].weight, j1.net[0].weight)
    states = [SOLVED, apply_move(SOLVED, 3)]
    out = j0(states, torch.Generator().manual_seed(0))
    assert out.shape == (2,)
    assert torch.equal(out, j0(np.array(states, dtype=np.uint8)))
    oh = cells.RawJudge.one_hot([SOLVED])
    assert oh.shape == (1, 144) and float(oh.sum()) == 24.0


def test_oxf_is_xs_trained_region_under_oxts_fresh_head(tmp_path):
    """Catches O-Xf built on X's trained head, on the pretrained encoder instead of X's trained
    region, or with a region that is not frozen; and O-Xt not built from the encoder."""
    x = _fake_x(tmp_path, 12)
    oxf = cells.make_oxf(12, tmp_path)
    oxt = cells.make_oxt(12, tmp_path)
    for k, v in oxf.head.state_dict().items():
        assert torch.equal(v, oxt.head.state_dict()[k])
    assert not torch.equal(oxf.head[0].weight, x.head[0].weight)
    assert torch.equal(oxf.sensory.fc1.weight, x.sensory.fc1.weight)
    assert not torch.equal(oxt.sensory.fc1.weight, x.sensory.fc1.weight)
    enc = c75.load_pretrained("X", 12, tmp_path)
    assert torch.equal(oxt.sensory.fc1.weight, enc.fc1.weight)
    assert all(not p.requires_grad for p in oxf.sensory.parameters())
    assert all(p.requires_grad for p in oxt.sensory.parameters())


def test_optimizers_are_the_spec_ones(tmp_path):
    """Catches O-Xf's optimizer touching the region, O-Xt's missing it, or a wrong lr."""
    _fake_x(tmp_path, 12)
    oxf, oxt, raw = (cells.make_judge(a, 12, tmp_path, tmp_path) for a in ("O-Xf", "O-Xt", "O-raw"))
    f, t, r = (cells.optimizer_for(a, j) for a, j in (("O-Xf", oxf), ("O-Xt", oxt), ("O-raw", raw)))
    f_params = {id(p) for g in f.param_groups for p in g["params"]}
    assert f_params == {id(p) for p in oxf.head.parameters()}
    assert [g["lr"] for g in t.param_groups] == [1e-3, 1e-4]
    assert {id(p) for p in t.param_groups[1]["params"]} == {id(p) for p in oxt.sensory.parameters()}
    assert [g["lr"] for g in r.param_groups] == [1e-3]
    with pytest.raises(ValueError):
        cells.optimizer_for("S", raw)


def test_reference_sensory_is_xs_region_for_oxf_and_the_encoder_for_oxt(tmp_path):
    """Catches a drift measured from the wrong starting point, which would make O-Xf's 0.0 or
    O-Xt's > 0 meaningless."""
    x = _fake_x(tmp_path, 12)
    rf = cells.reference_sensory("O-Xf", 12, tmp_path, tmp_path)
    rt = cells.reference_sensory("O-Xt", 12, tmp_path, tmp_path)
    assert torch.equal(rf["fc1.weight"], x.sensory.fc1.weight)
    assert torch.equal(rt["fc1.weight"], c75.load_pretrained("X", 12, tmp_path).fc1.weight)
    assert cells.drift(cells.make_oxf(12, tmp_path).sensory.state_dict(), rf) == 0.0
    assert cells.drift(cells.make_oxt(12, tmp_path).sensory.state_dict(), rt) == 0.0


def test_load_trained_refuses_a_missing_or_unfinished_judge(tmp_path):
    """Catches evaluating a judge that is absent or short of its budget (judge.pt exists from the
    first checkpoint, so its existence proves nothing)."""
    with pytest.raises(SystemExit, match="missing"):
        cells.load_trained("O-Xt", 12, tmp_path)
    j = cells.make_oraw(12)
    ck = cells.ckpt_dir("O-raw", 12, tmp_path)
    ck.mkdir(parents=True)
    torch.save(j.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"epochs": 3}))
    with pytest.raises(SystemExit, match="epochs 3"):
        cells.load_trained("O-raw", 12, tmp_path)
    got = cells.load_trained("O-raw", 12, tmp_path, required=3)
    assert torch.equal(got.net[0].weight, j.net[0].weight) and not got.training
    x = _fake_x(tmp_path / "x", 12)
    ck = cells.ckpt_dir("O-Xt", 12, tmp_path)
    ck.mkdir(parents=True)
    torch.save(x.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": 250}))
    with pytest.raises(SystemExit, match="update 250"):
        cells.load_trained("O-Xt", 12, tmp_path)
    (ck / "state.json").write_text(json.dumps({"update": 4000}))
    got = cells.load_trained("O-Xt", 12, tmp_path)
    assert torch.equal(got.sensory.fc1.weight, x.sensory.fc1.weight)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q` (tool timeout 600000)
Expected: collection ERROR, `experiments/080_oracle_ordering/cells.py` does not exist.

- [ ] **Step 3: Write cells.py**

Create `experiments/080_oracle_ordering/cells.py`:

```python
"""EXP-080 cells: three judges trained on the TRUE distance, as instruments for whether X's
encoder can carry far ordering. None is a candidate system; distance-to-solved is never a model
input in the method, and these arms exist to price it.

O-raw is a plain MLP on one-hot facelets (the protocol check). O-Xf is EXP-075's committed X
region, frozen, under a fresh wide head. O-Xt is EXP-079's S recipe (X's pretrained encoder, a
fresh wide head, arm W's optimizer) with the true distance as its target.

Spec: docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md, section 3.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c75 = _load("exp075_cells", REPO / "experiments" / "075_wide_region" / "cells.py")
e75 = _load("exp075_evaluate", REPO / "experiments" / "075_wide_region" / "evaluate.py")
c79 = _load("exp079_cells", REPO / "experiments" / "079_judge_trained_deep" / "cells.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402
from neuromorphic.training.lookahead import tree_levels  # noqa: E402

ARMS = ("O-raw", "O-Xf", "O-Xt")
# Longest first, so a pool's last wave is the cheap arms.
TRAIN_ORDER = ("O-Xt", "O-Xf", "O-raw")
EVAL_SEEDS = tuple(c75.EVAL_SEEDS)
PILOT_SEEDS = tuple(c75.PILOT_SEEDS)
MAX_LEN = 40
N_UPDATES = 4000
BATCH = 1000
PROBE_EVERY = 250
RAW_STATES = 600_000
RAW_EPOCHS = 12
RAW_BATCH = 512
RAW_HIDDEN = 512
N_COLOURS = 6
N_FACELETS = 24
# Gate L's tree: the leaves leaf_rank_margin ranks, which training must never see.
LEAF_K, LEAF_LO, LEAF_HI = 3, 7, 11

E75_OUT = REPO / "experiments" / "075_wide_region" / "outputs"
E75_PILOT = REPO / "experiments" / "075_wide_region" / "outputs_pilot"
E76_OUT = REPO / "experiments" / "076_deeper_tree" / "outputs"
E79_OUT = REPO / "experiments" / "079_judge_trained_deep" / "outputs"
OUT = HERE / "outputs"
PILOT_OUT = HERE / "outputs_pilot"

check_seeds = c75.check_seeds
exclusion_set = c75.exclusion_set
probe_set = c75.probe_set
drift = c79.drift


def x_judge_root(pilot: bool) -> Path:
    """Where EXP-075's trained X judges live: committed outputs, or the laptop's pilot folder."""
    return E75_PILOT if pilot else E75_OUT


def encoder_root(pilot: bool) -> Path:
    """Where X's pretrained encoders live, exactly as EXP-079 read them (its pilot copies were
    proved to be EXP-075's pilot encoders)."""
    return c79.encoder_root(pilot)


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def ckpt_dir(arm: str, seed: int, out_dir=None) -> Path:
    _check_arm(arm)
    return (Path(out_dir) if out_dir is not None else OUT) / f"judge_{arm}_s{seed}"


def record_name(arm: str, seed: int) -> str:
    _check_arm(arm)
    return f"exp080_train_{arm}_s{seed}.json"


def leaf_exclusion(probe, n_actions=N_ACTIONS, k=LEAF_K, lo=LEAF_LO, hi=LEAF_HI) -> set:
    """Every k-move leaf of every probe state at true distance lo..hi: the states
    leaf_rank_margin ranks. With true labels, a judge trained on one would have seen the answer."""
    out: set = set()
    for s, d in probe:
        if lo <= d <= hi:
            out.update(tree_levels(s, k, n_actions)[k])
    return out


def oracle_exclusion(seed: int, provider, probe=None) -> set:
    """Held-out states, probe states, and the probe leaves (spec section 3)."""
    probe = probe_set(seed, provider) if probe is None else probe
    return exclusion_set(seed, provider) | {s for s, _ in probe} | leaf_exclusion(probe)


class RawJudge(nn.Module):
    """O-raw: an MLP on one-hot facelets. `forward` takes and ignores a generator, so
    leaf_rank_margin and probe_judge call it as they call a spiking judge."""

    def __init__(self, hidden: int = RAW_HIDDEN):
        super().__init__()
        n_in = N_FACELETS * N_COLOURS
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 1))

    @staticmethod
    def one_hot(states) -> torch.Tensor:
        idx = torch.as_tensor(np.asarray(states), dtype=torch.long)
        return F.one_hot(idx, N_COLOURS).flatten(1).float()

    def forward(self, states, generator=None) -> torch.Tensor:
        return self.net(self.one_hot(states)).squeeze(-1)


def make_oraw(seed: int) -> RawJudge:
    torch.manual_seed(seed)
    return RawJudge()


def make_oxf(seed: int, x_root) -> vi.Judge:
    """X's committed trained region (refused short of 4000 updates), frozen, under the fresh wide
    head `c75.build_judge` seeds: the same head O-Xt and S start from."""
    x = e75.load_judge("X", seed, x_root)
    judge = c75.build_judge(seed, "X", c75.fresh_sensory("X", seed))
    judge.sensory.load_state_dict(x.sensory.state_dict())
    for p in judge.sensory.parameters():
        p.requires_grad_(False)
    return judge


def make_oxt(seed: int, enc_root) -> vi.Judge:
    """EXP-079's S before training: X's pretrained encoder under a fresh wide head."""
    return c79.make_judge(seed, enc_root)


def make_judge(arm: str, seed: int, x_root, enc_root):
    _check_arm(arm)
    if arm == "O-raw":
        return make_oraw(seed)
    if arm == "O-Xf":
        return make_oxf(seed, x_root)
    return make_oxt(seed, enc_root)


def optimizer_for(arm: str, judge):
    if arm == "O-raw":
        return torch.optim.Adam(judge.parameters(), lr=1e-3)
    if arm == "O-Xf":
        return vi.make_optimizer(judge, "B")
    if arm == "O-Xt":
        return vi.make_optimizer(judge, "W")
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def reference_sensory(arm: str, seed: int, x_root, enc_root) -> dict:
    """The region state an arm starts from, which its encoder_drift is measured against."""
    if arm == "O-Xf":
        return e75.load_judge("X", seed, x_root).sensory.state_dict()
    if arm == "O-Xt":
        return c75.load_pretrained("X", seed, enc_root).state_dict()
    raise ValueError(f"{arm!r} has no region")


def load_trained(arm: str, seed: int, judge_dir=None, required=None):
    """A trained judge from its checkpoint, refused unless training finished: O-raw at
    `required` epochs (default RAW_EPOCHS), O-Xf and O-Xt at `required` updates (default
    N_UPDATES)."""
    ck = ckpt_dir(arm, seed, judge_dir)
    path, state_path = ck / "judge.pt", ck / "state.json"
    for p in (path, state_path):
        if not p.exists():
            raise SystemExit(f"missing {p} for {arm} seed {seed}; train it with train.py first")
    state = json.loads(state_path.read_text())
    if arm == "O-raw":
        want = RAW_EPOCHS if required is None else required
        if state.get("epochs") != want:
            raise SystemExit(f"{state_path}: {arm} seed {seed} is at epochs {state.get('epochs')}, "
                             f"needs {want}; finish training first")
        judge = RawJudge()
    else:
        want = N_UPDATES if required is None else required
        if state.get("update") != want:
            raise SystemExit(f"{state_path}: {arm} seed {seed} is at update {state.get('update')}, "
                             f"needs {want}; finish training first")
        judge = c75.build_judge(seed, "X", c75.fresh_sensory("X", seed))
    judge.load_state_dict(torch.load(path, map_location="cpu"))
    judge.eval()
    return judge
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q` (tool timeout 600000)
Expected: 8 passed.

- [ ] **Step 5: Mutation check**

Apply each mutation, run the named test, confirm it FAILS, then revert (`git diff` must be empty for `cells.py` afterwards):
1. In `make_oxf`, delete the `requires_grad_(False)` loop: `test_oxf_is_xs_trained_region_under_oxts_fresh_head` fails.
2. In `oracle_exclusion`, drop `| leaf_exclusion(probe)`: `test_oracle_exclusion_is_heldout_plus_probe_plus_leaves` fails.
3. In `leaf_exclusion`, use `tree_levels(s, k, n_actions)[k - 1]`: `test_leaf_exclusion_is_every_3_move_leaf_of_probe_states_7_to_11` fails.
4. In `make_oxf`, return `x` (X's own judge, trained head): `test_oxf_is_xs_trained_region_under_oxts_fresh_head` fails.

- [ ] **Step 6: Commit**

```bash
git add experiments/080_oracle_ordering/cells.py tests/experiments/test_exp080.py
git commit -m "EXP-080 cells: the three oracle judges, the probe-leaf exclusion, trained-judge loading"
```

---

### Task 2: mint.py (labels, probe and probe-leaf distances, once per seed)

**Files:**
- Create: `experiments/080_oracle_ordering/mint.py`
- Modify: `tests/experiments/test_exp080.py` (append)

**Interfaces:**
- Consumes: Task 1's `cells.oracle_exclusion`, `cells.leaf_exclusion`, `cells.probe_set`, `cells.N_UPDATES`, `cells.BATCH`, `cells.MAX_LEN`, `cells.check_seeds`, `cells.OUT`, `cells.PILOT_OUT`, `cells.EVAL_SEEDS`; `vi.random_walk_states(n, max_len, rng, n_actions, exclude)`.
- Produces:
  - `mint_dir(out_dir) -> Path` (`<out_dir>/mint`), `npz_name(seed)`, `meta_name(seed)`.
  - `draw_blocks(n_blocks, block, max_len, rng_seed, exclude, distance, n_actions=N_ACTIONS) -> (states uint8 (n, 24), labels uint8 (n,))`.
  - `leaf_table(probe, distance, n_actions=N_ACTIONS) -> (leaf_states uint8 (m, 24), leaf_d uint8 (m,))`.
  - `digest(*arrays) -> str` (sha256 hex).
  - `write_mint(out_dir, seed, states, labels, probe, leaf_states, leaf_d, meta) -> dict` (the sidecar written).
  - `class Mint(NamedTuple)` with fields `seed, states, labels, probe, leaf_dist, sha, meta`; `probe` is `list[(tuple[int, ...], int)]`, `leaf_dist` is `dict[tuple[int, ...], int]`.
  - `read_mint(out_dir, seed) -> Mint`.
  - `mint_seed(seed, out_dir, n_blocks=N_UPDATES, block=BATCH, max_len=MAX_LEN, provider=None) -> dict`.
  - `pending_seeds(seeds, out_dir) -> list[int]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp080.py`:

```python
mint = _load("exp080_mint", EXP / "mint.py")


class FakeDistance:
    """A deterministic stand-in for the BFS table. Mint plumbing does not care whether a label
    is physically right, only that it is the provider's answer for that state."""

    def distance(self, s):
        return sum(i * int(v) for i, v in enumerate(s)) % 15


def _probe(n=5, seed=99):
    states = vi.random_walk_states(n, 6, random.Random(seed), N_ACTIONS, set())
    return [(s, 7 + i % 5) for i, s in enumerate(states)]


def _fake_mint(root, seed, n_blocks=4, block=8, max_len=6, probe=None):
    fd = FakeDistance()
    probe = _probe() if probe is None else probe
    states, labels = mint.draw_blocks(n_blocks, block, max_len, seed, set(), fd.distance)
    ls, ld = mint.leaf_table(probe, fd.distance)
    mint.write_mint(root, seed, states, labels, probe, ls, ld,
                    {"seed": seed, "n_blocks": n_blocks, "block": block, "max_len": max_len})
    return mint.read_mint(root, seed)


def test_draw_blocks_is_random_walk_states_called_once_per_block():
    """Catches draws that are not the library's walks in update order (one call per block from
    one random.Random(seed)), or labels that are not the provider's."""
    fd = FakeDistance()
    states, labels = mint.draw_blocks(3, 5, 7, 4, set(), fd.distance)
    rng = random.Random(4)
    want = [s for _ in range(3) for s in vi.random_walk_states(5, 7, rng, N_ACTIONS, set())]
    assert states.dtype == np.uint8 and states.shape == (15, 24)
    assert [tuple(int(x) for x in r) for r in states] == want
    assert labels.dtype == np.uint8 and labels.tolist() == [fd.distance(s) for s in want]


def test_draw_blocks_never_returns_an_excluded_state():
    """Catches an exclusion that is ignored. The control draw proves the excluded states are ones
    the walks actually reach, so the assertion can fail."""
    fd = FakeDistance()
    probe = [(apply_move(SOLVED, a), 7) for a in range(N_ACTIONS)]
    leaves = cells.leaf_exclusion(probe)
    control, _ = mint.draw_blocks(20, 50, 4, 0, set(), fd.distance)
    assert sum(tuple(int(x) for x in r) in leaves for r in control) > 50
    states, _ = mint.draw_blocks(20, 50, 4, 0, leaves, fd.distance)
    assert not any(tuple(int(x) for x in r) in leaves for r in states)


def test_draw_blocks_refuses_an_unlabelled_state():
    """Catches a bounded provider silently minting None (stored as 0) for a far state."""
    with pytest.raises(SystemExit, match="no distance"):
        mint.draw_blocks(1, 4, 5, 0, set(), lambda s: None)


def test_mint_round_trips_probe_and_leaf_distances(tmp_path):
    """Catches a probe whose order or labels change on disk (leaf_rank_margin consumes its
    generator in probe order), or leaf keys that do not match what tree_levels produces."""
    probe = _probe()
    m = _fake_mint(tmp_path, 12, probe=probe)
    assert m.probe == probe
    assert all(type(x) is int for x in m.probe[0][0]) and type(m.probe[0][1]) is int
    leaf = tree_levels(probe[0][0], 3, N_ACTIONS)[3][17]
    assert m.leaf_dist[leaf] == FakeDistance().distance(leaf)
    assert len(m.leaf_dist) == len(cells.leaf_exclusion(probe))
    assert m.meta["sha256"] == m.sha and m.seed == 12


def test_read_mint_refuses_a_tampered_file(tmp_path):
    """Catches training on a mint whose arrays are not the ones its sidecar describes (a partial
    write, or a file swapped after a re-mint)."""
    _fake_mint(tmp_path, 12)
    npz = mint.mint_dir(tmp_path) / mint.npz_name(12)
    z = dict(np.load(npz))
    z["labels"] = z["labels"].copy()
    z["labels"][0] = (int(z["labels"][0]) + 1) % 15
    np.savez(npz, **z)
    with pytest.raises(SystemExit, match="sha256"):
        mint.read_mint(tmp_path, 12)


def test_read_mint_refuses_a_missing_mint_or_a_wrong_seed(tmp_path):
    """Catches training a seed on another seed's labels, or on nothing."""
    with pytest.raises(SystemExit, match="mint.py"):
        mint.read_mint(tmp_path, 12)
    _fake_mint(tmp_path, 12)
    d = mint.mint_dir(tmp_path)
    (d / mint.npz_name(13)).write_bytes((d / mint.npz_name(12)).read_bytes())
    (d / mint.meta_name(13)).write_text((d / mint.meta_name(12)).read_text())
    with pytest.raises(SystemExit, match="seed 12"):
        mint.read_mint(tmp_path, 13)


def test_mint_is_pending_until_its_sidecar_exists(tmp_path):
    """Catches a relaunch that skips a seed whose mint was killed after the arrays and before
    the sidecar (the 03:31 restart)."""
    _fake_mint(tmp_path, 12)
    d = mint.mint_dir(tmp_path)
    (d / mint.npz_name(13)).write_bytes(b"partial")
    assert mint.pending_seeds([12, 13], tmp_path) == [13]
    assert not list(d.glob("*.tmp*"))


def test_mint_seed_excludes_probe_leaves_and_records_its_settings(tmp_path, monkeypatch):
    """Catches the glue dropping the probe leaves from the exclusion, or a sidecar that does not
    state the draw it holds."""
    probe = [(apply_move(SOLVED, a), 7) for a in range(N_ACTIONS)]
    leaves = cells.leaf_exclusion(probe)
    # mint.py loads its own copy of cells.py, so patch that copy.
    monkeypatch.setattr(mint.cells, "probe_set", lambda seed, provider: list(probe))
    monkeypatch.setattr(mint.cells, "exclusion_set", lambda seed, provider: set())
    meta = mint.mint_seed(12, tmp_path, n_blocks=20, block=50, max_len=4, provider=FakeDistance())
    m = mint.read_mint(tmp_path, 12)
    assert not any(tuple(int(x) for x in r) in leaves for r in m.states)
    assert m.probe == probe and set(m.leaf_dist) == leaves
    for k, v in {"seed": 12, "n_blocks": 20, "block": 50, "max_len": 4,
                 "rng": "random.Random(seed)", "n_probe": 6, "n_leaves": len(leaves)}.items():
        assert meta[k] == v and m.meta[k] == v
    assert meta["n_exclude"] == len(leaves | {s for s, _ in probe})
    assert sum(meta["label_hist"].values()) == 1000
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q` (tool timeout 600000)
Expected: collection ERROR, `mint.py` does not exist.

- [ ] **Step 3: Write mint.py**

Create `experiments/080_oracle_ordering/mint.py`:

```python
"""EXP-080 mint: the only place this experiment builds the unbounded BFS table.

Per seed: draws the training states (4000 blocks of 1000, `random_walk_states(1000, 40, ...)`
once per block from one `random.Random(seed)`, excluding held-out states, probe states and probe
leaves), looks up their true distances, and writes them with the seed's probe set and the true
distance of every probe leaf. Training and every margin cell read this file, so no other worker
holds the 1.9 GB table. Arrays are written first and the sidecar (which carries their sha256)
last, each atomically, so a mint killed mid-write leaves the seed pending.

Spec: docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md, sections 3
and 4.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/080_oracle_ordering/mint.py --pilot --seeds 12 13 --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp080_cells", HERE / "cells.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

N_FACELETS = cells.N_FACELETS


def mint_dir(out_dir) -> Path:
    return Path(out_dir) / "mint"


def npz_name(seed: int) -> str:
    return f"mint_s{seed}.npz"


def meta_name(seed: int) -> str:
    return f"mint_s{seed}.json"


def draw_blocks(n_blocks, block, max_len, rng_seed, exclude, distance, n_actions=N_ACTIONS):
    """`n_blocks` calls of random_walk_states(block, ...) from one random.Random(rng_seed), in
    order, with each state's label from `distance`."""
    rng = random.Random(rng_seed)
    n = n_blocks * block
    states = np.empty((n, N_FACELETS), dtype=np.uint8)
    labels = np.empty(n, dtype=np.uint8)
    i = 0
    for _ in range(n_blocks):
        for s in vi.random_walk_states(block, max_len, rng, n_actions, exclude):
            d = distance(s)
            if d is None:
                raise SystemExit(f"state {s} has no distance: mint needs the unbounded table")
            states[i] = s
            labels[i] = d
            i += 1
    return states, labels


def leaf_table(probe, distance, n_actions=N_ACTIONS):
    """Every probe leaf leaf_rank_margin ranks, sorted, with its true distance."""
    leaves = sorted(cells.leaf_exclusion(probe, n_actions))
    ls = np.array(leaves, dtype=np.uint8).reshape(-1, N_FACELETS)
    ld = np.array([distance(s) for s in leaves], dtype=np.uint8)
    return ls, ld


def digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _arrays(states, labels, probe, leaf_states, leaf_d) -> dict:
    return {"states": states, "labels": labels,
            "probe_states": np.array([s for s, _ in probe], dtype=np.uint8).reshape(-1, N_FACELETS),
            "probe_d": np.array([d for _, d in probe], dtype=np.uint8),
            "leaf_states": leaf_states, "leaf_d": leaf_d}


_ORDER = ("states", "labels", "probe_states", "probe_d", "leaf_states", "leaf_d")


def write_mint(out_dir, seed, states, labels, probe, leaf_states, leaf_d, meta) -> dict:
    d = mint_dir(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    arrays = _arrays(states, labels, probe, leaf_states, leaf_d)
    sidecar = {**meta, "seed": seed, "sha256": digest(*(arrays[k] for k in _ORDER))}
    tmp = d / f"mint_s{seed}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, d / npz_name(seed))
    tmp_meta = d / f"mint_s{seed}.tmp.json"
    tmp_meta.write_text(json.dumps(sidecar, indent=1, sort_keys=True) + "\n")
    os.replace(tmp_meta, d / meta_name(seed))
    return sidecar


# A NamedTuple, not a dataclass: dataclasses look the class's module up in sys.modules,
# and a module loaded through importlib.util is not registered there.
class Mint(NamedTuple):
    seed: int
    states: np.ndarray
    labels: np.ndarray
    probe: list
    leaf_dist: dict
    sha: str
    meta: dict


def _tuple(row) -> tuple:
    return tuple(int(x) for x in row)


def read_mint(out_dir, seed) -> Mint:
    d = mint_dir(out_dir)
    npz, js = d / npz_name(seed), d / meta_name(seed)
    for p in (npz, js):
        if not p.exists():
            raise SystemExit(f"missing {p}: run mint.py for seed {seed} first")
    meta = json.loads(js.read_text())
    if meta.get("seed") != seed:
        raise SystemExit(f"{js} is the mint of seed {meta.get('seed')}, not seed {seed}")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in _ORDER}
    sha = digest(*(arrays[k] for k in _ORDER))
    if sha != meta.get("sha256"):
        raise SystemExit(f"{npz}: sha256 {sha} does not match its sidecar's "
                         f"{meta.get('sha256')}; re-mint seed {seed}")
    probe = [(_tuple(s), int(dd)) for s, dd in zip(arrays["probe_states"], arrays["probe_d"])]
    leaf_dist = {_tuple(s): int(dd) for s, dd in zip(arrays["leaf_states"], arrays["leaf_d"])}
    return Mint(seed, arrays["states"], arrays["labels"], probe, leaf_dist, sha, meta)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def mint_seed(seed, out_dir, n_blocks=cells.N_UPDATES, block=cells.BATCH, max_len=cells.MAX_LEN,
              provider=None) -> dict:
    t0 = time.time()
    if provider is None:
        provider = ExactBFSDistance(max_depth=None)
    probe = cells.probe_set(seed, provider)
    excl = cells.oracle_exclusion(seed, provider, probe)
    states, labels = draw_blocks(n_blocks, block, max_len, seed, excl, provider.distance)
    ls, ld = leaf_table(probe, provider.distance)
    hist = Counter(int(x) for x in labels)
    meta = {"seed": seed, "n_blocks": n_blocks, "block": block, "max_len": max_len,
            "rng": "random.Random(seed)", "n_exclude": len(excl), "n_probe": len(probe),
            "n_leaves": int(len(ld)), "label_hist": {str(k): hist[k] for k in sorted(hist)},
            "git_commit": _git_commit(), "wall_s": round(time.time() - t0, 1)}
    return write_mint(out_dir, seed, states, labels, probe, ls, ld, meta)


def pending_seeds(seeds, out_dir) -> list[int]:
    return [s for s in seeds if not (mint_dir(out_dir) / meta_name(s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="outputs/ (outputs_pilot/ with --pilot) when omitted")
    args = ap.parse_args()
    cells.check_seeds(args.seeds, args.pilot)
    out = args.out_dir if args.out_dir is not None else (
        cells.PILOT_OUT if args.pilot else cells.OUT)
    jobs = pending_seeds(args.seeds, out)
    print(f"EXP-080 mint: {len(jobs)} seeds, {args.workers} workers, into {mint_dir(out)}",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(mint_seed, s, out): s for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            m = f.result()
            print(f"  {i}/{len(jobs)}  seed {m['seed']}  n_exclude {m['n_exclude']}  "
                  f"n_leaves {m['n_leaves']}  wall_s {m['wall_s']}  sha {m['sha256'][:12]}",
                  flush=True)
    print("done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q` (tool timeout 600000)
Expected: 16 passed.

- [ ] **Step 5: Mutation check**

Apply each, confirm the named test FAILS, revert:
1. In `mint_seed`, pass `cells.exclusion_set(seed, provider)` instead of `excl` to `draw_blocks`: `test_mint_seed_excludes_probe_leaves_and_records_its_settings` fails.
2. In `read_mint`, delete the sha comparison: `test_read_mint_refuses_a_tampered_file` fails.
3. In `pending_seeds`, test `npz_name(s)` instead of `meta_name(s)`: `test_mint_is_pending_until_its_sidecar_exists` fails.
4. In `draw_blocks`, create `rng = random.Random(rng_seed)` inside the block loop: `test_draw_blocks_is_random_walk_states_called_once_per_block` fails.

- [ ] **Step 6: Commit**

```bash
git add experiments/080_oracle_ordering/mint.py tests/experiments/test_exp080.py
git commit -m "EXP-080 mint: true-distance labels, probe and probe-leaf distances once per seed, sha-checked"
```

---

### Task 3: train.py (MSE on minted labels, resumable, never touches BFS)

**Files:**
- Create: `experiments/080_oracle_ordering/train.py`
- Modify: `tests/experiments/test_exp080.py` (append)

**Interfaces:**
- Consumes: Task 1's `cells.make_judge`, `cells.optimizer_for`, `cells.reference_sensory`, `cells.drift`, `cells.ckpt_dir`, `cells.record_name`, `cells.load_trained`, `cells.RawJudge`, `cells.x_judge_root`, `cells.encoder_root`, budgets; Task 2's `mint.read_mint(out_dir, seed) -> Mint` (fields `states`, `labels`, `probe`, `leaf_dist`, `sha`, `meta`); `gate_l.leaf_rank_margin(judge, probe, distance, seed, n_actions)`; `vi.probe_judge(judge, probe, generator)`.
- Produces:
  - `train_x_arm(arm, judge, m, *, n_updates, batch, probe_every, seed, ckpt_dir, log=print) -> dict` with keys `updates, resumed_from, wall_s, loss_last, probes, losses, final_probe`.
  - `train_oraw(judge, m, *, n_states, epochs, batch, seed, ckpt_dir, log=print) -> dict` with keys `updates, epochs, resumed_from, wall_s, loss_last, probes, losses, final_probe`.
  - `run(arm, seed, out_dir, *, n_updates=N_UPDATES, batch=BATCH, probe_every=PROBE_EVERY, raw_states=RAW_STATES, raw_epochs=RAW_EPOCHS, raw_batch=RAW_BATCH, mint_root=None, x_root=None, enc_root=None) -> dict` (writes `<out_dir>/exp080_train_<arm>_s<seed>.json`).
  - Record fields (read by Task 5): `arm, seed, mint_sha, mint_meta, gate_l, final_probe, probes, losses, loss_last, wall_s, resumed_from, git_commit`; O-Xf and O-Xt add `n_updates, batch, probe_every, updates, encoder_drift`; O-raw adds `raw_states, raw_epochs, raw_batch, raw_hidden, updates`.
  - `pending_jobs(jobs, out_dir) -> list[(arm, seed)]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp080.py`:

```python
train = _load("exp080_train", EXP / "train.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")


def _small(arm, tmp_path, n_updates=2, mint_root=None, **kw):
    """One tiny training run on seed 12: fake X layout, fake mint of 4 blocks of 8."""
    if not (tmp_path / "x").exists():
        _fake_x(tmp_path / "x", 12)
    if mint_root is None:
        mint_root = tmp_path / "m"
        if not (mint.mint_dir(mint_root) / mint.meta_name(12)).exists():
            _fake_mint(mint_root, 12)
    args = dict(n_updates=n_updates, batch=8, probe_every=1, raw_states=32, raw_epochs=2,
                raw_batch=8, mint_root=mint_root, x_root=tmp_path / "x", enc_root=tmp_path / "x")
    args.update(kw)
    return train.run(arm, 12, tmp_path / "out", **args)


def test_oxt_trains_its_region_and_oxf_does_not(tmp_path):
    """Catches a freeze that does not hold (O-Xf's region moves) and a gradient that never
    arrives (O-Xt's region does not), CLAUDE.md's MemoryReadout lesson; and a head that never
    trains."""
    rf = _small("O-Xf", tmp_path)
    rt = _small("O-Xt", tmp_path)
    assert rf["encoder_drift"] == 0.0
    assert rt["encoder_drift"] > 0.0
    f = cells.load_trained("O-Xf", 12, tmp_path / "out", required=2)
    t = cells.load_trained("O-Xt", 12, tmp_path / "out", required=2)
    fresh = cells.make_oxt(12, tmp_path / "x")
    assert not torch.equal(f.head[0].weight, fresh.head[0].weight)
    assert not torch.equal(t.head[0].weight, fresh.head[0].weight)
    x = c75.ckpt_dir("X", 12, tmp_path / "x") / "judge.pt"
    assert torch.equal(f.sensory.fc1.weight, torch.load(x)["sensory.fc1.weight"])


def test_record_carries_the_mint_and_a_gate_l_from_minted_distances(tmp_path):
    """Catches a record that cannot be paired across arms (no mint sha), or a Gate L that is not
    leaf_rank_margin on the minted probe and leaf distances."""
    rec = _small("O-Xt", tmp_path)
    m = mint.read_mint(tmp_path / "m", 12)
    on_disk = json.loads((tmp_path / "out" / cells.record_name("O-Xt", 12)).read_text())
    assert on_disk["mint_sha"] == m.sha and on_disk["mint_meta"]["n_blocks"] == 4
    assert (on_disk["arm"], on_disk["seed"], on_disk["n_updates"], on_disk["batch"]) == (
        "O-Xt", 12, 2, 8)
    j = cells.load_trained("O-Xt", 12, tmp_path / "out", required=2)
    want = gate_l.leaf_rank_margin(j, m.probe, m.leaf_dist.__getitem__, 12, N_ACTIONS)
    assert on_disk["gate_l"] == json.loads(json.dumps(want))
    assert on_disk["gate_l"]["n"] == 5 and rec["gate_l"] == want


def test_training_uses_block_u_at_update_u(tmp_path, monkeypatch):
    """Catches O-Xf and O-Xt reading different states at the same update (they must be paired on
    identical draws), or blocks read out of order."""
    seen = []
    real = train._block

    def spy(m, u, batch):
        xs, y = real(m, u, batch)
        seen.append((u, xs[0].tolist()))
        return xs, y

    monkeypatch.setattr(train, "_block", spy)
    _small("O-Xf", tmp_path, n_updates=3)
    m = mint.read_mint(tmp_path / "m", 12)
    assert seen == [(u, m.states[u * 8].tolist()) for u in range(3)]


def test_resume_equals_a_straight_run(tmp_path):
    """Catches a resume that restarts the data, the optimizer or the encoding generator, so a
    run interrupted by the 03:31 restart is not the run it would have been."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    for d in (a, b):
        _fake_x(d / "x", 12)
        _fake_mint(d / "m", 12)
    _small("O-Xt", a, n_updates=4, probe_every=2)
    _small("O-Xt", b, n_updates=2, probe_every=2)
    rb = _small("O-Xt", b, n_updates=4, probe_every=2)
    assert rb["resumed_from"] == 2
    ja = torch.load(cells.ckpt_dir("O-Xt", 12, a / "out") / "judge.pt")
    jb = torch.load(cells.ckpt_dir("O-Xt", 12, b / "out") / "judge.pt")
    assert all(torch.equal(ja[k], jb[k]) for k in ja)


def test_resume_refuses_a_checkpoint_from_a_different_mint(tmp_path):
    """Catches splicing two training sets into one judge after a seed is re-minted."""
    _small("O-Xt", tmp_path, n_updates=1)
    (tmp_path / "out" / cells.record_name("O-Xt", 12)).unlink()
    other = tmp_path / "m2"
    fd = FakeDistance()
    states, labels = mint.draw_blocks(4, 8, 6, 777, set(), fd.distance)
    probe = _probe()
    ls, ld = mint.leaf_table(probe, fd.distance)
    mint.write_mint(other, 12, states, labels, probe, ls, ld,
                    {"seed": 12, "n_blocks": 4, "block": 8, "max_len": 6})
    with pytest.raises(SystemExit, match="different mint"):
        _small("O-Xt", tmp_path, n_updates=2, mint_root=other)


def test_short_mint_is_refused_before_training(tmp_path):
    """Catches a run that would index past its mint and train on a truncated last block."""
    with pytest.raises(SystemExit, match="needs 40"):
        _small("O-Xt", tmp_path, n_updates=5)
    with pytest.raises(SystemExit, match="needs 64"):
        _small("O-raw", tmp_path, raw_states=64)


def test_oraw_trains_its_epochs_and_records_them(tmp_path):
    """Catches O-raw training a different number of states or epochs than its record says, or a
    record that a margin reader would mistake for a spiking arm."""
    rec = _small("O-raw", tmp_path)
    on_disk = json.loads((tmp_path / "out" / cells.record_name("O-raw", 12)).read_text())
    assert (on_disk["raw_states"], on_disk["raw_epochs"], on_disk["raw_batch"],
            on_disk["raw_hidden"]) == (32, 2, 8, 512)
    assert on_disk["updates"] == 8 and len(on_disk["losses"]) == 2
    assert "encoder_drift" not in on_disk and "n_updates" not in on_disk
    j = cells.load_trained("O-raw", 12, tmp_path / "out", required=2)
    assert not torch.equal(j.net[0].weight, cells.make_oraw(12).net[0].weight)
    assert rec["gate_l"]["n"] == 5


def test_training_never_builds_a_bfs_table(tmp_path, monkeypatch):
    """Catches a training worker that loads the 1.9 GB table (spec section 3: the worker count is
    set by the encoder alone)."""
    from neuromorphic.envs import cube_distance

    def boom(self, *a, **k):
        raise AssertionError("training built a BFS table")

    monkeypatch.setattr(cube_distance.ExactBFSDistance, "__init__", boom)
    _small("O-Xt", tmp_path)
    _small("O-raw", tmp_path)


def test_pending_jobs_skips_finished_runs_and_keeps_train_order(tmp_path):
    """Catches a relaunch that retrains finished runs or puts the long arm last."""
    (tmp_path / cells.record_name("O-Xf", 12)).write_text("{}")
    jobs = [(a, s) for a in cells.TRAIN_ORDER for s in (12, 13)]
    assert train.pending_jobs(jobs, tmp_path) == [
        ("O-Xt", 12), ("O-Xt", 13), ("O-Xf", 13), ("O-raw", 12), ("O-raw", 13)]
    args = train.build_parser().parse_args(["--n-updates", "4000"])
    assert args.arms == list(cells.TRAIN_ORDER)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q` (tool timeout 600000)
Expected: collection ERROR, `train.py` does not exist.

- [ ] **Step 3: Write train.py**

Create `experiments/080_oracle_ordering/train.py`:

```python
"""EXP-080 training driver: one (arm, seed) per job, MSE against the minted true distance.

O-Xf and O-Xt train on block u of the seed's mint at update u (paired on identical states),
4000 updates of 1000, checkpointing every 250 so a run killed by the 03:31 restart resumes where
it stopped. O-raw trains 12 epochs over the mint's first 600,000 states. No BFS table is built
here: labels, the probe set and every probe leaf's distance all come from the mint, and Gate L is
measured at the end from those minted distances.

Spec: docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md, section 3.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/080_oracle_ordering/train.py --pilot --seeds 12 13 \
        --n-updates 4000 --workers 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp080_cells", HERE / "cells.py")
mint = _load("exp080_mint", HERE / "mint.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

torch.set_num_threads(1)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _block(m, u, batch):
    """Update u's states and labels: rows u*batch to (u+1)*batch of the mint."""
    lo, hi = u * batch, (u + 1) * batch
    return m.states[lo:hi], torch.as_tensor(m.labels[lo:hi], dtype=torch.float32)


def _probe(judge, m, seed):
    return vi.probe_judge(judge, m.probe, torch.Generator().manual_seed(seed))


def train_x_arm(arm, judge, m, *, n_updates, batch, probe_every, seed, ckpt_dir, log=print):
    need = n_updates * batch
    if len(m.labels) < need:
        raise SystemExit(f"mint for seed {seed} holds {len(m.labels)} states; {arm} at "
                         f"{n_updates} x {batch} needs {need}")
    ck = Path(ckpt_dir)
    ck.mkdir(parents=True, exist_ok=True)
    state_path = ck / "state.json"
    opt = cells.optimizer_for(arm, judge)
    gen = torch.Generator().manual_seed(seed)
    start, probes, losses = 0, [], []
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("mint_sha") != m.sha:
            raise SystemExit(f"{state_path}: checkpoint was trained on a different mint "
                             f"({state.get('mint_sha')}) than seed {seed}'s now ({m.sha}); delete "
                             f"{ck} and retrain")
        judge.load_state_dict(torch.load(ck / "judge.pt"))
        opt.load_state_dict(torch.load(ck / "opt.pt"))
        gen.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))
        start, probes, losses = state["update"], state["probes"], state["losses"]
    t0 = time.time()
    loss_last = None
    for u in range(start, n_updates):
        xs, y = _block(m, u, batch)
        loss = F.mse_loss(judge(xs, gen), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_last = loss.item()
        if u % probe_every == 0 or u == n_updates - 1:
            pr = _probe(judge, m, seed)
            probes.append(pr)
            losses.append([u, loss_last])
            torch.save(judge.state_dict(), ck / "judge.pt")
            torch.save(opt.state_dict(), ck / "opt.pt")
            _write_json(state_path, {"arm": arm, "update": u + 1, "mint_sha": m.sha,
                                     "generator_state": gen.get_state().tolist(),
                                     "probes": probes, "losses": losses})
            log(f"{arm} s{seed} update {u}: loss {loss_last:.6f} "
                f"spearman_7_11 {pr['spearman_7_11']}")
    return {"updates": n_updates, "resumed_from": start, "wall_s": time.time() - t0,
            "loss_last": loss_last, "probes": probes, "losses": losses,
            "final_probe": probes[-1] if probes else _probe(judge, m, seed)}


def train_oraw(judge, m, *, n_states, epochs, batch, seed, ckpt_dir, log=print):
    """Minutes per seed, so no mid-run checkpoint: an interrupted run retrains from scratch."""
    if len(m.labels) < n_states:
        raise SystemExit(f"mint for seed {seed} holds {len(m.labels)} states; O-raw needs "
                         f"{n_states}")
    ck = Path(ckpt_dir)
    ck.mkdir(parents=True, exist_ok=True)
    x = cells.RawJudge.one_hot(m.states[:n_states])
    y = torch.as_tensor(m.labels[:n_states], dtype=torch.float32)
    opt = cells.optimizer_for("O-raw", judge)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    losses, steps, loss_last = [], 0, None
    for ep in range(epochs):
        perm = torch.randperm(n_states, generator=g)
        for i in range(0, n_states, batch):
            b = perm[i:i + batch]
            loss = F.mse_loss(judge.net(x[b]).squeeze(-1), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            loss_last = loss.item()
        losses.append([ep, loss_last])
        log(f"O-raw s{seed} epoch {ep}: loss {loss_last:.6f}")
    pr = _probe(judge, m, seed)
    torch.save(judge.state_dict(), ck / "judge.pt")
    _write_json(ck / "state.json", {"arm": "O-raw", "epochs": epochs, "mint_sha": m.sha})
    return {"updates": steps, "epochs": epochs, "resumed_from": 0, "wall_s": time.time() - t0,
            "loss_last": loss_last, "probes": [pr], "losses": losses, "final_probe": pr}


def run(arm, seed, out_dir, *, n_updates=cells.N_UPDATES, batch=cells.BATCH,
        probe_every=cells.PROBE_EVERY, raw_states=cells.RAW_STATES, raw_epochs=cells.RAW_EPOCHS,
        raw_batch=cells.RAW_BATCH, mint_root=None, x_root=None, enc_root=None) -> dict:
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mint_root = out_dir if mint_root is None else mint_root
    judge = cells.make_judge(arm, seed, x_root, enc_root)  # a missing input fails first
    m = mint.read_mint(mint_root, seed)
    ck = cells.ckpt_dir(arm, seed, out_dir)
    if arm == "O-raw":
        result = train_oraw(judge, m, n_states=raw_states, epochs=raw_epochs, batch=raw_batch,
                            seed=seed, ckpt_dir=ck)
        settings = {"raw_states": raw_states, "raw_epochs": raw_epochs, "raw_batch": raw_batch,
                    "raw_hidden": cells.RAW_HIDDEN}
    else:
        result = train_x_arm(arm, judge, m, n_updates=n_updates, batch=batch,
                             probe_every=probe_every, seed=seed, ckpt_dir=ck)
        settings = {"n_updates": n_updates, "batch": batch, "probe_every": probe_every,
                    "encoder_drift": cells.drift(judge.sensory.state_dict(),
                                                 cells.reference_sensory(arm, seed, x_root,
                                                                         enc_root))}
    judge.eval()
    rec = {**result, **settings, "arm": arm, "seed": seed, "mint_sha": m.sha,
           "mint_meta": m.meta, "n_actions": N_ACTIONS,
           "gate_l": gate_l.leaf_rank_margin(judge, m.probe, m.leaf_dist.__getitem__, seed,
                                             N_ACTIONS),
           "git_commit": _git_commit()}
    _write_json(out_dir / cells.record_name(arm, seed), rec)
    return rec


def pending_jobs(jobs, out_dir) -> list:
    return [(a, s) for a, s in jobs if not (Path(out_dir) / cells.record_name(a, s)).exists()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=list(cells.ARMS), default=list(cells.TRAIN_ORDER))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    # Required, with no default: a run must state its budget.
    ap.add_argument("--n-updates", type=int, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="outputs/ (outputs_pilot/ with --pilot) when omitted; also the mint root")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cells.check_seeds(args.seeds, args.pilot)
    out = args.out_dir if args.out_dir is not None else (
        cells.PILOT_OUT if args.pilot else cells.OUT)
    x_root, enc_root = cells.x_judge_root(args.pilot), cells.encoder_root(args.pilot)
    jobs = pending_jobs([(a, s) for a in args.arms for s in args.seeds], out)
    print(f"EXP-080 train: {len(jobs)} runs, {args.workers} workers, pilot={args.pilot}, "
          f"n_updates={args.n_updates}, out {out}, X judges {x_root}, encoders {enc_root}",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, a, s, out, n_updates=args.n_updates, x_root=x_root,
                            enc_root=enc_root): (a, s) for a, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            drift = r.get("encoder_drift")
            # None when the session resumed a finished checkpoint and trained nothing.
            loss = "None" if r["loss_last"] is None else format(r["loss_last"], ".6f")
            print(f"  {i}/{len(jobs)}  {r['arm']} seed {r['seed']}  updates {r['updates']}  "
                  f"loss_last {loss}  wall_s {r['wall_s']:.1f}  "
                  f"drift {drift}  gate_l margin {r['gate_l']['margin']:.4f}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q` (tool timeout 600000)
Expected: 25 passed.

- [ ] **Step 5: Mutation check**

Apply each, confirm the named test FAILS, revert:
1. In `cells.make_oxf`, delete the freeze loop: `test_oxt_trains_its_region_and_oxf_does_not` fails (O-Xf drift > 0).
2. In `cells.optimizer_for`, return `vi.make_optimizer(judge, "B")` for O-Xt (head only, region frozen): `test_oxt_trains_its_region_and_oxf_does_not` fails (O-Xt drift 0).
3. In `train_x_arm`, delete the `mint_sha` comparison: `test_resume_refuses_a_checkpoint_from_a_different_mint` fails.
4. In `train_x_arm`, drop the `gen.set_state(...)` line: `test_resume_equals_a_straight_run` fails.
5. In `_block`, use `lo = u * batch + 1`: `test_training_uses_block_u_at_update_u` fails.

- [ ] **Step 6: Commit**

```bash
git add experiments/080_oracle_ordering/train.py tests/experiments/test_exp080.py
git commit -m "EXP-080 train: MSE on minted labels for O-raw, O-Xf and O-Xt, resumable, Gate L from minted leaf distances"
```

---

### Task 4: evaluate.py (Gate 0's X cell, the J4V-O-Xt search cells, the pilot timing cell)

**Files:**
- Create: `experiments/080_oracle_ordering/evaluate.py`
- Modify: `tests/experiments/test_exp080.py` (append)

**Interfaces:**
- Consumes: Task 1's `cells.load_trained("O-Xt", seed, judge_dir)`, `cells.e75` (`load_judge`, `judge_updates`, `JudgeCritic`, `load_cell(depth, seed) -> (agent, head, states, train_seed)`), `cells.E75_OUT`; Task 2's `mint.read_mint`; `gate_l.leaf_rank_margin`; `neuromorphic.training.lookahead.evaluate_lookahead`.
- Produces:
  - `ARM = "J4V-O-Xt"`, `K = 4`, `DEPTH = 11`.
  - `record_name(seed) -> "exp080_J4V-O-Xt_d11_s<seed>.json"`, `time_record_name(judge_seed, cell_seed) -> "exp080_time_J4V-O-Xt_j<judge_seed>_c<cell_seed>.json"`, `gate0_record_name(seed) -> "exp080_gate0_X_s<seed>.json"`.
  - `gate0_cell(seed, mint_root, out_dir, x_root=cells.E75_OUT) -> dict` (fields `kind="gate0-X", seed, gate_l, mint_sha, judge_updates, wall_s, git_commit`).
  - `run_cell(seed, out_dir, judge_dir=None, judge_seed=None, limit_states=None, name=None) -> dict` (fields of `evaluate_lookahead` plus `arm, judge="O-Xt", depth, seed, judge_seed, fired4, wall_s, git_commit, limit_states, judge_updates`).
  - `pending_search(seeds, out_dir) -> list[int]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp080.py`:

```python
ev = _load("exp080_evaluate", EXP / "evaluate.py")


def test_gate0_cell_is_xs_gate_l_through_the_mint(tmp_path):
    """Catches a Gate 0 that measures X some other way than the arms are measured (other probe,
    other leaf distances, other generator), so equality with EXP-075 would not license the X
    side of the claims."""
    _fake_x(tmp_path / "x", 12)
    m = _fake_mint(tmp_path / "m", 12)
    rec = ev.gate0_cell(12, tmp_path / "m", tmp_path / "out", x_root=tmp_path / "x")
    x = cells.e75.load_judge("X", 12, tmp_path / "x")
    want = gate_l.leaf_rank_margin(x, m.probe, m.leaf_dist.__getitem__, 12, N_ACTIONS)
    on_disk = json.loads((tmp_path / "out" / ev.gate0_record_name(12)).read_text())
    assert on_disk["gate_l"] == json.loads(json.dumps(want))
    assert (on_disk["kind"], on_disk["seed"], on_disk["mint_sha"], on_disk["judge_updates"]) == (
        "gate0-X", 12, m.sha, 4000)
    assert rec["gate_l"] == want


def _spy_search(monkeypatch, n_states=7):
    calls = {}

    def fake_cell(depth, seed):
        calls["cell"] = (depth, seed)
        return "agent", "head", [SOLVED] * n_states, 5 + seed

    def fake_eval(agent, head, states, **kw):
        calls["eval"] = dict(kw, n=len(states), agent=agent, head=head)
        return {"success_rate": 0.5, "solved": 1, "n": len(states), "mode": kw["mode"],
                "k": kw["k"], "k_goal": kw["k_goal"], "no_revisit": kw["no_revisit"],
                "first_fire_len": {"4": 2}, "mean_steps": 3.0, "optimality": 0.9}

    monkeypatch.setattr(ev.e75, "load_cell", fake_cell)
    monkeypatch.setattr(ev, "evaluate_lookahead", fake_eval)
    return calls


def _trained_oxt(tmp_path, seed=12, update=4000):
    x = _fake_x(tmp_path / "x", seed)
    ck = cells.ckpt_dir("O-Xt", seed, tmp_path / "j")
    ck.mkdir(parents=True)
    torch.save(x.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": update}))
    return x


def test_search_cell_is_j4v_with_the_oxt_judge(tmp_path, monkeypatch):
    """Catches a search arm that is not EXP-076's J4V (mode C, k 4, goal check at 4,
    no-revisit, depth 11), runs another judge, or seeds its streams differently."""
    x = _trained_oxt(tmp_path)
    calls = _spy_search(monkeypatch)
    rec = ev.run_cell(12, tmp_path / "out", judge_dir=tmp_path / "j")
    kw = calls["eval"]
    assert calls["cell"] == (11, 12)
    assert (kw["depth"], kw["mode"], kw["k"], kw["k_goal"], kw["no_revisit"]) == (
        11, "C", 4, 4, True)
    assert (kw["rng_seed"], kw["imag_seed"], kw["n"]) == (17, 17, 7)
    assert torch.equal(kw["critic"].judge.sensory.fc1.weight, x.sensory.fc1.weight)
    on_disk = json.loads((tmp_path / "out" / ev.record_name(12)).read_text())
    assert (on_disk["arm"], on_disk["judge"], on_disk["depth"], on_disk["seed"],
            on_disk["judge_seed"], on_disk["judge_updates"], on_disk["fired4"],
            on_disk["limit_states"]) == ("J4V-O-Xt", "O-Xt", 11, 12, 12, 4000, 2, None)
    assert rec["success_rate"] == 0.5


def test_timing_cell_runs_a_pilot_judge_on_another_seeds_cell_and_is_marked(tmp_path, monkeypatch):
    """Catches a timing record that could pass for a claim record: it must name both seeds and
    carry limit_states, which the aggregator refuses."""
    _trained_oxt(tmp_path, seed=12)
    calls = _spy_search(monkeypatch, n_states=30)
    rec = ev.run_cell(0, tmp_path / "out", judge_dir=tmp_path / "j", judge_seed=12,
                      limit_states=20, name=ev.time_record_name(12, 0))
    assert calls["cell"] == (11, 0) and calls["eval"]["n"] == 20
    assert (rec["seed"], rec["judge_seed"], rec["limit_states"]) == (0, 12, 20)
    assert (tmp_path / "out" / "exp080_time_J4V-O-Xt_j12_c0.json").exists()
    assert not (tmp_path / "out" / ev.record_name(0)).exists()


def test_search_refuses_an_unfinished_oxt_judge(tmp_path, monkeypatch):
    """Catches a search cell evaluated from a partly trained judge."""
    _trained_oxt(tmp_path, update=250)
    _spy_search(monkeypatch)
    with pytest.raises(SystemExit, match="update 250"):
        ev.run_cell(12, tmp_path / "out", judge_dir=tmp_path / "j")


def test_pending_search_skips_finished_cells(tmp_path):
    """Catches a relaunch that re-runs finished search cells."""
    (tmp_path / ev.record_name(3)).write_text("{}")
    assert ev.pending_search([2, 3, 4], tmp_path) == [2, 4]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q` (tool timeout 600000)
Expected: collection ERROR, `evaluate.py` does not exist.

- [ ] **Step 3: Write evaluate.py**

Create `experiments/080_oracle_ordering/evaluate.py`:

```python
"""EXP-080 evaluation cells.

Gate 0: EXP-075's committed X judge for a seed, re-measured by leaf_rank_margin on that seed's
MINTED probe and leaf distances. Equality with EXP-075's committed gate_l proves both that
today's code reproduces X's side of every claim and that the mint's distances are the BFS's.

Search: EXP-076's J4V (mode C, k 4, goal check at 4, no-revisit) at depth 11 with O-Xt as the
judge, EXP-075's held-out sets, streams and policy cells unchanged. The pilot timing cell runs
seed 12's pilot O-Xt judge on seed 0's cell (seed 12 has no depth-9 policy cell), 20 states,
written under its own name with limit_states set so it can never enter a verdict.

Spec: docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md, sections 3,
5 and 6.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/080_oracle_ordering/evaluate.py --gate0 --seed 0
    .venv/bin/python -u experiments/080_oracle_ordering/evaluate.py --search --workers 6
    .venv/bin/python -u experiments/080_oracle_ordering/evaluate.py --time \
        --judge-dir experiments/080_oracle_ordering/outputs_pilot \
        --out-dir experiments/080_oracle_ordering/outputs_pilot
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


cells = _load("exp080_cells", HERE / "cells.py")
mint = _load("exp080_mint", HERE / "mint.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")
e75 = cells.e75

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.training.lookahead import evaluate_lookahead  # noqa: E402

torch.set_num_threads(1)

ARM = "J4V-O-Xt"
K = 4
DEPTH = 11
TIME_JUDGE_SEED = 12
TIME_CELL_SEED = 0
TIME_STATES = 20


def record_name(seed: int) -> str:
    return f"exp080_{ARM}_d{DEPTH}_s{seed}.json"


def time_record_name(judge_seed: int, cell_seed: int) -> str:
    return f"exp080_time_{ARM}_j{judge_seed}_c{cell_seed}.json"


def gate0_record_name(seed: int) -> str:
    return f"exp080_gate0_X_s{seed}.json"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _write(out_dir, name, rec) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out_dir / name)


def gate0_cell(seed, mint_root, out_dir, x_root=cells.E75_OUT) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    judge = e75.load_judge("X", seed, x_root)
    m = mint.read_mint(mint_root, seed)
    rec = {"kind": "gate0-X", "seed": seed, "mint_sha": m.sha,
           "gate_l": gate_l.leaf_rank_margin(judge, m.probe, m.leaf_dist.__getitem__, seed,
                                             N_ACTIONS),
           "judge_updates": e75.judge_updates("X", seed, x_root),
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit()}
    _write(out_dir, gate0_record_name(seed), rec)
    return rec


def _updates(seed, judge_dir) -> int:
    path = cells.ckpt_dir("O-Xt", seed, judge_dir) / "state.json"
    return int(json.loads(path.read_text())["update"])


def run_cell(seed, out_dir, judge_dir=None, judge_seed=None, limit_states=None, name=None) -> dict:
    """J4V-O-Xt on `seed`'s depth-11 cell. `judge_seed` defaults to `seed`; only the timing cell
    passes another."""
    torch.set_num_threads(1)
    t0 = time.time()
    judge_seed = seed if judge_seed is None else judge_seed
    critic = e75.JudgeCritic(cells.load_trained("O-Xt", judge_seed, judge_dir))
    agent, head, states, train_seed = e75.load_cell(DEPTH, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=DEPTH, mode="C", k=K, k_goal=K,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    rec = {**res, "depth": DEPTH, "seed": seed, "judge_seed": judge_seed, "arm": ARM,
           "judge": "O-Xt", "fired4": res["first_fire_len"].get("4", 0),
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "limit_states": limit_states, "judge_updates": _updates(judge_seed, judge_dir)}
    _write(out_dir, name or record_name(seed), rec)
    return rec


def pending_search(seeds, out_dir) -> list[int]:
    return [s for s in seeds if not (Path(out_dir) / record_name(s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate0", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    ap.add_argument("--mint-root", type=Path, default=None, help="--out-dir when omitted")
    ap.add_argument("--judge-dir", type=Path, default=None, help="--out-dir when omitted")
    args = ap.parse_args()
    if args.gate0 + args.search + args.time != 1:
        raise SystemExit("pass exactly one of --gate0, --search or --time")
    judge_dir = args.judge_dir if args.judge_dir is not None else args.out_dir
    if args.gate0:
        r = gate0_cell(args.seed, args.mint_root or args.out_dir, args.out_dir)
        print(f"gate0 X s{r['seed']} margin {r['gate_l']['margin']!r} wall {r['wall_s']}s")
        return
    if args.time:
        r = run_cell(TIME_CELL_SEED, args.out_dir, judge_dir, judge_seed=TIME_JUDGE_SEED,
                     limit_states=TIME_STATES,
                     name=time_record_name(TIME_JUDGE_SEED, TIME_CELL_SEED))
        print(f"time {ARM} judge s{r['judge_seed']} cell s{r['seed']} {r['n']} states "
              f"solved {r['solved']} wall {r['wall_s']}s")
        return
    cells.check_seeds(args.seeds, False)
    jobs = pending_search(args.seeds, args.out_dir)
    print(f"EXP-080 search: {len(jobs)} cells, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_cell, s, args.out_dir, judge_dir): s for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  {ARM} d{DEPTH} s{r['seed']} solved {r['solved']}/{r['n']} "
                  f"wall {r['wall_s']}s", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q` (tool timeout 600000)
Expected: 30 passed.

- [ ] **Step 5: Mutation check**

Apply each, confirm the named test FAILS, revert:
1. In `run_cell`, pass `k_goal=None`: `test_search_cell_is_j4v_with_the_oxt_judge` fails.
2. In `run_cell`, drop `limit_states` from the record: `test_timing_cell_runs_a_pilot_judge_on_another_seeds_cell_and_is_marked` fails.
3. In `gate0_cell`, use `cells.c75.probe_set` order reversed (`m.probe[::-1]`): `test_gate0_cell_is_xs_gate_l_through_the_mint` fails.

- [ ] **Step 6: Commit**

```bash
git add experiments/080_oracle_ordering/evaluate.py tests/experiments/test_exp080.py
git commit -m "EXP-080 evaluate: Gate 0's X cell through the mint, J4V-O-Xt search cells, the pilot timing cell"
```

---

### Task 5: aggregate.py (gates, claims, search reading, the verdict, the pilot report)

**Files:**
- Create: `experiments/080_oracle_ordering/aggregate.py`
- Modify: `tests/experiments/test_exp080.py` (append)

**Interfaces:**
- Consumes: Tasks 1 to 4 (record names and fields listed there); `experiments/070_lookahead_existing/aggregate.py` (`one_sided_p(diffs)`, `gate1_verdict(a_mean, b_mean) -> "RESOLVED" | "UNRESOLVED"`); `experiments/072_p3v_frontier/aggregate.py` (`pair_diffs(a: dict, b: dict) -> list`); `experiments/074_wide_judge/aggregate.py` (`gate_l_verdict(margins, threshold) -> bool`); `experiments/075_wide_region/aggregate.py` (`seconds_per_update(rec)`); `experiments/079_judge_trained_deep/aggregate.py` (`margin_band(gate_l, lo=9, hi=11)`).
- Produces:
  - constants `CLAIM_ALPHA = 0.025`, `CLAIM_MIN = 0.05`, `GATE_V_ALPHA = 0.05`, `GATE_V_MIN = 0.05`, `SEARCH_ALPHA = 0.05`, `SEARCH_MIN = 0.03`, `GATE_T_THRESHOLD = 0.1356358024691358`, `STOP_BAR = 0.14532407407407405`, `AMENDMENT: dict | None = None`.
  - `check_train_record(rec, name, arm, seed)`, `read_train(out, arm, seed)`, `check_mint_pairing(recs)`, `far(rec)`, `overall(rec)`.
  - `claim_verdict(diffs, gate_ok) -> (verdict, p, mean)`, `gate_v(oraw_far, x_far) -> dict`, `gate_t(margins) -> bool`, `search_reading(diffs, a_mean, b_mean) -> (reading, p, mean)`, `verdict(g0, gv, t, claim1, search) -> str`.
  - `gate0(out) -> "PASS" | "FAIL"`, `stage_gates(out) -> dict`, `format_gate0(v) -> str`, `format_stage_gates(res) -> list[str]`.
  - `pilot_report(pilot_dir) -> dict`.
  - CLI: `--pilot-report`, `--stage gate0`, `--stage gates`, and the full read.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp080.py`:

```python
agg = _load("exp080_aggregate", EXP / "aggregate.py")
E75 = REPO / "experiments" / "075_wide_region"


def _x75(seed, pilot=False):
    d = E75 / ("outputs_pilot" if pilot else "outputs")
    return json.loads((d / f"exp075_train_X_s{seed}.json").read_text())


def _shift(gl, delta):
    """A copy of a gate_l dict with every distance's hit moved by `delta` (margin moves too)."""
    out = json.loads(json.dumps(gl))
    for v in out["by_distance"].values():
        v["hit"] += delta
    out["hit"] += delta
    out["margin"] += delta
    return out


def _train_rec(arm, seed, gl, sha="s", drift=None):
    rec = {"arm": arm, "seed": seed, "gate_l": gl, "mint_sha": f"{sha}{seed}",
           "mint_meta": {"n_blocks": 4000, "block": 1000, "max_len": 40},
           "final_probe": {"spearman_7_11": 0.3, "mean_j_by_distance": {"9": 9.0, "10": 10.0,
                                                                         "11": 11.0}},
           "loss_last": 1.0, "wall_s": 100.0, "resumed_from": 0, "limit_states": None}
    if arm == "O-raw":
        rec.update(raw_states=600000, raw_epochs=12, raw_batch=512, raw_hidden=512, updates=14064)
    else:
        rec.update(n_updates=4000, batch=1000, probe_every=250, updates=4000,
                   encoder_drift=(0.0 if arm == "O-Xf" else 12.0) if drift is None else drift)
    return rec


def _write_train(out, shifts, seeds=range(12), pilot=False):
    out.mkdir(parents=True, exist_ok=True)
    for arm, delta in shifts.items():
        for s in seeds:
            rec = _train_rec(arm, s, _shift(_x75(s, pilot)["gate_l"], delta))
            (out / cells.record_name(arm, s)).write_text(json.dumps(rec))


def test_bars_are_the_committed_x_records():
    """Catches a bar retyped wrong or computed from other records: Gate T is half X's 12-seed
    mean overall margin, the stop bar is X's pilot far-band mean plus 0.05 (spec sections 5, 6)."""
    import statistics as st
    overall = st.mean(_x75(s)["gate_l"]["margin"] for s in range(12))
    assert agg.GATE_T_THRESHOLD == overall / 2
    assert round(agg.GATE_T_THRESHOLD, 5) == 0.13564
    pilot_far = st.mean(agg.margin_band(_x75(s, True)["gate_l"]) for s in (12, 13))
    assert agg.STOP_BAR == pilot_far + 0.05
    assert round(agg.STOP_BAR, 5) == 0.14532
    assert (agg.CLAIM_ALPHA, agg.CLAIM_MIN, agg.GATE_V_ALPHA, agg.GATE_V_MIN, agg.SEARCH_ALPHA,
            agg.SEARCH_MIN) == (0.025, 0.05, 0.05, 0.05, 0.05, 0.03)
    assert agg.AMENDMENT is None


def test_claim_verdict_bands():
    """Catches a claim confirmed at alpha 0.05, a small gain reported as CONFIRMED, or a VOID
    gate that still prints a verdict."""
    big = [0.08] * 12
    small = [0.02] * 12
    mixed = [0.08] * 7 + [-0.08] * 5
    assert agg.claim_verdict(big, True)[0] == "CONFIRMED"
    assert agg.claim_verdict(small, True)[0] == "CONFIRMED (SMALL)"
    assert agg.claim_verdict(mixed, True)[0] == "REFUTED"
    assert agg.claim_verdict(big, False)[0] == "VOID"
    # p 0.0322 with mean 0.10: CONFIRMED at alpha 0.05, REFUTED at the registered 0.025.
    edge = [0.2] * 8 + [-0.1] * 4
    assert agg.one_sided_p(edge) == pytest.approx(0.0322265625)
    assert agg.claim_verdict(edge, True)[0] == "REFUTED"


def test_gate_v_needs_both_p_and_size():
    """Catches Gate V passing on significance alone or on size alone."""
    assert agg.gate_v([0.3] * 12, [0.1] * 12)["pass"]
    assert not agg.gate_v([0.12] * 12, [0.1] * 12)["pass"]  # p tiny, mean 0.02
    assert not agg.gate_v([0.5] * 5 + [0.0] * 7, [0.1] * 12)["pass"]  # mean 0.108, p 0.109


def test_search_reading_and_its_gate1():
    """Catches PAYS without the 0.03 bar, or a floor contrast read as a result."""
    assert agg.search_reading([0.05] * 12, 0.245, 0.195)[0] == "PAYS"
    assert agg.search_reading([0.01] * 12, 0.205, 0.195)[0] == "DOES NOT PAY"
    assert agg.search_reading([0.0] * 12, 0.01, 0.01)[0] == "UNRESOLVED"


def test_verdict_has_a_line_for_every_gate_outcome():
    """Catches an outcome the spec names that the aggregator would print as something else
    (gate-calibration rule: the unresolved band is a verdict of its own)."""
    ok = {"O-raw": True, "O-Xf": True, "O-Xt": True}
    assert agg.verdict(False, True, ok, "CONFIRMED", "PAYS") == "UNRESOLVED"
    assert agg.verdict(True, False, ok, "CONFIRMED", "PAYS") == "UNRESOLVED"
    assert agg.verdict(True, True, {**ok, "O-raw": False}, "CONFIRMED", "PAYS") == "UNRESOLVED"
    assert agg.verdict(True, True, {**ok, "O-Xt": False}, "VOID", None) == (
        "UNRESOLVED (O-Xt UNTRAINED)")
    assert agg.verdict(True, True, ok, "REFUTED", None) == "ENCODER-LIMITED"
    assert agg.verdict(True, True, ok, "CONFIRMED (SMALL)", None) == "ORDERING LEARNABLE BUT SMALL"
    assert agg.verdict(True, True, ok, "CONFIRMED", "PAYS") == "ORDERING LEARNABLE, PAYS"
    assert agg.verdict(True, True, ok, "CONFIRMED", "DOES NOT PAY") == (
        "ORDERING LEARNABLE, DOES NOT PAY AT DEPTH 11")
    assert agg.verdict(True, True, ok, "CONFIRMED", "UNRESOLVED") == (
        "ORDERING LEARNABLE, SEARCH UNRESOLVED")
    with pytest.raises(SystemExit, match="search"):
        agg.verdict(True, True, ok, "CONFIRMED", None)


def test_train_record_refuses_a_moved_oxf_or_a_still_oxt():
    """Catches the freeze failing silently or the gradient never arriving, after the fact."""
    gl = _x75(0)["gate_l"]
    agg.check_train_record(_train_rec("O-Xf", 0, gl), "f", "O-Xf", 0)
    agg.check_train_record(_train_rec("O-Xt", 0, gl), "t", "O-Xt", 0)
    with pytest.raises(SystemExit, match="O-Xf"):
        agg.check_train_record(_train_rec("O-Xf", 0, gl, drift=1e-9), "f", "O-Xf", 0)
    with pytest.raises(SystemExit, match="O-Xt"):
        agg.check_train_record(_train_rec("O-Xt", 0, gl, drift=0.0), "t", "O-Xt", 0)


def test_smoke_and_misnamed_records_are_refused(tmp_path):
    """Catches a record at another budget, a smoke or timing record, or one whose arm or seed is
    not its file name's, entering a gate."""
    gl = _x75(0)["gate_l"]
    for bad, match in ((dict(n_updates=3), "n_updates"), (dict(seed=1), "file name"),
                       (dict(arm="O-Xf"), "file name"), (dict(limit_states=20), "smoke")):
        rec = {**_train_rec("O-Xt", 0, gl), **bad}
        (tmp_path / cells.record_name("O-Xt", 0)).write_text(json.dumps(rec))
        with pytest.raises(SystemExit, match=match):
            agg.read_train(tmp_path, "O-Xt", 0)
    rec = _train_rec("O-raw", 0, gl)
    rec["raw_epochs"] = 2
    (tmp_path / cells.record_name("O-raw", 0)).write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="raw_epochs"):
        agg.read_train(tmp_path, "O-raw", 0)


def test_mint_pairing_refuses_arms_trained_on_different_draws():
    """Catches O-Xf and O-Xt (or O-raw) of one seed trained on different mints, which unpairs
    the comparison spec section 3 pairs."""
    gl = _x75(0)["gate_l"]
    recs = {a: {0: _train_rec(a, 0, gl)} for a in cells.ARMS}
    agg.check_mint_pairing(recs)
    recs["O-Xf"][0]["mint_sha"] = "other"
    with pytest.raises(SystemExit, match="seed 0"):
        agg.check_mint_pairing(recs)


def test_gate0_compares_with_exp075s_committed_record(tmp_path):
    """Catches a Gate 0 that compares anything less than the whole committed gate_l."""
    gl = _x75(0)["gate_l"]
    (tmp_path / ev.gate0_record_name(0)).write_text(json.dumps({"gate_l": gl}))
    assert agg.gate0(tmp_path) == "PASS"
    (tmp_path / ev.gate0_record_name(0)).write_text(json.dumps({"gate_l": _shift(gl, 1e-12)}))
    assert agg.gate0(tmp_path) == "FAIL"


def test_stage_gates_reads_the_registered_contrasts(tmp_path, monkeypatch):
    """Catches claims paired against the wrong arm, Gate V built from the wrong arm, or a chain
    signal that would run the search when Claim 1 is not CONFIRMED."""
    monkeypatch.setattr(agg, "AMENDMENT", {"date": "2026-10-22"})
    (tmp_path / ev.gate0_record_name(0)).write_text(json.dumps({"gate_l": _x75(0)["gate_l"]}))
    _write_train(tmp_path, {"O-raw": 0.30, "O-Xf": 0.01, "O-Xt": 0.08})
    res = agg.stage_gates(tmp_path)
    assert res["g0"] == "PASS" and res["v"]["pass"]
    assert res["v"]["mean"] == pytest.approx(0.30)
    assert res["c1"][0] == "CONFIRMED" and res["c1"][2] == pytest.approx(0.08)
    assert res["c2"][0] == "CONFIRMED (SMALL)" and res["c2"][2] == pytest.approx(0.01)
    lines = "\n".join(agg.format_stage_gates(res))
    assert re.search(CHAIN_SEARCH, lines, re.I)
    _write_train(tmp_path, {"O-Xt": 0.02})
    lines = "\n".join(agg.format_stage_gates(agg.stage_gates(tmp_path)))
    assert not re.search(CHAIN_SEARCH, lines, re.I)
    monkeypatch.setattr(agg, "AMENDMENT", None)
    with pytest.raises(SystemExit, match="AMENDMENT"):
        agg.stage_gates(tmp_path)


def test_stage_gates_voids_the_claims_when_gate_v_fails(tmp_path, monkeypatch):
    """Catches claims read when the protocol could not show a gain."""
    monkeypatch.setattr(agg, "AMENDMENT", {"date": "2026-10-22"})
    (tmp_path / ev.gate0_record_name(0)).write_text(json.dumps({"gate_l": _x75(0)["gate_l"]}))
    _write_train(tmp_path, {"O-raw": 0.01, "O-Xf": 0.2, "O-Xt": 0.2})
    res = agg.stage_gates(tmp_path)
    assert not res["v"]["pass"] and res["c1"][0] == "VOID" and res["c2"][0] == "VOID"


def test_pilot_report_states_the_stop_rule(tmp_path):
    """Catches a stop rule read from the wrong arm or against the wrong bar."""
    _write_train(tmp_path, {"O-raw": 0.06, "O-Xf": 0.0, "O-Xt": 0.0}, seeds=(12, 13), pilot=True)
    rep = agg.pilot_report(tmp_path)
    assert rep["far"]["O-raw"] == pytest.approx(agg.STOP_BAR + 0.01)
    assert rep["stop"] is False
    _write_train(tmp_path, {"O-raw": 0.04}, seeds=(12, 13), pilot=True)
    assert agg.pilot_report(tmp_path)["stop"] is True
    assert rep["x_far"] == pytest.approx(agg.STOP_BAR - 0.05)
    assert rep["seconds_per_update"]["O-Xt"][12] == pytest.approx(100.0 / 4000)
```

Add these constants near the top of the test file, after `cells = ...` (Task 6's chain test reuses them):

```python
# The regexes chain080.ps1 applies to the aggregator's stage output (PowerShell -match is
# case-insensitive). Kept here verbatim so a change to either side fails a test.
CHAIN_GATE0 = r"GATE 0 \(X seed 0 re-measured through the mint\): PASS"
CHAIN_SEARCH = r"SEARCH: RUN"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q` (tool timeout 600000)
Expected: collection ERROR, `aggregate.py` does not exist.

- [ ] **Step 3: Write aggregate.py**

Create `experiments/080_oracle_ordering/aggregate.py`:

```python
"""EXP-080 aggregator. Every gate outcome is a verdict of its own: Gate 0 or Gate V failing, or
O-raw failing Gate T, prints UNRESOLVED; O-Xt failing Gate T prints UNRESOLVED (O-Xt UNTRAINED);
a confirmed gain under 0.05 prints CONFIRMED (SMALL) and its verdict line says so; a search
contrast on the floor or ceiling prints SEARCH UNRESOLVED. The one-line VERDICT is spec section
8's table.

Spec: docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md, sections 5
to 9.

The sign-flip test and Gate 1 are EXP-070's, seed pairing is EXP-072's, Gate T is EXP-074's
Gate L verdict, s/update is EXP-075's, and the far band is EXP-079's margin_band, all reused
through importlib.

Usage:
    .venv/bin/python experiments/080_oracle_ordering/aggregate.py --pilot-report
    .venv/bin/python experiments/080_oracle_ordering/aggregate.py --stage gate0
    .venv/bin/python experiments/080_oracle_ordering/aggregate.py --stage gates
    .venv/bin/python experiments/080_oracle_ordering/aggregate.py
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
a72 = _load("exp072_aggregate", REPO / "experiments" / "072_p3v_frontier" / "aggregate.py")
a74 = _load("exp074_aggregate", REPO / "experiments" / "074_wide_judge" / "aggregate.py")
a75 = _load("exp075_aggregate", REPO / "experiments" / "075_wide_region" / "aggregate.py")
a79 = _load("exp079_aggregate", REPO / "experiments" / "079_judge_trained_deep" / "aggregate.py")
cells = _load("exp080_cells", HERE / "cells.py")
ev = _load("exp080_evaluate", HERE / "evaluate.py")
one_sided_p = a70.one_sided_p
margin_band = a79.margin_band

SEEDS = cells.EVAL_SEEDS
ARMS = cells.ARMS
CLAIM_ALPHA = 0.05 / 2
CLAIM_MIN = 0.05
GATE_V_ALPHA = 0.05
GATE_V_MIN = 0.05
SEARCH_ALPHA = 0.05
SEARCH_MIN = 0.03
# Spec section 5: half X's 12-seed mean overall margin (EXP-075's committed records, 0.27127...).
GATE_T_THRESHOLD = 0.1356358024691358
# Spec section 6: X's mean pilot far-band margin on seeds 12 and 13 (0.09532...) plus 0.05.
STOP_BAR = 0.14532407407407405
MINT_SETTINGS = {"n_blocks": cells.N_UPDATES, "block": cells.BATCH, "max_len": cells.MAX_LEN}
X_SETTINGS = {"n_updates": cells.N_UPDATES, "batch": cells.BATCH,
              "probe_every": cells.PROBE_EVERY}
RAW_SETTINGS = {"raw_states": cells.RAW_STATES, "raw_epochs": cells.RAW_EPOCHS,
                "raw_batch": cells.RAW_BATCH, "raw_hidden": cells.RAW_HIDDEN}

# Set by the dated post-pilot amendment (spec section 6), before any seed 0-11 is minted: the
# worker counts. The launcher refuses every phase but check and the pilot phases while it is None.
AMENDMENT: dict | None = None


def require_amendment() -> None:
    if AMENDMENT is None:
        raise SystemExit("AMENDMENT is unset: the dated post-pilot amendment sets it; no gate or "
                         "claim may be read before then")


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path.name}: the run is incomplete")
    r = json.loads(path.read_text())
    if r.get("limit_states") is not None:
        raise SystemExit(f"{path.name} is a smoke record and may not enter a verdict")
    return r


def _x_train(seed, pilot=False) -> dict:
    d = cells.E75_PILOT if pilot else cells.E75_OUT
    return _read(d / cells.c75.record_name("X", seed))


def check_train_record(rec, name, arm, seed) -> None:
    for k, want in (("arm", arm), ("seed", seed)):
        if rec.get(k) != want:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, the file name says {want!r}")
    meta = rec.get("mint_meta", {})
    for k, v in MINT_SETTINGS.items():
        if meta.get(k) != v:
            raise SystemExit(f"{name}: mint {k} is {meta.get(k)!r}, spec says {v!r}")
    for k, v in (RAW_SETTINGS if arm == "O-raw" else X_SETTINGS).items():
        if rec.get(k) != v:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, spec says {v!r}")
    drift = rec.get("encoder_drift")
    if arm == "O-Xf" and drift != 0.0:
        raise SystemExit(f"{name}: O-Xf's region moved (encoder_drift {drift!r}); the freeze "
                         f"did not hold")
    if arm == "O-Xt" and not (isinstance(drift, float) and drift > 0.0):
        raise SystemExit(f"{name}: O-Xt's region did not move (encoder_drift {drift!r}); the "
                         f"gradient never reached it")


def read_train(out, arm, seed) -> dict:
    name = cells.record_name(arm, seed)
    rec = _read(Path(out) / name)
    check_train_record(rec, name, arm, seed)
    return rec


def check_mint_pairing(recs) -> None:
    """Every arm of a seed trained on the same mint (spec section 3)."""
    for seed in sorted(next(iter(recs.values()))):
        shas = {a: recs[a][seed]["mint_sha"] for a in recs}
        if len(set(shas.values())) != 1:
            raise SystemExit(f"seed {seed}: arms trained on different mints {shas}")


def far(rec) -> float:
    return margin_band(rec["gate_l"])


def overall(rec) -> float:
    return rec["gate_l"]["margin"]


def claim_verdict(diffs, gate_ok) -> tuple[str, float, float]:
    p, m = one_sided_p(diffs), st.mean(diffs)
    if not gate_ok:
        return "VOID", p, m
    if p < CLAIM_ALPHA and m > 0:
        return ("CONFIRMED" if m >= CLAIM_MIN else "CONFIRMED (SMALL)"), p, m
    return "REFUTED", p, m


def gate_v(oraw_far, x_far) -> dict:
    diffs = [a - b for a, b in zip(oraw_far, x_far)]
    p, m = one_sided_p(diffs), st.mean(diffs)
    return {"diffs": diffs, "p": p, "mean": m, "wins": sum(d > 0 for d in diffs),
            "pass": bool(p < GATE_V_ALPHA and m >= GATE_V_MIN)}


def gate_t(margins) -> bool:
    return a74.gate_l_verdict(margins, GATE_T_THRESHOLD)


def search_reading(diffs, a_mean, b_mean) -> tuple[str, float, float]:
    p, m = one_sided_p(diffs), st.mean(diffs)
    if a70.gate1_verdict(a_mean, b_mean) == "UNRESOLVED":
        return "UNRESOLVED", p, m
    return ("PAYS" if p < SEARCH_ALPHA and m >= SEARCH_MIN else "DOES NOT PAY"), p, m


def verdict(g0: bool, gv: bool, t: dict, claim1: str, search) -> str:
    """Spec section 8, one line."""
    if not (g0 and gv and t["O-raw"]):
        return "UNRESOLVED"
    if not t["O-Xt"]:
        return "UNRESOLVED (O-Xt UNTRAINED)"
    if claim1 == "REFUTED":
        return "ENCODER-LIMITED"
    if claim1 == "CONFIRMED (SMALL)":
        return "ORDERING LEARNABLE BUT SMALL"
    if claim1 != "CONFIRMED":
        raise SystemExit(f"Claim 1 is {claim1!r} with every gate passed; no verdict row fits")
    if search is None:
        raise SystemExit("Claim 1 is CONFIRMED: run the search cells before reading the verdict")
    return {"PAYS": "ORDERING LEARNABLE, PAYS",
            "DOES NOT PAY": "ORDERING LEARNABLE, DOES NOT PAY AT DEPTH 11",
            "UNRESOLVED": "ORDERING LEARNABLE, SEARCH UNRESOLVED"}[search]


def gate0(out) -> str:
    p = Path(out) / ev.gate0_record_name(0)
    if not p.exists():
        raise SystemExit(f"missing {p.name}: run the gate0 phase first")
    got = json.loads(p.read_text())["gate_l"]
    return "PASS" if got == _x_train(0)["gate_l"] else "FAIL"


def format_gate0(v: str) -> str:
    return f"GATE 0 (X seed 0 re-measured through the mint): {v}"


def stage_gates(out) -> dict:
    require_amendment()
    out = Path(out)
    g0 = gate0(out)
    recs = {a: {s: read_train(out, a, s) for s in SEEDS} for a in ARMS}
    check_mint_pairing(recs)
    x_far = [far(_x_train(s)) for s in SEEDS]
    v = gate_v([far(recs["O-raw"][s]) for s in SEEDS], x_far)
    t = {a: gate_t([overall(recs[a][s]) for s in SEEDS]) for a in ARMS}
    t_mean = {a: st.mean(overall(recs[a][s]) for s in SEEDS) for a in ARMS}
    base = g0 == "PASS" and v["pass"]

    def claim(arm):
        diffs = [far(recs[arm][s]) - x for s, x in zip(SEEDS, x_far)]
        return claim_verdict(diffs, base and t[arm]) + (sum(d > 0 for d in diffs),)

    c1, c2 = claim("O-Xt"), claim("O-Xf")
    return {"g0": g0, "v": v, "t": t, "t_mean": t_mean, "c1": c1, "c2": c2,
            "run_search": c1[0] == "CONFIRMED", "recs": recs, "x_far": x_far}


def format_stage_gates(res) -> list[str]:
    v = res["v"]
    lines = [format_gate0(res["g0"]),
             f"GATE V (protocol resolves): {'PASS' if v['pass'] else 'FAIL'}, O-raw - X far-band "
             f"mean {v['mean']:+.4f}, ahead on {v['wins']} of {len(v['diffs'])}, p {v['p']:.4f}"]
    for a in ARMS:
        lines.append(f"GATE T (trained) {a}: {'PASS' if res['t'][a] else 'FAIL'}, mean overall "
                     f"margin {res['t_mean'][a]:.4f} vs {GATE_T_THRESHOLD:.5f}")
    for key, label in (("c1", "CLAIM 1 O-Xt - X far-band"), ("c2", "CLAIM 2 O-Xf - X far-band")):
        verdict_, p, m, wins = res[key]
        lines.append(f"{label}: diff {m:+.4f}, ahead on {wins} of 12, p {p:.4f} -> {verdict_}")
    lines.append("SEARCH: RUN (Claim 1 CONFIRMED)" if res["run_search"]
                 else f"SEARCH: NOT RUN (Claim 1 {res['c1'][0]}; the search is not read)")
    return lines


def _read_search(out, seed) -> dict:
    name = ev.record_name(seed)
    r = _read(Path(out) / name)
    for k, want in (("arm", ev.ARM), ("judge", "O-Xt"), ("depth", ev.DEPTH), ("seed", seed),
                    ("judge_seed", seed), ("k", ev.K), ("k_goal", ev.K), ("mode", "C"),
                    ("no_revisit", True), ("judge_updates", cells.N_UPDATES)):
        if r.get(k) != want:
            raise SystemExit(f"{name}: {k} is {r.get(k)!r}, spec says {want!r}")
    return r


def pilot_report(pilot_dir=cells.PILOT_OUT) -> dict:
    pilot_dir = Path(pilot_dir)
    seeds = cells.PILOT_SEEDS
    recs = {a: {s: read_train(pilot_dir, a, s) for s in seeds} for a in ARMS}
    x = {s: _x_train(s, pilot=True) for s in seeds}
    t = pilot_dir / ev.time_record_name(ev.TIME_JUDGE_SEED, ev.TIME_CELL_SEED)
    timing = json.loads(t.read_text()) if t.exists() else None
    fars = {a: st.mean(far(recs[a][s]) for s in seeds) for a in ARMS}
    return {
        "far": fars, "overall": {a: st.mean(overall(recs[a][s]) for s in seeds) for a in ARMS},
        "x_far": st.mean(far(x[s]) for s in seeds),
        "x_overall": st.mean(overall(x[s]) for s in seeds),
        "far_by_seed": {a: {s: far(recs[a][s]) for s in seeds} for a in ARMS},
        "stop": fars["O-raw"] < STOP_BAR, "stop_bar": STOP_BAR,
        "seconds_per_update": {a: {s: a75.seconds_per_update(recs[a][s]) for s in seeds}
                               for a in ("O-Xf", "O-Xt")},
        "oraw_wall_s": {s: recs["O-raw"][s]["wall_s"] for s in seeds},
        "encoder_drift": {a: {s: recs[a][s]["encoder_drift"] for s in seeds}
                          for a in ("O-Xf", "O-Xt")},
        "timing": None if timing is None else {"n": timing["n"], "wall_s": timing["wall_s"],
                                               "solved": timing["solved"]},
    }


def _mean(recs, field):
    return st.mean(r[field] for r in recs.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    ap.add_argument("--pilot-dir", type=Path, default=cells.PILOT_OUT)
    ap.add_argument("--pilot-report", action="store_true")
    ap.add_argument("--stage", choices=("gate0", "gates"))
    args = ap.parse_args()
    if args.pilot_report:
        rep = pilot_report(args.pilot_dir)
        for k, v in rep.items():
            print(f"{k}: {v}")
        if rep["stop"]:
            print(f"STOP: O-raw's mean pilot far-band margin {rep['far']['O-raw']:.4f} is below "
                  f"{STOP_BAR:.5f}. Report; there is no amendment (spec section 6).")
        return
    require_amendment()
    out = args.out_dir
    if args.stage == "gate0":
        print(format_gate0(gate0(out)))
        return
    res = stage_gates(out)
    for line in format_stage_gates(res):
        print(line)
    if args.stage == "gates":
        return
    search = None
    srecs = x76 = None
    if res["run_search"]:
        srecs = {s: _read_search(out, s) for s in SEEDS}
        x76 = {s: _read(cells.E76_OUT / f"exp076_J4V-X_d11_s{s}.json") for s in SEEDS}
        a = {s: r["success_rate"] for s, r in srecs.items()}
        b = {s: r["success_rate"] for s, r in x76.items()}
        diffs = a72.pair_diffs(a, b)
        search, p, m = search_reading(diffs, st.mean(a.values()), st.mean(b.values()))
        print(f"SEARCH d11 J4V-O-Xt - J4V-X: {st.mean(a.values()):.4f} vs "
              f"{st.mean(b.values()):.4f}, diff {m:+.4f}, ahead on {sum(d > 0 for d in diffs)} "
              f"of 12, p {p:.4f} -> {search}")
    print("VERDICT: " + verdict(res["g0"] == "PASS", res["v"]["pass"], res["t"], res["c1"][0],
                                search))

    print("\nReported (spec section 9, never confirmations):")
    s79 = {s: json.loads((cells.E79_OUT / f"exp079_train_S_s{s}.json").read_text())
           for s in SEEDS}
    d = [far(res["recs"]["O-Xt"][s]) - far(s79[s]) for s in SEEDS]
    print(f"  O-Xt - S far-band: {st.mean(d):+.4f}, ahead on {sum(x > 0 for x in d)} of 12, "
          f"p {one_sided_p(d):.4f}")
    x = {s: _x_train(s) for s in SEEDS}
    for dist in ("9", "10", "11"):
        row = {a: st.mean(res["recs"][a][s]["gate_l"]["by_distance"][dist]["hit"]
                          - res["recs"][a][s]["gate_l"]["by_distance"][dist]["chance"]
                          for s in SEEDS) for a in ARMS}
        xm = st.mean(x[s]["gate_l"]["by_distance"][dist]["hit"]
                     - x[s]["gate_l"]["by_distance"][dist]["chance"] for s in SEEDS)
        print(f"  margin at distance {dist}: " + "  ".join(f"{a} {v:.4f}" for a, v in row.items())
              + f"  X {xm:.4f}")
    for a in ARMS:
        sp = _mean({s: res["recs"][a][s]["final_probe"] for s in SEEDS}, "spearman_7_11")
        mj = "  ".join(
            f"J@{dd} {st.mean(res['recs'][a][s]['final_probe']['mean_j_by_distance'][dd] for s in SEEDS):.3f}"
            for dd in ("9", "10", "11"))
        print(f"  {a}: spearman_7_11 {sp:.4f}  {mj}  loss_last "
              f"{_mean(res['recs'][a], 'loss_last'):.4f}")
    if srecs is not None:
        s79e = {s: json.loads((cells.E79_OUT / f"exp079_J4V-S_d11_s{s}.json").read_text())
                for s in SEEDS}
        for label, r in (("J4V-O-Xt", srecs), ("J4V-X", x76), ("J4V-S", s79e)):
            print(f"  d11 {label}: success {_mean(r, 'success_rate'):.4f}  mean steps "
                  f"{_mean(r, 'mean_steps'):.2f}  optimality {_mean(r, 'optimality'):.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q` (tool timeout 600000)
Expected: 42 passed.

- [ ] **Step 5: Mutation check**

Apply each, confirm the named test FAILS, revert:
1. `CLAIM_ALPHA = 0.05`: `test_bars_are_the_committed_x_records` fails.
2. In `gate_v`, drop `and m >= GATE_V_MIN`: `test_gate_v_needs_both_p_and_size` fails.
3. In `verdict`, delete the `not t["O-Xt"]` row: `test_verdict_has_a_line_for_every_gate_outcome` fails.
4. In `check_train_record`, delete the O-Xf drift check: `test_train_record_refuses_a_moved_oxf_or_a_still_oxt` fails.
5. In `stage_gates`, set `run_search` to `c1[0].startswith("CONFIRMED")`: `test_stage_gates_reads_the_registered_contrasts` fails (the SMALL case would run the search).
6. In `stage_gates`, build Gate V from `recs["O-Xt"]`: `test_stage_gates_reads_the_registered_contrasts` fails (mean 0.08, not 0.30).

- [ ] **Step 6: Commit**

```bash
git add experiments/080_oracle_ordering/aggregate.py tests/experiments/test_exp080.py
git commit -m "EXP-080 aggregate: Gates 0, V, T and 1, Claims 1 and 2, the search reading, the one-line verdict, the pilot report"
```

---

### Task 6: launch080.ps1 and chain080.ps1, test count

**Files:**
- Create: `experiments/080_oracle_ordering/launch080.ps1`
- Create: `experiments/080_oracle_ordering/chain080.ps1`
- Modify: `tests/experiments/test_exp080.py` (append)
- Modify: `CLAUDE.md` (test count line)

**Interfaces:**
- Consumes: CLIs of `mint.py` (`--pilot`, `--seeds`, `--workers`, `--out-dir`), `train.py` (`--pilot`, `--arms`, `--seeds`, `--n-updates`, `--workers`, `--out-dir`), `evaluate.py` (`--gate0 --seed 0`, `--search --workers N`, `--time --judge-dir D --out-dir D`), `aggregate.py` (`AMENDMENT` with keys `mint_workers`, `train_workers`, `eval_workers`; `--stage gate0`, `--stage gates`); Task 5's printed lines matched by `CHAIN_GATE0` and `CHAIN_SEARCH`.
- Produces: launcher phases `check, pilot-mint, pilot-train, pilot-time, mint, gate0, train, search`; chain order `mint, gate0, gate0check, train, gates, search`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp080.py`:

```python
def test_launcher_phases_guard_and_memory_sampling():
    """Catches a launcher that can run a full-run phase before the amendment, samples memory
    from the venv redirector instead of every python process (EXP-079's 2 MB reading), lifts
    the worker cap, or does not check the laptop-only pilot inputs."""
    text = (EXP / "launch080.ps1").read_text()
    assert ('ValidateSet("check","pilot-mint","pilot-train","pilot-time","mint","gate0","train",'
            '"search")') in text
    assert "a.AMENDMENT" in text and '-match "None"' in text
    assert "-gt 6" in text and "$null = $proc.Handle" in text
    assert 'Where-Object { $_.ProcessName -match "^python" }' in text
    assert "PrivateMemorySize64" in text and "max peak commit MB per python process" in text
    assert "max total commit MB across python processes" in text
    assert "judge_X_s" in text and "enc_X_s" in text and "exp079_train_S_s" in text
    assert '"--n-updates", $Updates' in text and "$Updates = 4000" in text
    assert '"--time"' in text and '"--gate0"' in text and '"--search"' in text
    assert "--pilot" in text


def test_chain_order_and_gate_regexes_match_the_aggregator():
    """Catches a chain that trains before Gate 0, runs the search when Claim 1 is not CONFIRMED,
    or whose regex no longer matches what the aggregator prints."""
    text = (EXP / "chain080.ps1").read_text()
    assert '@("mint","gate0","gate0check","train","gates","search")' in text
    assert CHAIN_GATE0 in text and CHAIN_SEARCH in text
    assert "--stage gate0" in text and "--stage gates" in text
    for k in ("mint_workers", "train_workers", "eval_workers"):
        assert k in text
    assert re.search(CHAIN_GATE0, agg.format_gate0("PASS"), re.I)
    assert not re.search(CHAIN_GATE0, agg.format_gate0("FAIL"), re.I)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q -k "launcher or chain"` (tool timeout 600000)
Expected: 2 failed (`FileNotFoundError` for the `.ps1` files).

- [ ] **Step 3: Write launch080.ps1**

Create `experiments/080_oracle_ordering/launch080.ps1`:

```powershell
# EXP-080 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File <repo>\experiments\080_oracle_ordering\launch080.ps1 -Phase check
#   ... -Phase pilot-mint  -Workers 2
#   ... -Phase pilot-train -Workers 3
#   ... -Phase pilot-time
#   ... -Phase mint   -Workers N   (N from the amendment)
#   ... -Phase gate0
#   ... -Phase train  -Workers N
#   ... -Phase search -Workers N
#
# Order (spec section 11): pilot-mint -> pilot-train -> pilot-time -> stop rule and dated
# amendment (sets aggregate.AMENDMENT) -> chain080.ps1: mint -> gate0 -> Gate 0 -> train ->
# Gates V and T, Claims 1 and 2 -> search (only when Claim 1 is CONFIRMED). Every phase but check
# and the pilot phases refuses to run while AMENDMENT is None. Every driver skips finished work,
# and training resumes its checkpoints, so a phase is re-runnable after the 03:31 restart.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Seed and arm lists therefore live in this file.
# Launch long phases with OS-level redirection (playbook section 3), never Start-Process from an
# ssh session.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","pilot-mint","pilot-train","pilot-time","mint","gate0","train","search")][string]$Phase,
    [int]$Workers = 0
)

$Updates = 4000

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\080_oracle_ordering"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# EXP-075's 12 X encoders and 12 X judges at 4000 updates, EXP-076's 12 depth-11 J4V-X records,
# EXP-075's 2 X pilot records and 2 X pilot judges at 4000 (laptop only, untracked), EXP-079's 2
# proved pilot encoders (laptop only) and its 12 S training records.
$present = & $py -c "import json, sys; from pathlib import Path; o=Path('experiments/075_wide_region/outputs'); p=Path('experiments/075_wide_region/outputs_pilot'); r6=Path('experiments/076_deeper_tree/outputs'); q=Path('experiments/079_judge_trained_deep'); done=lambda d, s: (d/('judge_X_s'+str(s))/'judge.pt').exists() and json.loads((d/('judge_X_s'+str(s))/'state.json').read_text())['update']==4000; e=sum((o/'encoders'/('enc_X_s'+str(s)+'.pt')).exists() for s in range(12)); j=sum(done(o, s) for s in range(12)); r=sum((r6/('exp076_J4V-X_d11_s'+str(s)+'.json')).exists() for s in range(12)); t=sum((p/('exp075_train_X_s'+str(s)+'.json')).exists() for s in (12, 13)); pj=sum(done(p, s) for s in (12, 13)); pe=sum((q/'outputs_pilot'/'encoders'/('enc_X_s'+str(s)+'.pt')).exists() for s in (12, 13)); sr=sum((q/'outputs'/('exp079_train_S_s'+str(s)+'.json')).exists() for s in range(12)); sys.stdout.write(' '.join(str(x) for x in (e, j, r, t, pj, pe, sr)))"
if ($LASTEXITCODE -ne 0 -or $present -ne "12 12 12 2 2 2 12") { Write-Error "expected '12 12 12 2 2 2 12' (X enc, X judges, J4V-X d11, X pilot records, X pilot judges, pilot encoders, S records), found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                     = $where"
"checkout                    = $head"
"inputs                      = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs                     = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

$isPilot = $Phase -like "pilot-*"
if (-not $isPilot) {
    $amend = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','$($exp.Replace('\','/'))/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.AMENDMENT))"
    if ($LASTEXITCODE -ne 0 -or $amend -eq "") { Write-Error "could not read AMENDMENT (got '$amend'); refusing to run."; exit 1 }
    if ($amend -match "None") { Write-Error "AMENDMENT is unset ($amend): commit the dated post-pilot amendment before this phase."; exit 1 }
    "amendment                   = $amend"
}

if ($Workers -eq 0) { $Workers = 2 }
if ($Workers -gt 6) { Write-Error "Workers capped at 6; got $Workers."; exit 1 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$outDir   = Join-Path $repo "$exp\outputs"
$pilotDir = Join-Path $repo "$exp\outputs_pilot"
$evalSeeds = @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")

switch ($Phase) {
    "pilot-mint"  { $driver = "$exp\mint.py"; $cliArgs = @("--pilot", "--seeds", "12", "13", "--out-dir", $pilotDir, "--workers", $Workers) }
    "pilot-train" { $driver = "$exp\train.py"; $cliArgs = @("--pilot", "--seeds", "12", "13", "--n-updates", $Updates, "--out-dir", $pilotDir, "--workers", $Workers) }
    "pilot-time"  { $driver = "$exp\evaluate.py"; $cliArgs = @("--time", "--judge-dir", $pilotDir, "--out-dir", $pilotDir) }
    "mint"        { $driver = "$exp\mint.py"; $cliArgs = @("--seeds") + $evalSeeds + @("--out-dir", $outDir, "--workers", $Workers) }
    "gate0"       { $driver = "$exp\evaluate.py"; $cliArgs = @("--gate0", "--seed", "0", "--out-dir", $outDir) }
    "train"       { $driver = "$exp\train.py"; $cliArgs = @("--seeds") + $evalSeeds + @("--n-updates", $Updates, "--out-dir", $outDir, "--workers", $Workers) }
    "search"      { $driver = "$exp\evaluate.py"; $cliArgs = @("--search", "--seeds") + $evalSeeds + @("--out-dir", $outDir, "--workers", $Workers) }
}

# Every phase is one python call that runs its own pool. The venv's python.exe is a ~2 MB
# redirector that starts python3.13.exe, so sampling the launched process reads the redirector
# (EXP-079's 2 MB disclosure). Sample every python process instead, in every phase.
"script                      = $driver $cliArgs"
"log                         = $log"
""
$errLog = $log + ".err"
$argList = @("-u", $driver) + @($cliArgs | ForEach-Object { "$_" })
$proc = Start-Process -FilePath $py -ArgumentList $argList -PassThru -NoNewWindow -WorkingDirectory $repo -RedirectStandardOutput $log -RedirectStandardError $errLog
# Read .Handle at once: without it Windows PowerShell 5.1 reports ExitCode as $null after exit.
$null = $proc.Handle
$maxEach = 0
$maxTotal = 0
while (-not $proc.HasExited) {
    $ms = @(Get-Process | Where-Object { $_.ProcessName -match "^python" } | ForEach-Object { [int]($_.PrivateMemorySize64 / 1MB) })
    if ($ms.Count -gt 0) {
        $each = ($ms | Measure-Object -Maximum).Maximum
        $tot = ($ms | Measure-Object -Sum).Sum
        if ($each -gt $maxEach) { $maxEach = $each }
        if ($tot -gt $maxTotal) { $maxTotal = $tot }
    }
    Start-Sleep -Seconds 10
}
Get-Content $log
if (Test-Path $errLog) { Get-Content $errLog }
"max peak commit MB per python process = $maxEach"
"max total commit MB across python processes = $maxTotal"
if ($proc.ExitCode -ne 0) { Write-Error "driver exited $($proc.ExitCode); do not aggregate."; exit 1 }
"PHASE OK"
exit 0
```

- [ ] **Step 4: Write chain080.ps1**

Create `experiments/080_oracle_ordering/chain080.ps1`:

```powershell
# EXP-080 full-run chain, after the dated amendment: mint -> gate0 -> Gate 0 -> train -> Gates V
# and T, Claims 1 and 2 -> search (only when the aggregator prints SEARCH: RUN, i.e. Claim 1 is
# CONFIRMED). Re-runnable with -From: every phase skips finished work and training resumes.
# Launch it with OS-level redirection (playbook section 3):
#   ssh -n laptop 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\experiments\080_oracle_ordering\chain080.ps1 > C:\Users\mlgbr\chain080.out 2>&1"'
param([string]$From = "mint")
$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py = Join-Path $repo ".venv\Scripts\python.exe"
$l = Join-Path $repo "experiments\080_oracle_ordering\launch080.ps1"
$agg = "experiments\080_oracle_ordering\aggregate.py"
$log = "C:\Users\mlgbr\chain080.log"
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo "src"
function Say($m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m; Add-Content -Path $log -Value $line; $line }
Say "chain start from $From at $(git rev-parse --short HEAD)"
$w = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','experiments/080_oracle_ordering/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); m=a.AMENDMENT; sys.stdout.write(str(m['mint_workers'])+' '+str(m['train_workers'])+' '+str(m['eval_workers']))"
if ($LASTEXITCODE -ne 0 -or $w -notmatch "^\d+ \d+ \d+$") { Say "STOP: could not read mint_workers, train_workers and eval_workers from AMENDMENT (got '$w')"; exit 1 }
$mintW, $trainW, $evalW = $w.Split(" ")
foreach ($wk in @($mintW, $trainW, $evalW)) {
    if ([int]$wk -lt 1 -or [int]$wk -gt 6) { Say "STOP: AMENDMENT workers must each be an integer from 1 to 6 (mint $mintW, train $trainW, eval $evalW)"; exit 1 }
}
Say "workers: mint $mintW, train $trainW, eval $evalW"
$order = @("mint","gate0","gate0check","train","gates","search")
$go = $false
foreach ($p in $order) {
    if ($p -eq $From) { $go = $true }
    if (-not $go) { continue }
    if ($p -eq "gate0check") {
        $g = & $py $agg --stage gate0 2>&1 | Out-String
        Say ("gate0: " + $g.Trim())
        if ($g -notmatch "GATE 0 \(X seed 0 re-measured through the mint\): PASS") { Say "STOP: Gate 0 did not pass, the verdict is UNRESOLVED"; exit 2 }
        continue
    }
    if ($p -eq "gates") {
        $g = & $py $agg --stage gates 2>&1 | Out-String
        Say ("gates: " + $g.Trim())
        if ($g -notmatch "SEARCH: RUN") { Say "chain done: the search is not read (see the gates line); aggregate and report"; exit 0 }
        continue
    }
    $n = $evalW
    if ($p -eq "mint") { $n = $mintW }
    if ($p -eq "train") { $n = $trainW }
    if ($p -eq "gate0") { $n = 1 }
    Say "phase $p workers $n"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $n
    Say "phase $p exit $LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Say "STOP: phase $p failed"; exit 1 }
}
Say "chain done"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/experiments/test_exp080.py -q` (tool timeout 600000)
Expected: 44 passed.

- [ ] **Step 6: Mutation check**

1. In `chain080.ps1`, change the order to `@("mint","train","gate0","gate0check","gates","search")`: `test_chain_order_and_gate_regexes_match_the_aggregator` fails.
2. In `aggregate.format_gate0`, change the text to `GATE 0 (X re-measured): ...`: the same test fails.
3. In `launch080.ps1`, replace the `Where-Object { $_.ProcessName -match "^python" }` sampler with `$proc.PrivateMemorySize64`: `test_launcher_phases_guard_and_memory_sampling` fails.
Revert each.

- [ ] **Step 7: Update the test count**

Run: `PYTHONPATH=src /root/projects/neuromorphic-nn-snn-research-project/.venv/bin/python -m pytest tests/ --collect-only -q | tail -1` and `... --collect-only -q -m "not slow" | tail -1` (tool timeout 600000).
Expected: 1216 total, 1189 not slow, 27 slow (1172 + 44). If the numbers differ, use the measured ones.

In `CLAUDE.md`, replace the line `**1172 as of 2026-10-19 (1145 not slow, 27 slow; EXP-079 amendment)**, verified by `--collect-only`.` with two lines:

```
1172 2026-10-19 (EXP-079 amendment),
**<total> as of 2026-10-21 (<not slow> not slow, <slow> slow; EXP-080 code)**, verified by `--collect-only`.
```

filling in the measured numbers.

- [ ] **Step 8: Commit**

```bash
git add experiments/080_oracle_ordering/launch080.ps1 experiments/080_oracle_ordering/chain080.ps1 tests/experiments/test_exp080.py CLAUDE.md
git commit -m "EXP-080 laptop launcher and chain: amendment guard, every-python memory sampling, search only when Claim 1 confirms; test count"
```

---

## After the build (controller, not a subagent task)

1. **Whole-branch verification, in chunks (never one call):** `tests/experiments/test_exp080.py` plus `tests/experiments/test_exp079.py` and `test_exp075.py` (the modules EXP-080 imports), each with a 600000 ms timeout. Then merge `--no-ff` as `Merge exp080-oracle-ordering: ...` and push.
2. **Laptop checkout.** The laptop's main checkout is at `7d39ea2` and holds EXP-079's records and judges as untracked files, which `18cd4f5` now tracks, so `git pull` will refuse to overwrite them. Move exactly the files `git pull` names into `C:\Users\mlgbr\exp079_untracked_backup\` (keeping paths), pull, then compare each backup file with the tracked one (`Get-FileHash`); all must match. Report any that do not before going on.
3. **Pilot:** `launch080.ps1 -Phase check` (expects `12 12 12 2 2 2 12`), then in one background chain with the standard redirection form: `pilot-mint -Workers 2`, `pilot-train -Workers 3`, `pilot-time`. Start a background watcher on the laptop log and name its task id before ending the turn.
4. **Pilot report:** `aggregate.py --pilot-report --pilot-dir experiments/080_oracle_ordering/outputs_pilot` on the laptop. Read: O-raw's far mean against the stop bar 0.14532; each arm's far and overall margin against X's pilot (far 0.0953); O-Xf drift exactly 0.0 and O-Xt drift above 0; seconds per update; the timing cell's wall time for 20 states (times 10 for one 200-state cell); `max peak commit MB per python process` above 500.
5. **Stop rule or amendment.** If `STOP`, report and stop. Otherwise write spec section 13 (dated amendment: mint, train and eval workers with total committed memory under 24 GB, schedule) and set `aggregate.AMENDMENT = {"date": ..., "mint_workers": ..., "train_workers": ..., "eval_workers": ..., "spec_section": 13}` with a test asserting those values; commit with the pilot records (`git add -f`), push, and `pm ask` on neuromorphic for the go.
