# EXP-082 X Long Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue EXP-075's X value-iteration judge from update 4000 to 16,000 on all 12 seeds, then search with it at depths 9 and 11 against EXP-076's J4V-X, and print spec section 8's verdict.

**Architecture:** A new `experiments/082_x_long_budget/` folder mirroring EXP-081's layout: `cells.py` (constants and names), `gatec.py` (continuity gate and checkpoint copy), `train.py` (two legs through `vi.train_judge`, EXP-075's settings), `evaluate.py` (EXP-081's J4V cell with the X16k judge), `aggregate.py` (gates, claims, verdict, reported block), and two PowerShell drivers for the laptop. EXP-075, EXP-076, EXP-080 and EXP-081 modules are imported unchanged; EXP-075's checkpoint folders are only ever read.

**Tech Stack:** Python 3.13 on the laptop (3.10 on the VPS), torch, pytest, PowerShell 5.1 on Windows.

**Spec:** `docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md`

## Global Constraints

- No em-dashes in any file, docstring, comment or commit message.
- Commit messages are plain: no `Co-Authored-By` trailer, no "Generated with" line. Stage explicit paths; never `git add -A`.
- Always run python via the venv with `PYTHONPATH=src`: `PYTHONPATH=src .venv/bin/python -m pytest ...`.
- Always pass an explicit Bash tool timeout (the default is 120 s; the hard ceiling is 600 s). Call test commands directly and let them block; buffered output means silence mid-run is normal.
- Never write an assertion that cannot fail; never weaken a threshold to make a test pass.
- Distance-to-solved is the probe's yardstick and the search's success test only, never a training target or a model input.
- Settings are EXP-075's X, unchanged: batch 1000, sync_every 100, probe_every 250, draws 1, max walk length 14, VI arm "W", readout "wide", hidden 512.
- START 4000, LEGS (8000, 16000), search depths (9, 11), J4V: mode C, k 4, k_goal 4, no-revisit.
- Claim 1: one-sided exact sign-flip, p < 0.025 and mean >= 0.03 (smaller positive mean with p < 0.025 is CONFIRMED (SMALL)). Claim 2: EXP-080's `claim_verdict` (p < 0.025, mean >= 0.05). Gate T: EXP-080's `gate_t` (0.13564).
- Workers: Gate C 4, train 4, search 12.
- EXP-075's `outputs/judge_X_s<seed>/` is read, never written.

## Review Focus

1. **A 03:31 restart mid-leg-2.** The chain re-runs from the top: leg 1's record must survive, leg 2 must resume from its checkpoint, and a checkpoint already past 8000 must never be recorded as the 8000 point. Pinned in Task 3 (`test_a_rerun_skips_finished_legs_and_resumes_leg_2`, `test_a_checkpoint_past_leg_1_is_never_recorded_as_leg_1`).
2. **Gate C re-run after training started.** A re-run must never copy EXP-075's 4000 checkpoint over a continuation in progress, which would silently restart training. Pinned in Task 2 (`test_gate_c_never_overwrites_a_started_continuation`).
3. **A search started on an unfinished judge** (someone runs the search phase by hand after a crash). It must refuse any judge not at exactly 16,000. Pinned in Task 4 (`test_search_cell_refuses_a_judge_short_of_16000`).
4. **A smoke record, a record from another checkout, or a fresh run in the verdict.** The aggregator must refuse each. Pinned in Task 5 (`test_a_record_that_is_not_the_continuation_is_refused`, `test_mixed_checkouts_are_refused`, `test_a_search_record_that_is_not_the_registered_cell_is_refused`).
5. **The chain and the aggregator drifting apart.** The chain must stop on a failed gates stage and must find the search line the aggregator prints. Pinned in Task 6 (`test_chain_reads_the_search_lines_the_aggregator_prints`, `test_drivers_use_the_spec_worker_counts`).

---

### Task 1: cells.py, constants and names

**Files:**
- Create: `experiments/082_x_long_budget/cells.py`
- Create: `tests/experiments/test_exp082.py`

**Interfaces:**
- Consumes: `experiments/075_wide_region/cells.py` (`ckpt_dir(arm, seed, out_dir)`, `record_name(arm, seed)`, `N_UPDATES`, `EVAL_SEEDS`, `check_seeds`, `HIDDEN`).
- Produces: module `cells` with `c75`, `REPO`, `E75`, `E75_OUT`, `E76_OUT`, `E81_OUT`, `OUT`, `ARM = "X16k"`, `SRC_ARM = "X"`, `START = 4000`, `LEGS = (8000, 16000)`, `BATCH = 1000`, `SYNC_EVERY = 100`, `PROBE_EVERY = 250`, `DRAWS = 1`, `DEPTHS = (9, 11)`, `EVAL_SEEDS`, `check_seeds`, `ckpt_dir(seed, root=None) -> Path`, `record_name(update, seed) -> str`, `gatec_name(seed) -> str`, `snapshot_name(update) -> str`, `search_name(depth, seed) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/test_exp082.py`:

```python
"""EXP-082: EXP-075's X judge continued by value iteration from 4000 to 16,000 updates."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "082_x_long_budget"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp082_cells", EXP / "cells.py")
c75 = cells.c75
t75 = _load("exp075_train", REPO / "experiments" / "075_wide_region" / "train.py")


def test_constants_are_the_spec_values():
    """Catches a setting drifting from EXP-075's X: a continuation on other settings is not the
    same run as a straight 16,000-update X."""
    a = t75.build_parser().parse_args(["--n-updates", "1"])
    assert (cells.BATCH, cells.SYNC_EVERY, cells.PROBE_EVERY, cells.DRAWS) == (
        a.batch, a.sync_every, a.probe_every, a.draws) == (1000, 100, 250, 1)
    assert t75.MAX_LEN == 14 and t75.VI_ARM == "W"
    assert cells.START == c75.N_UPDATES == 4000 and cells.LEGS == (8000, 16000)
    assert cells.DEPTHS == (9, 11) and cells.ARM == "X16k" and cells.SRC_ARM == "X"
    assert list(cells.EVAL_SEEDS) == list(range(12))


def test_names_and_layout():
    """The checkpoint folder is EXP-075's layout under EXP-082's outputs, so EXP-075's
    load_judge reads an X16k judge unchanged; every record name carries the experiment."""
    assert cells.ckpt_dir(3) == c75.ckpt_dir("X", 3, cells.OUT)
    assert cells.ckpt_dir(3).parent == EXP / "outputs"
    assert cells.record_name(8000, 3) == "exp082_X16k_u8000_s3.json"
    assert cells.gatec_name(3) == "exp082_gatec_s3.json"
    assert cells.snapshot_name(8000) == "judge_u8000.pt"
    assert cells.search_name(11, 3) == "exp082_J4V-X16k_d11_s3.json"
    assert cells.E75_OUT != cells.OUT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q` (tool timeout 300000)
Expected: FAIL at collection, `FileNotFoundError` for `experiments/082_x_long_budget/cells.py`.

- [ ] **Step 3: Write the implementation**

Create `experiments/082_x_long_budget/cells.py`:

```python
"""EXP-082 cells: EXP-075's X judge continued by value iteration from update 4000 to 16,000, arm
X16k. Distance-to-solved is the probe's yardstick and the search's success test only.

Spec: docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md, sections 2 to 5.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E75 = REPO / "experiments" / "075_wide_region"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c75 = _load("exp075_cells", E75 / "cells.py")

ARM = "X16k"
# EXP-075's arm. Also the checkpoint folder's name, so EXP-075's load_judge reads ours.
SRC_ARM = "X"
START = c75.N_UPDATES
LEGS = (8000, 16000)
# EXP-075's X settings (its train.py defaults), unchanged.
BATCH = 1000
SYNC_EVERY = 100
PROBE_EVERY = 250
DRAWS = 1
DEPTHS = (9, 11)
EVAL_SEEDS = c75.EVAL_SEEDS
OUT = HERE / "outputs"
E75_OUT = E75 / "outputs"
E76_OUT = REPO / "experiments" / "076_deeper_tree" / "outputs"
E81_OUT = REPO / "experiments" / "081_oxt_long_budget" / "outputs"

check_seeds = c75.check_seeds


def ckpt_dir(seed: int, root=None) -> Path:
    return c75.ckpt_dir(SRC_ARM, seed, OUT if root is None else root)


def record_name(update: int, seed: int) -> str:
    return f"exp082_{ARM}_u{update}_s{seed}.json"


def gatec_name(seed: int) -> str:
    return f"exp082_gatec_s{seed}.json"


def snapshot_name(update: int) -> str:
    return f"judge_u{update}.pt"


def search_name(depth: int, seed: int) -> str:
    return f"exp082_J4V-{ARM}_d{depth}_s{seed}.json"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q` (tool timeout 300000)
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/082_x_long_budget/cells.py tests/experiments/test_exp082.py
git commit -m "EXP-082 cells: X16k constants (EXP-075's X settings), names and checkpoint layout"
```

---

### Task 2: gatec.py, Gate C and the continuation it seeds

**Files:**
- Create: `experiments/082_x_long_budget/gatec.py`
- Modify: `tests/experiments/test_exp082.py` (append)

**Interfaces:**
- Consumes: Task 1's `cells`; EXP-075 `train.py` (`gate_l`, `MAX_LEN`, `VI_ARM`, `_encoder_drift`, `_resumed_from`); EXP-075 `evaluate.py` (`load_judge(train_arm, seed, judge_dir, required_updates)`, `JudgeCritic`, `load_cell`); EXP-081 `gatec.py` (`check_opt(opt_sd, judge, step)`, `_write_json(path, obj)`, `_copy_atomic(src, dst)`, `_git_commit()`).
- Produces: module `gatec` with `cells`, `c75`, `t75`, `e75`, `g81`, `gate_l`, `CKPT_FILES`, `sets(seed) -> (exclude, probe, distance)`, `measure(judge, probe, distance, seed) -> dict`, `blob_of(path) -> str`, `tracked_blob(path) -> str | None`, `check_identity(src, tracked, blob) -> (list[str], dict)`, `check_target(seed, src) -> bool`, `seed_continuation(seed, src_root, out) -> str`, `gate_c(seed, out=cells.OUT, src_root=cells.E75_OUT, tracked=tracked_blob, blob=blob_of) -> dict`, `passed(seed, out) -> bool`, `pending(seeds, out) -> list[int]`. Gate C records hold `continued_from = {"experiment": "EXP-075", "update": START}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp082.py`:

```python
gatec = _load("exp082_gatec", EXP / "gatec.py")
t75tests = _load("test_exp075_helpers", REPO / "tests" / "experiments" / "test_exp075.py")
vi = gatec.t75.vi

SEED = 12
START, LEGS, BATCH = 2, (3, 4), 4


def _probe():
    return [(apply_move(SOLVED, 0), 1), (apply_move(apply_move(SOLVED, 1), 3), 2)]


def fake_measure(judge, probe, distance, seed):
    """Stands in for Gate L (which needs the full BFS table): a number every weight moves."""
    return {"margin": sum(float(p.detach().double().sum()) for p in judge.parameters())}


def _train_straight(root, n, sync_every=1):
    """EXP-075's X in miniature: fake pretrained encoder, n updates of value iteration."""
    judge = gatec.c75.make_judge(SEED, "X", root)
    vi.train_judge(judge, gatec.t75.VI_ARM, n_updates=n, batch=BATCH, max_len=gatec.t75.MAX_LEN,
                   n_actions=N_ACTIONS, exclude=set(), probe=_probe(), seed=SEED,
                   sync_every=sync_every, probe_every=1, draws=1,
                   ckpt_dir=gatec.c75.ckpt_dir("X", SEED, root), log=lambda m: None)
    return judge


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fake EXP-075 X at update START with its record, and an empty EXP-082 out dir."""
    monkeypatch.setattr(gatec.cells, "START", START)
    monkeypatch.setattr(gatec, "sets", lambda seed: (set(), _probe(), None))
    monkeypatch.setattr(gatec, "measure", fake_measure)
    src = tmp_path / "e75"
    t75tests._fake_encoder(src, "X", SEED)
    judge = _train_straight(src, START)
    (src / gatec.c75.record_name("X", SEED)).write_text(json.dumps(
        {"gate_l": fake_measure(judge, None, None, SEED), "n_probe": 2}))
    return src, tmp_path / "out"


def _gc(src, out, tracked=lambda p: "b", blob=lambda p: "b"):
    return gatec.gate_c(SEED, out, src_root=src, tracked=tracked, blob=blob)


def _src_ck(src):
    return gatec.c75.ckpt_dir("X", SEED, src)


def test_gate_c_passes_and_copies_all_four_files(world):
    """Catches a copy missing the optimizer or the target net (the continuation would crash or,
    worse, restart Adam), and a Gate C record without the provenance training reads."""
    src, out = world
    rec = _gc(src, out)
    assert rec["pass"] is True and rec["continuation"] == "seeded"
    dst = gatec.cells.ckpt_dir(SEED, out)
    for f in ("judge.pt", "jt.pt", "opt.pt", "state.json"):
        assert (dst / f).read_bytes() == (_src_ck(src) / f).read_bytes(), f
    assert rec["continued_from"] == {"experiment": "EXP-075", "update": START}
    assert rec["start_update"] == START and rec["opt_bad"] == [] and rec["jt_ok"] is True
    assert gatec.passed(SEED, out)


@pytest.mark.parametrize("tracked,blob,why", [
    (lambda p: "b", lambda p: "other", "differs"),
    (lambda p: None, lambda p: "b", "untracked"),
])
def test_gate_c_fails_on_a_source_that_is_not_the_tracked_file(world, tracked, blob, why):
    src, out = world
    with pytest.raises(SystemExit, match="GATE C FAILED"):
        _gc(src, out, tracked, blob)
    rec = json.loads((out / gatec.cells.gatec_name(SEED)).read_text())
    assert rec["pass"] is False and any(why in b for b in rec["identity_bad"])
    assert not gatec.cells.ckpt_dir(SEED, out).exists()


def test_gate_c_fails_on_a_start_that_is_not_exp075s(world):
    """Catches a Gate L comparison that is not exact, or that compares the wrong judge."""
    src, out = world
    p = src / gatec.c75.record_name("X", SEED)
    r = json.loads(p.read_text())
    r["gate_l"]["margin"] += 1e-9
    p.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="GATE C FAILED"):
        _gc(src, out)
    assert not gatec.passed(SEED, out)


def test_gate_c_fails_when_exp075_is_not_at_the_start_update(world, monkeypatch):
    src, out = world
    monkeypatch.setattr(gatec.cells, "START", START + 1)
    with pytest.raises(SystemExit, match="start update"):
        _gc(src, out)


def test_gate_c_fails_on_an_optimizer_at_the_wrong_step(world):
    """Catches a stale opt.pt (the laptop's opt.pt predates its judge.pt by two days)."""
    src, out = world
    p = _src_ck(src) / "opt.pt"
    sd = torch.load(p)
    for st in sd["state"].values():
        st["step"] = torch.tensor(float(START - 1))
    torch.save(sd, p)
    with pytest.raises(SystemExit, match="GATE C FAILED"):
        _gc(src, out)
    rec = json.loads((out / gatec.cells.gatec_name(SEED)).read_text())
    assert any("step" in b for b in rec["opt_bad"])


def test_gate_c_fails_on_a_target_net_of_another_architecture(world):
    src, out = world
    other = gatec.c75.build_judge(SEED, "Y", gatec.c75.fresh_sensory("Y", SEED))
    torch.save(other.state_dict(), _src_ck(src) / "jt.pt")
    with pytest.raises(SystemExit, match="GATE C FAILED"):
        _gc(src, out)
    assert json.loads((out / gatec.cells.gatec_name(SEED)).read_text())["jt_ok"] is False


def test_gate_c_never_overwrites_a_started_continuation(world):
    """REVIEW FOCUS 2. A re-run after training began must keep the continuation, or training
    silently restarts from 4000."""
    src, out = world
    _gc(src, out)
    marker = gatec.cells.ckpt_dir(SEED, out) / "judge.pt"
    marker.write_bytes(b"started")
    (out / gatec.cells.gatec_name(SEED)).unlink()
    rec = _gc(src, out)
    assert rec["continuation"] == "kept" and marker.read_bytes() == b"started"


def test_gate_c_never_writes_exp075s_folder(world):
    src, out = world
    before = {f.name: f.read_bytes() for f in _src_ck(src).iterdir()}
    _gc(src, out)
    assert {f.name: f.read_bytes() for f in _src_ck(src).iterdir()} == before


def test_measure_is_exp075s_gate_l_call(monkeypatch):
    """Catches Gate C re-measuring with other arguments than EXP-075's train.run did."""
    seen = {}
    monkeypatch.setattr(gatec.gate_l, "leaf_rank_margin",
                        lambda *a, **k: seen.update(a=a, k=k) or {"m": 1})
    assert gatec.measure("J", "P", "D", 5) == {"m": 1}
    assert seen == {"a": ("J", "P", "D", 5, N_ACTIONS), "k": {}}


def test_identity_reads_git_for_a_tracked_file_and_none_for_an_untracked_one(tmp_path):
    f = gatec.cells.E75_OUT / "judge_X_s0" / "state.json"
    assert len(gatec.blob_of(f)) == 40 and gatec.tracked_blob(f) == gatec.blob_of(f)
    p = tmp_path / "x.json"
    p.write_text("{}")
    assert gatec.tracked_blob(p) is None


def test_pending_rechecks_failed_seeds(tmp_path):
    (tmp_path / gatec.cells.gatec_name(1)).write_text(json.dumps({"pass": True}))
    (tmp_path / gatec.cells.gatec_name(2)).write_text(json.dumps({"pass": False}))
    assert gatec.pending([1, 2, 3], tmp_path) == [2, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q` (tool timeout 300000)
Expected: FAIL at collection, `FileNotFoundError` for `gatec.py`.

- [ ] **Step 3: Write the implementation**

Create `experiments/082_x_long_budget/gatec.py`:

```python
"""EXP-082 Gate C, continuity (voiding), and the continuation it seeds.

Per seed: (a) EXP-075's X judge.pt and state.json are the tracked files and state.json says
update 4000; (b) opt.pt holds a step-4000 arm-W Adam state shaped like the judge and jt.pt loads
into the judge's architecture; (c) the judge, re-measured by EXP-075's Gate L call on EXP-075's
probe, equals EXP-075's committed gate_l exactly. Only then are the four checkpoint files copied
into EXP-082's directory. EXP-075's directory is only ever read.

Spec: docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md, section 4.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/082_x_long_budget/gatec.py --workers 4
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp082_cells", HERE / "cells.py")
c75 = cells.c75
t75 = _load("exp075_train", cells.E75 / "train.py")
e75 = _load("exp075_evaluate", cells.E75 / "evaluate.py")
g81 = _load("exp081_gatec", cells.REPO / "experiments" / "081_oxt_long_budget" / "gatec.py")
gate_l = t75.gate_l

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402

torch.set_num_threads(1)

# state.json last: its presence marks a seeded continuation.
CKPT_FILES = ("judge.pt", "jt.pt", "opt.pt", "state.json")
TRACKED = ("judge.pt", "state.json")


def sets(seed):
    """EXP-075's exclusion and probe sets and the true distance, from the full BFS table (about
    65 s). The exclusion set shapes the walks, so a continuation must use EXP-075's exactly."""
    provider = ExactBFSDistance(max_depth=None)
    return c75.exclusion_set(seed, provider), c75.probe_set(seed, provider), provider.distance


def measure(judge, probe, distance, seed) -> dict:
    """EXP-075's Gate L call (its train.run), unchanged."""
    return gate_l.leaf_rank_margin(judge, probe, distance, seed, N_ACTIONS)


def blob_of(path) -> str:
    return subprocess.run(["git", "hash-object", str(path)], cwd=cells.REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def tracked_blob(path):
    """The blob id git records for `path` at HEAD, or None when it is not tracked there."""
    try:
        rel = Path(path).resolve().relative_to(cells.REPO).as_posix()
    except ValueError:
        return None
    r = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=cells.REPO, capture_output=True,
                       text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def check_identity(src: Path, tracked=tracked_blob, blob=blob_of) -> tuple[list[str], dict]:
    bad, ids = [], {}
    for name in TRACKED:
        want = tracked(src / name)
        got = blob(src / name) if (src / name).exists() else None
        ids[name] = got
        if want is None:
            bad.append(f"{name} untracked")
        elif got != want:
            bad.append(f"{name} differs from the tracked file")
    return bad, ids


def check_target(seed, src: Path) -> bool:
    """jt.pt loads, strictly, into the judge's architecture."""
    jt = c75.build_judge(seed, cells.SRC_ARM, c75.fresh_sensory(cells.SRC_ARM, seed))
    try:
        jt.load_state_dict(torch.load(src / "jt.pt", map_location="cpu"))
    except (RuntimeError, FileNotFoundError):
        return False
    return True


def seed_continuation(seed, src_root, out) -> str:
    """Copy EXP-075's checkpoint into EXP-082's directory. Never overwrites: a continuation that
    has started is left alone."""
    dst = cells.ckpt_dir(seed, out)
    if (dst / "state.json").exists():
        return "kept"
    src = c75.ckpt_dir(cells.SRC_ARM, seed, src_root)
    dst.mkdir(parents=True, exist_ok=True)
    for name in CKPT_FILES:
        g81._copy_atomic(src / name, dst / name)
    return "seeded"


def gate_c(seed, out=cells.OUT, src_root=cells.E75_OUT, tracked=tracked_blob,
           blob=blob_of) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    out, src_root = Path(out), Path(src_root)
    src = c75.ckpt_dir(cells.SRC_ARM, seed, src_root)
    rec75 = json.loads((src_root / c75.record_name(cells.SRC_ARM, seed)).read_text())
    update = json.loads((src / "state.json").read_text()).get("update")
    identity_bad, ids = check_identity(src, tracked, blob)
    opt_bad, jt_ok, start_ok, gl = ["not checked"], False, False, None
    if update == cells.START:
        judge = e75.load_judge(cells.SRC_ARM, seed, src_root, required_updates=cells.START)
        opt_bad = g81.check_opt(torch.load(src / "opt.pt", map_location="cpu"), judge,
                                step=cells.START)
        jt_ok = check_target(seed, src)
        _, probe, distance = sets(seed)
        gl = json.loads(json.dumps(measure(judge, probe, distance, seed)))
        start_ok = gl == rec75["gate_l"]
    ok = update == cells.START and not identity_bad and not opt_bad and jt_ok and start_ok
    rec = {"seed": seed, "pass": ok, "start_update": update, "identity_bad": identity_bad,
           "blobs": ids, "opt_bad": opt_bad, "jt_ok": jt_ok, "start_gate_l_equal": start_ok,
           "start_margin": None if gl is None else gl["margin"],
           "continued_from": {"experiment": "EXP-075", "update": update},
           "git_commit": g81._git_commit()}
    if ok:
        rec["continuation"] = seed_continuation(seed, src_root, out)
    rec["wall_s"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    g81._write_json(out / cells.gatec_name(seed), rec)
    if not ok:
        raise SystemExit(f"GATE C FAILED for seed {seed}: start update {update}, identity "
                         f"{identity_bad}, opt {opt_bad}, jt loads {jt_ok}, gate_l equal "
                         f"{start_ok}")
    return rec


def passed(seed, out=cells.OUT) -> bool:
    p = Path(out) / cells.gatec_name(seed)
    return p.exists() and json.loads(p.read_text()).get("pass") is True


def pending(seeds, out=cells.OUT) -> list[int]:
    """Seeds without a passing Gate C record; a failed record is re-checked."""
    return [s for s in seeds if not passed(s, out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = pending(args.seeds, args.out_dir)
    print(f"EXP-082 Gate C: {len(jobs)} seeds, {args.workers} workers", flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(gate_c, s, args.out_dir): s for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                print(f"  {i}/{len(futs)}  seed {r['seed']}  PASS  margin {r['start_margin']!r}  "
                      f"{r['continuation']}  wall {r['wall_s']}s", flush=True)
            except SystemExit as e:
                failed += 1
                print(f"  {i}/{len(futs)}  seed {futs[f]}  {e}", flush=True)
    if failed:
        raise SystemExit(f"Gate C failed on {failed} seed(s): the experiment is VOID")
    print("done.")


if __name__ == "__main__":
    main()
```

Note for the implementer: `test_gate_c_fails_when_exp075_is_not_at_the_start_update` matches on `"start update"`, which the SystemExit message contains. `e75.load_judge` refuses a checkpoint whose state.json update is not exactly `required_updates`, so it is only called when `update == cells.START`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q` (tool timeout 600000)
Expected: all pass (2 from Task 1, 12 new including the 2 parametrized cases).

- [ ] **Step 5: Mutation check**

Temporarily change `start_ok = gl == rec75["gate_l"]` to `start_ok = True`, re-run `test_gate_c_fails_on_a_start_that_is_not_exp075s`, confirm it FAILS, then revert. Temporarily drop `"opt.pt"` from `CKPT_FILES`, re-run `test_gate_c_passes_and_copies_all_four_files`, confirm it FAILS, then revert. Confirm `git diff experiments/082_x_long_budget/gatec.py` shows only the intended file content (no mutation left).

- [ ] **Step 6: Commit**

```bash
git add experiments/082_x_long_budget/gatec.py tests/experiments/test_exp082.py
git commit -m "EXP-082 Gate C: tracked identity, step-4000 optimizer, target net, exact Gate L start; seeds the continuation without touching EXP-075"
```

---

### Task 3: train.py, two legs on the continuation

**Files:**
- Create: `experiments/082_x_long_budget/train.py`
- Modify: `tests/experiments/test_exp082.py` (append)

**Interfaces:**
- Consumes: Task 2's `gatec` (`passed`, `sets`, `measure`, `cells`, `c75`, `t75`, `g81`); `vi.train_judge` from `neuromorphic.training.value_iteration`.
- Produces: module `train` with `gatec`, `cells`, `vi`, `run(seed, out=cells.OUT, *, legs=cells.LEGS, batch=cells.BATCH, sync_every=cells.SYNC_EVERY, probe_every=cells.PROBE_EVERY, draws=cells.DRAWS, src_root=cells.E75_OUT) -> list[dict]`, `pending(seeds, out, legs) -> list[int]`. Leg records hold: every `vi.train_judge` result key (`updates`, `wall_s`, `loss_last`, `probes`, `final_probe`), plus `arm`, `seed`, `hidden`, `readout`, `n_updates`, `resumed_from`, `batch`, `max_len`, `n_actions`, `sync_every`, `probe_every`, `draws`, `n_exclude`, `n_probe`, `continued_from`, `encoder_drift`, `gate_l`, `git_commit`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp082.py`:

```python
train = _load("exp082_train", EXP / "train.py")


def _run(out, src, legs=LEGS):
    return train.run(SEED, out, legs=legs, batch=BATCH, sync_every=1, probe_every=1, src_root=src)


@pytest.fixture
def tworld(world, monkeypatch):
    """`world`, with train's own gatec module patched the same way."""
    monkeypatch.setattr(train.gatec.cells, "START", START)
    monkeypatch.setattr(train.gatec, "sets", lambda seed: (set(), _probe(), None))
    monkeypatch.setattr(train.gatec, "measure", fake_measure)
    src, out = world
    train.gatec.gate_c(SEED, out, src_root=src, tracked=lambda p: "b", blob=lambda p: "b")
    return src, out


def test_continuation_equals_a_straight_run(tworld, tmp_path):
    """The spec's section 2 premise: catches a continuation that restarts the optimizer, skips
    the target net, or trains on other settings. Compared weight for weight."""
    src, out = tworld
    _run(out, src)
    straight = tmp_path / "straight"
    shutil.copytree(src / "encoders", straight / "encoders")
    want = _train_straight(straight, LEGS[-1]).state_dict()
    got = torch.load(train.cells.ckpt_dir(SEED, out) / "judge.pt")
    assert got.keys() == want.keys()
    assert all(torch.equal(got[k], want[k]) for k in want)


def test_leg_records_carry_what_the_aggregator_checks(tworld):
    src, out = tworld
    r3, r4 = _run(out, src)
    assert (r3["n_updates"], r4["n_updates"]) == LEGS
    assert (r3["resumed_from"], r4["resumed_from"]) == (START, LEGS[0])
    assert r4["continued_from"] == {"experiment": "EXP-075", "update": START}
    assert (r4["arm"], r4["hidden"], r4["readout"], r4["max_len"]) == ("X16k", 512, "wide", 14)
    assert (r4["sync_every"], r4["batch"], r4["n_probe"]) == (1, BATCH, 2)
    assert r4["encoder_drift"] > 0.0
    assert len(r4["probes"]) > len(r3["probes"]) > START
    on_disk = json.loads((out / train.cells.record_name(LEGS[1], SEED)).read_text())
    assert on_disk["gate_l"] == json.loads(json.dumps(r4["gate_l"]))
    snap = torch.load(train.cells.ckpt_dir(SEED, out) / train.cells.snapshot_name(LEGS[0]))
    assert any(not torch.equal(snap[k], v) for k, v in
               torch.load(train.cells.ckpt_dir(SEED, out) / "judge.pt").items())


def test_a_rerun_skips_finished_legs_and_resumes_leg_2(tworld):
    """REVIEW FOCUS 1. The 03:31 restart re-runs the chain: leg 1's record must be untouched."""
    src, out = tworld
    _run(out, src)
    p1 = out / train.cells.record_name(LEGS[0], SEED)
    before = p1.read_text()
    (out / train.cells.record_name(LEGS[1], SEED)).unlink()
    recs = _run(out, src)
    assert [r["n_updates"] for r in recs] == [LEGS[1]]
    assert p1.read_text() == before


def test_a_checkpoint_past_leg_1_is_never_recorded_as_leg_1(tworld):
    """REVIEW FOCUS 1. A checkpoint at 16,000 with leg 1's record lost must not become the 8000
    point."""
    src, out = tworld
    _run(out, src)
    (out / train.cells.record_name(LEGS[0], SEED)).unlink()
    with pytest.raises(SystemExit, match=f"past leg end {LEGS[0]}"):
        _run(out, src)


def test_training_refuses_a_seed_without_a_passing_gate_c(world, monkeypatch):
    src, out = world
    with pytest.raises(SystemExit, match="Gate C"):
        _run(out, src)


def test_pending_is_seeds_without_the_last_leg(tmp_path):
    (tmp_path / train.cells.record_name(16000, 1)).write_text("{}")
    (tmp_path / train.cells.record_name(8000, 2)).write_text("{}")
    assert train.pending([1, 2, 3], tmp_path) == [2, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q` (tool timeout 600000)
Expected: FAIL at collection, `FileNotFoundError` for `train.py`.

- [ ] **Step 3: Write the implementation**

Create `experiments/082_x_long_budget/train.py`:

```python
"""EXP-082 training: EXP-075's value iteration for arm X, its settings unchanged, continuing
each seed's Gate C-seeded checkpoint in two legs, to update 8000 (recorded, judge snapshotted)
and then to 16,000. The body is EXP-075's train.run with only the checkpoint and record paths
changed; the pretrained encoder (the reference for encoder drift) is EXP-075's.

A leg runs only when its record is missing, and refuses a checkpoint already past its end, so a
seed killed in leg 2 resumes in leg 2 and its checkpoint is never recorded as the 8000 point.

Spec: docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md, section 5.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/082_x_long_budget/train.py --workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gatec = _load("exp082_gatec", HERE / "gatec.py")
cells = gatec.cells  # one module object, so a patched constant reaches both
c75 = gatec.c75
t75 = gatec.t75
g81 = gatec.g81

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

torch.set_num_threads(1)


def run(seed, out=cells.OUT, *, legs=cells.LEGS, batch=cells.BATCH,
        sync_every=cells.SYNC_EVERY, probe_every=cells.PROBE_EVERY, draws=cells.DRAWS,
        src_root=cells.E75_OUT) -> list:
    torch.set_num_threads(1)
    out = Path(out)
    if not gatec.passed(seed, out):
        raise SystemExit(f"seed {seed} has no passing Gate C record in {out}; run gatec.py first")
    ck = cells.ckpt_dir(seed, out)
    state_path = ck / "state.json"
    if not state_path.exists():
        raise SystemExit(f"{state_path} missing: Gate C passed but seeded no continuation")
    # From the Gate C record, never state.json: train_judge rewrites state.json at every
    # checkpoint with its own keys.
    continued_from = json.loads((out / cells.gatec_name(seed)).read_text())["continued_from"]
    exclude, probe, distance = gatec.sets(seed)
    written = []
    for n in legs:
        path = out / cells.record_name(n, seed)
        if path.exists():
            continue
        at = json.loads(state_path.read_text())["update"]
        if at > n:
            raise SystemExit(f"{state_path}: checkpoint is at update {at}, past leg end {n}, "
                             f"and {path.name} is missing; it cannot be recorded as the {n} point")
        # The pretrained judge first, as EXP-075's run does; train_judge loads judge.pt over it.
        judge = c75.make_judge(seed, cells.SRC_ARM, src_root)
        resumed_from = t75._resumed_from(ck)
        res = vi.train_judge(
            judge, t75.VI_ARM, n_updates=n, batch=batch, max_len=t75.MAX_LEN,
            n_actions=N_ACTIONS, exclude=exclude, probe=probe, seed=seed,
            sync_every=sync_every, probe_every=probe_every, draws=draws, ckpt_dir=ck)
        if n != legs[-1]:
            g81._copy_atomic(ck / "judge.pt", ck / cells.snapshot_name(n))
        judge.eval()
        rec = {**res, "arm": cells.ARM, "seed": seed, "hidden": c75.HIDDEN[cells.SRC_ARM],
               "readout": judge.readout, "n_updates": n, "resumed_from": resumed_from,
               "batch": batch, "max_len": t75.MAX_LEN, "n_actions": N_ACTIONS,
               "sync_every": sync_every, "probe_every": probe_every, "draws": draws,
               "n_exclude": len(exclude), "n_probe": len(probe),
               "continued_from": continued_from,
               "encoder_drift": t75._encoder_drift(judge, cells.SRC_ARM, seed, src_root),
               "gate_l": gatec.measure(judge, probe, distance, seed),
               "git_commit": g81._git_commit()}
        g81._write_json(path, rec)
        written.append(rec)
    return written


def pending(seeds, out=cells.OUT, legs=cells.LEGS) -> list[int]:
    return [s for s in seeds if not (Path(out) / cells.record_name(legs[-1], s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = pending(args.seeds, args.out_dir)
    print(f"EXP-082 train: {len(jobs)} seeds, {args.workers} workers, legs {cells.LEGS}, "
          f"out {args.out_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, s, args.out_dir): s for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            for r in f.result():
                loss = "None" if r["loss_last"] is None else format(r["loss_last"], ".6f")
                print(f"  {i}/{len(jobs)}  seed {r['seed']}  update {r['n_updates']}  resumed "
                      f"{r['resumed_from']}  loss_last {loss}  wall_s {r['wall_s']:.1f}  drift "
                      f"{r['encoder_drift']:.3f}  gate_l margin {r['gate_l']['margin']:.4f}",
                      flush=True)
    print("done.")


if __name__ == "__main__":
    main()
```

Note for the implementer: `tworld` patches `train.gatec`, which is a different module object from the `gatec` loaded at the top of the test file (each `_load` executes the file anew). Both must be patched, which is why `tworld` exists.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q` (tool timeout 600000)
Expected: all pass.

- [ ] **Step 5: Mutation check**

Temporarily pass `sync_every=100` instead of `sync_every=sync_every` to `vi.train_judge`, re-run `test_continuation_equals_a_straight_run`, confirm it FAILS, revert. Temporarily delete the `if at > n:` block, re-run `test_a_checkpoint_past_leg_1_is_never_recorded_as_leg_1`, confirm it FAILS, revert. Check `git diff experiments/082_x_long_budget/train.py` is clean of mutations.

- [ ] **Step 6: Commit**

```bash
git add experiments/082_x_long_budget/train.py tests/experiments/test_exp082.py
git commit -m "EXP-082 training: EXP-075's value iteration continued in two legs (8000 recorded and snapshotted, then 16,000); continuation equals a straight run"
```

---

### Task 4: evaluate.py, the search cells at depths 9 and 11

**Files:**
- Create: `experiments/082_x_long_budget/evaluate.py`
- Modify: `tests/experiments/test_exp082.py` (append)

**Interfaces:**
- Consumes: Task 2's `gatec` (`cells`, `e75`, `g81`); `evaluate_lookahead` from `neuromorphic.training.lookahead`; `e75.load_judge`, `e75.JudgeCritic`, `e75.load_cell(depth, seed) -> (agent, head, states, train_seed)`.
- Produces: module `evaluate` with `gatec`, `cells`, `ARM = "J4V-X16k"`, `K = 4`, `record_name(depth, seed)`, `run_cell(depth, seed, out=cells.OUT, limit_states=None) -> dict`, `pending(depths, seeds, out) -> list[tuple[int, int]]`. Search records hold every `evaluate_lookahead` key (including `success_rate`, `solved`, `n`, `k`, `k_goal`, `mode`, `no_revisit`, `optimality`, `mean_steps`, `goal_fired_frac`) plus `depth`, `seed`, `judge_seed`, `arm`, `judge = "X16k"`, `fired4`, `wall_s`, `git_commit`, `limit_states`, `judge_updates`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp082.py`:

```python
evaluate = _load("exp082_evaluate", EXP / "evaluate.py")


def test_search_cell_is_exp076s_j4v():
    e76 = _load("exp076_evaluate", REPO / "experiments" / "076_deeper_tree" / "evaluate.py")
    assert e76.ARMS["J4V-X"] == (evaluate.K, evaluate.K) == (4, 4)
    assert evaluate.ARM == "J4V-X16k"
    assert evaluate.record_name(9, 2) == cells.search_name(9, 2)


def test_search_cell_refuses_a_judge_short_of_16000(tmp_path):
    """REVIEW FOCUS 3. Catches a search run by hand on a judge still training."""
    t75tests._fake_judge_ckpt(tmp_path, "X", 0, update=8000)
    with pytest.raises(SystemExit, match="evaluation needs 16000"):
        evaluate.run_cell(11, 0, tmp_path, limit_states=1)
    assert not (tmp_path / evaluate.record_name(11, 0)).exists()


def test_search_cell_refuses_an_unregistered_depth(tmp_path):
    with pytest.raises(SystemExit, match="depth 7"):
        evaluate.run_cell(7, 0, tmp_path, limit_states=1)


def test_a_16k_judge_drives_a_real_j4v_cell_at_both_depths(tmp_path):
    """Catches a record missing a field the aggregator checks, on EXP-075's real cells."""
    t75tests._fake_judge_ckpt(tmp_path, "X", 0, update=16000)
    for d in (9, 11):
        r = evaluate.run_cell(d, 0, tmp_path, limit_states=1)
        assert (r["arm"], r["judge"], r["depth"], r["seed"], r["judge_seed"]) == (
            "J4V-X16k", "X16k", d, 0, 0)
        assert (r["k"], r["k_goal"], r["mode"], r["no_revisit"]) == (4, 4, "C", True)
        assert r["judge_updates"] == 16000 and r["limit_states"] == 1 and r["n"] == 1
        assert json.loads((tmp_path / evaluate.record_name(d, 0)).read_text())["depth"] == d


def test_search_pending_runs_depth_11_first_and_skips_finished(tmp_path):
    (tmp_path / evaluate.record_name(11, 1)).write_text("{}")
    assert evaluate.pending([9, 11], [1, 2], tmp_path) == [(11, 2), (9, 1), (9, 2)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q -k "search"` (tool timeout 600000)
Expected: FAIL at collection, `FileNotFoundError` for `evaluate.py`.

- [ ] **Step 3: Write the implementation**

Create `experiments/082_x_long_budget/evaluate.py`:

```python
"""EXP-082 search cells: EXP-081's J4V cell (mode C, k 4, goal check at 4, no-revisit, EXP-075's
held-out sets and streams), unchanged except for the judge, which is X16k, at depths 9 and 11.
The search runs on every seed whatever the probe says (spec section 8). A judge not at exactly
16,000 updates is refused.

Spec: docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md, sections 7 and 9.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/082_x_long_budget/evaluate.py --workers 12
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gatec = _load("exp082_gatec", HERE / "gatec.py")
cells = gatec.cells
e75 = gatec.e75

from neuromorphic.training.lookahead import evaluate_lookahead  # noqa: E402

torch.set_num_threads(1)

ARM = f"J4V-{cells.ARM}"
K = 4


def record_name(depth: int, seed: int) -> str:
    return cells.search_name(depth, seed)


def run_cell(depth, seed, out=cells.OUT, limit_states=None) -> dict:
    torch.set_num_threads(1)
    if depth not in cells.DEPTHS:
        raise SystemExit(f"depth {depth} is not one of the registered depths {cells.DEPTHS}")
    t0 = time.time()
    out = Path(out)
    judge = e75.load_judge(cells.SRC_ARM, seed, out, required_updates=cells.LEGS[-1])
    critic = e75.JudgeCritic(judge)
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=K, k_goal=K,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    state = json.loads((cells.ckpt_dir(seed, out) / "state.json").read_text())
    rec = {**res, "depth": depth, "seed": seed, "judge_seed": seed, "arm": ARM,
           "judge": cells.ARM, "fired4": res["first_fire_len"].get("4", 0),
           "wall_s": round(time.time() - t0, 1), "git_commit": gatec.g81._git_commit(),
           "limit_states": limit_states, "judge_updates": int(state["update"])}
    gatec.g81._write_json(out / record_name(depth, seed), rec)
    return rec


def pending(depths, seeds, out=cells.OUT) -> list[tuple[int, int]]:
    """Depth 11 first: its cells are the longest, so they start in the first wave."""
    return [(d, s) for d in sorted(depths, reverse=True) for s in seeds
            if not (Path(out) / record_name(d, s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--depths", type=int, nargs="+", default=list(cells.DEPTHS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = pending(args.depths, args.seeds, args.out_dir)
    print(f"EXP-082 search: {len(jobs)} cells, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_cell, d, s, args.out_dir): (d, s) for d, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  {ARM} d{r['depth']} s{r['seed']} solved {r['solved']}/"
                  f"{r['n']} wall {r['wall_s']}s", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q -k "search"` (tool timeout 600000)
Expected: 5 passed. `test_a_16k_judge_drives_a_real_j4v_cell_at_both_depths` takes the longest (one scramble per depth through a 4-deep tree); if it alone runs past about 120 s, report the measured time rather than marking it slow.

- [ ] **Step 5: Commit**

```bash
git add experiments/082_x_long_budget/evaluate.py tests/experiments/test_exp082.py
git commit -m "EXP-082 search cells: EXP-081's J4V cell with the X16k judge at depths 9 and 11; refuses a judge not at 16,000"
```

---

### Task 5: aggregate.py, gates, claims, verdict and the reported block

**Files:**
- Create: `experiments/082_x_long_budget/aggregate.py`
- Modify: `tests/experiments/test_exp082.py` (append)

**Interfaces:**
- Consumes: Task 4's `evaluate` (`ARM`, `K`, `cells`); EXP-080 `aggregate.py` (`one_sided_p`, `far(rec)`, `overall(rec)`, `claim_verdict(diffs, gate_ok)`, `gate_t(margins)`, `_read(path)`, `CLAIM_ALPHA`, `CLAIM_MIN`, `GATE_T_THRESHOLD`, `a72.pair_diffs(a, b)`).
- Produces: module `aggregate` with `C1_MIN = 0.03`, `CLAIM_ALPHA`, `GATE_T_THRESHOLD`, `verdict(gate_c_ok, t_ok, c1, c2) -> str`, `claim1_verdict(diffs) -> (str, p, mean)`, `stage_gates(out, e75_root) -> dict`, `format_stage_gates(res) -> list[str]`, `run_all(out, e75_root, e76_root, e81_root) -> dict` with keys `verdict` (str), `lines` (list[str]), `reported` (list[str]). The gates stage prints exactly one of `SEARCH: RUN ...` or `SEARCH: NOT RUN ...`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp082.py`:

```python
agg = _load("exp082_aggregate", EXP / "aggregate.py")
acells = agg.cells


def _gl(far, overall=0.3):
    by = {str(d): {"hit": 0.2 + (far if d >= 9 else 0.5), "chance": 0.2, "n": 50}
          for d in range(7, 12)}
    return {"by_distance": by, "margin": overall, "chance": 0.2, "hit": 0.2 + overall, "n": 250}


def _final_probe():
    return {"spearman_7_11": 0.3, "mean_j_by_distance": {str(d): float(d) for d in range(7, 12)}}


def _x_rec(seed, far):
    return {"arm": "X", "seed": seed, "n_probe": 483, "gate_l": _gl(far),
            "final_probe": _final_probe(), "loss_last": 0.2}


def _leg(seed, u, far, overall=0.3, **over):
    r = {"arm": "X16k", "seed": seed, "n_updates": u, "batch": 1000, "sync_every": 100,
         "probe_every": 250, "draws": 1, "max_len": 14, "readout": "wide", "hidden": 512,
         "resumed_from": 4000 if u == 8000 else 8000,
         "continued_from": {"experiment": "EXP-075", "update": 4000}, "encoder_drift": 13.0,
         "n_probe": 483, "gate_l": _gl(far, overall), "final_probe": _final_probe(),
         "loss_last": 0.2, "wall_s": 100.0, "git_commit": "abc"}
    r.update(over)
    return r


def _search(seed, depth, rate, arm="J4V-X16k", **over):
    r = {"arm": arm, "judge": "X16k", "depth": depth, "seed": seed, "judge_seed": seed, "k": 4,
         "k_goal": 4, "mode": "C", "no_revisit": True, "judge_updates": 16000,
         "limit_states": None, "success_rate": rate, "optimality": 0.8, "mean_steps": 10.0,
         "goal_fired_frac": 0.1, "n": 200}
    r.update(over)
    return r


def _jit(s):
    """Per-seed noise of mean -0.00033 and mixed sign: a zero gain must not confirm."""
    return ((s * 7) % 5 - 2) * 0.002


def _agg_world(tmp, *, far16=0.12, rate11=0.195, rate9=0.40, overall16=0.30, gc=True,
               **leg_over):
    roots = [tmp / n for n in ("out", "e75", "e76", "e81")]
    out, e75, e76, e81 = roots
    for d in roots:
        d.mkdir(parents=True)
    for s in range(12):
        j, lvl = _jit(s), 0.01 * (s % 4)
        (out / acells.gatec_name(s)).write_text(json.dumps({"seed": s, "pass": gc or s != 3}))
        (e75 / acells.c75.record_name("X", s)).write_text(json.dumps(_x_rec(s, 0.12 + lvl)))
        (out / acells.record_name(8000, s)).write_text(json.dumps(_leg(s, 8000, 0.12 + lvl)))
        (out / acells.record_name(16000, s)).write_text(json.dumps(
            _leg(s, 16000, far16 + lvl + j, overall16, **leg_over)))
        for d, rate, base in ((9, rate9, 0.40), (11, rate11, 0.195)):
            lv = 0.01 * (s % 3)
            (e76 / f"exp076_J4V-X_d{d}_s{s}.json").write_text(json.dumps(
                _search(s, d, base + lv, arm="J4V-X")))
            (out / acells.search_name(d, s)).write_text(json.dumps(_search(s, d, rate + lv + j)))
        (e81 / f"exp081_J4V-O-Xt16k_d11_s{s}.json").write_text(json.dumps(
            _search(s, 11, 0.41, arm="J4V-O-Xt16k")))
    return roots


def test_verdict_has_a_line_for_every_row_of_spec_section_8():
    v = agg.verdict
    assert v(False, True, "CONFIRMED", "CONFIRMED") == "VOID"
    assert v(True, False, "CONFIRMED", "CONFIRMED") == "UNRESOLVED (X16k UNTRAINED)"
    assert v(True, True, "CONFIRMED", "REFUTED") == "BUDGET-LIMITED, PAYS"
    assert v(True, True, "CONFIRMED (SMALL)", "CONFIRMED") == \
        "ORDERING IMPROVES, SEARCH DOES NOT PAY"
    assert v(True, True, "REFUTED", "CONFIRMED (SMALL)") == \
        "VALUE ITERATION SATURATED AT 4X BUDGET"


def test_bars_are_the_spec_values():
    assert agg.CLAIM_ALPHA == 0.025 and agg.C1_MIN == 0.03
    assert agg.a80.CLAIM_MIN == 0.05
    assert agg.GATE_T_THRESHOLD == pytest.approx(0.13564, abs=1e-5)


def test_a_large_search_gain_pays(tmp_path):
    res = agg.run_all(*_agg_world(tmp_path, rate11=0.295))
    assert res["verdict"] == "BUDGET-LIMITED, PAYS"


def test_a_small_search_gain_with_a_far_band_gain_reads_ordering_improves(tmp_path):
    """Catches Claim 1's 0.03 minimum missing: +0.02 on every seed is CONFIRMED (SMALL) and
    falls through to Claim 2."""
    res = agg.run_all(*_agg_world(tmp_path, rate11=0.215, far16=0.22))
    assert any("CLAIM 1" in ln and "CONFIRMED (SMALL)" in ln for ln in res["lines"])
    assert res["verdict"] == "ORDERING IMPROVES, SEARCH DOES NOT PAY"


def test_nothing_moving_reads_saturated(tmp_path):
    res = agg.run_all(*_agg_world(tmp_path))
    assert res["verdict"] == "VALUE ITERATION SATURATED AT 4X BUDGET"


def test_claim_1_reads_depth_11_not_depth_9(tmp_path):
    """Catches the depths swapped: a depth-9 gain alone is reported, never the primary."""
    res = agg.run_all(*_agg_world(tmp_path, rate9=0.60))
    assert res["verdict"] == "VALUE ITERATION SATURATED AT 4X BUDGET"
    assert any(ln.startswith("depth 9") and "+0.19" in ln for ln in res["reported"])


def test_untrained_16k_is_unresolved_whatever_the_search(tmp_path):
    res = agg.run_all(*_agg_world(tmp_path, rate11=0.295, overall16=0.10))
    assert res["verdict"] == "UNRESOLVED (X16k UNTRAINED)"


def test_a_failed_gate_c_is_void(tmp_path):
    res = agg.run_all(*_agg_world(tmp_path, rate11=0.295, gc=False))
    assert res["verdict"] == "VOID"
    assert "GATE C (continuity): FAIL on seeds [3]" in res["lines"]


@pytest.mark.parametrize("over,msg", [
    ({"resumed_from": 0}, "fresh run"),
    ({"continued_from": {"experiment": "EXP-080", "update": 4000}}, "continued_from"),
    ({"encoder_drift": 0.0}, "did not move"),
    ({"n_probe": 250}, "n_probe"),
    ({"sync_every": 1}, "sync_every"),
    ({"limit_states": 2}, "smoke record"),
])
def test_a_record_that_is_not_the_continuation_is_refused(tmp_path, over, msg):
    """REVIEW FOCUS 4."""
    with pytest.raises(SystemExit, match=msg):
        agg.run_all(*_agg_world(tmp_path, **over))


def test_mixed_checkouts_are_refused(tmp_path):
    """REVIEW FOCUS 4. Spec section 6: one git_commit across the 16,000-update records."""
    roots = _agg_world(tmp_path)
    p = roots[0] / acells.record_name(16000, 5)
    r = json.loads(p.read_text())
    r["git_commit"] = "def"
    p.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="checkouts"):
        agg.run_all(*roots)


@pytest.mark.parametrize("over,msg", [
    ({"judge_updates": 8000}, "judge_updates"),
    ({"limit_states": 1}, "smoke record"),
    ({"k": 3}, "k is 3"),
])
def test_a_search_record_that_is_not_the_registered_cell_is_refused(tmp_path, over, msg):
    """REVIEW FOCUS 4."""
    roots = _agg_world(tmp_path)
    p = roots[0] / acells.search_name(11, 4)
    p.write_text(json.dumps(_search(4, 11, 0.2, **over)))
    with pytest.raises(SystemExit, match=msg):
        agg.run_all(*roots)


def test_gates_stage_prints_exactly_one_search_line(tmp_path):
    ok = agg.format_stage_gates(agg.stage_gates(*_agg_world(tmp_path / "a")[:2]))
    bad = agg.format_stage_gates(agg.stage_gates(*_agg_world(tmp_path / "b", gc=False)[:2]))
    assert sum(ln.startswith("SEARCH: RUN") for ln in ok) == 1
    assert not any(ln.startswith("SEARCH: NOT RUN") for ln in ok)
    assert sum(ln.startswith("SEARCH: NOT RUN") for ln in bad) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q -k "verdict or bars or gain or saturated or depth_11 or untrained or gate_c_is_void or refused or checkouts or one_search_line"` (tool timeout 600000)
Expected: FAIL at collection, `FileNotFoundError` for `aggregate.py`.

- [ ] **Step 3: Write the implementation**

Create `experiments/082_x_long_budget/aggregate.py`:

```python
"""EXP-082 aggregator. Every gate outcome is a verdict of its own: a failed Gate C prints VOID, a
failed Gate T prints UNRESOLVED (X16k UNTRAINED), and the VERDICT line is spec section 8's table.
Claim 1 is the depth-11 search (p < 0.025 and mean >= 0.03); Claim 2 is the far band at EXP-080's
Claim 1 bar (claim_verdict). CONFIRMED (SMALL) on either falls through to the next row.

Spec: docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md, sections 6 to 9.

Usage:
    .venv/bin/python experiments/082_x_long_budget/aggregate.py --stage gates
    .venv/bin/python experiments/082_x_long_budget/aggregate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("exp082_evaluate", HERE / "evaluate.py")
cells = ev.cells
a80 = _load("exp080_aggregate", cells.REPO / "experiments" / "080_oracle_ordering" / "aggregate.py")
one_sided_p = a80.one_sided_p
far = a80.far
overall = a80.overall
pair_diffs = a80.a72.pair_diffs

SEEDS = cells.EVAL_SEEDS
CLAIM_ALPHA = a80.CLAIM_ALPHA
C1_MIN = 0.03
GATE_T_THRESHOLD = a80.GATE_T_THRESHOLD
U8, U16 = cells.LEGS
SETTINGS = {"arm": cells.ARM, "batch": cells.BATCH, "sync_every": cells.SYNC_EVERY,
            "probe_every": cells.PROBE_EVERY, "draws": cells.DRAWS, "max_len": 14,
            "readout": "wide", "hidden": 512}


def read_gatec(out, seed) -> dict:
    p = Path(out) / cells.gatec_name(seed)
    if not p.exists():
        raise SystemExit(f"missing {p.name}: run the gatec phase first")
    return json.loads(p.read_text())


def x_record(seed, e75_root=cells.E75_OUT) -> dict:
    return a80._read(Path(e75_root) / cells.c75.record_name(cells.SRC_ARM, seed))


def check_leg_record(rec, name, update, seed, x_rec) -> None:
    for k, want in {**SETTINGS, "seed": seed, "n_updates": update}.items():
        if rec.get(k) != want:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, spec says {want!r}")
    r = rec.get("resumed_from")
    if not (isinstance(r, int) and r >= cells.START):
        raise SystemExit(f"{name}: resumed_from {r!r} is under {cells.START}; that is a fresh "
                         f"run, not EXP-075's X continued")
    if (rec.get("continued_from") or {}).get("experiment") != "EXP-075":
        raise SystemExit(f"{name}: continued_from is {rec.get('continued_from')!r}")
    drift = rec.get("encoder_drift")
    if not (isinstance(drift, float) and drift > 0.0):
        raise SystemExit(f"{name}: the region did not move (encoder_drift {drift!r})")
    if rec.get("n_probe") != x_rec.get("n_probe"):
        raise SystemExit(f"{name}: n_probe {rec.get('n_probe')!r} differs from EXP-075's "
                         f"{x_rec.get('n_probe')!r}")
    got_n = {k: v["n"] for k, v in rec["gate_l"]["by_distance"].items()}
    want_n = {k: v["n"] for k, v in x_rec["gate_l"]["by_distance"].items()}
    if got_n != want_n:
        raise SystemExit(f"{name}: gate_l n per distance {got_n} differs from X's {want_n}")


def read_leg(out, update, seed, x_rec) -> dict:
    name = cells.record_name(update, seed)
    rec = a80._read(Path(out) / name)
    check_leg_record(rec, name, update, seed, x_rec)
    return rec


def check_one_checkout(recs: dict) -> str:
    commits = {r.get("git_commit") for r in recs.values()}
    if len(commits) != 1:
        raise SystemExit(f"the {U16}-update records come from {len(commits)} checkouts "
                         f"{sorted(map(str, commits))}; spec section 6 needs one")
    return commits.pop()


def claim1_verdict(diffs) -> tuple[str, float, float]:
    p, m = one_sided_p(diffs), st.mean(diffs)
    if p < CLAIM_ALPHA and m > 0:
        return ("CONFIRMED" if m >= C1_MIN else "CONFIRMED (SMALL)"), p, m
    return "REFUTED", p, m


def verdict(gate_c_ok: bool, t_ok: bool, c1: str, c2: str) -> str:
    """Spec section 8, one line."""
    if not gate_c_ok:
        return "VOID"
    if not t_ok:
        return "UNRESOLVED (X16k UNTRAINED)"
    if c1 == "CONFIRMED":
        return "BUDGET-LIMITED, PAYS"
    if c2 == "CONFIRMED":
        return "ORDERING IMPROVES, SEARCH DOES NOT PAY"
    return "VALUE ITERATION SATURATED AT 4X BUDGET"


def stage_gates(out=cells.OUT, e75_root=cells.E75_OUT) -> dict:
    out = Path(out)
    gc = {s: read_gatec(out, s) for s in SEEDS}
    gc_fail = [s for s, r in gc.items() if r.get("pass") is not True]
    if gc_fail:
        return {"gate_c": False, "gate_c_fail": gc_fail}
    x = {s: x_record(s, e75_root) for s in SEEDS}
    r8 = {s: read_leg(out, U8, s, x[s]) for s in SEEDS}
    r16 = {s: read_leg(out, U16, s, x[s]) for s in SEEDS}
    commit = check_one_checkout(r16)
    t_ok = a80.gate_t([overall(r16[s]) for s in SEEDS])
    d2 = [far(r16[s]) - far(x[s]) for s in SEEDS]
    c2 = a80.claim_verdict(d2, t_ok) + (sum(d > 0 for d in d2),)
    return {"gate_c": True, "t": t_ok, "c2": c2, "commit": commit,
            "recs": {cells.START: x, U8: r8, U16: r16}}


def format_stage_gates(res) -> list[str]:
    if not res["gate_c"]:
        return [f"GATE C (continuity): FAIL on seeds {res['gate_c_fail']}",
                "SEARCH: NOT RUN (Gate C failed; the experiment is VOID)"]
    recs = res["recs"]
    pts = "  ".join(f"u{u} far {st.mean(far(recs[u][s]) for s in SEEDS):.4f} overall "
                    f"{st.mean(overall(recs[u][s]) for s in SEEDS):.4f}"
                    for u in (cells.START, U8, U16))
    v, p, m, wins = res["c2"]
    return [f"GATE C (continuity): PASS on {len(SEEDS)} of {len(SEEDS)}",
            f"GATE T (trained) X16k: {'PASS' if res['t'] else 'FAIL'}, mean overall margin vs "
            f"{GATE_T_THRESHOLD:.5f}; {pts}",
            f"CLAIM 2 X16k - X far-band: diff {m:+.4f}, ahead on {wins} of {len(SEEDS)}, "
            f"p {p:.4f} -> {v}",
            "SEARCH: RUN (spec section 8: the search runs whatever the probe says)"]


def read_search(out, depth, seed) -> dict:
    name = cells.search_name(depth, seed)
    r = a80._read(Path(out) / name)
    for k, want in (("arm", ev.ARM), ("judge", cells.ARM), ("depth", depth), ("seed", seed),
                    ("judge_seed", seed), ("k", ev.K), ("k_goal", ev.K), ("mode", "C"),
                    ("no_revisit", True), ("judge_updates", U16)):
        if r.get(k) != want:
            raise SystemExit(f"{name}: {k} is {r.get(k)!r}, spec says {want!r}")
    return r


def _rates(recs) -> dict:
    return {s: r["success_rate"] for s, r in recs.items()}


def _contrast(a: dict, b: dict) -> tuple[list, float, float, int]:
    d = pair_diffs(a, b)
    return d, one_sided_p(d), st.mean(d), sum(x > 0 for x in d)


def search_stage(out, e76_root, e81_root) -> dict:
    s = {d: {x: read_search(out, d, x) for x in SEEDS} for d in cells.DEPTHS}
    b = {d: {x: a80._read(Path(e76_root) / f"exp076_J4V-X_d{d}_s{x}.json") for x in SEEDS}
         for d in cells.DEPTHS}
    o81 = {x: a80._read(Path(e81_root) / f"exp081_J4V-O-Xt16k_d11_s{x}.json") for x in SEEDS}
    d11, _, _, wins = _contrast(_rates(s[11]), _rates(b[11]))
    return {"c1": claim1_verdict(d11) + (wins,), "search": s, "base": b, "o81": o81}


def _band(rec, d) -> float:
    v = rec["gate_l"]["by_distance"][d]
    return v["hit"] - v["chance"]


def reported(res, srch) -> list[str]:
    recs, pts = res["recs"], (cells.START, U8, U16)
    lines = ["far band per seed (X u4000 / u8000 / u16000):"]
    for s in SEEDS:
        lines.append(f"  s{s}: " + " / ".join(f"{far(recs[u][s]):.4f}" for u in pts))
    for d in ("7", "8", "9", "10", "11"):
        lines.append(f"margin at distance {d}: " + "  ".join(
            f"u{u} {st.mean(_band(recs[u][s], d) for s in SEEDS):.4f}" for u in pts))
    for u in pts:
        r = recs[u]
        sp = st.mean(r[s]["final_probe"]["spearman_7_11"] for s in SEEDS)
        mj = "  ".join(
            f"J@{d} {st.mean(r[s]['final_probe']['mean_j_by_distance'][d] for s in SEEDS):.3f}"
            for d in ("9", "10", "11"))
        loss = [r[s]["loss_last"] for s in SEEDS if r[s]["loss_last"] is not None]
        lines.append(f"u{u}: overall {st.mean(overall(r[s]) for s in SEEDS):.4f}  "
                     f"spearman_7_11 {sp:.4f}  {mj}  loss_last "
                     f"{st.mean(loss) if loss else None}")
    lines.append(f"encoder drift at u{U16}: "
                 f"{st.mean(recs[U16][s]['encoder_drift'] for s in SEEDS):.3f}")
    a9, b9 = _rates(srch["search"][9]), _rates(srch["base"][9])
    _, p9, m9, w9 = _contrast(a9, b9)
    lines.append(f"depth 9 J4V-X16k - J4V-X: {st.mean(a9.values()):.4f} vs "
                 f"{st.mean(b9.values()):.4f}, diff {m9:+.4f}, ahead on {w9} of 12, "
                 f"p {p9:.4f} (reported, not a claim)")
    for k in ("optimality", "mean_steps", "goal_fired_frac"):
        a = st.mean(srch["search"][11][s][k] for s in SEEDS)
        b = st.mean(srch["base"][11][s][k] for s in SEEDS)
        lines.append(f"d11 {k}: X16k {a:.4f}  J4V-X {b:.4f}")
    a11, o11 = _rates(srch["search"][11]), _rates(srch["o81"])
    _, _, mo, wo = _contrast(a11, o11)
    lines.append(f"d11 X16k - O-Xt16k (EXP-081, oracle ceiling, descriptive): "
                 f"{st.mean(a11.values()):.4f} vs {st.mean(o11.values()):.4f}, diff {mo:+.4f}, "
                 f"ahead on {wo} of 12")
    w = [recs[U8][s]["wall_s"] + recs[U16][s]["wall_s"] for s in SEEDS]
    lines.append(f"training wall per seed, both legs: {min(w):.0f} to {max(w):.0f} s")
    return lines


def run_all(out=cells.OUT, e75_root=cells.E75_OUT, e76_root=cells.E76_OUT,
            e81_root=cells.E81_OUT) -> dict:
    res = stage_gates(out, e75_root)
    lines = format_stage_gates(res)
    if not res["gate_c"]:
        return {"verdict": verdict(False, False, "VOID", "VOID"), "lines": lines, "reported": []}
    srch = search_stage(out, e76_root, e81_root)
    v, p, m, wins = srch["c1"]
    a, b = _rates(srch["search"][11]), _rates(srch["base"][11])
    lines.append(f"CLAIM 1 J4V-X16k - J4V-X d11: {st.mean(a.values()):.4f} vs "
                 f"{st.mean(b.values()):.4f}, diff {m:+.4f}, ahead on {wins} of 12, "
                 f"p {p:.4f} -> {v}")
    return {"verdict": verdict(True, res["t"], v, res["c2"][0]), "lines": lines,
            "reported": reported(res, srch)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    ap.add_argument("--stage", choices=("gates",))
    args = ap.parse_args()
    if args.stage == "gates":
        for line in format_stage_gates(stage_gates(args.out_dir)):
            print(line)
        return
    res = run_all(args.out_dir)
    for line in res["lines"]:
        print(line)
    print("VERDICT: " + res["verdict"])
    if res["reported"]:
        print("\nReported (spec section 9, never confirmations):")
        for line in res["reported"]:
            print("  " + line)


if __name__ == "__main__":
    main()
```

Note for the implementer: `a80._read` refuses any record whose `limit_states` is not None with "is a smoke record", which is what the `limit_states` cases match. In `_agg_world`, `rate9=0.60` against a base of 0.40 gives a depth-9 diff of +0.20 less the jitter mean (+0.1997), which the reported line prints as `+0.1997`; the test matches the prefix `+0.19`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q -k "verdict or bars or gain or saturated or depth_11 or untrained or gate_c_is_void or refused or checkouts or one_search_line"` (tool timeout 600000)
Expected: all selected pass.

- [ ] **Step 5: Mutation check**

Temporarily set `C1_MIN = 0.0`, re-run `test_a_small_search_gain_with_a_far_band_gain_reads_ordering_improves`, confirm it FAILS, revert. Temporarily make `search_stage` compute Claim 1 from depth 9 (`s[9]`, `b[9]`), re-run `test_claim_1_reads_depth_11_not_depth_9`, confirm it FAILS, revert. Check `git diff` is clean of mutations.

- [ ] **Step 6: Commit**

```bash
git add experiments/082_x_long_budget/aggregate.py tests/experiments/test_exp082.py
git commit -m "EXP-082 aggregator: Gate C VOID, Gate T, Claim 1 depth-11 search (0.03 bar), Claim 2 far band, spec section 8 verdict, reported block with depth 9"
```

---

### Task 6: laptop drivers, launch082.ps1 and chain082.ps1

**Files:**
- Create: `experiments/082_x_long_budget/launch082.ps1`
- Create: `experiments/082_x_long_budget/chain082.ps1`
- Modify: `tests/experiments/test_exp082.py` (append)

**Interfaces:**
- Consumes: the four Python drivers (`gatec.py`, `train.py`, `evaluate.py`, `aggregate.py`), each taking `--seeds`, `--out-dir`, `--workers`; `aggregate.py --stage gates` printing `SEARCH: RUN` or `SEARCH: NOT RUN`.
- Produces: `launch082.ps1 -Phase check|gatec|train|search [-Workers N]` and `chain082.ps1 [-From gatec|train|gates]`, which writes `C:\Users\mlgbr\verdict082.txt` only on a clean aggregate and logs to `C:\Users\mlgbr\chain082.log`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp082.py`:

```python
def test_chain_reads_the_search_lines_the_aggregator_prints(tmp_path):
    """REVIEW FOCUS 5. Catches the chain and the aggregator drifting apart, and a chain that
    reads a crashed gates stage as 'search not run'."""
    chain = (EXP / "chain082.ps1").read_text()
    assert '-match "SEARCH: NOT RUN"' in chain and '-notmatch "SEARCH: RUN"' in chain
    assert "STOP: the gates stage exited" in chain
    yes = "\n".join(agg.format_stage_gates(agg.stage_gates(*_agg_world(tmp_path / "a")[:2])))
    assert re.search("SEARCH: RUN", yes) and not re.search("SEARCH: NOT RUN", yes)
    no = "\n".join(agg.format_stage_gates(
        agg.stage_gates(*_agg_world(tmp_path / "b", gc=False)[:2])))
    assert re.search("SEARCH: NOT RUN", no)


def test_drivers_use_the_spec_worker_counts():
    """REVIEW FOCUS 5. Spec sections 5 and 10: Gate C 4, train 4 (X commits 3.6 GB per worker
    and throughput does not rise above 4), search 12."""
    spec = '@{ "gatec" = 4; "train" = 4; "search" = 12 }'
    assert spec in (EXP / "chain082.ps1").read_text()
    assert spec in (EXP / "launch082.ps1").read_text()


def test_drivers_have_no_em_dashes_or_backticks_in_strings():
    for f in ("chain082.ps1", "launch082.ps1"):
        text = (EXP / f).read_text()
        assert "\u2014" not in text, f
        assert not re.search(r'"[^"\n]*`[^"\n]*"', text), f
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q -k "chain or drivers"` (tool timeout 600000)
Expected: FAIL, `FileNotFoundError` for `chain082.ps1`.

- [ ] **Step 3: Write launch082.ps1**

Create `experiments/082_x_long_budget/launch082.ps1`:

```powershell
# EXP-082 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File <repo>\experiments\082_x_long_budget\launch082.ps1 -Phase check
#   ... -Phase gatec  -Workers 4
#   ... -Phase train  -Workers 4
#   ... -Phase search -Workers 12
#
# Order (spec sections 4 to 8): Gate C -> train (two legs) -> gates stage -> search (depths 9 and
# 11, always) -> aggregate. chain082.ps1 runs them. Every driver skips finished work and training
# resumes its checkpoints, so a phase is re-runnable after the 03:31 restart.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Launch long phases with OS-level redirection (playbook
# section 3), never Start-Process from an ssh session.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","gatec","train","search")][string]$Phase,
    [int]$Workers = 0
)

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\082_x_long_budget"
$caps = @{ "gatec" = 4; "train" = 4; "search" = 12 }

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# EXP-075's 12 X checkpoints at 4000 with all four files, 12 X encoders, 12 X records; EXP-076's
# 12 depth-9 and 12 depth-11 J4V-X records; EXP-081's 12 depth-11 O-Xt16k records.
$present = & $py -c "import json, sys; from pathlib import Path; o=Path('experiments/075_wide_region/outputs'); r6=Path('experiments/076_deeper_tree/outputs'); r81=Path('experiments/081_oxt_long_budget/outputs'); ck=lambda s: o/('judge_X_s'+str(s)); c=sum(all((ck(s)/f).exists() for f in ('judge.pt','jt.pt','opt.pt','state.json')) and json.loads((ck(s)/'state.json').read_text())['update']==4000 for s in range(12)); x=sum((o/'encoders'/('enc_X_s'+str(s)+'.pt')).exists() for s in range(12)); xr=sum((o/('exp075_train_X_s'+str(s)+'.json')).exists() for s in range(12)); j9=sum((r6/('exp076_J4V-X_d9_s'+str(s)+'.json')).exists() for s in range(12)); j11=sum((r6/('exp076_J4V-X_d11_s'+str(s)+'.json')).exists() for s in range(12)); q=sum((r81/('exp081_J4V-O-Xt16k_d11_s'+str(s)+'.json')).exists() for s in range(12)); sys.stdout.write(' '.join(str(v) for v in (c, x, xr, j9, j11, q)))"
if ($LASTEXITCODE -ne 0 -or $present -ne "12 12 12 12 12 12") { Write-Error "expected '12 12 12 12 12 12' (X checkpoints, X encoders, X records, J4V-X d9, J4V-X d11, O-Xt16k d11), found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                     = $where"
"checkout                    = $head"
"inputs                      = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs                     = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

if ($Workers -eq 0) { $Workers = 2 }
if ($Workers -gt $caps[$Phase]) { Write-Error "Workers capped at $($caps[$Phase]) for $Phase; got $Workers."; exit 1 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$outDir = Join-Path $repo "$exp\outputs"
$evalSeeds = @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")

switch ($Phase) {
    "gatec"  { $driver = "$exp\gatec.py" }
    "train"  { $driver = "$exp\train.py" }
    "search" { $driver = "$exp\evaluate.py" }
}
$cliArgs = @("--seeds") + $evalSeeds + @("--out-dir", $outDir, "--workers", $Workers)

# Sample every python process: the venv's python.exe is a redirector (EXP-079's disclosure).
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

- [ ] **Step 4: Write chain082.ps1**

Create `experiments/082_x_long_budget/chain082.ps1`:

```powershell
# EXP-082 chain: Gate C -> train (two legs) -> gates stage (record checks, Gate T, Claim 2) ->
# search (depths 9 and 11; always run once the gates stage passes, spec section 8) -> full
# aggregate into C:\Users\mlgbr\verdict082.txt. Re-runnable from the top: every phase skips
# finished work and training resumes its checkpoints. Workers are spec section 10's.
# Launch it with OS-level redirection (playbook section 3):
#   ssh -n laptop 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\experiments\082_x_long_budget\chain082.ps1 > C:\Users\mlgbr\chain082.out 2>&1"'
param([ValidateSet("gatec","train","gates")][string]$From = "gatec")
$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py = Join-Path $repo ".venv\Scripts\python.exe"
$l = Join-Path $repo "experiments\082_x_long_budget\launch082.ps1"
$agg = "experiments\082_x_long_budget\aggregate.py"
$log = "C:\Users\mlgbr\chain082.log"
$verdictFile = "C:\Users\mlgbr\verdict082.txt"
$workers = @{ "gatec" = 4; "train" = 4; "search" = 12 }
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo "src"
function Say($m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m; Add-Content -Path $log -Value $line; $line }
Say "chain start from $From at $(git rev-parse --short HEAD)"
$order = @("gatec","train","gates","search","aggregate")
$go = $false
$runSearch = $true
foreach ($p in $order) {
    if ($p -eq $From) { $go = $true }
    if (-not $go) { continue }
    if ($p -eq "gates") {
        # Fail closed: a refused record raises SystemExit, prints neither search line, and must
        # stop the chain, not read as "search not run".
        $g = & $py $agg --stage gates 2>&1 | Out-String
        $code = $LASTEXITCODE
        Say ("gates: " + $g.Trim())
        if ($code -ne 0) { Say "STOP: the gates stage exited $code; fix the records before going on"; exit 1 }
        if ($g -match "SEARCH: NOT RUN") { $runSearch = $false; continue }
        if ($g -notmatch "SEARCH: RUN") { Say "STOP: the gates stage printed no search line"; exit 2 }
        continue
    }
    if ($p -eq "search" -and -not $runSearch) { Say "search not run (see the gates line)"; continue }
    if ($p -eq "aggregate") {
        $a = & $py $agg 2>&1 | Out-String
        $code = $LASTEXITCODE
        Say ("aggregate exit " + $code)
        # Written only on success, so the watcher never mistakes an error for a verdict.
        if ($code -ne 0) { Say ("STOP: the aggregate exited " + $code + ": " + $a.Trim()); exit 1 }
        Set-Content -Path ($verdictFile + ".tmp") -Value $a
        Move-Item -Force ($verdictFile + ".tmp") $verdictFile
        continue
    }
    Say "phase $p workers $($workers[$p])"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $workers[$p]
    Say "phase $p exit $LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Say "STOP: phase $p failed"; exit 1 }
}
Say "chain done"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py -q -k "chain or drivers"` (tool timeout 600000)
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add experiments/082_x_long_budget/launch082.ps1 experiments/082_x_long_budget/chain082.ps1 tests/experiments/test_exp082.py
git commit -m "EXP-082 launcher and chain: input check, Gate C 4 / train 4 / search 12 workers, fail-closed gates stage, verdict file only on a clean aggregate"
```

---

### Task 7: whole-branch verification, test count, laptop pre-flight (controller only)

This task is the controller's, not a subagent's: it runs the suite in chunks and touches the laptop.

**Files:**
- Modify: `CLAUDE.md` (test count line)

- [ ] **Step 1: Run the EXP-082 file and its neighbours**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp082.py tests/experiments/test_exp081.py tests/experiments/test_exp075.py -q -m "not slow"` (tool timeout 600000)
Expected: all pass. If this call approaches 600 s, split it by file.

- [ ] **Step 2: Run the rest of the fast suite in chunks**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -m "not slow" --ignore=tests/training --ignore=tests/experiments` (tool timeout 600000), then `tests/experiments` minus the three files above (tool timeout 600000), then `tests/training/test_cube_baseline.py` (tool timeout 600000), and the remainder of `tests/training` as a deliberate background run (it is about 837 s and cannot fit one call).
Expected: all pass; report any failure with its output.

- [ ] **Step 3: Record the test count**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ --collect-only -q | tail -1` and `... -m "not slow" --collect-only -q | tail -1` (tool timeout 300000). In `CLAUDE.md`, change the bold "as of 2026-10-21" count entry into a plain line and append `**<total> as of 2026-10-26 (<not slow> not slow, <slow> slow; EXP-082 code)**`, matching the existing list's format.

```bash
git add CLAUDE.md
git commit -m "CLAUDE.md: test count after EXP-082 code"
git push
```

- [ ] **Step 4: Laptop pre-flight**

```bash
ssh laptop 'powershell -NoProfile -Command "cd C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project; git pull --ff-only; git rev-parse --short HEAD"'
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\082_x_long_budget\launch082.ps1 -Phase check'
```

(tool timeout 120000 each). Expected: the pull reaches the pushed HEAD, and the check prints `inputs = 12 12 12 12 12 12` and `CHECK OK`. Any other count stops the build; report it.

- [ ] **Step 5: Log**

Run: `secretary log "EXP-082 built: Gate C, two-leg X continuation, depth 9 and 11 search, aggregator, laptop drivers; laptop pre-flight CHECK OK; run scheduled Wednesday 2026-10-28"` (tool timeout 60000).

---

## Run procedure (Wednesday 2026-10-28, not part of the build)

1. On the laptop's pulled checkout: `ssh -n laptop 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\082_x_long_budget\chain082.ps1 > C:\Users\mlgbr\chain082.out 2>&1"'` in a background Bash call.
2. Read the Gate C phase log within the first 30 minutes: 12 PASS lines, each with the margin equal to EXP-075's committed `gate_l.margin` for its seed. A failure is VOID; stop and report it.
3. Start a named VPS watcher (a `run_in_background` poll loop every 10 minutes) that prints chain state, python count, record counts and `verdict082.txt` existence, re-runs the chain when the chain process is gone and no verdict exists, and exits when the verdict file appears. Re-arm it on every exit and after any resume. Remember the watcher prints VPS time, about 3h50m ahead of laptop time.
4. On the verdict: copy back `verdict082.txt` and the records, re-run `aggregate.py` on the VPS and diff it byte for byte, then write `RESULTS.md` (verdict, gates, claims, per-seed table, reported block, disclosures, provenance), force-add the records and the X16k `judge.pt`, `judge_u8000.pt` and `state.json` (not `opt.pt` or `jt.pt`), and update the README, roadmap, report and vault.
