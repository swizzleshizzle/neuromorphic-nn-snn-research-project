# EXP-079 Judge Trained Where Scrambles Live Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build EXP-079's experiment folder: train judge S (EXP-075's X recipe on random walks of length 1 to 40 instead of 1 to 14), evaluate it in EXP-076's 4-move tree against EXP-076's committed J4V-X records, and aggregate the pre-registered gates and claims.

**Architecture:** One folder, `experiments/079_judge_trained_deep/`, reusing earlier experiments through `importlib` exactly as EXP-075 to EXP-078 do: `cells.py` (paths, S judge construction and loading, the pilot-encoder proof), `train.py` (EXP-075's driver with `--max-len` and a walk recorder that observes, never alters, the library's walks), `evaluate.py` (J4V-S and J4V-X cells, Gate R rank cells), `aggregate.py` (Gates 0, L, E, M, R, 1; Claims 1 and 2; the pilot amendment numbers and stop rule), `launch079.ps1` and `chain079.ps1` for the laptop. No library change.

**Tech Stack:** Python 3 via `.venv/bin/python`, PyTorch (CPU), pytest, Windows PowerShell 5.1 on the laptop.

**Spec:** `docs/superpowers/specs/2026-10-18-exp079-judge-trained-where-scrambles-live-design.md` (read it before starting; section 8 was tightened 2026-10-19 so Gate E also voids).

## Global Constraints

- Walk length for S: `max_len` **40**, fixed (spec section 3). EXP-075's X used 14.
- Training settings, all unchanged from EXP-075's X: 4000 updates, batch 1000, target sync every 100, probe every 250, 1 draw, `make_optimizer(judge, "W")`, wide readout, hidden 512, head seeded by `torch.manual_seed(seed)` right before it is built.
- S starts each evaluation seed from `experiments/075_wide_region/outputs/encoders/enc_X_s<seed>.pt` (committed).
- Search: EXP-076's J4V: `mode="C"`, `k=4`, `k_goal=4`, `no_revisit=True`, depths 9 and 11, seeds 0 to 11.
- Pilot seeds 12 and 13 only (no evaluation seed is seen by the pilot). No 10-seed sensitivity line.
- Claims: one-sided exact sign-flip over 4096 flips, alpha **0.025** each (0.05 / 2). CONFIRMED (SMALL) under a mean of 0.01.
- Gate L: per-seed margin > 0 by exact one-sided sign-flip at p < 0.05 AND mean margin >= half S's mean pilot margin (EXP-074's `gate_l_verdict`).
- Pilot stop rule: S's mean pilot Gate L margin below **0.1298** (EXP-075's X threshold, `GATE_L_THRESHOLD["X"]` = 0.12983333333333333, imported, never retyped) stops the experiment.
- Gate M alpha 0.05, probe distances 9 to 11, reported, binds the reading.
- VOID when Gate 0(a), Gate 0(b), Gate L or Gate E fails.
- Worker cap 6 on the laptop. Pilot training uses 2 workers.
- Run python through the venv, never bare `python`. In a git worktree prefix every command with `PYTHONPATH=src` (the editable install points at the main checkout).
- **Always pass an explicit Bash tool timeout; the default is 120 s.** Every test command in this plan fits in one call with `timeout: 600000`. Call it directly and let it block; piped output buffers, so silence mid-run is normal and is not a hang. Never run the whole suite from a task.
- No em-dashes anywhere (code, docs, commit messages). Plain commit messages, no `Co-Authored-By`, no "Generated with". Stage explicit paths, never `git add -A`.
- Test-strength rule (CLAUDE.md): every assertion must be able to fail against the defect it names. Never weaken a threshold to make a test pass.

## Review Focus

1. **A training session resumed after a 03:31 Windows Update restart** sees only its own updates, so the realised histogram covers fewer than 250 updates. Expected: the record states how many (`train_hist_updates`) and is not refused. Pinned in Task 2 (`test_recorder_reports_how_many_updates_it_covers`).
2. **A crash after the final checkpoint and before the record is written** makes the relaunch train zero updates, so no walk length is observed. Expected: the aggregator refuses that record with a message naming the remedy, rather than reading a record that cannot prove its walk length. Pinned in Task 4 (`test_train_record_must_prove_its_walk_length`).
3. **Only one of the two EXP-075 pilot encoders is proved.** Expected: nothing is copied, so the pilot never mixes a copied encoder with a re-pretrained one. Pinned in Task 1 (`test_pilot_encoders_copy_all_or_nothing`).
4. **EXP-076's J4V-X records have no `judge` field**, while this experiment's J4V-S records must say `"S"`. Expected: the aggregator reads EXP-076's records for J4V-X and refuses a J4V-S record whose judge is anything but S. Pinned in Task 4 (`test_load_records_reads_x_from_exp076_and_s_from_out`).
5. **`train.py --pilot` run without `--out-dir`.** Expected: pilot records and checkpoints go to `outputs_pilot/`, never among the evaluation records in `outputs/`. Pinned in Task 2 (`test_pilot_defaults_to_the_pilot_directories`).

---

## File Structure

| file | responsibility |
|---|---|
| `experiments/079_judge_trained_deep/cells.py` | constants, paths, S construction from X's encoder, loading a trained S judge, proving and copying the pilot encoders (CLI `prepare-pilot-encoders`) |
| `experiments/079_judge_trained_deep/train.py` | value-iteration driver for S, the walk recorder, the training record |
| `experiments/079_judge_trained_deep/evaluate.py` | one J4V cell per call (J4V-S or J4V-X), Gate R rank cells for S, Gate M's X continuity cell |
| `experiments/079_judge_trained_deep/aggregate.py` | record checks, every gate, both claims, readings, pilot amendment numbers |
| `experiments/079_judge_trained_deep/launch079.ps1` | laptop launcher, one phase per call, amendment guard, memory sampling |
| `experiments/079_judge_trained_deep/chain079.ps1` | the full run after the amendment: cont, Gate 0(b), train, Gates L and E, eval, det, rank |
| `tests/experiments/test_exp079.py` | every test in this plan |

---

### Task 1: cells.py (paths, S construction, the pilot-encoder proof)

**Files:**
- Create: `experiments/079_judge_trained_deep/cells.py`
- Create: `tests/experiments/test_exp079.py`

**Interfaces:**
- Consumes: EXP-075's `cells.py` (`build_judge(seed, arm, sensory)`, `fresh_sensory(arm, seed)`, `load_pretrained(arm, seed, out_dir)`, `encoder_path(arm, seed, out_dir)`, `ckpt_dir(arm, seed, out_dir)`, `record_name(arm, seed)`, `check_seeds(seeds, pilot)`, `exclusion_set(seed, provider)`, `probe_set(seed, provider)`, `EVAL_SEEDS`, `PILOT_SEEDS`).
- Produces: module constants `ARM = "S"`, `BASE_ARM = "X"`, `MAX_LEN = 40`, `N_UPDATES = 4000`, `EVAL_SEEDS`, `PILOT_SEEDS`, `E75_OUT`, `E75_PILOT`, `E76_OUT`, `E78_OUT`, `OUT`, `PILOT_OUT`, `DRIFT_TOL`; the module object `c75`; functions `encoder_root(pilot: bool) -> Path`, `ckpt_dir(seed, out_dir=None) -> Path`, `record_name(seed) -> str`, `make_judge(seed, enc_root) -> vi.Judge`, `judge_updates(seed, judge_dir=None) -> int`, `load_trained(seed, judge_dir=None, required_updates=N_UPDATES) -> vi.Judge`, `drift(sensory_state, e0_state) -> float`, `pilot_encoder_matches(seed, e75_pilot_dir=E75_PILOT) -> tuple[bool, str]`, `prepare_pilot_encoders(e75_pilot_dir=E75_PILOT, dest_root=PILOT_OUT) -> list[str]`; aliases `check_seeds`, `exclusion_set`, `probe_set`.

- [ ] **Step 1: Write the failing tests**

Create `tests/experiments/test_exp079.py`:

```python
"""EXP-079: judge S, EXP-075's X recipe trained on walks of length 1 to 40."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.encoder_pretrain import save_encoder

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "079_judge_trained_deep"
E75 = REPO / "experiments" / "075_wide_region"
E76 = REPO / "experiments" / "076_deeper_tree"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp079_cells", EXP / "cells.py")


def _fake_x_pilot(d, seed, perturb=0.0):
    """EXP-075's pilot layout for one seed under `d`: encoder, X judge whose region has moved,
    and a training record whose encoder_drift is the true drift (plus `perturb`)."""
    enc = cells.c75.fresh_sensory("X", seed)
    path = cells.c75.encoder_path("X", seed, d)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder(enc, path)
    judge = cells.c75.build_judge(seed, "X", cells.c75.fresh_sensory("X", seed))
    with torch.no_grad():
        judge.sensory.fc1.weight.add_(0.01)
    ck = cells.c75.ckpt_dir("X", seed, d)
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(judge.state_dict(), ck / "judge.pt")
    e0 = cells.c75.load_pretrained("X", seed, d).state_dict()
    true = cells.drift(judge.sensory.state_dict(), e0)
    (d / cells.c75.record_name("X", seed)).write_text(
        json.dumps({"encoder_drift": true + perturb}))
    return true


def test_constants_are_the_spec_values():
    """Catches the walk length, update count, seeds or source folders drifting from spec
    sections 3 and 6."""
    assert cells.MAX_LEN == 40 and cells.N_UPDATES == 4000
    assert cells.ARM == "S" and cells.BASE_ARM == "X"
    assert cells.EVAL_SEEDS == tuple(range(12)) and cells.PILOT_SEEDS == (12, 13)
    assert cells.E75_OUT.name == "outputs" and cells.E75_OUT.parent.name == "075_wide_region"
    assert cells.E76_OUT.parent.name == "076_deeper_tree"
    assert cells.encoder_root(False) == cells.E75_OUT
    assert cells.encoder_root(True) == cells.PILOT_OUT
    assert cells.record_name(3) == "exp079_train_S_s3.json"
    assert cells.ckpt_dir(3, Path("x")) == Path("x") / "judge_S_s3"


def test_make_judge_is_x_built_from_the_committed_encoder():
    """Catches S built at another width, with another readout, from another encoder, or with a
    head not seeded like X's (spec section 3: the only difference is max_len)."""
    s = cells.make_judge(0, cells.E75_OUT)
    x = cells.c75.make_judge(0, "X", cells.E75_OUT)
    assert s.readout == "wide" and s.sensory.fc1.out_features == 512
    sd_s, sd_x = s.state_dict(), x.state_dict()
    assert sd_s.keys() == sd_x.keys()
    for k in sd_s:
        assert torch.equal(sd_s[k], sd_x[k]), k


def test_load_trained_refuses_a_partly_trained_or_missing_judge(tmp_path):
    """Catches an S judge evaluated before training finished (judge.pt exists from update 0)."""
    with pytest.raises(SystemExit, match="judge_S_s0"):
        cells.load_trained(0, tmp_path)
    ck = cells.ckpt_dir(0, tmp_path)
    ck.mkdir(parents=True)
    judge = cells.c75.build_judge(0, "X", cells.c75.fresh_sensory("X", 0))
    torch.save(judge.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": 3999}))
    with pytest.raises(SystemExit, match="3999"):
        cells.load_trained(0, tmp_path)
    (ck / "state.json").write_text(json.dumps({"update": 4000}))
    loaded = cells.load_trained(0, tmp_path)
    assert torch.equal(loaded.head[0].weight, judge.head[0].weight)
    assert cells.judge_updates(0, tmp_path) == 4000


def test_drift_is_exp075s_l2_over_the_encoder_tensors():
    """Catches a drift that is not EXP-075's (so the pilot proof would compare different
    quantities): one tensor moved by 0.01 everywhere gives 0.01 * sqrt(numel)."""
    a = {"w": torch.zeros(4, 25), "b": torch.zeros(3)}
    b = {"w": torch.full((4, 25), 0.01), "b": torch.zeros(3)}
    assert cells.drift(b, a) == pytest.approx(0.01 * 10.0, rel=1e-6)


def test_pilot_encoder_is_proved_only_when_the_drift_reproduces(tmp_path):
    """Catches a pilot encoder accepted without checking it is the one EXP-075's pilot record
    describes (spec section 6), and a missing file read as a match."""
    _fake_x_pilot(tmp_path, 12)
    ok, why = cells.pilot_encoder_matches(12, tmp_path)
    assert ok, why
    _fake_x_pilot(tmp_path, 13, perturb=1e-3)
    ok, why = cells.pilot_encoder_matches(13, tmp_path)
    assert not ok and "seed 13" in why
    ok, why = cells.pilot_encoder_matches(12, tmp_path / "nowhere")
    assert not ok and "missing" in why


def test_pilot_encoders_copy_all_or_nothing(tmp_path):
    """REVIEW FOCUS 3. Catches one proved encoder copied while the other failed, which would mix
    a copied encoder with a re-pretrained one in the same pilot."""
    src, dest = tmp_path / "e75", tmp_path / "e79"
    _fake_x_pilot(src, 12)
    _fake_x_pilot(src, 13, perturb=1e-3)
    with pytest.raises(SystemExit, match="pilot-pretrain"):
        cells.prepare_pilot_encoders(src, dest)
    assert not cells.c75.encoder_path("X", 12, dest).exists()
    _fake_x_pilot(src, 13)
    lines = cells.prepare_pilot_encoders(src, dest)
    assert len(lines) == 2
    for s in (12, 13):
        a = cells.c75.encoder_path("X", s, src).read_bytes()
        assert cells.c75.encoder_path("X", s, dest).read_bytes() == a
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (Bash tool `timeout: 600000`): `.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m "not slow"`
Expected: collection error, `FileNotFoundError` for `experiments/079_judge_trained_deep/cells.py`.

- [ ] **Step 3: Write `cells.py`**

```python
"""EXP-079 cells: judge S is EXP-075's X recipe trained on walks of length 1 to 40.

S starts each evaluation seed from EXP-075's committed pretrained X encoder and is built exactly
as X was (wide readout, hidden 512, head seeded by the seed). Only the walk length differs.
Pilot seeds 12 and 13 read their encoders from this experiment's outputs_pilot/encoders/, filled
either by copying EXP-075's laptop pilot encoders once each is proved to be the one its pilot
record describes, or by re-pretraining with EXP-075's pretrain.py.

Spec: docs/superpowers/specs/2026-10-18-exp079-judge-trained-where-scrambles-live-design.md,
sections 3, 4 and 6.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python experiments/079_judge_trained_deep/cells.py prepare-pilot-encoders
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c75 = _load("exp075_cells", REPO / "experiments" / "075_wide_region" / "cells.py")

from neuromorphic.training import value_iteration as vi  # noqa: E402

ARM = "S"
# S is built at X's width, from X's pretrained encoder, by X's construction.
BASE_ARM = "X"
# Spec section 3: the only lever. EXP-075's X trained on walks of length 1 to 14.
MAX_LEN = 40
N_UPDATES = 4000
EVAL_SEEDS = tuple(c75.EVAL_SEEDS)
PILOT_SEEDS = tuple(c75.PILOT_SEEDS)
E75_OUT = REPO / "experiments" / "075_wide_region" / "outputs"
E75_PILOT = REPO / "experiments" / "075_wide_region" / "outputs_pilot"
E76_OUT = REPO / "experiments" / "076_deeper_tree" / "outputs"
E78_OUT = REPO / "experiments" / "078_full_five_move_tree" / "outputs"
OUT = HERE / "outputs"
PILOT_OUT = HERE / "outputs_pilot"
# Same machine, same float code path: a genuine match reproduces the drift to rounding.
DRIFT_TOL = 1e-6

check_seeds = c75.check_seeds
exclusion_set = c75.exclusion_set
probe_set = c75.probe_set


def encoder_root(pilot: bool) -> Path:
    """Where a seed's pretrained X encoder lives (as `<root>/encoders/enc_X_s<seed>.pt`):
    EXP-075's committed outputs for evaluation seeds, this experiment's outputs_pilot for the
    pilot seeds."""
    return PILOT_OUT if pilot else E75_OUT


def ckpt_dir(seed, out_dir=None) -> Path:
    return (Path(out_dir) if out_dir is not None else OUT) / f"judge_S_s{seed}"


def record_name(seed) -> str:
    return f"exp079_train_S_s{seed}.json"


def make_judge(seed, enc_root) -> vi.Judge:
    """S before training: X's pretrained region for `seed` under a fresh wide head, seeded as
    X's was. A missing encoder raises SystemExit naming the file."""
    return c75.build_judge(seed, BASE_ARM, c75.load_pretrained(BASE_ARM, seed, enc_root))


def judge_updates(seed, judge_dir=None) -> int:
    path = ckpt_dir(seed, judge_dir) / "state.json"
    if not path.exists():
        raise SystemExit(f"missing {path} for S seed {seed}: cannot tell how far the judge "
                         f"trained; train it with train.py first")
    return int(json.loads(path.read_text())["update"])


def load_trained(seed, judge_dir=None, required_updates=N_UPDATES) -> vi.Judge:
    """A trained S judge from its checkpoint, refused unless training reached
    `required_updates` (judge.pt is written at update 0, so its existence proves nothing)."""
    path = ckpt_dir(seed, judge_dir) / "judge.pt"
    if not path.exists():
        raise SystemExit(f"missing judge checkpoint {path} for S seed {seed}; train it with "
                         f"train.py first")
    done = judge_updates(seed, judge_dir)
    if done != required_updates:
        raise SystemExit(f"{path.parent / 'state.json'}: S seed {seed} is at update {done}, "
                         f"evaluation needs {required_updates}; finish training first")
    judge = c75.build_judge(seed, BASE_ARM, c75.fresh_sensory(BASE_ARM, seed))
    judge.load_state_dict(torch.load(path, map_location="cpu"))
    judge.eval()
    return judge


def drift(sensory_state: dict, e0_state: dict) -> float:
    """EXP-075's encoder_drift: the L2 distance over the pretrained encoder's tensors."""
    return sum(float(((sensory_state[k] - e0_state[k]) ** 2).sum()) for k in e0_state) ** 0.5


def pilot_encoder_matches(seed, e75_pilot_dir=E75_PILOT) -> tuple[bool, str]:
    """Whether EXP-075's pilot X encoder for `seed` is the one its pilot record describes:
    recomputing the drift between the X pilot judge's region and that encoder must reproduce
    the record's encoder_drift within DRIFT_TOL."""
    d = Path(e75_pilot_dir)
    enc = c75.encoder_path(BASE_ARM, seed, d)
    jpt = c75.ckpt_dir(BASE_ARM, seed, d) / "judge.pt"
    rec = d / c75.record_name(BASE_ARM, seed)
    for p in (enc, jpt, rec):
        if not p.exists():
            return False, f"seed {seed}: missing {p}"
    e0 = c75.load_pretrained(BASE_ARM, seed, d).state_dict()
    judge = c75.build_judge(seed, BASE_ARM, c75.fresh_sensory(BASE_ARM, seed))
    judge.load_state_dict(torch.load(jpt, map_location="cpu"))
    got = drift(judge.sensory.state_dict(), e0)
    want = json.loads(rec.read_text())["encoder_drift"]
    if abs(got - want) > DRIFT_TOL:
        return False, f"seed {seed}: recomputed drift {got!r} against the record's {want!r}"
    return True, f"seed {seed}: drift {got!r} matches the record"


def prepare_pilot_encoders(e75_pilot_dir=E75_PILOT, dest_root=PILOT_OUT) -> list[str]:
    """Copy EXP-075's pilot X encoders to `dest_root/encoders/`, all or nothing: every seed is
    proved first, and one failure copies none (SystemExit naming each failure)."""
    proofs = [pilot_encoder_matches(s, e75_pilot_dir) for s in PILOT_SEEDS]
    bad = [why for ok, why in proofs if not ok]
    if bad:
        raise SystemExit("pilot encoders not proved: " + "; ".join(bad)
                         + ". Nothing was copied. Run the launcher's pilot-pretrain phase.")
    lines = []
    for s, (_, why) in zip(PILOT_SEEDS, proofs):
        dest = c75.encoder_path(BASE_ARM, s, dest_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c75.encoder_path(BASE_ARM, s, e75_pilot_dir), dest)
        lines.append(f"{why}; copied to {dest}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare-pilot-encoders",))
    ap.add_argument("--e75-pilot-dir", type=Path, default=E75_PILOT)
    ap.add_argument("--dest-root", type=Path, default=PILOT_OUT)
    args = ap.parse_args()
    for line in prepare_pilot_encoders(args.e75_pilot_dir, args.dest_root):
        print(line)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (Bash tool `timeout: 600000`): `.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m "not slow"`
Expected: 6 passed.

- [ ] **Step 5: Mutation check**

Temporarily change `prepare_pilot_encoders` to copy each proved seed inside the proof loop (before checking `bad`). Run the same command: `test_pilot_encoders_copy_all_or_nothing` must FAIL. Then change `DRIFT_TOL = 1e-6` to `DRIFT_TOL = 1.0`: `test_pilot_encoder_is_proved_only_when_the_drift_reproduces` must FAIL. Restore both and confirm `git diff experiments/079_judge_trained_deep/cells.py` shows no mutation left. Record both results in the report.

- [ ] **Step 6: Commit**

```bash
git add experiments/079_judge_trained_deep/cells.py tests/experiments/test_exp079.py
git commit -m "EXP-079: cells (S built from X's committed encoder, trained-judge loading, the pilot-encoder proof)"
```

---

### Task 2: train.py (the S driver and the walk recorder)

**Files:**
- Create: `experiments/079_judge_trained_deep/train.py`
- Modify: `tests/experiments/test_exp079.py` (append)

**Interfaces:**
- Consumes: Task 1's `cells` (`make_judge`, `ckpt_dir`, `record_name`, `encoder_root`, `exclusion_set`, `probe_set`, `check_seeds`, `drift`, `c75`, `OUT`, `PILOT_OUT`, `EVAL_SEEDS`); `neuromorphic.training.value_iteration` (`train_judge`, `random_walk_states`); EXP-074's `gate_l.leaf_rank_margin(judge, probe, distance, seed, n_actions)`.
- Produces: `VI_ARM = "W"`; class `WalkRecorder(walk_fn, distance, keep)` with `__call__(n, max_len, rng, n_actions, exclude, **kw)`, attributes `max_lens: list`, `calls: int`, method `summary() -> {"walk_max_lens": list[int], "train_hist": {str(d): float}, "train_hist_updates": int}`; context manager `recording(distance, keep)`; `train_core(judge, *, max_len, distance, n_updates, batch, exclude, probe, seed, sync_every, probe_every, draws, ckpt_dir, log=print) -> (dict, dict)`; `run(seed, n_updates, batch, sync_every, probe_every, draws, max_len, out_dir, enc_root, provider=None) -> dict`; `pending_jobs(seeds, out_dir) -> list[int]`; `build_parser()`; `resolve_dirs(args) -> (out_dir, enc_root)`. The training record has every key of `train_judge`'s result plus `walk_max_lens`, `train_hist`, `train_hist_updates`, `arm` ("S"), `seed`, `hidden` (512), `readout`, `n_updates`, `resumed_from`, `batch`, `max_len`, `n_actions`, `sync_every`, `probe_every`, `draws`, `n_exclude`, `n_probe`, `encoder_root`, `encoder_drift`, `gate_l`, `git_commit`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp079.py`:

```python
train = _load("exp079_train", EXP / "train.py")


def _tiny_judge(seed=0):
    return cells.c75.build_judge(seed, "X", cells.c75.fresh_sensory("X", seed))


def _tiny_probe():
    s1 = apply_move(SOLVED, 0)
    return [(SOLVED, 0), (s1, 1), (apply_move(s1, 2), 2)]


def _core_kwargs(tmp, max_len, n_updates=2):
    return dict(max_len=max_len, distance=lambda s: 0, n_updates=n_updates, batch=3,
                exclude=set(), probe=_tiny_probe(), seed=0, sync_every=1, probe_every=1,
                draws=1, ckpt_dir=tmp, log=lambda m: None)


def test_recorder_returns_the_walks_unchanged_and_keeps_the_last_interval():
    """Catches a recorder that alters what training sees, or a histogram over every update
    instead of the last probe interval (spec section 4)."""
    batches = iter([[0, 0], [1, 2], [2, 2]])
    rec = train.WalkRecorder(lambda n, m, rng, a, ex: next(batches), lambda s: s, keep=2)
    assert rec(2, 40, None, 6, set()) == [0, 0]
    assert rec(2, 40, None, 6, set()) == [1, 2]
    assert rec(2, 17, None, 6, set()) == [2, 2]
    out = rec.summary()
    assert out["walk_max_lens"] == [17, 40]
    assert out["train_hist"] == {"1": 0.25, "2": 0.75}
    assert out["train_hist_updates"] == 2 and rec.calls == 3


def test_recorder_reports_how_many_updates_it_covers():
    """REVIEW FOCUS 1. Catches a resumed session's short histogram passed off as a full probe
    interval: one call under keep=250 reports 1, not 250."""
    rec = train.WalkRecorder(lambda n, m, rng, a, ex: [5], lambda s: s, keep=250)
    rec(1, 40, None, 6, set())
    assert rec.summary()["train_hist_updates"] == 1


@pytest.mark.parametrize("max_len", [40, 17])
def test_max_len_reaches_random_walk_states(tmp_path, max_len):
    """THE SPEC SECTION 4 TEST. Catches a driver whose walk length never reaches the library:
    two different values must each arrive, so a hardcoded 14 (or 40) fails one case."""
    _, walks = train.train_core(_tiny_judge(), **_core_kwargs(tmp_path, max_len))
    assert walks["walk_max_lens"] == [max_len]
    assert walks["train_hist_updates"] == 1


def test_recorder_does_not_change_training(tmp_path):
    """Catches the recorder perturbing the RNG or the walks: a judge trained through train_core
    equals one trained by train_judge directly, tensor for tensor."""
    a, b = _tiny_judge(), _tiny_judge()
    train.train_core(a, **_core_kwargs(tmp_path / "a", 40))
    kw = _core_kwargs(tmp_path / "b", 40)
    kw.pop("distance")
    vi.train_judge(b, train.VI_ARM, n_actions=N_ACTIONS, **kw)
    sa, sb = a.state_dict(), b.state_dict()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_recording_restores_the_library_function_on_error():
    """Catches a patched random_walk_states left behind after a failed run, which would record
    (and slow) every later call in the process."""
    real = vi.random_walk_states
    with pytest.raises(RuntimeError):
        with train.recording(lambda s: 0, 5):
            assert vi.random_walk_states is not real
            raise RuntimeError("boom")
    assert vi.random_walk_states is real


def test_parser_requires_max_len_and_keeps_exp075s_settings():
    """Catches a driver that falls back to a default walk length, and settings drifting from
    EXP-075's X (spec section 3)."""
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["--n-updates", "1"])
    a = train.build_parser().parse_args(["--n-updates", "1", "--max-len", "40"])
    assert (a.batch, a.sync_every, a.probe_every, a.draws) == (1000, 100, 250, 1)
    assert a.max_len == 40 and a.seeds == list(range(12)) and train.VI_ARM == "W"


def test_pilot_defaults_to_the_pilot_directories():
    """REVIEW FOCUS 5. Catches pilot records written among the evaluation records, and the
    pilot reading EXP-075's evaluation encoders (seeds 12 and 13 have none there)."""
    a = train.build_parser().parse_args(["--n-updates", "1", "--max-len", "40", "--pilot",
                                         "--seeds", "12", "13"])
    assert train.resolve_dirs(a) == (cells.PILOT_OUT, cells.PILOT_OUT)
    b = train.build_parser().parse_args(["--n-updates", "1", "--max-len", "40"])
    assert train.resolve_dirs(b) == (cells.OUT, cells.E75_OUT)


def test_train_refuses_a_missing_encoder_before_building_anything(tmp_path):
    """Catches training on a random region when the encoder is absent, and the check running
    only after the 65 s BFS build."""
    import time
    t0 = time.time()
    with pytest.raises(SystemExit, match="enc_X_s12.pt"):
        train.run(12, 1, 8, 1, 1, 1, 40, tmp_path / "out", tmp_path)
    assert time.time() - t0 < 20


def test_train_relaunch_skips_runs_whose_record_exists(tmp_path):
    """Catches a relaunch that resubmits finished runs."""
    (tmp_path / cells.record_name(13)).write_text("{}")
    assert train.pending_jobs([12, 13], tmp_path) == [12]


@pytest.mark.slow
def test_run_writes_the_record_with_its_walk_evidence(tmp_path):
    """Catches a record missing the walk length evidence, the histogram, Gate L or the drift
    Gate E reads. Slow: builds the full BFS table, as the real driver does."""
    path = cells.c75.encoder_path("X", 12, tmp_path)
    path.parent.mkdir(parents=True)
    save_encoder(cells.c75.fresh_sensory("X", 12), path)
    rec = train.run(12, 2, 8, 1, 1, 1, 40, tmp_path / "out", tmp_path)
    on_disk = json.loads((tmp_path / "out" / cells.record_name(12)).read_text())
    assert on_disk["arm"] == "S" and on_disk["max_len"] == 40
    assert on_disk["walk_max_lens"] == [40] and on_disk["train_hist_updates"] == 1
    assert sum(on_disk["train_hist"].values()) == pytest.approx(1.0)
    assert on_disk["readout"] == "wide" and on_disk["hidden"] == 512
    assert on_disk["gate_l"]["n"] == 250 and on_disk["encoder_drift"] > 0.0
    assert (tmp_path / "out" / "judge_S_s12" / "judge.pt").exists()
    assert rec["gate_l"] == on_disk["gate_l"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (Bash tool `timeout: 600000`): `.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m "not slow"`
Expected: collection error, `FileNotFoundError` for `train.py`.

- [ ] **Step 3: Write `train.py`**

```python
"""EXP-079 training driver: judge S by EXP-075's value iteration on walks of length 1 to
`--max-len` (40 in every launched run), then Gate L.

Identical to EXP-075's X driver except for the walk length, where the starting encoder is read
from, and two diagnostics: the walk lengths `random_walk_states` was actually called with, and
the realised training-distance histogram over the last `probe_every` updates, counted from the
BFS instrument as the walks are drawn. Neither reaches the judge: the recorder returns the
library's walks unchanged, and the full BFS table is built only for the exclusion and probe sets,
Gate L and this histogram, never for training.

Spec: docs/superpowers/specs/2026-10-18-exp079-judge-trained-where-scrambles-live-design.md,
sections 3, 4 and 6.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/079_judge_trained_deep/train.py --pilot --seeds 12 13 \
        --n-updates 4000 --max-len 40 --workers 2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp079_cells", HERE / "cells.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

torch.set_num_threads(1)

# EXP-075's X optimizer: encoder lr 1e-4, head lr 1e-3.
VI_ARM = "W"


class WalkRecorder:
    """Wraps `random_walk_states`: returns its walks unchanged, notes every max_len it was
    called with, and keeps a distance histogram of the last `keep` calls (one call per update)."""

    def __init__(self, walk_fn, distance, keep):
        self.walk_fn = walk_fn
        self.distance = distance
        self.recent = deque(maxlen=keep)
        self.max_lens = []
        self.calls = 0

    def __call__(self, n, max_len, rng, n_actions, exclude, **kw):
        states = self.walk_fn(n, max_len, rng, n_actions, exclude, **kw)
        if max_len not in self.max_lens:
            self.max_lens.append(max_len)
        self.recent.append(Counter(self.distance(s) for s in states))
        self.calls += 1
        return states

    def summary(self) -> dict:
        total = Counter()
        for c in self.recent:
            total.update(c)
        n = sum(total.values())
        return {"walk_max_lens": sorted(self.max_lens),
                "train_hist": {str(d): total[d] / n for d in sorted(total)} if n else {},
                "train_hist_updates": len(self.recent)}


@contextmanager
def recording(distance, keep):
    """Route value_iteration's own `random_walk_states` lookups through a WalkRecorder, and
    restore the library function however the block exits."""
    real = vi.random_walk_states
    rec = WalkRecorder(real, distance, keep)
    vi.random_walk_states = rec
    try:
        yield rec
    finally:
        vi.random_walk_states = real


def train_core(judge, *, max_len, distance, n_updates, batch, exclude, probe, seed, sync_every,
               probe_every, draws, ckpt_dir, log=print):
    """`vi.train_judge` with S's settings, observed by a WalkRecorder. Returns train_judge's
    result and the recorder's summary."""
    with recording(distance, probe_every) as rec:
        result = vi.train_judge(
            judge, VI_ARM, n_updates=n_updates, batch=batch, max_len=max_len,
            n_actions=N_ACTIONS, exclude=exclude, probe=probe, seed=seed,
            sync_every=sync_every, probe_every=probe_every, draws=draws, ckpt_dir=ckpt_dir,
            log=log)
    return result, rec.summary()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _resumed_from(ckpt_dir) -> int:
    """Updates already done when this session starts (0 for a fresh run). train_judge's wall_s
    covers only the session that wrote the record, so s/update needs this."""
    path = Path(ckpt_dir) / "state.json"
    return int(json.loads(path.read_text())["update"]) if path.exists() else 0


def _rel(path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def run(seed, n_updates, batch, sync_every, probe_every, draws, max_len, out_dir, enc_root,
        provider=None) -> dict:
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = cells.make_judge(seed, enc_root)  # first: a missing encoder fails before the BFS
    if provider is None:
        provider = ExactBFSDistance(max_depth=None)
    excl = cells.exclusion_set(seed, provider)
    probe = cells.probe_set(seed, provider)
    ck = cells.ckpt_dir(seed, out_dir)
    resumed_from = _resumed_from(ck)
    result, walks = train_core(
        judge, max_len=max_len, distance=provider.distance, n_updates=n_updates, batch=batch,
        exclude=excl, probe=probe, seed=seed, sync_every=sync_every, probe_every=probe_every,
        draws=draws, ckpt_dir=ck)
    judge.eval()
    e0 = cells.c75.load_pretrained(cells.BASE_ARM, seed, enc_root).state_dict()
    rec = {
        **result, **walks,
        "arm": cells.ARM, "seed": seed, "hidden": cells.c75.HIDDEN[cells.BASE_ARM],
        "readout": judge.readout, "n_updates": n_updates, "resumed_from": resumed_from,
        "batch": batch, "max_len": max_len, "n_actions": N_ACTIONS, "sync_every": sync_every,
        "probe_every": probe_every, "draws": draws, "n_exclude": len(excl),
        "n_probe": len(probe), "encoder_root": _rel(enc_root),
        "encoder_drift": cells.drift(judge.sensory.state_dict(), e0),
        "gate_l": gate_l.leaf_rank_margin(judge, probe, provider.distance, seed, N_ACTIONS),
        "git_commit": _git_commit(),
    }
    (out_dir / cells.record_name(seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def pending_jobs(seeds, out_dir) -> list[int]:
    return [s for s in seeds if not (Path(out_dir) / cells.record_name(s)).exists()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--n-updates", type=int, required=True)
    # Required, with no default: a run must state its walk length (spec section 4).
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--sync-every", type=int, default=100)
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="outputs/ (outputs_pilot/ with --pilot) when omitted")
    ap.add_argument("--encoder-root", type=Path, default=None,
                    help="EXP-075's outputs/ (outputs_pilot/ with --pilot) when omitted")
    return ap


def resolve_dirs(args) -> tuple[Path, Path]:
    out = args.out_dir if args.out_dir is not None else (
        cells.PILOT_OUT if args.pilot else cells.OUT)
    enc = args.encoder_root if args.encoder_root is not None else cells.encoder_root(args.pilot)
    return out, enc


def main() -> None:
    args = build_parser().parse_args()
    cells.check_seeds(args.seeds, args.pilot)
    out_dir, enc_root = resolve_dirs(args)
    jobs = pending_jobs(args.seeds, out_dir)
    print(f"EXP-079: {len(jobs)} S training runs, {args.workers} workers, pilot={args.pilot}, "
          f"n_updates={args.n_updates}, max_len={args.max_len}, out {out_dir}, "
          f"encoders {enc_root}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, s, args.n_updates, args.batch, args.sync_every,
                            args.probe_every, args.draws, args.max_len, out_dir, enc_root): s
                for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            loss = "None" if r["loss_last"] is None else format(r["loss_last"], ".6f")
            print(f"  {i}/{len(jobs)}  S seed {r['seed']}  updates {r['updates']}  "
                  f"loss_last {loss}  wall_s {r['wall_s']:.1f}  "
                  f"encoder_drift {r['encoder_drift']:.6f}  "
                  f"gate_l margin {r['gate_l']['margin']:.4f}  "
                  f"walk_max_lens {r['walk_max_lens']}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (Bash tool `timeout: 600000`): `.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m "not slow"`
Expected: all non-slow tests pass (Task 1's 6 plus this task's 10 = 16 passed, 1 deselected).

Then the slow test alone (Bash tool `timeout: 600000`; it builds the full BFS table, about 70 s, plus Gate L):
`.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m slow`
Expected: 1 passed.

- [ ] **Step 5: Mutation check**

In `train_core`, replace `max_len=max_len` in the `vi.train_judge` call with `max_len=14`. Run the non-slow command: both `test_max_len_reaches_random_walk_states` cases must FAIL. Then in `WalkRecorder.__call__` change `return states` to `return states[::-1]`: `test_recorder_returns_the_walks_unchanged_and_keeps_the_last_interval` and `test_recorder_does_not_change_training` must FAIL. Restore both, confirm `git diff experiments/079_judge_trained_deep/train.py` is clean of the mutation, and record the results in the report.

- [ ] **Step 6: Commit**

```bash
git add experiments/079_judge_trained_deep/train.py tests/experiments/test_exp079.py
git commit -m "EXP-079: training driver for S (max_len required and recorded, walk recorder with the realised distance histogram)"
```

---

### Task 3: evaluate.py (J4V-S and J4V-X cells, Gate R rank cells)

**Files:**
- Create: `experiments/079_judge_trained_deep/evaluate.py`
- Modify: `tests/experiments/test_exp079.py` (append)

**Interfaces:**
- Consumes: Task 1's `cells` (`load_trained`, `judge_updates`, `ckpt_dir`, `E75_OUT`, `OUT`); EXP-075's `evaluate.py` (`JudgeCritic`, `load_cell(depth, seed) -> (agent, head, states, train_seed)`, `load_judge(train_arm, seed, judge_dir)`, `judge_updates(train_arm, seed, judge_dir)`); `neuromorphic.training.lookahead` (`evaluate_lookahead`, `imag_seed_for`, `imagined_values`, `tree_levels`).
- Produces: `ARMS = ("J4V-S", "J4V-X")`, `K = 4`, `DEPTHS = (9, 11)`, `record_name(arm, depth, seed) -> "exp079_<arm>_d<depth>_s<seed>.json"`, `rank_record_name(depth, seed) -> "exp079_rank_J-S4_d<depth>_s<seed>.json"`, `load_judge(arm, seed, judge_dir=None)`, `run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict`, `rank_cell(depth, seed, out_dir, judge_dir=None, limit_states=None, provider=None) -> dict`. `gatel_x_record_name(seed) -> "exp079_gatel_X_s<seed>.json"`, `gatel_x_cell(seed, out_dir, provider=None) -> dict` (Gate M's continuity cell: keys `kind` "gatel-X", `seed`, `gate_l`, `judge_updates`, `wall_s`, `git_commit`). An evaluation record holds `evaluate_lookahead`'s fields plus `depth`, `seed`, `arm`, `judge` ("S" or "X"), `fired4`, `wall_s`, `git_commit`, `limit_states`, `judge_updates`. A rank record holds `kind` ("J-S4"), `k`, `depth`, `seed`, `n`, `hit`, `chance`, `limit_states`, `judge_updates`, `wall_s`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp079.py`:

```python
evaluate = _load("exp079_evaluate", EXP / "evaluate.py")
e76 = _load("exp076_evaluate_for_079_tests", E76 / "evaluate.py")
OUTCOME = ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
           "greedy_modal_action_frac", "goal_fired_frac", "fallback_frac", "first_fire_len",
           "judge_evals")


def _fake_s_ckpt(jdir, seed, update=4000):
    """An untrained S-shaped judge saved as a finished checkpoint, with its head moved so it
    differs from X's committed judge."""
    ck = cells.ckpt_dir(seed, jdir)
    ck.mkdir(parents=True)
    judge = cells.c75.build_judge(seed, "X", cells.c75.fresh_sensory("X", seed))
    with torch.no_grad():
        judge.head[0].weight.mul_(-1.0)
    torch.save(judge.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": update}))
    return judge


def test_arms_and_constants_are_the_spec():
    """Catches an arm list, tree depth or depth list drifting from spec section 3."""
    assert evaluate.ARMS == ("J4V-S", "J4V-X") and evaluate.K == 4
    assert evaluate.DEPTHS == (9, 11)
    assert evaluate.record_name("J4V-S", 11, 3) == "exp079_J4V-S_d11_s3.json"
    assert evaluate.rank_record_name(9, 3) == "exp079_rank_J-S4_d9_s3.json"


def test_j4v_x_here_equals_exp076s_harness_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH. Catches this harness drifting from EXP-076's: J4V-X through it
    equals EXP-076's own run_cell on the same states, committed judge, in every outcome field."""
    mine = evaluate.run_cell("J4V-X", 9, 0, tmp_path / "a", limit_states=2)
    ref = e76.run_cell("J4V-X", 9, 0, tmp_path / "b", limit_states=2)
    for f in OUTCOME:
        assert mine[f] == ref[f], f
    assert mine["k"] == 4 and mine["k_goal"] == 4 and mine["judge"] == "X"


def test_each_arm_loads_its_own_judge(tmp_path):
    """Catches J4V-S silently running X's judge (the arms would then be one arm measured twice)
    and J4V-X reading anything but EXP-075's committed judge."""
    saved = _fake_s_ckpt(tmp_path, 0)
    s = evaluate.load_judge("J4V-S", 0, tmp_path)
    assert torch.equal(s.head[0].weight, saved.head[0].weight)
    x = evaluate.load_judge("J4V-X", 0)
    ref = torch.load(cells.E75_OUT / "judge_X_s0" / "judge.pt", map_location="cpu")
    assert torch.equal(x.head[0].weight, ref["head.0.weight"])
    assert not torch.equal(s.head[0].weight, x.head[0].weight)


def test_j4v_s_cell_writes_its_record(tmp_path):
    """Catches an S record that does not say which judge and tree it ran, so the aggregator
    could not refuse a mislabelled one."""
    _fake_s_ckpt(tmp_path / "j", 0)
    rec = evaluate.run_cell("J4V-S", 7, 0, tmp_path / "o", limit_states=1,
                            judge_dir=tmp_path / "j")
    assert rec["arm"] == "J4V-S" and rec["judge"] == "S"
    assert rec["k"] == 4 and rec["k_goal"] == 4 and rec["mode"] == "C"
    assert rec["no_revisit"] is True and rec["judge_updates"] == 4000
    assert rec["limit_states"] == 1 and rec["fired4"] == rec["first_fire_len"].get("4", 0)
    on_disk = json.loads((tmp_path / "o" / "exp079_J4V-S_d7_s0.json").read_text())
    assert on_disk["judge"] == "S"


def test_partly_trained_s_judge_is_refused_by_evaluation(tmp_path):
    """Catches a cell evaluated from an S judge that stopped short of 4000 updates."""
    _fake_s_ckpt(tmp_path / "j", 0, update=3750)
    with pytest.raises(SystemExit, match="3750"):
        evaluate.run_cell("J4V-S", 7, 0, tmp_path / "o", limit_states=1,
                          judge_dir=tmp_path / "j")


def test_unknown_arm_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="unknown arm"):
        evaluate.run_cell("J3V-S", 7, 0, tmp_path)


def test_rank_cell_ranks_four_move_leaves_with_s(tmp_path, monkeypatch):
    """Catches Gate R measured on 3-move leaves, with X's judge, or without its limit."""
    _fake_s_ckpt(tmp_path / "j", 0)
    seen, judges = [], []
    real_levels, real_load = evaluate.tree_levels, evaluate.load_judge

    def spy_levels(state, k, n):
        seen.append(k)
        return real_levels(state, k, n)

    def spy_load(arm, seed, judge_dir=None):
        judges.append(arm)
        return real_load(arm, seed, judge_dir)

    monkeypatch.setattr(evaluate, "tree_levels", spy_levels)
    monkeypatch.setattr(evaluate, "load_judge", spy_load)
    rec = evaluate.rank_cell(7, 0, tmp_path / "o", judge_dir=tmp_path / "j", limit_states=2,
                             provider=ExactBFSDistance(max_depth=11))
    assert set(seen) == {4} and judges == ["J4V-S"]
    assert rec["kind"] == "J-S4" and rec["k"] == 4 and rec["n"] == 2
    assert rec["limit_states"] == 2 and rec["judge_updates"] == 4000
    assert 0.0 < rec["chance"] < 1.0 and rec["hit"] in (0.0, 0.5, 1.0)
    assert (tmp_path / "o" / "exp079_rank_J-S4_d7_s0.json").exists()


@pytest.mark.slow
def test_gatel_x_cell_reproduces_the_committed_gate_l(tmp_path):
    """GATE M CONTINUITY. Catches today's Gate L code (tree_levels, the probe set, the judge
    load) measuring X differently from EXP-075, which would make Gate M compare S with a number
    the current code cannot produce. Measured 2026-10-19 on the VPS: equal, about 100 s. Slow:
    builds the full BFS table."""
    rec = evaluate.gatel_x_cell(0, tmp_path)
    ref = json.loads((cells.E75_OUT / "exp075_train_X_s0.json").read_text())["gate_l"]
    assert rec["gate_l"] == ref
    assert rec["kind"] == "gatel-X" and rec["judge_updates"] == 4000
    assert (tmp_path / "exp079_gatel_X_s0.json").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (Bash tool `timeout: 600000`): `.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m "not slow"`
Expected: collection error, `FileNotFoundError` for `evaluate.py`.

- [ ] **Step 3: Write `evaluate.py`**

```python
"""EXP-079 evaluation: EXP-076's full 4-move tree driven by judge S or by EXP-075's X.

J4V-S loads this experiment's trained S judge. J4V-X loads EXP-075's committed X judge and is
re-run here only for Gate 0(b); its claim records are EXP-076's. Held-out sets, budgets, streams
and the no-revisit rule are EXP-075's (its `load_cell` and `JudgeCritic` are used directly), and
the search is EXP-076's J4V unchanged (mode C, k 4, goal check at 4).

Spec: docs/superpowers/specs/2026-10-18-exp079-judge-trained-where-scrambles-live-design.md,
sections 3 and 7.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/079_judge_trained_deep/evaluate.py --arm J4V-S --depth 11 --seed 0
    .venv/bin/python -u experiments/079_judge_trained_deep/evaluate.py --rank --depth 11 --seed 0
    .venv/bin/python -u experiments/079_judge_trained_deep/evaluate.py --gatel-x --seed 0
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


cells = _load("exp079_cells", HERE / "cells.py")
e75 = _load("exp075_evaluate", REPO / "experiments" / "075_wide_region" / "evaluate.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training.lookahead import (  # noqa: E402
    evaluate_lookahead, imag_seed_for, imagined_values, tree_levels,
)

torch.set_num_threads(1)

ARMS = ("J4V-S", "J4V-X")
K = 4
DEPTHS = (9, 11)


def record_name(arm: str, depth: int, seed: int) -> str:
    return f"exp079_{arm}_d{depth}_s{seed}.json"


def rank_record_name(depth: int, seed: int) -> str:
    return f"exp079_rank_J-S4_d{depth}_s{seed}.json"


def gatel_x_record_name(seed: int) -> str:
    return f"exp079_gatel_X_s{seed}.json"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _check_arm(arm) -> None:
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; valid: {list(ARMS)}")


def load_judge(arm, seed, judge_dir=None):
    """J4V-S: this experiment's S checkpoint (`judge_dir`, default outputs/). J4V-X: EXP-075's
    committed X judge. Both refuse a judge short of 4000 updates."""
    _check_arm(arm)
    if arm == "J4V-S":
        return cells.load_trained(seed, judge_dir)
    return e75.load_judge("X", seed, cells.E75_OUT)


def _judge_updates(arm, seed, judge_dir=None) -> int:
    if arm == "J4V-S":
        return cells.judge_updates(seed, judge_dir)
    return e75.judge_updates("X", seed, cells.E75_OUT)


def _write(out_dir, name, rec) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")


def run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None) -> dict:
    torch.set_num_threads(1)
    _check_arm(arm)
    t0 = time.time()
    critic = e75.JudgeCritic(load_judge(arm, seed, judge_dir))
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=K, k_goal=K,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    rec = {**res, "depth": depth, "seed": seed, "arm": arm, "judge": arm[-1],
           "fired4": res["first_fire_len"].get("4", 0),
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit(),
           "limit_states": limit_states,
           "judge_updates": _judge_updates(arm, seed, judge_dir)}
    _write(out_dir, record_name(arm, depth, seed), rec)
    return rec


def rank_cell(depth, seed, out_dir, judge_dir=None, limit_states=None, provider=None) -> dict:
    """Gate R at k 4 for S (reported, not voiding; spec section 7): per-seed mean hit of 'the
    lowest-J 4-move leaf is closer to solved than the root', on evaluation held-out states. BFS
    is the yardstick only; unbounded by default because depth + 4 passes the 2x2's diameter."""
    t0 = time.time()
    torch.set_num_threads(1)
    critic = e75.JudgeCritic(load_judge("J4V-S", seed, judge_dir))
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    if provider is None:
        provider = ExactBFSDistance(max_depth=None)
    n_actions = head.head.out_features
    hits, chances = [], []
    for i, s in enumerate(states):
        g = torch.Generator().manual_seed(imag_seed_for(train_seed, i, 0))
        levels = tree_levels(s, K, n_actions)
        scores = imagined_values(agent, critic, levels[K], generator=g)
        d = provider.distance(s)
        leaf_d = [provider.distance(x) for x in levels[K]]
        hits.append(int(leaf_d[int(scores.argmax())] < d))
        chances.append(sum(x < d for x in leaf_d) / len(leaf_d))
    rec = {"kind": "J-S4", "k": K, "depth": depth, "seed": seed, "n": len(states),
           "hit": st.mean(hits), "chance": st.mean(chances), "limit_states": limit_states,
           "judge_updates": cells.judge_updates(seed, judge_dir),
           "wall_s": round(time.time() - t0, 1)}
    _write(out_dir, rank_record_name(depth, seed), rec)
    return rec


def gatel_x_cell(seed, out_dir, provider=None) -> dict:
    """Gate M's continuity cell: EXP-075's committed X judge for `seed`, re-measured by today's
    Gate L code on its own probe set. Gate M pairs S's margins with X's PUBLISHED ones, so this
    must reproduce the committed training record's gate_l exactly (CLAUDE.md: re-evaluate the
    baseline, never trust a published per-seed value unchecked)."""
    torch.set_num_threads(1)
    t0 = time.time()
    judge = load_judge("J4V-X", seed)
    if provider is None:
        provider = ExactBFSDistance(max_depth=None)
    probe = cells.probe_set(seed, provider)
    rec = {"kind": "gatel-X", "seed": seed,
           "gate_l": gate_l.leaf_rank_margin(judge, probe, provider.distance, seed, N_ACTIONS),
           "judge_updates": _judge_updates("J4V-X", seed),
           "wall_s": round(time.time() - t0, 1), "git_commit": _git_commit()}
    _write(out_dir, gatel_x_record_name(seed), rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--gatel-x", action="store_true",
                    help="Gate M's continuity cell: re-measure X's Gate L for --seed")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--judge-dir", type=Path, default=None,
                    help="where judge_S_s<seed>/ lives; this experiment's outputs/ when omitted")
    ap.add_argument("--limit-states", type=int, default=None)
    args = ap.parse_args()
    if (args.arm is not None) + args.rank + args.gatel_x != 1:
        raise SystemExit("pass exactly one of --arm, --rank or --gatel-x")
    if args.gatel_x:
        r = gatel_x_cell(args.seed, args.out_dir)
        print(f"gatel-X s{r['seed']} margin {r['gate_l']['margin']:.6f} wall {r['wall_s']}s")
        return
    if args.depth is None:
        raise SystemExit("--depth is required with --arm or --rank")
    if args.arm:
        r = run_cell(args.arm, args.depth, args.seed, args.out_dir, args.limit_states,
                     args.judge_dir)
        print(f"{r['arm']} d{r['depth']} s{r['seed']} solved {r['solved']}/{r['n']} "
              f"fired4 {r['fired4']} wall {r['wall_s']}s")
    else:
        r = rank_cell(args.depth, args.seed, args.out_dir, args.judge_dir, args.limit_states)
        print(f"J-S4 d{r['depth']} s{r['seed']} hit {r['hit']:.4f} chance {r['chance']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (Bash tool `timeout: 600000`): `.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m "not slow"`
Expected: 23 passed, 2 deselected. The two-state J4V-X slice runs twice and is the slowest test here.

Then the new slow test alone (Bash tool `timeout: 600000`; full BFS plus 250 probe states, about 100 s):
`.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m slow -k gatel`
Expected: 1 passed.

- [ ] **Step 5: Mutation check**

In `load_judge`, make the `J4V-S` branch return `e75.load_judge("X", seed, cells.E75_OUT)`. Run the command: `test_each_arm_loads_its_own_judge` must FAIL. Then change `k_goal=K` to `k_goal=None` in `run_cell`: `test_j4v_x_here_equals_exp076s_harness_on_a_slice` or `test_j4v_s_cell_writes_its_record` must FAIL (report which). Restore, confirm `git diff experiments/079_judge_trained_deep/evaluate.py` is clean of the mutation, and record the results.

- [ ] **Step 6: Commit**

```bash
git add experiments/079_judge_trained_deep/evaluate.py tests/experiments/test_exp079.py
git commit -m "EXP-079: evaluation cells (J4V-S on S's judge, J4V-X for Gate 0(b)), Gate R rank cells at k 4, and Gate M's X continuity cell"
```

---

### Task 4: aggregate.py (gates, claims, readings, the pilot amendment numbers)

**Files:**
- Create: `experiments/079_judge_trained_deep/aggregate.py`
- Modify: `tests/experiments/test_exp079.py` (append)

**Interfaces:**
- Consumes: Task 1's `cells` (`EVAL_SEEDS`, `PILOT_SEEDS`, `PILOT_OUT`, `E75_OUT`, `E75_PILOT`, `E76_OUT`, `E78_OUT`, `record_name`, `c75`); Task 3's `evaluate` (`ARMS`, `DEPTHS`, `K`, `record_name`, `rank_record_name`, `gatel_x_record_name`); EXP-070's aggregate (`one_sided_p(diffs)`, `gate1_verdict(a_mean, b_mean)`); EXP-071's (`gate0b_verdict(recs, ref)`, `gate_r_verdict(rows, hit_key, chance_key) -> (bool, p)`); EXP-072's (`pair_diffs(a, b)`); EXP-074's (`gate_l_verdict(margins, threshold)`); EXP-075's (`GATE_L_THRESHOLD`, `seconds_per_update(rec)`).
- Produces (all module level): constants `SEEDS`, `DEPTHS`, `ALPHA = 0.05 / 2`, `SMALL = 0.01`, `GATE_M_ALPHA = 0.05`, `M_BAND = (9, 11)`, `X_PILOT_GATE_L`, `TRAIN_SETTINGS`, `DET_IGNORED`, `AMENDMENT = None`, `GATE_L_THRESHOLD = None`; functions `claim_verdict`, `contrast`, `check_train_record(rec, name)`, `margin_band(gate_l, lo=9, hi=11)`, `gate_m(s_recs, x_recs) -> dict`, `pilot_amendment(pilot_dir=cells.PILOT_OUT) -> dict`, `require_amendment()`, `gate_m_continuity(out) -> str`, `stage_train(out) -> dict` (keys `l`, `e`, `m`, `m_cont`, `recs`), `format_stage_train(res) -> list[str]`, `check_eval_record(rec, path)`, `load_records(out_dir) -> dict`, `rates_of(recs)`, `stage_cont(out) -> str`, `format_stage_cont(verdict) -> str`, `gate0a(out_dir, det_dir) -> str`, `primary_verdicts(rates, gate_ok) -> dict`, `readings(full, gate_m_above) -> list[str]`, `main()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp079.py`:

```python
agg = _load("exp079_aggregate", EXP / "aggregate.py")


def _x_train(seed, pilot=False):
    d = cells.E75_PILOT if pilot else cells.E75_OUT
    return json.loads((d / f"exp075_train_X_s{seed}.json").read_text())


def _as_s(rec, shift=0.0, dists=("9", "10", "11")):
    """An S training record built from a real X record: same probe and per-distance n, hits at
    `dists` moved by `shift`, overall margin recomputed, S's walk evidence added."""
    r = copy.deepcopy(rec)
    bd = r["gate_l"]["by_distance"]
    for d in dists:
        bd[d]["hit"] += shift
    n = sum(v["n"] for v in bd.values())
    hit = sum(v["hit"] * v["n"] for v in bd.values()) / n
    chance = sum(v["chance"] * v["n"] for v in bd.values()) / n
    r["gate_l"].update(hit=hit, chance=chance, margin=hit - chance)
    r.update(arm="S", max_len=40, walk_max_lens=[40], train_hist_updates=250,
             train_hist={"10": 0.5, "11": 0.5})
    return r


def _write_s_train(out, shift=0.0, seeds=range(12), pilot=False, **over):
    out.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        r = _as_s(_x_train(s, pilot), shift)
        r.update(over)
        (out / cells.record_name(s)).write_text(json.dumps(r))


def _ev(arm, depth, seed, solved, **over):
    r = {"arm": arm, "depth": depth, "seed": seed, "solved": solved, "n": 200,
         "success_rate": solved / 200, "k": 4, "k_goal": 4, "mode": "C", "no_revisit": True,
         "judge_updates": 4000, "limit_states": None, "mean_steps": 10.0, "optimality": 0.9}
    if arm == "J4V-S":
        r["judge"] = "S"
    r.update(over)
    return r


def test_constants_are_the_spec():
    """Catches alpha, the stop bar, the Gate M band or the training settings drifting from
    spec sections 6 to 8, and an amendment committed before the pilot."""
    assert agg.ALPHA == 0.025 and agg.SMALL == 0.01 and agg.GATE_M_ALPHA == 0.05
    assert agg.X_PILOT_GATE_L == 0.12983333333333333 and agg.M_BAND == (9, 11)
    assert agg.TRAIN_SETTINGS == {"n_updates": 4000, "batch": 1000, "sync_every": 100,
                                  "probe_every": 250, "draws": 1, "max_len": 40}
    assert agg.DET_IGNORED == ("wall_s", "git_commit")
    assert agg.AMENDMENT is None and agg.GATE_L_THRESHOLD is None


def test_claim_verdicts_use_alpha_0025():
    """Catches a wrong alpha in either direction: p 0.0193 must confirm (it would not at
    0.05 / 3) and p 0.0327 must not (it would at 0.05)."""
    v, p = agg.claim_verdict([0.05] * 10 + [-0.05] * 2, 0.3, 0.2, True)
    assert v == "CONFIRMED" and p == pytest.approx(0.019287109375)
    v, p = agg.claim_verdict([0.02] * 9 + [-0.02] * 2 + [0.0], 0.3, 0.2, True)
    assert v == "NOT SIGNIFICANT" and p == pytest.approx(0.03271484375)
    assert agg.claim_verdict([0.005] * 12, 0.3, 0.2, True)[0] == "CONFIRMED (SMALL)"
    assert agg.claim_verdict([-0.01] * 12, 0.3, 0.2, True)[0] == "REFUTED"
    assert agg.claim_verdict([0.05] * 12, 0.3, 0.2, False)[0] == "VOID"
    assert agg.claim_verdict([0.005] * 12, 0.01, 0.005, True)[0] == "UNRESOLVED"


def test_margin_band_pools_over_distances_9_to_11():
    """Catches Gate M read over all probe distances, or averaged per distance instead of
    pooled by n."""
    gl = {"by_distance": {
        "7": {"hit": 0.9, "chance": 0.1, "n": 50},
        "9": {"hit": 0.5, "chance": 0.2, "n": 50},
        "10": {"hit": 0.4, "chance": 0.3, "n": 25},
        "11": {"hit": 0.7, "chance": 0.5, "n": 25}}}
    want = (0.5 * 50 + 0.4 * 25 + 0.7 * 25) / 100 - (0.2 * 50 + 0.3 * 25 + 0.5 * 25) / 100
    assert agg.margin_band(gl) == pytest.approx(want)


def test_gate_m_is_s_minus_x_by_sign_flip_and_refuses_a_different_probe():
    """Catches Gate M computed as S alone, at alpha other than 0.05, or across probes that
    differ (per-distance n must match X's, the cheap proof of the same probe)."""
    x = {s: _x_train(s) for s in range(12)}
    shifts = [0.02] * 9 + [-0.02] * 2 + [0.0]
    s_recs = {s: _as_s(x[s], shifts[s]) for s in range(12)}
    m = agg.gate_m(s_recs, x)
    assert m["p"] == pytest.approx(0.03271484375) and m["above"] is True
    assert m["mean"] == pytest.approx(sum(shifts) / 12)
    worse = {s: _as_s(x[s], v) for s, v in enumerate([0.05] * 9 + [-0.05] * 3)}
    assert agg.gate_m(worse, x)["above"] is False
    s_recs[4]["gate_l"]["by_distance"]["10"]["n"] += 1
    with pytest.raises(SystemExit, match="seed 4"):
        agg.gate_m(s_recs, x)


def test_train_record_must_prove_its_walk_length():
    """REVIEW FOCUS 2. Catches an S record trained on 14-move walks, one whose observed walk
    lengths disagree with its stated max_len, and one that observed no walk at all (a relaunch
    after the record write was lost), which cannot prove its walk length."""
    good = _as_s(_x_train(0))
    agg.check_train_record(good, "r.json")
    for over, msg in (({"max_len": 14}, "max_len"), ({"walk_max_lens": [14]}, "walk"),
                      ({"walk_max_lens": [14, 40]}, "walk"), ({"walk_max_lens": []}, "walk"),
                      ({"arm": "X"}, "arm"), ({"readout": "concept"}, "readout"),
                      ({"batch": 500}, "batch"), ({"hidden": 128}, "hidden")):
        bad = {**good, **over}
        with pytest.raises(SystemExit, match=msg):
            agg.check_train_record(bad, "r.json")


def test_pilot_amendment_states_the_stop_rule_and_the_threshold(tmp_path):
    """Catches the stop rule read against anything but X's 0.1298, and Gate L's threshold not
    half S's mean pilot margin (spec sections 6 and 7)."""
    _write_s_train(tmp_path, 0.0, seeds=(12, 13), pilot=True)
    a = agg.pilot_amendment(tmp_path)
    xs = [_x_train(s, pilot=True)["gate_l"]["margin"] for s in (12, 13)]
    assert a["mean_margin"] == pytest.approx(sum(xs) / 2)
    assert a["stop"] is False
    assert a["gate_l_threshold"] == pytest.approx(a["mean_margin"] / 2)
    assert set(a["by_distance_margin"][12]) == {"S", "X"}
    low = tmp_path / "low"
    low.mkdir()
    for s in (12, 13):
        r = _as_s(_x_train(s, pilot=True), -0.2, dists=("7", "8", "9", "10", "11"))
        (low / cells.record_name(s)).write_text(json.dumps(r))
    b = agg.pilot_amendment(low)
    assert b["mean_margin"] < agg.X_PILOT_GATE_L and b["stop"] is True


def _write_gatel_x(out, **gl_over):
    ref = _x_train(0)["gate_l"]
    (out / evaluate.gatel_x_record_name(0)).write_text(
        json.dumps({"kind": "gatel-X", "seed": 0, "gate_l": {**ref, **gl_over}}))


def test_gate_m_continuity_compares_with_the_committed_record(tmp_path):
    """Catches Gate M read against X's published margins without checking they reproduce, and
    a continuity check blind to a changed per-distance hit."""
    with pytest.raises(SystemExit, match="exp079_gatel_X_s0"):
        agg.gate_m_continuity(tmp_path)
    _write_gatel_x(tmp_path)
    assert agg.gate_m_continuity(tmp_path) == "PASS"
    bd = copy.deepcopy(_x_train(0)["gate_l"]["by_distance"])
    bd["10"]["hit"] += 0.02
    _write_gatel_x(tmp_path, by_distance=bd)
    assert agg.gate_m_continuity(tmp_path) == "FAIL"


def test_stage_train_needs_the_amendment_and_voids_on_l_or_e(tmp_path, monkeypatch):
    """Catches Gates L and E read before the amendment, Gate E not checked on every seed, and
    Gate L passing with a mean below its threshold."""
    _write_s_train(tmp_path, 0.0)
    _write_gatel_x(tmp_path)
    with pytest.raises(SystemExit, match="amendment"):
        agg.stage_train(tmp_path)
    monkeypatch.setattr(agg, "AMENDMENT", {"date": "2026-10-20"})
    monkeypatch.setattr(agg, "GATE_L_THRESHOLD", 0.1)
    res = agg.stage_train(tmp_path)
    assert res["l"]["pass"] is True and res["e"] is True and res["m"]["above"] is False
    assert res["m_cont"] == "PASS"
    monkeypatch.setattr(agg, "GATE_L_THRESHOLD", 0.9)
    assert agg.stage_train(tmp_path)["l"]["pass"] is False
    monkeypatch.setattr(agg, "GATE_L_THRESHOLD", 0.1)
    r = json.loads((tmp_path / cells.record_name(5)).read_text())
    r["encoder_drift"] = 0.0
    (tmp_path / cells.record_name(5)).write_text(json.dumps(r))
    assert agg.stage_train(tmp_path)["e"] is False


def test_load_records_reads_x_from_exp076_and_s_from_out(tmp_path):
    """REVIEW FOCUS 4. Catches J4V-X read from anywhere but EXP-076's committed records, which
    carry no judge field, and a J4V-S record labelled with another judge."""
    for d in (9, 11):
        for s in range(12):
            (tmp_path / evaluate.record_name("J4V-S", d, s)).write_text(
                json.dumps(_ev("J4V-S", d, s, 50)))
    recs = agg.load_records(tmp_path)
    ref = json.loads((cells.E76_OUT / "exp076_J4V-X_d11_s3.json").read_text())
    assert recs[("J4V-X", 11)][3]["solved"] == ref["solved"] and "judge" not in ref
    assert recs[("J4V-S", 9)][7]["solved"] == 50
    (tmp_path / evaluate.record_name("J4V-S", 9, 7)).write_text(
        json.dumps(_ev("J4V-S", 9, 7, 50, judge="X")))
    with pytest.raises(SystemExit, match="judge"):
        agg.load_records(tmp_path)


def test_eval_records_at_other_settings_or_smoke_are_refused(tmp_path):
    """Catches a smoke cell, a 3-move cell, a partly trained judge, or a record whose content
    is not the cell its filename names."""
    for over, msg in (({"limit_states": 20}, "smoke"), ({"k": 3}, "k is"),
                      ({"judge_updates": 3750}, "judge_updates"), ({"seed": 4}, "seed")):
        for d in (9, 11):
            for s in range(12):
                (tmp_path / evaluate.record_name("J4V-S", d, s)).write_text(
                    json.dumps(_ev("J4V-S", d, s, 50)))
        (tmp_path / evaluate.record_name("J4V-S", 9, 2)).write_text(
            json.dumps(_ev("J4V-S", 9, 2, 50, **over)))
        with pytest.raises(SystemExit, match=msg):
            agg.load_records(tmp_path)


def test_stage_cont_compares_against_exp076s_committed_records(tmp_path):
    """Catches Gate 0(b) checked against the wrong records, or blind to a changed outcome."""
    for d in (9, 11):
        ref = json.loads((cells.E76_OUT / f"exp076_J4V-X_d{d}_s0.json").read_text())
        (tmp_path / evaluate.record_name("J4V-X", d, 0)).write_text(json.dumps(ref))
    assert agg.stage_cont(tmp_path) == "PASS"
    p = tmp_path / evaluate.record_name("J4V-X", 11, 0)
    r = json.loads(p.read_text())
    r["solved"] += 1
    p.write_text(json.dumps(r))
    assert agg.stage_cont(tmp_path) == "FAIL"


def test_gate0a_ignores_only_the_session_fields(tmp_path):
    """Catches a determinism check that ignores an outcome field (judge_evals here) or that
    fails on wall_s and git_commit, which differ by design."""
    full, det = tmp_path / "o", tmp_path / "d"
    for d in (full, det):
        d.mkdir()
    base = _ev("J4V-S", 9, 0, 77, judge_evals=1296.0)
    (full / evaluate.record_name("J4V-S", 9, 0)).write_text(json.dumps(base))
    (det / evaluate.record_name("J4V-S", 9, 0)).write_text(
        json.dumps({**base, "wall_s": 1.0, "git_commit": "abc"}))
    assert agg.gate0a(full, det) == "PASS"
    (det / evaluate.record_name("J4V-S", 9, 0)).write_text(
        json.dumps({**base, "judge_evals": 1295.0}))
    assert agg.gate0a(full, det) == "FAIL"


def test_claims_compare_s_with_x_at_the_registered_depths():
    """Catches Claim 1 read at depth 9 or Claim 2 at depth 11, or the arms reversed."""
    rates = {("J4V-S", 11): {s: 0.30 for s in range(12)},
             ("J4V-X", 11): {s: 0.20 for s in range(12)},
             ("J4V-S", 9): {s: 0.40 for s in range(12)},
             ("J4V-X", 9): {s: 0.45 for s in range(12)}}
    full = agg.primary_verdicts(rates, True)
    assert full["claim1"][0] == "CONFIRMED" and full["claim1"][2] == pytest.approx(0.10)
    assert full["claim2"][0] == "REFUTED" and full["claim2"][2] == pytest.approx(-0.05)


def test_readings_lead_with_gate_m_and_name_a_trade():
    """Catches the spec's fixed readings missing from the output: a confirmed Claim 1 with
    Gate M not above 0 must lead RESULTS.md, a confirmed Claim 1 with a refuted Claim 2 is a
    trade, and a null Claim 1 is about this lever only."""
    c = ("CONFIRMED", 0.001, 0.05, 0.3, 0.25)
    r = ("REFUTED", 0.9, -0.02, 0.4, 0.42)
    ns = ("NOT SIGNIFICANT", 0.2, 0.01, 0.2, 0.19)
    lines = agg.readings({"claim1": c, "claim2": r}, gate_m_above=False)
    assert any("GATE M" in x and "lead" in x for x in lines)
    assert any("trade" in x for x in lines)
    assert agg.readings({"claim1": c, "claim2": c}, gate_m_above=True) == []
    assert any("BY THIS LEVER" in x for x in
               agg.readings({"claim1": ns, "claim2": c}, gate_m_above=True))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (Bash tool `timeout: 600000`): `.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m "not slow"`
Expected: collection error, `FileNotFoundError` for `aggregate.py`.

- [ ] **Step 3: Write `aggregate.py`**

```python
"""EXP-079 aggregator. Gates 0, L and E are VERDICTS: a claim whose gate failed prints VOID, a
contrast on the floor or ceiling prints UNRESOLVED, and a confirmed gain under 0.01 prints
CONFIRMED (SMALL). Gate M and Gate R are reported; Gate M binds the reading, and the fixed
readings of spec section 8 are printed as lines of their own.

Spec: docs/superpowers/specs/2026-10-18-exp079-judge-trained-where-scrambles-live-design.md,
sections 6 to 8.

The sign-flip test and Gate 1 are EXP-070's, Gate 0(b)'s field comparison and Gate R are
EXP-071's, seed pairing is EXP-072's, Gate L's verdict is EXP-074's, and the pilot stop bar is
EXP-075's X Gate L threshold, all reused through importlib. Alpha is 0.05 / 2.

Usage:
    .venv/bin/python experiments/079_judge_trained_deep/aggregate.py --pilot-report
    .venv/bin/python experiments/079_judge_trained_deep/aggregate.py --stage cont
    .venv/bin/python experiments/079_judge_trained_deep/aggregate.py --stage train
    .venv/bin/python experiments/079_judge_trained_deep/aggregate.py
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
a75 = _load("exp075_aggregate", REPO / "experiments" / "075_wide_region" / "aggregate.py")
cells = _load("exp079_cells", HERE / "cells.py")
ev = _load("exp079_evaluate", HERE / "evaluate.py")
one_sided_p = a70.one_sided_p

SEEDS = cells.EVAL_SEEDS
DEPTHS = ev.DEPTHS
ALPHA = 0.05 / 2
SMALL = 0.01
GATE_M_ALPHA = 0.05
M_BAND = (9, 11)
# Spec section 6: S must clear the Gate L threshold EXP-075's X cleared, or the pilot stops.
X_PILOT_GATE_L = a75.GATE_L_THRESHOLD["X"]
TRAIN_SETTINGS = {"n_updates": 4000, "batch": 1000, "sync_every": 100, "probe_every": 250,
                  "draws": 1, "max_len": cells.MAX_LEN}
# Evaluation runs without a checkpoint, so these are the only session fields a record has.
DET_IGNORED = ("wall_s", "git_commit")

# Set together by the dated post-pilot amendment (spec section 6), before any seed 0-11 trains:
# AMENDMENT = {"date", "train_workers", "eval_workers", "spec_section"}, and Gate L's threshold,
# half S's mean pilot margin. The launcher refuses every phase but check and pilot while either
# is None.
AMENDMENT: dict | None = None
GATE_L_THRESHOLD: float | None = None


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


def check_train_record(rec, name) -> None:
    for k, v in TRAIN_SETTINGS.items():
        if rec.get(k) != v:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, spec says {v!r}")
    for k, v in (("arm", cells.ARM), ("readout", "wide"),
                 ("hidden", cells.c75.HIDDEN[cells.BASE_ARM])):
        if rec.get(k) != v:
            raise SystemExit(f"{name}: {k} is {rec.get(k)!r}, spec says {v!r}")
    if rec.get("walk_max_lens") != [cells.MAX_LEN]:
        raise SystemExit(
            f"{name}: walk_max_lens is {rec.get('walk_max_lens')!r}; the walks must be observed "
            f"at exactly [{cells.MAX_LEN}]. An empty list means the session that wrote this record "
            f"trained no update (its record write was lost after the final checkpoint): delete "
            f"the record and judge_S_s<seed>/ and retrain that seed.")


def margin_band(gate_l, lo=M_BAND[0], hi=M_BAND[1]) -> float:
    """Gate L's margin restricted to probe distances lo..hi, pooled by n."""
    rows = [v for d, v in gate_l["by_distance"].items() if lo <= int(d) <= hi]
    n = sum(v["n"] for v in rows)
    return (sum(v["hit"] * v["n"] for v in rows) - sum(v["chance"] * v["n"] for v in rows)) / n


def gate_m(s_recs: dict, x_recs: dict) -> dict:
    """Gate M (spec section 7): per seed, S's margin at probe distances 9 to 11 minus X's on the
    same probe states, by exact one-sided sign-flip. The probes must agree in n per distance."""
    for s in sorted(s_recs):
        bs, bx = s_recs[s]["gate_l"]["by_distance"], x_recs[s]["gate_l"]["by_distance"]
        if {d: v["n"] for d, v in bs.items()} != {d: v["n"] for d, v in bx.items()}:
            raise SystemExit(f"seed {s}: S's Gate L probe differs from X's in n per distance; "
                             f"Gate M needs the same probe states")
    diffs = [margin_band(s_recs[s]["gate_l"]) - margin_band(x_recs[s]["gate_l"])
             for s in sorted(s_recs)]
    p = one_sided_p(diffs)
    m = st.mean(diffs)
    return {"diffs": diffs, "mean": m, "p": p, "wins": sum(d > 0 for d in diffs),
            "above": bool(p < GATE_M_ALPHA and m > 0)}


def _x_train(seed, pilot=False) -> dict:
    d = cells.E75_PILOT if pilot else cells.E75_OUT
    return _read(d / cells.c75.record_name(cells.BASE_ARM, seed))


def gate_m_continuity(out) -> str:
    """X seed 0's Gate L, re-measured here by evaluate.gatel_x_cell, must equal EXP-075's
    committed training record exactly, or Gate M's X side is a number today's code does not
    produce."""
    p = Path(out) / ev.gatel_x_record_name(0)
    if not p.exists():
        raise SystemExit(f"missing {p.name}: run the cont phase (Gate M continuity) first")
    got = json.loads(p.read_text())["gate_l"]
    return "PASS" if got == _x_train(0)["gate_l"] else "FAIL"


def pilot_amendment(pilot_dir=cells.PILOT_OUT) -> dict:
    """The numbers the dated amendment states (spec section 6), from the two S pilot records,
    with EXP-075's X pilot judges on the same seeds alongside."""
    pilot_dir = Path(pilot_dir)
    s, x = {}, {}
    for seed in cells.PILOT_SEEDS:
        name = cells.record_name(seed)
        s[seed] = _read(pilot_dir / name)
        check_train_record(s[seed], name)
        x[seed] = _x_train(seed, pilot=True)
    margins = {seed: s[seed]["gate_l"]["margin"] for seed in cells.PILOT_SEEDS}
    mean = st.mean(margins.values())

    def by_d(rec):
        return {d: v["hit"] - v["chance"] for d, v in sorted(rec["gate_l"]["by_distance"].items(),
                                                             key=lambda kv: int(kv[0]))}

    return {"margins": margins, "mean_margin": mean, "x_pilot_gate_l": X_PILOT_GATE_L,
            "stop": mean < X_PILOT_GATE_L, "gate_l_threshold": mean / 2,
            "by_distance_margin": {seed: {"S": by_d(s[seed]), "X": by_d(x[seed])}
                                   for seed in cells.PILOT_SEEDS},
            "seconds_per_update": {seed: a75.seconds_per_update(s[seed])
                                   for seed in cells.PILOT_SEEDS},
            "train_hist": {seed: s[seed]["train_hist"] for seed in cells.PILOT_SEEDS},
            "train_hist_updates": {seed: s[seed]["train_hist_updates"]
                                   for seed in cells.PILOT_SEEDS}}


def require_amendment() -> None:
    if AMENDMENT is None or GATE_L_THRESHOLD is None:
        raise SystemExit("AMENDMENT or GATE_L_THRESHOLD is unset: the dated post-pilot "
                         "amendment sets both; no gate or claim may be read before then")


def stage_train(out) -> dict:
    """Gates L and E, and Gate M (reported), from the 12 S training records alone."""
    require_amendment()
    out = Path(out)
    recs = {}
    for seed in SEEDS:
        name = cells.record_name(seed)
        recs[seed] = _read(out / name)
        check_train_record(recs[seed], name)
    margins = [recs[seed]["gate_l"]["margin"] for seed in SEEDS]
    return {"l": {"mean": st.mean(margins), "threshold": GATE_L_THRESHOLD,
                  "p": one_sided_p(margins),
                  "pass": a74.gate_l_verdict(margins, GATE_L_THRESHOLD)},
            "e": all(recs[seed]["encoder_drift"] > 0 for seed in SEEDS),
            "m": gate_m(recs, {seed: _x_train(seed) for seed in SEEDS}),
            "m_cont": gate_m_continuity(out),
            "recs": recs}


def format_stage_train(res) -> list[str]:
    l, m = res["l"], res["m"]
    return [f"GATE L (VI worked): {'PASS' if l['pass'] else 'FAIL'}, mean margin "
            f"{l['mean']:.4f} vs threshold {l['threshold']:.4f}, p {l['p']:.4f}",
            f"GATE E (encoders trained): {'PASS' if res['e'] else 'FAIL'}",
            f"GATE M continuity (X seed 0 Gate L re-measured here): {res['m_cont']}",
            f"GATE M (reported, binds the reading): S - X margin at probe distances 9-11, mean "
            f"{m['mean']:+.4f}, S ahead on {m['wins']} of {len(m['diffs'])}, p {m['p']:.4f} -> "
            f"{'ABOVE 0' if m['above'] else 'NOT ABOVE 0'}"]


def check_eval_record(rec, path) -> None:
    name = Path(path).name
    arm = rec.get("arm")
    if arm not in ev.ARMS:
        raise SystemExit(f"{name}: arm is {arm!r}, expected one of {list(ev.ARMS)}")
    if rec.get("k") != ev.K or rec.get("k_goal") != ev.K:
        raise SystemExit(f"{name}: k is {rec.get('k')!r} and k_goal {rec.get('k_goal')!r}; "
                         f"spec says {ev.K} and {ev.K}")
    if rec.get("mode") != "C" or rec.get("no_revisit") is not True:
        raise SystemExit(f"{name}: mode {rec.get('mode')!r}, no_revisit "
                         f"{rec.get('no_revisit')!r}; spec says C with no-revisit")
    if rec.get("judge_updates") != cells.N_UPDATES:
        raise SystemExit(f"{name}: judge_updates is {rec.get('judge_updates')!r}, spec says "
                         f"{cells.N_UPDATES}: it was evaluated from a partly trained judge")
    # EXP-076's J4V-X records predate the judge field; this experiment's S records must carry it.
    if arm == "J4V-S" and rec.get("judge") != "S":
        raise SystemExit(f"{name}: judge is {rec.get('judge')!r}, a J4V-S record must say 'S'")


def _read_cell(path: Path, arm: str, depth: int, seed: int) -> dict:
    """Read a record, check its spec fields, and check it is the cell its filename claims."""
    r = _read(path)
    check_eval_record(r, path)
    for field, want in (("arm", arm), ("depth", depth), ("seed", seed)):
        if r.get(field) != want:
            raise SystemExit(f"{Path(path).name}: {field} is {r.get(field)!r}, the file name "
                             f"says {want!r}")
    return r


def load_records(out_dir) -> dict:
    """(arm, depth) -> {seed: record}. J4V-X from EXP-076's committed records (licensed by
    Gate 0(b)); J4V-S from this experiment's outputs."""
    out_dir = Path(out_dir)
    recs = {}
    for d in DEPTHS:
        recs[("J4V-X", d)] = {s: _read_cell(cells.E76_OUT / f"exp076_J4V-X_d{d}_s{s}.json",
                                              "J4V-X", d, s) for s in SEEDS}
        recs[("J4V-S", d)] = {s: _read_cell(out_dir / ev.record_name("J4V-S", d, s),
                                              "J4V-S", d, s) for s in SEEDS}
    return recs


def rates_of(recs) -> dict:
    return {key: {s: r["success_rate"] for s, r in by_seed.items()}
            for key, by_seed in recs.items()}


def stage_cont(out) -> str:
    """Gate 0(b): J4V-X at seed 0, depths 9 and 11, re-run here, against EXP-076's records."""
    out = Path(out)
    return a71.gate0b_verdict(
        {("J4V-X", d): _read_cell(out / ev.record_name("J4V-X", d, 0), "J4V-X", d, 0)
         for d in DEPTHS},
        {("J4V-X", d): _read(cells.E76_OUT / f"exp076_J4V-X_d{d}_s0.json") for d in DEPTHS})


def format_stage_cont(verdict) -> str:
    return f"GATE 0(b) J4V-X continuity with EXP-076 (d9, d11, seed 0): {verdict}"


def gate0a(out_dir, det_dir) -> str:
    """Gate 0(a): the det re-run of J4V-S d9 s0 equals the full-run record in every field but
    the session fields."""
    recs = []
    for d in (Path(out_dir), Path(det_dir)):
        p = d / ev.record_name("J4V-S", 9, 0)
        if not p.exists():
            raise SystemExit(f"missing {p}: Gate 0(a) cannot be computed")
        recs.append(json.loads(p.read_text()))
    keep = lambda r: {k: v for k, v in r.items() if k not in DET_IGNORED}  # noqa: E731
    return "PASS" if keep(recs[0]) == keep(recs[1]) else "FAIL"


def primary_verdicts(rates: dict, gate_ok: bool) -> dict:
    """Claim 1: d11 J4V-S vs J4V-X. Claim 2: d9 J4V-S vs J4V-X."""
    return {"claim1": contrast(rates, ("J4V-S", 11), ("J4V-X", 11), gate_ok),
            "claim2": contrast(rates, ("J4V-S", 9), ("J4V-X", 9), gate_ok)}


def readings(full: dict, gate_m_above: bool) -> list[str]:
    """Spec sections 7 and 8, fixed before the numbers: what the output must say beside the
    verdicts."""
    c1, c2 = full["claim1"][0], full["claim2"][0]
    out = []
    if c1.startswith("CONFIRMED") and not gate_m_above:
        out.append("GATE M IS NOT ABOVE 0 WHILE CLAIM 1 IS CONFIRMED: RESULTS.md must lead with "
                   "this. The gain is not attributable to better ranking far from the goal.")
    if c1.startswith("CONFIRMED") and c2 == "REFUTED":
        out.append("CLAIM 1 CONFIRMED WITH CLAIM 2 REFUTED: report a trade, not a win. S gave up "
                   "near-goal training share and depth 9 paid for it.")
    if c1 in ("NOT SIGNIFICANT", "REFUTED"):
        out.append("CLAIM 1 READS: the judge is not the limit BY THIS LEVER. That is not "
                   "evidence that search is the limit.")
    return out


def _mean(recs, key, field):
    return st.mean(r[field] for r in recs[key].values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--pilot-dir", type=Path, default=cells.PILOT_OUT)
    ap.add_argument("--det-dir", type=Path, default=HERE / "outputs_det")
    ap.add_argument("--pilot-report", action="store_true",
                    help="print the amendment's numbers from the pilot records and stop")
    ap.add_argument("--stage", choices=("cont", "train"),
                    help="read only Gate 0(b), or only Gates L, E and M, and stop")
    args = ap.parse_args()
    if args.pilot_report:
        a = pilot_amendment(args.pilot_dir)
        for key, val in a.items():
            print(f"{key}: {val}")
        if a["stop"]:
            print(f"STOP: S's mean pilot Gate L margin {a['mean_margin']:.4f} is below X's "
                  f"threshold {X_PILOT_GATE_L:.4f}. Report; there is no amendment "
                  f"(spec section 6).")
        return
    require_amendment()
    out = args.out_dir
    if args.stage == "cont":
        print(format_stage_cont(stage_cont(out)))
        return
    tr = stage_train(out)
    if args.stage == "train":
        for line in format_stage_train(tr):
            print(line)
        return
    g0b = stage_cont(out)
    g0a = gate0a(out, args.det_dir)
    recs = load_records(out)
    rates = rates_of(recs)
    rank = {d: [_read(out / ev.rank_record_name(d, s)) for s in SEEDS] for d in DEPTHS}
    rank76 = {d: [json.loads((cells.E76_OUT / f"exp076_rank_J-X4_d{d}_s{s}.json").read_text())
                  for s in SEEDS] for d in DEPTHS}

    print(f"GATE 0(a) determinism (J4V-S d9 s0 re-run): {g0a}")
    print(format_stage_cont(g0b))
    for line in format_stage_train(tr):
        print(line)
    print("GATE R at k 4 (reported, not voiding): lowest-J 4-move leaf closer than root")
    for d in DEPTHS:
        ok, p = a71.gate_r_verdict(rank[d], "hit", "chance")
        print(f"  d{d}: S hit {st.mean(r['hit'] for r in rank[d]):.4f} chance "
              f"{st.mean(r['chance'] for r in rank[d]):.4f} p {p:.4f} -> {ok}   "
              f"(EXP-076 X: hit {st.mean(r['hit'] for r in rank76[d]):.4f} chance "
              f"{st.mean(r['chance'] for r in rank76[d]):.4f})")

    print("\nMean held-out success (mean steps to solve, optimality)")
    for d in DEPTHS:
        print(f"  d{d}: " + "  ".join(
            f"{a} {_mean(recs, (a, d), 'success_rate'):.4f} "
            f"({_mean(recs, (a, d), 'mean_steps'):.2f}, {_mean(recs, (a, d), 'optimality'):.3f})"
            for a in ev.ARMS))

    gate_ok = g0a == "PASS" and g0b == "PASS" and tr["l"]["pass"] and tr["e"]
    if not gate_ok:
        print("Gate 0, Gate L or Gate E did not pass. Every claim is VOID.")
    full = primary_verdicts(rates, gate_ok)
    for name, label in (("claim1", "CLAIM 1 (primary) d11 J4V-S - J4V-X"),
                        ("claim2", "CLAIM 2 (primary) d9 J4V-S - J4V-X")):
        v, p, diff, am, bm = full[name]
        wins = sum(x > 0 for x in a72.pair_diffs(rates[("J4V-S", 11 if name == "claim1" else 9)],
                                                  rates[("J4V-X", 11 if name == "claim1" else 9)]))
        print(f"{label}: {am:.4f} vs {bm:.4f}, diff {diff:+.4f}, S ahead on {wins} of 12, "
              f"p {p:.4f} -> {v}")
    for line in readings(full, tr["m"]["above"]):
        print("    " + line)

    print("\nSecondary (patterns, never confirmations):")
    j5 = st.mean(json.loads((cells.E78_OUT / f"exp078_J5V-X_d11_s{s}.json").read_text())
                 ["success_rate"] for s in SEEDS)
    print(f"  d11 EXP-078 full 5-move tree with X alongside: {j5:.4f}")
    s_sp = st.mean(r["final_probe"]["spearman_7_11"] for r in tr["recs"].values())
    x_sp = st.mean(_x_train(s)["final_probe"]["spearman_7_11"] for s in SEEDS)
    print(f"  probe spearman_7_11, mean over seeds: S {s_sp:.4f}  X {x_sp:.4f}")
    for d in ("9", "10", "11"):
        sj = st.mean(r["final_probe"]["mean_j_by_distance"][d] for r in tr["recs"].values())
        xj = st.mean(_x_train(s)["final_probe"]["mean_j_by_distance"][d] for s in SEEDS)
        print(f"  probe mean J at distance {d}: S {sj:.3f}  X {xj:.3f}")
    dists = sorted({d for r in tr["recs"].values() for d in r["train_hist"]}, key=int)
    hist = {d: st.mean(r["train_hist"].get(d, 0.0) for r in tr["recs"].values()) for d in dists}
    print("  realised training histogram, mean over seeds (last probe interval): "
          + "  ".join(f"{d}:{v:.3f}" for d, v in hist.items()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (Bash tool `timeout: 600000`): `.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m "not slow"`
Expected: 37 passed, 2 deselected.

- [ ] **Step 5: Mutation check**

Apply each mutation alone, run the command, confirm the named test FAILS, then restore:
1. `ALPHA = 0.05 / 2` to `ALPHA = 0.05`: `test_claim_verdicts_use_alpha_0025`.
2. In `check_train_record`, delete the `walk_max_lens` check: `test_train_record_must_prove_its_walk_length`.
3. In `gate_m`, delete the n-per-distance loop: `test_gate_m_is_s_minus_x_by_sign_flip_and_refuses_a_different_probe`.
4. In `gate_m_continuity`, return `"PASS"` unconditionally: `test_gate_m_continuity_compares_with_the_committed_record`.
5. In `pilot_amendment`, change `"stop": mean < X_PILOT_GATE_L` to `"stop": mean < X_PILOT_GATE_L / 2`: `test_pilot_amendment_states_the_stop_rule_and_the_threshold`.
6. Change `DET_IGNORED` to `("wall_s", "git_commit", "judge_evals")`: `test_gate0a_ignores_only_the_session_fields`.
Confirm `git diff experiments/079_judge_trained_deep/aggregate.py` is clean of mutations afterwards. Record all six results in the report.

- [ ] **Step 6: Commit**

```bash
git add experiments/079_judge_trained_deep/aggregate.py tests/experiments/test_exp079.py
git commit -m "EXP-079: aggregator (Gates 0, L, E as verdicts, Gate M binding the reading, Claims 1 and 2 at alpha 0.025, pilot stop rule)"
```

---

### Task 5: launcher and chain for the laptop

**Files:**
- Create: `experiments/079_judge_trained_deep/launch079.ps1`
- Create: `experiments/079_judge_trained_deep/chain079.ps1`
- Modify: `tests/experiments/test_exp079.py` (append)
- Modify: `CLAUDE.md` (the test-count line)

**Interfaces:**
- Consumes: `cells.py prepare-pilot-encoders` (Task 1), `train.py` flags (Task 2), `evaluate.py` flags and record names (Task 3), `aggregate.py` (`AMENDMENT`, `GATE_L_THRESHOLD`, `--stage cont`, `--stage train`, `format_stage_cont`, `format_stage_train`) (Task 4), EXP-075's `pretrain.py` (`--pilot --arms X --seeds 12 13 --out-dir D --workers N`).
- Produces: `launch079.ps1 -Phase check|pilot-encoders|pilot-pretrain|pilot-train|train|cont|eval|det|rank [-Workers N] [-MaxLen 40] [-SkipExisting]`; `chain079.ps1 [-From cont|gate0b|train|gatele|eval|det|rank]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_exp079.py`:

```python
# The regexes chain079.ps1 applies to the aggregator's stage output (PowerShell -match is
# case-insensitive). Kept here verbatim so a change to either side fails a test.
CHAIN_GATE0B = "continuity.*: PASS"
CHAIN_GATE_L = r"GATE L \(VI worked\): PASS"
CHAIN_GATE_E = r"GATE E \(encoders trained\): PASS"
CHAIN_GATE_MC = r"GATE M continuity \(X seed 0 Gate L re-measured here\): PASS"


def test_launcher_pins_the_walk_length():
    """Catches a launcher that can start S training at a walk length other than 40, or that
    does not pass it to the driver (spec section 4)."""
    text = (EXP / "launch079.ps1").read_text()
    assert "[int]$MaxLen = 40" in text
    assert "if ($MaxLen -ne 40)" in text
    assert text.count('"--max-len", $MaxLen') == 2  # pilot-train and train


def test_launcher_carries_the_fixes_and_the_registered_cells():
    """Catches a launcher without EXP-074's .Handle fix, without the amendment guard on both
    fields, with an eval list missing an arm or depth, or with the worker cap lifted."""
    text = (EXP / "launch079.ps1").read_text()
    assert "$null = $proc.Handle" in text
    assert "a.AMENDMENT" in text and "a.GATE_L_THRESHOLD" in text
    assert 'foreach ($d in 11, 9)' in text and '"J4V-S"' in text and '"J4V-X"' in text
    assert "-gt 6" in text and "outputs_det" in text
    assert "prepare-pilot-encoders" in text and '"--gatel-x"' in text
    assert "075_wide_region\\pretrain.py" in text and '"--arms", "X"' in text
    assert '$target = Join-Path $phaseDir $j[1]' in text


def test_launcher_samples_memory_in_driver_phases_too():
    """Catches training phases whose peak committed memory is never measured: the driver runs
    its own pool, so the launcher must sample every python process, not only its own child."""
    text = (EXP / "launch079.ps1").read_text()
    assert "RedirectStandardOutput" in text
    assert text.count("PrivateMemorySize64") >= 2
    assert "max total commit MB" in text and "peak commit MB" in text


def test_chain_order_and_gate_regexes_match_the_aggregator():
    """Catches a chain that skips a gate, runs eval before Gates L and E, or whose regex no
    longer matches what the aggregator prints, so a PASS would read as a STOP or a FAIL as a
    PASS."""
    text = (EXP / "chain079.ps1").read_text()
    assert '@("cont","gate0b","train","gatele","eval","det","rank")' in text
    for rx in (CHAIN_GATE0B, CHAIN_GATE_L, CHAIN_GATE_E, CHAIN_GATE_MC):
        assert rx in text
    assert "--stage cont" in text and "--stage train" in text
    assert "train_workers" in text and "eval_workers" in text
    assert re.search(CHAIN_GATE0B, agg.format_stage_cont("PASS"), re.I)
    assert not re.search(CHAIN_GATE0B, agg.format_stage_cont("FAIL"), re.I)
    m = {"mean": 0.01, "p": 0.2, "wins": 7, "diffs": [0.0] * 12, "above": False}
    ok = agg.format_stage_train({"l": {"pass": True, "mean": 0.2, "threshold": 0.1, "p": 0.001},
                                 "e": True, "m": m, "m_cont": "PASS"})
    bad = agg.format_stage_train({"l": {"pass": False, "mean": 0.05, "threshold": 0.1,
                                        "p": 0.3}, "e": False, "m": m, "m_cont": "FAIL"})
    assert re.search(CHAIN_GATE_L, "\n".join(ok), re.I)
    assert re.search(CHAIN_GATE_E, "\n".join(ok), re.I)
    assert not re.search(CHAIN_GATE_L, "\n".join(bad), re.I)
    assert not re.search(CHAIN_GATE_E, "\n".join(bad), re.I)
    assert re.search(CHAIN_GATE_MC, "\n".join(ok), re.I)
    assert not re.search(CHAIN_GATE_MC, "\n".join(bad), re.I)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (Bash tool `timeout: 600000`): `.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m "not slow" -k "launcher or chain"`
Expected: 4 failed (`FileNotFoundError` for `launch079.ps1` and `chain079.ps1`).

- [ ] **Step 3: Write `launch079.ps1`**

```powershell
# EXP-079 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase pilot-encoders
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase pilot-pretrain -Workers 2   (only if pilot-encoders refused)
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase pilot-train    -Workers 2
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase cont  -Workers 2
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase train -Workers N   (N from the amendment)
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase eval  -Workers N -SkipExisting
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase det   -Workers 1
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase rank  -Workers N -SkipExisting
#
# Order (spec section 10): pilot-encoders (or pilot-pretrain) -> pilot-train -> stop rule and
# dated amendment (sets aggregate.AMENDMENT and aggregate.GATE_L_THRESHOLD) -> chain079.ps1:
# cont (Gate 0(b)) -> train -> Gates L and E -> eval (depth 11 first) -> det (Gate 0(a), into
# outputs_det) -> rank. Every phase but check and the pilot phases refuses to run while either
# amendment field is None. Training resumes by itself (the driver skips finished seeds and
# train_judge resumes its checkpoint); evaluation phases take -SkipExisting.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Arm and seed lists therefore live in this file.
# Launch long phases with OS-level redirection (playbook section 3), never Start-Process
# from an ssh session.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","pilot-encoders","pilot-pretrain","pilot-train","train","cont","eval","det","rank")][string]$Phase,
    [int]$Workers = 0,
    [int]$MaxLen = 40,
    [switch]$SkipExisting
)

# Spec sections 3 and 4, fixed. The walk length is pinned: any other value is refused.
if ($MaxLen -ne 40) { Write-Error "MaxLen is pinned at 40 (spec section 4); got $MaxLen."; exit 1 }
$Updates = 4000
$SyncEvery = 100
$ProbeEvery = 250

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\079_judge_trained_deep"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# EXP-075's 12 X encoders and 12 X judges at 4000 updates, EXP-076's 24 J4V-X records, and
# EXP-075's two X pilot training records (the pilot comparison and the encoder proof).
$present = & $py -c "import json, sys; from pathlib import Path; o=Path('experiments/075_wide_region/outputs'); p=Path('experiments/075_wide_region/outputs_pilot'); r6=Path('experiments/076_deeper_tree/outputs'); e=sum((o/'encoders'/('enc_X_s'+str(s)+'.pt')).exists() for s in range(12)); j=sum((o/('judge_X_s'+str(s))/'judge.pt').exists() and json.loads((o/('judge_X_s'+str(s))/'state.json').read_text())['update']==4000 for s in range(12)); r=sum((r6/('exp076_J4V-X_d'+str(d)+'_s'+str(s)+'.json')).exists() for d in (9, 11) for s in range(12)); t=sum((p/('exp075_train_X_s'+str(s)+'.json')).exists() for s in (12, 13)); sys.stdout.write(str(e)+' '+str(j)+' '+str(r)+' '+str(t))"
if ($LASTEXITCODE -ne 0 -or $present -ne "12 12 24 2") { Write-Error "expected 12 X encoders, 12 X judges, 24 J4V-X records and 2 X pilot records, found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                     = $where"
"checkout                    = $head"
"X enc, X judges, J4V-X, pilot = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs                     = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

# Every phase after the pilot reads or produces numbers that decide a claim, and none may run
# before the dated amendment sets BOTH fields. Check the exit code too: an import failure
# prints nothing, and "" would not match "None".
$isPilot = $Phase -like "pilot-*"
if (-not $isPilot) {
    $amend = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','$($exp.Replace('\','/'))/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.AMENDMENT)+' | '+str(a.GATE_L_THRESHOLD))"
    if ($LASTEXITCODE -ne 0 -or $amend -eq "") { Write-Error "could not read AMENDMENT and GATE_L_THRESHOLD (got '$amend'); refusing to run."; exit 1 }
    if ($amend -match "None") { Write-Error "an amendment field is unset ($amend): commit the dated post-pilot amendment before this phase."; exit 1 }
    "amendment | Gate L          = $amend"
}

if ($Workers -eq 0) { $Workers = 2 }
if ($Workers -gt 6) { Write-Error "Workers capped at 6; got $Workers."; exit 1 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$outDir   = Join-Path $repo "$exp\outputs"
$pilotDir = Join-Path $repo "$exp\outputs_pilot"
$detDir   = Join-Path $repo "$exp\outputs_det"
$evalSeeds = @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")

if ($Phase -eq "pilot-encoders") {
    & $py "$exp\cells.py" prepare-pilot-encoders 2>&1 | Tee-Object -FilePath $log
    exit $LASTEXITCODE
}

# Driver phases: one python call that runs its own worker pool. Its output goes to the phase
# log, and every python process is sampled for committed memory while it runs, because the
# driver's workers are not this script's children.
$driver = $null
switch ($Phase) {
    "pilot-pretrain" { $driver = "experiments\075_wide_region\pretrain.py"; $cliArgs = @("--pilot", "--arms", "X", "--seeds", "12", "13", "--out-dir", $pilotDir, "--workers", $Workers) }
    "pilot-train"    { $driver = "$exp\train.py"; $cliArgs = @("--pilot", "--seeds", "12", "13", "--n-updates", $Updates, "--max-len", $MaxLen, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--out-dir", $pilotDir, "--workers", $Workers) }
    "train"          { $driver = "$exp\train.py"; $cliArgs = @("--seeds") + $evalSeeds + @("--n-updates", $Updates, "--max-len", $MaxLen, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--out-dir", $outDir, "--workers", $Workers) }
}
if ($driver) {
    "script                      = $driver $cliArgs"
    "log                         = $log"
    ""
    $errLog = $log + ".err"
    $argList = @("-u", $driver) + @($cliArgs | ForEach-Object { "$_" })
    $proc = Start-Process -FilePath $py -ArgumentList $argList -PassThru -NoNewWindow -WorkingDirectory $repo -RedirectStandardOutput $log -RedirectStandardError $errLog
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
}

# Evaluation phases: evaluate.py runs one cell per call, so this file throttles a pool of
# processes itself. Each job is @(label, expected-output-file, args...).
$phaseDir = switch ($Phase) { "det" { $detDir } default { $outDir } }
$seeds = 0..11
$jobs = New-Object System.Collections.ArrayList
switch ($Phase) {
    "cont" {
        foreach ($d in 9, 11) {
            [void]$jobs.Add(@("J4V-X d$d s0", "exp079_J4V-X_d${d}_s0.json", "--arm", "J4V-X", "--depth", $d, "--seed", 0)) }
        [void]$jobs.Add(@("gatel X s0", "exp079_gatel_X_s0.json", "--gatel-x", "--seed", 0))
    }
    "eval" {
        foreach ($d in 11, 9) { foreach ($s in $seeds) {
            [void]$jobs.Add(@("J4V-S d$d s$s", "exp079_J4V-S_d${d}_s${s}.json", "--arm", "J4V-S", "--depth", $d, "--seed", $s)) } }
    }
    "det" {
        [void]$jobs.Add(@("J4V-S d9 s0", "exp079_J4V-S_d9_s0.json", "--arm", "J4V-S", "--depth", 9, "--seed", 0))
    }
    "rank" {
        foreach ($d in 11, 9) { foreach ($s in $seeds) {
            [void]$jobs.Add(@("rank d$d s$s", "exp079_rank_J-S4_d${d}_s${s}.json", "--rank", "--depth", $d, "--seed", $s)) } }
    }
}

"phase                       = $Phase, $($jobs.Count) cells, $Workers workers, into $phaseDir"
"log                         = $log"
""
$running = @()
$failed = 0
$peak = @{}
$labels = @{}
$maxPeak = 0
$stamp = { param($m) Add-Content -Path $log -Value $m; $m }
# Sample each live child's committed memory (PrivateMemorySize64) and log the peak at exit.
$sample = {
    foreach ($p in $running) {
        try {
            if (-not $p.HasExited) {
                $p.Refresh()
                $m = [int]($p.PrivateMemorySize64 / 1MB)
                if ($m -gt $peak[$p.Id]) { $peak[$p.Id] = $m }
            }
        } catch { }
    }
}
$reap = {
    foreach ($p in $running) {
        if ($p.HasExited -and $labels.ContainsKey($p.Id)) {
            $v = [int]$peak[$p.Id]
            & $stamp "peak commit MB $($labels[$p.Id]) $v" | Out-Null
            if ($v -gt $script:maxPeak) { $script:maxPeak = $v }
            $labels.Remove($p.Id)
        }
    }
}
foreach ($j in $jobs) {
    $target = Join-Path $phaseDir $j[1]
    if ($SkipExisting -and (Test-Path $target)) { continue }
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Workers) { & $sample; & $reap; Start-Sleep -Seconds 5 }
    & $reap
    foreach ($p in @($running | Where-Object { $_.HasExited })) { if ($p.ExitCode -ne 0) { $failed++ } }
    $running = @($running | Where-Object { -not $_.HasExited })
    & $stamp "start $($j[0])" | Out-Null
    $argList = @("-u", "$exp\evaluate.py") + @($j[2..($j.Count - 1)] | ForEach-Object { "$_" }) + @("--out-dir", $phaseDir)
    # Read .Handle at once: without it Windows PowerShell 5.1 reports ExitCode as $null after the
    # process exits, and every cell counts as failed whatever it did (EXP-074, 2026-10-08).
    $proc = Start-Process -FilePath $py -ArgumentList $argList -PassThru -NoNewWindow -WorkingDirectory $repo
    $null = $proc.Handle
    $running += $proc
    $labels[$proc.Id] = $j[0]
    $peak[$proc.Id] = 0
}
while (@($running | Where-Object { -not $_.HasExited }).Count -gt 0) { & $sample; & $reap; Start-Sleep -Seconds 5 }
& $reap
"max peak commit MB = $maxPeak"
foreach ($p in $running) { if ($p.ExitCode -ne 0) { $failed++ } }
if ($failed -gt 0) { Write-Error "$failed cells exited non-zero; do not aggregate."; exit 1 }
"PHASE OK"
```

- [ ] **Step 4: Write `chain079.ps1`**

```powershell
# EXP-079 full-run chain, after the dated amendment: cont (Gate 0(b) cells and Gate M's X
# continuity cell) -> Gate 0(b) -> train -> Gates L, E and M continuity -> eval -> det -> rank. Re-runnable with -From: training resumes by itself and evaluation
# phases skip existing cells. Launch it with OS-level redirection (playbook section 3):
#   ssh -n laptop 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\experiments\079_judge_trained_deep\chain079.ps1 > C:\Users\mlgbr\chain079.out 2>&1"'
param([string]$From = "cont")
$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py = Join-Path $repo ".venv\Scripts\python.exe"
$l = Join-Path $repo "experiments\079_judge_trained_deep\launch079.ps1"
$agg = "experiments\079_judge_trained_deep\aggregate.py"
$log = "C:\Users\mlgbr\chain079.log"
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo "src"
function Say($m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m; Add-Content -Path $log -Value $line; $line }
Say "chain start from $From at $(git rev-parse --short HEAD)"
$w = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','experiments/079_judge_trained_deep/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.AMENDMENT['train_workers'])+' '+str(a.AMENDMENT['eval_workers']))"
if ($LASTEXITCODE -ne 0 -or $w -notmatch "^\d+ \d+$") { Say "STOP: could not read train_workers and eval_workers from AMENDMENT (got '$w')"; exit 1 }
$trainW, $evalW = $w.Split(" ")
Say "workers: train $trainW, eval $evalW"
$order = @("cont","gate0b","train","gatele","eval","det","rank")
$go = $false
foreach ($p in $order) {
    if ($p -eq $From) { $go = $true }
    if (-not $go) { continue }
    if ($p -eq "gate0b") {
        $g = & $py $agg --stage cont 2>&1 | Out-String
        Say ("gate0b: " + $g.Trim())
        if ($g -notmatch "continuity.*: PASS") { Say "STOP: Gate 0(b) did not pass"; exit 2 }
        continue
    }
    if ($p -eq "gatele") {
        $g = & $py $agg --stage train 2>&1 | Out-String
        Say ("gates L, E, M: " + $g.Trim())
        if ($g -notmatch "GATE L \(VI worked\): PASS") { Say "STOP: Gate L did not pass, every claim is VOID"; exit 2 }
        if ($g -notmatch "GATE E \(encoders trained\): PASS") { Say "STOP: Gate E did not pass, every claim is VOID"; exit 2 }
        if ($g -notmatch "GATE M continuity \(X seed 0 Gate L re-measured here\): PASS") { Say "STOP: X's Gate L does not reproduce, so Gate M cannot be read; re-measure X before going on"; exit 2 }
        continue
    }
    $n = $evalW
    if ($p -eq "cont") { $n = 2 }
    if ($p -eq "train") { $n = $trainW }
    if ($p -eq "det") { $n = 1 }
    Say "phase $p workers $n"
    if ($p -eq "eval" -or $p -eq "rank") { & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $n -SkipExisting }
    else { & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $n }
    Say "phase $p exit $LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Say "STOP: phase $p failed"; exit 1 }
}
Say "chain done"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run (Bash tool `timeout: 600000`): `.venv/bin/python -m pytest tests/experiments/test_exp079.py -q -m "not slow"`
Expected: 41 passed, 2 deselected.

- [ ] **Step 6: Mutation check**

Change `"if ($MaxLen -ne 40)"` to `"if ($MaxLen -lt 1)"` in the launcher: `test_launcher_pins_the_walk_length` must FAIL. Change the chain's `"GATE L \(VI worked\): PASS"` to `"GATE L: PASS"`: `test_chain_order_and_gate_regexes_match_the_aggregator` must FAIL. Restore both; confirm the diff is clean of mutations; record the results.

- [ ] **Step 7: Update the test count in CLAUDE.md**

Run (Bash tool `timeout: 600000`): `.venv/bin/python -m pytest tests/ --collect-only -q | tail -1` and `.venv/bin/python -m pytest tests/ --collect-only -q -m "not slow" | tail -1`. In `CLAUDE.md`, change the line `**1122 as of 2026-10-14 (1097 not slow, 25 slow; EXP-078 code)**, verified by \`--collect-only\`.` to the plain entry `1122 2026-10-14 (EXP-078 code),` and add after it a bold line in the same form with the new totals and `EXP-079 code`, dated the day you run it, for example `**1165 as of 2026-10-19 (1138 not slow, 27 slow; EXP-079 code)**, verified by \`--collect-only\`.` with the numbers the two commands print.

- [ ] **Step 8: Commit**

```bash
git add experiments/079_judge_trained_deep/launch079.ps1 experiments/079_judge_trained_deep/chain079.ps1 tests/experiments/test_exp079.py CLAUDE.md
git commit -m "EXP-079: laptop launcher (max_len pinned at 40, amendment guard, memory sampling in driver phases) and chain; test count"
```

---

## After the plan (controller, not a task)

These are run steps from spec sections 6 and 10, listed so the executor knows where the build ends:

1. Whole-branch review, merge `--no-ff`, push. Pull on the laptop; `launch079.ps1 -Phase check`.
2. `-Phase pilot-encoders`; if it refuses, `-Phase pilot-pretrain -Workers 2`. Then `-Phase pilot-train -Workers 2` with OS-level redirection (about 13 h).
3. Copy the pilot records back, `aggregate.py --pilot-report`. If it prints STOP, report and stop: there is no amendment. Otherwise write the dated amendment (spec section 12): `AMENDMENT` (date, train and eval worker counts from the measured peak commit, spec section) and `GATE_L_THRESHOLD`, commit, push, `pm ask`.
4. On Michael's go: `chain079.ps1` with OS-level redirection.
```
