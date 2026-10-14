# EXP-078 Full Five-Move Tree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Within-cell checkpoint and resume for `evaluate_lookahead`, then EXP-078's driver, aggregator and laptop launcher, so a full 5-move tree (J5V-X) can run 8 to 12 h cells that survive a Windows Update restart.

**Architecture:** `evaluate_lookahead` gains `checkpoint`, `checkpoint_every` and `stop_after`, all off by default, so every existing path is byte-identical. `experiments/078_full_five_move_tree/` copies EXP-077's driver, aggregator and launcher, reads J4V-X from EXP-076's records and J4VG5-X and B5-X from EXP-077's, and computes EXP-077's validity itself (Claims 3 and 4 depend on it).

**Tech Stack:** Python 3.10 (VPS) / 3.13 (laptop), PyTorch, pytest; PowerShell 5.1 on the laptop.

**Spec:** `docs/superpowers/specs/2026-10-14-exp078-full-five-move-tree-design.md` (read sections 2 to 9 before any task).

## Global Constraints

- Run Python only via the venv: `.venv/bin/python`. In a git worktree prefix every command with `PYTHONPATH=src`.
- **Always pass an explicit Bash `timeout`; the default is 120 s.** Call test commands directly and let them block (600000 ms covers every file named here). Silence mid-run is buffered output, not a hang. Never background a test run.
- No em-dashes anywhere. Plain commit messages (no `Co-Authored-By`, no "Generated with"). Stage explicit paths; never `git add -A`.
- Never write an assertion that cannot fail; a new test must fail against the pre-fix code. Never weaken a test to make it pass.
- `checkpoint=None` must leave `evaluate_lookahead`'s behaviour AND its returned keys exactly as today.
- Checkpoint interval N = 10 states in the driver; forced resume for Gate 0(a) after 50 states. Alpha 0.05 / 4 = 0.0125. These are constants, not CLI choices.

## Review Focus

1. **A checkpoint from a different cell** (another seed, depth, arm or state count) loaded by mistake: it must be refused, not resumed. Pinned in Task 1.
2. **A crash between the final checkpoint and the record write**: re-running must produce the record without re-evaluating, and the checkpoint is deleted only after the record exists. Pinned in Tasks 1 and 2.
3. **A torn checkpoint write** (killed mid-write): writes go to a temp file then `os.replace`, so the previous checkpoint survives. Pinned by reading the code in review; Task 1 asserts no `.tmp` file is left behind.
4. **EXP-077 not finished or VOID when EXP-078 aggregates**: Claims 3 and 4 must print VOID with the reason, and Claims 1 and 2 must still print. Pinned in Task 3.
5. **The launcher reading the redirector** instead of the worker: carried over from EXP-077 with its under-50 MB warning. Pinned in Task 4.

---

### Task 1: Within-cell checkpoint and resume in `evaluate_lookahead`

**Files:**
- Modify: `src/neuromorphic/training/lookahead.py` (`evaluate_lookahead`)
- Test: `tests/training/test_lookahead_resume.py` (new)

**Interfaces:**
- Produces: `evaluate_lookahead(..., checkpoint=None, checkpoint_every=10, stop_after=None)`.
  - `checkpoint`: a path. If it exists it is loaded and the loop continues from its index; it is (re)written atomically every `checkpoint_every` completed states and once after the last state.
  - `stop_after`: if set, after that many states completed IN THIS CALL, write the checkpoint and return `None` (a simulated kill, for tests and Gate 0(a)).
  - With `checkpoint` set, the result gains `"resumes"` (how many times this cell was loaded from a checkpoint) and `"elapsed_s"` (seconds spent in the loop, summed over sessions). With `checkpoint=None` neither key exists.
  - `trace` together with `checkpoint` raises `ValueError` (a trace cannot be resumed).
  - A checkpoint whose identity differs from the call's (`depth, mode, k, k_goal, beam_width, n_states, rng_seed, imag_seed, no_revisit`) raises `ValueError` naming the field.

- [ ] **Step 1: Write the failing tests** in `tests/training/test_lookahead_resume.py`:

```python
"""EXP-078 seam: resume a cell from a checkpoint. Each docstring names the bug it catches."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuromorphic.envs.cube import N_ACTIONS
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
    reads_states = True

    def __call__(self, states, generator):
        return torch.tensor([float(hash(s) % 1_000_003) for s in states])


STATES = _shell(5, 7)


def _run(agent, head, mode="E", critic=None, **kw):
    # Mode E follows the policy's logits, which are drawn from the evaluation generator, so
    # a resume that loses the generator state changes the outcome. Mode C ignores the logits
    # and could not see it.
    return la.evaluate_lookahead(agent, head, STATES, depth=5, mode=mode, k=2,
                                 generator=torch.Generator().manual_seed(11), rng_seed=11,
                                 imag_seed=11, critic=critic, no_revisit=True, **kw)


OUTCOME = ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
           "goal_fired_frac", "first_fire_len", "fallback_frac", "greedy_modal_action_frac")


def test_default_path_has_no_new_keys():
    """Catches the resume seam leaking into every record: without a checkpoint the result has
    no resumes or elapsed_s key."""
    agent, head = _agent_and_head()
    r = _run(agent, head)
    assert "resumes" not in r and "elapsed_s" not in r


@pytest.mark.parametrize("m", [3, 4])
def test_killed_and_resumed_equals_straight_through(tmp_path, m):
    """THE RESUME TEST. Catches a resume that does not restore the evaluation generator, a
    counter, or the state index: a cell stopped after m states (3 is not a multiple of the
    interval 2, 4 is) and resumed equals the same cell run straight through in every outcome
    field. Uses the real agent, so hidden agent state between episodes would also show."""
    agent, head = _agent_and_head()
    ref = _run(agent, head)
    ck = tmp_path / "cell.ckpt"
    agent, head = _agent_and_head()
    assert _run(agent, head, checkpoint=ck, checkpoint_every=2, stop_after=m) is None
    assert ck.exists()
    agent, head = _agent_and_head()
    got = _run(agent, head, checkpoint=ck, checkpoint_every=2)
    for f in OUTCOME:
        assert got[f] == ref[f], f
    assert got["resumes"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_must_restore_the_generator(tmp_path):
    """Catches the resume test being blind to the generator: with the saved generator state
    replaced by another stream's, the resumed cell must differ from the straight run."""
    agent, head = _agent_and_head()
    ref = _run(agent, head)
    ck = tmp_path / "cell.ckpt"
    agent, head = _agent_and_head()
    _run(agent, head, checkpoint=ck, checkpoint_every=2, stop_after=3)
    saved = torch.load(ck, weights_only=False)
    saved["gen"] = torch.Generator().manual_seed(999).get_state()
    torch.save(saved, ck)
    agent, head = _agent_and_head()
    got = _run(agent, head, checkpoint=ck, checkpoint_every=2)
    assert any(got[f] != ref[f] for f in OUTCOME)


def test_a_finished_checkpoint_returns_without_re_evaluating(tmp_path):
    """Catches a crash after the final checkpoint forcing a full re-run: a second call on a
    finished checkpoint returns the same result and evaluates no state (the critic is never
    called)."""
    agent, head = _agent_and_head()
    ck = tmp_path / "cell.ckpt"
    first = _run(agent, head, mode="C", critic=_HashCritic(), checkpoint=ck, checkpoint_every=2)

    class _Boom(_HashCritic):
        def __call__(self, states, generator):
            raise AssertionError("re-evaluated a finished cell")

    again = la.evaluate_lookahead(agent, head, STATES, depth=5, mode="C", k=2,
                                  generator=torch.Generator().manual_seed(11), rng_seed=11,
                                  imag_seed=11, critic=_Boom(), no_revisit=True,
                                  checkpoint=ck, checkpoint_every=2)
    for f in OUTCOME:
        assert again[f] == first[f], f


def test_a_checkpoint_from_another_cell_is_refused(tmp_path):
    """Catches resuming the wrong cell: a checkpoint written for k 2 is refused by a k 3 call,
    and one written for 7 states by a 6-state call."""
    agent, head = _agent_and_head()
    ck = tmp_path / "cell.ckpt"
    _run(agent, head, checkpoint=ck, checkpoint_every=2, stop_after=2)
    with pytest.raises(ValueError, match="k"):
        la.evaluate_lookahead(agent, head, STATES, depth=5, mode="E", k=3,
                              generator=torch.Generator().manual_seed(11), rng_seed=11,
                              imag_seed=11, critic=_HashCritic(), no_revisit=True,
                              checkpoint=ck)
    with pytest.raises(ValueError, match="n_states"):
        la.evaluate_lookahead(agent, head, STATES[:6], depth=5, mode="E", k=2,
                              generator=torch.Generator().manual_seed(11), rng_seed=11,
                              imag_seed=11, critic=_HashCritic(), no_revisit=True,
                              checkpoint=ck)


def test_trace_with_checkpoint_is_refused(tmp_path):
    """Catches a resumed trace silently missing its first episodes."""
    agent, head = _agent_and_head()
    with pytest.raises(ValueError):
        _run(agent, head, checkpoint=tmp_path / "c.ckpt", trace=[])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_lookahead_resume.py -q` (timeout 600000)
Expected: FAIL (`unexpected keyword argument 'checkpoint'`). `test_default_path_has_no_new_keys` may already pass; that is expected for a regression pin.

- [ ] **Step 3: Implement** in `evaluate_lookahead`:
  - New keywords `checkpoint=None, checkpoint_every=10, stop_after=None`. If `checkpoint is not None and trace is not None`: `raise ValueError("a trace cannot be resumed from a checkpoint")`.
  - Collect the loop's running state in one dict `acc` (solved, steps_solved, eval_revisits, eval_steps, fired_moves, fallback_moves, modal_fracs, fire_hist, judge_evals_total, searched_moves) so it can be saved and restored as a unit; keep the arithmetic identical.
  - `ident = {"depth", "mode", "k", "k_goal", "beam_width", "n_states": len(states), "rng_seed", "imag_seed", "no_revisit"}` (values from the call).
  - On start, if `checkpoint` exists: `ck = torch.load(checkpoint, weights_only=False)`; for each identity field that differs, `raise ValueError(f"checkpoint {field} is {ck['ident'][field]!r}, this call has {ident[field]!r}")`; restore `acc`, `start = ck["next"]`, `generator.set_state(ck["gen"])`, `resumes = ck["resumes"] + 1`, `elapsed = ck["elapsed_s"]`. Otherwise `start = 0`, `resumes = 0`, `elapsed = 0.0`.
  - Loop `for i in range(start, len(states))` with `state = states[i]` (the per-state imagination seed already uses `i`).
  - `_save(next_i)`: `torch.save({"ident": ident, "acc": acc, "next": next_i, "gen": generator.get_state(), "resumes": resumes, "elapsed_s": elapsed + (time.monotonic() - t0)}, tmp)` then `os.replace(tmp, checkpoint)` with `tmp = checkpoint.with_name(checkpoint.name + ".tmp")`. Call it after state i when `(i + 1) % checkpoint_every == 0`, and once after the loop. With `stop_after`, count states done in this call; when it reaches `stop_after`, `_save(i + 1)` and `return None`.
  - Build the result from `acc` exactly as now; if `checkpoint is not None`, add `"resumes"` and `"elapsed_s"`.
  - Accept `checkpoint` as `str` or `Path` (`Path(checkpoint)`).

- [ ] **Step 4: Run the new tests and the regression set** (each with timeout 600000):
`.venv/bin/python -m pytest tests/training/test_lookahead_resume.py tests/training/test_lookahead_beam.py tests/training/test_lookahead_k_goal.py -q`
`.venv/bin/python -m pytest tests/training -q -m "not slow" -k "lookahead"`
`.venv/bin/python -m pytest tests/experiments/test_exp077.py tests/experiments/test_exp076.py -q -m "not slow"`
Expected: all PASS. If the resume test fails because the agent carries state between episodes, do not weaken it: report which state (and whether `evaluate_states` resets it) as BLOCKED.

- [ ] **Step 5: Mutation check.** Temporarily (a) skip `generator.set_state` (the corrupted-generator test and the mode-E resume test must fail), (b) restart at index 0 instead of `ck["next"]`, (c) drop one `acc` counter from the restore; confirm a test fails for each; restore and confirm `git diff` shows only the intended change.

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/training/lookahead.py tests/training/test_lookahead_resume.py
git commit -m "Look-ahead: within-cell checkpoint and resume for evaluate_lookahead, off by default (EXP-078)"
```

---

### Task 2: EXP-078 driver

**Files:**
- Create: `experiments/078_full_five_move_tree/evaluate.py`
- Test: `tests/experiments/test_exp078.py` (new; Task 3 appends)

**Interfaces:**
- Consumes: Task 1's `evaluate_lookahead(..., checkpoint, checkpoint_every, stop_after)`; EXP-075's judge loading as in `experiments/077_beam_search/evaluate.py`.
- Produces: `ARMS = {"J5V-X": ("C", 5, 5, None), "J4V-X": ("C", 4, 4, None)}` (J4V-X only for Gate 0(b)); `DEPTHS = (9, 11)`; `CKPT_EVERY = 10`; `FORCED_STOP = 50`; `RANK_K = 5`; `E76_OUT`, `E77_OUT`, `E77_DET`; `record_name(arm, depth, seed) -> "exp078_{arm}_d{depth}_s{seed}.json"`; `rank_record_name(depth, seed) -> "exp078_rank_J-X5_d{depth}_s{seed}.json"`; `ckpt_path(out_dir, arm, depth, seed) -> out_dir / "ckpt" / (record stem + ".ckpt")`; `run_cell(arm, depth, seed, out_dir, limit_states=None, judge_dir=None, stop_after=None) -> dict | None`; `rank_cell(depth, seed, out_dir, judge_dir=None, limit_states=None, provider=None) -> dict`.
  - `run_cell` always passes `checkpoint=ckpt_path(...)`, `checkpoint_every=CKPT_EVERY`; writes the record (adds `depth, seed, arm, fired5, wall_s, git_commit, limit_states, judge_updates`, with `wall_s = round(res["elapsed_s"], 1)`), THEN deletes the checkpoint; returns `None` and writes nothing when stopped. CLI `--stop-after N` passes through.

- [ ] **Step 1: Write the failing tests** in `tests/experiments/test_exp078.py`:

```python
"""EXP-078: driver, aggregator and launcher. Each docstring names the bug it catches."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "experiments" / "078_full_five_move_tree"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("exp078_evaluate", HERE / "evaluate.py")


def test_arms_and_constants_are_the_spec():
    """Catches a drifted arm table, interval or forced-stop point (spec sections 3, 4, 7)."""
    assert ev.ARMS == {"J5V-X": ("C", 5, 5, None), "J4V-X": ("C", 4, 4, None)}
    assert ev.DEPTHS == (9, 11) and ev.CKPT_EVERY == 10 and ev.FORCED_STOP == 50
    assert ev.RANK_K == 5


def test_j4v_x_with_checkpointing_equals_exp076_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH. Catches checkpointing changing an outcome: J4V-X through this
    driver (checkpoint on) equals EXP-076's driver on the first 3 depth-9 states of seed 0."""
    e76 = _load("exp076_evaluate_t", REPO / "experiments" / "076_deeper_tree" / "evaluate.py")
    mine = ev.run_cell("J4V-X", 9, 0, tmp_path / "a", limit_states=3)
    ref = e76.run_cell("J4V-X", 9, 0, tmp_path / "b", limit_states=3)
    for f in ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
              "goal_fired_frac", "first_fire_len", "fallback_frac"):
        assert mine[f] == ref[f], f
    assert mine["resumes"] == 0


def test_stopped_cell_writes_no_record_and_resumes_to_the_same_record(tmp_path):
    """Catches a stopped cell leaving a record a launcher would skip as done, a deleted
    checkpoint before the record exists, and a resume that differs from a straight run."""
    straight = ev.run_cell("J4V-X", 9, 0, tmp_path / "s", limit_states=3)
    out = tmp_path / "r"
    assert ev.run_cell("J4V-X", 9, 0, out, limit_states=3, stop_after=2) is None
    assert not (out / ev.record_name("J4V-X", 9, 0)).exists()
    assert ev.ckpt_path(out, "J4V-X", 9, 0).exists()
    resumed = ev.run_cell("J4V-X", 9, 0, out, limit_states=3)
    assert (out / ev.record_name("J4V-X", 9, 0)).exists()
    assert not ev.ckpt_path(out, "J4V-X", 9, 0).exists()
    for f in ("solved", "success_rate", "mean_steps", "first_fire_len", "judge_evals"):
        assert resumed[f] == straight[f], f
    assert resumed["resumes"] == 1


def test_j5v_x_records_its_depths_and_fired5(tmp_path):
    """Catches a J5V-X record that does not say what ran (k 5, goal check 5, 7776 leaves)."""
    r = ev.run_cell("J5V-X", 9, 0, tmp_path, limit_states=1)
    assert (r["mode"], r["k"], r["k_goal"]) == ("C", 5, 5)
    assert r["judge_evals"] == 7776 and r["fired5"] == r["first_fire_len"].get("5", 0)


def test_unknown_arm_exits_cleanly(tmp_path):
    """Catches an arm typo running something unregistered."""
    with pytest.raises(SystemExit):
        ev.run_cell("J6V-X", 9, 0, tmp_path, limit_states=1)
```

- [ ] **Step 2: Run to verify they fail** (`tests/experiments/test_exp078.py`, timeout 600000).

- [ ] **Step 3: Implement** `experiments/078_full_five_move_tree/evaluate.py` by copying `experiments/077_beam_search/evaluate.py` and changing: docstring (EXP-078, spec path); ARMS and constants as in Interfaces; `record_name` prefix `exp078_`; `ckpt_path`; `run_cell` per Interfaces (keep `fired5`; `wall_s` from `elapsed_s`; delete the checkpoint with `.unlink()` only after `_write`); restore a `rank_cell` from `experiments/076_deeper_tree/evaluate.py` with `RANK_K = 5` and `kind "J-X5"` (BFS `ExactBFSDistance(max_depth=None)`, the yardstick only); CLI `--arm` or `--rank` (exactly one), `--stop-after`.

- [ ] **Step 4: Run tests** (timeout 600000). Expected: PASS. The J5V-X test evaluates one depth-9 state with 7776 leaves per move; allow a few minutes.

- [ ] **Step 5: Commit**

```bash
git add experiments/078_full_five_move_tree/evaluate.py tests/experiments/test_exp078.py
git commit -m "EXP-078: driver with within-cell checkpoints for J5V-X, J4V-X for Gate 0(b), and the k 5 rank cell"
```

---

### Task 3: EXP-078 aggregator

**Files:**
- Create: `experiments/078_full_five_move_tree/aggregate.py`
- Test: `tests/experiments/test_exp078.py` (append)

**Interfaces:**
- Consumes: Task 2's names; `experiments/077_beam_search/aggregate.py` (as `a77`: `claim_verdict` pattern, `_read`, `check_eval_record` shape, `stage_cont`, `gate0a`, `gate_d`, `gate_b`, `load_records`) through importlib.
- Produces: `ALPHA = 0.05 / 4`, `SMALL = 0.01`, `AMENDMENT: dict | None = None`, `exp077_valid() -> tuple[bool, str]`, `load_records(out_dir)` with keys `("J5V-X", d)`, `("J4V-X", d)` from EXP-076, `("J4VG5-X", d)` and `("B5-X", d)` from EXP-077's `outputs/`; `gate_d(recs)` (J5V-X `first_fire_len` keys other than "5"); `gate0a(out_dir, det_dir)` (J5V-X d9 s0; every field equal except `wall_s`, `git_commit`, `resumes`, `elapsed_s`); `stage_cont(out)`; `primary_verdicts(rates, gate_ok, e77_ok, seeds)` returning claim1..claim4; `report_lines(...)`; CLI `--pilot-report`, `--stage cont`, `--det-dir`.

- [ ] **Step 1: Write the failing tests** (append):

```python
ag = _load("exp078_aggregate", HERE / "aggregate.py")


def _rates(**over):
    rates = {(a, d): {s: 0.1 for s in range(12)}
             for a in ("J5V-X", "J4V-X", "J4VG5-X", "B5-X") for d in ev.DEPTHS}
    for key, val in over.items():
        arm, d = key.rsplit("_d", 1)
        rates[(arm.replace("_", "-"), int(d))] = {s: val + 0.001 * s for s in range(12)}
    return rates


def test_alpha_is_four_way():
    """Catches the three-claim alpha carried over from EXP-077."""
    assert ag.ALPHA == pytest.approx(0.0125) and ag.SMALL == 0.01


def test_four_claims_compare_the_registered_pairs():
    """Catches a claim wired to the wrong arm or depth (spec section 8)."""
    v = ag.primary_verdicts(_rates(J5V_X_d11=0.3, J5V_X_d9=0.5), True, True, range(12))
    assert [v[c][0] for c in ("claim1", "claim2", "claim3", "claim4")] == ["CONFIRMED"] * 4
    v = ag.primary_verdicts(_rates(J5V_X_d11=0.3, B5_X_d11=0.3), True, True, range(12))
    assert v["claim4"][0] == "REFUTED" and v["claim1"][0] == "CONFIRMED"


def test_exp077_void_voids_only_claims_3_and_4():
    """Catches EXP-077's validity being ignored, or voiding claims that do not read its arms."""
    v = ag.primary_verdicts(_rates(J5V_X_d11=0.3, J5V_X_d9=0.5), True, False, range(12))
    assert v["claim1"][0] == "CONFIRMED" and v["claim2"][0] == "CONFIRMED"
    assert v["claim3"][0] == "VOID" and v["claim4"][0] == "VOID"


def test_gate0a_ignores_only_the_session_fields(tmp_path):
    """Catches Gate 0(a) failing on a resumed det cell (resumes, elapsed_s differ by design) or
    passing a real difference."""
    rec = {"solved": 80, "wall_s": 1.0, "git_commit": "a", "resumes": 0, "elapsed_s": 1.0}
    (tmp_path / "o").mkdir(); (tmp_path / "d").mkdir()
    name = ev.record_name("J5V-X", 9, 0)
    (tmp_path / "o" / name).write_text(json.dumps(rec))
    (tmp_path / "d" / name).write_text(json.dumps({**rec, "wall_s": 9.0, "git_commit": "b",
                                                   "resumes": 1, "elapsed_s": 9.0}))
    assert ag.gate0a(tmp_path / "o", tmp_path / "d") == "PASS"
    (tmp_path / "d" / name).write_text(json.dumps({**rec, "solved": 79}))
    assert ag.gate0a(tmp_path / "o", tmp_path / "d") == "FAIL"


def test_gate_d_is_the_first_fire_clause():
    """Catches the first-fire instrument check missing for J5V-X."""
    recs = {("J5V-X", d): {s: {"first_fire_len": {"5": 3}} for s in range(12)} for d in ev.DEPTHS}
    assert ag.gate_d(recs) == []
    recs[("J5V-X", 11)][4] = {"first_fire_len": {"4": 1, "5": 2}}
    assert ag.gate_d(recs) == [(11, 4)]


def test_exp077_valid_reports_why_when_its_run_is_missing(tmp_path, monkeypatch):
    """Catches Claims 3 and 4 reading EXP-077 arms that never finished: with EXP-077's outputs
    pointed at an empty directory, exp077_valid is (False, reason)."""
    monkeypatch.setattr(ag.ev, "E77_OUT", tmp_path)
    ok, why = ag.exp077_valid()
    assert ok is False and why
```

- [ ] **Step 2: Run to verify they fail** (timeout 600000).

- [ ] **Step 3: Implement** `experiments/078_full_five_move_tree/aggregate.py` by copying `experiments/077_beam_search/aggregate.py` and changing:
  - docstring, `ev` = this experiment's `evaluate.py`, `a77` = EXP-077's `aggregate.py` (importlib), `ALPHA = 0.05 / 4`;
  - `check_eval_record` accepts J5V-X and J4V-X from `ev.ARMS`, and J4VG5-X and B5-X from `a77.ev.ARMS` (delegate those to `a77.check_eval_record`);
  - `load_records` per Interfaces (J4V-X from `ev.E76_OUT` as `exp076_J4V-X_...`; J4VG5-X and B5-X from `ev.E77_OUT` as `exp077_...`), each through an identity-checked read;
  - `exp077_valid()`: try `a77.stage_cont(ev.E77_OUT) == "PASS"`, `a77.gate0a(ev.E77_OUT, ev.E77_DET) == "PASS"`, `recs77 = a77.load_records(ev.E77_OUT)`, `not a77.gate_d(recs77)`, `a77.gate_b(recs77)["ok"]`; return `(True, "")` if all hold, else `(False, "<which failed>")`; catch `SystemExit` (missing records) as `(False, str(e))`;
  - `gate_d` per Interfaces; delete `gate_b` (no beam arm runs here); `gate0a` per Interfaces; `stage_cont` compares this run's J4V-X seed 0 d9/d11 with EXP-076's;
  - `primary_verdicts(rates, gate_ok, e77_ok, seeds)`: claim1 `("J5V-X", 11)` vs `("J4V-X", 11)`; claim2 the same at 9; claim3 `("J5V-X", 11)` vs `("J4VG5-X", 11)`; claim4 `("J5V-X", 11)` vs `("B5-X", 11)`; claims 3 and 4 get `gate_ok and e77_ok`;
  - Gate R: report per depth the k 5 hit and chance from `exp078_rank_J-X5_*` records against EXP-076's k 4 (`experiments/076_deeper_tree/outputs/exp076_rank_J-X4_*`), reported only;
  - `report_lines`: gates (0(a), 0(b), D, EXP-077 validity with its reason), means per arm and depth (success, mean steps, optimality; `judge_evals` only for J5V-X), seconds per state per arm, the four claims with 10-seed lines, secondaries (claims 3 and 4 at depth 9; Gate R);
  - `pilot_report`: seconds per state of the 20-state J5V-X pilot record, projected depth-11 cell hours, total eval cell hours with the EXP-076 J4V-X depth-9/depth-11 wall ratio.

- [ ] **Step 4: Run tests** (whole file, timeout 600000). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/078_full_five_move_tree/aggregate.py tests/experiments/test_exp078.py
git commit -m "EXP-078: aggregator with four claims at alpha 0.0125, EXP-077's validity computed for Claims 3 and 4, Gate 0(a) ignoring session fields"
```

---

### Task 4: EXP-078 laptop launcher and chain

**Files:**
- Create: `experiments/078_full_five_move_tree/launch078.ps1`, `experiments/078_full_five_move_tree/chain078.ps1`
- Test: `tests/experiments/test_exp078.py` (append)

**Interfaces:**
- Consumes: Task 2's CLI (`--arm/--rank --depth --seed --out-dir --limit-states --stop-after`), Task 3's `AMENDMENT` and `--stage cont`.
- Produces: phases `check, pilot, cont, eval, det, rank`.

- [ ] **Step 1: Write the failing tests** (append):

```python
def _ps(name):
    return (HERE / name).read_text()


def test_launcher_carries_the_fixes_and_the_registered_cells():
    """Catches a launcher missing a carried fix, the resume seam check, or a registered cell."""
    t = _ps("launch078.ps1")
    for s in ("$null = $proc.Handle", "AMENDMENT is unset", "Workers -gt 6", "ParentProcessId",
              "under 50 MB", "'workers'", "checkpoint"):
        assert s in t, s
    assert '"J5V-X"' in t and '"J4V-X d9 s0"' in t and '"J4V-X d11 s0"' in t
    assert '"J5V-X d9 s0"' in t and "--stop-after" in t


def test_eval_phase_does_not_skip_a_cell_that_only_has_a_checkpoint():
    """Catches -SkipExisting keyed on the checkpoint: a cell is done only when its RECORD
    exists, so an in-flight cell is relaunched and resumes."""
    t = _ps("launch078.ps1")
    assert "SkipExisting" in t and ".ckpt" not in t


def test_chain_order_and_the_forced_resume_det():
    """Catches the chain running eval before Gate 0(b), rank before eval, or det without its
    forced stop and relaunch."""
    t = _ps("chain078.ps1")
    assert t.index('"cont"') < t.index('"gate"') < t.index('"eval"') < t.index('"det"') < t.index('"rank"')
    assert "--stage cont" in t and "STOP" in t and "'workers'" in t
```

- [ ] **Step 2: Run to verify they fail** (timeout 600000).

- [ ] **Step 3: Implement.**
  - `launch078.ps1`: copy `experiments/077_beam_search/launch077.ps1`. Change `$exp`; the seam check tests `'checkpoint' in inspect.signature(l.evaluate_lookahead).parameters`; the presence check counts 12 trained X judges and 24 EXP-076 J4V-X records (unchanged from EXP-077); `ValidateSet("check","pilot","cont","eval","det","rank")`. Jobs, with literal labels where the test greps them:
    - pilot: `"J5V-X d11 s0 pilot"` with `--limit-states 20`;
    - cont: `"J4V-X d9 s0"`, `"J4V-X d11 s0"`;
    - eval: J5V-X, depth 11 then 9, seeds 0..11 (`"J5V-X"` appears as a literal in the loop);
    - det: `"J5V-X d9 s0"` run TWICE in sequence into `outputs_det`: first with `--stop-after 50` (it exits 0 having written only a checkpoint), then without, so the second resumes. Run det with 1 worker and wait for the first to exit before starting the second;
    - rank: depth 11 then 9, seeds 0..11, `--rank`.
    `-SkipExisting` stays keyed on the record file (the job's expected output), never on a checkpoint.
  - `chain078.ps1`: copy `experiments/077_beam_search/chain077.ps1`; launcher `launch078.ps1`, log `C:\Users\mlgbr\chain078.log`, order `@("cont","gate","eval","det","rank")`, gate `aggregate.py --stage cont` in this folder, eval and rank workers from `AMENDMENT['workers']` as EXP-077's chain reads them, det 1 worker without `-SkipExisting`.

- [ ] **Step 4: Run tests** (whole file, timeout 600000). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/078_full_five_move_tree/launch078.ps1 experiments/078_full_five_move_tree/chain078.ps1 tests/experiments/test_exp078.py
git commit -m "EXP-078: laptop launcher with checkpointed cells and a forced-resume det, and the cont-gate-eval-det-rank chain"
```
