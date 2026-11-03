# EXP-085 Outward Walks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue X16k from update 16,000 to 24,000 on the desktop GPU in two arms (C: walks up to 14; O: max walk length ramping 14 to 30 over 4000 updates), search both at depths 9 and 11, and read a one-line verdict with Gates C, T and H.

**Architecture:** One new experiment folder, `experiments/085_outward_walks/`, built on EXP-082's continuation (Gate C, legs, records), EXP-084's device path and EXP-080's statistics. The only library change is that `vi.train_judge` accepts a per-update max walk length, with an `int` keeping the old path byte-identical. Gate C and the laptop copy run on the VPS; rebase, pre-flight, training, search and Gate H run on the desktop through one PowerShell chain.

**Tech Stack:** Python 3, PyTorch (CPU on the VPS, ROCm 2.9.1 on the desktop), pytest, PowerShell 5.

**Spec:** `docs/superpowers/specs/2026-11-03-exp085-outward-walks-design.md`

## Global Constraints

- Seeds 0 to 11; start update 16,000; end 24,000; legs (20000, 23751, 24000); records at 20000 and 24000, a full checkpoint snapshot at 23751.
- Arm C max length 14 at every update. Arm O: `14 + (16 * (u - 16000)) // 4000` for 16000 <= u < 20000, then 30.
- Both arms exclude EXP-075's exclusion set UNION the seed's probe states, from update 16,000.
- EXP-082 settings unchanged: batch 1000, sync_every 100, probe_every 250, draws 1, arm W optimizer, readout wide, hidden 512.
- Search: J4V, mode C, k 4, k_goal 4, no-revisit, EXP-075 held-out sets and streams, 200 states, depths (9, 11), on the device.
- Gate T bar `a80.GATE_T_THRESHOLD` (0.1356358024691358) on each arm. Claim 1 bar 0.03, Claim 2 bar `a80.CLAIM_MIN` (0.05), alpha 0.025, guard trips at mean <= -0.05 with drop p < 0.025.
- Gate H selection: `random.Random(85)`; 2 search cells per depth, 2 training-replay seeds per arm.
- Verdict strings exactly as spec section 8.
- CPU path byte-identical: `tests/training/test_device_port.py` golden pins must keep passing.
- No em-dashes in repo files; plain commit messages, no trailer; stage explicit paths; `PYTHONPATH=src` in a worktree.
- Never write into `experiments/082_x_long_budget/` or `experiments/075_wide_region/`.

## Review Focus

1. A chain relaunched after a crash mid-leg: every stage must resume from checkpoints and records, never re-record a leg or re-run a finished cell (tests in Tasks 4 and 5).
2. A rebase run twice, or on a directory training already advanced: it must refuse, never reset a started arm to 16,000 (Task 3).
3. An arm-O checkpoint resumed mid-ramp: the max length must follow the absolute update number, not the number of updates since resume (Task 1 and Task 4).
4. A search or replay record from the wrong device, arm, update count or a `limit_states` smoke run reaching the aggregator (Task 6).
5. Gate H (b) when pre-flight says GPU training is not deterministic: the verdict must say "unavailable", never PASS (Task 6).

---

### Task 1: per-update max walk length in `train_judge`

**Files:**
- Modify: `src/neuromorphic/training/value_iteration.py` (`train_judge`)
- Test: `tests/training/test_value_iteration.py`

**Interfaces:**
- Produces: `train_judge(..., max_len: int | Callable[[int], int], ...)`. A callable is called with the absolute update index each update; an `int` is used as before.

- [ ] **Step 1: Write the failing tests**

```python
def test_max_len_schedule_is_called_with_the_absolute_update(tmp_path, monkeypatch):
    seen = []
    real = vi.random_walk_states
    monkeypatch.setattr(vi, "random_walk_states",
                        lambda n, ml, *a, **k: (seen.append(ml), real(n, ml, *a, **k))[1])
    sched = lambda u: 2 + u            # distinct per update
    # tiny judge as the module's other tests build it; 3 updates, then resume for 2 more
    _train(tmp_path, n_updates=3, max_len=sched)
    _train(tmp_path, n_updates=5, max_len=sched)
    assert seen == [2, 3, 4, 5, 6]

def test_int_max_len_path_is_unchanged(tmp_path):
    # same seed, int 4 versus a constant schedule lambda u: 4: identical judge.pt bytes
    a = _train(tmp_path / "a", n_updates=3, max_len=4)
    b = _train(tmp_path / "b", n_updates=3, max_len=lambda u: 4)
    assert (tmp_path / "a" / "judge.pt").read_bytes() == (tmp_path / "b" / "judge.pt").read_bytes()
```

`_train` is a local helper building the smallest judge the file already uses and calling `vi.train_judge` with `probe_every=1`, `sync_every=1`, `batch=4`.

- [ ] **Step 2: Run, expect FAIL** (`TypeError` from `rng.randint(1, <function>)`): `.venv/bin/python -m pytest tests/training/test_value_iteration.py -q -k max_len`
- [ ] **Step 3: Implement**: inside the loop, `ml = max_len(update) if callable(max_len) else max_len`. Docstring line naming EXP-085.
- [ ] **Step 4: Run Step 2's command, expect PASS; then `.venv/bin/python -m pytest tests/training/test_device_port.py -q` (timeout 600000), expect the golden pins to pass.**
- [ ] **Step 5: Commit** `src/neuromorphic/training/value_iteration.py tests/training/test_value_iteration.py`: "EXP-085: train_judge takes a per-update max walk length".

### Task 2: cells and Gate C on the VPS

**Files:**
- Create: `experiments/085_outward_walks/cells.py`, `experiments/085_outward_walks/gatec.py`
- Test: `tests/experiments/test_exp085.py`

**Interfaces:**
- Produces in `cells.py`: `ARMS = ("C", "O")`, `START = 16000`, `END = 24000`, `RAMP_END = 20000`, `LEGS = (20000, 23751, 24000)`, `RECORD_LEGS = (20000, 24000)`, `SNAP_UPDATE = 23751`, `MAX_LEN_C = 14`, `MAX_LEN_O = 30`, `DEPTHS = (9, 11)`, `SEEDS = range(12)`, `H_SEED = 85`, `OUT`, `E75_OUT`, `E82_OUT`, `max_len_schedule(arm) -> Callable[[int], int]`, `start_dir(seed, out=None)` = `out/start/judge_X_s<seed>`, `ckpt_dir(arm, seed, out=None)` = `out/arm_<arm>/judge_X_s<seed>` (EXP-075 layout, so `e75.load_judge("X", seed, out/arm_<arm>)` reads it), `snap_dir(arm, seed, out=None)` = `ckpt_dir(...)/snap_u23751`, `gatec_name(seed)` `exp085_gatec_s<seed>.json`, `record_name(arm, update, seed)` `exp085_<arm>_u<update>_s<seed>.json`, `search_name(arm, depth, seed)` `exp085_J4V-<arm>_d<depth>_s<seed>.json`, `h_search_name(arm, depth, seed)` `exp085_H_J4V-<arm>_d<depth>_s<seed>.json`, `h_train_name(arm, seed)` `exp085_Hb_<arm>_s<seed>.json`, `PREFLIGHT_NAME` `exp085_preflight.json`, `REBASE_NAME(arm, seed)` `exp085_rebase_<arm>_s<seed>.json`.
- Produces in `gatec.py`: `sets(seed) -> (exclude, probe, distance)` with `exclude` = exclusion set UNION probe states; `gate_c(seed, src_root, out=cells.OUT, tracked=..., blob=...) -> dict`; `passed(seed, out)`; `main --src-root PATH`.

- [ ] **Step 1: Failing tests** (model on `tests/experiments/test_exp082.py`'s `world` fixture, which fakes the BFS and git):
  - `test_constants_are_the_spec_values`: every constant above, and `cells.max_len_schedule("C")(u) == 14` for u in (16000, 19999, 23999); `cells.max_len_schedule("O")` gives 14 at 16000, 22 at 18000, 29 at 19999, 30 at 20000 and 23999.
  - `test_names_and_layout`: the paths and names above; `cells.ckpt_dir("O", 3).parent.name == "arm_O"`.
  - `test_exclusion_includes_every_probe_state`: `sets(seed)`'s exclude contains every `s` of `probe` and every EXP-075 held-out state.
  - `test_gate_c_passes_and_seeds_the_start_dir`: on a world whose source `judge.pt` and `state.json` blobs equal the tracked EXP-082 blobs (compared against `cells.E82_OUT/judge_X_s<seed>/`), update 16000, opt step 16000, and a re-measured gate_l equal to the EXP-082 u16000 record: record `pass` True, all four files in `start_dir`.
  - Parametrized fails, each with `pass` False and no `start_dir/state.json`: source judge.pt not the tracked blob; state update 8000; an Adam state at step 4000; gate_l off by one in the last digit.
  - `test_gate_c_does_not_gate_jt`: a `jt.pt` that does not load still passes (spec section 6: copied, not gated).
  - `test_gate_c_never_writes_exp082`: the tree under `cells.E82_OUT` hashes the same before and after.
- [ ] **Step 2: Run, expect FAIL** (module missing): `.venv/bin/python -m pytest tests/experiments/test_exp085.py -q` (timeout 300000)
- [ ] **Step 3: Implement** by adapting `experiments/082_x_long_budget/gatec.py`: identity compares `blob_of(src/name)` with `tracked_blob(cells.E82_OUT/judge_X_s<seed>/name)`; step check via `g81.check_opt(..., step=16000)`; start check against `exp082_X16k_u16000_s<seed>.json`'s `gate_l`; `continued_from = {"experiment": "EXP-082", "update": 16000}`. One worker by default (the VPS has 2 cores, 4 GB).
- [ ] **Step 4: Run Step 2's command, expect PASS.**
- [ ] **Step 5: Commit** the two modules and the test file: "EXP-085 cells and Gate C".

### Task 3: rebase and pre-flight (desktop)

**Files:**
- Create: `experiments/085_outward_walks/rebase.py`, `experiments/085_outward_walks/preflight.py`
- Test: `tests/experiments/test_exp085.py`

**Interfaces:**
- Consumes: Task 2's `cells`, `gatec.passed`.
- Produces: `rebase.rebase(arm, seed, device, out=cells.OUT) -> dict` copies `start_dir` into `ckpt_dir(arm, seed)` and rewrites its `state.json` with the walk `random_state` carried over, `generator_state` from `torch.Generator(device=device).manual_seed(rebase_seed(seed))`, `generator_device` = the device type, `update` 16000, probes unchanged; writes `REBASE_NAME` with the seed, the formula and hashes of the four source files. `rebase_seed(seed) = 85_000_000 + seed * 100_003 + 16000`. `preflight.gpu(device, out) -> dict` and `preflight.walks(out) -> dict`, both merged into `PREFLIGHT_NAME`.

- [ ] **Step 1: Failing tests**:
  - `test_rebase_carries_the_walk_state_and_resets_the_encoding_stream` (device "cpu" for the test, monkeypatching the generator factory to record the seed): random_state equal to start's; generator state equal to a fresh generator at `rebase_seed(seed)`; tensors byte-equal to start's.
  - `test_rebase_refuses_without_gate_c` and `test_rebase_refuses_a_started_arm`: an arm dir whose state.json update is above 16000, or already rebased, raises SystemExit and is not touched (Review Focus 2).
  - `test_rebased_cpu_checkpoint_resumes_in_train_judge`: one update of `vi.train_judge` from a rebased dir runs and reaches update 16001 (CPU, tiny world).
  - `test_preflight_walks_matches_the_spec_table` is `@pytest.mark.slow` (full BFS): arm O at update 20000 gives distance shares within 0.02 of spec row 30 (0.524, 0.108, 0.149, 0.141, 0.079), arm C within 0.02 of row 14.
  - `test_preflight_gpu_reports_determinism`: with the train call stubbed to return two state dicts, equal dicts give `deterministic` True, one changed tensor gives False.
- [ ] **Step 2: Run, expect FAIL**: `.venv/bin/python -m pytest tests/experiments/test_exp085.py -q -m "not slow"` (timeout 300000)
- [ ] **Step 3: Implement.** `preflight.gpu`: rebase seed 0 arm C into two temp dirs, train each to 16100 on the device, compare judge state dicts with `torch.equal` per tensor, record `deterministic`, `s_per_update`. `preflight.walks`: 20,000 walks per row through `vi.random_walk_states` with the schedule's length at updates 16000, 18000, 20000 per arm and the real exclusion, shares by true distance (instrument only). CLI: `preflight.py --part gpu|walks --device cuda`.
- [ ] **Step 4: Run Step 2's command, expect PASS; run the slow test once in the background on the VPS.**
- [ ] **Step 5: Commit**: "EXP-085 rebase onto the device and pre-flight".

### Task 4: training legs

**Files:**
- Create: `experiments/085_outward_walks/train.py`
- Test: `tests/experiments/test_exp085.py`

**Interfaces:**
- Consumes: Tasks 1 to 3.
- Produces: `train.run(arm, seed, device, out=cells.OUT) -> list[dict]` and `train.pending(out) -> list[(arm, seed)]` ordered seed-major, C before O. Record keys: EXP-082's leg record keys plus `arm`, `max_len_schedule` (the string `"14"` or `"14->30 by 20000"`), `device`, `device_name`, `torch_version`, `s_per_update`, `n_exclude`, `n_probe`.

- [ ] **Step 1: Failing tests** (tiny world, CPU):
  - `test_legs_record_20000_and_24000_and_snapshot_23751`: after `run`, both records exist with `n_updates` 20000 and 24000, `snap_dir` holds all four files with state update 23751, and no 23751 record.
  - `test_a_rerun_skips_finished_legs` and `test_a_checkpoint_past_a_leg_end_is_never_recorded_as_it` (EXP-082's two tests, adapted; Review Focus 1).
  - `test_arm_o_resumed_mid_ramp_uses_the_absolute_update` (Review Focus 3): kill after 17000 by running leg to a patched end, resume, and the recorded max lengths seen by the walk generator for 17000..17003 equal the schedule's.
  - `test_both_arms_exclude_the_probe` : the `exclude` handed to `train_judge` contains every probe state.
  - `test_run_refuses_an_arm_not_rebased`.
- [ ] **Step 2: Run, expect FAIL.** Same command as Task 3.
- [ ] **Step 3: Implement** from `experiments/082_x_long_budget/train.py`'s `run`, with `max_len=cells.max_len_schedule(arm)`, `device=`, `gatec.sets(seed)`, the 23751 leg copying the four checkpoint files into `snap_dir` (atomic copies, state.json last) instead of writing a record, and `encoder_drift`/`gate_l` as EXP-082. One process at a time; `main --device cuda`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit**: "EXP-085 training legs".

### Task 5: search cells and Gate H

**Files:**
- Create: `experiments/085_outward_walks/evaluate.py`, `experiments/085_outward_walks/gateh.py`
- Test: `tests/experiments/test_exp085.py`

**Interfaces:**
- Produces: `evaluate.run_cell(arm, depth, seed, device, out=cells.OUT, name=None, limit_states=None) -> dict` (EXP-082's `run_cell` with `e75.load_judge("X", seed, out/arm_<arm>, required_updates=24000, device=device)`, plus `device`, `device_name`, `torch_version`, `judge_updates`); `evaluate.pending(out)` depth 11 first. `gateh.search_picks() -> list[(arm, depth, seed)]` (2 per depth from `random.Random(85)` over ARMS x SEEDS), `gateh.train_picks() -> list[(arm, seed)]` (2 seeds per arm), `gateh.run_a(device, out)`, `gateh.run_b(device, out)` (skips and records `available: False` when `PREFLIGHT_NAME` says not deterministic; else copies `snap_dir` to a temp dir, trains to 24000 with the arm's schedule, and records `equal` = every tensor of the replayed `judge.pt` `torch.equal` to `ckpt_dir`'s).

- [ ] **Step 1: Failing tests**:
  - `test_search_cell_is_exp082s_j4v`: the arguments reaching `evaluate_lookahead` (patched) are mode C, k 4, k_goal 4, no_revisit True, EXP-075's states for the depth and seed.
  - `test_search_refuses_a_judge_short_of_24000` and `test_search_refuses_an_unregistered_depth`.
  - `test_search_pending_runs_depth_11_first_and_skips_finished`.
  - `test_picks_are_fixed_and_distinct`: two calls return the same lists; 2 per depth, 2 per arm, no duplicates.
  - `test_gate_h_a_compares_everything_but_wall_and_commit`: a replay record differing only in `wall_s` and `git_commit` is equal; differing in `solved` or `first_fire_len` is not.
  - `test_gate_h_b_is_unavailable_when_preflight_says_not_deterministic`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** `main --device cuda` on each; `gateh.py --part a|b`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit**: "EXP-085 search cells and Gate H".

### Task 6: aggregator

**Files:**
- Create: `experiments/085_outward_walks/aggregate.py`
- Test: `tests/experiments/test_exp085.py`

**Interfaces:**
- Produces: `verdict(c_ok, h_ok, t_ok, c1, c2, guard) -> str`; `guard_trips(d9_diffs) -> bool` (mean <= -0.05 and `one_sided_p([-d for d in d9_diffs]) < 0.025`); `run_all(out) -> dict` with `lines` and `verdict`; `main` prints the lines (exit 0) for the chain to write as the verdict file.

- [ ] **Step 1: Failing tests** (synthetic records written into tmp_path):
  - `test_verdict_has_a_line_for_every_row_of_spec_section_8`: each of the seven strings, including precedence: Gate C fail beats Gate H fail beats Gate T fail; Claim 1 CONFIRMED with the guard tripped reads "OUTWARD WALKS PAY AT DEPTH 11, COST DEPTH 9"; CONFIRMED (SMALL) falls through.
  - `test_guard_trips_only_on_a_significant_drop_of_005`: -0.06 on 12 of 12 seeds trips; -0.04 on 12 of 12 does not; -0.06 mean with signs mixed so p >= 0.025 does not.
  - `test_claim_1_is_o_minus_c_at_depth_11` and `test_claim_2_is_o_minus_c_far_band`.
  - `test_gate_h_unavailable_is_said_not_passed` (Review Focus 5): the Gate H line contains "(b) unavailable" and the verdict is not VOID.
  - `test_records_from_the_wrong_device_arm_updates_or_a_smoke_run_are_refused` (Review Focus 4): `device` "cpu", `judge_updates` 20000, `limit_states` 5, `n` 199, `arm` swapped each raise SystemExit naming the field.
  - `test_mixed_checkouts_are_refused`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** on `a80` (`one_sided_p`, `far`, `overall`, `gate_t`, `claim_verdict`) and EXP-082's `claim1_verdict` shape with `C1_MIN = 0.03`. Reported block per spec section 9; EXP-082's `exp082_X16k_u16000` records give the 16,000 margins and its J4V-X16k records the reported search baseline.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit**: "EXP-085 aggregator".

### Task 7: drivers, watcher and docs

**Files:**
- Create: `experiments/085_outward_walks/stage_vps.sh`, `experiments/085_outward_walks/chain085.ps1`, `experiments/085_outward_walks/status085.ps1`, `experiments/085_outward_walks/watch085.sh`
- Modify: `CLAUDE.md` (test count line)
- Test: `tests/experiments/test_exp085.py`

**Interfaces:**
- `stage_vps.sh`: scp the four files per seed from the laptop's EXP-082 folder into `outputs/x16k_src/judge_X_s<seed>/`, run `gatec.py --src-root outputs/x16k_src`, force-add and commit the 12 Gate C records, push, then scp each `start/judge_X_s<seed>/` to the desktop's `D:/gpu083/repo/experiments/085_outward_walks/outputs/start/`.
- `chain085.ps1` stages in order: check (import path under the repo's src, `torch.cuda.is_available()` True, 12 passing Gate C records, the EXP-084 verdict file says `GPU PORT REPRODUCES`, pytest `tests/experiments/test_exp085.py -m "not slow"`), rebase, preflight gpu, train, search, gateh a, gateh b, aggregate into `D:\gpu083\verdict085.txt`. Logs to `D:\gpu083\chain085.log`. `-From` parameter as chain084.
- `status085.ps1` prints `"$verdict|py N|T n|S n|H n|last log line"`.
- `watch085.sh` (VPS): polls every 10 min. An EMPTY status (ssh failed) means the desktop is unreachable, which after a bugcheck can last hours: it never relaunches on it, never counts it as a crash, and pushes ntfy once after 3 consecutive unreachable polls so Michael can bring Tailscale back. Only a reachable desktop with no python, no verdict and no STOP line counts as a crash: it relaunches from the top (every stage resumes), counting relaunches in a state file under `/root/scratch/exp085/`; on the third crash it stops relaunching and exits non-zero so the session asks Michael.

- [ ] **Step 1: Failing tests**: `test_chain_has_every_stage_in_spec_order`, `test_chain_checks_the_exp084_verdict`, `test_drivers_have_no_em_dashes_or_backticks_in_strings` (EXP-082's test, adapted), `test_watcher_stops_after_three_relaunches` (run `watch085.sh` with a stub `ssh` on PATH returning a dead status, `WATCH085_SLEEP=0`), `test_watcher_never_relaunches_on_an_unreachable_desktop` (stub `ssh` exits 255 with no output: no relaunch, crash count 0, one push request after 3 polls, via a stub `curl` or `WATCH085_PUSH` command).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Parse-check the ps1 files on the desktop at deploy.
- [ ] **Step 4: Run, expect PASS; run the whole suite in chunks per CLAUDE.md and update the count line from `--collect-only`.**
- [ ] **Step 5: Commit**: "EXP-085 drivers, watcher and test count".
