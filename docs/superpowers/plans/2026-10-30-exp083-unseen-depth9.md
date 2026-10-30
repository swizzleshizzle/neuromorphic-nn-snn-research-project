# EXP-083 Unseen Depth-9 Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run J4V-X16k against J4V-X at depths 9 and 10 on 200 states per seed that neither judge was ever trained on, with the gates, claims and one-line verdict of the spec.

**Architecture:** `experiments/083_unseen_depth9/`, importing EXP-082's `cells`/`gatec`/`evaluate`, EXP-075's `cells`/`evaluate` and EXP-080's aggregate helpers unchanged. `sets.py` replays the training walk stream and writes one sets file per seed; `evaluate.py` runs the registered search cell over those sets (plus Gate R); `floor.py` measures the random-policy floor; `aggregate.py` reads it all into one verdict. A PowerShell launcher and chain run it on the laptop.

**Tech Stack:** Python 3 / torch (CPU, 1 thread per worker), pytest, PowerShell 5.1 on the laptop.

**Spec:** `docs/superpowers/specs/2026-10-30-exp083-unseen-depth9-design.md`

## Global Constraints

- Seeds 0 to 11; depths (9, 10); 200 states per seed and depth.
- Set shuffle seed `83_000 + 100 * d + seed`, applied to the SORTED candidate list.
- "Trained on" = a state in a training batch of `vi.random_walk_states(1000, 14, rng, N_ACTIONS, exclude)`, `rng = random.Random(seed)`, updates 0 to 15,999; `exclude` = EXP-075's exclusion set (`gatec.sets(seed)[0]`).
- Gate V: replayed random state equals EXP-075's `judge_X_s<seed>/state.json` (update 4000) and EXP-082's `judge_X_s<seed>/state.json` (update 16,000).
- Gate R: J4V-X16k depth 9 seed 0 on EXP-082's own held-out states equals EXP-082's record exactly (`solved` 127, `success_rate` 0.635).
- Gate F: a claim at depth d is read only if the better arm's 12-seed mean exceeds the random-policy floor on U_d by at least 0.05.
- Claims: one-sided exact sign-flip (EXP-080's `one_sided_p`), alpha 0.025; Claim 1 bar 0.10, Claim 2 bar 0.05.
- Search cell: mode C, k 4, k_goal 4, no-revisit, `e75.JudgeCritic`, depth-9 agent/head/train seed from `e75.load_cell(9, seed)` for both depths, `depth=d` passed to `evaluate_lookahead`.
- Workers: sets 4, search 6, floor 4.
- Never write into EXP-075's or EXP-082's directories.
- No em-dashes anywhere; no backticks inside double-quoted PowerShell strings; plain commit messages, no trailer.
- Always run python via the venv: `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp083.py -q`.

## Review Focus

1. A Gate R run must never write into EXP-082's outputs (EXP-082's `ev.run_cell` writes to its `out`, which is also where it loads the judge from). Test: Gate R's record lands in EXP-083's out and EXP-082's out is unchanged.
2. A sets file whose Gate V failed must never feed a search cell. Test: `run_cell` refuses it.
3. A search record from the wrong judge (X16k checkpoint read for arm J4V-X, or wrong update count) must be refused by the aggregator, not silently paired. Test in Task 4.
4. Gate F failing only at depth 10 must still let Claim 1 decide (STOPS SHORT row). Test in Task 4.
5. A restart mid-phase: every driver skips finished records. Test: pending lists skip existing files.

---

### Task 1: cells and the replayed sets

**Files:**
- Create: `experiments/083_unseen_depth9/cells.py`, `experiments/083_unseen_depth9/sets.py`
- Test: `tests/experiments/test_exp083.py`

**Interfaces:**
- Produces (`cells.py`): `DEPTHS = (9, 10)`, `CAP = 200`, `SHUFFLE_BASE = 83_000`, `X_END = 4000`, `X16K_END = 16000`, `ARMS = ("J4V-X16k", "J4V-X")`, `EVAL_SEEDS`, `OUT = HERE / "outputs"`, `E75_OUT`, `E82_OUT`; `sets_name(seed) -> "exp083_sets_s{seed}.json"`, `search_name(arm, depth, seed) -> "exp083_{arm}_d{depth}_s{seed}.json"`, `GATE_R_NAME = "exp083_gateR_s0.json"`, `floor_name(depth, seed) -> "exp083_floor_d{depth}_s{seed}.json"`; module handles `c75`, `e75`, `g82` (EXP-082 gatec), `ev82` (EXP-082 evaluate).
- Produces (`sets.py`): `replay(seed, exclude, ends, *, batch=1000, max_len=14) -> tuple[set, dict[int, list]]` (seen states; random state as `[v, list, gauss]` at each update count in `ends`); `pick(shell, seen, exclude, depth, seed, cap=CAP) -> list`; `build(seed, out=OUT, *, sets_fn=g82.sets, shell_fn=..., e75_ck=..., e82_ck=...) -> dict` writing `sets_name(seed)`; `pending(seeds, out) -> list[int]`; `main()` with `--seeds --workers --out-dir`.
- Sets file keys: `seed`, `gate_v` (`{"u4000": bool, "u16000": bool}`), `pass` (both true), `n_seen`, `candidates` (`{"9": int, "10": int}`), `sets` (`{"9": [...], "10": [...]}`, empty lists when `pass` is false), `git_commit`.

- [ ] **Step 1: Write failing tests**
  - `test_constants_are_the_spec_values`: the Global Constraints numbers and names above.
  - `test_replay_matches_a_real_training_run`: train `vi.train_judge` for 3 updates, batch 4, on a tiny judge (mirror `tests/experiments/test_exp082.py`'s `_train_straight` fixture), then `replay(seed, exclude, ends=(3,), batch=4)` gives the same random state as the run's `state.json`. This pins "the rng feeds nothing but the walks".
  - `test_replay_seen_is_exactly_the_batches`: `seen` equals the union of `n` direct `vi.random_walk_states` calls from a fresh `Random(seed)`.
  - `test_pick_drops_seen_and_excluded_and_caps`: with a toy shell of 500 ints, seen={0..99}, exclude={100..149}: 200 picked, none in seen or exclude, all distinct.
  - `test_pick_is_independent_of_shell_order` and `test_pick_differs_by_seed_and_depth`.
  - `test_build_writes_sets_only_when_gate_v_passes`: with stub `sets_fn`/`shell_fn` and stub checkpoint dirs, a matching random state gives `pass` True and 200-state sets; a tampered EXP-082 `state.json` gives `pass` False, `gate_v.u16000` False and empty sets.
  - `test_sets_pending_skips_written_files`.
- [ ] **Step 2: Run, verify they fail** (`ModuleNotFoundError` / missing attributes).
- [ ] **Step 3: Implement `cells.py` and `sets.py`.** `pick` sorts, filters, then shuffles with `random.Random(SHUFFLE_BASE + 100 * depth + seed)`. `build` uses one `ExactBFSDistance(max_depth=None)` provider for both shells. Sets are stored as lists of lists (JSON); readers convert back to tuples.
- [ ] **Step 4: Run tests, verify they pass.**
- [ ] **Step 5: Commit** `EXP-083 cells and replayed sets (Gate V)`.

### Task 2: the search cell and Gate R

**Files:**
- Create: `experiments/083_unseen_depth9/evaluate.py`
- Test: `tests/experiments/test_exp083.py`

**Interfaces:**
- Consumes: Task 1's `cells`, sets file.
- Produces: `load_states(seed, depth, out) -> list[tuple]` (refuses a missing file or `pass` False); `judge_for(arm, seed) -> Judge` (J4V-X16k: `e75.load_judge("X", seed, E82_OUT, required_updates=16000)`; J4V-X: `e75.load_judge("X", seed, E75_OUT, required_updates=4000)`); `run_cell(arm, depth, seed, out=OUT, limit_states=None) -> dict` writing `search_name`; `gate_r(out=OUT) -> dict` writing `GATE_R_NAME` with `solved`, `success_rate`, `expected_solved`, `expected_success_rate`, `pass`; `pending(seeds, out) -> list[tuple]` (Gate R first, then depth 10, then depth 9); `main()`.
- Record keys: EXP-082's (`arm`, `judge`, `depth`, `seed`, `judge_seed`, `k`, `k_goal`, `mode`, `no_revisit`, `judge_updates`, `wall_s`, `git_commit`, `limit_states`, `fired4`) plus `n` and `states_from` (the sets file name).

- [ ] **Step 1: Failing tests:** `test_cell_refuses_a_missing_or_failed_sets_file`; `test_judge_for_reads_each_arm_from_its_own_experiment` (monkeypatch `e75.load_judge`, assert root and `required_updates` per arm); `test_cell_runs_a_real_j4v_cell_at_depth_10` (tiny judge checkpoints as in test_exp082's real-cell test, `limit_states=2`, record has `depth` 10, `n` 2, the right arm); `test_gate_r_writes_only_into_exp083` (monkeypatched search; EXP-082 out dir file listing unchanged, record in EXP-083 out, `pass` False when solved differs from 127); `test_search_pending_order_and_skip`.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** One private `_search(judge, depth, seed, states)` shared by `run_cell` and `gate_r`; agent, head and train seed always from `e75.load_cell(9, seed)`; Gate R's states are `e75.load_cell(9, 0)`'s own states. `torch.set_num_threads(1)`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** `EXP-083 search cell over the replayed sets, and Gate R`.

### Task 3: the random-policy floor

**Files:**
- Create: `experiments/083_unseen_depth9/floor.py`
- Test: `tests/experiments/test_exp083.py`

**Interfaces:**
- Produces: `random_rollout(state, depth, rng) -> bool` (uniform action from `rng.randrange(N_ACTIONS)` for up to `max_steps_for(depth)` steps via `apply_move`, success on `is_solved`); `floor_cell(depth, seed, out=OUT) -> dict` writing `floor_name` with `solved`, `n`, `success_rate`, using `random.Random(train_seed)` where `train_seed` is `e75.load_cell(9, seed)[3]`; `pending`, `main()`.

- [ ] **Step 1: Failing tests:** `test_rollout_solves_a_one_move_state_sometimes_and_solved_always` (SOLVED returns True at step 0; over 200 rng seeds a 1-move state is solved at a rate between 0.15 and 0.6); `test_rollout_budget_is_2d_plus_3` (count `apply_move` calls with a counting stub: never above `2*depth+3`); `test_floor_cell_reads_the_sets_file` (stub sets, record fields).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** `EXP-083 random-policy floor`.

### Task 4: aggregator

**Files:**
- Create: `experiments/083_unseen_depth9/aggregate.py`
- Test: `tests/experiments/test_exp083.py`

**Interfaces:**
- Produces: `C1_MIN = 0.10`, `C2_MIN = 0.05`, `FLOOR_MARGIN = 0.05`; `claim_verdict(diffs, minimum, floor_ok) -> tuple[str, float, float]` returning `"UNRESOLVED"` when `not floor_ok`, else `"CONFIRMED"` / `"CONFIRMED (SMALL)"` (p < 0.025, mean > 0, below `minimum`) / `"NOT CONFIRMED"`; `verdict(v_ok, r_ok, s_ok, f9_ok, c1, c2) -> str` (spec section 7, exact strings); `read_search(out, arm, depth, seed) -> dict` (checks `arm`, `judge`, `depth`, `seed`, `judge_seed`, `k`, `k_goal`, `mode`, `no_revisit`, `n` 200, `limit_states` None, `judge_updates` 16000 or 4000 by arm); `run_all(out=OUT, e75_root, e76_root, e82_root) -> dict` and `main()` printing one `VERDICT: ...` line plus the reported block; exit non-zero on any refused record or mixed `git_commit`.
- Verdict strings: `VOID`, `UNRESOLVED`, `DEPTH-9 GAIN CONFIRMED, REACHES DEPTH 10`, `DEPTH-9 GAIN CONFIRMED, STOPS SHORT OF DEPTH 10`, `DEPTH-9 GAIN SMALLER THAN REPORTED`, `DEPTH-9 GAIN DOES NOT HOLD ON UNSEEN STATES`.
- Reported block (spec section 9): both arms on U_9 vs EXP-082 / EXP-076 old-set means; optimality, mean steps, goal-fired and fallback fractions; floors; per seed `n_seen` and candidate counts.

- [ ] **Step 1: Failing tests** (synthetic records in tmp dirs, like test_exp082's `_agg_world`): one per verdict row; `test_bars_are_the_spec_values`; `test_small_significant_depth9_gain_reads_smaller_than_reported`; `test_gate_f_failing_at_depth_10_only_reads_stops_short`; `test_gate_f_failing_at_depth_9_is_unresolved`; `test_a_failed_gate_v_or_gate_r_is_void`; `test_a_short_set_is_unresolved`; `test_a_record_from_the_wrong_judge_is_refused` (J4V-X record with `judge_updates` 16000); `test_mixed_checkouts_are_refused`.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement**, reusing `a80.one_sided_p` and `a80.a72.pair_diffs`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** `EXP-083 aggregator: gates V, R, S, F, two claims, one-line verdict`.

### Task 5: laptop launcher and chain

**Files:**
- Create: `experiments/083_unseen_depth9/launch083.ps1`, `experiments/083_unseen_depth9/chain083.ps1`
- Test: `tests/experiments/test_exp083.py`

**Interfaces:**
- `launch083.ps1 -Phase check|sets|search|floor -Workers N`: EXP-082's launcher shape (library check, input check, python-process guard, Start-Process with redirected logs, commit-memory sampling). Input check prints `12 12 12` for: EXP-075 X checkpoints at 4000 with `state.json`; EXP-082 X16k checkpoints at 16,000 with `state.json`; EXP-082's 12 depth-9 search records. Caps: sets 4, search 6, floor 4.
- `chain083.ps1 [-From sets|search|floor]`: sets, then a gates check (`aggregate.py --stage sets` must exit 0 and print `SETS: OK`, else write the VOID verdict if it printed `VERDICT: VOID`, then stop), then search, floor, aggregate; writes `C:\Users\mlgbr\verdict083.txt` only on a clean aggregate; logs to `C:\Users\mlgbr\chain083.log`.
- So `aggregate.py` gains `--stage sets`: reads the 12 sets files; prints `SETS: OK` when all pass Gate V and Gate S, else `VERDICT: VOID` (Gate V) or `VERDICT: UNRESOLVED` (Gate S) and exits 0.

- [ ] **Step 1: Failing tests:** `test_drivers_use_the_spec_worker_counts`; `test_drivers_have_no_em_dashes_or_backticks_in_strings`; `test_chain_reads_the_lines_the_aggregator_prints` (the strings `SETS: OK` and `VERDICT: VOID` appear in both chain083.ps1 and aggregate.py's output for the matching worlds).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement both scripts and `--stage sets`.**
- [ ] **Step 4: Run the whole file:** `PYTHONPATH=src .venv/bin/python -m pytest tests/experiments/test_exp083.py -q` all pass; then `tests/experiments/test_exp082.py` still passes.
- [ ] **Step 5: Commit** `EXP-083 laptop launcher and chain`.
