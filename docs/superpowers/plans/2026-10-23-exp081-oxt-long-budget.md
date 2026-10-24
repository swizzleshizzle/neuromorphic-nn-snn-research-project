# EXP-081 O-Xt long budget Implementation Plan

> **For agentic workers:** executed natively (Michael: "build then run", 2026-10-23); one
> whole-branch review subagent at the end.

**Goal:** continue EXP-080's 12 O-Xt runs from update 4000 to 16,000 on an extended mint and read
whether the budget was the limit.

**Architecture:** `experiments/081_oxt_long_budget/` imports EXP-080's `cells`, `mint`, `train`,
`evaluate` and `aggregate` through importlib and adds only a mint length, a Gate C continuity
check that seeds the continuation, a two-leg trainer, a 16k search cell and a new aggregator.

**Tech stack:** Python 3.13 venv, torch, numpy; PowerShell chain on the laptop.

**Spec:** `docs/superpowers/specs/2026-10-23-exp081-oxt-long-budget-design.md`

## Global constraints

- No em-dashes. Plain commit messages, explicit paths staged.
- Distance-to-solved is a diagnostic target only.
- EXP-080 code is imported, never edited.
- Workers: mint 6, train 6, eval 6 (spec section 10).
- Every test must fail against the defect it names.

## Review focus

1. A seed killed mid leg 2 must never be re-recorded as the 8000 point.
2. Gate C must refuse a mint whose prefix differs in ANY array (states, labels, probe, leaves).
3. Gate C must never overwrite a continuation that has already started.
4. The aggregator must refuse a record with `resumed_from` under 4000 or a mint other than the
   one Gate C certified.
5. Gate T's failure line must show far band and overall at 4000 and 16,000.

## Tasks

### Task 1: cells and mint (`cells.py`, `mint.py`)
- Constants: `N_BLOCKS = 16000`, `START = 4000`, `LEGS = (8000, 16000)`, `OUT`, `E80_OUT`.
- `record_name(update, seed)`, `gatec_name(seed)`, `ckpt_dir(seed, root)` (EXP-080's layout, so
  EXP-080's `load_trained` reads it), `snapshot_name(update)`.
- `mint.py` calls EXP-080's `mint_seed(seed, out, n_blocks=N_BLOCKS)` with the pending pattern.
- Test: a longer `draw_blocks` has the shorter one as its prefix (the property the design rests
  on).

### Task 2: Gate C (`gatec.py`)
- `check_prefix(new, old)`, `check_start(seed, judge, mint, e80_record)`,
  `check_opt(opt_state, judge)`, `gate_c(seed, ...)` writing `exp081_gatec_s<seed>.json` and
  initialising the continuation directory only when every check passes and no continuation
  exists.
- Tests: each check passes on the true case and fails on a one-element mutation of each array;
  opt with a wrong step or shape fails; an existing continuation is not overwritten; the copied
  state carries the new mint sha.

### Task 3: two-leg trainer (`train.py`)
- `run(seed, out, ...)`: refuses without a passing Gate C record; for each leg in `LEGS` whose
  record is missing, calls EXP-080's `train_x_arm`; refuses when the checkpoint is past the leg;
  snapshots the leg-1 judge; writes the record.
- Tests: two legs equal a straight run; a continuation from a copied short checkpoint equals a
  straight run; an existing leg-1 record is not rewritten after leg 2; a checkpoint past leg 1
  without its record is refused.

### Task 4: search cell (`evaluate.py`)
- EXP-080's `run_cell` shape with `load_trained(..., required=16000)` and its own record name.
- Test: refuses a judge at 8000; record carries `judge_updates` 16000.

### Task 5: aggregator (`aggregate.py`)
- Record checks, Gate C, Gate T, Claims 1 and 2, verdict table, search reading, reported lines.
- Tests: each verdict row from synthetic records shifted off EXP-080's committed ones; record
  refusals (resumed_from, n_blocks, mint sha, drift); Gate T line content.

### Task 6: launcher and chain (`launch081.ps1`, `chain081.ps1`) plus the laptop procedure
- Phases: check, mint, gatec, train, gates, search; each skips finished work.
- Chain regexes pinned by a test as in EXP-080.
