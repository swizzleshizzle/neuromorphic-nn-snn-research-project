# Session Handoff - 2026-08-03 (Mon) - Week 18 session 1

> **EXP-036 IS RUNNING ON THE LAPTOP.** Dispatched 2026-08-03 around 20:15 ET, expected to take
> **11.8 hours**, so it should finish around **08:00 ET on 2026-08-04**. Repo is clean at
> `87a965b` and pushed. The laptop is on the same commit.
>
> Read `CLAUDE.md`, then this file, then
> `docs/superpowers/specs/2026-08-03-exp036-generalisation-gap-design.md` for the
> pre-registered contract this run is answering.

## 0. State check

```bash
git log --oneline -1                                    # expect 87a965b
git status --short                                      # expect clean
ssh -n laptop 'powershell -NoProfile -Command "(Get-Process | Where-Object { $_.Name -match \"^python\" }).Count"'
```

## 1. First thing: check on EXP-036

```bash
ssh -n laptop 'powershell -NoProfile -Command "$d=\"C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\036_generalisation_gap\outputs\"; \"pyprocs=\" + (Get-Process | Where-Object { $_.Name -match \"^python\" }).Count; \"records=\" + (Get-ChildItem $d -Filter *.json).Count; Get-Content (Join-Path $d run.log) -Tail 40"'
```

**96 records expected** (48 trained + 48 floors), plus 48 `*_head.pt` checkpoints.

> [!warning] **Zero records for the first ~3 HOURS is HEALTHY, not dead.**
> `sweep_configs` submits all 48 trained runs before the 48 floors, so the first wave of 16
> workers is entirely depth-3 training and nothing completes until a whole run finishes. A
> depth-3 run at 10,000 episodes is 70,000 steps, which at the measured 153 ms/step is
> **about 3.0 hours**. The cheap floors are queued LAST and produce nothing early.
>
> Expected first-record times, per wave of 16 on 16 workers:
>
> | wave | depth | steps/run | first records at |
> |---|---|---|---|
> | 1 | 3 | 70,000 | ~3.0 h |
> | 2 | 4 | 80,000 | ~6.4 h |
> | 3 | 5, 6 | 90,000 / 100,000 | ~10 h, then the floors |
>
> A record-count check with a short grace period would call this run dead while it is working
> perfectly. Same shape as the five monitoring bugs logged on 2026-08-02: **before trusting a
> health check, ask what it would report about a run that is perfectly healthy.**
>
> The real early health signal is worker count (expect 16 plus the parent) and memory. Measured
> 90 min in: 18 processes, about 95 MB each, 10.2 GB free. **The 195 MB per worker figure in
> the playbook is roughly 2x conservative for these cube runs.**

Fetch the results when it is done:

```bash
scp "laptop:C:/Users/mlgbr/Desktop/Projects/neuromorphic-nn-snn-research-project/experiments/036_generalisation_gap/outputs/*" experiments/036_generalisation_gap/outputs/
```

The `*_head.pt` checkpoints **are tracked in git**, unlike the `.json` records. Commit them.

## 2. What EXP-036 answers, and the thresholds it must be held to

Full contract in the spec and in `experiments/036_generalisation_gap/run.py`'s docstring. The
driver prints its own verdicts. Two claims:

1. **The gap.** `train_success - heldout_success` at each depth, on 200-capped samples.
   Mean gap under 0.05 with p > 0.05 **refutes coverage as a lever** and the vault's Stage-1a
   train-fraction sweep is cancelled, not run. At or above 0.15 with p <= 0.05 justifies it.
2. **The break point.** Held-out success below twice the measured random floor.
   **Pre-registered prediction: it breaks at depth 5.** If depth 6 still works, that refutes
   the prediction and means Wall 1 is further out than EXP-033 implied. Log it as a refutation.

**Replication gate before trusting anything:** the depth-3 cell is exactly EXP-035's 10k cell
and must reproduce **0.397**. The driver checks this and prints MISMATCH if it is off by more
than 0.02. If it mismatches, resolve that before reading any other row.

## 3. Three errors caught during design, each worth a run

Recorded because the pattern matters more than the instances.

1. **The vault's Stage 1 premise was already false.** `road-to-a-solved-cube.md` asks whether
   the depth-3 policy memorises its 90 training states. It does not: `split_shell` partitions
   depth 3 into 90 train / 30 eval, and **EXP-035's 50.0% headline was always held-out
   success**. The vault note has not been corrected yet; see section 6.

2. **The first gap threshold sat below its own noise floor.** The draft contract tested a
   single-seed gap against 0.05. The `random` arm cannot overfit, so its gap is zero by
   construction, yet over seeds 0-5 it produced single-seed gaps from **-0.100 to +0.011** -
   the depth-3 held-out side is 30 states, so its resolution is 1/30 and three lucky solves
   read as a tenth of a gap. Now stated on the twelve-seed mean with an exact paired
   permutation test over 4096 sign flips, against the random arm as a **measured** null.

3. **The cost estimate was 43% low.** It used CLAUDE.md's "`brain.step` costs about 90 ms",
   which is single-step latency, not what a 16-worker run achieves. See section 4.

## 4. `brain.step` is 90 ms; a parallel run achieves 153 ms/step

Calibrated against EXP-035's own wall clock on the same machine: 3,359,916 steps in 8h55m
across 16 workers is 143 core-hours, so **153 ms/step**. The rule of thumb understates a
parallel run by about 70%. EXP-036 is 11.8 h, not the 6.7 h first written into the spec.

`docs/playbooks/remote-experiment-runs.md` now carries the calibration and this step-counting
formula. Use it for every future estimate:

```python
def steps(depth, episodes):
    stages = list(range(1, depth + 1))
    per = episodes // len(stages)
    return sum(per * (2 * d + 3) for d in stages)
# wall_hours = total_steps * 0.153 / 3600 / workers, plus n_states * (2d+3) per evaluation
```

## 5. Code changes that outlive this experiment

- **Head checkpointing.** No trained weights were saved anywhere before today, so every
  "evaluate that policy differently" question cost a full retrain. `head_filename(cfg)` writes
  `<record stem>_head.pt` beside the record. **This makes the EXP-030 memory re-ask cheap**,
  which is the single biggest consequence of this session for what comes next.
- **Checkpoints are tracked in git**, records are not. Two `.gitignore` fixes were needed and
  **one was already silently broken**: `experiments/*/outputs/` excluded the *directory*, and
  git does not descend into an excluded directory, so the `!.gitkeep` negation beneath it had
  never worked. Now `experiments/*/outputs/*`, with `!experiments/*/outputs/*_head.pt` placed
  after the global `*.pt` because last match wins.
- **`sample_train_eval()`**, capped at `heldout_cap`. **The cap is load-bearing.** Uncapped at
  depth 6 the train-side evaluation is 8,769 states x 15 steps, about 3.3 h per seed and 40
  core-hours over twelve seeds, against roughly 2.6 h of actual training per run. It also
  matches the sampling noise on the two sides, which a gap measurement needs.
- **New record fields:** `train_success_rate`, `n_train_eval`, `generalisation_gap`.
- **Ordering is load-bearing.** Both evaluations run after training, and the train side runs
  strictly after the held-out side, because `greedy_action` draws on the shared torch
  generator. The five EXP-030 reference values are what catch a swap: all reproduce exactly.

## 6. Open, and carried forward

**New, from this session:**

1. **The vault note `road-to-a-solved-cube.md` still states the false Stage-1 premise.** It
   should be corrected to say the depth-3 policy already generalises and that the live question
   is the gap. Not done tonight.
2. `scripts/verify_instrument_neutrality.py` still fails on zero shared filenames, so it cannot
   compare `exp035` against `exp036` records. Needs comparison keyed on
   (arm, depth, seed, sigma) with tag excluded. The EXP-036 driver does its own replication
   check, so this is not blocking, but the script remains unusable across tags.
3. Vault `experiment-log.md` is stale since 2026-07-15 and is missing EXP-029 through EXP-035
   entirely. `weekly-notes-index.md` stale since 2026-06-23. There is no week-18 weekly note yet.
4. Phase 0 and Phase 1 checkpoints in the vault `progress-tracker` are still unticked although
   every week beneath them is done. Unverifiable from here, needs Michael.
5. The Google Calendar milestone on 2026-08-08, "MILESTONE: First 1-Move Cube Solve Attempt",
   is stale by months and says "do not rush past 1-move scrambles until solve rate > 80%".

**Carried forward from 2026-08-02, still open:** items 1 to 10 of that handoff's section 6, all
unchanged. Notably item 7, `test_random_arm_scores_above_zero_but_well_below_one` at
`tests/training/test_cube_baseline.py:71`, still asserts `0.0 <= x <= 1.0` and still cannot fail.

## 7. What was updated outside the repo

`progress-tracker.md` in the vault was rewritten for Phase 3: weeks 16 and 17 marked done with
what actually happened, week 18 in progress, weeks 19 to 25 re-mapped onto the Stage 1-5 plan
from `road-to-a-solved-cube`, and the empty metric lines filled in with the depth-3 result, the
390-parameter framing, and EXP-033's measured probe table. The vault is live-synced, so that is
already on every device.

## 8. Pointers

- Pre-registration: `docs/superpowers/specs/2026-08-03-exp036-generalisation-gap-design.md`
- Driver and its contract: `experiments/036_generalisation_gap/run.py`
- Launcher, with the three Windows failure modes documented: `experiments/036_generalisation_gap/launch036.ps1`
- Strategy: vault `Neuromorphic Development/road-to-a-solved-cube.md` (Stage 1 premise needs fixing)
- Remote runs: `docs/playbooks/remote-experiment-runs.md` (now correct on ssh and on throughput)
- Previous handoff: `docs/handoffs/SESSION-HANDOFF-2026-08-02.md`
