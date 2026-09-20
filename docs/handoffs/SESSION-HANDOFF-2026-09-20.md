# Session Handoff - 2026-09-20 - PHASE 3 CLOSED OUT. Nothing is running, nothing is required.

> **Nothing is running. The laptop is FREE and idle. `main` is clean at `5c55ef1`, no branches,
> no PRs.** Suite: **703 tests, 681 not slow, 22 slow**, plus **88 vitest + 2 e2e** in
> `dashboard/`. Tags: `phase-2-complete`, `phase-3-checkpoint`.
>
> **Phase 3 is checkpointed, and its last two experiments are done and banked.** Week 24 and
> week 25's deliverables are both complete. **Phase 4 starts Oct 5 and is laptop-free.**
>
> **There is no required next task.** Read section 4 before inventing one.

## 0. The one thing that still needs Michael

**Windows Update is still unpaused** (`PauseUpdatesExpiryTime = NONE`). It destroyed one run
outright (~19 CPU-h) and damaged another. EXP-063 and EXP-064 both survived the 03:31 window
purely because nothing was pending. **Confirm Settings shows "Updates paused until <date>" before
any dispatch that will still be running at 03:31 local.**

## 1. EXP-064 - the run completed, the experiment did not

**The capstone's central claim was tested for the first time.** `Brain.step` computes
prefrontal -> router -> motor -> an action on **every step of every experiment ever run**, and
that action was consumed **only by the dashboard**. So **34,912 of 61,728 brain parameters
(56.6%) were frozen at random init forever**. `readout="motor"` put that pathway on the policy
path and `region_lr` trained it.

**The mechanism worked.** `region_drift` **2.7051**, `motor_rate_mean` **0.1534**, zero silent
steps, gradient verified to reach every region parameter with the encoder at exactly 0.0 delta.
Both gates passed and both were right to.

**The experiment still failed.** Both arms scored **exactly 0.0000** held-out and train-side on
all 12 seeds. The primary is 0 by construction.

> **The finding is the CONTROL.** EXP-043 ran the same config with a **linear** head at depth 5
> and scored **0.3229**. Adding `head_hidden=218` to match parameters dropped it to **0.0000**,
> entropy **0.0120**, modal action **1.000**. Matching capacity cost 0.32 and bought a dead
> policy - and the floor was calibratable **before dispatch** from a number already in the repo.

**Arm P did not fail the way the control did**: entropy stayed healthy (1.758 -> 0.915 across the
curriculum) and it revisits less than half as often. It trained, fired, explored, never learned.
`region_drift` 2.71 means the regions moved nearly 3x their own initial magnitude, which is a
**hypothesis** about `region_lr=1e-2`, not a finding. The lr was deliberately not swept.

## 2. The two rules this bought, both in `CLAUDE.md`

- **GATE THE COMPARISON'S RESOLUTION, NOT JUST THE ARM'S MECHANISM.** Both gates guarded arm P.
  Neither could see the contrast had none. Ask what reading would prove the comparison could have
  come out either way, and gate on that too.
- **A CONTROL MUST BE A WORKING REFERENCE, NOT JUST A MATCHED ONE.** Before using an arm as a
  reference, confirm it still clears the floor in the regime it will run in.

**This is the third distinct matching failure and they are worth reading together:** EXP-030
matched a control until it was bit-identical to its arm; EXP-063 matched a control to its arm but
neither to the baseline; EXP-064 matched capacity but not competence.

## 3. EXP-065 - NEURO-SCOPE renders a cube, and the dashboard's honest caveat

The cube path was **fully built and never called**, so through all of Phase 3 the dashboard only
rendered a 5x5 grid from July. One trace surfaced three defects reachable only on that path: a
numpy `int64` from `env.action_space.n` crashing the monitor's own header, a distance table that
must cover the **reachable set** rather than the scramble depth, and an Export PNG button
overlapping the 3D Cloud toggle by 42.5 x 24px.

> **`record_episode` uses `out["action"]`, the brain's OWN pathway.** Every experiment's policy
> used `head(concept)` and ignored it. **So every NEURO-SCOPE frame ever rendered shows a decision
> the agent never made.** In the cube trace the untrained brain plays **U' seven times in a row**.
> Worth saying out loud in the Phase 4 write-up if any dashboard image appears in it.

**To view it:** a preview server may still be in tmux session `neuroscope` on this VPS at
`127.0.0.1:4173`. Restart with `npx vite preview --port 4173 --host 127.0.0.1` from `dashboard/`.
Reach it with `ssh -N -L 4173:localhost:4173 root@69.167.169.11`. The cube build needs
`VITE_TRACE_URL=/cube_dashboard_trace.jsonl`; `node scripts/sync-trace.mjs cube` prints the line.

## 4. What is left, and what is NOT

**Nothing is required.** Weeks 24 and 25 are both complete and the checkpoint is tagged.

1. **Sep 28 to Oct 4 is still an unscheduled gap week**, now surfaced in `progress-tracker`.
2. **The compute window is effectively closed.** Phase 4 is laptop-free documentation. The
   topology question is now a **v2 ARCHITECTURE** question, not a run.
3. **Michael's todos**: `0576` (see the dashboard render - now possible, see section 3) and
   `e491` (the magnitude-matched arm T, which its own todo argues against).

**What NOT to do:**
- **Do not re-run EXP-064 without a control demonstrated to clear the floor first**, plus a
  resolution gate. Otherwise it buys another 14 hours of 0.0000.
- **Do not re-run the monolithic contrast** to close criterion 2: it is vacuous, and
  `tests/training/test_topology_contrast_is_vacuous.py` proves why.
- **Do not fold any follow-up into EXP-063's or EXP-064's numbers.** Separate experiment or
  optional stopping.
- **Do not schedule work against a publishing date.** Content Day is defunct.

## 5. Standing facts

- **Confirm completion from COUNTS, never the log.** EXP-064's aggregator was written and tested
  before any record was opened, and this handoff can say so.
- **Price each ARM separately.** EXP-064 came in at 14.0 h against 13.7 h (2%) because the motor
  arm (104.69 ms/step) and the control (57.66) were priced apart and scaled by ratio against an
  arm with a known laptop cost.
- **Anything the record dict reads must be initialised in BOTH branches** of
  `run_cube_baseline`. EXP-064 repeated EXP-042's `UnboundLocalError` six lines below its warning.
- **An unreachable tailscale peer is an UNKNOWN**, not a paused job.
- **READ `docs/retired-instruments.md` and the gate-calibration rule in `CLAUDE.md` before
  writing a spec.**
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 6. Pointers

- `experiments/064_motor_policy_path/RESULTS.md` - the run that completed and the experiment that did not
- `experiments/065_cube_dashboard_trace/run.py` - the first cube trace
- `docs/phase3-honest-assessment.md` - the checkpoint, with its own dated correction
- `tests/training/test_topology_contrast_is_vacuous.py` - why criterion 2 cannot be closed
- Vault: `experiment-log.md` (through EXP-065 plus three addenda), `progress-tracker.md`,
  weekly notes 21-25
