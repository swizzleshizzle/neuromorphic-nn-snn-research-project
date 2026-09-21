# Session Handoff - 2026-09-21 - THE TOPOLOGY LINE IS CLOSED. Nothing is running.

> **Nothing is running. The laptop is FREE and idle. `main` is clean at `04471c7`, no branches,
> no PRs.** Suite: **714 tests, 692 not slow, 22 slow**, plus **88 vitest + 2 e2e** in
> `dashboard/`. Tags: `phase-2-complete`, `phase-3-checkpoint`.
>
> **Week 24 (Sep 14-20) is complete and over-delivered**: EXP-062, EXP-063, the whole Phase 3
> checkpoint, and then EXP-064, EXP-065, EXP-066 over the weekend. **Week 25 (Sep 21-27) has no
> deliverable left** - its checkpoint was delivered early on Sep 17.
>
> **Phase 4 starts Oct 5 and is laptop-free.** There is no required next task.

## 0. The one thing that still needs Michael

**Windows Update is still unpaused** (`PauseUpdatesExpiryTime = NONE`). Three runs in a row have
now crossed the 03:31 window and survived only because nothing was pending. It has previously
destroyed ~19 CPU-h outright. **Confirm Settings shows "Updates paused until <date>" before any
dispatch that will still be running at 03:31 local.**

## 1. The headline: the five-region topology was tested and it does not carry a policy

**This is new since the checkpoint, and it changes what the write-up can say.**

Measured 2026-09-17: with `recall=False`, **only the sensory region is on the policy path**.
`Brain.step` computes prefrontal -> router -> motor -> an action on **every step of every
experiment ever run**, and that action was consumed **only by the dashboard**. So **34,912 of
61,728 brain parameters (56.6%) were frozen at random initialisation forever**.

**EXP-064 (Sat 19)** put that pathway on the policy path and trained it. The mechanism worked:
drift 2.7051, firing 0.1534, gradient verified at every region parameter, encoder at exactly 0.0
delta. **The experiment still failed** - its capacity-matched control collapsed (an MLP head took
EXP-043's 0.3229 to 0.0000), both arms hit the floor, and the difference was 0 by construction.

**EXP-066 (Sun 20)** fixed that by **removing the control rather than repairing it**, asking a
threshold question one-sample against a fixed bar.

| `region_lr` | `region_drift` | held-out success | modal action |
|---|---|---|---|
| 1e-4 | 0.079 | **0.0017** | 0.850 |
| 1e-3 | 0.460 | **0.0000** | 0.969 |
| 1e-2 | 2.705 | **0.0000** | 1.000 |

**All three at the floor**, against a 390-parameter linear head's **0.3229** on the same config.
All six gates passed, so nothing failed for a boring reason.

> **The learning signal DESTROYS the policy.** The ordering is monotone and inverse: the further
> the regions moved, the more completely the policy collapsed onto a single move. **The limit of
> not training the pathway is where it is least bad.** It is *unstable*, not incapable: arm L4
> seed 0 solves **11.1%** at curriculum depth 3 against a 1.4% floor, then decays to 0.2% by
> depth 5 as entropy falls 1.763 -> 0.219.

**What this does NOT say:** that the topology is worthless in principle. The recipe, head,
curriculum and step caps were all tuned around a concept-reading linear head, and this pathway
inherited every one unchanged. That is the fair comparison available and it is **not** a neutral
starting point. Say so in the write-up.

## 2. Method lessons banked this weekend, all in `CLAUDE.md` and the vault

1. **When a contrast keeps failing for want of a trustworthy control, check whether the question
   can be re-posed as a THRESHOLD.** EXP-066 needs no control, so EXP-064's failure mode is
   structurally unreachable rather than merely guarded against.
2. **Gate the comparison's resolution, not just the arm's mechanism.** EXP-064's two gates passed
   and were right to; neither could see the contrast had none.
3. **A control must be a working reference, not just a matched one.** Third distinct matching
   failure after EXP-030 (bit-identical to its arm) and EXP-063 (matched to its arm, not the
   baseline).
4. **Do not gate a quantity you have only measured in a different regime.** A drift ceiling
   guessed from a 120-episode calibration would have been ~0.3; arm L3 measured 0.4597 and would
   have been voided - and drift carried the run's most interesting finding.
5. **A per-arm cost decomposed from a MIXED run is not validated by the total matching.** It
   confirms only the weighted sum. First pure-motor run: ~5.1 h/cell, not the 4.40 assumed.

## 3. EXP-065 - the dashboard renders a cube, and its honest caveat

The cube path was fully built and never called, so through all of Phase 3 NEURO-SCOPE only
rendered a 5x5 grid from July. One trace surfaced three defects reachable only on that path
(numpy `int64` crashing the monitor's own header, a distance table that must cover the reachable
set, an Export PNG button overlapping the 3D Cloud toggle by 42.5 x 24px).

> **`record_episode` uses `out["action"]`, the brain's OWN pathway - so every NEURO-SCOPE frame
> ever rendered shows a decision the agent never made.** Worth saying out loud if any dashboard
> image appears in the Phase 4 write-up.

**To view it:** tmux session `neuroscope` on this VPS serves `127.0.0.1:4173`. Restart with
`npx vite preview --port 4173 --host 127.0.0.1` from `dashboard/`. Reach it with
`ssh -N -L 4173:localhost:4173 root@69.167.169.11`. The cube build needs
`VITE_TRACE_URL=/cube_dashboard_trace.jsonl`; `node scripts/sync-trace.mjs cube` prints the line.

## 4. What is left, and what is NOT

**Nothing is required.** Weeks 24 and 25 are both complete.

1. **Sep 28 to Oct 4 is an unscheduled gap week**, surfaced in `progress-tracker`.
2. **The compute window is effectively closed.** Phase 4 is laptop-free documentation.
3. **Michael's todos**: `0576` (see the dashboard render - now possible, section 3) and `e491`
   (the magnitude-matched arm T, which its own todo argues against).

**What NOT to do:**
- **Do not re-run the topology question.** Three orders of magnitude of `region_lr` all sit at
  the floor and the collapse is monotone, so an interior optimum runs the wrong way. Any further
  attempt needs a different ARCHITECTURE, not a different hyperparameter.
- **Do not re-run the monolithic contrast** to close criterion 2: it is vacuous, and
  `tests/training/test_topology_contrast_is_vacuous.py` proves why.
- **Do not fold any follow-up into EXP-063's, EXP-064's or EXP-066's numbers.** Separate
  experiment or optional stopping.
- **Do not schedule work against a publishing date.** Content Day is defunct.

## 5. Standing facts

- **Confirm completion from COUNTS, never the log.** EXP-066's aggregator and its 11 tests were
  committed before dispatch, and completion was confirmed from 24 records, 24 checkpoints and
  zero python processes.
- **Anything the record dict reads must be initialised in BOTH branches** of `run_cube_baseline`.
  EXP-064 repeated EXP-042's `UnboundLocalError` six lines below its own warning comment.
- **Mid-run progress is not a rate.** 19 of 24 cells at 20.3 h looked like a 42% overrun; the
  remaining five were one in-flight wave and it finished 13% over, not 42%. Waves, not cells.
- **An unreachable tailscale peer is an UNKNOWN**, not a paused job.
- **READ `docs/retired-instruments.md` and the gate-calibration rule in `CLAUDE.md` before
  writing a spec.**
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 6. Pointers

- `experiments/066_region_lr_sweep/RESULTS.md` - the topology line closed
- `experiments/064_motor_policy_path/RESULTS.md` - the run that completed and the experiment that did not
- `experiments/065_cube_dashboard_trace/run.py` - the first cube trace
- `docs/phase3-honest-assessment.md` - the checkpoint, with its own dated correction
- Vault: `experiment-log.md` (through EXP-066 plus six addenda), `road-to-a-solved-cube.md`
  (topology update at the top), `progress-tracker.md`, weekly notes 21-25
