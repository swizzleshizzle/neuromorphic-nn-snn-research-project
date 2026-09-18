# Session Handoff - 2026-09-18 - PHASE 3 IS CHECKPOINTED. Nothing is required next.

> **Nothing is running. The laptop is FREE and idle. `main` is clean at `39b9454`, no branches, no
> PRs.** Suite: **678 tests, 656 not slow, 22 slow.** Tags: `phase-2-complete`,
> **`phase-3-checkpoint`**.
>
> **Week 24 (Sep 14-20) is DONE and over-delivered**: EXP-062, EXP-063, **and the whole Phase 3
> checkpoint, which was week 25's work.** So **week 25 has no remaining deliverable.**
>
> **There is no required next task.** Read section 3 before inventing one.

## 0. The one thing that still needs Michael

**Windows Update is still unpaused.** `PauseUpdatesExpiryTime` reads **NONE**. It has destroyed one
run outright (EXP-059 attempt 1, ~19 CPU-h) and damaged another (EXP-062 attempt 1). EXP-063's 18.5
hours survived the 03:31 window purely because nothing was pending. **Confirm Settings shows
"Updates paused until <date>" before any dispatch that will still be running at 03:31 local.**

## 1. What this session did

**EXP-063 ran and closed the memory line.** A learned attention over a **perfect** episodic cache
scored **0.1154** against amnesic's **0.3138** (-0.1983, p 0.0000) and the raw attractor read's
**0.2183** (-0.1029, p 0.0059). Both secondaries refuted in the wrong direction. The primary
(T - U = +0.0612, p 0.0358) is **confirmed and is not good news**: 14 of the control's 24 seeds
scored exactly 0.000, and on the 10 non-collapsed seeds it is +0.0320 at p 0.5078. **It largely
measures how often training collapsed.** All four gates passed.

**Phase 3 was checkpointed**: `docs/phase3-honest-assessment.md`, merged and tagged
`phase-3-checkpoint` (deliberately not `phase-3-complete`). README rewritten (it said "Phase 2,
Week 12" and carried no results). Vault current through week 25.

**Three stale-statement bugs found and fixed**, all the same shape:
- Phase 1's checkpoint looked unmet for four months. It was **completed 2026-05-25 and never
  ticked** (todo `0817`, now closed along with `eef1` and `4f13`).
- EXP-012's log row said **"(not built)"** for four months after it was built, copied from a
  read-only audit snapshot taken hours before the demo landed.
- EXP-061's log row rendered broken from a literal `||` inside a table cell.

**The criterion-2 topology experiment was specified, then proven VACUOUS before a cell ran.** See
section 2, because it is the most transferable thing here.

## 2. The finding that matters most

**A control matched on every axis you can name may be the same computation.**

`docs/phase3-honest-assessment.md` proposed fixing criterion 2 with an on-path-capacity-matched
monolithic arm at 192 neurons. `Brain` builds its sensory region as `SensoryCortex(n_obs=144,
hidden=128, concept=64, seed=s)`; `MonolithicBrain(total_neurons=192, content=64)` builds
`SensoryCortex(n_obs=144, hidden=192-64, concept=64, seed=s)`. **Measured over 6 seeds through the
real `make_agent` path: every weight bit-identical, concept `max|diff| = 0.000e+00`.**

**This is EXP-030's lesson recurring, 33 experiments later.** There a shuffle-null that varied the
query state also varied "features of the current observation". Here the capacity-match removed the
only thing that differed. **EXP-030 found its confound from the numbers, after paying for them;
this was checked for literal equality before dispatch and cost nothing instead of 24 to 48 cells.**

**The conclusion is stronger than the experiment would have been.** The topology question is not
answerable by ANY arm-versus-arm contrast while the topology is off the policy path (318 of 510
neurons are, whenever `recall=False`). Total-matching measures **width**, in the control's favour.
On-path-matching measures **nothing**. **It is a v2 architecture question now, not an experiment.**

`tests/training/test_topology_contrast_is_vacuous.py` (15 tests, 3 mutations all caught) locks it
in. **A failure of that file is GOOD news** and its docstring says so, so nobody deletes it to get
green.

## 3. What is actually left, and what is NOT

**Nothing is required.** Do not invent a week-25 deliverable; it was consumed early.

**Three real items, none urgent:**

1. **Sep 28 to Oct 4 is an unscheduled gap week.** Phase 3 ends Sep 27, Phase 4 starts Oct 5, and
   `progress-tracker` has **no row** for the week between. Worth deciding rather than drifting.
2. **The compute window closes.** The laptop is idle now and **Phase 4 is laptop-free
   documentation work**, so the next two weeks are the last natural slot for any experiment. The
   only scientifically live candidate is todo `e491` (the magnitude-matched arm T, 24 cells,
   ~2.3 h/cell at 6 workers), and **its own todo says decide whether it is worth running at all**:
   the ceiling result already closes the memory line, and pooling a rerun with existing numbers
   would be optional stopping.
3. **Michael's two todos**: `0576` (see the NEURO-SCOPE dashboard render in a browser once) and
   `e491` above.

**What NOT to do:**
- **Do not re-run the monolithic contrast** to close criterion 2. Section 2 explains why.
- **Do not fold the magnitude-matched arm into EXP-063's numbers.** It is a separate experiment or
  it is optional stopping.
- **Do not schedule work against a publishing date.** Content Day is defunct.

## 4. Standing facts, unchanged

- **Confirm completion from COUNTS, never the log**, which prints one success rate per line.
- **An unreachable tailscale peer is an UNKNOWN**, not a paused job. Use the three-reading test.
- **Price runs from a SAME-worker-count measurement**, and check the arms have the same per-step
  cost: EXP-063 came in **24% under** its estimate because the attention arms skip the attractor
  unroll the memory arms run.
- **Worker count should DIVIDE the cell count.**
- **READ `docs/retired-instruments.md` before putting an instrument in a spec.** Five retired.
- **READ the gate-calibration rule in `CLAUDE.md` before writing a gate.** 2 of 8 were wrong.
- **A fixture can disarm its own test** (EXP-063's encoder-freeze test built its input under
  `no_grad`, so the mutation survived). Only mutation testing finds it.
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 5. Pointers

- `docs/phase3-honest-assessment.md` - the checkpoint, with its own dated correction
- `experiments/063_learned_readout/RESULTS.md` - the memory line closed at its ceiling
- `tests/training/test_topology_contrast_is_vacuous.py` - why criterion 2 cannot be closed
- `docs/retired-instruments.md`, `CLAUDE.md`, `docs/playbooks/remote-experiment-runs.md`
- Vault: `experiment-log.md` (through EXP-063 plus the vacuous-control addendum),
  `progress-tracker.md`, `road-to-a-solved-cube.md`, weekly notes 21-25
