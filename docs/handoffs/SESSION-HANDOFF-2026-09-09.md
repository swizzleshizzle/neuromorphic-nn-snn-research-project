# Session Handoff - 2026-09-09 (Week 23) - EXP-059 RE-DISPATCHED after being destroyed once

> **EXP-059 IS RUNNING AGAIN. Launched 2026-09-09 23:14 UTC**, 72 cells, 6 workers, **expect ~45 h**
> (the ~30 h figure was refuted; see section 0c). Verified by probing for 8 python processes, not by
> an exit code.
>
> **The FIRST attempt was destroyed by a Windows Update reboot at 07:35 UTC** with zero records
> written, ~19 CPU-hours lost. **That post-mortem is section 0 and it is the most useful thing in
> this document**, because the failure was misdiagnosed as a sleep for 13 hours.
>
> **WINDOWS UPDATE IS NOT YET DEFERRED - this is the one open risk.** The registry write was blocked
> by the permission classifier, correctly, so it is Michael's to do: Settings > Windows Update >
> Pause updates > 5 weeks. **Until that is done, a second TrustedInstaller reboot can repeat the
> loss**, though after the first cell completes `--skip-existing` limits it to the in-flight wave
> rather than everything.
>
> **`main` is at `60245b6` and clean. Work sits on branch `exp-059-memory-depth5`, pushed. The
> laptop worktree is at `a709511`, whose only diff to the branch tip is docs - no code difference -
> so it was deliberately NOT re-synced, avoiding any risk to the 24 untracked encoders.**

## 0. WHAT HAPPENED, AND THE DIAGNOSIS THAT WAS WRONG FOR 13 HOURS

| UTC | elapsed | CPU-h/worker | records | |
|---|---|---|---|---|
| 04:16 | 0.43 h | 0.42 | 0 | healthy |
| 05:50 | 2.00 h | 1.72 | 0 | healthy |
| 07:23 | 3.55 h | **3.21** | 0 | healthy, last good reading |
| **07:35** | 3.75 h | - | - | **Windows Update reboot. Run destroyed.** |
| 08:41 - 18:43 | - | unreachable | - | reported as "asleep, nothing lost" - **WRONG** |
| 21:43 | - | **0 procs** | **0** | machine awake, empty outputs dir |

**The misdiagnosis is the durable lesson.** From 08:41 the laptop read `offline, last seen 1h ago`,
and the playbook's own table said that meant "the machine slept, the job is paused, not dead". So
it was reported as paused, five times over 13 hours. **The last-seen field cannot tell a sleep from
a reboot**, and the difference is the entire run. The playbook table has been corrected, with the
three-reading test (uptime, python process count, record count) that actually decides it.

**The evidence, once the machine answered:**

```
BOOT   = 2026-09-09 03:35:58 laptop-local = 07:35:58 UTC   (uptime 14.12 h)
PYPROC = 0        ALLPROC = 348        outputs dir = 0 files
log mtime = 2026-09-08 23:50:22 laptop-local, i.e. unchanged since launch
event 1074 = TrustedInstaller.exe ... "Operating System: Upgrade (Planned)"
```

## 0b. THE DECISION, RESOLVED

**Michael chose to re-dispatch**, on 2026-09-09, over the two alternatives on the table: re-running
at 12 seeds (~22 h, half the exposure, but n=24 was chosen deliberately to halve the standard error)
or leaving it. **Re-dispatched unchanged at n=24**, so the spec and its calibrated gate stand exactly
as pre-registered.

**The stated precondition was not met, and that is a live risk rather than a closed item.** The
advice was not to re-dispatch without deferring Windows Update; the deferral needs Michael, since
the registry write is blocked by the permission classifier. **The run is going regardless.** The
mitigation that does apply automatically: once the first cell completes, `--skip-existing` caps a
repeat loss at the in-flight wave instead of the whole run.

## 0c. THE RE-DISPATCH, and what is known about its cost

| | |
|---|---|
| Launched | **2026-09-09 23:14 UTC** (19:14 laptop-local, EDT) |
| Command | `launch059_wt.ps1 -Phase rl -Workers 6 -SkipExisting` |
| Worktree | `exp-059-memory-depth5 @ a709511`, library verified resolving under the worktree |
| Pre-flight | 24/24 encoders, recall probe present, 0 stray python procs, `CHECK OK` |

`-SkipExisting` is passed so the identical command resumes losslessly after any future
interruption. It bought nothing on this launch, because the outputs directory was empty.

**Per-cell cost is known only as a lower bound, `>3.21 CPU-h`**, measured in the destroyed attempt.
The 2.5 h/cell estimate it was dispatched under is refuted: that figure came from scaling EXP-058's
measured 3.37 h depth-6 cell by `13/15` for the step budget times `5/6` for the stage count, which
**double-counts**. A curriculum SPLITS a fixed episode count across its stages, so the correct
ratio is 0.90, not 0.72. **The first record of this run finally pins it** - the log's `success`
lines carry per-cell seconds directly.

At ~3.4 h/cell and the measured 0.904 CPU/wall ratio, 12 waves is **~45 h wall, ETA around
2026-09-11 20:00 UTC**, plus any sleep.

## 0d. ARM A IS COMPLETE - the cost is now measured cleanly, and a units error is corrected

**2026-09-10 11:27 UTC: all 24 arm-A cells are done.** Arms M and S have produced **zero** records,
because the driver is arm-major (`sweep_configs` iterates `for key in arms` outside
`for seed in seeds`). Readings at 13:07 UTC: uptime 29.52 h against a run age of 13.88 h, 8
processes, six workers at 12.27 CPU-h, ratio **0.884**, no sleep, no reboot.

> [!warning] **CORRECTION: earlier entries said "~3.05 CPU-h per cell". That is the WALL figure,
> not the CPU one.** The two were conflated across several readings. **3.05 h is wall time per
> cell; the CPU cost is 2.70 CPU-h.** Projections must use the wall figure, and the CPU figure is
> only for the sleep check.

**Arm A, measured cleanly rather than extrapolated** - 24 cells on 6 workers is exactly 4 per
worker, so there is no fractional-wave guesswork:

| quantity | value |
|---|---|
| arm A, launch to last completion | 23:14 -> 11:27 UTC, **12.22 h wall** |
| per cell | **3.054 h wall**, 2.70 CPU-h |
| **projected total IF M and S match A** | 12 cells per worker x 3.054 = **36.7 h** |
| **ETA on that assumption** | **2026-09-11 11:53 UTC** |

**That is EARLIER than the 16:30 previously quoted**, because the old figure used wave 1's 3.44 h,
which included process startup and was never representative.

### Arm M is now priced too, and it is only 7% dearer

**2026-09-10 14:43 UTC: arm M's first wave of 6 is done** (30/72 total: amnesic 24, memory 6,
shuffled 0). Arm A's last completion was 11:27 and arm M's sixth was 14:43, so one arm-M wave is
**3.27 h wall per cell** against arm A's 3.054 - a **7% premium**, not the step change the extra
hippocampal recall and the every-8-steps probe might have caused.

| segment | cells | per cell | span |
|---|---|---|---|
| arm A | 24, done | 3.054 h | 12.22 h |
| arm M | 24, 6 done | **3.27 h** | ~13.1 h |
| arm S | 24, 0 done | assumed 3.27 h | ~13.1 h |
| **total** | | | **~38.4 h** |

**Revised ETA: 2026-09-11 13:40 UTC**, plus any sleep. Arm S remains an assumption, though a
shuffled read costs what a real one does, so it is a much safer one than arm M was.

> [!warning] **DO NOT COMPUTE A PRELIMINARY M-VERSUS-A CONTRAST.** Six arm-M seeds against their
> arm-A partners is a runnable contrast right now, and looking at it would contaminate the analysis
> the pre-registration exists to protect - the aggregator checks the validity gate FIRST for exactly
> this reason, and the gate cannot be evaluated until arm M is complete. **Cost readings are the
> only thing to take from a partial run.**

## 0e. SESSION 2 OF WEEK 23 - DONE. All three items, all laptop-free.

The laptop is occupied until ~2026-09-11 16:30 UTC, so session 2 was scoped to work that needs no
compute. **All of it is committed and pushed.**

### 1. EXP-059's aggregator - DONE, and written while ZERO records existed

`experiments/059_memory_depth5/aggregate.py` plus 22 tests in
`tests/experiments/test_exp059_aggregate.py`. **Provenance verified by probe, not asserted: 0
records on the laptop at 2026-09-09 23:47 UTC, workers still in wave 1.** That is the only condition
under which an aggregator can be honestly written.

Three things in it worth knowing before touching it:

- **It deliberately does NOT import `describe_contrast`** from EXP-055/056/057. All three hardcode
  `T95_DF11 = 2.201`, the multiplier at df=11 for n=12. **This is n=24, where it is 2.069**, so
  reusing that code would report every interval about 6% too wide. There is a df-indexed table
  instead.
- **A missing `recall_content_cos` is a gate FAILURE, never a pass.** `None` coerced to 0.0 would
  sail under a "below 0.95" ceiling while measuring nothing - the mirror of EXP-058's gate that
  could not pass.
- **The sampled permutation uses an add-one estimator**, because plain `hits/draws` can print
  `p = 0.0000` and claim an exactness the sampling cannot support.

**Every test was verified to fail against the specific bug it names**, by mutating the aggregator
eleven ways and confirming each mutation is caught. **That found a real gap**: nothing covered a
PARTIAL probe failure, where one seed is `None` and the mean of the other 23 sits comfortably under
the ceiling, so the gate would have passed on 23 of 24 arms while claiming to check all of them.

**Reading the output on a synthetic world found a second gap the tests had not.** At a significant
`-0.045` the wording reported only the magnitude. That is the likeliest outcome given EXP-058's
ordering, and EXP-057's Claim 2 at `-0.0446` needed "do NOT call this a null" written into its prose
because the wording did not say it. The sub-bar branch now names the direction.

### 2. The retired-instruments note - DONE

`docs/retired-instruments.md`, with a pointer from `CLAUDE.md`. Deferred across three handoffs.

**The synthesis is that these are not bad measurements: every one works as a THRESHOLD and fails as
a GRADIENT.** Each detects that an intervention happened; none measures how much it helped. The
recurring error is seeing an instrument and the outcome move together across one coarse contrast and
reading that as if it held per seed.

It also records **why this system generates them so readily**: with `recall=False` only the sensory
region is on the policy path, so an instrument reading a representation can measure something the
policy never consults. And it carries four checks to run before adopting a new instrument, the
sharpest being that **a pilot selecting on an unvalidated proxy can cancel the arm that was going to
work** - which is what EXP-053's lr pilot nearly did to arm B.

### 3. EXP-060, the EXP-056 replication - SPECED, not dispatched

`docs/superpowers/specs/2026-09-10-exp060-flattened-critic-replication-design.md`.

**Its real problem is statistical, not computational.** Extending an experiment because its p-value
was marginal and then pooling is **optional stopping**. So the primary is an **independent
replication on seeds 14-23 alone**, which no decision was based on, and the pooled n=22 is secondary
with that caveat attached whenever quoted. **The primary is underpowered at ~50-60%, and EXP-056's
`-0.0646` is itself upward-biased because it was selected for significance**, so a null primary is
pre-registered as a bound.

> [!note] **COST CORRECTED DOWNWARD: ~14 h, not the ~25 h this handoff carried.**
> Three reasons, and two of them were simply never checked:
> 1. **E1 encoders already exist for seeds 0-13, not 0-11** - EXP-047's pilot ran on seeds 12 and 13
>    and those encoders were kept. **10 seeds are needed, not 12.**
> 2. **E0 is already manufactured for all 24 seeds**, as a side effect of EXP-059. The ~1.7 h every
>    previous estimate carried is already spent.
> 3. **Worker counts chosen to DIVIDE the cell count.** 10 and 20 cells both divide by 10. Note the
>    playbook measures per-cell time as WORSE at 10 workers than 6; the gain is in removing a ragged
>    final wave, not in parallelism.

## 1. What EXP-059 is, and why the design is shaped this way

**The memory question, at depth 5, at n=24, with a gate that works.** Three arms varying ONLY the
readout, on `exp043_capped_d5`'s config field for field: a working depth-5 policy at **0.3229**,
frozen encoder, already measured at 24 seeds.

| arm | readout | tag |
|---|---|---|
| A, amnesic | `memory_amnesic` | `exp059_amnesic_d5` |
| M, memory | `memory` | `exp059_memory_d5` |
| S, shuffled | `memory_shuffled` | `exp059_shuffled_d5` |

**It exists because EXP-058 was VOID.** That experiment's gate required `mean_n_stored > 10`, a
quantity bounded by episode length, where episodes average 7.76 steps. **Unsatisfiable by
construction**, and it gated on **storing** when the arms differ at the **read** site. Its seeds are
also burned: runs here are byte-identical, so re-running its cells under a corrected gate
reproduces exactly the void records.

**Changing venue to depth 5 fixes both at once**: a different measurement rather than a repeat,
seeds 12-23 fresh, n=24 halving the standard error, and **no encoder manufacturing** because all 24
`exp040_encoder_s*.pt` exist.

### The gate was calibrated BEFORE the spec, and that caught a third mistake

The first instinct was "arm A's recall norm is exactly zero". **Wrong**: `MemoryReadout`'s own
docstring records that 65% of the recall block is a memory-free transform of the current concept.
The gate now measures the read site across its full range, at `bb1efa5`:

| attractor state | `recall_content_cos` |
|---|---|
| empty, nothing stored | **1.000000** (the gate CAN fail) |
| random loaded | 0.9437 |
| **real depth-5 run** | **0.8128** (the gate CAN pass) |
| docstring, independent, 79 steps | 0.802 |

**Condition: arm M below 0.95**, plus arm S's `unshuffled_frac` below 0.20. A cosine is bounded and
scale-free, so it cannot repeat EXP-057's regime-dependence.

### Two limits recorded up front

- **Claim 1 is NOT directional.** EXP-058's unlicensed ordering suggests memory may HURT, so a
  significant **-0.05** is pre-registered as a real finding, not a failed confirmation.
- **n=24 is still only ~30-40% powered at a 0.03 effect**, which is roughly what EXP-058 saw.
  `revisit_rate` is the better-powered instrument and is why Claim 2 is a claim, not a footnote.

## 2. What else changed this session

- **`bbd0` audit done and ticked.** The uniform modal floor is budget-dependent; 0.354 is the
  depth-3 figure. Two wrong citations fixed (EXP-038's spec body, EXP-037's depth-4 Claim 5);
  neither changed a conclusion. The recurrence fix is the budget-by-depth table now in
  `modal_action_fraction`'s docstring, extended to depth 7 (0.3010).
- **`c16a` done.** `requirements.txt` gained `-e .[server]`; without it a fresh checkout could not
  COLLECT `tests/server`, and a collection error fails the whole run.
- **`3ef2` done.** The ssh exit-code guidance now points at the **tailscale last-seen field**, with
  both real strings recorded verbatim.
- **`7741` reworded** into `4f13`: the memory re-ask is NOT cheap. The CLI has no edit command, so
  the old entry is ticked and its text is now misleading in the completed list.
- **The playbook gained the comma trap and the background-kill behaviour**, out of rotating
  handoffs and into the durable doc.

## 3. Open items

**Everything laptop-free is done.** What remains is either waiting on compute or waiting on Michael.

1. **Finish EXP-059.** Running, ETA ~2026-09-11 16:30 UTC. When 72 records exist: `scp` them and
   the `*_head.pt` back, run the **already-written** `aggregate.py` (do NOT rewrite it - it was
   authored from the spec before any number existed), produce `RESULTS.md`, run the suite in chunks,
   merge `--no-ff`, delete the branch, add the vault row.
2. **Dispatch EXP-060** once the laptop frees. Spec is written and pre-registered; ~14 h. Vault todo
   added.
3. **DEFER WINDOWS UPDATE ON THE LAPTOP - needs Michael.** Settings > Windows Update > Pause
   updates. The registry write is blocked by the permission classifier here. This is what destroyed
   the first EXP-059 attempt.
4. **EXP-055's two leads**, both needing compute and neither specced: what one epoch builds that
   helps policy while making `S` worse than random, and whether `e2` is a cheaper encoder recipe at
   74% of `e10` for a fifth of the cost. **Deliberately not specced** - EXP-060 has a stronger claim
   on the next slot, and specs written far ahead of their dispatch tend to be re-derived anyway.
5. **Vault, needs Michael**: `0576` see the dashboard render once, `0817` decide the Phase 0/1
   progress-tracker checkpoints.

**Vault `4f13` (re-ask the EXP-030 memory question) is what EXP-059 IS**, and it is deliberately
left open: it is not done until `RESULTS.md` exists, and per the secretary convention a task is
ticked when Michael says so, not when the work looks finished.

## 4. Standing facts

- **READ "THE GATE-CALIBRATION RULE" IN `CLAUDE.md` BEFORE WRITING A VALIDITY GATE.** Two of three
  experiments that used one got it wrong, both the same shape: a threshold chosen in one regime and
  applied in another. **Compute a gate's attainable range before committing it.**
- **The critic question is CLOSED.** Its benefit is within-episode state-dependence, not
  calibration. Everything without it sits 0.1358-0.1558; the full critic sits at 0.2004.
- **FIVE instruments move against policy quality**: the EXP-033 probe, pretraining move-accuracy,
  the entropy trace, `S`, and `critic_ev`. **Use `revisit_rate` and `optimality`.**
- **Unanimity at p 0.0005 measures an instrument's consistency, not its link to the outcome.**
- **Week 20's spiking-encoder training is the ONE neuromorphic change.**
- **Costs, measured:** depth-7 frozen cell ~2.8 h; depth-6 memory cell 3.37 h; depth-5 memory cell
  **unmeasured, estimated 2.5 h**. More workers buy no wall clock on RL cells.
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 5. Operational

- **Never pass comma-separated arguments over ssh**; the failure **exits zero**.
- **Long background commands are killed here around 2-3 h** with empty output. The remote run
  survives; only the notification is lost. `| tail` buffers until exit, so a killed piped command
  tells you nothing. Foreground chunks under 600 s are reliable.
- **The Bash default timeout is 120 s and 600 s is a hard ceiling.**
- **Sync the laptop worktree with `sync_repo.ps1 -Repo ... -Branch ...`, never a bare checkout**,
  and remember `exp040_encoder_s*.pt` are NOT tracked in git: all 24 live on the MAIN checkout and
  were copied into the worktree with hash verification.
- **`C:\Users\mlgbr\repo-attic` holds 18+ folders** that nothing prunes.

## 6. Pointers

- `docs/superpowers/specs/2026-09-09-exp059-memory-depth5-design.md` - the live contract
- `experiments/058_memory_reask/RESULTS.md` - the VOID write-up and the EXP-030 trap it reproduced
- `experiments/057_constant_critic/RESULTS.md` - calibration ruled out
- `CLAUDE.md` - the gate-calibration rule; the re-measured test runtimes
- `docs/playbooks/remote-experiment-runs.md` - dispatch, the comma trap, background kills,
  last-seen diagnosis
- Vault: `experiment-log.md` (through EXP-058 plus four dated corrections)
