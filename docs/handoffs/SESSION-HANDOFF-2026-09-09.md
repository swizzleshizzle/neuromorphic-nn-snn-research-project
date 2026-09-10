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

## 0e. WAVE 1 LANDED - the per-cell cost is MEASURED, and the playbook formula was right

**2026-09-10 02:40:48 UTC: the first 6 records exist**, all arm A, seeds 0-5. Readings at
04:07 UTC: uptime 20.52 h against a run age of 4.88 h (no reboot), 8 processes, six workers at
**4.32 CPU-h**, ratio **0.885**, no sleep.

| quantity | value |
|---|---|
| wave 1, launch to 6th completion | 23:14 -> 02:40 UTC, **3.44 h wall** |
| per cell | **~3.05 CPU-h** at the measured ratio |
| projected total, 12 waves | **~41.3 h wall** |
| **ETA** | **2026-09-11 16:30 UTC**, plus any sleep |

> [!note] **THE CORRECTED `steps()` FORMULA PREDICTED 3.03 CPU-H. MEASURED ~3.05.**
> That is the playbook fix validating itself on the first real number. The hand-scaled 2.5 h
> estimate that priced this run was the error, and its cause is recorded there: multiplying the
> step-budget ratio by the stage-count ratio double-counts, because a curriculum SPLITS a fixed
> episode count across stages. **Use `steps()`; do not scale a measured cell by hand.**

**Two corrections to expectations set earlier this session:**

1. **The log does NOT carry per-cell seconds.** It prints `0s` on every line, because
   `run.py` reads `r.get("seconds", 0)` and the record has no `seconds` field. Per-cell cost had
   to be inferred from the log's mtime instead. Any future check that promises "the log carries
   per-cell seconds" is wrong; **the mtime of the last completion is the usable signal**.
2. **41.3 h is a FLOOR, not a symmetric estimate.** All six wave-1 cells were arm A. Arms M and S
   are not yet priced, and the run is `--skip-existing`-resumable, so treat the ETA as the
   earliest plausible finish.

**Descriptive, decides nothing, but worth a second look at write-up time:** arm A's first six
successes are 0.395, 0.340, 0.310, 0.280, 0.160 and **0.000**, mean ~0.248 against the
`exp043_capped_d5` context of 0.3229. A seed at exactly 0.000 is a collapsed policy rather than a
noisy one. **No claim is affected** - the contrasts are paired by seed, so a low arm-A seed is
subtracted from the same seed's arm M - but it belongs in `RESULTS.md` rather than being noticed
for the first time by a reader.

**Attempt 1 showed 3.21 CPU-h with zero records at a comparable age**, which does not fit a 3.05
CPU-h cell. The plausible reading is contention from Windows Update staging its upgrade, which
rebooted the machine 12 minutes later. **That is a hypothesis, not an established cause**, and it
is recorded as one.

## 0d. PLAN FOR SESSION 2 OF WEEK 23 - the laptop is occupied for ~45 h

**That single fact sets the agenda: session 2 is the work that needs no laptop.** No second dispatch
is possible until roughly the evening of 2026-09-11, so anything requiring compute is blocked and
everything below is deliberately chosen to be independent of it.

### 1. Write `experiments/059_memory_depth5/aggregate.py` from the spec. FIRST, and time-critical.

**Its whole value depends on being written before any number exists**, and the run has just started,
so the window is open now and closes when the first records land. Write it against
`docs/superpowers/specs/2026-09-09-exp059-memory-depth5-design.md` only.

What the spec pre-registers, and the aggregator must implement without reinterpretation:

- **Claim 3 is a CONDITION and is checked FIRST.** Arm M mean `recall_content_cos` **below 0.95**,
  and arm S `unshuffled_frac` **below 0.20**. If either fails, **every claim is void and the
  aggregator must refuse to print the rest** rather than print them with a warning.
- **Claim 1, PRIMARY, `M` minus `A` on success, NOT directional.** Three pre-registered readings:
  `>= +0.05` at `p <= 0.05` confirms; **`<= -0.05` at `p <= 0.05` is a real finding that memory
  HURTS and belongs in the headline**; anything else is a BOUND with its interval and **never an
  equivalence**.
- **Claim 2, `M` minus `A` on `revisit_rate`, confirmed at `<= -0.02` and `p <= 0.05`.** Mind the
  sign: memory is supposed to REDUCE cycling. All four combinations with Claim 1 are pre-registered.
- **Claim 4, `M` minus `S`**, reported with the sentence that it measures the harm of INCORRECT
  memory and is **not** evidence about the benefit of correct memory.
- **Bonferroni 0.0167** across the three inferential contrasts. Claim 3 is a condition, not a
  contrast.
- **The permutation test switches on n**: exact when `n <= 20`, otherwise a fixed-seed
  200,000-sample permutation, and **it must PRINT which it used**. At n=24 exhaustive is 16.8M
  sign flips.

**Test it against synthetic records, not the real ones**, and per the test-strength rule make each
assertion fail against a deliberately broken aggregator: a gate that cannot fail, a sign flipped on
Claim 2, an equivalence claimed from a bound.

### 2. The standing note on the five retired instruments.

Deferred three handoffs running and now has enough instances to write once: the EXP-033 probe,
pretraining move-accuracy, the entropy trace, `S`, and `critic_ev`. **The unifying point is that
unanimity at `p 0.0005` measures an instrument's consistency, not its link to the outcome.** Include
`critic_ev` must not gate critic work, since EXP-056's worse arm had the better-fitting critic at
every stage.

### 3. Spec the EXP-056 repeat so it is ready the moment the laptop frees.

~25 h, of which **14 h is re-manufacturing encoders**. Worth doing because `p 0.0234` against a
`0.025` threshold is thin for a result two experiments lean on. Writing the spec now costs nothing
and removes the design work from the critical path.

### Not in session 2

- **Anything needing the laptop.** Blocked until ~2026-09-11 evening.
- **Merging the branch.** It carries EXP-059's driver and cannot merge before its `RESULTS.md`.
- Vault `0576` and `0817` still need Michael.

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

1. **Finish EXP-059** (section 0).
2. **Repeat EXP-056 at higher n, ~25 h**, of which 14 h is re-manufacturing encoders. `p 0.0234`
   against `0.025` is thin for a result two experiments lean on.
3. **EXP-055's two leads**: what one epoch builds that helps policy while making `S` worse than
   random, and whether `e2` is a cheaper recipe at 74% of `e10` for a fifth of the cost.
4. **A standing note on the five retired instruments.** Enough instances now to write once.
5. **Vault, needs Michael**: `0576` see the dashboard render once, `0817` decide the Phase 0/1
   progress-tracker checkpoints.

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
