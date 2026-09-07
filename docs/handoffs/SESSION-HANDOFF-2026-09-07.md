# Session Handoff - 2026-09-07 (Mon) - end of Week 22 / start of Week 23

> **Nothing is running. The laptop is idle. `main` is at `99dcfdd`, clean, no branches, no PRs.**
>
> **Full suite: 588 tests.** Every original Week 22 item is closed. **EXP-057 answered the critic
> question. EXP-058 is VOID by its own pre-registered gate, and that is the most important thing
> in this document.**

## 0. Read this first: the 09-03 handoff is superseded, and 08-31 is DANGEROUS

**The 08-31 handoff must not be acted on at all.** Its item 2 proposes a per-episode batch-mean
baseline arm. That arm forms `G_t - mean(G)`, exactly zero on a one-step episode, and depth 1
averages 1.22 steps per episode, so it would be starved of gradient precisely where the critic
predicts best. **The confound is aligned with the hypothesis.** It was disqualified before running
and the question it wanted to ask has since been answered properly by EXP-057.

The 09-03 handoff is merely out of date: its top item is done.

## 1. What changed

### EXP-057 - calibration is NOT the mechanism. The critic question is closed.

A critic fitted by MSE but **blind to the state** beats the lagging EMA baseline by **+0.0088 at
p 0.7822**. Not merely non-significant: a near-zero point estimate on the one question the arm
existed to answer.

| arm | success | what it has |
|---|---|---|
| `B`, full critic | **0.2004** | calibration + between-episode + within-episode |
| `C`, constant critic | 0.1558 | calibration only |
| EXP-051 EMA | 0.1471 | neither, and it lags |
| `F`, flattened (EXP-056) | 0.1358 | calibration + between-episode |

**Everything without within-episode state-dependence sits between 0.1358 and 0.1558; the one arm
that has it sits at 0.2004.** That is a **shape across two pre-registered experiments, not a
resolved decomposition**: only EXP-056's `F` vs `B` cleared a threshold (p 0.0234, and by 6.4% of
its Bonferroni margin), and every EXP-057 contrast is a bound.

### EXP-058 - VOID, and the fault was in the spec

Three arms at depth 6 on the best config on record, 36 cells, **20.2 h**. **Claim 3's gate required
`mean_n_stored > 10`.** That quantity is bounded by episode length, and depth-6 episodes average
**7.76 steps** because this policy *solves* them before the 15-step cap. **Unsatisfiable by
construction.** It also could not discriminate: `use_memory = (readout != "concept")` means all
three arms store, and the amnesic arm zeroes `W_rec` at the READ site only.

**No threshold was edited after the numbers existed.** The claims are not reportable.

**But the ordering reproduces EXP-030's trap exactly**, and that is worth carrying forward:

| arm | success |
|---|---|
| A, amnesic | **0.3425** |
| M, memory | 0.3154 |
| S, shuffled | 0.2979 |

**Memory beats the shuffle-null (+0.0175) and does NOT beat the amnesic control (-0.0271).**
EXP-030 found the identical structure on a policy 15x worse. **The same two-arm design would have
reported memory as a win both times.** The largest contrast is **S - A at -0.0446, p 0.0859**:
*incorrect* memory hurts, which is what a memory-versus-shuffle-null comparison actually measures.
Fixing M vs A as primary before any number existed was vindicated even though the gate never
licensed reading it.

### The gate-calibration rule is now in `CLAUDE.md`

**Two of the three experiments that used a validity gate got it wrong, both this month, both the
same shape: a threshold chosen in one regime and applied in another.** EXP-057's was calibrated on a
depth-3 smoke run and evaluated at depth 7, passing with 2.0x margin instead of the three orders its
spec claimed. EXP-058's was unsatisfiable. **Read that section before writing another gate.**

### The `bbd0` audit is done

The uniform modal floor is budget-dependent and **0.354 is the depth-3, 9-step figure**. Two wrong
citations found and fixed (EXP-038's own spec body, and EXP-037's Claim 5 which is depth 4); neither
changed a conclusion, since the correct lower floor makes both policies look *more* collapsed. The
recurrence fix is the full budget-by-depth table now in `modal_action_fraction`'s docstring,
validated two ways and extended to depth 7 (0.3010).

## 2. THE LAPTOP IS NOT READY TO DISPATCH

Same trap as last time, and it does not error.

1. **The worktree `C:\Users\mlgbr\wt-exp053` is on `exp-058-memory-reask` at `4a96137`**, two
   commits behind a branch that **no longer exists on origin**.
2. **Sync with `sync_repo.ps1`, never a bare checkout:**
   ```bash
   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1 -Repo C:\Users\mlgbr\wt-exp053 -Branch main'
   ```
3. **The worktree has no `.venv`**, so the only interpreter imports the MAIN checkout's `src`
   unless `PYTHONPATH` overrides it. That failure is silent and produces a plausible wrong result.
   Every launcher from `launch055_wt.ps1` onward refuses to start unless
   `neuromorphic.__file__` resolves under the worktree, and each also proves the specific symbol
   its experiment needs is present. **Copy that pattern.**
4. **Housekeeping: `C:\Users\mlgbr\repo-attic` now holds 18 timestamped folders.** Every sync moves
   colliding untracked files there. Nothing prunes them and nobody has checked whether any holds
   something not since committed.

## 3. Open items

1. **A successor to EXP-058, and it needs a design decision first.** The claim thresholds were never
   contaminated and can be reused. **The seeds are burned**: runs here are byte-identical, so
   re-running seeds 0-11 under a corrected gate reproduces exactly these records, which is
   laundering rather than replication. An uncontaminated re-test needs either **new seeds** (which
   requires re-running the EXP-047 fine-tune and EXP-049 second round first, because E2 encoders
   exist only for seeds 0-11) or **a different measurement**, such as the same question at another
   depth on a base config whose encoders already exist more widely. **The second is cheaper and is
   a genuinely new measurement rather than a repeat.**
   - The successor has a prior: effects around **-0.03** for M vs A and **-0.045** for S vs A, both
     under the +0.05 bar EXP-058 set. Raise n or lower the bar deliberately, and say which first.
   - Use a gate on **recall**, not storing: non-zero recall-block norm in M against exactly zero
     in A.
2. **Repeat EXP-056 at higher n. COST CORRECTED: this is ~25 h, not the ~6 h an earlier note
   claimed.** Its contrasts are paired by seed and every arm exists only at seeds 0-11, and E1
   encoders only reach seed 13, so it needs pretrain E0 (~1.9 h) plus the EXP-047 fine-tune (~12 h)
   plus both arms at 24 cells (~11.4 h). **14 of the 25 h is re-manufacturing encoders.** Still
   worth doing eventually: `p 0.0234` against `0.025` is thin for a result two experiments lean on.
3. **EXP-055's two leads.** What does one epoch build that helps policy while making `S` worse than
   random, and is `e2` a cheaper encoder recipe at 74% of `e10` for a fifth of the cost.
4. **A standing note on the retired instruments.** There are **five**, and the pattern now has
   enough instances to be written once instead of rediscovered.
5. **Vault chores**: `0576` dashboard render, `0817` progress-tracker checkpoints, `c16a`
   requirements extra, `3ef2` playbook ssh-255 wording. **`bbd0` is complete and can be ticked.**

## 4. Standing facts

- **FIVE instruments move against policy quality**: the EXP-033 probe (both directions at p 0.0005,
  plus a per-seed argument from the re-analysis), pretraining move-accuracy, the entropy trace,
  **`S`**, and **`critic_ev`**. **Use `revisit_rate` and `optimality`.**
- **`critic_ev` must not gate critic work.** EXP-056's worse-performing arm had the better-fitting
  critic at every stage.
- **Unanimity at p 0.0005 measures the instrument's consistency, not its link to the outcome.** The
  probe was unanimous at every depth and still ranked seeds no better than chance.
- **Week 20's spiking-encoder training is the ONE neuromorphic change.** Arm G made the bus
  load-bearing in code and bought nothing measurable; the pre-registered rule retires it.
- **Costs, measured:** depth-7 frozen cell ~2.8 h; depth-6 memory cell **3.37 h**; more workers buy
  no wall clock on RL cells, so pick the count from memory headroom (~1.05 GB private each).
- **A long run survives the laptop sleeping.** EXP-058 slept 26 h of its 39.7 h wall clock in
  transit and resumed with nothing lost. Judge progress by CPU-hours per worker, not wall clock.
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 5. Operational notes

- **Never pass comma-separated arguments over ssh.** `cmd.exe` eats the commas and the failure
  **exits zero**. Verify a launch by probing for records and worker processes, never by exit code.
- **Client-side exit codes say nothing about the job.** EXP-055 reported 143 because the
  controller's own timeout fired one minute before the run finished successfully.
- **Long background commands are killed in this environment**, repeatedly, with empty output and no
  OOM evidence, at roughly the 2-3 h mark. It never cost work, only wake-ups. Foreground chunks
  under 600 s are reliable; `| tail` buffers until exit, so a killed piped command tells you
  nothing.
- **`CLAUDE.md`'s test chunking is re-measured** and the `tests/training` remainder must be
  backgrounded or split further.

## 6. Pointers

- `experiments/057_constant_critic/RESULTS.md`, `experiments/058_memory_reask/RESULTS.md` (read the
  VOID banner and the EXP-030 trap callout)
- `CLAUDE.md` - **the gate-calibration rule**, and the re-measured test runtimes
- `docs/superpowers/specs/2026-09-04-exp058-memory-reask-design.md` - the gate diagnosis is
  annotated in place
- `experiments/probe_reanalysis/RESULTS.md` - why unanimity is not relevance
- Vault: `experiment-log.md` (current through EXP-058 plus four dated corrections),
  `road-to-a-solved-cube.md`
