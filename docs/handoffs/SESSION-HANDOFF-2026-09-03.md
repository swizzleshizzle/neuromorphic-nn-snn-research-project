# Session Handoff - 2026-09-03 (Wed) - end of Week 22

> **Nothing is running. The laptop is idle. `main` is at `08c60bd`, clean, and carries EXP-055,
> EXP-056 and the probe re-analysis, all merged with committed `RESULTS.md`.**
>
> **No branches exist. No PRs are open.** Full suite green: **580 tests, 561 not slow and 19 slow.**

## 0. Read this first: the 08-31 handoff is superseded, and one of its items is ACTIVELY WRONG

That document is three experiments out of date and **its item 2 proposes an arm that the data has
since disqualified.** Do not act on it.

It said: *"replace the EMA baseline with a per-episode batch-mean constant and no critic."* **That
arm forms `G_t - mean(G)`, which is exactly zero on a one-step episode, and depth 1 averages 1.22
steps per episode with 78% of episodes solved.** It would be starved of gradient precisely in the
stages where the critic predicts best, so a loss against the critic would be uninterpretable: the
confound is aligned with the hypothesis. **A fresh session following that handoff would spend six
hours building a control that cannot answer its own question.**

The reason it looked sensible is also recorded below: EXP-053's mechanism claim was wrong.

## 1. What changed

### EXP-055 - the left edge is real, and `S` is refuted as a policy predictor

Four arms at 1, 2, 3 and 5 pretraining epochs, 12 seeds, every arm frozen at 390 trainable.

| epochs | 0 | **1** | **2** | **3** | **5** | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|---|---|---|
| policy | 0.0000 | **0.0854** | **0.1483** | **0.1746** | **0.1762** | 0.2012 | 0.1850 | 0.1800 | 0.0887 |

**Claim 1 CONFIRMED: `e10 - e1 = +0.1158 at p 0.0098.`** One epoch reaches only **42.5%** of the
plateau, so pretraining is **not** almost entirely "stop being randomly initialised" and **the
EXP-039/040 framing survives**. Roughly 42.5% from escaping random init, 57.5% from the objective.

Claim 3 resolves exactly one step, `e1 -> e2` at **+0.0629, p 0.0068**, the largest resolved step
anywhere in the 0-to-80 curve after the jump off zero. **The other three are UNRESOLVED, not flat**,
at a pre-registered ~10% power. Do not read them as a plateau.

**The result neither phase could produce alone:** at the 0-to-1 transition `S` falls **below its own
initialisation** (11/12 seeds, p 0.0010) while policy rises from exactly 0.0000 to 0.0854 (12/12).
Both sides resolved, and E0 was verified to BE the initialisation pretraining starts from
(reproduced from `make_sensory(seed)` to within 1e-9). **`S` is refuted as a policy predictor**, not
merely unevaluated as EXP-054 left it.

### EXP-056 - the critic's benefit IS its within-episode state-dependence

| | success |
|---|---|
| arm `B`, full critic (EXP-053) | 0.2004 |
| arm `F`, flattened to episode mean | **0.1358** |
| EXP-051 EMA baseline | 0.1471 |

**Claim 1: `F` BELOW `B`, -0.0646 at p 0.0234, W-L-T 3-9-0.** Flattening `V(s_t)` to its own episode
mean collapses the arm to the level of the EMA baseline it was meant to beat (-0.0112 against it,
p 0.6709). **EXP-053's +0.0533 is entirely its within-episode state-dependence.**

The validity gate was checked BEFORE the verdict and passed emphatically: `V`'s within-episode RMS
is **2.14x the returns' own** at depth 1 and never below 0.415x, against a 0.05 threshold.

**Honest limits, not buried:** p 0.0234 clears Bonferroni 0.025 by **6.4% of its margin**, the same
knife-edge EXP-053's Claim 1 sat on. Claim 2 is a bound, not an equivalence. n=12.

### The probe re-analysis - direction right, ranks no seeds

Closes item 5 of the old handoff. **Not an experiment**, no data generated,
`experiments/probe_reanalysis/`.

EXP-039's probe gain was unanimous at every depth at p 0.0005. Put beside behaviour, the direction
holds: success rose significantly at depths 4, 5 and 6, so **the pretrained-encoder line is not
weakened**. But across **15 correlations between a probe movement and a behavioural one, 2 were
nominally significant and 0 survived Bonferroni 0.0033**, and **both nominal hits run the wrong
way**.

> **Unanimity at p 0.0005 measures consistency of the INSTRUMENT, not its connection to the
> outcome.** The probe was unanimous at every depth and still ranked the seeds no better than chance.

**EXP-033's width finding must now be quoted narrowly**: "wider random projections decode better",
never "width would not have helped the policy", because no policy was ever trained at width 128, 256
or 512. **EXP-033's Finding 2 survives untouched** because it is not a probe-based inference at all:
it ran the probe AS A POLICY and compared success rates directly. **EXP-047 did not over-claim** and
its numbers reproduce exactly.

### Corrections that landed, each made while nothing pending could use it

- **EXP-053's "the mechanism is measurably absent" was generalised from ONE stage.** `critic_ev` is
  **+0.4702 at depth 1** and +0.2054 at depth 2, going negative only from depth 4, with every seed
  positive at depths 1 to 3. The 0.0021 figure is the final stage only. **This correction is what
  disqualified the old handoff's item 2 and produced EXP-056.**
- **EXP-054's Claim 4 gate**: now prints UNEVALUATED rather than a meaningless PASSED, computed from
  a resolvability test on both axes.
- **EXP-055's parked `describe_contrast` interval gating**: it asserted the interval reached the bar
  without consulting the interval.
- **EXP-053's decision-table row** for "G does not beat its control, yet beats R", added with
  nothing pending and verified not to change that run's reported verdict.
- **EXP-053's 66 orphaned checkpoints** committed. It was the only experiment from 036 onward with
  none tracked, and it was two `.gitignore` negations missed, not one.
- **`CLAUDE.md`'s per-file test runtimes** were wrong in both directions, and **the playbook's
  implied wall-clock return from more workers** was wrong.

## 2. THE LAPTOP IS NOT READY TO DISPATCH AS-IS

Three things must be fixed before the next run, and none of them errors loudly.

1. **The worktree `C:\Users\mlgbr\wt-exp053` is on `exp-056-flattened-critic` at `e7e0d73`**, which
   is **one commit behind the branch tip and the branch no longer exists on origin.** It must be
   synced to `main` before anything is dispatched.
2. **Use `sync_repo.ps1`, not a bare checkout.** It is fully parameterised and was used successfully
   this week:
   ```bash
   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1 -Repo C:\Users\mlgbr\wt-exp053 -Branch main'
   ```
   Expect it to move untracked files aside. It already did once: **114 files** (EXP-053's 66 and
   EXP-055's 48) are in `C:\Users\mlgbr\repo-attic\20260902-002436`. Moved, never deleted; the
   tracked copies on `main` are authoritative.
3. **The launchers carry a wrong-library gate and it is load-bearing.** The worktree has no `.venv`,
   so the only interpreter installs the project editable with an absolute path to the MAIN
   checkout's `src`. Without `PYTHONPATH` it imports the old library and produces a complete,
   plausible, wrong result with no error. `launch055_wt.ps1` and `launch056_wt.ps1` both refuse to
   start unless `neuromorphic.__file__` resolves under the worktree, and `launch056_wt.ps1`
   additionally proves the library HAS the symbol the experiment needs. **Copy that pattern.**

## 3. Operational notes earned this week

- **NEVER pass comma-separated arguments over ssh.** `cmd.exe` treats commas as separators, so
  `-Epochs 1,2,3,5` arrived as the single token `1235`. The `ValidateSet` refusal was correct and
  **the ssh still exited ZERO**, because PowerShell parameter binding fails before the script body
  runs. Use a switch. **Verify a launch by probing for records and worker processes, never by the
  ssh exit code.**
- **A client-side exit code says nothing about the job.** EXP-055's dispatch reported `exit 143`
  because the controller's own `timeout 64800` killed the ssh client **one minute before the job
  finished successfully**. Probe before reacting.
- **More workers buys no wall clock on RL cells.** EXP-055 phase 3 took **18.0 h at 8 workers**
  against a pre-registered 17.2 h estimate at 6. Per-cell time rises with contention as fast as the
  workers add throughput. **Pick the count from memory headroom**: RL workers measure ~1.05 GB
  private each, so 10 would have breached this box's commit line.
- **Pretraining's contention knee is BELOW 8**, not between 8 and 10: 86.7 s/epoch/seed at 8 against
  87.3 at 10 and 24.9 at 2.
- **`CLAUDE.md`'s test table is re-measured and the old chunking recipe does not complete.** The
  `tests/training` remainder is ~837 s and must be backgrounded. To find expensive files, loop with
  a per-file `timeout` inside ONE call rather than bisecting.
- **Two `-m slow` background runs were killed with empty output and no OOM evidence.** Foreground
  chunks worked both times. If it happens again, do not retry the same way.

## 4. Open items, roughly by value

1. **The CONSTANT-CRITIC arm.** Already pre-registered in EXP-056's spec section 5. A single learned
   scalar, no state input, fitted by the same MSE loss with the same optimizer and rate. It differs
   from arm B in exactly one way (cannot see the state) and from the EMA baseline in exactly one way
   (fitted rather than exponentially averaged), and it has **no one-step degeneracy**. It separates
   calibration from between-episode state-dependence, which EXP-056 deliberately could not. ~4-6 h.
2. **Repeat EXP-056 at higher n.** p 0.0234 against a 0.025 threshold is thin for a result this
   load-bearing.
3. **Re-ask the memory question** (vault todo `7741`). EXP-030 was measured on a **2.2%** policy;
   depth 6 now runs at 0.30. **The honest version needs the three-arm design (memory, shuffle-null,
   amnesic) at depth 6: 36 cells, roughly 15 h**, and its own pre-registration. The vault todo says
   head checkpoints make it "cheap (no retrain)"; that is true only of a narrower eval-only
   question, and evaluating a head trained without memory with recall switched on is a distribution
   shift, not a controlled comparison. **EXP-030's 144 records were single-copy on the laptop and
   are now on the VPS**, inside restic.
4. **EXP-055's two leads.** What does the encoder learn in one epoch that helps policy while making
   its distance structure worse than random? And `e2` reaches 74% of `e10` for a fifth of the
   pretraining cost.
5. **A standing note on the retired instruments.** There are now **five**, and the pattern is
   consistent enough to be worth writing once rather than rediscovering per experiment.

## 5. Standing facts that shape every decision

- **FIVE instruments now move against policy quality**: the EXP-033 probe (both directions at
  p 0.0005, plus the per-seed argument above), pretraining move-accuracy, the entropy trace,
  **`S`** (refuted outright by EXP-055, not merely unevaluated), and now **`critic_ev`**.
  **Use `revisit_rate` and `optimality`.**
- **`critic_ev` MUST NOT gate future critic work.** EXP-056 showed the worse-performing arm had the
  better-fitting critic at every stage. A REINFORCE baseline is unbiased for any function of state
  and its job is within-episode variance reduction; `critic_ev` measures pooled stage-level
  prediction instead. EXP-053's own lr pilot predicted arm B would be null on exactly this basis,
  and running the pre-registered arm anyway is what caught it.
- **Week 20's spiking-encoder training is the ONE neuromorphic change.** Arm G made the bus
  load-bearing in code and **bought nothing measurable**; the pre-registered rule retires it. It
  must never be written up as a second change. Recorded in the vault's `road-to-a-solved-cube`.
- **Budget is log-linear**: ~0.22 per log10 at depth 6, 0.210 at depth 7, no knee.
- **The encoder line is bounded**: it does not compound (EXP-049) and its advantage over budget
  erodes with depth (EXP-051).
- **n >= 12 seeds. Measure the chance floor. No scipy** - exact permutation over `2**12` is 4096.
- **Do not run depth 8 on the current recipe expecting a bargain**: ~194,000 episodes.

## 6. Pointers

- `experiments/055_pretraining_left_edge/RESULTS.md`, `experiments/056_flattened_critic/RESULTS.md`,
  `experiments/probe_reanalysis/RESULTS.md`
- `experiments/053_neuromod_stage3/RESULTS.md` - read the **2026-09-02 correction callout**
- `docs/superpowers/specs/2026-09-02-exp056-flattened-critic-design.md` - **section 5 pre-registers
  the constant-critic successor**
- `docs/playbooks/remote-experiment-runs.md` - dispatch, and the corrected worker guidance
- `CLAUDE.md` - re-measured test runtimes and the chunking that actually works
- Vault: `300 Efforts/Active/Coding/Neuromorphic Development/experiment-log.md` (current through
  EXP-056 plus two dated corrections) and `road-to-a-solved-cube.md`
