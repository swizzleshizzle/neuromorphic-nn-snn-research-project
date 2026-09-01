# Session Handoff - 2026-08-31 (Mon) - for Week 22 session 1

> **Nothing is running. The laptop is idle. `main` is at `0e4ddfe` and carries EXP-053 and
> EXP-054, both merged and both with committed `RESULTS.md`.**
>
> **EXP-055 is pre-registered, built, reviewed and ready to dispatch on branch
> `exp-055-left-edge` (6 commits, `386851e..9a35a2a`, unmerged and no PR).**

## 0. Read this first: the 08-27 handoff is three experiments out of date

The previous handoff presented the Stage 3 fork as unmade. **It was settled on 2026-08-27 and
both branches of it have since been built, run and written up.** Do not act on that document.

One long session ran Thu 08-27 through Mon 08-31 and did roughly three sessions' work: EXP-053,
EXP-054, and EXP-055's design and build.

## 1. What changed

### EXP-053 - Stage 3. The critic works; the neuromorphic claim does not.

Michael chose to run both halves of the fork as one paired design, with the learned critic as the
gate's control. Three arms, 12 seeds, plus a critic-lr pilot selected blind to success rate.

| | arm | control | delta | W-L-T | exact p | bar | verdict |
|---|---|---|---|---|---|---|---|
| Claim 1, critic, d7 | **0.2004** | 0.1471 | **+0.0533** | 8-4-0 | **0.0498** | +0.05 | **CONFIRMED** |
| Claim 2, gate, d6 | 0.3004 | 0.2700 | +0.0304 | 9-3-0 | 0.1323 | +0.05 | not confirmed |
| Claim 3, G vs R, d6 | 0.3004 | 0.2654 | +0.0350 | 7-5-0 | 0.1167 | +0.03 | not confirmed |

**Claim 1 clears both thresholds by a hair - 6% and 0.4% of margin - and its mechanism is
measurably absent.** Critic explained variance is **0.0021**, meaning `V(s)` is no better than a
constant, and both trajectory metrics are null. So *"a learned critic raises depth-7 success"* is
supported and *"because a state-dependent baseline reduces gradient variance"* is not.

**The leading alternative is cheap and untested**: a critic fit by MSE may simply be a
better-calibrated constant than a lagging `beta=0.1` EMA. One arm settles it.

**Claim 5 is the quiet win**: the critic rescued the one dead seed, **0/12 against 1/12**.

**The pre-registered rule retires the neuromorphic claim rather than deferring it.** The bus is
now genuinely load-bearing in code - `NeuromodBus.learning_enabled` and `Brain.learn()` have their
first readers since L11, and the running-median gate produced a realized rate of **0.4987** with no
tuned constant - but it bought nothing measurable. **`road-to-a-solved-cube` and `CLAUDE.md` should
record that Week 20's "nothing neuromorphic participates in the learning" changed ONCE, when the
spiking encoder started training. Arm G must not be written up as a second change.**

One mechanism number moved: **G revisits significantly less than R at an identical update rate,
p 0.0083.** Recorded as a lead and explicitly not a rescue - six descriptive tests were run, so
that p sits exactly at Bonferroni.

### EXP-054 - sequence-blindness is REFUTED

| epochs | 0 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|
| policy | 0.0000 | **0.2012** | 0.1850 | 0.1800 | **0.0887** |
| `S` | **0.0100** | **0.0242** | 0.0241 | 0.0246 | 0.0244 |

**Claim 1 NOT CONFIRMED, 0 of 3 contrasts.** The four trained arms differ by **0.08x their own
within-arm sd** while the policy halves. Whatever causes EXP-052's collapse, **it is not loss of
distance structure in the concept code.** Four experiments had leaned on that hypothesis.

**The spec's own Johnson-Lindenstrauss prediction was refuted too** - it was pre-registered so it
could not be invented afterwards. A random encoder scores **lowest**: E0 minus E10 is **-0.0143 at
p 0.0020**. Pretraining *builds* distance structure, entirely in its first 10 epochs, then
saturates. That independently confirms EXP-052's efficiency finding from a different instrument.

**Not an artifact**: `S_cross` and the `level` control both track `S` exactly, and the
per-separation table shows a genuine gradient rather than a clustering step.

> [!warning] `S` IS NEITHER CLEARED NOR RETIRED, AND ITS CLAIM 4 "PASSED" IS UNINTERPRETABLE
> The aggregator printed CLAIM 4 PASSED from a rank correlation over four arm means whose spread
> is 0.08x the within-arm sd. That is **EXP-052's own process failure** - naming a shape from
> indistinguishable means - **reappearing in the aggregator written to prevent it.** Written up as
> a defect with a named fix (gate the between-arm comparison on the arms being distinguishable),
> not cited as a clearance. Do not cite "S was checked and cleared" in a later spec.

## 2. EXP-055 is ready to dispatch and has NOT been run

Branch `exp-055-left-edge`, unmerged, no PR. Pre-registered at `386851e`, amended at `9a35a2a`,
**both before any number exists**.

**The gap it closes.** Two independent instruments agree everything happens before 10 epochs and
nothing after, and nobody has measured a point inside that window. If one epoch already reaches the
plateau, pretraining's contribution is almost entirely "stop being randomly initialised" and the
EXP-039/040 framing needs rewriting.

**Four arms at 1, 2, 3 and 5 epochs, 12 seeds, every field copied from EXP-052's Phase 2 with one
variable.** Every arm frozen at 390 trainable.

| phase | what | cost |
|---|---|---|
| 1 | pretrain 48 encoders, 10 workers | ~0.5 h |
| 2 | `S` over those encoders, VPS only | seconds |
| 3 | four RL arms, 6 workers | ~17.2 h |

**The final review found two Critical defects in the pre-registration, both fixed before any run:**

1. **The "bound" wording was factually false.** Measured paired-difference sd on the real
   comparison arms is 0.102 to 0.137, so at n=12 a non-significant result is consistent with a true
   effect near **+0.09**, nearly twice the +0.05 bar. The function written to stop null-to-claim
   conversion was performing one. It now reports `mean +/- 2.201 * se` and says n=12 does not
   resolve it.
2. **Claim 1 could fire on the wrong sign.** `abs(delta) >= bar` would have printed CONFIRMED for a
   significant **-0.08** on a directional claim.

**Claim 3 is badly powered on purpose-of-record**: about 28% for its own effect size, so four
"indistinguishable" lines are the likely output whatever the truth is. That is printed by the
aggregator and stated in the spec so a null is not read as flatness. The headline case is fine - if
`e1` is near zero the delta is ~0.20 at power near 1.00.

**PARKED, and it must be fixed before `describe_contrast` is reused anywhere else**: the interval
branch prints "the interval includes effects larger than the bar" unconditionally rather than
gating on the computed upper bound. True for every sd this project has measured, single call site,
same class as the bug it replaced.

## 3. Dispatching on the laptop - THE SETUP IS DIFFERENT NOW

**The laptop's main checkout could NOT switch branches.** 48 EXP-052 `.pt` files are untracked
there but tracked in the branch, and **24 of them differ by hash**. A forced checkout would have
destroyed them. Michael chose a worktree.

- **Worktree: `C:\Users\mlgbr\wt-exp053`.** The main checkout at
  `C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project` is untouched, still on
  `main` at `ea5e482`, with every untracked output where it was.
- **THE WORKTREE HAS NO `.venv`, AND THIS IS A TRAP.** The main checkout's venv installs the
  project editable with an **absolute** path to *its own* `src`. Running the worktree's scripts
  with that interpreter imports the **old** library while looking completely normal. **`PYTHONPATH`
  overrides it and this was verified**, not assumed:
  ```
  $env:PYTHONPATH = "C:\Users\mlgbr\wt-exp053\src"
  $py = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\.venv\Scripts\python.exe"
  ```
- **Launcher: `C:\Users\mlgbr\launch053_wt.ps1`**, which carries both. **Not** the committed
  `launch053.ps1`, which points at the main checkout. EXP-055 needs the worktree switched to
  `exp-055-left-edge` and a launcher variant, not a fresh worktree.
- **`Start-Process ... -WindowStyle Hidden` over ssh DIES when the session ends.** First launch
  produced 8 processes, then 0, with a log stopping at the header and no traceback. Use the
  playbook pattern: run the launcher in the ssh **FOREGROUND** and background it on the controller
  side. Windows has no SIGHUP semantics, so a dropped ssh does not kill it.
- **`exp040_encoder_s*.pt` are NOT TRACKED IN GIT.** They exist only on disk. Arms that need them
  will refuse to start until they are copied in - that guard is deliberate.
- Copy control records in too: EXP-051's and EXP-047's `outputs/*.json` are gitignored and the
  pre-flight guards check for them.

## 4. Operational notes earned this stretch

- **The Bash tool's default timeout is 120 s, and 600 s is a hard ceiling.** Exceeding either makes
  the harness auto-background the command, and a subagent then waits forever for a notification
  that never arrives. **Six stalls traced to this.** The instruction that works is *"always pass an
  explicit timeout, the default is 120 s"* - say the number. Split any suite so no single call
  approaches 600 s. **Never put "run the whole suite" in a task-scoped brief**: one stall was
  scripted by a plan that told an implementer to run a 13-minute suite.
- **Measured per-file test runtimes are now in `CLAUDE.md`.** `test_encoder_finetune_seam.py` is
  519 s and must be run ALONE. The fast suite cannot complete in one call at any timeout.
- **`concept_rates` is batched**, so encoder-level analysis is seconds, not hours. EXP-054's 60
  cells took 8.8 s against a 20-40 minute estimate.
- Quoting through `cmd.exe` still mangles nested PowerShell. For anything non-trivial, `scp` a
  `.ps1` or `.py` over and run it from a file.

## 5. Open items, roughly by value

1. **Run EXP-055.** Built, reviewed, ready. ~18 h. Phase 1 and 2 are cheap and can run first.
2. **Why does the critic work?** Its explained variance is 0.0021, so replace the EMA baseline with
   a per-episode batch-mean constant and no critic. If that matches +0.0533, the state-dependence
   is irrelevant. One arm, ~6.3 h, and it attacks the mechanism of the only claim that confirmed.
3. **Fix EXP-054's Claim 4 gate** - condition the between-arm comparison on the arms being
   distinguishable - and the parked `describe_contrast` gating in EXP-055.
4. **Add the missing row to EXP-053's decision table**: "G does not beat its control, yet beats R".
   It was close to what happened and the table could not express it. Add it BEFORE running anything
   that would use it.
5. **Redo the probe-based inferences** in EXP-033/039/047 with trajectory metrics beside them.
6. **Re-ask the memory question.** EXP-030 was measured on a 2.2% policy; depth 6 now runs at 0.30.

## 6. Standing facts that shape every decision

- **Budget is log-linear**: ~0.22 per log10 at depth 6, 0.210 at depth 7, no knee. Every depth
  number before EXP-046 means "at 10,000 episodes".
- **The encoder line is bounded**: it does not compound (EXP-049) and its advantage over budget
  erodes with depth (EXP-051: 1.69x cheaper at depth 6, 1.09x at depth 7).
- **FOUR instruments now move against policy quality**: the EXP-033 probe (both directions, p
  0.0005), pretraining move-accuracy, the entropy trace (Spearman +0.881 within an arm, opposite
  between arms), and `S` is unevaluated rather than cleared. **Use `revisit_rate` and
  `optimality`.**
- **n >= 12 seeds. Measure the chance floor. No scipy - exact permutation over 2**12 is 4096 and
  cheap.**
- **Do not run depth 8 on the current recipe expecting a bargain**: ~194,000 episodes.

## 7. Pointers

- `experiments/053_neuromod_stage3/RESULTS.md` and `experiments/054_sequence_blindness/RESULTS.md`
- `docs/superpowers/specs/2026-08-31-exp055-pretraining-left-edge-design.md` - pre-registered, amended
- `docs/playbooks/remote-experiment-runs.md` - dispatch, and the corrected estimator
- `CLAUDE.md` - measured test runtimes and the timeout gotcha
