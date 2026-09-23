# Session Handoff - 2026-09-23 - WEEK 25 DONE. Reproducibility measured; nothing running.

> **Nothing is running. The laptop is FREE and idle. `main` is clean at `8bbb0d3`, no branches,
> no PRs.** Suite: **714 tests, 692 not slow, 22 slow**, plus **88 vitest + 2 e2e** in
> `dashboard/`. Tags: `phase-2-complete`, `phase-3-checkpoint`.
>
> **Weeks 24 and 25 are both complete.** Week 25 delivered its checkpoint early (Sep 17) and then
> a **reproducibility audit** (Sep 22-23). **Phase 4 starts Oct 5 and is laptop-free.**
>
> **There is no required next task.** Read section 4 before inventing one.

## 0. The one thing that still needs Michael

**Windows Update is still unpaused** (`PauseUpdatesExpiryTime = NONE`). It has destroyed ~19
CPU-h outright. Confirm Settings shows "Updates paused until <date>" before any dispatch that
will still be running at 03:31 local.

## 1. The headline, and it changes what Phase 4 may claim

==**Published numbers reproduce by RE-EVALUATING the tracked checkpoints. They do NOT reproduce
by retraining.**==

**Retraining is not portable, and not by a little.** Two probes, two machines, weights compared
byte for byte:

| probe | laptop (made the originals) | this VPS (second x86) |
|---|---|---|
| E0 encoder pretraining | **12 of 12 byte-identical** | **0 of 2**, 7.1% of params, cosine 0.77 |
| **EXP-036 depth 3, no pretrained encoder** | (originals) | **0 of 1** - **0 of 390 params, cosine 0.524** |

**EXP-036 is the load-bearing probe**: its only inputs are code and a seed, so its failure rules
out "this is a quirk of encoder pretraining". **No seeded training run in this project reproduces
off the machine it ran on.**

**But re-evaluating a tracked checkpoint reproduces exactly.** Loading the published EXP-036
heads on both machines: `success_rate`, `mean_steps`, `optimality` and `eval_revisit_rate` all
identical to **full float repr**; only `greedy_modal_action_frac` differs, by **1 ULP**, being a
mean whose summation order moves.

So `.gitignore`'s stated reason for tracking `*_head.pt` - *"what makes never retrain to
re-evaluate true from a fresh checkout"* - is the load-bearing decision in this repo, and it is
now measured rather than assumed.

**Say this in the write-up:** every published number reproduces from a fresh clone by loading the
tracked checkpoints and re-evaluating. **Do not say** "seeded runs are byte-identical" without
the qualifier **within a machine**.

## 2. Artifacts secured, and one correction worth keeping

| artifact | before | now |
|---|---|---|
| E0 pretrained encoders (24) | untracked, laptop-only | **tracked**, 2.7 MB |
| EXP-043 depth-6 heads, seeds 14-23 (10) | untracked, laptop-only | **tracked** |
| EXP-047 fine-tuned heads, seeds 14-23 (10) | untracked, laptop-only | **tracked** |
| EXP-047 fine-tuned **encoders**, seeds 14-23 | already tracked | unchanged |

**Fifteen experiments load the E0 encoders frozen** (EXP-041 through EXP-066), so they carried
every result from week 18 onward.

> **A first reading looked worse than it was.** The EXP-047 *encoders* for seeds 14-23 were
> already tracked - that is the expensive artifact, a ~12 h fine-tuning run. Only the cheap heads
> were missing. "A replication is unreproducible" and "some checkpoints were untidy" are very
> different claims and the first reading was wrong.

**A regenerated artifact can be VALID without being THE artifact**: a regenerated E0 encoder
scores move-accuracy 0.4346 against the original's 0.4373, inside the seed spread 0.4300-0.4373.
So the **findings** would plausibly survive retraining while the **numbers** would not. Say which
one you mean.

## 3. Documentation defects fixed

- **EXP-040's `RESULTS.md` claimed its `*_encoder_*.pt` files were tracked. Zero were.** Anyone
  following it to reproduce would have hunted for artifacts not in the checkout.
- **`CLAUDE.md`'s seeding bullet** now says "within a machine". It was not false - it said
  "across worker scheduling" - but it invited the inference that would have made the release
  claim wrong.

## 4. What is left, and what is NOT

**Nothing is required.** Weeks 24 and 25 are complete.

1. **Sep 28 to Oct 4 is an unscheduled gap week.**
2. **The compute window is effectively closed.** Phase 4 is laptop-free documentation.
3. **Michael's todos**: `0576` (see the dashboard render - possible now, tunnel command below)
   and `e491` (magnitude-matched arm T, which its own todo argues against).
4. **Untested by this audit**: whether findings survive retraining (an argument, not a
   measurement); any experiment other than EXP-036 and EXP-040 phase 1; non-x86 or GPU hosts;
   the `dashboard/` JS toolchain.

**What NOT to do:**
- **Do not re-run the topology question.** Three orders of magnitude of `region_lr` all sit at
  the floor and the collapse is monotone. It needs a different ARCHITECTURE, not a hyperparameter.
- **Do not re-run the monolithic contrast** for criterion 2: it is vacuous and
  `tests/training/test_topology_contrast_is_vacuous.py` proves why.
- **Do not claim reproducibility by retraining.** Section 1.
- **Do not schedule work against a publishing date.** Content Day is defunct.

## 5. Tools added this week

| tool | answers |
|---|---|
| `experiments/040_pretrained_encoder_policy/verify_regeneration.py` | is encoder pretraining reproducible, here vs there |
| `scripts/verify_e2e_reproduction.py` | does retraining a published cell reproduce its checkpoint |
| `scripts/verify_eval_portability.py` | does re-evaluating a tracked checkpoint reproduce its metrics |
| `scripts/wt-switch-branch.ps1` | switch the laptop worktree without losing untracked checkpoints |

**Run the first three on any machine that will host the public release.** The fourth replaces a
dance hand-patched four times, once leaving the worktree as the sole copy of 24 checkpoints;
`git checkout -f` would work and is wrong, because it discards untracked files without comparing.

## 6. Standing facts

- **Confirm completion from COUNTS, never the log.**
- **Mid-run progress is not a rate.** 19 of 24 cells looked like a 42% overrun; the remaining
  five were one in-flight wave and it finished 13% over. Waves, not cells.
- **Anything the record dict reads must be initialised in BOTH branches** of `run_cube_baseline`.
- **Drive PowerShell from a FILE, never `ssh ... -Command`** for anything non-trivial: the
  quoting eats backslashes and has silently produced no output twice.
- **`fd` respects `.gitignore`**, so it reports 0 files in a gitignored `outputs/` even where git
  tracks them via a negation. Use `ls`/`git ls-files` when auditing artifacts.
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 7. Viewing the dashboard

tmux session `neuroscope` on this VPS serves `127.0.0.1:4173`. Restart with
`npx vite preview --port 4173 --host 127.0.0.1` from `dashboard/`. Reach it with
`ssh -N -L 4173:localhost:4173 root@69.167.169.11`. The cube build needs
`VITE_TRACE_URL=/cube_dashboard_trace.jsonl`.

**Every NEURO-SCOPE frame ever rendered shows a decision the agent never made** - it draws
`out["action"]`, the brain's own pathway, which no experiment's policy used.

## 8. Pointers

- `docs/reproducibility-audit.md` - the measurements, the claim, and what was not checked
- `experiments/066_region_lr_sweep/RESULTS.md` - the topology line closed
- `docs/phase3-honest-assessment.md` - the checkpoint, with its own dated correction
- Vault: `experiment-log.md` (through EXP-066 plus eight addenda), `progress-tracker.md`,
  `road-to-a-solved-cube.md`, weekly notes 21-25
