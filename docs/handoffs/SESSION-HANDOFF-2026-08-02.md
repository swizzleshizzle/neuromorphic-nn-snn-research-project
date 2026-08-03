# Session Handoff - 2026-08-02 (Sat/Sun) -> Week 18

> **Nothing is in flight. The repo is clean at `b3c89bf` and pushed. The laptop is idle.**
>
> **Depth-3 success went from 2.2% to 50.0% this weekend**, with no architectural change. Five
> experiments landed (EXP-031 through EXP-035). Read `CLAUDE.md` first, then this file, then
> `experiments/035_budget_scaling/RESULTS.md` for the current state of the art.
>
> Strategy for what comes next is NOT in this repo: it is in the Obsidian vault at
> `300 Efforts/Active/Coding/Neuromorphic Development/road-to-a-solved-cube.md`.

## 0. State check

```bash
git log --oneline -1                 # expect b3c89bf
git status --short                   # expect clean
ls experiments/ | tail -5            # 031..035 all present with RESULTS.md
```

The laptop was unreachable at handoff time (asleep or off). That is expected, nothing is running on
it, and its last known state was in sync at `b3c89bf`. Verify with a `git log` there before dispatching.

## 1. What happened, in one line each

| exp | question | answer |
|---|---|---|
| **031** | Is the depth-3 policy collapsed to a constant action? | **Yes.** 7/12 seeds play one action for all 9 steps. Was already latent in EXP-030's own revisit numbers. |
| **032** | Do the ADR-0001 stabilizers fix it? | **Partially, and it does not help.** Collapsed seeds 11/12 -> 3/12, success still ~0. Normalization ALONE is actively harmful. |
| **033** | Is the representation the bottleneck? | **No.** The same frozen concept@64 supports 0.481 under supervised fit vs 0.022 under REINFORCE. Width refuted as a lever. |
| **034** | Is it the learning signal? | **Yes, and it is an interaction.** Curriculum 1->2->3 at a matched budget beats direct; extra episodes alone are worth -0.003 at p=1.000. |
| **035** | Does the curriculum gain saturate? | **No.** 600 -> 3k -> 10k -> 30k gives 0.097 -> 0.256 -> 0.397 -> **0.500**. Still climbing. |

Current standing: **50.0% at depth 3**, measured random floor 1.4%, v1 baseline 2.2%. Same frozen
randomly-initialised brain, same 64-wide concept, same `Linear(64 -> 6)` head, **390 trainable
parameters**. Only the order the problems were presented in, and for how long.

## 2. Code changes that outlive the experiments

- **`CubeConfig.curriculum`** (default `()`): splits the episode budget across depths rather than
  multiplying it, so a curriculum arm never buys extra compute. Evaluation always at `cfg.depth`.
- **`CubeConfig.encoder_seed` / `train_seed` / `split_seed`** (default `None` -> fall back to `seed`).
  `cfg.seed` previously drove five independent things at once, which made seed variance
  unattributable. `None` is the only sentinel because `encoder_seed=0` must not read as unset.
- **`record_filename()`** extracted, with a warning listing everything it does NOT encode
  (`entropy_beta`, `normalize_advantages`, `episodes`, `curriculum`, the three new seeds). Any sweep
  over those must make `tag` unique per cell.
- **Two collapse instruments** on `cube_baseline`: `greedy_modal_action_frac` and `mean_train_entropy`.
- **`scripts/verify_instrument_neutrality.py`** compares two record sets field by field and fails on
  zero shared filenames rather than reporting a vacuous pass.

**All default paths verified byte-identical** across both refactors against reference values measured
on this machine: `success_rate 0.6666666666666666`, `revisit_rate 0.16633266533066132`,
`eval_revisit_rate 0.25`, `greedy_modal_action_frac 1.0`, `mean_train_entropy 0.5422023858095053`
(depth 1, seed 0, 600 episodes, `tag=exp030_concept`). **Use these to check any future refactor.**

## 3. THE JOB NEXT

Full reasoning in the vault note. In priority order:

1. **Re-ask the EXP-030 memory question.** That null was measured on a 2.2% policy that had learned
   nothing, which EXP-033 and EXP-035 together now make explicit. There is finally a policy at 50%.
   "Does episodic memory reduce cycling and improve solving?" is a live question for the first time and
   all the machinery exists. **This is also the only item that speaks to the project's actual thesis
   rather than around it.**
2. **Tune the curriculum** before buying more episodes. Still fixed `(1,2,3)`, equal thirds, no adaptive
   advancement. Free compared with another 3x budget step, which now costs 5.5 h per run.
3. **Decompose the residual variance** with `encoder_seed` x `train_seed`, now possible. Hold
   `split_seed` FIXED, do not cross it: EXP-034 scored every seed on a different 30-state test set,
   worth about 1% of variance and pure noise with no upside.
4. **Then the encoder.** EXP-033 Finding 1 stands: the encoder discards information a linear probe
   recovers from raw facelets. The probe gives a measured way to check whether pretraining raised it.

**Do not pursue width.** Refuted in EXP-033 against a bar set before the numbers existed.

## 4. Things that will bite

**Seeded runs are NOT reproducible across platforms.** Laptop (Windows, Python 3.13) and VPS (Linux,
3.10) diverge on identical seeds; byte-identity holds *within* a machine only. EXP-032's 24/24 baseline
audit passed only because both sides ran on the laptop. The playbook's "seeded runs are byte-identical"
is about worker scheduling, not machines. **Any cross-machine comparison must account for this.**

**Launcher discipline** (learned expensively, all three failure modes documented in
`launch3.ps1`/`launch032.ps1` headers on the laptop):
- No `Start-Process` detaching. Windows OpenSSH tears down the session job object.
- No `$ErrorActionPreference='Stop'` with `2>&1`. PowerShell escalates a native command's first stderr
  line into a terminating error, and `reinforce.py:188` warns on the first episode. This killed a run
  and orphaned 16 workers that looked healthy for 35 minutes.
- Plain file redirection, no pipeline. Orphans inherit the stdout handle and block a `Tee-Object` reader
  forever.

**Monitoring: five separate bugs this weekend, all the same shape** - a check that cannot distinguish
the states it exists to separate. Process count (alive vs orphaned), absolute file age (fresh vs
leftover), `age < -1` (producing vs never-started), a probe pointed at a stale directory, a 1-hour grace
on a run whose first record takes 5 hours. **Before trusting a health check, ask what it would report
about a run that is perfectly healthy, and about one that is dead.** The repo's test-strength rule is
written about assertions but generalises exactly.

## 5. Correction logged this weekend

**0.481 was called an "oracle ceiling" and that was wrong.** EXP-035 has 7/12 seeds exceeding it, up to
0.633. The probe was optimised for per-step move-optimality while the task rewards solving, and a
9-step budget on a 3-move scramble leaves slack. Corrected in EXP-033, EXP-034 and EXP-035, not only
the newest file. EXP-033's actual claim is unaffected.

## 6. Open debt

Carried forward, still open:

1. Nothing in the dashboard cube work has been seen rendering in a browser.
2. `cubeNet.ts` uses row-major on all six faces; B and D are probably mis-oriented (cosmetic).
3. Move correctness and net geometry pinned independently, composition untested.
4. DLB corner highlight has no test pinning its placement.
5. Shuffle-control dilution (`unshuffled_frac` 0.321 at depth 1) if a shuffle control is used again.
6. Live trace streaming during training is a spec, not built.
7. `test_random_arm_scores_above_zero_but_well_below_one` at `tests/training/test_cube_baseline.py:70`
   asserts `0.0 <= x <= 1.0`, which cannot fail.
8. `requirements.txt` omits the `[server]` extra, so a fresh checkout fails `tests/server/` collection.

New:

9. EXP-025 has `run.py` and `aggregate.py` but no committed `RESULTS.md`. The collapse-fix record lives
   only in ADR 0001 Amendment 2. **Cite the ADR, not "EXP-025".**
10. Nothing past depth 3 has been tested. Shells grow fast: 534 at depth 4, 2,256 at depth 5.

## 7. Pointers

- Standing knowledge: `CLAUDE.md`
- **Strategy / gap analysis: vault `Neuromorphic Development/road-to-a-solved-cube.md`**
- Week log: vault `Weekly Notes/week-17-memory-engagement.md`
- Remote runs: `docs/playbooks/remote-experiment-runs.md` (see the 07-31 handoff section 5 for its
  two errors)
- Results: `experiments/03{0,1,2,3,4,5}_*/RESULTS.md`
- The collapse fix as actually established: `docs/adr/0001-multi-region-training-strategy.md`
- Previous handoff: `docs/handoffs/SESSION-HANDOFF-2026-07-31.md` (superseded; keep for sections 1 and 5)
