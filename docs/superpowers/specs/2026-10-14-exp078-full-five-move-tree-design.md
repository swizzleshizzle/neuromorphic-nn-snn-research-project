# EXP-078: the full 5-move tree with the committed wide judges

**Written 2026-10-14, week 28.** EXP-076 confirmed that a full 4-move tree beats the 3-move tree
with the same judges (depth 11 0.1950 vs 0.1200, depth 9 0.4004 vs 0.2571, 12 of 12 seeds). Michael
chose to run both 5-move searches, the cheap one first (EXP-077, a width-64 beam,
`docs/superpowers/specs/2026-10-14-exp077-beam-search-design.md`) and then this one: every 5-move
sequence scored by EXP-075's 12 committed X judges, nothing retrained. Roadmap:
`docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md`.

## 1. The questions

1. Does a fifth judged level pay off again, as the fourth did?
2. What does EXP-077's beam give up by pruning? The full tree is the beam's ceiling at depth 5:
   same judge, same goal check, every leaf seen.

## 2. Cost, and why a cell must survive a restart

EXP-076 section 2 measured one real move on the VPS at 0.49 s for k 4 and **4.06 s for k 5**
(7776 leaves), 8.3x. On the laptop a J4V-X depth-11 cell took 1.35 h (EXP-076 records). If the
judge's share scales by 8.3x, a J5V-X depth-11 cell is roughly 8 to 12 h, and depth 9 about three
quarters of that: **about 200 to 250 cell-hours** for 24 cells. The pilot (section 6) replaces
this estimate.

A cell that long will often be in flight at 03:31, when the laptop restarts for Windows Update
whenever one is pending. Banking per cell, which made every earlier run restart-safe, is not
enough. **Each cell therefore checkpoints within itself** (section 4) and resumes where it
stopped, losing at most the states since its last checkpoint.

## 3. Arms (EXP-075's committed X judge for each seed, 576-unit readout, no-revisit)

| arm | goal check (`k_goal`) | search | source |
|---|---|---|---|
| **J5V-X** | 5 | full 5-move tree (mode C, k 5) | new cells |
| **J4V-X** | 4 | full 4-move tree | EXP-076's committed records, reused under Gate 0(b) |
| **J4VG5-X** | 5 | full 4-move tree | EXP-077's committed records |
| **B5-X** | 5 | beam, width 64, depth 5 | EXP-077's committed records |

Everything else is EXP-076 section 3 unchanged. Depths 9 and 11, seeds 0 to 11, paired across arms.
EXP-077's arms are reused only after EXP-077's own Gates 0 and D have passed; if EXP-077 is VOID,
Claims 3 and 4 below are VOID with it and Claims 1 and 2 stand.

**What each difference measures.**
- J5V-X minus J4V-X: a fifth judged level plus a goal check at 5, together (as EXP-076's Claim 1).
- J5V-X minus J4VG5-X: the fifth judged level alone, with the goal check held at 5.
- J5V-X minus B5-X: what the beam's pruning costs at depth 5.

## 4. Code change

- **No search code changes.** `choose_move` and `tree_levels` are generic in k.
- **Within-cell checkpoint and resume** in `evaluate_lookahead`, off by default:
  `checkpoint=None | path, checkpoint_every=N`. Every N held-out states (and at the end) it writes,
  atomically (write then rename), the index of the next state, every running counter and list the
  record is built from, the histogram, and **the evaluation generator's state**
  (`generator.get_state()`), which the Poisson encoder draws from. On start, an existing checkpoint
  is loaded and the loop continues from its index. Imagination seeds are already per (cell, state,
  step), so they need nothing. The environment's own RNG must be checked in the plan: if
  `reset(options={"state": ...})` draws from it, its state is checkpointed too.
- `wall_s` in a resumed record is the sum over sessions; a `resumes` count is recorded.
- With `checkpoint=None` the code path is byte-identical to today, pinned by a unit test.
- **The resume test (test-strength rule):** a small cell run straight through, and the same cell
  killed after M states and resumed, give records equal in every outcome field, for at least two
  values of M including one not a multiple of N. It must fail if the generator state is not
  restored; mutation-test that by deleting the restore.
- `experiments/078_full_five_move_tree/`: `evaluate.py` (passes the checkpoint path and N = 10),
  `aggregate.py`, launcher on EXP-077's pattern, with peak commit read from the `python3.13`
  children.

## 5. Disclosure

The cost probe in EXP-076 section 2 ran one k-5 move on one depth-11 state of seed 0 and did not
look at its outcome. The pilot sees the first 20 depth-11 states of seed 0. **Every primary prints
the 10-seed sensitivity line (seeds 0 and 3 dropped).** If it disagrees in verdict with the 12-seed
line, RESULTS.md leads with the disagreement.

## 6. Pilot (seed 0, depth 11, first 20 held-out states, into `outputs_pilot/`)

Run J5V-X with `--limit-states 20`, once straight through. Measure seconds per state and **peak
committed memory per worker**: 7776 leaves in one judge batch is six times EXP-076's largest, and
memory is the likely limit on the worker count. The pilot may run on the laptop while EXP-077's
evaluation runs, if EXP-077's amendment leaves room for one more worker under 24 GB.

**A dated amendment then fixes**, before any full cell, and is put to Michael with `pm ask`:
- the worker count (total committed memory under 24 GB, at most 6 workers);
- the schedule, depth-11 cells first, so Claims 1, 3 and 4 bank before Claim 2.

No gate threshold or checkpoint interval is set from the pilot.

## 7. Gates

**Gate 0.**
- (a) Determinism: J5V-X at depth 9, seed 0, re-run into `outputs_det/` **with one forced resume**
  (the run is stopped after 50 states and relaunched). Identical except `wall_s`, `git_commit` and
  `resumes`. This tests determinism and resume on the real regime at once.
- (b) Continuity: J4V-X at seed 0, depths 9 and 11, re-run through this experiment's code with
  checkpointing on, must equal EXP-076's records in every outcome field (78/200 and 36/200).

**Gate D (voiding).** On every (depth, seed) cell, J5V-X `first_fire_len` has no key other than
"5": the first time a walk from depth 9 or 11 comes within 5 moves of solved it is exactly 5 away.
A violation is a bug and voids every claim. There is no dominance clause: J5V-X plays different
moves from J4V-X before distance 5, so neither arm bounds the other by construction.

**Gate 1 (resolution).** As EXP-077: UNRESOLVED if both arms are below 0.02 or above 0.98.

**Gate R at k 5 (reported, not voiding).** As EXP-076 at k 4: per depth, the fraction of held-out
states whose lowest-J 5-move leaf is closer to solved than the root, against chance. One rank cell
per (depth, seed), run after the evaluation cells so it never delays a claim.

## 8. Pre-registered claims (one-sided exact sign-flip, 4096 flips; alpha 0.0125 each, 0.05 / 4)

**Claim 1 (primary): a full 5-move tree solves more of the typical random scramble than the full
4-move tree.** mean(J5V-X - J4V-X) > 0 at depth 11.

**Claim 2 (primary): the same at depth 9.** mean(J5V-X - J4V-X) > 0 at depth 9.

**Claim 3 (primary): the fifth judged level adds to the longer goal check.**
mean(J5V-X - J4VG5-X) > 0 at depth 11.

**Claim 4 (primary): the full tree beats the beam at depth 5.** mean(J5V-X - B5-X) > 0 at
depth 11. A NOT SIGNIFICANT here says the beam's pruning cost is below what this run can detect,
not that it is zero.

Verdicts: CONFIRMED, CONFIRMED (SMALL) under 0.01, REFUTED (mean <= 0), NOT SIGNIFICANT, UNRESOLVED
(Gate 1), VOID (Gate 0 or Gate D; Claims 3 and 4 also if EXP-077 is VOID). Each prints its 10-seed
line.

**Secondary (patterns):** Claims 3 and 4 at depth 9; seconds per state per arm; mean steps and
optimality; Gate R against EXP-076's k 4 rates.

**What the primaries can detect.** EXP-076's paired per-seed sd was 0.031 to 0.032, so the se is
about 0.009 at n = 12. At alpha 0.0125 a true difference of about 0.028 is detectable with 80%
power.

## 9. Order

1. Code, tests, review (subagent-driven, after EXP-077's build).
2. Pilot (section 6).
3. Dated amendment, committed and put to Michael with `pm ask`.
4. `cont`: Gate 0(b). Starts only after EXP-077's evaluation has finished, so the two never share
   the laptop's memory beyond the pilot.
5. `eval`: J5V-X, depth 11 then depth 9, `-SkipExisting`, resuming in-flight cells.
6. `det`: Gate 0(a) with its forced resume.
7. `rank`: Gate R.
8. Aggregate (after EXP-077's RESULTS.md exists), RESULTS.md, records tracked.

## 10. Not in scope

k 6 or deeper as a full tree (another 6x); other judges; retraining; beam widths other than
EXP-077's.

## 11. Amendment 2026-10-15: pilot results, worker count and schedule (before any full cell)

The pilot ran on the laptop (checkout `b756c20`, J5V-X, the first 20 depth-11 held-out states of
seed 0, 1 worker), beside EXP-077's `cont` phase (2 workers) as section 6 allows. No full cell had
run.

| arm | seconds per state | depth-11 cell (200 states) | judge evals per move | peak commit |
|---|---|---|---|---|
| J5V-X | 115.0 | 6.39 h | 7776 | 2785 MB |

- `judge_evals` is exactly 6^5, so every move scored the full tree. `first_fire_len` is
  `{"5": 9}`, the shape Gate D requires.
- Against EXP-076's J4V-X depth-11 cell on the laptop (1.35 h), a J5V-X cell is 4.7x, below the
  8.3x of section 2. The pilot shared the laptop with two EXP-077 workers, so if anything it
  overstates the cost.
- Projected evaluation: **125.5 cell-hours** for the 24 J5V-X cells (76.7 at depth 11, 48.8 at
  depth 9 by EXP-076's depth-9 over depth-11 wall ratio, 0.637). Section 2's 200 to 250 is
  superseded.
- **Workers: 6.** 6 x 2.8 GB is 16.7 GB committed, under the 24 GB budget of section 6.
- **Schedule:** `chain078.ps1`, started only after EXP-077's eval has finished: cont (Gate 0(b),
  2 workers, about 1.4 h), eval (depth 11 first, 12 cells in two waves, about 12.8 h, so Claims 1,
  3 and 4 bank first; then depth 9, about 8.1 h), det (J5V-X d9 s0 with the forced resume, about
  4.1 h), rank (Gate R, last, not measured by the pilot). About 27 h before rank, so the run will
  cross at least one 03:31 restart; within-cell checkpoints cover it, and the chain is relaunched
  with `-From` at the phase in flight.
- Disclosed, decides nothing: on the same 20 states J5V-X solved 5, against B5-X 3 and J4VG5-X 2
  in EXP-077's pilot. Seed 0 is outside the 10-seed line.
- **Correction 2026-10-15 (before any full cell, time estimate only):** EXP-077's full run measured
  6-worker contention: its depth-11 cells took 1.4x to 1.8x their 2-worker pilot time. Applying that
  factor, the schedule above is **about 40 to 48 h**, with depth 11 taking about 19 to 23 h. Worker
  count, order and memory are unchanged.
