# EXP-077: a 5-move beam search with the committed wide judges

**Written 2026-10-14, week 28.** EXP-076 (`experiments/076_deeper_tree/RESULTS.md`) confirmed that
a full 4-move tree with EXP-075's 12 committed X judges beats the 3-move tree everywhere: depth 11
0.1950 vs 0.1200, depth 9 0.4004 vs 0.2571, 12 of 12 seeds each. A 4-move goal check alone changed
no cell, so the gain is the judge scoring leaves nearer the goal. A full 5-move tree scores 7776
leaves per move, about 8x the 4-move tree's cost (EXP-076 spec section 2). Michael chose to run
both (relayed 2026-10-14): this cheaper deeper search first, then the full 5-move tree (EXP-078,
`docs/superpowers/specs/2026-10-14-exp078-full-five-move-tree-design.md`). Nothing is retrained.
Roadmap: `docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md`.

## 1. The question

Can a search reach five moves deep at about the judge budget of the full 4-move tree, and does
that depth pay off the way EXP-076's did? A beam keeps only the B most promising nodes at each
level, scored by the judge, and expands only those. Its cost grows linearly with depth instead of
as 6^k, which is what any search past five moves will need. Its risk is the converse of its
saving: a good line whose early nodes look poor to the judge is pruned before its leaves are
seen, and EXP-075's Gate R showed the judge is weakest far from the goal, which is exactly where
a beam's early levels sit.

## 2. The beam (fixed here, not from the pilot)

`beam_move(state, *, width, depth, critic, agent, visited, imag_generator, info)`:

1. The frontier starts as the root alone, and a `seen` set holds the root. For level l = 1 to
   `depth`:
   - expand every frontier node in frontier order, moves 0 to 5 in order;
   - drop a child if it is in `visited` (the real path, when no-revisit is on) or in `seen`; add
     every kept child to `seen` at once, so the first occurrence wins. **Each state is scored at
     most once, at the shallowest level the beam reaches it**, which also rules out any path that
     returns to its own ancestors;
   - if no child survives, stop and use the last non-empty frontier;
   - score the survivors with the judge in ONE batched `imagined_values` call;
   - the new frontier is the top `width` survivors by judge value (the critic returns -J, higher
     is better, as in mode C), ties to the earlier child.
2. Return the first move of the best node of the final frontier.
3. If level 1 has no survivor, fall back to `_greedy_unvisited`, set `info["fallback"]`, exactly
   as mode C does.

The goal check runs first, as in every judged arm: `solving_prefix(state, k_goal)`, here
`k_goal = 5`. If it fires, its first move is played and the beam is not run.

*Revised 2026-10-14 while planning, before any code or number existed: the pruning rule was
"drop a child equal to an ancestor on its own path, or repeated within its level", under which a
full-width beam can miss a leaf a simple path reaches (U2 is reached by U U and by U' U'). The
shallowest-level rule above replaces it and has an independent reference (a BFS).*

**Width B = 64, depth 5.** Judge evaluations per real move, at most: 6 + 36 + 216 + 384 +
384 = **1026** (the shallowest-level rule only lowers this; on the 2x2 only 27 distinct states
are 2 moves from any state, not 36), against the full 4-move tree's 1296 (and the full 5-move tree's 7776). Pruning
only lowers it. So the beam reaches one level deeper on about 79% of J4V-X's leaf budget. It makes
five batched judge calls per move instead of one, which the pilot prices.

**Stream discipline (new arms only).** The judge's Poisson encoder draws from the per-step
imagination generator (`imag_seed_for`), now across five calls in level order. That is
deterministic, but it is not the stream of a single batched call, so the beam's leaf scores are
not comparable value-for-value with mode C's. Every existing arm is untouched.

## 3. Arms (every arm uses EXP-075's committed X judge for that seed, read through the 576-unit readout)

| arm | goal check (`k_goal`) | search | source |
|---|---|---|---|
| **J4V-X** | 4 | full 4-move tree (mode C) | EXP-076's committed records, reused under Gate 0(b) |
| **J4VG5-X** | 5 | full 4-move tree (mode C) | new cells |
| **B5-X** | 5 | beam, width 64, depth 5 | new cells |

All three use no-revisit. Everything else is EXP-076 section 3 unchanged: held-out sets, budgets
(`2d + 3`), the evaluation stream, the per-step imagination seeds, and the depth-11 cell. Depths 9
and 11, seeds 0 to 11, paired across arms.

**What each difference measures.**
- J4VG5-X minus J4V-X: the longer goal check alone. Every start is at depth 9 or 11 and a move
  changes the distance by at most 1, so the first time a walk comes within 5 moves of solved it is
  exactly 5 away. Until then the two arms play identical moves (`solving_prefix` tries shorter
  lengths first and every other input is the same); from that move J4VG5-X plays an optimal
  5-move solve. J4V-X needs at least 5 more moves from the same state, so whenever the budget
  truncates J4VG5-X it also truncates J4V-X. **J4VG5-X therefore solves at least as many states
  as J4V-X on every cell** (Gate D). EXP-076 measured the same quantity one level down at exactly
  zero on 24 of 24 cells.
- B5-X minus J4VG5-X: what the beam's judged fifth level (and its pruning) does, with the goal
  check held at 5. This has no by-construction sign.

## 4. Code change

- `src/neuromorphic/training/lookahead.py`: a new mode `"B"` in `choose_move` with `width` and
  `depth` arguments (refused for every other mode), implemented by `beam_move` as section 2.
  Existing modes and defaults are byte-identical, which a unit test pins and Gate 0(b) checks on
  the laptop.
- `evaluate_lookahead` records `judge_evals` (mean judge-scored states per real move that reached
  the search, so goal fires are excluded) for every judged arm. For mode C it is exactly 6^k,
  an instrument check.
- Unit tests, beyond the default-path pin:
  - a full-width beam (width >= every level's size) at depth 4 has, as its final frontier,
    exactly the states at BFS distance 4 from the root in the graph with `visited` removed
    (computed independently by a plain BFS), and picks the best-scoring one, on many states, with
    deterministic test critics;
  - at depth 1 the beam, at any width, plays the same move as mode C at k 1, with and without a
    `visited` set, including the fallback;
  - a narrow beam (width 1) is greedy hill-climbing and is checked against a hand-computed walk;
  - the pruning rules each have a case that fails if the rule is deleted;
  - the fallback sets `info["fallback"]`.
- `experiments/077_beam_search/`: `evaluate.py`, `aggregate.py`, launcher, all on EXP-076's
  pattern. The launcher's peak-commit sampling must read the `python3.13` child, not the venv's
  `python.exe` redirector (EXP-076 section 12).

## 5. Disclosure

The pilot (section 6) sees the first 20 held-out states of seed 0 at depth 11 for both new arms.
**Every primary prints the 10-seed sensitivity line (seeds 0 and 3 dropped)** beside the 12-seed
verdict, as in EXP-075 and EXP-076. If the two disagree in verdict, RESULTS.md leads with the
disagreement.

## 6. Pilot (seed 0, depth 11, first 20 held-out states, into `outputs_pilot/`)

Run J4VG5-X and B5-X with `--limit-states 20`. Measure seconds per state for each arm and peak
committed memory per worker (`PrivateMemorySize64` of the `python3.13` workers).

**A dated amendment then fixes**, before any full cell runs, and is put to Michael with `pm ask`:
- the worker count, keeping total committed memory under 24 GB and at most 6 workers;
- the schedule.

No gate threshold, beam width or depth is set from the pilot.

## 7. Gates

**Gate 0.**
- (a) Determinism: B5-X at depth 9, seed 0, re-run into `outputs_det/`. Identical except `wall_s`
  and `git_commit`.
- (b) Continuity: J4V-X at seed 0, depths 9 and 11, re-run on the laptop through this experiment's
  code, must equal EXP-076's committed records in every outcome field (78/200 and 36/200). This
  licenses reusing J4V-X's records and proves the new mode leaves mode C unchanged.

**Gate D (voiding).** On every one of the 24 (depth, seed) cells:
- J4VG5-X `solved` >= J4V-X `solved`;
- for J4VG5-X and B5-X, `first_fire_len` has no key other than "5";
- J4VG5-X `fired5` >= J4V-X `solved`, since every J4V-X solve passes distance 5, where J4VG5-X
  fires.

All hold by construction (section 3), so any violation is a bug and **voids every claim**. The
beam has no dominance clause.

**Gate 1 (resolution).** A contrast is UNRESOLVED if both arms are below 0.02 or above 0.98
(EXP-070's `gate1_verdict`). J4V-X is 0.1950 and 0.4004, clear of both.

**Gate B (beam instrument, reported).** Mean `judge_evals` per move for B5-X at each depth, which
cannot exceed 1026, and the fraction of beam moves that hit the fallback. A mean above 1026 is a
bug and voids every claim; otherwise reported only.

There is no Gate R: a beam has no fixed leaf set to rank.

## 8. Pre-registered claims (one-sided exact sign-flip, 4096 flips; alpha 0.0167 each, 0.05 / 3)

**Claim 1 (primary): a 5-move beam solves more of the typical random scramble than the full
4-move tree.** mean(B5-X - J4V-X) > 0 at depth 11.

**Claim 2 (primary): the same at depth 9.** mean(B5-X - J4V-X) > 0 at depth 9.

**Claim 3 (primary): the beam's judged depth adds to the longer goal check.**
mean(B5-X - J4VG5-X) > 0 at depth 11.

Verdicts: CONFIRMED, CONFIRMED (SMALL) when the mean difference is under 0.01, REFUTED (mean <= 0),
NOT SIGNIFICANT, UNRESOLVED (Gate 1), VOID (Gate 0, Gate D or Gate B). Each primary also prints
its 10-seed line.

**How a null reads.** These are superiority claims. A beam that merely matches the full 4-move tree
returns NOT SIGNIFICANT or REFUTED, and RESULTS.md says so in those words. "It matches at lower
cost" is **not** a claim of this experiment and may not be read off a null: the cost lines below
are reported patterns only.

**Secondary (patterns, never confirmations):**
- Claim 3 at depth 9.
- J4VG5-X minus J4V-X at both depths, with summed `fired5`. Non-negative by construction; only its
  size is informative.
- Seconds per state and mean `judge_evals` per move, per arm.
- Mean steps to solve and optimality per arm.

**What the primaries can detect.** EXP-076's paired per-seed sd of J4V-X minus J3V-X was 0.0312 at
depth 9 and 0.0322 at depth 11, so the se is about 0.009 at n = 12. At alpha 0.0167 a true
difference of about 0.027 is detectable with 80% power. A beam against a full tree with the same
judge may well pair more tightly; that is not assumed.

## 9. Order

1. Code, tests, review (subagent-driven, one subagent at a time on the VPS).
2. Pilot (section 6) on the laptop.
3. Dated amendment: worker count and schedule. Committed and put to Michael with `pm ask` before
   step 4.
4. `cont`: Gate 0(b), J4V-X seed 0 at depths 9 and 11; stop unless it passes.
5. `eval`: J4VG5-X and B5-X, 48 cells, `-SkipExisting`.
6. `det`: Gate 0(a).
7. Aggregate, RESULTS.md, records tracked with `git add -f`.

Estimated cost: J4VG5-X about J4V-X's 26.6 cell-hours (EXP-076); B5-X between that and twice it,
depending on the five-call overhead. About 55 to 80 cell-hours, 9 to 14 h at 6 workers.

## 10. Not in scope

Other widths or depths (a width or depth sweep would be its own experiment, and choosing one from
this run's numbers would be outcome-dependent); beam search with the policy head; retraining;
the full 5-move tree (EXP-078).
