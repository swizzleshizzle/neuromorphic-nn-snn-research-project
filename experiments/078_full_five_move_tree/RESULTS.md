# EXP-078 results: the full 5-move tree with the committed wide judges

**Claim 4 splits: CONFIRMED on 12 seeds, NOT SIGNIFICANT on the 10-seed line.** Spec section 5
requires this to lead. The full 5-move tree (J5V-X) beats EXP-077's width-64 beam (B5-X) at depth
11 by +0.0379 (0.1854 vs 0.1475), winning on 10 of 12 seeds, p 0.0039. With seeds 0 and 3 dropped
the difference is +0.0300 and still positive on 8 of 10 seeds, but p is 0.0156 on the 1024-flip
grid, above alpha 0.0125. The direction agrees on both lines; the 10-seed line misses the bar. The
two dropped seeds are two of the largest wins (+15 and +16 solved).

**Claim 2 CONFIRMED, Claims 1 and 3 REFUTED, every gate passed.** At depth 9 the full 5-move tree
solves **55.6%** against **40.0%** for the full 4-move tree (J4V-X), winning on 12 of 12 seeds. At
depth 11, the typical random scramble, it solves **18.5%** against **19.5%**. That is REFUTED by
the pre-registered rule (mean <= 0), but the difference is -0.0096, about a third of what the run
could detect (about 0.028). The honest reading is **no detectable gain from a fifth judged level
at depth 11**, not a measured loss.

Spec: `docs/superpowers/specs/2026-10-14-exp078-full-five-move-tree-design.md` (section 11 is the
dated pilot amendment). Plan: `docs/superpowers/plans/2026-10-14-exp078-full-five-move-tree.md`.

## Primaries (spec section 8)

One-sided exact sign-flip over all 4096 (12 seeds) or 1024 (10 seeds) flips, alpha 0.0125 each
(0.05 / 4). The 10-seed line drops seeds 0 and 3.

| claim | 12 seeds | 10 seeds (0 and 3 dropped) |
|---|---|---|
| 1. J5V-X > J4V-X, depth 11 | -0.0096, J5V-X wins 4, loses 7, ties 1, p 0.8750, **REFUTED** | -0.0160, p 0.9551, **REFUTED** |
| 2. J5V-X > J4V-X, depth 9 | +0.1554, wins 12 of 12, p 0.0002, **CONFIRMED** | +0.1460, p 0.0010, **CONFIRMED** |
| 3. J5V-X > J4VG5-X, depth 11 | -0.0096, p 0.8750, **REFUTED** | -0.0160, p 0.9551, **REFUTED** |
| 4. J5V-X > B5-X, depth 11 | +0.0379, wins 10, loses 2, p 0.0039, **CONFIRMED** | +0.0300, p 0.0156, **NOT SIGNIFICANT** |

Claim 3 is identical to Claim 1 because J4VG5-X equals J4V-X on every cell (EXP-077's
decomposition), so its verdict adds nothing of its own.

## Gates

| gate | result |
|---|---|
| 0(a) determinism with one forced resume: J5V-X d9 seed 0 into `outputs_det/`, stopped at 50 states and resumed | **PASS**: every outcome field equal; the det record has `resumes` 1 |
| 0(b) continuity: J4V-X d9 and d11 seed 0 re-run with checkpointing on, against EXP-076's records | **PASS** (78/200 and 36/200) |
| D first-fire instrument (voiding) | **PASS** on all 24 cells: `first_fire_len` only "5" |
| EXP-077 validity (Claims 3 and 4 depend on it) | **PASS** |
| 1 floor 0.02 and ceiling 0.98 | no claim cell outside it |

## Mean held-out success (200 states per cell, 12 seeds)

| depth | J5V-X | J4V-X | J4VG5-X | B5-X |
|---|---|---|---|---|
| 9 | 0.5558 | 0.4004 | 0.4004 | **0.5571** |
| 11 | 0.1854 | **0.1950** | **0.1950** | 0.1475 |

Patterns, not claims:

| depth | arm | mean steps to solve | optimality | judge evals per move | seconds per state |
|---|---|---|---|---|---|
| 9 | J5V-X | 9.51 | 0.947 | 7776 | 89.6 |
| 9 | B5-X | 9.45 | 0.953 | 719 | 10.7 |
| 9 | J4V-X | 10.75 | 0.839 | 1296 | 15.5 |
| 11 | J5V-X | 14.18 | 0.777 | 7776 | 168.4 |
| 11 | B5-X | 13.22 | 0.834 | 717 | 20.2 |
| 11 | J4V-X | 15.84 | 0.696 | 1296 | 24.3 |

Secondary: at depth 9 the full tree and the beam are level (J5V-X - B5-X -0.0012, 6 wins, 5
losses, 1 tie, p 0.5542). So the beam's pruning costs nothing measurable at depth 9 and 0.038 at
depth 11. The seconds per state of J4V-X and B5-X come from earlier runs under different
contention.

## Per seed (success rate; J4VG5-X equals J4V-X on every row)

| seed | d9 J4V-X | d9 B5-X | d9 J5V-X | d11 J4V-X | d11 B5-X | d11 J5V-X | d11 J5V-X fired5 |
|---|---|---|---|---|---|---|---|
| 0 | 0.390 | 0.500 | 0.595 | 0.180 | 0.135 | 0.210 | 48 |
| 1 | 0.460 | 0.645 | 0.625 | 0.210 | 0.155 | 0.200 | 42 |
| 2 | 0.335 | 0.420 | 0.425 | 0.165 | 0.100 | 0.155 | 34 |
| 3 | 0.365 | 0.545 | 0.565 | 0.215 | 0.150 | 0.230 | 50 |
| 4 | 0.455 | 0.645 | 0.655 | 0.175 | 0.160 | 0.190 | 39 |
| 5 | 0.440 | 0.535 | 0.515 | 0.195 | 0.095 | 0.145 | 33 |
| 6 | 0.310 | 0.505 | 0.460 | 0.185 | 0.170 | 0.140 | 30 |
| 7 | 0.400 | 0.610 | 0.610 | 0.250 | 0.160 | 0.235 | 49 |
| 8 | 0.410 | 0.595 | 0.520 | 0.190 | 0.165 | 0.140 | 31 |
| 9 | 0.415 | 0.550 | 0.560 | 0.170 | 0.185 | 0.200 | 43 |
| 10 | 0.430 | 0.565 | 0.595 | 0.230 | 0.155 | 0.205 | 41 |
| 11 | 0.395 | 0.570 | 0.545 | 0.175 | 0.140 | 0.175 | 38 |

## Gate R at k 5 (reported, not voiding)

The fraction of held-out states whose lowest-J leaf is closer to solved than the root, against the
fraction of all leaves that are. Each k is compared with its own chance rate.

| depth | k 5 hit | k 5 chance | k 5 over chance | k 4 hit (EXP-076) | k 4 chance | k 4 over chance |
|---|---|---|---|---|---|---|
| 9 | 0.5021 | 0.1382 | **+0.364** | 0.2308 | 0.0478 | +0.183 |
| 11 | 0.5312 | 0.5395 | **-0.008** | 0.2838 | 0.1628 | +0.121 |

**Read the instrument's floor first.** From depth 11 a 5-move leaf can only sit at an even
distance, so "closer than the root" means distance 10 or less, and 54% of all leaves already
qualify. The criterion is binary: it cannot tell a leaf at 6 from one at 10. So at depth 11 and
k 5 it has little headroom and coarse resolution. With that said, the judge's lowest-J leaf is no
more often closer than a random leaf, where at k 4 it was 0.12 above chance and at depth 9 k 5 it
is 0.36 above. This is a pattern consistent with the roadmap's account (the judge ranks
reliably near the goal and poorly far from it). It is not a mechanism measurement, and nothing
here tests it.

## What it means

- **A fifth judged level pays at depth 9 and not at depth 11.** The fourth level paid at both
  (EXP-076). At depth 9 both 5-move searches add about 0.155 over the 4-move tree. At depth 11
  neither does: the full tree is level with the 4-move tree, and the beam is below it.
- **Pruning costs something only where the judge is weak.** The beam matches the full tree at
  depth 9 and falls 0.038 below it at depth 11 (12-seed CONFIRMED, 10-seed NOT SIGNIFICANT).
- **At depth 11 more search with the same judge has stopped helping.** Together with Gate R, this
  points at the judge's ranking far from the goal rather than at the search. EXP-079 (spec
  `docs/superpowers/specs/2026-10-18-exp079-judge-trained-where-scrambles-live-design.md`) tests
  one lever on that judge.

## Disclosures

- **Claim 4's 10-seed disagreement**, which leads above. The verdict label is the 12-seed one, as
  pre-registered. The 10-seed line is a miss of the bar at a coarser resolution, not a reversal.
- **The pilot was on the full tree's side and decided nothing.** On the first 20 depth-11 states of
  seed 0, J5V-X solved 5, B5-X 3 and J4VG5-X 2 (spec section 11). The full seed-0 depth-11 cells
  are 42, 27 and 36 of 200.
- **Gate 0(b) compares EXP-071's `OUTCOME_FIELDS`**, which omit `fallback_frac`, as in EXP-074 to
  EXP-077.
- **`wall_s` is the evaluation loop's time only**, not process start-up or judge loading.
- **The 24 rank records carry no `git_commit` field**; the rank records never wrote one. Every
  evaluation, continuity and det record says `eb4b3eb`, the laptop's checkout when the chain
  started. `git diff 007521f eb4b3eb` is empty under `src/`, `tests/` and this experiment's
  folder, so nothing changed since the amendment.
- **No evaluation cell resumed.** `resumes` is 0 on all 24 evaluation cells. The run crossed two
  03:31 windows with no restart. The checkpoint path was exercised only by Gate 0(a)'s forced
  resume (`launch078.ps1` passes `--stop-after 50` to the first det job, and checks that a
  checkpoint exists and no record does before the resume job starts).
- **Rank used more memory than evaluation.** Peak commit was 3761 MB per rank worker against 2844
  per evaluation worker. At 6 workers that is 22.6 GB, under the 24 GB budget, but the
  amendment's 16.7 GB was sized for evaluation only.
- **Six workers ran slower than the pilot.** Depth-11 cells took 8.5 to 10.2 h against the
  pilot's 6.39 h, and depth-9 cells 3.9 to 6.6 h. The evaluation took 172.0 cell-hours against
  the amendment's 125.5.

## Cost and provenance

- Machine: SwizzlesDuo (laptop), checkout `eb4b3eb`, run by
  `experiments/078_full_five_move_tree/chain078.ps1 -From cont` (cont, Gate 0(b), eval, det,
  rank). Started 2026-10-16 23:17 UTC, finished 2026-10-18 08:38 UTC: 33.4 h of wall time,
  against the amendment's 27 h before rank and the later 40 to 48 h correction.
- Evaluation: 112.3 cell-hours at depth 11 and 59.7 at depth 9, 6 workers. Continuity: 1.75.
  Determinism: 1.9, 1 worker. Rank: 7.4.
- J4V-X is EXP-076's committed records, re-run only at seed 0 for Gate 0(b). J4VG5-X and B5-X are
  EXP-077's committed records.
- Records: `outputs/` (2 continuity, 24 evaluation, 24 rank), `outputs_det/` (1),
  `outputs_pilot/` (1), all tracked.

Regenerate the verdicts from the tracked records:

```bash
.venv/bin/python experiments/078_full_five_move_tree/aggregate.py
```
