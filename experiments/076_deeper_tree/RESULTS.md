# EXP-076 results: a deeper look-ahead tree with the committed wide judges

**All three claims CONFIRMED, on all 12 seeds and on the 10-seed sensitivity line, and every gate
passed.** Searching four moves ahead instead of three, with EXP-075's 12 committed X judges and
nothing retrained, solves **19.5%** of depth-11 held-out scrambles (the typical random scramble)
against **12.0%** for the 3-move tree, and **40.0%** at depth 9 against **25.7%**. J4V-X wins
every seed at both depths. **The gain is the judge's, not the goal check's:** extending only the
goal check to four moves (J3VG4-X) changed the solved count on **none of the 24 cells**.

Spec: `docs/superpowers/specs/2026-10-11-exp076-deeper-tree-design.md` (section 11 is the
amendment from the final code review, section 12 the dated pilot amendment). Plan:
`docs/superpowers/plans/2026-10-11-exp076-deeper-tree.md`.

## Primaries (spec section 8)

One-sided exact sign-flip over all 4096 (12 seeds) or 1024 (10 seeds) flips, alpha 0.0167 each
(0.05 / 3). The 10-seed line drops seeds 0 and 3, the cells disclosed in spec section 5.

| claim | 12 seeds | 10 seeds (0 and 3 dropped) |
|---|---|---|
| 1. J4V-X > J3V-X, depth 11 | +0.0750, 12 of 12, p 0.0002, **CONFIRMED** | +0.0790, p 0.0010, **CONFIRMED** |
| 2. J4V-X > J3V-X, depth 9 | +0.1433, 12 of 12, p 0.0002, **CONFIRMED** | +0.1460, p 0.0010, **CONFIRMED** |
| 3. J4V-X > J3VG4-X, depth 11 | +0.0750, 12 of 12, p 0.0002, **CONFIRMED** | +0.0790, p 0.0010, **CONFIRMED** |

Every difference is far above the 0.01 band, so none is CONFIRMED (SMALL). p 0.0002 is the
smallest value 4096 flips can give.

## Gates

| gate | result |
|---|---|
| 0(a) determinism: J4V-X d9 seed 0 re-run into `outputs_det/` | **PASS**: every field equal except `wall_s` |
| 0(b) continuity: J3V-X d9 and d11 seed 0 re-run here, against EXP-075's records | **PASS** |
| D dominance and first-fire instrument (voiding) | **PASS** on all 24 cells: J3VG4-X >= J3V-X solved; `first_fire_len` only "4" for both new arms; J3VG4-X `fired4` >= J3V-X solved |
| G the 4-move goal check fired (reported) | fired on 629 episodes at depth 9 and 320 at depth 11 (it cannot be unresolved, spec section 11) |
| R at k 4 (reported, not voiding): lowest-J 4-move leaf closer to solved than the root | depth 9 hit 0.2308 vs chance 0.0478; depth 11 hit 0.2838 vs chance 0.1628; both p 0.0002 |
| 1 floor 0.02 and ceiling 0.98 | no claim cell outside it |

## Mean held-out success (200 states per cell, 12 seeds)

| depth | J4V-X | J3VG4-X | J3V-X |
|---|---|---|---|
| 9 | **0.4004** | 0.2571 | 0.2571 |
| 11 | **0.1950** | 0.1200 | 0.1200 |

Mean steps to solve (solved episodes) and optimality, a pattern rather than a claim:

| depth | J4V-X | J3V-X and J3VG4-X |
|---|---|---|
| 9 | 10.75 steps, optimality 0.839 | 11.99, 0.753 |
| 11 | 15.84 steps, optimality 0.696 | 17.57, 0.627 |

The deeper tree solves more and wastes fewer moves on the cubes it solves. With no-revisit on,
revisit rate and fallback fraction are 0.0 for J4V-X at both depths.

## Per seed

| seed | d9 J3V-X | d9 J4V-X | d9 J3VG4-X fired4 | d11 J3V-X | d11 J4V-X | d11 J3VG4-X fired4 |
|---|---|---|---|---|---|---|
| 0 | 0.280 | 0.390 | 59 | 0.150 | 0.180 | 32 |
| 1 | 0.250 | 0.460 | 52 | 0.110 | 0.210 | 26 |
| 2 | 0.215 | 0.335 | 43 | 0.100 | 0.165 | 22 |
| 3 | 0.215 | 0.365 | 44 | 0.135 | 0.215 | 30 |
| 4 | 0.285 | 0.455 | 58 | 0.125 | 0.175 | 28 |
| 5 | 0.315 | 0.440 | 63 | 0.130 | 0.195 | 30 |
| 6 | 0.175 | 0.310 | 35 | 0.100 | 0.185 | 23 |
| 7 | 0.290 | 0.400 | 60 | 0.120 | 0.250 | 26 |
| 8 | 0.295 | 0.410 | 60 | 0.110 | 0.190 | 22 |
| 9 | 0.280 | 0.415 | 57 | 0.140 | 0.170 | 32 |
| 10 | 0.265 | 0.430 | 54 | 0.105 | 0.230 | 23 |
| 11 | 0.220 | 0.395 | 44 | 0.115 | 0.175 | 26 |

J3VG4-X's success rate equals J3V-X's on every row, so it is not repeated.

## What the decomposition says

The spec's confound (section 1) was that raising k extends both the goal check and the judged
tree. The third arm settles it more sharply than the spec expected. **J3VG4-X solved exactly as
many states as J3V-X on all 24 cells**: whenever a 3-move walk came within four moves of solved,
it went on to solve anyway, so a 4-move goal check had nothing to add. Claim 3's difference is
therefore identical to Claim 1's. The whole gain is the judge scoring leaves one move nearer the
goal, which is where spec section 1 predicted it to be more reliable.

`fired4` exceeds solved in some cells (for example d11 seed 1: 26 fires, 22 solves). That is the
step budget, not a fault: a goal check firing at step t with a 4-move solution needs t + 3 <= 2d + 3,
and a fire later than that is truncated before it completes (`evaluate_lookahead`'s loop). The
same episodes fail in J3V-X.

Gate R at k 4 reads lower than EXP-075's k 3 hit rates (depth 9 0.2308 against 0.3271), but
chance is far lower too (0.0478 against 0.1751), because a 4-move tree has 1296 leaves. Relative
to chance, the 4-move leaf ranking is stronger at both depths (depth 11: 0.284 vs 0.163, against
k 3's 0.570 vs 0.521). It is reported, not interpreted further.

## Disclosures

- **The pilot pointed the other way and decided nothing.** On the first 20 depth-11 states of
  seed 0, J4V-X solved 2 and J3VG4-X 4 (spec section 12). The full seed-0 depth-11 cells are 36
  and 30 of 200. Seed 0 is in the 12-seed line and excluded from the 10-seed line; both agree.
- **The launcher's peak-memory log is not an instrument.** It reads about 2 MB for every cell
  because it samples the venv's `python.exe` redirector, not the `python3.13.exe` worker (spec
  section 12). The pilot's probe measured real workers at up to 2.0 GB committed; the full run was
  not probed and nothing failed.
- **Rank `wall_s` includes the unbounded BFS build**, about 67 s per process (spec section 11).
- **Gate 0(b) compares EXP-071's `OUTCOME_FIELDS`**, which omit `fallback_frac`, as in EXP-074
  and EXP-075.

## Cost and provenance

- Machine: SwizzlesDuo (laptop), checkout `79930b2`, run by
  `experiments/076_deeper_tree/chain076.ps1` (`launch076.ps1` phases cont, rank, eval and det).
  Started 2026-10-12 20:05 and done 2026-10-13 02:44 laptop time: **6.6 h of wall time**, 6
  workers.
- Evaluation: 26.6 cell-hours for J4V-X (1.35 h per depth-11 cell) and 7.4 for J3VG4-X, 34.0 in
  all against the pilot's projection of 41.9. Rank: 1.9 cell-hours.
- J3V-X is EXP-075's committed records, not re-run except seed 0 for Gate 0(b).
- Records: `outputs/` (2 continuity, 24 rank, 48 evaluation), `outputs_det/` (1), `outputs_pilot/`
  (3), all tracked.

Regenerate the verdicts from the tracked records:

```bash
.venv/bin/python experiments/076_deeper_tree/aggregate.py --determinism-ok
```

## What it means for the roadmap

A 4-move search with the same judges moves the typical random scramble from about one cube in
eight to about one in five. It searches 6 times the leaves (6^4 against 6^3) and costs about
2.6 times the evaluation time per state (pilot, against J3VG4-X). The judge is better near the goal than far from it, and a deeper tree exploits that
directly. A 5-move tree would cost about 20 h per depth-11 cell on the laptop (spec section 2),
so further depth needs a cheaper search (beam or pruned) rather than a full tree. The next stage is
Michael's call.
