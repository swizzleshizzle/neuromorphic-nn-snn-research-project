# EXP-077 results: a width-64 beam searched five moves deep

**Claim 2 CONFIRMED, Claims 1 and 3 REFUTED, every gate passed, and the 10-seed line agrees on
all three.** With EXP-075's 12 committed X judges and nothing retrained, a width-64 beam searched
five moves deep (B5-X) solves **55.7%** of depth-9 held-out scrambles against **40.0%** for the
full 4-move tree, and wins on **12 of 12** seeds. At depth 11, the typical random scramble, it
solves **14.75%** against **19.5%** and loses on **11 of 12**. The beam helps where the root is
already near the goal and hurts where it is far from it.

Spec: `docs/superpowers/specs/2026-10-14-exp077-beam-search-design.md` (section 11 is the dated
pilot amendment). Plan: `docs/superpowers/plans/2026-10-14-exp077-beam-search.md`.

## Primaries (spec section 8)

One-sided exact sign-flip over all 4096 (12 seeds) or 1024 (10 seeds) flips, alpha 0.0167 each
(0.05 / 3). The 10-seed line drops seeds 0 and 3.

| claim | 12 seeds | 10 seeds (0 and 3 dropped) |
|---|---|---|
| 1. B5-X > J4V-X, depth 11 | -0.0475, B5-X wins 1 of 12, p 0.9998, **REFUTED** | -0.0460, p 0.9990, **REFUTED** |
| 2. B5-X > J4V-X, depth 9 | +0.1567, B5-X wins 12 of 12, p 0.0002, **CONFIRMED** | +0.1590, p 0.0010, **CONFIRMED** |
| 3. B5-X > J4VG5-X, depth 11 | -0.0475, B5-X wins 1 of 12, p 0.9998, **REFUTED** | -0.0460, p 0.9990, **REFUTED** |

The refutations are not nulls. The beam is worse at depth 11, by more than five times the
detectable difference of spec section 8 (about 0.027). p 0.0002 is the smallest value 4096 flips
can give.

## Gates

| gate | result |
|---|---|
| 0(a) determinism: B5-X d9 seed 0 re-run into `outputs_det/` | **PASS**: every field equal except `wall_s` |
| 0(b) continuity: J4V-X d9 and d11 seed 0 re-run here, against EXP-076's records | **PASS** (78/200 and 36/200) |
| D dominance and first-fire instrument (voiding) | **PASS** on all 24 cells: J4VG5-X >= J4V-X solved; `first_fire_len` only "5"; J4VG5-X `fired5` >= J4V-X solved |
| B beam judge evaluations <= 1026 per move (voiding) | **PASS** on every cell: mean 718.8 at depth 9 and 717.4 at depth 11; `fallback_frac` 0.0 at both depths |
| 1 floor 0.02 and ceiling 0.98 | no claim cell outside it |

## Mean held-out success (200 states per cell, 12 seeds)

| depth | B5-X | J4V-X | J4VG5-X |
|---|---|---|---|
| 9 | **0.5571** | 0.4004 | 0.4004 |
| 11 | 0.1475 | **0.1950** | **0.1950** |

Patterns, not claims:

| depth | arm | mean steps to solve | optimality | judge evals per move | seconds per state |
|---|---|---|---|---|---|
| 9 | B5-X | 9.45 | 0.953 | 719 | 10.7 |
| 9 | J4V-X / J4VG5-X | 10.75 | 0.839 | 1296 | 15.5 / 23.8 |
| 11 | B5-X | 13.22 | 0.834 | 717 | 20.2 |
| 11 | J4V-X / J4VG5-X | 15.84 | 0.696 | 1296 | 24.3 / 32.6 |

At both depths the beam solves its cubes in fewer moves and scores fewer leaves per move than the
4-move tree. The J4V-X seconds per state come from EXP-076's run, under different contention, so
the column compares B5-X with J4VG5-X only.

## Per seed (success rate; J4V-X equals J4VG5-X on every row)

| seed | d9 J4V-X | d9 B5-X | d9 J4VG5-X fired5 | d11 J4V-X | d11 B5-X | d11 J4VG5-X fired5 |
|---|---|---|---|---|---|---|
| 0 | 0.390 | 0.500 | 84 | 0.180 | 0.135 | 40 |
| 1 | 0.460 | 0.645 | 97 | 0.210 | 0.155 | 49 |
| 2 | 0.335 | 0.420 | 72 | 0.165 | 0.100 | 36 |
| 3 | 0.365 | 0.545 | 73 | 0.215 | 0.150 | 45 |
| 4 | 0.455 | 0.645 | 91 | 0.175 | 0.160 | 39 |
| 5 | 0.440 | 0.535 | 89 | 0.195 | 0.095 | 44 |
| 6 | 0.310 | 0.505 | 64 | 0.185 | 0.170 | 40 |
| 7 | 0.400 | 0.610 | 85 | 0.250 | 0.160 | 54 |
| 8 | 0.410 | 0.595 | 86 | 0.190 | 0.165 | 44 |
| 9 | 0.415 | 0.550 | 88 | 0.170 | 0.185 | 38 |
| 10 | 0.430 | 0.565 | 87 | 0.230 | 0.155 | 53 |
| 11 | 0.395 | 0.570 | 80 | 0.175 | 0.140 | 36 |

## What the decomposition says

**The 5-move goal check adds nothing on its own, as the 4-move one did not in EXP-076.**
J4VG5-X solved exactly as many states as J4V-X on all 24 cells, with identical mean steps and
optimality. It is not a wiring fault. J4VG5-X's goal check does fire at 5 (`first_fire_len` only
"5"), and it fires on more walks than J4V-X's fires at 4 (depth 11 seed 10: 53 against 48). Its
greedy modal-action fraction differs in the fourth decimal, so it does play different moves. The
extra fires are walks that reach distance 5 too late to finish within the step budget (`2d + 3`).
Every walk that does finish solves the same way under both arms. Claim 3's difference is therefore
identical to Claim 1's.

**So what separates B5-X from J4V-X is the beam itself: a fifth judged level, bought by pruning
to 64 states per level.** At depth 9 the gain is large and on every seed. At depth 11 the loss is
on 11 of 12 seeds. The roadmap's working account (EXP-076: the judge is more reliable near the
goal than far from it) predicts this shape. A beam keeps only the states the judge ranks highest
at each level, so it is only as good as the judge's ranking at the shallow levels. From depth 11
those rankings are made far from the goal, and a pruned-away state cannot be recovered. A full
tree scores every leaf and never prunes. This account is consistent with the data, and **it is
not tested here**. EXP-078's Claim 4 (full 5-move tree against B5-X at depth 11) is the direct
test of what the pruning costs.

## Disclosures

- **The pilot was on the beam's side and decided nothing.** On the first 20 depth-11 states of
  seed 0, B5-X solved 3 and J4VG5-X 2 (spec section 11). The full seed-0 depth-11 cells are 27 and
  36 of 200. Seed 0 is in the 12-seed line and excluded from the 10-seed line; both agree.
- **Gate B is applied per cell, on the cell's mean `judge_evals` per move.** The bound of 1026
  holds per move by construction (`beam_move` scores each state at most once). The record keeps
  the mean, not the maximum.
- **`fallback_frac` is over all moves.** It reads 0.0 everywhere: no beam move ever found its
  first level empty under no-revisit.
- **The eval phase reported a failure it did not have.** The laptop dropped off the tailnet for
  about 2 minutes at 13:55 UTC and did not reboot. The ssh session carrying the chain's stdout
  died with it, and the cells, started with `Start-Process -NoNewWindow`, inherit that stdout.
  Each cell that finished afterwards wrote its record and then failed on its final `print`, so it
  exited 1. `launch077.ps1` counted those exits and the chain stopped before `det`. All 48
  evaluation records were checked complete (200 states each) before going on. `det` was then run
  alone with OS-level redirection (`cmd /c "powershell ... > file 2>&1"`). The determinism gate
  is the check that nothing was lost.
- **Every record says `git_commit` 007521f, though the chain started at b756c20.**
  `git_commit` is read when a record is written. The laptop was pulled to 007521f before the first
  record was written. 007521f only adds EXP-078's amendment files to b756c20 and changes nothing
  under `src/`.
- **Six workers ran slower than the pilot.** Depth-11 cells took 1.8 h (J4VG5-X) and 1.1 h (B5-X)
  against the pilot's 0.99 h and 0.79 h, which ran at 2 workers. The evaluation took 9.8 h of
  wall time against the amendment's 6 to 7 h.
- **Gate 0(b) compares EXP-071's `OUTCOME_FIELDS`**, which omit `fallback_frac`, as in EXP-074
  to EXP-076.

## Cost and provenance

- Machine: SwizzlesDuo (laptop), checkout `b756c20` pulled to `007521f` before any record, run by
  `experiments/077_beam_search/chain077.ps1` (cont, Gate 0(b), eval), then `-From det`. Started
  2026-10-15 01:48 laptop time. Eval finished at 12:36 and det at 13:48. Eval used 6 workers.
- Evaluation: 20.6 cell-hours for B5-X and 37.6 for J4VG5-X, 58.2 in all against the amendment's
  35.0. Continuity: 1.8. Determinism: 0.29 h, run alone.
- J4V-X is EXP-076's committed records, re-run only at seed 0 for Gate 0(b).
- Records: `outputs/` (2 continuity, 48 evaluation), `outputs_det/` (1), `outputs_pilot/` (2),
  all tracked.

Regenerate the verdicts from the tracked records:

```bash
.venv/bin/python experiments/077_beam_search/aggregate.py --det-dir experiments/077_beam_search/outputs_det
```

## What it means for the roadmap

The cheap 5-move search is not a cheaper substitute for the full 4-move tree on the typical random
scramble. It is worse at depth 11, though much better at depth 9. EXP-078, the full 5-move tree,
is already specified and piloted. Its pilot solved 5 of the same 20 states that B5-X solved 3 of.
It is the direct measure of whether depth 11 needs the judge's full ranking or a better judge, and
it runs next once Michael approves its amendment (neuromorphic#21).
