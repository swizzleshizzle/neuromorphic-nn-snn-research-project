# EXP-081 results: was the 4000-update budget the limit?

**Verdict: BUDGET-LIMITED, PAYS. Every gate passed, both claims CONFIRMED, and the search pays at
depth 11.** EXP-080 gave a judge on X's spiking encoder the true BFS distance as its target and
found it barely beat X far from the goal at 4000 updates (+0.0228, p 0.0547), and read
ENCODER-LIMITED. Continued to 16,000 updates on the same seeds, the same judge clears X by
**+0.0678** in the far band (10 of 12 seeds, p 0.0012), and as the search's judge at depth 11 it
solves **0.4088 against J4V-X's 0.1950** (ahead on 12 of 12 seeds, p 0.0002).

**This is a ceiling, not a solver result.** O-Xt is trained on true distance-to-solved, which the
repo admits only as an instrument (spec section 1: "an instrument that prices what X's encoder can
carry, never a candidate system"). The 0.41 is what search over X's encoder reaches when its judge
is given the answer key and 4x EXP-080's budget. It says the encoder can carry a far-out ordering
that doubles depth-11 solves; it does not say the system solves two cubes in five.

| | far band (9 to 11) | minus X | depth 11 solved (J4V) |
|---|---|---|---|
| X (EXP-075 judge, EXP-076 search) | 0.1195 | | 0.1950 |
| O-Xt at 4000 (EXP-080) | 0.1423 | +0.0228 | not run |
| O-Xt at 8000 | 0.1606 | +0.0411 | not run |
| **O-Xt at 16,000** | **0.1873** | **+0.0678** | **0.4088** |
| O-raw at 4000 (EXP-080, raw facelets) | 0.2973 | +0.1778 | not run |

Spec: `docs/superpowers/specs/2026-10-23-exp081-oxt-long-budget-design.md`. Plan:
`docs/superpowers/plans/2026-10-23-exp081-oxt-long-budget.md`.

## Gates

| gate | result |
|---|---|
| C (continuity: the 16,000-block mint's first 4000 blocks, probe and leaves byte-identical to EXP-080's; EXP-080's checkpoint at update 4000 re-measures to its published Gate L on the new mint; every `opt.pt` state at step 4000) | **PASS** on 12 of 12. Start margins equal EXP-080's (seed 0: 0.27037) |
| T (trained: mean overall margin above 0.13564) | **PASS**: 0.3356 at 16,000 (0.2499 at 4000) |

Every leg record resumed from the right point (8000 leg from 4000, 16,000 leg from 8000), carries
`continued_from` EXP-080, trained on the mint Gate C certified, and matches X's Gate L n per
distance. Encoder drift (from X's encoder): 19.2 to 20.4 at 8000, 28.1 to 29.9 at 16,000
(EXP-080's 4000: 12.7 to 13.5).

## Primaries (spec sections 6 and 7)

One-sided exact sign-flip over all 4096 flips, alpha 0.025. Claim 1 is EXP-080's Claim 1 bar
unchanged (CONFIRMED also needs a mean of at least 0.05, otherwise SMALL); Claim 2 needs a
positive mean only.

| claim | diff | ahead | p | verdict |
|---|---|---|---|---|
| 1. O-Xt16k > X, far band | +0.0678 | 10 of 12 | 0.0012 | **CONFIRMED** |
| 2. O-Xt16k > O-Xt4k, far band | +0.0450 | 11 of 12 | 0.0007 | **CONFIRMED** |

## Search (spec section 9)

EXP-080's search cell unchanged (J4V: mode C, k 4, goal check at 4, no-revisit, depth 11,
EXP-075's held-out sets and streams, 200 states per seed) with O-Xt16k as the judge, paired with
EXP-076's J4V-X records, read by EXP-080's `search_reading` (alpha 0.05, minimum 0.03, Gate 1
floor and ceiling).

| | J4V-O-Xt16k | J4V-X | diff | ahead | p | reading |
|---|---|---|---|---|---|---|
| depth 11 success | 0.4088 | 0.1950 | **+0.2137** | 12 of 12 | 0.0002 | **PAYS** |

| secondary (mean over seeds) | J4V-O-Xt16k | J4V-X |
|---|---|---|
| optimality | 0.7562 | 0.6962 |
| mean steps | 14.56 | 15.84 |
| goal check fired (fraction of steps) | 0.081 | 0.035 |
| revisit rate | 0.0 | 0.0 |
| greedy modal action fraction | 0.298 | 0.296 |

## Per seed

Far band is margin (hit rate minus chance) at probe distances 9 to 11, 50 probe states per
distance.

| seed | X far | u4000 | u8000 | u16000 | u16000 - X | X overall | u16000 overall | d11 O-Xt16k | d11 X |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1745 | 0.2145 | 0.2145 | 0.2545 | +0.0800 | 0.3024 | 0.3664 | 79/200 | 36/200 |
| 1 | 0.1110 | 0.1510 | 0.1710 | 0.2044 | +0.0933 | 0.2597 | 0.3277 | 77/200 | 42/200 |
| 2 | 0.0851 | 0.1718 | 0.1718 | 0.2118 | +0.1267 | 0.2005 | 0.3525 | 74/200 | 33/200 |
| 3 | 0.0483 | 0.1217 | 0.1283 | 0.1550 | +0.1067 | 0.2399 | 0.3079 | 86/200 | 43/200 |
| 4 | 0.1793 | 0.1659 | 0.2326 | 0.2526 | +0.0733 | 0.3017 | 0.3777 | 96/200 | 35/200 |
| 5 | 0.1108 | 0.1441 | 0.1375 | 0.2241 | +0.1133 | 0.2649 | 0.3649 | 88/200 | 39/200 |
| 6 | 0.1087 | 0.0887 | 0.1687 | 0.1754 | +0.0667 | 0.2745 | 0.3225 | 72/200 | 37/200 |
| 7 | 0.1552 | 0.1419 | 0.1352 | 0.1486 | -0.0067 | 0.3026 | 0.3346 | 79/200 | 50/200 |
| 8 | 0.0750 | 0.1617 | 0.1617 | 0.1884 | +0.1133 | 0.2597 | 0.3437 | 78/200 | 38/200 |
| 9 | 0.1109 | 0.1242 | 0.1509 | 0.1042 | -0.0067 | 0.2789 | 0.2869 | 83/200 | 34/200 |
| 10 | 0.1563 | 0.1363 | 0.1563 | 0.1963 | +0.0400 | 0.2881 | 0.3361 | 84/200 | 46/200 |
| 11 | 0.1191 | 0.0857 | 0.0991 | 0.1324 | +0.0133 | 0.2822 | 0.3062 | 85/200 | 35/200 |

The search gain is more uniform than the far-band gain. Seeds 7 and 9 sit level with X on the
probe and still solve 29 and 49 more depth-11 cubes than J4V-X.

## Reported, not gating (spec section 11)

| quantity | u4000 | u8000 | u16000 | X |
|---|---|---|---|---|
| margin at distance 7 | 0.4964 | 0.5714 | 0.6514 | **0.6914** |
| margin at distance 8 | 0.3263 | 0.4047 | 0.4647 | 0.3063 |
| margin at distance 9 | 0.1640 | 0.2107 | 0.2590 | 0.1524 |
| margin at distance 10 | 0.1629 | 0.1445 | 0.1712 | 0.1245 |
| margin at distance 11 | 0.1000 | 0.1267 | 0.1317 | 0.0817 |
| overall margin (7 to 11) | 0.2499 | 0.2916 | 0.3356 | 0.2713 |
| probe spearman_7_11 | 0.3315 | 0.3652 | 0.3955 | 0.2983 (EXP-079) |
| probe mean J at distance 9 / 10 / 11 | 9.70 / 9.92 / 10.06 | 9.69 / 9.95 / 10.07 | 9.75 / 10.01 / 10.13 | |
| training loss at the end (MSE, moves squared) | 2.94 | 2.58 | 2.32 | |

- **The budget was not exhausted.** Loss fell 2.94, 2.58, 2.32 across the three points, and the
  far band rose by about the same amount in the 8000 to 16,000 leg (+0.027) as in the 4000 to
  8000 leg (+0.018). This is not a plateau; a longer run is the obvious follow-on, and this
  experiment does not measure where it stops.
- **Budget closed part of the gap to raw facelets, not all of it.** O-Xt16k's far band (0.1873)
  is still 0.11 under O-raw's at a quarter of the updates (0.2973 at 4000), and at distance 7 it
  is still below X (0.6514 against 0.6914). EXP-080's encoder gap is smaller than it read, and it
  is still there.
- **The near-goal deficit EXP-080 reported is closing with budget**: distance-7 margin 0.50, 0.57,
  0.65 against X's 0.69. EXP-080's reading ("X's near-goal order or a calibrated far-out value,
  not both") looks like a budget effect too.

## What it means

- **EXP-080's ENCODER-LIMITED was a 4000-update reading.** At 4x the budget the same arm,
  continued on the same seeds, clears EXP-080's own bar. EXP-080's conclusion that "any judge on
  X's encoder is close to its far-band ceiling already, so further judge objectives on this
  encoder are not worth a run" does not hold. The far-band ordering is reachable through X's
  encoder; it is expensive to reach.
- **Far-band ordering pays in search.** Same tree, same goal check, same states: swapping X's
  judge for one with a better far band moves depth 11 from 0.195 to 0.41, on every seed. Since
  EXP-076, more search (EXP-077, EXP-078) and a retrained judge (EXP-079) had not moved depth 11.
  That links the probe's far band to the outcome the project is after.
- **It reopens EXP-079's question.** EXP-079 changed the judge's training mass (longer walks) and
  moved nothing at its budget. The headroom is now shown to exist, so an experience-based judge
  objective trained at a comparable budget is worth a run; EXP-079 does not rule it out.
- **The next stage is Michael's call.** The admissible question is how a judge trained from
  experience (no BFS labels) reaches this ordering; O-Xt16k itself stays an instrument.

## Disclosures

- **The baseline is EXP-076's published J4V-X records, not a re-evaluation tonight**, as the spec
  pre-registered (section 9). Both arms are evaluated the same way: `evaluate_lookahead` with
  generators freshly seeded from the cell's training seed, on EXP-075's held-out sets, so the
  pairing is on the same states and streams. Re-evaluating J4V-X here was not done; the repo has
  measured that re-evaluation of a tracked checkpoint is portable, and the gap (+0.2137, 12 of 12)
  is far outside the per-seed differences between published and re-evaluated cells seen before.
- **Wave 2 ran its second leg about 50% slower**: seeds 6 to 11 took about 15,450 s for 8000 to
  16,000 against about 10,060 s for seeds 0 to 5, for the same work. The cause is unknown; it is a
  wall-time observation only. Every record passed the aggregator's checks, and the laptop did not
  restart (last boot 2026-10-20).
- **Seeds 0, 2 and 8 have the same far band at 4000 and 8000** (0.2145, 0.1718, 0.1617). Checked:
  their per-distance hit rates differ at every point (seed 0 at distance 7: 0.52 then 0.64) and
  sum to the same far-band total by coincidence, and the 8000 snapshot hashes differently from
  EXP-080's checkpoint on all three.
- **The run was continued, not retrained.** Spec section 2 argues that continuing from EXP-080's
  checkpoint with its optimizer state equals a fresh 16,000-update run on the same machine and
  checkout; Gate C checked the inputs that argument rests on. It is not a cross-machine claim.
- **The depth-11 search is 200 held-out states per seed.** The secondaries are reported, not
  tested.

## Provenance

- Machine: SwizzlesDuo (Intel Ultra 9 185H, 22 cores, 31.4 GB), laptop checkout `9d2390a`.
- Date: 2026-10-23 20:15 to 2026-10-24 09:40 laptop time (ET). Mint 35 min (12 seeds, about
  1,000 to 1,060 s each, 2.26 GB committed per process); Gate C 81 s; training 20:52 to 06:42
  (6 workers, two waves); search 06:43 to 09:40 (5,014 to 5,499 s per cell).
- Seeds 0 to 11, the same as EXP-075, EXP-076 and EXP-080.
- Re-aggregated on the VPS from the copied records: output byte-identical to the laptop's
  `verdict081.txt` (bundle SHA256 `da4c4af6...4d7f` on both ends).
- Tracked: the 48 records (`exp081_gatec_s*.json`, `exp081_O-Xt_u8000_s*.json`,
  `exp081_O-Xt_u16000_s*.json`, `exp081_J4V-O-Xt16k_d11_s*.json`) and, per seed,
  `judge_O-Xt_s*/judge.pt` (16,000), `judge_u8000.pt` and `state.json`. Not tracked: the
  16,000-block mints (regenerable) and `opt.pt`.
- Regenerate the verdict from the tracked records:

```bash
PYTHONPATH=src .venv/bin/python experiments/081_oxt_long_budget/aggregate.py
```

- Re-run on the laptop (skips finished work):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments\081_oxt_long_budget\chain081.ps1
```
