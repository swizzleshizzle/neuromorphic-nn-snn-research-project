# EXP-082 results: is value iteration budget-limited too?

**Verdict: VALUE ITERATION SATURATED AT 4X BUDGET.** Both gates passed. Claim 1 (the primary)
was not confirmed: at depth 11 the search with X's judge continued to 16,000 updates solves
**0.2083 against J4V-X's 0.1950** (+0.0133, ahead on 9 of 12 seeds, p 0.1035). Claim 2 came out
CONFIRMED (SMALL): X16k beats X in the far band by **+0.0278** (8 of 12, p 0.0112), which is
significant but under the 0.05 bar. Spec section 8 sends SMALL to the next row, so with Claim 1
unconfirmed the verdict is the last row. The aggregator did exactly what the spec says.

**What 4x budget did buy is the near band, and it is large.** Margins at distances 7 and 8 rose
from 0.69 to 0.84 and from 0.31 to 0.57. In the search, depth 9 went from **0.4004 to 0.6200**,
ahead on 12 of 12 seeds (p 0.0002). That is the largest depth-9 success rate the repo has
recorded, above EXP-077's 0.5571 (a width-64 beam over 5 moves, a different cell). It is a real
solver number, since X is trained by value iteration with no true distances. It is **reported,
not a claim** (spec section 9), and it does not change the verdict.

| | far band (9 to 11) | minus X | overall margin (7 to 11) | depth 9 solved (J4V) | depth 11 solved (J4V) |
|---|---|---|---|---|---|
| X (EXP-075 judge, EXP-076 search) | 0.1195 | | 0.2713 | 0.4004 | 0.1950 |
| X at 8000 | 0.1318 | +0.0123 | 0.3183 | not run | not run |
| **X16k (X at 16,000)** | **0.1473** | **+0.0278** | **0.3689** | **0.6200** | **0.2083** |
| O-Xt16k (EXP-081, oracle-trained ceiling) | 0.1873 | +0.0678 | 0.3356 | not run | 0.4088 |

Spec: `docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md`. Plan:
`docs/superpowers/plans/2026-10-25-exp082-x-long-budget.md`.

## Gates

| gate | result |
|---|---|
| C (continuity: source `judge.pt` and `state.json` hash equal to the tracked EXP-075 files at update 4000; every Adam state in `opt.pt` at step 4000 with the right shape; `jt.pt` loads; the source judge re-measures to EXP-075's committed `gate_l` exactly) | **PASS** on 12 of 12. Start margins equal EXP-075's (seed 0: 0.30237) |
| T (trained: mean overall margin above 0.13564) | **PASS**: 0.3689 at 16,000 (0.2713 at 4000, 0.3183 at 8000) |

Every 8000 record resumed from 4000 and every 16,000 record from 8000. All carry
`continued_from` EXP-075 update 4000 and EXP-075's `n_probe`. All 60 records (Gate C, both
legs, both search depths) carry the same commit, `4e2c17f`. Encoder drift at 16,000 is 22.0 to
28.0 (mean 24.03).

## Primaries (spec section 7)

One-sided exact sign-flip test over all 4096 flips, alpha 0.025, paired by seed.

| claim | contrast | diff | ahead | p | verdict |
|---|---|---|---|---|---|
| 1 (primary) | J4V-X16k minus J4V-X, depth-11 success | +0.0133 | 9 of 12 | 0.1035 | **not confirmed** (aggregator: REFUTED) |
| 2 (mechanism) | X16k minus X, far band | +0.0278 | 8 of 12 | 0.0112 | **CONFIRMED (SMALL)** |

**Claim 1 bounds the gain. It does not show zero.** Spec section 7 put Claim 1's se at about
0.0124 and said gains under about 0.02 do not resolve. The measured +0.0133 is about 2.7 more
solves per seed out of 200, inside that band. The aggregator prints REFUTED, the label for a
non-confirmation. What the data support is that 4x budget did not buy the 0.03 the spec asked
for at depth 11, and that any true gain there is most likely small.

## Search

EXP-081's search cell unchanged except for the judge: J4V, mode C, k 4, goal check at 4,
no-revisit, EXP-075's held-out sets and streams, 200 scrambles per seed, through
`e75.JudgeCritic`. It is paired with EXP-076's published J4V-X records at each depth.

| | J4V-X16k | J4V-X | diff | ahead | p |
|---|---|---|---|---|---|
| depth 11 success (Claim 1) | 0.2083 | 0.1950 | +0.0133 | 9 of 12 | 0.1035 |
| depth 9 success (reported) | 0.6200 | 0.4004 | **+0.2196** | 12 of 12 | 0.0002 |

| depth 11 secondary (mean over seeds) | J4V-X16k | J4V-X |
|---|---|---|
| optimality | 0.7950 | 0.6962 |
| mean steps | 13.86 | 15.84 |
| goal check fired (fraction of steps) | 0.0376 | 0.0345 |

At depth 11 X16k's solves are shorter and closer to optimal than J4V-X's, though there are
barely more of them. The X16k minus O-Xt16k contrast at depth 11 (descriptive, spec section 9)
is 0.2083 against 0.4088, -0.2004, with X16k ahead on 0 of 12.

## Per seed

Far band is margin (hit rate minus chance) at probe distances 9 to 11. Overall is margin over
distances 7 to 11.

| seed | X far | u8000 far | u16000 far | X overall | u16000 overall | d9 X16k | d9 X | d11 X16k | d11 X |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1745 | 0.1879 | 0.1945 | 0.3024 | 0.4104 | 127/200 | 78/200 | 42/200 | 36/200 |
| 1 | 0.1110 | 0.1377 | 0.1310 | 0.2597 | 0.3557 | 139/200 | 92/200 | 35/200 | 42/200 |
| 2 | 0.0851 | 0.1451 | 0.1785 | 0.2005 | 0.3445 | 106/200 | 67/200 | 38/200 | 33/200 |
| 3 | 0.0483 | 0.0483 | 0.0950 | 0.2399 | 0.3319 | 117/200 | 73/200 | 52/200 | 43/200 |
| 4 | 0.1793 | 0.1726 | 0.1726 | 0.3017 | 0.3897 | 128/200 | 91/200 | 39/200 | 35/200 |
| 5 | 0.1108 | 0.1241 | 0.1108 | 0.2649 | 0.3489 | 130/200 | 88/200 | 43/200 | 39/200 |
| 6 | 0.1087 | 0.1154 | 0.1621 | 0.2745 | 0.3665 | 91/200 | 62/200 | 36/200 | 37/200 |
| 7 | 0.1552 | 0.1552 | 0.1419 | 0.3026 | 0.3746 | 137/200 | 80/200 | 37/200 | 50/200 |
| 8 | 0.0750 | 0.0950 | 0.1084 | 0.2597 | 0.3717 | 126/200 | 82/200 | 40/200 | 38/200 |
| 9 | 0.1109 | 0.1242 | 0.1842 | 0.2789 | 0.4029 | 132/200 | 83/200 | 40/200 | 34/200 |
| 10 | 0.1563 | 0.1430 | 0.1363 | 0.2881 | 0.3561 | 127/200 | 86/200 | 53/200 | 46/200 |
| 11 | 0.1191 | 0.1324 | 0.1524 | 0.2822 | 0.3742 | 128/200 | 79/200 | 45/200 | 35/200 |

Overall margin rose on all 12 seeds, and depth 9 rose on all 12 by 29 to 57 solves. The far band
fell on 3 seeds (4, 7, 10) and held flat on one (5). Depth 11 fell on 3 seeds (1, 6, 7). Seed 7
lost 13 depth-11 solves, the largest single move either way.

## Reported, not gating (spec section 9)

| quantity | X (u4000) | u8000 | u16000 |
|---|---|---|---|
| margin at distance 7 | 0.6914 | 0.7731 | **0.8364** |
| margin at distance 8 | 0.3063 | 0.4230 | **0.5663** |
| margin at distance 9 | 0.1524 | 0.1390 | 0.1790 |
| margin at distance 10 | 0.1245 | 0.1329 | 0.1495 |
| margin at distance 11 | 0.0817 | 0.1234 | 0.1134 |
| overall margin (7 to 11) | 0.2713 | 0.3183 | 0.3689 |
| probe spearman_7_11 | 0.2983 | 0.3178 | 0.3521 |
| probe mean J at distance 9 / 10 / 11 | 7.14 / 7.27 / 7.38 | 7.43 / 7.56 / 7.69 | 7.67 / 7.81 / 7.94 |
| training loss at the end | 0.1974 | 0.1948 | 0.1849 |

- **The gain is concentrated near the goal.** Distance 8 rose by 0.26 and distance 7 by 0.15.
  Distances 9 to 11 rose by 0.025 to 0.032 each. This matches how value iteration moves
  information: each target is the judge's own estimate one look-ahead away, so ordering spreads
  outward from the solved state, and 12,000 more updates pushed it about one distance further.
  The depth-9 search needs good ordering at distances 5 to 9, which is where X16k improved, so
  it pays. Depth 11 starts at distance 11, where X16k barely improved, so it does not.
- **The curve is not flat, so SATURATED is a reading at 16,000 against this bar, not a
  plateau.** Loss fell 0.197, 0.195, 0.185. The far band rose +0.012 over the first leg and
  +0.016 over the second. At that slope the far band would need several more times the budget
  to reach O-Xt16k's 0.1873. This experiment does not measure where it stops.
- **The limit is the objective, as spec section 12 pre-registered.** On the same encoder at the
  same 16,000 updates, the judge trained on true distances reaches far band 0.1873 and depth 11
  0.4088. X16k reaches 0.1473 and 0.2083. The encoder can carry the far-out ordering. Value
  iteration, at this budget, does not extract it.
- **EXP-079's near-goal trade-off did not appear.** Spec section 9 ran depth 9 to catch a judge
  that orders far states better but loses near ones. X16k improved both, and near far more.
- J's scale is the value-iteration judge's own cost estimate, not moves. Its spacing between
  distances (about 0.11 to 0.14 per distance at all three points) did not widen with budget.

## What it means

- **More budget is not the route to depth 11 for this objective.** 4x the updates bought a
  third more overall margin, a near-band gain worth +0.22 at depth 9, and at most a small,
  unresolved gain at depth 11. Since EXP-076, nothing that uses no true distances has moved
  depth 11: not more search (EXP-077, EXP-078), not longer walks (EXP-079), not 4x budget
  (here).
- **The far-band gap between X16k and the oracle ceiling is the objective's.** EXP-081
  established that X's encoder can carry an ordering that doubles depth-11 solves. EXP-082
  shows X's own objective does not reach it in 4x the budget. The open question is how a judge
  trained from experience reaches the far band. Candidates are a target that looks further
  than one look-ahead (multi-step or search-backed), and a curriculum that pushes training
  mass outward as the near band converges.
- **The depth-9 result changes the shallow end of the road.** Depth 9 at 0.62 with a plain
  4-move tree and no oracle is the strongest real number the project has at that depth. It is
  reported, not confirmed, so if the next stage builds on it, it should be pre-registered there.
- **The next stage is Michael's call.**

## Disclosures

- **The baseline is EXP-076's published J4V-X records, not a re-evaluation**, as the spec
  pre-registered (section 3), and as EXP-081 did. Both arms use `evaluate_lookahead` with
  generators freshly seeded from the cell's training seed, on EXP-075's held-out sets, so the
  pairing is on the same states and streams.
- **The run was continued, not retrained.** Spec section 2 argues that this equals a fresh
  16,000-update run on the same machine and checkout. Gate C checked the inputs that argument
  rests on. It is not a cross-machine claim.
- **The search ran slower at 12 workers than EXP-081's did at 6.** Depth-11 cells took 12,080
  to 13,043 s each here, against EXP-081's 5,014 to 5,499 s at 6 workers for the same cell.
  Twelve depth-11 cells in one wave took about 3.6 h. EXP-081 took about 2.9 h in two waves.
  Doubling the workers more than doubled the per-cell time, so the pool bought no throughput
  and the search took 5.5 h against the spec's 4 h estimate. This is the training-pool lesson
  (no gain above 4 to 6 workers) applied to search. Future search phases on this laptop should
  use 6 workers.
- **Training took about 65 h against the spec's 42 h.** The measured rate was about 577
  updates per hour per seed at 4 workers (6.0 to 6.6 s per update in the first leg, 6.2 to 6.9 s in
  the second), not EXP-075's 4.2 s. Wave timing reversed EXP-081's pattern: the second leg took
  about 55,300 s in wave 1, 53,000 s in wave 2 and 49,800 s in wave 3.
- **There was no restart.** The laptop last booted 2026-10-20, and the chain ran once from
  start to verdict, so every record's `wall_s` and `loss_last` cover its whole leg.
- **Deferred minors from the build's final review**, none of which bit on this run:
  1. The checkpoint writer saves its four files in sequence, not atomically. A restart inside
     that window could tear `state.json`. No restart happened.
  2. A record's `probes` list has two extra entries (at updates 3999 and 7999) and no update
     labels, so plot the curve by update, not by position.
  3. `wall_s` and `loss_last` cover only the last session after a restart. With no restart,
     they cover the whole leg here.
  4. The aggregator's single-commit check covers only the 16,000 records. Checked by hand: all
     60 records carry `4e2c17f`.
  5. The reported block gives distance-7 and distance-8 margins as averages only, and the
     training curve as three points.
  6. The test that a continued run equals an uninterrupted one uses 1-update spacing. The real
     100/250 spacing lines up by arithmetic, but no test pins it.
  7. The chain cannot be restarted directly at the search or aggregate stage. It was not needed.
- **The depth-11 and depth-9 searches are 200 held-out states per seed.** The secondaries and
  the depth-9 contrast are reported, not tested against a bar.

## Provenance

- Machine: SwizzlesDuo (Intel Ultra 9 185H, 22 cores, 31.4 GB), laptop checkout `4e2c17f`.
- Date: 2026-10-26 16:07 to 2026-10-29 14:39 laptop time (ET). Gate C 6 min (4 workers);
  training 16:13 Oct 26 to 09:11 Oct 29 (4 workers, three waves of four seeds, 75,995 to
  80,096 s per seed for both legs); search 09:12 to 14:39 (12 workers; depth-11 cells 12,080 to
  13,043 s, depth-9 cells 5,715 to 6,903 s).
- Seeds 0 to 11, the same as EXP-075, EXP-076 and EXP-081.
- Re-aggregated on the VPS from the copied records. The output's content is identical to the
  laptop's `verdict082.txt`. The only difference is one trailing blank line in the laptop file,
  which its PowerShell write adds. The 96-file bundle (60 records plus 36 checkpoint files)
  hashes the same on both ends: SHA256 over the sorted per-file hashes, `f3842c43...b621b0d2`.
- Tracked: the 60 records (`exp082_gatec_s*.json`, `exp082_X16k_u8000_s*.json`,
  `exp082_X16k_u16000_s*.json`, `exp082_J4V-X16k_d9_s*.json`, `exp082_J4V-X16k_d11_s*.json`)
  and, per seed, `judge_X_s*/judge.pt` (16,000), `judge_u8000.pt` and `state.json`. Not tracked:
  `opt.pt` and `jt.pt`, which only a further continuation would need. They remain on the laptop.
- Regenerate the verdict from the tracked records:

```bash
PYTHONPATH=src .venv/bin/python experiments/082_x_long_budget/aggregate.py
```

- Re-run on the laptop (skips finished work):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments\082_x_long_budget\chain082.ps1
```
