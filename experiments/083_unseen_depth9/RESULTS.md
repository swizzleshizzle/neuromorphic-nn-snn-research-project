# EXP-083 results: does X16k's depth-9 gain hold on states it never trained on, and reach depth 10?

**Verdict: DEPTH-9 GAIN CONFIRMED, REACHES DEPTH 10.** All four gates passed. On 200 depth-9
states per seed that neither judge ever saw in training, the search with EXP-082's X16k judge
solves **0.5537 against J4V-X's 0.3279** (+0.2258, ahead on 12 of 12 seeds, p 0.0002). At
depth 10, a depth with no earlier cell, it solves **0.2904 against 0.1992** (+0.0912, 12 of 12,
p 0.0002). Both clear their pre-registered bars (0.10 and 0.05) by more than double.

EXP-082 reported +0.2196 at depth 9 outside its pre-registration. The pre-registered rerun on
new states measures +0.2258, so the gain did not shrink. Both arms are lower on the new states
(X16k 0.6200 to 0.5537, X 0.4004 to 0.3279), so the new population is harder for both, and the
gap between them held.

| | depth 9 (new sets) | depth 9 (EXP-082, old held-out) | depth 10 (new sets) | depth 11 (EXP-082) |
|---|---|---|---|---|
| **J4V-X16k** | **0.5537** | 0.6200 | **0.2904** | 0.2083 |
| J4V-X | 0.3279 | 0.4004 | 0.1992 | 0.1950 |
| diff | **+0.2258** | +0.2196 | **+0.0912** | +0.0133 |
| ahead | 12 of 12 | 12 of 12 | 12 of 12 | 9 of 12 |

Read across, the gain fades with depth: about +0.22 at 9, +0.09 at 10, +0.01 at 11. That is the
outward spread of value iteration that EXP-082 described (4x budget pushed good ordering about
one distance further out), now measured at the depth in between.

Spec: `docs/superpowers/specs/2026-10-30-exp083-unseen-depth9-design.md`. Plan:
`docs/superpowers/plans/2026-10-30-exp083-unseen-depth9.md`.

## Gates

| gate | result |
|---|---|
| V (voiding: the replayed walk stream is the training stream; the replayed random state equals EXP-075's `state.json` at 4000 and EXP-082's at 16,000) | **PASS** on 12 of 12, at both checkpoints |
| R (voiding: J4V-X16k depth 9 seed 0 on EXP-082's own held-out states reproduces EXP-082's record) | **PASS**: solved 127, want 127; success 0.635 |
| S (each set holds 200 states) | **PASS** on 12 of 12 at both depths |
| F (the better arm is at least 0.05 above the measured random-policy floor) | **PASS**: depth 9 0.5537 against floor 0.0000; depth 10 0.2904 against 0.0000 |

The random-policy floor is exactly zero: 0 of 2,400 states solved at each depth, with the
cell's 2d+3 step budget.

## Claims (spec section 6)

One-sided exact sign-flip test over all 4096 flips, alpha 0.025, paired by seed.

| claim | contrast | diff | ahead | p | bar | verdict |
|---|---|---|---|---|---|---|
| 1 (primary) | J4V-X16k minus J4V-X, success on U_9 | +0.2258 | 12 of 12 | 0.0002 | 0.10 | **CONFIRMED** |
| 2 | J4V-X16k minus J4V-X, success on U_10 | +0.0912 | 12 of 12 | 0.0002 | 0.05 | **CONFIRMED** |

p 0.0002 is the smallest p-value 12 seeds can produce (1 in 4096). The smallest per-seed gain
is +0.155 at depth 9 and +0.040 at depth 10. The paired sd is 0.0358 at depth 9, the same as
EXP-082's 0.0358 on the old sets, and 0.0233 at depth 10. The spec allowed for up to twice
that.

## Per seed

Solved out of 200. "Old" is EXP-082's J4V-X16k depth-9 count on EXP-075's held-out set.

| seed | d9 X16k | d9 X | diff | d10 X16k | d10 X | diff | old d9 X16k |
|---|---|---|---|---|---|---|---|
| 0 | 124 | 77 | +47 | 67 | 45 | +22 | 127 |
| 1 | 115 | 66 | +49 | 57 | 41 | +16 | 139 |
| 2 | 93 | 61 | +32 | 60 | 43 | +17 | 106 |
| 3 | 99 | 52 | +47 | 50 | 32 | +18 | 117 |
| 4 | 107 | 62 | +45 | 64 | 38 | +26 | 128 |
| 5 | 111 | 64 | +47 | 52 | 44 | +8 | 130 |
| 6 | 93 | 62 | +31 | 58 | 34 | +24 | 91 |
| 7 | 114 | 62 | +52 | 58 | 43 | +15 | 137 |
| 8 | 125 | 72 | +53 | 60 | 41 | +19 | 126 |
| 9 | 111 | 69 | +42 | 56 | 38 | +18 | 132 |
| 10 | 118 | 73 | +45 | 63 | 43 | +20 | 127 |
| 11 | 119 | 67 | +52 | 52 | 36 | +16 | 128 |

## Reported, never confirmations (spec section 9)

| | J4V-X16k d9 | J4V-X d9 | J4V-X16k d10 | J4V-X d10 |
|---|---|---|---|---|
| success | 0.5537 | 0.3279 | 0.2904 | 0.1992 |
| optimality | 0.9315 | 0.7972 | 0.8264 | 0.7295 |
| mean steps | 9.67 | 11.30 | 12.12 | 13.73 |
| goal check fired (fraction of steps) | 0.1523 | 0.0748 | 0.0600 | 0.0395 |
| fallback | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- **The new population moves both arms down by a similar amount**: X16k by 0.066, X by 0.073.
  U_d holds the states the training walks happened to miss, not a uniform draw from the shell
  (spec section 4). This run does not measure why those states are harder. The paired contrast
  is unaffected, since both arms see the same states.
- X16k's depth-9 solves are near optimal (0.93) and use 1.6 fewer steps than X's.
- Per seed, the replay saw 1,175,730 to 1,177,761 distinct training states, leaving 75,964 to
  76,916 depth-9 candidates and 496,144 to 497,533 depth-10 candidates before the 200 cap.

## What it means

- **The depth-9 result is now a confirmed number.** A plain 4-move tree with a judge trained by
  value iteration, using no true distances, solves 55% of depth-9 scrambles it never trained
  on, against 33% for the same judge at a quarter of the budget. It can be built on.
- **Depth 10 is the new edge.** The gain is real there (+0.09, 12 of 12) but less than half the
  depth-9 gain, and at depth 11 EXP-082 found almost none. Budget moved the frontier about one
  depth, not three. This fits EXP-082's mechanism: value iteration spreads ordering outward
  about one distance per 12,000 updates at this setup.
- **What this does not answer** (spec section 12): whether ordering keeps spreading with more
  training, which is options 1 and 2 on neuromorphic#29 and needs the GPU, and whether the gain
  holds for a judge retrained on another machine.
- **The next stage is Michael's call.**

## Disclosures

- **Both arms were rerun on the new sets.** EXP-076's published J4V-X records are on the old
  held-out set and cannot be the paired baseline (spec section 4).
- **Depth 10 uses the seed's depth-9 policy agent, head and train seed**, as EXP-074's depth-11
  cell did. Mode C scores moves with the judge, and both arms share the head.
- **The depth-9 policy head was trained on EXP-070's training side of the depth-9 shell**, which
  U_9 may overlap. Both arms share the head, so the paired contrast is unaffected (spec section 4).
- **"Trained on" means a state appeared in a training batch.** Children scored only through the
  frozen target network are not counted, since they receive no gradient (spec section 2).
- **The search ran slower than the spec's 7.5 h.** It took 10.6 h (00:58 to 11:36), with
  per-cell times from 2,602 to 4,977 s at depth 9 and 3,441 to 6,104 s at depth 10. The first
  depth-10 wave took about 3,500 s per cell, close to the spec's 3,900 s estimate; later cells
  ran up to 75% slower at the same 6 workers. The cause was not measured.
- **There was no restart.** The laptop last booted 2026-10-29 23:01 ET, before the chain started
  at 00:41, and the chain ran once from start to verdict.
- **Deferred minors from the build's final review**, none of which bit on this run:
  1. A Gate R failure does not stop the chain early; the search would run to the end and the
     aggregator would print VOID. Gate R passed.
  2. A sets file whose Gate V failed is skipped on a rerun and must be deleted by hand. No seed
     failed Gate V.
  3. Floor records carry no `git_commit`, so the aggregator's single-checkout check covers only
     the sets, Gate R and search records. Those 61 records all carry `9ffe730`; the 24 floor
     records were written by the same chain run at that checkout.
- **Rulings made during the build** (from its ledger): the sets builder takes one BFS provider
  and injectable exclusion, shell and provider functions, for testability; Gate R reads its
  expected values from the EXP-082 record rather than from constants; the sets-only check is
  `aggregate.py --stage sets`; the review finding that a short set crashed the full aggregate
  was regraded Important and fixed (it now prints UNRESOLVED).

## Provenance

- Machine: SwizzlesDuo (Intel Ultra 9 185H, 22 cores, 31.4 GB), laptop checkout `9ffe730`.
- Date: 2026-10-30, 00:41 to 11:37 laptop time (ET). Sets 00:41 to 00:58 (4 workers); search
  and Gate R 00:58 to 11:36 (6 workers; Gate R alone 2,161 s); floor 11:36 to 11:37 (4 workers).
- Seeds 0 to 11, the same as EXP-075, EXP-076, EXP-081 and EXP-082.
- Re-aggregated on the VPS from the copied records. The output is identical to the laptop's
  `verdict083.txt`, except for one trailing blank line that the laptop's PowerShell write adds.
- Tracked: all 85 records (`exp083_sets_s*.json`, `exp083_gateR_s0.json`,
  `exp083_J4V-X16k_d{9,10}_s*.json`, `exp083_J4V-X_d{9,10}_s*.json`,
  `exp083_floor_d{9,10}_s*.json`). The judges are EXP-075's and EXP-082's tracked checkpoints.
- Regenerate the verdict from the tracked records:

```bash
PYTHONPATH=src .venv/bin/python experiments/083_unseen_depth9/aggregate.py --stage all
```

- Re-run on the laptop (skips finished work):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experiments\083_unseen_depth9\chain083.ps1
```
