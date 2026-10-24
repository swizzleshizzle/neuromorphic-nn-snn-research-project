# EXP-080 results: the oracle ordering ceiling

> **Superseded on the budget question by EXP-081 (2026-10-24).** Continued from this run's
> checkpoints to 16,000 updates, O-Xt beats X's far band by +0.0678 (p 0.0012) and as the depth-11
> judge solves 0.4088 against J4V-X's 0.1950 (12 of 12): BUDGET-LIMITED, PAYS. The ENCODER-LIMITED
> reading below is a 4000-update reading. See `experiments/081_oxt_long_budget/RESULTS.md`.

**Verdict: ENCODER-LIMITED. Both claims REFUTED, every gate passed.** The question was whether a
judge built on EXP-075's X encoder could order states far from the goal (true distances 9 to 11)
better than X itself, if it were trained directly on true BFS distance. It cannot, by enough to
count:

| arm | what it is | margin at distances 9 to 11 | minus X |
|---|---|---|---|
| X (EXP-075) | the reference | 0.1195 | |
| O-Xt | fresh head on X's encoder, encoder fine-tuned | 0.1423 | **+0.0228** |
| O-Xf | fresh head on X's frozen encoder | 0.1301 | +0.0106 |
| O-raw | plain MLP on raw facelets, no encoder | 0.2973 | **+0.1778** |

The same true labels, the same states and the same probe give a raw-facelet MLP 0.18 of
far-band margin over X (12 of 12 seeds, p 0.0002), and give X's encoder only 0.02 to 0.01.
**The ordering is learnable from the cube and not reachable through X's encoder** at this budget.
The search cells were not run: they are read only when Claim 1 confirms (spec section 8), and the
chain stopped before them.

Spec: `docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md` (section 13 is
the dated pilot amendment). Plan: `docs/superpowers/plans/2026-10-21-exp080-oracle-ordering.md`.

## Gates

| gate | result |
|---|---|
| 0 (X seed 0's Gate L re-measured through the mint, and the mint's sha) | **PASS** |
| V (the protocol resolves: O-raw minus X far-band, p < 0.05 and mean at least 0.05) | **PASS**: +0.1778, 12 of 12, p 0.0002 |
| T (trained: mean overall margin above 0.13564) | **PASS** on all three: O-raw 0.4703, O-Xf 0.2319, O-Xt 0.2499 |

Encoder drift: O-Xf exactly 0.0 on every seed (frozen, as designed); O-Xt 12.73 to 13.54 (the
encoder trained).

## Primaries (spec section 7)

One-sided exact sign-flip over all 4096 flips, alpha 0.025; CONFIRMED also needs a mean of at
least 0.05, otherwise SMALL.

| claim | arm | X | diff | ahead | p | verdict |
|---|---|---|---|---|---|---|
| 1. O-Xt > X, far band | 0.1423 | 0.1195 | +0.0228 | 7 of 12 | 0.0547 | **REFUTED** |
| 2. O-Xf > X, far band | 0.1301 | 0.1195 | +0.0106 | 7 of 12 | 0.2407 | **REFUTED** |

Claim 1 misses on both counts. Had p cleared 0.025, a mean of +0.0228 would still read SMALL,
under half the 0.05 bar, so the verdict does not hinge on the p value.

## Per seed (margin, hit rate minus chance, 50 probe states per distance)

| seed | X far | O-raw far | O-Xf far | O-Xt far | O-Xt - X | X overall | O-Xt overall |
|---|---|---|---|---|---|---|---|
| 0 | 0.1745 | 0.3545 | 0.2079 | 0.2145 | +0.0400 | 0.3024 | 0.2704 |
| 1 | 0.1110 | 0.3044 | 0.0644 | 0.1510 | +0.0400 | 0.2597 | 0.2437 |
| 2 | 0.0851 | 0.3118 | 0.1451 | 0.1718 | +0.0867 | 0.2005 | 0.2605 |
| 3 | 0.0483 | 0.3083 | 0.1217 | 0.1217 | +0.0733 | 0.2399 | 0.2479 |
| 4 | 0.1793 | 0.2459 | 0.1526 | 0.1659 | -0.0133 | 0.3017 | 0.2377 |
| 5 | 0.1108 | 0.2441 | 0.1241 | 0.1441 | +0.0333 | 0.2649 | 0.2769 |
| 6 | 0.1087 | 0.2954 | 0.1421 | 0.0887 | -0.0200 | 0.2745 | 0.2225 |
| 7 | 0.1552 | 0.2952 | 0.0819 | 0.1419 | -0.0133 | 0.3026 | 0.2626 |
| 8 | 0.0750 | 0.3484 | 0.1417 | 0.1617 | +0.0867 | 0.2597 | 0.2757 |
| 9 | 0.1109 | 0.2775 | 0.1509 | 0.1242 | +0.0133 | 0.2789 | 0.2309 |
| 10 | 0.1563 | 0.3163 | 0.1230 | 0.1363 | -0.0200 | 0.2881 | 0.2481 |
| 11 | 0.1191 | 0.2657 | 0.1057 | 0.0857 | -0.0333 | 0.2822 | 0.2222 |

O-Xt gains most on the seeds where X is weakest far out (seeds 2, 3 and 8), and loses where X is
strong (4, 7, 10).

## Reported, not gating (spec section 9)

| quantity | O-raw | O-Xf | O-Xt | X |
|---|---|---|---|---|
| margin at distance 7 | 0.8064 | 0.5031 | 0.4964 | **0.6914** |
| margin at distance 8 | 0.6530 | 0.2663 | 0.3263 | 0.3063 |
| margin at distance 9 | 0.4290 | 0.1424 | 0.1640 | 0.1524 |
| margin at distance 10 | 0.2895 | 0.1229 | 0.1629 | 0.1245 |
| margin at distance 11 | 0.1734 | 0.1250 | 0.1000 | 0.0817 |
| overall margin (7 to 11) | 0.4703 | 0.2319 | 0.2499 | 0.2713 |
| probe spearman_7_11 | 0.4855 | 0.2815 | 0.3315 | 0.2983 (EXP-079) |
| probe mean J at distance 9 / 10 / 11 | 10.03 / 10.20 / 10.40 | 9.61 / 9.78 / 9.92 | 9.70 / 9.92 / 10.06 | |
| training loss at the end (MSE, moves squared) | 1.49 | 3.45 | 2.94 | |

O-Xt minus S (EXP-079's judge, same walks, different target) far band: +0.0111, 6 of 12, p 0.1938.

- **True labels through X's encoder rank WORSE than X near the goal.** At distance 7 both oracle
  arms sit near 0.50 against X's 0.69, and their overall margin is below X's. X was trained by
  value iteration on walks up to 14 moves, mostly near the goal; the oracle arms spread their fit
  over walks to 40. One reading, not tested here: through this encoder, with this head and
  budget, a judge gets X's near-goal order or a calibrated far-out value, not both.
- **The encoder arms' loss plateaus near 3 (about 1.7 to 1.9 moves RMS) from update 1000 on**,
  O-raw's near 1.5. Fine-tuning the encoder (O-Xt over O-Xf) lowered the loss by 0.5 and moved
  the far band by 0.012. The limit is not the head: both arms put a fresh wide head on the same
  features, and the frozen one is only 0.012 behind.
- **The 2x2 cube's state is in the facelets, and a 144-512-512-1 MLP recovers a far-band order
  2.5 times X's** in 12 epochs over 600,000 states, about 13 minutes per run. The information
  is in the input. The spiking encoder's representation is where it is lost.

## What it means

- **Spec section 8 points the next stage at the encoder or region, not at the judge's training
  signal.** EXP-079 changed the judge's training mass and moved nothing; this run gave the judge
  the answer key and it still moved only 0.02. That suggests why EXP-079's lever could not
  have worked: if the answer key buys 0.02, a better-placed training mass buys less.
- **It does not say search is the limit, and it does not say a better encoder pays at depth 11.**
  The search arm was not run. What it does say: any judge on X's encoder is close to its far-band
  ceiling already, so further judge objectives on this encoder are not worth a run.
- The next stage is Michael's call.

## Disclosures

- **The pilot predicted this** (spec section 13): O-Xt minus X far was +0.043 on seeds 12 and 13,
  under the 0.05 bar, and O-Xf was -0.013. The full run's +0.023 and +0.011 are smaller still.
- **The full run started about 13 h after the pilot finished**, because the go-or-hold question
  sat on neuromorphic#26 when Michael's "run it" had already covered it. Nothing changed in that
  gap: the laptop's checkout stayed at `c33c821`.
- **O-raw is the protocol check, not a matched competitor.** It trains on the first 600,000 of the
  4,000,000 minted states for 12 epochs; O-Xf and O-Xt see all 4,000,000 once. O-raw saw fewer
  distinct states and still won by 0.15 to 0.17.
- **The budget is 4000 updates**, fixed before the pilot. O-Xt's loss was still drifting down
  slowly at the end (about 3.1 to 2.9 over its last 1,250 updates; O-Xf's was flat). A much longer
  run might close some of the gap; this experiment does not measure that.
- **Every full-run record says `c33c821`**, the amendment commit. The pilot records say
  `005773e`; `git diff 005773e c33c821` is the amendment only (spec section 13, `AMENDMENT` in
  `aggregate.py`, its test, the pilot records).
- **No training run resumed** (`resumed_from` 0 on all 36). The run ended at 23:32, before the
  03:31 window.
- The mint files (states, labels, leaf distances, about 4M states per seed) are untracked and stay
  on the laptop under `outputs/mint/`; each training record carries its mint's sha256.

## Cost and provenance

- Machine: SwizzlesDuo (laptop), checkout `c33c821`, run by
  `experiments/080_oracle_ordering/chain080.ps1` (mint, Gate 0, train, gates). Started
  2026-10-21 18:24 laptop time (UTC-4), finished 23:32: **5.1 h of wall time**, under the
  amendment's 6 to 9 h because the search did not run.
- Mint: 12 seeds at 6 workers, 11 min. Training: 36 runs at 6 workers, 4.95 h. O-Xt 5,000 to
  5,500 s per run (about 1.3 s per update, the pilot's speed), O-Xf 2,400 to
  3,200 s, O-raw 750 to 800 s.
- Peak committed memory: 2,227 MB per python process in training, 12,234 MB across all.
- X is EXP-075's committed records; S is EXP-079's.
- Pilot: 01:14 to 03:04 the same day, seeds 12 and 13, records in `outputs_pilot/`.
- Records: `outputs/` (36 training records, 1 Gate 0 record, and each judge's
  `judge_<arm>_s<seed>/judge.pt` and `state.json`), `outputs_pilot/` (6 training, 1 timing), all
  tracked.

Regenerate the verdict from the tracked records:

```bash
PYTHONPATH=src .venv/bin/python experiments/080_oracle_ordering/aggregate.py
```
