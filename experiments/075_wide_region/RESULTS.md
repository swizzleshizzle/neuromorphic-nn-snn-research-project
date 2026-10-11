# EXP-075 results: a wider sensory region for the learned judge

**Claims 1 and 2 CONFIRMED at depth 9, on all 12 seeds and on the 10-seed sensitivity line.
Claim 3 (depth 11) NOT SIGNIFICANT.** A judge whose sensory region is four times wider (hidden 512,
inverse-model pretrained, then trained by value iteration) solves 25.7% of depth-9 held-out
scrambles, against 19.3% for EXP-074's best judge and 17.0% for the same recipe at hidden 128.
It wins every depth-9 seed against both. At depth 11, the typical random scramble, it solves 12.0%
against 10.1%, a positive difference that this experiment cannot distinguish from zero.

Spec: `docs/superpowers/specs/2026-10-09-exp075-wide-sensory-region-design.md` (section 12 is the
dated pilot amendment, sections 13 and 14 the gate records). Plan:
`docs/superpowers/plans/2026-10-09-exp075-wide-sensory-region.md`.

## Primaries (spec section 9)

One-sided exact sign-flip over all 4096 (12 seeds) or 1024 (10 seeds) flips, alpha 0.0167 each
(0.05 / 3). The 10-seed line drops seeds 0 and 3, the evaluation cells disclosed in spec section 2.
**The two lines agree on every claim.**

| claim | 12 seeds | 10 seeds (0 and 3 dropped) |
|---|---|---|
| 1. J3V-X > J3V-W, depth 9 | +0.0638, p 0.0002, **CONFIRMED** | +0.0650, p 0.0010, **CONFIRMED** |
| 2. J3V-X > J3V-Y, depth 9 | +0.0867, p 0.0002, **CONFIRMED** | +0.0880, p 0.0010, **CONFIRMED** |
| 3. J3V-X > J3V-W, depth 11 | +0.0188, p 0.0757, NOT SIGNIFICANT | +0.0130, p 0.1777, NOT SIGNIFICANT |

**Claim 3 is not a refutation.** The mean is positive and X wins 8 of 12 seeds (1 tie, 3 losses).
The measured paired sd is 0.041, so the se is about 0.012 and +0.019 is 1.6 se. The spec's power
statement put the detectable difference at about 0.03; the observed difference is below it.

## Mean held-out success (200 states per cell, 12 seeds)

| depth | J3V-X | J3V-Y | J3V-W | P3V | R3V |
|---|---|---|---|---|---|
| 7 | **0.8908** | 0.7592 | 0.7908 | 0.3217 | 0.0033 |
| 8 | **0.5158** | 0.3492 | 0.3792 | 0.1496 | 0.0013 |
| 9 | **0.2571** | 0.1704 | 0.1933 | 0.0629 | 0.0004 |
| 11 | **0.1200** | 0.0879 | 0.1013 | 0.0308 | 0.0004 |

J3V-X: the wide region (144 -> 512 -> 64, inverse-model pretrained by EXP-039's recipe, then
value iteration), read through the wide readout (576 inputs). J3V-Y: the same recipe at hidden 128
(192 inputs). J3V-W: EXP-074's committed W judge (E1's region, hidden 128, RL fine-tuned before
value iteration), reused under Gate 0(b). P3V and R3V: EXP-074's reference rows (the best policy
recipe and the random floor), not re-run here.

## Depth 9 and depth 11, per seed

| seed | d9 X | d9 Y | d9 W | d11 X | d11 Y | d11 W |
|---|---|---|---|---|---|---|
| 0 | 0.280 | 0.155 | 0.180 | 0.150 | 0.095 | 0.075 |
| 1 | 0.250 | 0.140 | 0.200 | 0.110 | 0.055 | 0.105 |
| 2 | 0.215 | 0.155 | 0.160 | 0.100 | 0.085 | 0.105 |
| 3 | 0.215 | 0.180 | 0.200 | 0.135 | 0.105 | 0.115 |
| 4 | 0.285 | 0.170 | 0.175 | 0.125 | 0.120 | 0.050 |
| 5 | 0.315 | 0.165 | 0.185 | 0.130 | 0.050 | 0.095 |
| 6 | 0.175 | 0.165 | 0.160 | 0.100 | 0.070 | 0.080 |
| 7 | 0.290 | 0.205 | 0.250 | 0.120 | 0.125 | 0.165 |
| 8 | 0.295 | 0.210 | 0.225 | 0.110 | 0.125 | 0.090 |
| 9 | 0.280 | 0.160 | 0.195 | 0.140 | 0.075 | 0.070 |
| 10 | 0.265 | 0.150 | 0.185 | 0.105 | 0.080 | 0.150 |
| 11 | 0.220 | 0.190 | 0.205 | 0.115 | 0.070 | 0.115 |

At depth 9, X beats W on **12 of 12** seeds and Y on **12 of 12** (paired sd 0.039 and 0.044).
At depth 11, X beats W on 8 (1 tie, 3 losses) and Y on 10 (2 losses).

## Gates (all PASS)

| gate | result |
|---|---|
| P pretraining | every seed's move accuracy at or above 0.9 x its arm's pilot mean: X min 0.5518 (threshold 0.5024), Y min 0.4508 (0.4097). Y's 12 seeds sit at EXP-039's 0.449 to 0.457. |
| L training worked | mean margin X 0.2713 (threshold 0.1298), Y 0.2246 (0.1048); p 0.0002 each. |
| E encoders | drift > 0 on every seed: X 11.26 to 14.44, Y 7.54 to 8.56. |
| R leaf ranking | J-X and J-Y at depths 7, 8, 9, 11 and J-W at 9 and 11, all above chance. Table below. |
| 0(a) determinism | J3V-X depth 9 seed 0 re-run into a separate directory differs from its full-run copy only in `wall_s` (56/200). Record in `outputs_det/`. |
| 0(b) continuity | J3V-W at seed 0, depths 9 and 11, re-evaluated from EXP-074's committed judge, equals EXP-074's records in every outcome field (36/200 and 15/200). |
| 1 resolution | no contrast in the 0.02 / 0.98 band, encoded in the verdict function. |
| Y working control | Y's mean pilot margin 0.2097 cleared W's Gate L threshold 0.1018 (spec section 6). |

**Gate R, the mechanism line.** The fraction of held-out states whose lowest-J 3-move leaf is
closer to solved than the root (mean hit / chance):

| depth | chance | J-X | J-Y | J-W |
|---|---|---|---|---|
| 7 | 0.1344 | **0.8329** | 0.6754 | (0.7063, EXP-074) |
| 8 | 0.1486 | **0.4571** | 0.3692 | (0.3658, EXP-074) |
| 9 | 0.1751 | **0.3271** | 0.3125 | 0.3092 |
| 11 | 0.5214 | 0.5700 | 0.5692 | 0.5517 |

The X - Y leaf-rate advantage shrinks with depth: +0.158, +0.088, +0.015, +0.001. The solve-rate
advantage does not shrink the same way (+0.132, +0.167, +0.087, +0.032). As in EXP-074, one-step
leaf ranking at the root is a floor check and does not predict the episode-level difference: at
depth 9 X, Y and W have leaf rates within 0.018 of each other while X solves 0.06 to 0.09 more.

## Secondary patterns (never confirmations)

- Depths 7 and 8, X - W: +0.100 and +0.137; X - Y: +0.132 and +0.167. All p 0.0002.
- Depth 11, X - Y: +0.032, p 0.0024.
- Depth 9, Y - W: -0.023, p 0.9998. With the region width held at 128, the laptop-pretrained Y
  solves slightly less than EXP-074's W, whose region also had EXP-040's RL fine-tune before value
  iteration (Y and W also differ in machine and pretraining run). Whatever separates them is
  about 0.02 at depth 9; the width step, Y to X, is about 0.09.

## What this says about the goal

The spec's diagnostic (section 1) found the frozen 192-unit readout capped a supervised head at
about where value iteration already was, and a raw-facelet head at 88/200 on depth 9, seed 0.
Widening the region moved the VI judge from 36/200 to 56/200 on that cell, and from 0.193 to 0.257
on the 12-seed mean. The ceiling is still well above: at depth 11, which is what a random scramble
looks like, the best SNN judge solves about one cube in eight.

## Caveats and recorded rulings

- **One training relaunch (spec section 14).** About 15 h into training, Windows logged a low
  virtual memory condition and two X workers faulted; the pool died with no reboot. The watcher
  relaunched the phase, which skipped 18 finished runs and resumed six from their last checkpoint:
  X seeds 7 and 8 from update 3001, X 9 from 2501, X 10 and 11 from 1501, Y 11 from 251. Their
  records carry `resumed_from`, and their `wall_s` covers only the resumed part. Resuming equals an
  uninterrupted run by test (`test_resume_equals_an_uninterrupted_run`). Root cause: the amendment
  budgeted 8 workers by working set (2.82 GB each), but X workers committed about 3.6 GB.
  Evaluation workers committed 1.0 to 1.5 GB, so 12 evaluation workers were safe.
- **Throughput was below the amendment's estimate.** Pretraining took about 5.5 h (estimated 3 h)
  and training about 19 h (estimated 16 h). At 8 workers an X update cost about 12 s, against
  4.16 s at 4 workers in the pilot, so the laptop's total throughput did not rise above 4 workers.
- **The pilot's seconds-per-update was briefly misread as a stall.** A directory timestamp was
  taken for a checkpoint time; the X pilot runs never stopped. Recorded because it was reported
  to Michael mid-run and then corrected.
- **W, P3V and R3V are reused**, not re-run: W from EXP-074 under Gate 0(b), P3V and R3V as
  EXP-074's reference rows. All EXP-075 cells and the Gate 0(b) re-evaluation ran on the same
  laptop.
- **Committed judges reproduce across machines.** `judge_X_s7` (a resumed run) and `judge_Y_s0`,
  loaded on the VPS, recompute their training records' Gate L margins to full float repr
  (0.3026481481481482 and 0.26237037037037036).

## Provenance

- Machine: SwizzlesDuo laptop (Intel Ultra 9 185H, 22 cores, 31.4 GB), Windows, `.venv` Python 3.13.
- Pilot 2026-10-08/09, checkout `b20b2cf`, 4 workers, seeds 12 and 13. Amendment `47246ae`.
- Pretraining 2026-10-09, `47246ae`, 8 workers, about 5.5 h. Training 2026-10-09/10, `4e65594`,
  8 workers, about 19 h, one relaunch. Cont 2026-10-10, `54095ca`. Rank, eval and det
  2026-10-10/11, `b503077`, 12 workers (det 1).
- Seeds 0 to 11. Records: `outputs/exp075_pretrain_*`, `exp075_train_*`, `exp075_rank_*`,
  `exp075_J3V-{X,Y}_d{7,8,9,11}_s*`, `exp075_J3V-W_d{9,11}_s0`; encoders
  `outputs/encoders/enc_{X,Y}_s*.pt`; judges `outputs/judge_{X,Y}_s*/judge.pt` with `state.json`
  (evaluation refuses a judge whose `state.json` is not at update 4000); pilot `outputs_pilot/`;
  determinism `outputs_det/`.

Regenerate, on the laptop from the main checkout:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File launch075.ps1 -Phase pilot-pretrain -Workers 4
powershell -NoProfile -ExecutionPolicy Bypass -File launch075.ps1 -Phase pilot-train    -Workers 4
powershell -NoProfile -ExecutionPolicy Bypass -File launch075.ps1 -Phase pretrain       -Workers 8
powershell -NoProfile -ExecutionPolicy Bypass -File launch075.ps1 -Phase train          -Workers 8
powershell -NoProfile -ExecutionPolicy Bypass -File launch075.ps1 -Phase cont           -Workers 2
powershell -NoProfile -ExecutionPolicy Bypass -File launch075.ps1 -Phase rank           -Workers 12
powershell -NoProfile -ExecutionPolicy Bypass -File launch075.ps1 -Phase eval           -Workers 12 -SkipExisting
powershell -NoProfile -ExecutionPolicy Bypass -File launch075.ps1 -Phase det            -Workers 1
```

Aggregate (any machine with the committed records):

```bash
PYTHONPATH=src .venv/bin/python experiments/075_wide_region/aggregate.py --determinism-ok
```

Retraining from a seed is not portable across machines (CLAUDE.md, reproducibility audit);
re-evaluating the committed judges is.
