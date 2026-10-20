# EXP-079 results: the judge trained where the scrambles live

**Both claims REFUTED, every gate passed.** Training the judge on walks up to 40 moves (S) instead
of 14 (X, EXP-075) made the 4-move tree search worse at both depths, not better. At depth 11, the
typical random scramble, J4V-S solves **17.1%** against J4V-X's **19.5%** (-0.0238, S ahead on 3
of 12 seeds, p 0.9443). At depth 9 it solves **34.5%** against **40.0%** (-0.0550, S ahead on 1 of
12, p 0.9995).

**Gate M was not above 0** (+0.0117 at probe distances 9 to 11, S ahead on 7 of 12, p 0.2310). S
was trained to rank far-from-goal states better and does not measurably do so. Spec section 7 makes
Gate M bind the reading only if Claim 1 confirms; it did not, so Gate M is the mechanism reading
behind the refutation rather than a caveat on a gain. **The lever moved the training mass as
designed and did not move the ranking where it was aimed.** The aggregator's own reading: the
judge is not the limit BY THIS LEVER, which is not evidence that search is the limit.

Spec: `docs/superpowers/specs/2026-10-18-exp079-judge-trained-where-scrambles-live-design.md`
(section 12 is the dated pilot amendment). Plan: `docs/superpowers/plans/2026-10-19-exp079-judge-trained-deep.md`.

## Primaries (spec section 8)

One-sided exact sign-flip over all 4096 flips, alpha 0.025 each (0.05 / 2).

| claim | S | X | diff | S ahead | p | verdict |
|---|---|---|---|---|---|---|
| 1. J4V-S > J4V-X, depth 11 | 0.1713 | 0.1950 | -0.0238 | 3 of 12 (1 tie) | 0.9443 | **REFUTED** |
| 2. J4V-S > J4V-X, depth 9 | 0.3454 | 0.4004 | -0.0550 | 1 of 12 (1 tie) | 0.9995 | **REFUTED** |

## Gates

| gate | result |
|---|---|
| 0(a) determinism: J4V-S d9 seed 0 re-run into `outputs_det/` | **PASS** (62/200 both times) |
| 0(b) continuity: J4V-X d9 and d11 seed 0 re-run here against EXP-076's records | **PASS** (78/200 and 36/200) |
| L (VI worked) | **PASS**: mean margin 0.2479 vs threshold 0.1118, p 0.0002 |
| E (encoders trained) | **PASS**: drift 11.91 to 14.07 on every seed |
| M continuity (X seed 0 Gate L re-measured here) | **PASS** |
| M (S - X margin at probe distances 9 to 11, reported) | **NOT ABOVE 0**: +0.0117, 7 of 12, p 0.2310 |
| 1 floor 0.02 and ceiling 0.98 | no claim cell outside it |

## Mean held-out success (200 states per cell, 12 seeds)

| depth | J4V-S | J4V-X | S mean steps to solve | X mean steps | S optimality | X optimality |
|---|---|---|---|---|---|---|
| 9 | 0.3454 | **0.4004** | 11.43 | 10.75 | 0.789 | 0.839 |
| 11 | 0.1713 | **0.1950** | 16.40 | 15.84 | 0.672 | 0.696 |

S is worse on every axis: fewer solves, longer solves, lower optimality.

## Per seed (success rate; X is EXP-076's committed records)

| seed | d9 S | d9 X | d11 S | d11 X | S Gate L margin | S encoder drift | S spearman_7_11 |
|---|---|---|---|---|---|---|---|
| 0 | 0.310 | 0.390 | 0.185 | 0.180 | 0.2344 | 13.22 | 0.290 |
| 1 | 0.325 | 0.460 | 0.150 | 0.210 | 0.2317 | 12.60 | 0.343 |
| 2 | 0.285 | 0.335 | 0.240 | 0.165 | 0.2605 | 14.02 | 0.449 |
| 3 | 0.285 | 0.365 | 0.170 | 0.215 | 0.1919 | 13.91 | 0.234 |
| 4 | 0.425 | 0.455 | 0.150 | 0.175 | 0.2377 | 11.91 | 0.268 |
| 5 | 0.440 | 0.440 | 0.165 | 0.195 | 0.2809 | 12.09 | 0.241 |
| 6 | 0.250 | 0.310 | 0.125 | 0.185 | 0.2225 | 13.96 | 0.346 |
| 7 | 0.380 | 0.400 | 0.165 | 0.250 | 0.2386 | 12.19 | 0.295 |
| 8 | 0.415 | 0.410 | 0.165 | 0.190 | 0.2557 | 14.07 | 0.282 |
| 9 | 0.345 | 0.415 | 0.215 | 0.170 | 0.2869 | 12.64 | 0.315 |
| 10 | 0.340 | 0.430 | 0.150 | 0.230 | 0.2841 | 13.11 | 0.195 |
| 11 | 0.345 | 0.395 | 0.175 | 0.175 | 0.2502 | 13.11 | 0.254 |

## Secondaries and Gate R (patterns, never confirmations)

| quantity | S | X |
|---|---|---|
| probe spearman_7_11, mean over seeds | 0.2926 | 0.2983 |
| probe mean J at true distance 9 | 7.980 | 7.135 |
| probe mean J at true distance 10 | 8.137 | 7.270 |
| probe mean J at true distance 11 | 8.283 | 7.380 |
| Gate R k 4, d9: lowest-J leaf closer than root (chance 0.0478) | 0.2271 | 0.2308 |
| Gate R k 4, d11 (chance 0.1628) | 0.2921 | 0.2838 |

Realised training histogram, mean over seeds (last probe interval): distance 10 0.176, 11 0.196,
12 0.107, against EXP-075's walks to 14 where distance 11 held about 1.3% (spec section 2).

- **S's J is better calibrated in absolute terms far out and no better at ordering.** Its mean J
  at true distances 9 to 11 sits about 0.85 to 0.9 higher than X's, nearer the truth, but the
  rank correlation over 7 to 11 is level (0.293 vs 0.298) and Gate M is level. Search uses only
  the order of the leaves, so calibration alone buys nothing.
- **Gate R is level at both depths** (S 0.227 vs X 0.231 at d9, 0.292 vs 0.284 at d11). The
  judge's lowest-J leaf is closer than the root as often as X's, yet the episodes solve less. So
  the loss is not in the first-step pick that Gate R reads. Nothing here localises it further.
- **The depth-11 refutation is not a small miss.** -0.0238 is about the size of the detectable
  difference EXP-078 reported (about 0.028), in the wrong direction. The depth-9 loss, -0.055 with
  S behind on 10 of 12 seeds, is larger.

## What it means

- **Moving the judge's training mass to where the scrambles live did not help the search, and
  hurt it.** The roadmap's account after EXP-078 was that the judge ranks well near the goal and
  poorly far from it. This lever targeted "far from it" directly, and S does not rank better
  there (Gate M), and it ranked worse at distances 7 and 8 on both pilot seeds.
- **It is one lever, refuted.** It does not show the judge is not the limit, and it does not show
  search is. Spec section 11 lists what was left out: deeper training targets (the ST arm), other
  walk lengths or a sampled mixture, and warm-starting from X.

## Disclosures

- **The pilot warned about this.** S ranked below X overall on both pilot seeds and gained at
  distances 9 to 11 only on seed 12 (spec section 12). The pre-registered stop rule (competence)
  passed, Michael gave the go on neuromorphic#24 with that table in front of him, and the run was
  not stopped on the warning.
- **Gate 0(b) compares EXP-071's 8 `OUTCOME_FIELDS`**, which omit `fallback_frac`, as in EXP-074
  to EXP-078.
- **`wall_s` is the evaluation loop's time only**, not process start-up or judge loading.
- **The launcher's per-cell `peak commit MB` reads 2 MB** in the cont, eval, det and rank phases:
  it sampled the venv redirector, not the worker. A manual sample 2 h into evaluation read 1.1 to
  1.3 GB committed per evaluation worker, under the amendment's estimate. Training memory (4210
  MB per worker) is from the pilot, where the reading was correct.
- **Every full-run record says `7d39ea2`**, the laptop's checkout when the chain started. The pilot
  records say `7604414`. `git diff 7604414 7d39ea2` under `src/`, `tests/` and this experiment's
  folder is the amendment only (the two constants in `aggregate.py`, the test update, the pilot
  records).
- **No training run resumed** (`resumed_from` 0 on all 12) and no evaluation cell restarted. The run
  crossed one 03:31 window with no restart.

## Cost and provenance

- Machine: SwizzlesDuo (laptop), checkout `7d39ea2`, run by
  `experiments/079_judge_trained_deep/chain079.ps1` (cont, Gate 0(b), train, Gates L, E, M, eval,
  det, rank). Started 2026-10-19 11:19 laptop time (UTC-4), finished 2026-10-20 18:05: 30.8 h of
  wall time, inside the amendment's 30 to 40 h.
- Pilot: 2 runs, 5.6 h each at 2 workers, finished 2026-10-19 04:20.
- Training: 12 runs at 4 workers in three waves, 6.9 to 7.4 h per run, 21.8 h of wall time.
- Evaluation: 23.9 cell-hours at depth 11 and 16.5 at depth 9, 6 workers, 6.8 h of wall time.
  Continuity 1.15 h; determinism 0.5 h at 1 worker; rank 0.5 h.
- J4V-X is EXP-076's committed records, re-run only at seed 0 for Gate 0(b).
- Records: `outputs/` (12 training, 24 evaluation, 24 rank, 2 continuity, 1 Gate M X continuity,
  and each seed's `judge_S_s<seed>/judge.pt` and `state.json`), `outputs_det/` (1),
  `outputs_pilot/` (2), all tracked.

Regenerate the verdicts from the tracked records:

```bash
.venv/bin/python experiments/079_judge_trained_deep/aggregate.py
```
