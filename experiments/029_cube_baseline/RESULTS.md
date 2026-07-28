# EXP-029 Results - v1 Cube Baseline (fail-first)

> **Why this file exists:** the standing habit adopted after the 2026-07-13 Phase-2 audit. Every
> experiment commits a curated, in-repo results record so the authoritative numbers never live only in a
> gitignored `outputs/` folder or a vault note. **Provenance:** 12 seeds (0-11), 2x2 cube, exact-distance
> shells, run 2026-07-27 on the laptop `SwizzlesDuo` (Intel Core Ultra 9 185H, 22 cores) over SSH with
> `--workers 16`, 264 records in about 2 hours. **The interpretation contract in
> `docs/superpowers/specs/2026-07-25-cube-baseline-design.md` section 5 was committed BEFORE any numbers
> existed** (git `41e7e6b`, with amendments at `4c2d971`). Regenerate with the command at the bottom.

## What this tests

The v1 recipe (frozen random sensory encoder + trainable linear REINFORCE head, `recall=False`) on a 2x2
cube, across **exact** distance-to-solved shells rather than move counts. Three arms:

- `regionalized` - the five-region `Brain`
- `monolithic` - an unregionalized stack matched on **total** neuron count (510)
- `random` - uniform policy, evaluation only, measuring the true chance floor

Depths 1 and 2 are evaluated exhaustively on the whole shell (6 and 27 states) and are
**training-distribution**. Depths 3 to 6 use a per-shell train/held-out split with the held-out side
capped at 200, and are **held-out**.

## Results

Success rate, mean over 12 seeds. Standard deviation across seeds in brackets.

| depth | eval | regionalized | monolithic | random floor | eval n |
|---|---|---|---|---|---|
| 1 | train-dist | **87.5%** (sd 14.4) | 87.5% (sd 17.6) | 20.8% (sd 14.4) | 6 |
| 2 | train-dist | **38.0%** (sd 24.5) | 30.6% (sd 23.8) | 4.3% (sd 4.7) | 27 |
| 3 | held-out | 2.2% (sd 5.9) | 0.8% (sd 2.1) | 1.4% (sd 3.0) | 30 |
| 4 | held-out | 0.0% | 0.0% | 0.3% | 133 |
| 5 | held-out | 0.0% | 0.0% | 0.0% | 200 |
| 6 | held-out | 0.0% | 0.0% | 0.1% | 200 |

**The knee is between depth 2 and depth 3.**

## Pre-registered contract, claim by claim

**1. "Chance is measured, not assumed." CONFIRMED and it mattered.**
The measured floor is **20.8%** at depth 1, not the 1/6 = 16.7% that a naive reading gives, because a
random walk with a `2d+3` step budget can stumble into solved. Every near-chance statement below is made
against the measured curve.

**2. "Solid success at depth 1, sharp degradation at depth 2, near-chance by depth 3." CONFIRMED.**
87.5% against a 20.8% floor, then 38.0% against 4.3%, then 2.2% against 1.4%. The predicted shape held
exactly.

**3. "A weak depth-1 result would indict the encoding or the training setup, not the architecture."
NOT TRIGGERED.** Depth 1 came in strong, so the encoding and harness are sound and the collapse is a real
capability limit rather than a broken setup. This was the pre-registered escape hatch and it was not needed.

**4. "The regionalization comparison is paired per seed." DONE, and the answer is NULL.**

| depth | mean paired diff | sd | reg wins | mono wins | ties |
|---|---|---|---|---|---|
| 1 | +0.0 pts | 15.9 | 3 | 4 | 5 |
| 2 | +7.4 pts | 38.1 | 7 | 5 | 0 |
| 3 | +1.4 pts | 6.6 | 2 | 2 | 8 |
| 4-6 | +0.0 pts | 0.0 | 0 | 0 | 12 |

The only non-zero cell is depth 2 at +7.4 points, but with a standard deviation of 38.1 across 12 seeds
and a 7-5 win split, **that is a coin flip, not an effect.** No regionalization claim is made.

**5. "A null result is a result." APPLIED.** See claim 4 and the section below.

**6. "A monolithic win would be confounded with policy-path width." NOT TRIGGERED, but the confound
stands.** With `recall=False`, 318 of the regionalized arm's 510 neurons are off the policy path, so the
comparison is a 128-wide frozen stack against a 446-wide one (26,816 vs 93,278 policy-path parameters).
Neither arm won, so the confound did not have to be adjudicated. It remains the reason this experiment
cannot answer the regionalization question at all.

**7. "The depth-1 cell is optimistically biased." CONFIRMED and disclosed.** Depth 1 is selected by max
mean success over the swept sigmas on the same data it reports. The winner reads 87.5%; the mean across
all three swept sigmas is **81%** for both arms. Depths 2 to 6 use a fixed pre-selected sigma and are
unbiased.

## The finding worth carrying forward: noise regularization did not transfer

Both arms selected **sigma = 0.0**. EXP-028's headline result on the grid was that Gaussian concept noise
at sigma 0.4 roughly doubled held-out navigation (43% to 83%). On the cube, no noise beat no noise, for
either topology.

This is precisely why the design swept sigma on the cube rather than importing the grid-tuned constant.
Had 0.4 been imported as "best-v1", both arms would have run handicapped and the collapse curve would
have been pessimistic for the wrong reason.

## Secondary observations

- **Beyond depth 2 the trained policy is indistinguishable from chance**, and at depths 3 and 4 it is
  nominally *below* the measured floor (regionalized 2.2% vs random 1.4% at depth 3; 0.0% vs 0.3% at
  depth 4). All of these are within noise of zero; the honest reading is that training buys nothing past
  the knee, not that it actively hurts.
- **Optimality at depth 1 is 0.565 (regionalized) against 0.843 for the random arm.** The random arm looks
  more "optimal" only because it succeeds almost exclusively when it stumbles onto the solution
  immediately; conditioning on success selects its short episodes. Mean steps for a 1-step problem were
  1.81 (regionalized), 1.67 (monolithic), 1.37 (random).
- **Per-seed variance is large** at depths 1 and 2 (sd 14 to 25 points), consistent with EXP-028's grid
  experience. n=12 is doing real work here; n=5 would have been noise.

## Verdict

> [!success] v1 solves the cube reactively at distance 1, degrades sharply at 2, and is at chance by 3.
> The encoding and harness are sound (87.5% vs a 20.8% floor at depth 1), so the collapse at depth 3 is a
> genuine capability limit of a frozen random encoder plus a linear head. **The regionalization question
> is not answered and cannot be answered by this design**: with `recall=False` only the sensory region is
> on the policy path, so the two arms differ in width (128 vs 446) rather than in topology, and the paired
> comparison is null in any case. EXP-028's noise regularization did not transfer to the cube.

**Lead for EXP-030:** the informative depths are 1 to 3. Depths 4 and beyond sit on the random floor,
where no intervention can show an effect. Engagement (putting the deferred regions on the policy path) is
the prerequisite for any regionalization claim.

## Regenerate

```powershell
.venv\Scripts\python.exe experiments\029_cube_baseline\run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16
.venv\Scripts\python.exe experiments\029_cube_baseline\aggregate.py
```
