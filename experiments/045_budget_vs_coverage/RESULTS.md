# EXP-045 Results - it was TOTAL BUDGET. Back-loading a fixed budget makes depth 7 much worse.

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract and
> `aggregate.py` were both committed at `d8958ec`, **before the run was dispatched**. No threshold
> was edited.
>
> **Provenance:** 12 records + 12 head checkpoints. Depth 7, **10,000 episodes** (arm A's budget),
> `curriculum_weights=(1,1,1,1,1,1,10)`, EXP-040 pretrained encoders, `max_steps_by_depth=((1,2),)`,
> `entropy_beta=0.0`, `heldout_cap=200`, seeds 0-11. Dispatched 2026-08-17 02:45 UTC on the laptop
> `SwizzlesDuo`, `--workers 12`, finished 11:28 UTC, **8.7 h**, zero tracebacks. Paired baseline is
> **EXP-044 arm A**, not re-run.

## Headline

**H-coverage is refuted, and not by a null - by a significant loss.**

| | mean | sd | zeros | modal action frac |
|---|---|---|---|---|
| EXP-044 arm A (uniform, 10k) | 0.0621 | 0.0551 | 1 / 12 | 0.561 |
| **EXP-045 (back-loaded, 10k)** | **0.0142** | **0.0320** | **9 / 12** | **0.847** |
| EXP-044 arm B (uniform, 44k) | 0.1971 | 0.0428 | 0 / 12 | - |

Giving stage 7 arm B's episode count **within arm A's budget** did not recover arm B's gain. It
**destroyed** what arm A had: paired delta **-0.0479**, **W-L-T 0-11-1**, exact **p 0.0010**.

**The operative variable in EXP-044 arm B was TOTAL BUDGET.** Deepest-shell coverage is not it, and
raising coverage at fixed budget is actively harmful.

## Claim 1 (PRIMARY) - does deepest-shell coverage reproduce arm B's gain? REFUTED

Pre-registered: **>= +0.05** at **p <= 0.05** would confirm H-coverage.

| | value |
|---|---|
| paired delta | **-0.0479** |
| W-L-T | **0-11-1** |
| exact p (2^12 flips) | **0.0010** |
| share of arm B's +0.1350 recovered | **-35%** |

Eleven of twelve seeds got worse and the twelfth was already at zero. **Not a single seed
improved.** The pre-registration allowed for a positive-but-under-bar result and said to report
that as "refuted without showing the effect is zero" - that escape hatch is not needed. The sign
is wrong and the p-value is small.

## Claim 2 - does it clear the working bar? NO, by a mile

mean 0.0142 against 0.10, margin **-9.30 SE**, **0 of 12** seeds above. Depth 7 back-loaded is the
worst depth-7 cell measured, worse than the starved arm A.

## Claim 3 - MECHANISM: it hurt by driving collapse, exactly as EXP-037 saw

| | EXP-044 arm A | **EXP-045** |
|---|---|---|
| modal action fraction | 0.561 | **0.847** |
| eval revisit rate | 0.4806 | **0.6920** |
| mean train entropy | 0.5120 | **0.3810** |
| mean steps (on solves) | 9.25 | **2.56** |

And inside the long stage-7 block itself, from `stage_trace`:

| | value |
|---|---|
| entropy, first 10% of the stage | 0.5914 |
| entropy, last 10% | **0.0979** |
| entropy minimum | **2.7e-06** |
| fraction of training episodes solved | **0.0218** |

**The policy collapsed during the very stage that was supposed to help it.** 6,250 consecutive
episodes at a depth it solves 2.2% of the time is 6,250 episodes of almost no reward signal;
entropy fell to effectively zero, the modal action rose to 0.847, and the revisit rate - the rate
at which the greedy policy walks back into a state it has already visited - went to 0.69.

EXP-037 measured the same signature at depth 4 with a frozen random encoder: modal 0.685 -> 0.757
as the share at the evaluated depth rose 25% -> 75%. **That finding now generalises** to depth 7,
a pretrained encoder, and the depth-1 cap, and the mechanism reproduces with it.

## Claim 4 - failure counts, DESCRIPTIVE, no p-value

**9 of 12** seeds at exactly 0.000, against arm A's 1 of 12. Reported as a count, no test:
**n=12 cannot show a count went to zero** and it cannot show one arrived there either.

## Claim 5 - what this means. The pre-committed reading, unchanged

1. **The operative variable is total budget.** EXP-044's gain came from 4.4x more training
   everywhere, not from the deepest shell specifically.
2. **The word "coverage" leaves EXP-044's write-up.** Done, at the same time as this file.
3. **Every "depth N stopped working" result in this project is confounded with episodes.** All of
   them were taken at a fixed 10,000, and depth 7 shows a depth can look broken purely for want of
   budget. That does not invalidate them; it means they measure *depth at 10,000 episodes*.
4. **The next experiment is depth 6 at raised total budget**, not anything weighted.

## The thing worth taking away, beyond the claim

**The shallow curriculum stages are not warm-up. They are what keeps the policy from collapsing.**

Halving them (1,428 -> 625 episodes each) while quadrupling the deep stage produced the most
collapsed policy in the depth-7 series. The curriculum's job is not to introduce difficulty
gradually - it is to keep the reward signal dense enough that entropy survives. A long stint at a
depth the policy almost never solves is not training; it is a slow collapse with a reward of
approximately zero.

That also explains why EXP-042's depth-1 **cap** worked and why deleting the depth-1 stage did not:
the shallow stages must be present and cheap, not absent and not dominant.

## Limitations

- **One weighting** (10x on the last stage). The decline may not be monotone all the way, though
  EXP-037 found it monotone across 25/50/75% at depth 4.
- **Total budget and per-stage episodes are still not fully separated.** This arm shows that moving
  budget *toward* the deep end hurts; it does not isolate which shallow stage matters, or whether
  the shallow stages help by training or merely by keeping entropy alive.
- **A budget-matched control at the shallow end is untested**: nothing here tries 10,000 episodes
  weighted *toward* the shallow stages, which EXP-037's 12.5% arm hints would be roughly neutral.
- Depths 1-7 are **0.9%** of the state space; a random scramble sits at **depth 11**.

## Lead for the next experiment

1. **Depth 6 at raised total budget** (~51,000 episodes for depth-5-like exposure, ~25 h). If it
   moves the way depth 7 did, the whole depth series is a budget series and every published number
   here needs restating as "at 10,000 episodes".
2. **Fine-tune the encoder during RL** - still untested, and now the only untried lever that is not
   about budget.
3. **Re-run depth 5 with 24+ seeds** to settle EXP-043's Claim 1 (+0.1108 at p 0.0815).

**Still refuted and CLOSED:** width (EXP-033), volume alone (EXP-034), curriculum stage weighting
(EXP-037, now **confirmed to generalise** by this experiment), starvation at depth 6 (EXP-037),
trainer stabilizers (EXP-038), deleting the depth-1 stage (EXP-042), **deepest-shell coverage as
the explanation for EXP-044 arm B (this experiment)**.

## Regenerate

```bash
.venv/bin/python -u experiments/045_budget_vs_coverage/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 12 --skip-existing
.venv/bin/python experiments/045_budget_vs_coverage/aggregate.py
# EXP-044 arm A's records are the paired baseline and MUST be present; run.py refuses without them.
# Add --dry-run to inspect the pre-flight banner without starting anything.
```
