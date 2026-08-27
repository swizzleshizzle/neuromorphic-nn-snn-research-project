# EXP-052 Results - pretraining saturates early, and its own metric cannot tell you that

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract was
> committed at `3fcf21e` **before any number existed**, including the single-comparison rule and
> the reading of each curve shape. No threshold was edited while filling this in. **One
> post-run change was made to `aggregate.py`'s interpreter and it is described below**, because it
> was correcting an over-read of my own.
>
> **Provenance:** 24 records, 24 encoders. Phase 1: from-scratch inverse-model pretraining at 10
> and 20 epochs, 12 seeds each, EXP-040's config and `rl_heldout_union` exclusions unchanged, 10
> workers, 1.3 h. Phase 2: depth 6, 10,000 episodes, each encoder **frozen** (390 trainable),
> curriculum `(1..6)`, `max_steps_by_depth=((1,2),)`, 6 workers. Laptop `SwizzlesDuo`,
> 2026-08-26 19:04 to 08-27 ~05:00. Zero tracebacks. The 40- and 80-epoch arms are EXP-043 and
> EXP-050 and were **not** re-run.

## Headline

**Pretraining saturates early. 10 epochs buys everything 40 does - and the pretraining
objective's own metric cannot tell you that, because it keeps improving all the way to 80.**

| epochs | move-accuracy | depth-6 policy | sd | |
|---|---|---|---|---|
| 0 | - | **0.0000** | - | EXP-036, random frozen encoder |
| **10** | 0.383 | **0.2012** | 0.0983 | |
| **20** | 0.414 | **0.1850** | 0.0899 | |
| 40 | 0.437 | **0.1800** | 0.0943 | EXP-043 - the inherited value |
| 80 | 0.452 | **0.0887** | 0.0842 | EXP-050 |

## Claim 1 (PRIMARY) - is 40 epochs past the optimum? REFUTED

Pre-registered single comparison, **20 minus 40**: **+0.0050**, W-L-T 7-5-0, **p 0.8418**.

The inherited 40 was **not** wrong. This closes a doubt that EXP-050 opened under every prior
result, which is what Claim 5 pre-committed as the value of a null here.

Exploratory, carrying no bar: **10 minus 40 = +0.0213, p 0.4902.**

## Claim 2 - the shape, read through significance

| comparison | delta | p |
|---|---|---|
| 10 vs 20 | +0.0162 | 0.6909 |
| 10 vs 40 | +0.0213 | 0.4902 |
| 20 vs 40 | +0.0050 | 0.8418 |
| **40 vs 80** | **+0.0912** | **0.0078** |

**A plateau from 10 to 40, then a collapse by 80.** None of 10/20/40 differ from each other; only
40-vs-80 does.

So the useful statement is **not** "fewer epochs is better" - it is that **pretraining saturates
early for policy purposes**. Ten epochs buys what forty does at **a quarter of the cost**, and
there is **no free performance sitting there**. No baseline in the series moves.

> [!warning] A correction to my own aggregator, made before this file was written
> The first version of `aggregate.py` declared *"MONOTONE DECREASING - the peak is BELOW 10
> epochs - this substantially weakens the EXP-039/040 premise"*, purely from the **ordering of
> four means**. Three of those four are pairwise indistinguishable at p 0.49-0.84. It was ranking
> noise and calling it a finding - the exact over-read this project has refused since EXP-037,
> arriving from a direction the threshold discipline does not cover, because **the thresholds
> were all obeyed**. The interpreter now requires significance before naming a shape. No bar
> moved.

## Claim 3 - does the pretraining metric predict policy? CONFIRMED: NO

| | 10 | 20 | 40 | 80 |
|---|---|---|---|---|
| move-accuracy | 0.383 | 0.414 | 0.437 | **0.452** |
| depth-6 policy | **0.2012** | 0.1850 | 0.1800 | 0.0887 |

**Move-accuracy is monotone increasing across the whole range and is tight within each arm**
(spread ~0.009 over 12 seeds). It says 80 epochs is the best encoder. **80 is the worst measured
by a factor of two.**

**You cannot tune pretraining length by watching pretraining.** That matters practically: the
move-accuracy is the *only* signal available without paying for a full RL arm, and it is
anti-informative for the choice it looks like it should inform.

This is the same shape as the probe finding (EXP-049/050), now in a third instrument: **metrics
that improve on the pretext task move against downstream policy quality.**

## Claim 4 - trajectory metrics. Suggestive, NOT significant.

| | 10 | 20 | 40 | 80 |
|---|---|---|---|---|
| `eval_revisit_rate` | 0.4300 | 0.4286 | 0.4652 | 0.4745 |
| `optimality` | 0.7068 | 0.6644 | 0.6445 | 0.5779 |

Both trend monotonically with epochs, in the direction the sequence-blindness hypothesis predicts:
more pretraining -> more revisiting, less optimal paths. **But none of it is significant** -
optimality 10-vs-40 is p 0.1973 and 10-vs-80 is p 0.2041.

Four points trending together is more than EXP-050 had, and it is still not evidence. Recorded as
the same unproven mechanism, now with a fourth data point pointing the same way.

**The probe was deliberately not run** (EXP-049/050: it moves opposite to policy quality here).

## What this does and does not license

**Licensed:**
- "Pretraining saturates by 10 epochs for policy purposes; 10, 20 and 40 are indistinguishable."
- "Pretraining cost can be cut 4x with no measured loss."
- "The pretraining objective's own metric is anti-informative for choosing its own duration."
- "The inherited 40-epoch configuration was not a hidden error."

**NOT licensed:**
- "Fewer epochs is better." 10 > 40 by +0.0213 at p 0.49. That is noise.
- "The peak is below 10." Untested - the lowest measured point is 10, and 0 gives 0.0000.
- "Sequence-blindness is established." Four consistent trends, all non-significant.

## What to do next

1. **Cut pretraining to 10 epochs for future work.** Free 4x on every encoder build, no measured
   cost. Not urgent, since it changes no result - purely an efficiency win.
2. **The plateau's left edge is unmeasured.** Between 0 (0.0000) and 10 (0.2012) the curve rises
   steeply and nobody has looked. If 3 epochs also gives ~0.20, pretraining is doing far less work
   than the EXP-039/040 story implies, and that *would* be a substantive finding rather than an
   efficiency note.
3. **Stop treating pretext metrics as progress indicators.** Two instruments now - the probe and
   move-accuracy - improve while policy degrades. Any future auxiliary objective needs a policy
   measurement, not a proxy.

## Regenerate

```bash
.venv/bin/python -u experiments/052_pretraining_optimum/pretrain_sweep.py --workers 10
.venv/bin/python -u experiments/052_pretraining_optimum/run.py --workers 6
.venv/bin/python experiments/052_pretraining_optimum/aggregate.py
```
