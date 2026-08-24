# EXP-049 Results - constant returns, as predicted, and the probe now moves the wrong way twice

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract was
> committed at `aaca17c` **before any number existed**, `aggregate.py` was written before dispatch,
> and **the primary claim was predicted to refute in the spec itself**. No threshold was edited
> while filling this in.
>
> **Provenance:** 24 records (12 arm D + 12 arm E), 12 twice-fine-tuned encoders. Depth 6, 10,000
> episodes per arm, curriculum `(1..6)`, `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`,
> `normalize_advantages=False`. Arm D starts from EXP-047's encoders at `encoder_lr=1e-4`
> (**reused, deliberately not re-piloted**); arm E freezes arm D's output, `encoder_lr=None`.
> Laptop `SwizzlesDuo`, 6 workers. Arm D 2026-08-23 22:01 to 08-24 04:28 (6.4 h), arm E 04:28 to
> 08:21 (3.9 h); **10.3 h against a 15 h estimate.** Zero tracebacks. Arms A, B and C were not
> re-run. Records in `outputs/` (gitignored); `*_head.pt` and `*_encoder.pt` tracked.

## Headline

**The recipe does not compound. Each round buys a fixed increment over its own compute.**

| arm | encoder | head | compute | budget-equiv | actual | **excess** |
|---|---|---|---|---|---|---|
| A (EXP-043) | E0, frozen | fresh | 1.00 | 0.1800 | 0.1800 | - |
| C (EXP-047) | E1, joint | co-adapted | 1.33 | 0.2072 | 0.2700 | +0.0628 |
| B (EXP-048) | E1, frozen | fresh | 2.33 | 0.2608 | 0.3112 | +0.0504 |
| **D** | **E2, joint** | co-adapted | 2.66 | 0.2735 | **0.3525** | **+0.0790** |
| E | E2, frozen | fresh | 3.66 | 0.3040 | 0.3579 | +0.0540 |

**Claim 2, the quantity the spec said would carry the answer: +0.0628, +0.0504, +0.0540.** Flat.

So **iterating is a better way to spend compute, not an escape from the budget wall EXP-046
priced.** The cost of a target score is now calculable in advance, and "just iterate it" is
closed as a strategy.

## Claim 1 (PRIMARY) - landed exactly in the band written for it

**E - B = +0.0467, W-L-T 8-4-0, exact p 0.2456.**

The spec pre-registered `>= +0.05 at p <= 0.05` **and predicted refutation**, on the grounds that
a constant-return model puts `E - B` near **+0.043**. The observed +0.0467 sits between the
**+0.0432** the extra round's compute alone buys and the +0.05 bar.

> **VERDICT: UNINTERPRETABLE**, which is a pre-registered verdict and not a near-miss. The band
> was written into the spec, the driver banner and the aggregator before the run, precisely
> because the bar sat only 1.16x above the confound. Reporting +0.0467 as "nearly confirmed"
> would be the move this project has refused since EXP-037.

## Claim 3 - a second round DOES help the joint arm. **D - C = +0.0825, 11-1-0, p 0.0020.**

Arm D was only run to produce E2; its score is the strongest single result here.

## The recipe EXP-048 recommended is wrong, and this experiment retires it

EXP-048 proposed a **two-stage recipe**: fine-tune, then throw the head away and retrain it. That
was based on `B - C = +0.0412`. It does not survive round 2.

| | fresh-head gain | W-L | p |
|---|---|---|---|
| round 1 (B - C) | +0.0412 | 9-2 | 0.2393 |
| round 2 (E - D) | **+0.0054** | 6-6 | 0.6816 |

A fresh-head stage costs **1.0 compute unit**, whose budget-equivalent at this point on the curve
is **+0.0305**. In round 2 it returned **+0.0054** - it **lost ground against simply spending that
compute on episodes**.

**Arm D has the highest excess in the whole series (+0.0790).** The best recipe found is
therefore *fine-tune twice and keep the co-adapted head*, at 2.66 units - not the 3.66-unit
two-stage version.

> [!warning] **And my explanation for why fresh heads help was wrong.**
> EXP-048 argued the round-1 head was stale from *chasing a moving target*. That predicts the
> benefit scales with how far the encoder moved. It does not:
>
> | | relative `|delta fc1|` | fresh-head gain |
> |---|---|---|
> | round 1 (E0 -> E1) | 2.91% | +0.0412 |
> | round 2 (E1 -> E2) | 2.36% | +0.0054 |
>
> Nearly the same movement, an eightfold smaller benefit. **The moving-target mechanism does not
> hold.** The honest position is stronger than a replacement story: **neither fresh-head delta was
> ever significant** (p 0.2393 and p 0.6816), so the fresh-head stage was never established as
> valuable, and EXP-048 over-read a p 0.24 point estimate into a named recipe.

## Claim 5 - THE PROBE MOVES THE WRONG WAY, AND NOW IT IS UNANIMOUS

Depth-6 probe accuracy, same instrument, across both rounds:

| encoder | probe @ depth 6 | delta | W-L | p | best policy on it |
|---|---|---|---|---|---|
| E0 | 0.6839 | - | - | - | 0.1800 |
| E1 | 0.6742 | -0.0097 | 1-11 | - | 0.3112 |
| **E2** | **0.6554** | **-0.0188** | **0-12** | **0.0005** | **0.3525** |

**Every seed's probe got worse in round 2, and p 0.0005 is the floor of the exact test at n=12.**

Across two rounds the probe falls **monotonically** while policy success nearly doubles. This is
no longer a puzzle about one measurement - it is a **reproducible anti-correlation**, and it
promotes EXP-048's finding from "the probe missed it" to something sharper:

> **The linear probe on isolated states is not merely a weak predictor of policy quality on this
> task. Over this sequence it points the wrong way.**

That matters because the probe has been the mechanism instrument since EXP-033 and carried
EXP-039's headline. It measures what it always measured - how linearly decodable a
distance-reducing move is from an independently drawn state - but **a probe delta must not be
read as a policy prediction**, and anywhere one was, the inference needs redoing.

## Claim 4 - mechanism, directionally right and not significant

EXP-048 localised the round-1 gain to trajectories. Round 2:

| | B | E | delta | p | predicted |
|---|---|---|---|---|---|
| `eval_revisit_rate` | 0.3808 | 0.3586 | -0.0222 | 0.5972 | lower |
| `optimality` | 0.7716 | 0.7874 | +0.0159 | 0.5513 | higher |

Both moved as predicted; neither is established. Expected, since `E - B` is about a third of
`B - A`, so there is far less signal to detect. **Not evidence the mechanism failed, and not
evidence it held.**

## What this does and does not license

**Licensed:**
- "The recipe does not compound: each fine-tuning round buys about +0.05 over its own compute."
- "A second fine-tuning round helps the joint arm, +0.0825 at p 0.0020."
- "Depth 6 reaches 0.3525 at 2.66x compute, against EXP-046's 0.3225 at 4.4x."
- "The probe falls monotonically across two rounds while policy success rises, unanimously in
  round 2."

**NOT licensed:**
- "Throw the head away." Retired above; it lost to its own compute in round 2.
- "A third round will add another +0.05." The series is three points and Claim 2 is a trend, not
  a fitted law. Extrapolation is exactly what EXP-046 warned about.
- "The RL objective is what improved the encoder." Still uncontrolled, third experiment running.

## What to do next

1. **Stop iterating.** Claim 6 pre-committed this: with constant returns, a round 3 is priced at
   about +0.05 for 1.33 units and answers nothing new. **The next move is a different
   second-stage objective**, not another round.
2. **Retire the probe as a policy predictor, and say so where it was used.** EXP-033, EXP-039 and
   EXP-047 each read a probe delta as evidence about policy quality. Their probe numbers stand;
   the inferences drawn from them need a trajectory measurement beside them. `revisit_rate` and
   `optimality` have been in every record since EXP-029.
3. **Settle "objective or just more gradient"** - the control deferred twice now. It needs an
   honest exchange rate between RL episodes and pretraining epochs, which is the hard part and
   why it keeps being deferred. Continuing EXP-039's inverse-model pretraining for a matched
   number of *gradient steps* is the most defensible version.

## Regenerate

```bash
.venv/bin/python -u experiments/049_second_round/run.py --arm D --workers 6
.venv/bin/python -u experiments/049_second_round/run.py --arm E --workers 6   # needs D's encoders
.venv/bin/python experiments/049_second_round/aggregate.py
```
