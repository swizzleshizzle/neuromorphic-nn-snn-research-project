# EXP-050 Results - the objective is what matters, and more pretraining is actively harmful

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract was
> committed at `f569540` **before any number existed**, `aggregate.py` was written before dispatch,
> and the third control asymmetry was added to the spec **from the smoke test, before the run**.
> No threshold was edited while filling this in.
>
> **Provenance:** 12 arm-F records plus 12 E0+ encoders. Phase 1: 40 more inverse-model epochs
> warm-started from E0, EXP-040's config and held-out exclusions unchanged, 10,360 encoder updates
> per seed, 10 workers, 2.49 h. Phase 2: depth 6, 10,000 episodes, E0+ **frozen**
> (`encoder_lr=None`, 390 trainable), curriculum `(1..6)`, `max_steps_by_depth=((1,2),)`, 6
> workers. Laptop `SwizzlesDuo`, 2026-08-24 19:02 to 08-25 13:30. Zero tracebacks. Arms A
> (EXP-043) and B (EXP-048) not re-run.

## Headline

**More pretraining does not merely fail to help. It halves the policy.**

| arm | encoder | how it was made | mean | sd |
|---|---|---|---|---|
| A (EXP-043) | E0 | 40 epochs inverse-model | **0.1800** | 0.0943 |
| **F** | **E0+** | **80 epochs inverse-model** | **0.0887** | 0.0842 |
| B (EXP-048) | E1 | 40 epochs + 10,000 RL updates | **0.3112** | 0.1091 |

All three have **identical RL compute**. The only difference is how the encoder's extra ~10,000
gradient steps were spent.

- **Claim 1 REFUTED**: `F - A` = **-0.0912**, W-L-T 3-9-0, **p 0.0078**.
- **Claim 2**: `F - B` = **-0.2225**, W-L-T 1-11-0, **p 0.0010**.
- **Claim 3, the pre-registered grid: RL'S OBJECTIVE IS WHAT MATTERS.** The "more gradient"
  alternative is **closed**, and EXP-047/048/049 can lift their scope caveats rather than restate
  them.

**The control was favoured on four axes and still lost.** Step-matched (10,360 against RL's
10,000), **17x more data per update**, a clean supervised gradient instead of a noisy policy
gradient, and it moved the encoder **12.7x further** (37.0% of `|fc1|` against RL's 2.91%). Every
one of those advantages was stated in the spec before the run.

## This was not a no-op control - the pretraining objective genuinely improved

Move-naming accuracy rose on **all 12 seeds**, from EXP-040's 0.436-0.438 to **0.448-0.455**.
EXP-040's RESULTS listed *"longer pretraining, which is cheap and whose objective had not
saturated"* as a promising open item. **It is cheap, it had not saturated, and it makes the policy
worse.** Worth having tested rather than assumed.

So the finding is sharper than the grid cell it landed in. The grid's null was *"more pretraining
does nothing"*. What happened is that **more of the objective that created the good encoder
destroys it for policy use, while getting better at its own task.** That is proxy-objective
divergence: past roughly 40 epochs, optimising move-naming discards whatever the policy needs.

## Claim 4 - THE PROBE IS ANTI-CORRELATED WITH POLICY IN BOTH DIRECTIONS

Depth-6 probe, E0 against E0+:

| depth | E0 | E0+ | delta | W-L | p |
|---|---|---|---|---|---|
| 4 | 0.7998 | 0.7910 | -0.0087 | 5-6 | 0.4307 |
| **6** | **0.6839** | **0.6996** | **+0.0157** | **12-0** | **0.0005** |

Set that beside EXP-049, at the same depth, on the same instrument:

| intervention | depth-6 probe | | policy |
|---|---|---|---|
| RL fine-tuning (EXP-049) | 0.6742 -> 0.6554 | **0-12**, p 0.0005 | 0.3112 -> 0.3525 **up** |
| more pretraining (EXP-050) | 0.6839 -> 0.6996 | **12-0**, p 0.0005 | 0.1800 -> 0.0887 **down** |

**Two different training objectives, opposite probe movements, both unanimous, both at p 0.0005 -
and in both cases the probe moved OPPOSITE to policy quality.**

This is no longer "the probe is a weak predictor" or "the probe missed it". At depth 6 the probe
is a **reliable inverse indicator** of policy quality across two unrelated interventions.

### My pre-registered inference here was badly specified, and would have licensed a wrong answer

The spec said:

> *"Pre-registered prediction: E0+ probes HIGHER than E0, while arm F's policy gain is smaller
> than arm B's. **If both hold, the anti-correlation is a property of the RL objective**, not of
> gradient descent on the encoder in general."*

**Both stated conditions held.** E0+ does probe higher at depth 6, and arm F's gain is smaller
than arm B's. **The conclusion is nevertheless wrong**, because I wrote "gain is smaller" without
considering that arm F might post a *loss*. With a loss, probe-up-policy-down is the
anti-correlation appearing **again**, which is the opposite of evidence that it is RL-specific.

A pre-registered inference can be satisfied on its stated terms and still be invalid, if the terms
were carved without considering an outcome. The threshold discipline this project keeps does not
protect against that; only checking the *reasoning* against the actual numbers does. Recorded
rather than quietly re-derived, because the spec is committed.

## The mechanism this suggests - coherent, NOT established

The inverse model is trained to name the move between `s` and `s'`. That is a **purely
single-step** objective. Over-training it plausibly makes the code single-step-optimal and
**sequence-blind** - and EXP-049 located the policy gain precisely in sequences, in not undoing
one's own work.

That predicts arm F should revisit *more* and be *less* optimal than arm A:

| metric | A (E0) | F (E0+) | delta | p |
|---|---|---|---|---|
| `eval_revisit_rate` | 0.4652 | 0.4745 | +0.0093 | 0.8433 |
| `optimality` | 0.6445 | 0.5779 | -0.0666 | 0.5786 |

**Both move as predicted; neither is significant.** `optimality` is scored only over solved
episodes, and arm F solves 8.9% of the time, so it has very little to average. **The mechanism is
consistent with the data and is not demonstrated by it.** Stated that way deliberately: this is
the point in a story where a plausible mechanism gets promoted to a finding without earning it.

## What this does and does not license

**Licensed:**
- "The encoder improvement in EXP-047/048/049 is attributable to the RL objective, not to
  additional gradient steps. A control favoured on four axes lost by 0.22."
- "Doubling inverse-model pretraining halves depth-6 policy success, while improving the
  pretraining task itself."
- "At depth 6 the decodability probe moves opposite to policy quality under two different
  objectives, unanimously in both."

**NOT licensed:**
- "Pretraining is bad." 40 epochs is what makes any of this work; **80 is worse than 40**. There
  is an optimum and it has not been located - only two points have been measured.
- "The probe is broken." It measures what it always measured. What is established is that its
  delta is anti-predictive of policy quality here.
- "Sequence-blindness is the mechanism." Directionally consistent, p 0.84 and p 0.58.

## What to do next

1. **Locate the pretraining optimum.** Only 40 and 80 epochs exist, and 40 was never chosen by
   measurement - EXP-039 calibrated the learning *rate*, not the epoch count. If 20 beats 40, a
   substantial part of the frozen-encoder era was leaving performance on the table for free, and
   this is the cheapest experiment in the queue.
2. **Stop reporting probe deltas as evidence about policy.** Three experiments now read one that
   way. Their numbers stand; the inferences need `revisit_rate` and `optimality` beside them.
3. **Test sequence-blindness properly** with a metric that does not depend on solving - for
   example, whether the encoder's concept distinguishes `s` from a state two moves away versus
   one move away.

## Regenerate

```bash
.venv/bin/python -u experiments/050_objective_vs_gradient/extend_pretrain.py --workers 10
.venv/bin/python -u experiments/050_objective_vs_gradient/run.py --workers 6
.venv/bin/python experiments/050_objective_vs_gradient/aggregate.py
.venv/bin/python experiments/050_objective_vs_gradient/probe_e0plus.py     # Claim 4
```
