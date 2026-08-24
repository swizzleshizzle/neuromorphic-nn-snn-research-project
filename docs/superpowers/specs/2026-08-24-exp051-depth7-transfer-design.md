# EXP-051 design - does the encoder gain transfer to the frontier?

> **PRE-REGISTERED. Committed before any number exists.** Thresholds fixed at commit time.
> **Queued to dispatch when EXP-050 finishes**; it does not depend on EXP-050's outcome.

## 1. The gap this closes

**Every result in weeks 19 and 20 is measured at depth 6.** The goal is the full 2x2, where a
random scramble sits at **depth 11**. Nothing yet says the encoder work helps anywhere else.

There is a sharper reason to doubt it than "untested". **E1 was fine-tuned on the curriculum
`(1..6)`. Depth 7 was never in its training distribution.** So this is a genuine
out-of-distribution generalisation test, not a formality, and a refutation would mean the gain is
**depth-specific** - which would bound everything EXP-047/048/049 established to the depth they
happened to be run at.

## 2. Design

**One arm.** The baseline exists and is not re-run: EXP-044 arm A, depth 7 at 10,000 episodes on
the E0 encoder, **0.0621** across the same 12 seeds (verified locally, matching its published
value exactly).

| arm | encoder | during RL | head | depth | episodes | mean |
|---|---|---|---|---|---|---|
| EXP-044 A | E0 pretrained, frozen | frozen | fresh | 7 | 10,000 | **0.0621** |
| **EXP-051** | **E1 fine-tuned, frozen** | frozen | fresh | 7 | 10,000 | **?** |

**One variable: which frozen encoder.** Every other field is copied from EXP-044 arm A -
curriculum `(1..7)`, `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`,
`normalize_advantages=False`, `max_depth=7`, same seeds.

**E1, not E2, deliberately.** E1 is the encoder EXP-048 measured at depth 6, so this is the same
comparison moved to a new depth and is directly readable against EXP-048's **+0.1312**. Using the
better E2 would answer "how well can we do at depth 7" while confounding "did the gain transfer".

## 3. Claims

Paired per seed, exact permutation over `2**12`.

### Claim 1 - PRIMARY. Does it transfer? **EXP-051 minus EXP-044 arm A.**

CONFIRMED at **>= +0.05** with **p <= 0.05**, the standing bar.

### Claim 2 - THE POINT PREDICTION. Descriptive, stated in advance.

EXP-048 measured **+0.1312** at depth 6. **Complete transfer predicts `0.0621 + 0.1312` =
0.1933.**

Report the **transfer fraction** = `(observed - 0.0621) / 0.1312`. 1.0 is complete transfer,
0.0 is none.

### Claim 3 - THE HEADLINE COMPARISON. Does the encoder buy at the frontier what BUDGET buys?

**EXP-044 arm B reached 0.1971 at depth 7 - and it cost 4.4x the episodes.** Complete transfer
predicts **0.1933 at 1x the episodes**.

If those land together, the encoder work buys at the frontier exactly what 4.4x budget buys, for
**a quarter of the episodes**. That is the single most useful sentence this line could produce.

### Claim 4 - CONSISTENCY. Does the constant-return model hold across depth?

The depth-7 budget rate, back-solved from EXP-044's own two arms, is **0.210 per log10**
(`(0.1971 - 0.0621) / log10(4.4)`) - essentially depth 6's measured **0.22**. So this arm's
budget-equivalent, carrying E1's 1.33 units plus its own 1.0, is:

```
0.0621 + 0.210 * log10(2.33) = 0.1392
```

**Complete transfer would therefore show an excess of +0.0541** - against the **+0.0628, +0.0504,
+0.0540** measured at depth 6 for arms C, B and E. If the observed excess lands near +0.05 again,
the constant-return model holds **across depth as well as across rounds**, and the cost of any
target at any depth becomes calculable. Descriptive; four points is a trend, not a law.

### Claim 5 - MECHANISM, using the instrument that replaced the probe.

EXP-049 retired the probe as a policy predictor and put `eval_revisit_rate` and `optimality` in
its place. **Predicted: revisits lower and optimality higher than EXP-044 arm A**, as at depth 6
(0.4652 -> 0.3808 and 0.6445 -> 0.7716).

**The probe is deliberately NOT run here.** It has been shown to move opposite to policy quality
over this encoder sequence; running it would add a number nobody should use.

### Claim 6 - the null is a real and important result

A refuted Claim 1 means the encoder improvement is **depth-specific** and does not generalise
beyond the shells it was fine-tuned on. That would bound EXP-047/048/049 to depth 6, make the
two-stage recipe far less interesting, and redirect week 21 toward **why** - most likely that the
encoder is fitting shell-specific structure rather than cube structure, which the curriculum's
`(1..6)` cap would explain.

## 4. Execution

12 cells, one arm, depth 7. 105,672 training steps plus 6,800 evaluation steps per cell.

**Six workers** - two clean waves at the measured 0.115 s/step:

```
ceil(12/6) * 112,472 * 0.115 s = 2 * 12,934 s = 7.2 h
```

Dispatch **after EXP-050 completes**; the two must not share the machine.
