# EXP-051 Results - the gain transfers, but its advantage over budget does not

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract was
> committed at `520d7ab` **before any number existed**, including the point prediction and the
> budget arithmetic. No threshold was edited while filling this in.
>
> **Provenance:** 12 records. Depth 7, 10,000 episodes, curriculum `(1..7)`,
> `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`, `normalize_advantages=False`, `max_depth=7`,
> EXP-047's E1 encoders **frozen** (`encoder_lr=None`, 390 trainable). Laptop `SwizzlesDuo`,
> 6 workers, 2026-08-25 13:35 to 19:51 (**6.3 h against a 7.2 h estimate**). Zero tracebacks.
> Baseline is EXP-044 arm A and was **not** re-run. **The probe was deliberately not run** - see
> below.

## Headline

**The encoder gain does transfer to a depth it was never trained on. But its advantage over
simply buying budget nearly vanishes there.**

| | mean | sd |
|---|---|---|
| EXP-044 arm A, E0 frozen, depth 7 | 0.0621 | 0.0528 |
| **EXP-051, E1 frozen, depth 7** | **0.1471** | 0.0750 |

**Claim 1 CONFIRMED: +0.0850, W-L-T 10-1-1, exact p 0.0039.**

E1 was fine-tuned on the curriculum `(1..6)`. **Depth 7 was never in its training distribution**,
and the gain still appears - so the encoder learned something about the cube, not about the
particular shells it saw. That was the real risk this experiment was built to test, and it is
retired.

**Then the arithmetic the spec fixed in advance turns the result around.**

## Claim 2 - the point prediction. Transfer is PARTIAL.

Complete transfer of EXP-048's depth-6 **+0.1312** would have predicted **0.1933**. Observed
**0.1471**, a **transfer fraction of 0.65**.

## Claim 3 - does the encoder buy at the frontier what budget buys? NO.

EXP-044 arm B reached **0.1971** at this depth **and it cost 4.4x the episodes**. Complete
transfer would have matched it at 1x. **0.1471 does not.**

## Claim 4 - THE FINDING. Constant returns do NOT hold across depth.

The spec fixed the budget-equivalent in advance: depth 7's rate back-solves to **0.210 per log10**
from EXP-044's own two arms, so this arm carrying E1's 1.33 units plus its own 1.0 is worth
`0.0621 + 0.210 * log10(2.33)` = **0.1392**.

| arm | depth | excess over budget-equivalent |
|---|---|---|
| C (EXP-047) | 6 | **+0.0628** |
| B (EXP-048) | 6 | **+0.0504** |
| E (EXP-049) | 6 | **+0.0540** |
| **EXP-051** | **7** | **+0.0079** |

**At depth 6 the encoder route beat the budget curve by a consistent ~+0.05. At depth 7 it beats
it by +0.008 - essentially nothing.**

The same fact as a compute ratio, which is the form to remember:

| depth | encoder route | same score by budget alone | **encoder route is** |
|---|---|---|---|
| 6 | 0.3112 at 2.33x | would need **3.95x** | **1.69x cheaper** |
| 7 | 0.1471 at 2.33x | would need **2.54x** | **1.09x cheaper** |

**At depth 7 the encoder pipeline is within 9% of break-even against just spending the compute.**

### This corrects week 20's closing claim

Week 20 closed on *"a lever that beats buying budget by about +0.05 per round"*. **That is a
depth-6 statement.** One depth further out it is +0.008, and the advantage erodes in exactly the
direction that matters - the goal is depth 11, not depth 6.

Two points do not fix a trend, and the erosion could be depth-7-specific rather than monotone.
But the direction is unambiguous and it was measured against a prediction fixed beforehand.

## Claim 5 - mechanism, and it does NOT reproduce here

EXP-049 located the depth-6 gain in trajectories: revisits fell (p 0.0454) and optimality rose
(p 0.0132).

| metric | arm A | EXP-051 | delta | p | predicted |
|---|---|---|---|---|---|
| `eval_revisit_rate` | 0.4806 | 0.4160 | -0.0647 | 0.1025 | lower |
| `optimality` | 0.6487 | 0.6440 | **-0.0047** | **0.9546** | higher |

**Revisits fall in the predicted direction but not significantly. Optimality is flat** - it was
predicted to rise, and it did not move at all.

So the trajectory mechanism that explained depth 6 **does not clearly operate at depth 7**, which
sits alongside the eroded budget advantage rather than contradicting it: whatever the encoder is
adding, it adds less here, and the instrument agrees.

**The probe was deliberately not run.** EXP-049 and EXP-050 between them established it moves
opposite to policy quality at depth 6 - down 0-12 while success doubled, up 12-0 while success
halved, both at p 0.0005. Running it would have produced a number nobody should use.

## What this does and does not license

**Licensed:**
- "The encoder gain generalises to an unseen depth: +0.0850 at depth 7, 10-1, p 0.0039."
- "Transfer is partial - 65% of the depth-6 gain."
- "The encoder route's advantage over buying budget falls from 1.69x cheaper at depth 6 to 1.09x
  at depth 7."

**NOT licensed:**
- "The encoder gain does not transfer." It does, significantly, on a depth it never saw.
- "The advantage decays monotonically with depth." Two depths. The direction is clear; the shape
  is not.
- "The trajectory mechanism is refuted." Not significant is not refuted, and depth 7 solves rarely
  enough that `optimality` has little to average.

## What to do next

1. **Decide whether the encoder line is still the frontier strategy.** At 1.09x cheaper it is
   nearly break-even, and EXP-049 already showed it does not compound. If the advantage really
   does erode with depth, the honest conclusion is that this line **buys a fixed, modest amount
   and then stops** - which makes the case for Stage 3's dense signal stronger, not weaker,
   because Wall 2 (sparse reward compounding with depth) is the thing that actually scales.
2. **Locate the pretraining optimum** (from EXP-050) - still the cheapest open experiment, and it
   moves the whole series if 20 epochs beats 40.
3. **Do not run depth 8 on this recipe** expecting a bargain. The exchange rate at depth 7 is
   already nearly the budget curve's.

## Regenerate

```bash
.venv/bin/python -u experiments/051_depth7_transfer/run.py --workers 6
.venv/bin/python experiments/051_depth7_transfer/aggregate.py
```
