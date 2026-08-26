# EXP-052 design - where is the pretraining optimum?

> **PRE-REGISTERED. Committed before any number exists.** Thresholds fixed at commit time.

## 1. Why this is the cheapest high-leverage experiment open

EXP-050 established that **doubling inverse-model pretraining halves depth-6 policy success**
(0.1800 -> 0.0887, p 0.0078) *while the pretraining task itself improved* (move-accuracy
0.436 -> 0.453). More of the objective that built the good encoder makes it worse for policy use.

That raises a question nobody has asked: **40 epochs was never chosen by measurement.** EXP-039
calibrated the *learning rate* (3e-3 against 1e-3, by the pretraining objective, explicitly not by
the probe), and recorded the epoch count as a fixed configuration. Every cube result since
EXP-040 - the whole frozen-encoder era, EXP-043's 0.1800, EXP-047/048/049's encoder line, and
EXP-051's frontier test - starts from a 40-epoch encoder that was never compared against any
other.

**If 20 epochs beats 40, every baseline in the series moves.**

## 2. The curve is already bracketed, so this only has to locate the peak

| epochs | depth-6 policy | source |
|---|---|---|
| 0 (random frozen encoder) | **0.0000** | EXP-036 |
| 40 | **0.1800** | EXP-043 / EXP-050 arm A |
| 80 (warm-started) | **0.0887** | EXP-050 arm F |

The curve **rises from 0 and falls by 80**, so an interior optimum exists. This is not a blind
sweep; it is a bisection with both ends already pinned.

## 3. Design

Two new arms. Everything is copied from EXP-043's depth-6 cell except the encoder, so all arms
have **identical RL compute** and the single variable is **how many epochs the encoder was
pretrained for**.

| arm | pretraining | status |
|---|---|---|
| 10 epochs | from scratch | **new** |
| **20 epochs** | from scratch | **new, PRIMARY** |
| 40 epochs | from scratch | exists (E0) |
| 80 epochs | warm-started | exists (E0+) |

**From scratch, not warm-started**, so each arm is "N epochs of pretraining" rather than "40 then
N-40 more with a fresh head".

### 20 epochs is the primary, and 10 is exploratory - to avoid multiplicity

Testing both against 40 and reporting whichever wins would inflate the false-positive rate.
**The single pre-registered comparison is 20 against 40.** 20 is the natural first probe: 40 -> 80
is a doubling that hurt, so 40 -> 20 is the matching halving. The 10-epoch arm is **exploratory
and carries no bar**; it exists to give the curve a fourth point and to say whether the peak is
below 20.

## 4. Claims

Paired per seed, exact permutation over `2**12`.

### Claim 1 - PRIMARY. Is 40 epochs past the optimum? **20 epochs minus 40 epochs.**

CONFIRMED at **>= +0.05** with **p <= 0.05**, the standing bar.

**A confirmed Claim 1 is a correction to the whole series, not a new capability.** It would mean
every frozen-encoder result since EXP-040 was run from a needlessly over-trained encoder.

### Claim 2 - THE CURVE. Descriptive, four points, no bar.

Report 10 / 20 / 40 / 80 with per-seed spread. Three shapes, and the reading of each is fixed
here:

| shape | reading |
|---|---|
| peak at 40 (both new arms below it) | **The inherited value was right.** EXP-050's result stands as "past the optimum", and nothing in the series moves. |
| peak at 20 | The series has been over-training by 2x. Re-baselining is cheap and everything shifts up. |
| **monotone decreasing (10 > 20 > 40)** | The peak is **below 10 epochs**, and the honest question becomes how much pretraining is doing at all. This would substantially weaken the EXP-039/040 premise, and is the outcome most worth being ready to report. |

### Claim 3 - Does the pretraining metric predict policy? **Predicted NO.**

Report final move-naming accuracy per arm beside policy success. EXP-050 already saw them
diverge - accuracy up, policy halved. **Prediction: move-accuracy rises monotonically with epochs
while policy peaks and falls.** If so, the pretraining objective's own metric is useless for
choosing how long to pretrain, which is worth knowing because it is the only signal available
without running the RL arm.

### Claim 4 - Trajectory metrics, the instrument that replaced the probe.

`eval_revisit_rate` and `optimality` across the four arms. **The probe is deliberately not run** -
EXP-049 and EXP-050 established it moves opposite to policy quality at this depth, unanimously in
both directions.

### Claim 5 - the null is a real result

A refuted Claim 1 with the peak at 40 means the inherited configuration was accidentally correct.
That is worth establishing: it converts an unexamined assumption into a measured one, and it
closes this question rather than leaving it as a standing doubt under every prior result.

## 5. What this does not resolve

**EXP-050's arm F was warm-started with a fresh inverse-model head**, so its "80 epochs" is not
80-from-scratch. If a randomly-initialised head damages the encoder in its first epochs, that -
rather than epoch count - could explain part of EXP-050's collapse. **This experiment does not
settle that**, and an 80-from-scratch arm would.

It does bear on it indirectly: if the curve declines monotonically from 20 upward through 40, the
decline needs no warm-start explanation.

## 6. Execution

| phase | what | workers | cost |
|---|---|---|---|
| 1 | pretrain 12 x 10 epochs and 12 x 20 epochs, from scratch | 10 | ~1.9 h |
| 2 | 24 RL cells, each encoder frozen + fresh head, depth 6 | 6 | ~7.8 h |

Total **~9.7 h**. Phase-1 cost scaled from EXP-050's measured 2.49 h for 40 epochs at 10 workers;
phase-2 from EXP-049 arm E's measured 3.9 h for 12 cells at 6 workers.
