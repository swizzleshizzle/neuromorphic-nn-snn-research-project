# EXP-066 design - can the brain's own pathway learn AT ALL? A `region_lr` sweep

> **PRE-REGISTERED. Committed before any EXP-066 number exists.** Thresholds fixed at commit time.
> **Date:** 2026-09-20 · **Phase:** 3 (post-checkpoint) · **Grounds:** EXP-064, EXP-043.

## 0. The question, and why it is asked WITHOUT a control

EXP-064 put prefrontal -> router -> motor on the policy path and trained it. **The mechanism
worked**: `region_drift` 2.7051, `motor_rate_mean` 0.1534, zero silent steps, gradient verified
to reach every region parameter. **The experiment still failed**: both arms scored exactly
0.0000, because the capacity-matched control collapsed (EXP-043's 0.3229 became 0.0000 on adding
`head_hidden=218`), and a contrast between two arms on the floor is 0 by construction.

**So this experiment does not use a control.** It asks a THRESHOLD question with a one-sample
reading, the same shape as EXP-062's depth-8 containment test:

> **Can the brain's own pathway learn a policy at all, at any learning rate?**

==That reframing is the entire design point.== There is no control arm, therefore no control can
break and zero out the comparison. EXP-064's failure mode is unreachable here.

**The hypothesis being tested** comes from EXP-064's own results file, where it is recorded as a
hypothesis and not a finding: `region_drift` of **2.71** means the regions moved nearly three
times their own initial magnitude, which is consistent with `region_lr=1e-2` being too large and
destroying the pathway rather than training it.

## 1. The arms

12 seeds each (0-11), depth 5, EXP-064 arm P's configuration field for field, frozen EXP-040 E0
encoder. **The only variable is `region_lr`.**

| arm | `region_lr` | tag | status |
|---|---|---|---|
| **L4** | **1e-4** | `exp066_motor_lr1e4` | NEW |
| **L3** | **1e-3** | `exp066_motor_lr1e3` | NEW |
| L2 | 1e-2 | `exp064_motor_d5` | **REUSED from EXP-064**, already measured at **0.0000** |

**Three points across three orders of magnitude, one of them free.** The reuse is legitimate
without an inertness proof because EXP-066 adds **no code at all**: `readout="motor"` and
`region_lr` shipped in EXP-064 and are unchanged. Only a configuration differs.

## 2. The reference, which is NOT a control

**EXP-043 ran this same configuration with a linear head at depth 5 and scored 0.3229 across
seeds 0-23.** Its config differs from EXP-064's motor arm only in `readout`, `region_lr`, a BFS
table bound (`max_depth` 6 vs 5, an instrument bound that cannot affect training), and three
fields that did not exist when it ran (`constant_critic`, `flatten_critic`, `gate_rate_by_seed`,
all inert).

==It is used as **context**, never as a control.== Its trainable surface is **390** against the
motor arm's **15,540**, a 40x difference. Any comparison against it is **descriptive** and is
reported as such. Calling it a control would repeat EXP-063's mistake in the other direction.

## 3. Claims

### Claim 1, PRIMARY - can the pathway learn? A one-sample threshold, per arm.

For each new arm independently:

- **CONFIRMED** at **mean held-out success `>= 0.02`** AND **the 95% CI lower bound `> 0.0`**.
- **REFUTED** otherwise: the arm is at the floor.

**0.02 is EXP-062's bar**, reused unchanged. That experiment judged depth 9 at **0.0163** to be
*at the floor* despite 7 of 12 seeds beating zero, and this uses the same standard so the two
readings stay comparable. The 95% CI uses the paired `t` at **df=11, 2.201**.

> **If all three arms are at the floor, that is a REAL and strong result**, not a failed
> experiment: the brain's own pathway does not learn a policy anywhere across three orders of
> magnitude of learning rate, on a configuration where a 390-parameter linear head scores 0.3229.
> ==Written here before any number exists so it cannot later be described as inconclusive.==

### Claim 2, SECONDARY and DESCRIPTIVE - the best arm against the reference

Reported as a ratio to EXP-043's **0.3229**, with **no p-value and no verdict**, because the
capacity mismatch is 40x. An unregistered test here would inflate the fixed multiplicity.

### Claim 3, VALIDITY GATE 1 - the regions actually trained

`region_drift` **arm mean `>= 0.001`** for each new arm. A frozen pathway reads exactly **0.0**,
so the gate can fail; EXP-064 measured **2.7051** at 1e-2, so it can pass.

> **There is deliberately NO UPPER bound on drift**, even though "the regions exploded" is the
> hypothesis under test. The 10,000-episode drift at 1e-3 and 1e-4 has never been measured, and a
> ceiling guessed from a 120-episode calibration is exactly the regime error that killed EXP-057's
> and EXP-058's gates. ==Drift is **reported** and interpreted; it is not gated from above.==

### Claim 4, VALIDITY GATE 2 - the spiking pathway actually fires

`motor_rate_mean` **arm mean `>= 0.02`** for each new arm. Calibrated at depth 5 on the real
frozen encoder at 0.105-0.150 before EXP-064, and EXP-064's arm measured **0.1534**. A silent
pathway reads 0.0 and hands the head a constant zero vector.

### The resolution argument, which EXP-064 did not make

==The new rule from EXP-064 is **gate the comparison's resolution, not just the arm's
mechanism**.== Here the resolution is structural rather than gated: the reading is **one-sample
against a fixed bar**, the measured chance floor at this depth is **0.000**, and a working arm on
this exact configuration scores **0.3229**. **The dynamic range is the full 0.000 to 0.3229 and
nothing another arm does can collapse it.** That is why this design needs no resolution gate,
and the argument is stated rather than assumed.

## 4. Cost, priced from EXP-064's measurement

| | |
|---|---|
| Cells | **24** (2 new arms x 12 seeds), all motor arms |
| Per cell | **~4.50 h**, EXP-064's motor arm measured at 6 workers, adjusted for its 14.0 h actual against a 13.7 h estimate |
| Workers | **6**, which DIVIDES 24 into exactly 4 waves |
| Compute | **~108 CPU-h, ~18 h wall** |

**Every cell is a motor arm, so unlike EXP-064 there is no per-arm cost asymmetry to price
separately.** EXP-064's estimate came in at 2% because its two arms were priced apart; here one
figure is correct because there is one kind of cell.

## 5. What this cannot answer

- **Whether the topology HELPS.** There is no control. A confirmed arm shows the pathway *can*
  carry a policy, not that it carries one better than a conventional head.
- **Whether some `region_lr` between the sampled points is better.** Three points across three
  orders is a coarse sweep, chosen to bound the question rather than optimise it.
- **Whether a different head, curriculum or budget would change it.** All inherited unchanged
  from EXP-064, which inherited them from a recipe tuned around a concept-reading linear head.
- **Anything at depth 7 or 8.**
