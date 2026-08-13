# EXP-043 design - does the depth-1 cap move the frontier at depths 5 and 6?

**Status: pre-registration. Written 2026-08-13, before any EXP-043 number exists.**
Thresholds are fixed. If one is edited after data arrives, that edit is the finding.

## 1. Why

EXP-042 capped the depth-1 **training** step budget at 2, which drops a constant-action policy
from 0.3333 to 0.1667 - below a random policy's 0.2208 - and removes the attractor EXP-041 found.
At depth 4 it was worth **+0.188** (0.3471 -> 0.5351), took seeds at zero from 2/12 to **0/12**,
and more than halved the spread (sd 0.2242 -> 0.1012).

Critically it also helped the ten seeds that never failed, by **+0.1195**. The trap was degrading
every run.

**EXP-040 measured depths 5 and 6 with the trap still in place**: 0.2304 and 0.1037. Those
numbers are now known to be depressed by a defect that has since been fixed.

## 2. Arms - 24 runs

| arm | depths | n | source |
|---|---|---|---|
| **capped** | 5, 6 | 12 each | new; `max_steps_by_depth=((1,2),)` |
| baseline | 5, 6 | - | **EXP-040, not re-run** |

EXP-040's depth-5 and depth-6 cells use the **same pretrained encoders, the same seeds, the same
machine, the same budget and the same curriculum**. They differ from this arm by the cap and
nothing else, so they pair directly.

> [!warning] The baseline has no `stage_trace`, so the mechanism cannot be paired
> EXP-040 predates that telemetry. **The primary claim is therefore on SUCCESS, not on entropy** -
> unlike EXP-042, where the baseline was re-run for exactly this reason. The capped arm's stage
> traces are reported descriptively only.
>
> Re-running the baselines would cost another 24 runs to re-measure quantities EXP-040 already
> has. The mechanism is established at depth 4; this experiment asks whether the fix *transfers*.

## 3. The contract

Paired per seed against EXP-040, exact permutation over 2^12 = 4096 sign flips, two-sided.

### Claim 1 (PRIMARY) - does the cap transfer?

> **CONFIRMED at a depth** if mean delta is **>= +0.05** with **p <= 0.05**.

+0.05 is the EXP-042 bar, and about a third of what the cap bought at depth 4.

### Claim 2 - does depth 6 become genuinely "working"?

EXP-036's rule: **>= 2x the measured floor AND >= 0.10 absolute**. The depth-6 floor is 0.0008,
so the binding bar is **0.10**.

EXP-040 reached 0.1037 - which cleared the rule by **0.11 standard errors**, with 5 of 12 seeds
above it. That was reported as *at* the bar, not above it.

> **WORKING** requires mean **>= 0.10** AND a margin of **>= 1.0 SE** AND **>= 8 of 12 seeds**
> individually above 0.10.

The SE and seed-count conditions exist because EXP-040 showed the bare rule fires on noise.

### Claim 3 - the failure count, descriptive, no p-value

EXP-040 had **2/12** zeros at depth 5 and **3/12** at depth 6. Report the capped counts. **No
test is attached**: Fisher's exact at these n cannot distinguish them, exactly as in EXP-042.

### Claim 4 - does the variance collapse repeat?

At depth 4 the cap took sd from 0.2242 to 0.1012. Report sd per depth. A repeat is evidence that
EXP-040's "powerful but unreliable" caveat was **largely the trap**, not the encoder.

### Claim 5 - the null is pre-committed

> If Claim 1 refutes at both depths while EXP-042 stands, the finding is that **the depth-1 trap
> is a depth-4 phenomenon** - plausible, because deeper runs spend proportionally less of their
> budget in stage 1 and have more stages in which to recover. Report that as a scoping result,
> not as a failure.

## 4. Cost

| depth | steps/run | runs |
|---|---|---|
| 5 | ~84,000 | 12 |
| 6 | ~95,000 | 12 |
| **total** | | **~2.15M steps, about 13 h** |

## 5. What this cannot say

- **Nothing about depth 7+.** If the break point moves again, finding the new one is a separate
  run and needs `ExactBFSDistance(max_depth=8)`.
- **Nothing about a frozen encoder**, or about cap values other than 2.
- **Nothing about depths 2-3 budgets.** Their constant-action rates (0.037, 0.000) are below
  random by enumeration, so no trap exists there - but "no trap" is not "well chosen".
