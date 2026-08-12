# EXP-042 design - closing the depth-1 curriculum trap

**Status: pre-registration. Written 2026-08-12, before any EXP-042 number exists.**
Thresholds are fixed. If one is edited after data arrives, that edit is the finding.

## 1. What EXP-041 established

Curriculum stage 1 pays a **constant-action policy 0.3333** against a uniform-random policy's
**0.2208**. A face move has order 4, so from a one-move scramble any repeated move either
inverts it (1 step) or cycles back to solved (3 steps), and the shipped `2d+3 = 5` budget covers
both. **Depth 1 is the only stage where degeneracy beats exploring** - by depth 2 it is 0.037
against 0.051, and by depth 3 it is exactly zero.

EXP-040's two dead seeds die there: entropy 0.005 leaving stage 1 and 0.0002-0.0012 entering the
final stage, against working seeds' 0.41-0.49.

## 2. Arms - 36 runs, depth 4, the EXP-040 pretrained encoders

| arm | change | depth-1 constant-action reward |
|---|---|---|
| **A baseline** | none; `curriculum=(1,2,3,4)`, shipped budget | 0.3333 |
| **B capped** | `max_steps_by_depth=((1,2),)` | **0.1667** |
| **C skipped** | `curriculum=(2,3,4)` | n/a |

Arm B admits the inverse and not the cycle, putting degeneracy **below** random. Arm C removes
the stage entirely and needs no code change at all.

**The baseline is re-run rather than reused.** EXP-040's records predate the `stage_trace`
telemetry, so its per-stage entropy does not exist. Comparing against a differently-instrumented
arm would confound the fix with the instrument.

`max_steps_by_depth` is **training-only**; evaluation always uses `max_steps_for(cfg.depth)`, so
no arm is scored on a different yardstick. Default `()` verified byte-identical to the
pre-change baseline (`tests/training/test_encoder_seam.py`, slow marker).

## 3. The statistical problem, stated before the data

> [!danger] No arrangement of 12 seeds can prove the failures were eliminated
> The effect is **2 of 12 seeds**. In a paired permutation test where two seeds carry the whole
> difference and ten are ~0, only 2 of the 4 sign assignments on those two exceed the observed
> sum, so **p is approximately 0.5 by construction**. Fisher's exact on 2/12 against 0/12 gives
> **p about 0.48**. Both are structurally incapable of significance at this n.
>
> This is not a reason to skip the experiment. It is a reason to **put the pre-registered claim
> on a quantity that varies across every seed**, and to report the failure count descriptively
> with its limitation named rather than dressed in a p-value.

## 4. The contract

All tests paired per seed against arm A, exact permutation over 2^12 = 4096 sign flips.

### Claim 1 (PRIMARY, mechanism) - does the fix keep exploration alive?

**Entropy entering the final curriculum stage**, from `stage_trace`.

> **CONFIRMED** for an arm if mean delta vs A is **>= +0.05** at **p <= 0.05**.

This is the quantity that decides whether the policy can still learn at the evaluated depth, and
EXP-041 measured it as cleanly bimodal: 0.0002-0.0012 collapsed against 0.41-0.49 healthy.

> [!important] What confirmation and refutation each mean here
> If the fix helps **every** seed, the differences are non-zero throughout and the test has real
> power. **If the effect is confined to the two previously-failing seeds, this claim CANNOT
> confirm** - see section 3 - and a refutation must be reported as *"underpowered by
> construction"*, **not** as *"the fix does not work"*.

### Claim 2 (SECONDARY, outcome) - descriptive, deliberately untested

Report per arm: seeds at exactly 0.000, mean success, sd. **No p-value is attached to the
failure count**, because none would be meaningful at 2/12. A drop from 2/12 to 0/12 is
*suggestive and no more*; confirming it needs roughly 40+ seeds per arm and is a separate run.

### Claim 3 - does the fix cost anything?

The bootstrap stages do more work than expected (EXP-037), so removing or shrinking depth 1
might hurt seeds that were fine.

> **A COST IS FLAGGED** if an arm's mean success falls **>= 0.05** below A among the ten seeds
> that did **not** fail in EXP-040.

Arm C is the more likely to pay this: it deletes the stage rather than repricing it.

### Claim 4 - the mechanism check, which is deterministic

Enumerated, not sampled, and already true: constant-action reward at depth 1 is **0.3333**
shipped and **0.1667** capped, against random's **0.2208**. Pinned in
`test_depth1_cap_removes_the_constant_action_exploit`. If arm B fails to raise entropy despite
this, the mechanism story from EXP-041 is wrong and that is the finding.

## 5. Cost

12 seeds x 3 arms at depth 4. Arm A about 80,000 env steps per run, B about 72,500, C about
90,000 - roughly **2.9M steps**, about **18 h** at EXP-038's measured 46 steps/s on 10 workers.
One overnight on the laptop.

## 6. What this cannot say

- **Nothing at depths 5-6**, where the same seeds also failed. Depth 4 is the powered arm.
- **Nothing about a frozen encoder.** The trap predates pretraining but only fires when learning
  is fast enough to reach the attractor; this tests only the pretrained setting.
- **Nothing about budgets other than 2** at depth 1, nor about the `2d+3` rule generally.
- **Nothing about whether 0/12 means the failure rate is zero.** See section 3.
