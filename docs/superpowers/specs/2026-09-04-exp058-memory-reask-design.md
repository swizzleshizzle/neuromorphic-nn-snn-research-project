# EXP-058 design - does episodic memory help a policy that actually works?

> **PRE-REGISTERED. Committed before any number exists.** Thresholds fixed at commit time.
> **Date:** 2026-09-04 · **Phase:** 3 · **Grounds:** EXP-030, EXP-031, EXP-049.

## 0. The gap this closes

EXP-030 asked whether episodic memory reduces cycling and improves solving. It was **null**, and it
is the oldest unresolved question in the project. **But it was measured on a policy that had
learned almost nothing: 2.2% success at depth 3.** EXP-031 then found that policy was collapsed to
a constant action on 7 of 12 seeds.

**A memory that helps you avoid revisiting states cannot help an agent that is not navigating.**
Depth 6 now runs at **0.3579** (`exp049_fresh2_d6`), so the question can finally be asked of a
policy that solves cubes.

### EXP-030's real lesson, and what it dictates about this design

EXP-030's headline was *"memory beat the shuffle-null by 10.8 points (p 0.078)"*. Its own three-arm
design showed that was the wrong comparison: **memory beat the AMNESIC control by 1.2 points at
p 0.91.** The shuffle-null contrast was measuring **the harm of incorrect memory**, not the benefit
of correct memory. **Two arms would have published a false positive.**

So: **the primary contrast here is memory against AMNESIC**, and the shuffle-null is reported as a
secondary that answers a different question. That ordering is fixed now, before any number exists,
precisely because the tempting comparison is the wrong one.

## 1. The arms

**Three arms, 12 seeds, 36 cells. `exp049_fresh2_d6` copied field for field, varying ONLY the
readout.**

| arm | `readout` | what the policy head reads |
|---|---|---|
| **A**, amnesic | `memory_amnesic` | concept, a ZEROED recall block, familiarity |
| **M**, memory | `memory` | concept, the hippocampal recall code, familiarity |
| **S**, shuffled | `memory_shuffled` | concept, recall and familiarity computed from a WRONG query |

Base config, from `exp049_fresh2_d6` (the best depth-6 policy on record, 0.3579): `arm="regionalized"`,
depth 6, 10,000 episodes, `curriculum=(1..6)`, `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`,
`normalize_advantages=False`, `max_depth=6`, **encoder FROZEN** (`encoder_lr=None`), EXP-049's E2
encoder `exp049_ft2_d6_..._s{seed}_sig0.0_encoder.pt`, which is **tracked in git for all 12 seeds**.

**THE AMNESIC ARM IS THE CONTROL, NOT `concept`.** `CLAUDE.md`'s architecture invariant is that a
readout change alters the feature width, so comparing `memory` against `concept` would confound
memory with width. The amnesic arm has the identical width and the identical code path with the
stored content zeroed. **`exp049_fresh2_d6`'s 0.3579 is context, not a control**, and no claim below
is paired against it.

## 2. Claims

Paired by seed, exact permutation over all `2**12` sign flips.

### Claim 1, PRIMARY - `M` minus `A` on held-out success

**CONFIRMED at `delta >= +0.05` and `p <= 0.05`.** Does correct memory help a working policy?

### Claim 2, THE MECHANISM - `M` minus `A` on `revisit_rate`

**`CLAUDE.md`: prefer a mechanism measurement to a performance measurement.** Memory is supposed to
reduce cycling. **CONFIRMED at `delta <= -0.02` and `p <= 0.05`** - note the sign: a *fall* in
revisit rate is the predicted direction.

**This claim is reported whatever Claim 1 does, and the four combinations are pre-registered:**

| Claim 1 | Claim 2 | reading |
|---|---|---|
| confirmed | confirmed | memory works, and it works through the mechanism claimed for it |
| confirmed | not | it helps, but not by reducing revisits. The stated mechanism is wrong and the real one is unknown. |
| not | **confirmed** | **the mechanism engages and does not convert.** This is EXP-030's localisation done properly, and it points at the readout rather than the hippocampus. |
| not | not | memory is not engaging at depth 6 either. With Claim 3's gate passed, that is a real null and not an implementation failure. |

### Claim 3, THE VALIDITY GATE - is the hippocampus actually storing?

**A CONDITION, not a report, and it exists because this exact defect happened here.** `CLAIM.md`
records that `Hippocampus.store()` once assigned instead of accumulating, so it held **exactly one
pattern**, and the test that should have caught it asserted only `count_nonzero(W_rec) > 0`.

**Pre-registered condition: arm `M`'s `mean_n_stored` must exceed 10.** If it does not, the
hippocampus is not accumulating, arms `M` and `A` are the same arm under two names, and **every
claim above is void**. The aggregator checks this first and refuses the rest.

Also gated: arm `S`'s `unshuffled_frac` must be **below 0.20**. If the shuffled query mostly
coincides with the true one, the shuffle-null is not a null and Claim 4 means nothing.

### Claim 4, SECONDARY - `M` minus `S`

**The harm of INCORRECT memory, and it is not evidence about the benefit of correct memory.**
Reported with that sentence attached. EXP-030's headline came from this contrast and was
misread; the write-up must not repeat that.

### Multiplicity

**Three inferential contrasts** (Claims 1, 2, 4), **Bonferroni 0.0167**. Claim 3 is a condition with
no p-value.

### Power, stated before the numbers exist

The paired-difference sd on this project's depth-6 arms runs **0.10 to 0.14**, so **n=12 gives
roughly 28% power for a +0.05 effect**. An indistinguishable verdict is a likely outcome whatever
is true, and every null is a **bound reported with its interval**, never an equivalence.

**`revisit_rate` is the better-powered instrument** and that is why Claim 2 exists: it is a
per-episode rate rather than a success count, so its within-arm spread is smaller. The experiment
is designed so that the mechanism can resolve even if the performance cannot.

## 3. Cost, and how it will be measured rather than assumed

**The honest position: the per-cell cost of a memory arm at depth 6 is NOT known, and two attempts
to measure it were inconclusive.** A concept cell at this config runs about 2.8 h on the laptop. On
this VPS a 60-episode memory run exceeded the 570 s that a concept run took 373 s to finish, so
memory costs **at least 1.5x** concept and the upper bound was not established.

So the plan is:

1. **Dispatch all 36 cells at 6 workers**, which is 6 clean waves.
2. **Read the true per-cell cost from wave 1**, which every record carries in `seconds`.
3. **Report it and decide whether to continue.** The driver takes `--skip-existing`, and seeded runs
   are byte-identical, so stopping and resuming is free and lossless.

Range: **17 h if memory is free, past 30 h if it is 2x.** This is written down now so the number is
compared against a stated expectation rather than rationalised afterwards.

## 4. What this experiment CANNOT answer

- **Nothing about depth 7**, or about whether memory would help where the policy is weaker again.
- **Nothing about a better readout.** If Claim 2 confirms while Claim 1 does not, that localises the
  failure to the readout but does not design a replacement.
- **It cannot separate the hippocampus from the readout as the source of a null.** Claim 3's gate
  rules out a dead hippocampus, which is different from ruling out a poorly-conditioned one.
