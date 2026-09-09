# EXP-059 design - the memory question, at depth 5, at n=24, with a gate that works

> **PRE-REGISTERED. Committed before any number exists.** Thresholds fixed at commit time.
> **Date:** 2026-09-09 · **Phase:** 3 · **Grounds:** EXP-030, EXP-031, EXP-043b, EXP-058.

## 0. Why this exists, and why it is not EXP-058 again

EXP-058 asked whether episodic memory helps a policy that works. **It was VOID**: its Claim 3 gate
required `mean_n_stored > 10`, a quantity bounded by episode length, in a setting where episodes
average 7.76 steps. Unsatisfiable by construction, and it gated on **storing** when the arms differ
at the **read** site.

**Its seeds are also burned.** Runs here are byte-identical, so re-running EXP-058's cells under a
corrected gate reproduces exactly the void records. That is laundering, not replication.

**This design fixes both problems with one change of venue.** `exp043_capped_d5` is a working
depth-5 policy at **0.3229**, frozen encoder, **already measured at 24 seeds**, and every encoder it
needs (`exp040_encoder_s0..23.pt`) exists. So:

- **Depth 5 is a different measurement**, not a repeat of the cells already seen.
- **Seeds 12-23 have never been used for this question.**
- **n=24 instead of 12**, which roughly halves the standard error.
- **No encoder manufacturing**, unlike every other option priced this week.

### What EXP-058 nonetheless established, and it shapes the claims below

Its numbers were never licensed, but its **ordering** reproduced EXP-030's trap exactly: memory beat
the shuffle-null (+0.0175) and **did not** beat the amnesic control (-0.0271), just as EXP-030 found
+10.8 and +1.2 on a policy 15x worse. **The same two-arm design would have reported a win both
times.** The amnesic arm is therefore non-negotiable, and **M vs A is the primary**, fixed here
before any number exists for the same reason it was in EXP-058.

## 1. The arms

**Three arms, 24 seeds, 72 cells. `exp043_capped_d5` copied field for field, varying ONLY the
readout.**

| arm | `readout` | what the policy head reads |
|---|---|---|
| **A**, amnesic | `memory_amnesic` | concept, recall from an EMPTIED attractor, familiarity |
| **M**, memory | `memory` | concept, the hippocampal recall code, familiarity |
| **S**, shuffled | `memory_shuffled` | concept, recall and familiarity from a WRONG query |

Base config: `arm="regionalized"`, depth 5, 10,000 episodes, `curriculum=(1..5)`,
`max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`, `normalize_advantages=False`, `max_depth=6`,
**encoder FROZEN**, `exp040_encoder_s{seed}.pt`.

**The amnesic arm is the control, not `concept`.** A readout change alters the feature width, so
comparing memory against concept would confound memory with width. `exp043_capped_d5`'s 0.3229 is
**context, not a control**, and no claim is paired against it.

## 2. Claims

Paired by seed, exact permutation. **At n=24 the exhaustive `2**24` is 16.8M sign flips**, which is
affordable but slow; the aggregator uses the exact test when `n <= 20` and a fixed-seed
200,000-sample permutation otherwise, and **prints which it used**.

### Claim 1, PRIMARY - `M` minus `A` on held-out success

**Not directional.** EXP-058's unlicensed ordering suggests memory may HURT, so collapsing the signs
would discard the more likely outcome. Three readings, fixed now:

| outcome | reading |
|---|---|
| `delta >= +0.05`, `p <= 0.05` | **CONFIRMED.** Correct memory helps a working policy. |
| `delta <= -0.05`, `p <= 0.05` | **Memory HURTS**, and that is a real finding, not a failed confirmation. Report it in the headline. |
| otherwise | a BOUND with its interval, never an equivalence |

### Claim 2, THE MECHANISM - `M` minus `A` on `revisit_rate`

**CONFIRMED at `delta <= -0.02` and `p <= 0.05`.** Note the sign: memory is supposed to REDUCE
cycling. `revisit_rate` is a per-episode rate rather than a success count, so its within-arm spread
is smaller and **this is the better-powered instrument**. All four combinations with Claim 1 are
pre-registered as in EXP-058; the interesting one remains **mechanism confirms, performance does
not**, which localises the failure to the readout rather than the hippocampus.

### Claim 3, THE VALIDITY GATE - calibrated BEFORE this spec was written

**A CONDITION, not a report.** The arms differ at the read site, so the gate measures the read site:
`recall_content_cos`, the cosine between the real recall and a `W_rec`-zeroed one. **Near 1.0 means
the attractor's stored content changed nothing and an M-versus-A contrast has nothing to measure.**

**Measured across the full range at `bb1efa5`, before this document existed:**

| attractor state | `recall_content_cos` |
|---|---|
| empty, nothing ever stored | **1.000000** |
| random loaded | 0.9437 |
| **real depth-5 run** | **0.8128** |
| `MemoryReadout` docstring, independent, 79 policy steps | 0.802 |

**Condition: arm M's mean `recall_content_cos` must be below 0.95.** It sits between a real run and
total failure, so it catches a near-empty attractor and nothing more. **A cosine is bounded and
scale-free**, so it cannot repeat EXP-057's regime-dependence, where an absolute threshold
calibrated at depth 3 passed with 2.0x margin at depth 7.

Also gated: **arm S's `unshuffled_frac` below 0.20** (EXP-058 measured 0.1652, so this is known
satisfiable).

**If either gate fails, every claim is void.** The aggregator checks both first and refuses the rest.

### Claim 4, SECONDARY - `M` minus `S`

**The harm of INCORRECT memory, and not evidence about the benefit of correct memory.** Reported
with that sentence attached, because EXP-030's headline came from this contrast and was misread.

### Multiplicity

**Three inferential contrasts** (Claims 1, 2, 4), **Bonferroni 0.0167**. Claim 3 is a condition.

### Power, stated honestly before the numbers exist

**n=24 helps but does not make this decisive.** At the paired-difference sd this project measures on
depth-6 arms (0.10 to 0.14), `se` at n=24 is about 0.020 to 0.029, so:

| true effect | approx power at alpha 0.05 |
|---|---|
| 0.05 | **~60-70%** |
| 0.03 | ~30-40% |

**EXP-058's unlicensed ordering suggested about -0.03 for M vs A**, which sits in the poorly-powered
row. **So an indistinguishable Claim 1 remains a plausible outcome even if memory genuinely does
something small.** That is stated now so it is not rationalised later. `revisit_rate` is the
better-powered instrument and is why Claim 2 exists as a claim rather than a footnote.

## 3. Cost

**72 cells at 6 workers is 12 clean waves.** A depth-6 memory cell measured **3.37 h** in EXP-058;
depth 5 runs `2d+3 = 13` steps against 15 and one fewer curriculum stage, so **expect roughly 2.5 h
per cell, about 30 h**, and **that is an estimate, not a measurement**. Wave 1's per-cell time
settles it, `--skip-existing` makes stopping free, and the run is expected to span a laptop sleep.

## 4. What this cannot answer

- **Nothing about depth 6 or 7.** EXP-058's depth-6 cells exist but are void; this is depth 5.
- **It cannot separate a poorly-conditioned hippocampus from a poorly-conditioned readout.** Claim 3
  rules out an *empty* attractor, which is narrower than ruling out a badly scaled one.
- **It cannot resolve a small effect.** See the power table. A null Claim 1 with a confirmed Claim 2
  would be the informative outcome, and a null on both is a bound, not a refutation.
