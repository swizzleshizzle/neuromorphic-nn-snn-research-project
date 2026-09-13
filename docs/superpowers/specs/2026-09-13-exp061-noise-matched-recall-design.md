# EXP-061 design - WHY does the memory read hurt? Matched-magnitude noise in the recall block

> **PRE-REGISTERED. Committed before any EXP-061 number exists.** Thresholds fixed at commit time.
> **Date:** 2026-09-13 · **Phase:** 3 · **Grounds:** EXP-030, EXP-059.

## 0. The question EXP-059 raised and could not answer

EXP-059 found that **episodic memory HURTS** a working depth-5 policy: `M` minus `A` was
**-0.0954 at p 0.0056**, and **correct memory was indistinguishable from wrong memory**
(`M` minus `S` = +0.0204, p 0.4268). Arms: amnesic **0.3138**, memory **0.2183**, shuffled
**0.1979**.

**Two incompatible explanations survive that result, and EXP-059 cannot separate them:**

| | hypothesis | what it predicts |
|---|---|---|
| **H1** | The recall is **noise on the policy path**. Stored content carries nothing the policy can use, so reading it is equivalent to reading garbage. | matched-magnitude noise performs like `M` |
| **H2** | The stored content is **actively misleading** - worse than nothing, because it asserts something false about the current state. | noise performs BETTER than `M` |

Both predict `M` ≈ `S`, which is what EXP-059 measured. **Matched-magnitude noise is the arm that
separates them**, and it is the cheapest possible discriminator: one new arm.

**What arm A is, precisely, because the whole design turns on it.** `memory_amnesic` is not "no
recall". It reads the same hippocampus with `W_rec` zeroed, so its recall block is a **feed-forward
nonlinear transform of the CURRENT concept** - genuinely informative about the state the agent is
looking at. `MemoryReadout`'s docstring records that **65% of the recall block's energy is that
memory-free transform** (cosine 0.802). So:

- **A**: recall = informative transform of the current state.
- **M**: recall = that transform with stored content mixed in.
- **N**: recall = **noise at the same magnitude**, informative about nothing.

## 1. The arms

**One new arm, 24 seeds, 24 cells. `exp059_memory_d5` copied field for field, changing only the
readout.** Arms A and M are **REUSED from EXP-059 at all 24 seeds** and are not re-run.

| arm | `readout` | source |
|---|---|---|
| **A**, amnesic | `memory_amnesic` | EXP-059, `exp059_amnesic_d5`, 24 cells on disk |
| **M**, memory | `memory` | EXP-059, `exp059_memory_d5`, 24 cells on disk |
| **N**, noise | **`memory_noise`** | **NEW, this experiment, `exp061_noise_d5`** |

Base config: `arm="regionalized"`, depth 5, 10,000 episodes, `curriculum=(1..5)`,
`max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`, `normalize_advantages=False`, `max_depth=5`,
**encoder FROZEN**, `exp040_encoder_s{seed}.pt`.

> [!important] **REUSING EXP-059's ARMS IS ONLY LEGITIMATE IF THE CODE CHANGE WAS INERT, AND THAT
> WAS VERIFIED RATHER THAN ASSUMED.** Adding `memory_noise` also added norm instruments touching
> the `memory` and `memory_amnesic` paths. Both modes were run before and after the change at
> depth 1, seed 3: `success_rate`, `revisit_rate`, `mean_train_entropy` and `recall_content_cos`
> are **identical to full float repr**. The dedicated `torch.Generator` is constructed **only**
> for `memory_noise`, so no other mode's RNG stream shifts.
> `test_preexisting_modes_are_numerically_inert` locks this in.

**`memory_noise` replaces the recall block with a random vector of the SAME L2 norm**, leaving the
concept block and familiarity untouched. So N differs from M in the recall block's **information
content and nothing else** - not magnitude, not head width, not familiarity.

## 2. Claims

Paired by seed, exact permutation where affordable. **n=24 makes the exhaustive 2**24 = 16.8M**, so
the aggregator uses exact when `n <= 20` and a fixed-seed 200,000-sample permutation otherwise, and
**prints which it used.**

### Claim 1, PRIMARY - `M` minus `N` on held-out success. The H1-vs-H2 discriminator.

**NOT directional.** H2 predicts `N` above `M` (negative delta); H1 predicts a null.

| outcome | reading |
|---|---|
| `delta <= -0.05`, `p <= 0.05` | **H2 SUPPORTED.** Noise beats real memory: the stored content is **actively misleading**, not merely useless. Report in the headline. |
| `delta >= +0.05`, `p <= 0.05` | **Stored content is worth something after all** - it beats noise, even though it loses to the amnesic transform. A third outcome neither hypothesis predicted, and it would need its own explanation. |
| otherwise | a **BOUND**, consistent with H1 and **never proof of it** |

> [!warning] **THE INTERESTING HYPOTHESIS PREDICTS A NULL, AND THAT IS THIS DESIGN'S CENTRAL
> WEAKNESS. It is stated here, before any number exists, so it cannot be quietly dropped later.**
> H1 ("the recall is noise") is supported by `M` ≈ `N`, and a null at n=24 is a bound, not
> evidence. At this project's measured paired sd of 0.10-0.15, `se` is about **0.020-0.031**, so
> power is roughly **50-60% at a 0.05 effect** and much less below that. **An indistinguishable
> Claim 1 will be reported as a bound that is CONSISTENT WITH H1, and H1 will not be called
> confirmed.** Claim 2 is what carries a positive prediction.

### Claim 2 - `N` minus `A`. The claim with a real directional prediction.

**CONFIRMED at `delta <= -0.05` and `p <= 0.05`.** Under **both** hypotheses, destroying the
recall block's information should cost roughly what EXP-059 measured for memory: **-0.0954 at
p 0.0056**, same n, same seeds, same config. So this is the well-grounded, well-powered contrast.

**What it establishes if confirmed:** the recall block's value is its **memory-free transform of
the current concept**, and replacing that with anything uninformative costs about 0.1 - which
reframes EXP-059's headline. "Memory hurts" would become **"the recall block is valuable for what
it says about the CURRENT state, and stored content destroys that as thoroughly as noise does."**

**If NOT confirmed** - if noise costs much less than memory did - then `M` is not simply
information-destroying and H2 gains support from a second direction.

### Claim 3, THE VALIDITY GATE - calibrated at depth 5 BEFORE this spec was written

**A CONDITION, not a report.** The manipulation is a substitution inside the recall block. If that
block is negligible beside the concept block, the policy head barely sees it and **any result here
is vacuous rather than informative.**

**Gate quantity: `recall_concept_norm_ratio` = mean `||recall||` / mean `||concept||`, arm N.**

**Measured directly on the representation at depth 5 with the real frozen encoder, 4 seeds, before
this document existed** (no training or evaluation needed - with a frozen encoder the ratio is a
property of the representation):

| arm | ratio across seeds 0-3 |
|---|---|
| `M`, memory | 0.1653, 0.2138, 0.3258, 0.2983 |
| `A`, amnesic | 0.2648, 0.2218, 0.3550, 0.2794 |
| **`N`, noise** | **0.1653, 0.2138, 0.3217, 0.2831** |

**Condition: arm N's mean ratio must be `>= 0.05`.** The worst observed value is **0.1653, which is
3.3x the threshold**, and a negligible recall block would sit near 0. **The gate can fail** (zero
the block and it reads 0.0) **and can pass** (every real measurement is 0.165 or above), which is
the pair of checks EXP-057's and EXP-058's gates failed.

**It is a RATIO, so it is bounded and scale-free** - the property that stopped EXP-059's cosine
gate repeating EXP-057's regime-dependence.

**Also reported, never gated: `noise_real_cos`**, the cosine between the real recall and the
substituted block. It is ~0 by construction, so **gating on it could not fail** - the mirror of
EXP-058's unsatisfiable gate. It appears as a sanity reading only. *(Mutation testing found the
first version of this instrument compared the real recall to the noise vector rather than to what
was actually returned, which made it blind to a 50/50 leak. It now compares against the
substituted value.)*

### Multiplicity

**Two inferential contrasts** (Claims 1, 2), **Bonferroni 0.025**. Claim 3 is a condition.

## 3. Cost

**24 cells, one arm, 6 workers, 4 clean waves.** EXP-059 measured its depth-5 amnesic cells at
**3.054 h wall each at 6 workers** - the same depth, config and worker count - so this estimate
comes from a same-worker-count measurement rather than a cross-worker-count scaling. **Expect
~12.2 h, and treat it as a floor**: four consecutive cost estimates in this project have come in
low, and the playbook now says to add 20-40% to anything scaled. **Call it 12-16 h.**

`--skip-existing` makes an interruption cost only the in-flight wave.

## 4. What this cannot answer

- **It cannot prove H1.** H1 predicts a null and a null is a bound. The most this design can do is
  *fail to find* the asymmetry H2 predicts, while Claim 2 confirms the information-destruction
  account.
- **It cannot separate "uninformative" from "badly scaled".** Magnitude is held fixed by
  construction, which is the point - but it means a badly-conditioned recall code and an
  uninformative one look identical here.
- **Nothing about depth 6 or 7**, and nothing about a non-frozen encoder.
- **It does not test a better readout.** Whether some other readout could use the stored content is
  the obvious follow-up and is out of scope.
