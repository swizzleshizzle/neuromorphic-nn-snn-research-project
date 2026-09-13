# EXP-061 results - the hippocampal recall is NOISE, measured rather than inferred

> **COMPLETE.** 24 cells, one new arm, no tracebacks. **Validity gate PASSED**, so the claims
> below may be read.
>
> **HEADLINE: matched-magnitude noise is indistinguishable from real memory** (`M` minus `N` =
> **+0.0210 at p 0.4989**), **and both cost about 0.1 against the amnesic transform** (`N` minus
> `A` = **-0.1165 at p 0.0001**, clearing Bonferroni).
>
> **This reframes EXP-059's headline.** "Memory hurts" becomes: **the recall block is valuable for
> what it says about the CURRENT state, and mixing in stored content destroys that value as
> thoroughly as replacing it with random noise.**
>
> **H1 is SUPPORTED, NOT CONFIRMED.** The primary is a null, and the spec pre-registered that a
> null here is a bound rather than evidence - precisely so it could not be upgraded afterwards.

**Pre-registration:** `docs/superpowers/specs/2026-09-13-exp061-noise-matched-recall-design.md`.
**The aggregator was written before any EXP-061 number was seen**: completion was confirmed from
counts alone - 24 records, 24 checkpoints, zero python processes - **without reading the run log**,
which prints one success rate per line. EXP-060's could not claim that.

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo`, worktree `C:\Users\mlgbr\wt-exp053` at `04c9b25`, 6 workers |
| Wall clock | 2026-09-13 04:01 to 20:53 UTC, **16.9 h including a ~5 h sleep**, so **~11.9 h of compute** |
| Per cell | wave 1 3.58 h (with startup), then **3.05 h** - matching EXP-059's measured 3.054 h at the same worker count |
| Arms A, M, S | **REUSED from EXP-059**, 24 seeds each, not re-run |
| Arm N | NEW, `memory_noise`, `exp061_noise_d5` |

**The cost estimate held for the first time in five.** It came from a **same-worker-count**
measurement rather than a cross-worker-count scaling; the four that came in low were all scaled.

```bash
powershell -File C:\Users\mlgbr\launch061_wt.ps1 -Phase rl -Workers 6 -SkipExisting
.venv/bin/python -u experiments/061_noise_matched_recall/aggregate.py
```

> [!important] **REUSING EXP-059's ARMS REQUIRED PROVING THE CODE CHANGE WAS INERT, NOT ASSUMING
> IT.** Adding `memory_noise` also added norm instruments touching the `memory` and
> `memory_amnesic` paths. Both were run before and after the change at depth 1, seed 3:
> `success_rate`, `revisit_rate`, `mean_train_entropy` and `recall_content_cos` are **identical to
> full float repr**. The noise generator is constructed **only** for `memory_noise`, so no other
> mode's RNG stream shifts. `test_preexisting_modes_are_numerically_inert` locks it in.

## Claim 3, the validity gate - PASSED

| | |
|---|---|
| arm N mean `recall_concept_norm_ratio` | **0.3041** (min 0.2190, max 0.4512) |
| floor | **>= 0.05** |
| seeds individually below the floor | **none** |

**Six times the floor at the worst seed.** Calibrated at depth 5 on the real frozen encoder before
the spec was written (arm M measured 0.1653-0.3258 across seeds 0-3), and a negligible recall block
would read near 0 - so the gate **can** fail and **can** pass, the pair of checks EXP-057's and
EXP-058's gates failed.

**Sanity, never a gate: `noise_real_cos` = +0.000146** across 24 cells (range -0.00089 to
+0.00111). Essentially exactly zero, so the substitution leaked nothing.

> [!note] **That instrument was blind when first written, and mutation testing is what caught it.**
> It originally compared the real recall to the *noise vector* - both pre-substitution, so ~0 by
> construction and unable to detect a leak. `recall = 0.5 * recall + 0.5 * noise` passed every
> assertion while the policy saw half the real recall. It now compares against the **substituted**
> value, and leaks at 50/50 and 85/15 are both caught.

## The four arms together - this is the substantive result

| arm | readout | success | sd | `revisit_rate` | `optimality` |
|---|---|---|---|---|---|
| **A**, amnesic | `memory_amnesic` | **0.3138** | 0.1318 | 0.3217 | 0.6707 |
| **M**, memory | `memory` | 0.2183 | 0.1346 | 0.3374 | 0.6477 |
| **S**, shuffled | `memory_shuffled` | 0.1979 | 0.1510 | 0.3417 | 0.5745 |
| **N**, noise | **`memory_noise`** | **0.1973** | 0.1189 | 0.3308 | 0.6079 |

**`A` stands alone at 0.3138. `M`, `S` and `N` sit within 0.021 of each other.**

> [!warning] **`N` minus `S` is -0.0006 - noise and WRONG memory are near-identical. That contrast
> is DESCRIPTIVE and carries no p-value here**, because it was not pre-registered and computing one
> post hoc would inflate the multiplicity the spec fixed at two contrasts. It is reported because
> it corroborates the picture from a third direction, not as a result.

## Claim 1, PRIMARY - `M` minus `N`. **+0.0210 at p 0.4989. INDISTINGUISHABLE.**

Approx 95% interval **[-0.0417, +0.0838]**, W-L-T 13-10-1, paired sd 0.1485.

**A BOUND, CONSISTENT WITH H1 AND NOT PROOF OF IT.** H1 ("the recall is noise on the policy path")
predicts exactly this, which is *why* a null cannot confirm it - and the interval still reaches the
0.05 bar in both directions, so n=24 does not resolve the question.

**H2 is what this rules against.** H2 predicted noise would BEAT real memory, i.e. a significantly
negative delta. The point estimate runs the other way (+0.0210, real memory marginally ahead) and
is nowhere near significance. **So "the stored content is actively misleading" gets no support.**

## Claim 2 - `N` minus `A`. **-0.1165 at p 0.0001. CONFIRMED.**

W-L-T **2-20-2**, paired sd 0.1235, clears Bonferroni 0.025 by a wide margin.

**Replacing the recall block's information with noise costs 0.1165**, against the **-0.0954** real
memory cost in EXP-059 on the same seeds, config and encoder. Both land in the same place.

**This is the contrast that carried a positive prediction, and it is the one that fired.** The
recall block's value is what it says about the **current state** - and any uninformative
substitute, stored content or random noise alike, destroys it.

## What this changes

1. **EXP-059's finding now has a mechanism.** "Memory hurts" was a performance result; this makes
   it structural. The hippocampal content is **noise on the policy path**, measured directly rather
   than inferred from `M` ≈ `S`.
2. **The three-way near-identity is the evidence, not any single contrast.** Real content (0.2183),
   wrong content (0.1979) and pure noise (0.1973) are within 0.021 of each other while the
   memory-free transform sits 0.10-0.12 above all three. **Three different kinds of
   uninformative-ness cost the same amount**, which is what "the content carries nothing usable"
   predicts.
3. **The next question is the readout, not the hippocampus.** Nothing here says the stored content
   is *unreadable* - only that THIS readout cannot use it. `MemoryReadout` concatenates a raw
   hippocampal read; a readout that learned what to attend to is untested.
4. **A same-worker-count cost estimate held exactly**, after four scaled ones came in low. Price
   from a measurement at the worker count you will use.

## What is NOT claimed

- **Not that H1 is confirmed.** The primary is a null and a null is a bound. The honest statement is
  that H1 is **consistent with everything measured** and H2 got no support.
- **Not that `N` equals `S`.** That -0.0006 is descriptive, unregistered, and has no p-value.
- **Not that the hippocampus is broken.** EXP-059's gate showed stored content genuinely changes
  the recall (`recall_content_cos` 0.8514, well below 1.0). The content is **there**; the policy
  cannot use it.
- **Not that memory is useless in general.** Depth 5, a frozen EXP-040 encoder, this readout, this
  hippocampus. A different readout is the obvious follow-up and is out of scope.
- **Not a magnitude effect.** Noise is matched to the real recall's L2 norm per step by
  construction, and `noise_real_cos` = +0.000146 confirms the substitution is clean, so this
  cannot be explained by the recall block getting louder or quieter.
- **Nothing about depths other than 5.**
