# EXP-063 design - can ANY readout use the stored content? A learned attention over episodic memory

> **PRE-REGISTERED. Committed before any EXP-063 number exists.** Thresholds fixed at commit time.
> **Date:** 2026-09-17 · **Phase:** 3 · **Grounds:** EXP-030, EXP-059, EXP-061.

## 0. The question the checkpoint has to answer

EXP-059 found that memory **hurts** (`M - A` = -0.0954, p 0.0056). EXP-061 found **why it looks
that way**: matched-magnitude noise performs the same as real recall (`M - N` = +0.0210,
p 0.4989), so the recall block is **noise on the policy path**.

Neither experiment can say whether the episodic information is **unusable** or merely **unused**,
and the difference decides how Phase 3's memory line is written up:

| reading | what the checkpoint would say |
|---|---|
| unusable | "episodic memory does not help on this task" |
| unused | "episodic memory does not help **through a raw readout**" |

EXP-059's own validity gate is what forces the question. It measured `recall_content_cos` at
**0.8514**, well below 1.0, which says the hippocampus's **stored content genuinely changes the
recall**. The information is present and this readout cannot use it.

Every cube experiment to date has measured **one** readout: a raw hippocampal read concatenated
onto the concept, with a linear head on top. Nothing has ever learned **what to attend to**.

## 1. The arms

Two NEW arms, 24 seeds each, at EXP-059's depth-5 configuration field for field. Four existing
arms are **reused at the same 24 seeds and are NOT re-run**.

| arm | readout | tag | source |
|---|---|---|---|
| **T** | `memory_attn` | `exp063_attn_d5` | **NEW** |
| **U** | `memory_attn_noise` | `exp063_attnnoise_d5` | **NEW** |
| A | `memory_amnesic` | `exp059_amnesic_d5` | EXP-059, mean **0.3138** |
| M | `memory` | `exp059_memory_d5` | EXP-059, mean **0.2183** |
| N | `memory_noise` | `exp061_noise_d5` | EXP-061 |

**T attends over a PERFECT cache of prior concepts and bypasses the hippocampal attractor
entirely. That is deliberate, and T is a CEILING INSTRUMENT, not a proposed architecture.** If a
learned readout with perfect episodic recall cannot beat a control with nothing real to attend
to, then no readout over the lossy attractor read can. The result is decisive in both
directions and is cheap in a way that attending over `n` hippocampal reads would not be
(`brain.step` is ~90 ms and dominates every runtime estimate).

**The head width is identical across all five arms** - `[concept, read-block, familiarity]`,
`64*2+1 = 129` - so the contrast carries **no width confound**. What T and U add is
**parameters**: `W_q` and `W_k` at `d_k=16` with no biases, `2 * 64 * 16 = 2,048` on top of a
780-parameter head, a **3.6x trainable surface**. That is a real confound against A, M and N,
and holding it fixed is exactly what U is for.

**U is matched on everything except content.** Same module, same parameter count, same feature
width, and the same read-block **magnitude**: its keys are per-entry norm-matched, non-negative
random vectors, generated once per visited state and stable for the episode, and the resulting
read is rescaled step by step to the norm the real-content read would have had. Non-negative
because concept codes are spike **rates** and live in the positive orthant - signed noise would
cancel under a convex combination and shrink the block by roughly `1/sqrt(n)`, which would make
U differ from T in magnitude as well as in content, the single confound EXP-061 exists to remove.

**The read is over STRICTLY PRIOR states.** The current snapshot is appended after the read, so
attention cannot put its mass on the state the agent is already looking at. Without that, T
would quietly become a wider amnesic arm - the confound A was built to control, re-introduced
through the back door. On the first step of an episode the read block is exactly zero.

## 2. Reuse of EXP-059's and EXP-061's arms

Legitimate ONLY because the change was proven inert rather than assumed to be. All five
pre-existing readouts (`concept`, `memory`, `memory_shuffled`, `memory_amnesic`, `memory_noise`)
were run at depth 1, seed 3 against the pre-change code in a clean worktree and against the
post-change code, on `success_rate`, `revisit_rate`, `mean_train_entropy`,
`recall_concept_norm_ratio`, `optimality`, `trainable_params`, `mean_n_stored`,
`generalisation_gap` and `greedy_modal_action_frac`: **identical to full float repr**. The
attention module and the noise generator are constructed **only** for the two new modes, so no
other mode's torch RNG stream shifts. `test_preexisting_modes_are_numerically_inert` locks three
of them in, and mutation M8 - building the attention for every non-concept mode - fails it.

## 3. Claims

**Test:** paired over the 24 shared seeds, sign-flip permutation, sampled with the add-one
estimator (exhaustive would be `2**24 = 16,777,216`). Approximate 95% intervals from the paired
`t` at `df = 23`.

**Multiplicity:** the primary is read at **p <= 0.05**. The two secondaries are read at
**Bonferroni 0.05/3 = 0.0167**.

### Claim 1, PRIMARY - `T` minus `U` on held-out success

**The capacity-matched test of whether episodic content is usable at all.** Directional:
if the content is usable, `T > U`.

- **CONFIRMED** at `>= +0.05` and `p <= 0.05`.
- **REFUTED** at `<= -0.05` and `p <= 0.05`: the added capacity actively hurt.
- **NULL otherwise, and a null is a BOUND, not evidence of absence.** Stated here so it cannot
  be upgraded afterwards. The honest sentence for a null is: *with a perfect episodic cache and
  a learned attention over it, no usable episodic signal was found at n=24.*

### Claim 2, SECONDARY - `T` minus `A` on held-out success

The practical bar. A is the **best** memory-width arm (0.3138) and is memory-free by
construction. Confirmed at `>= +0.05`, `p <= 0.0167`.

### Claim 3, SECONDARY - `T` minus `M` on held-out success

Does a learned readout beat the raw attractor read (M = 0.2183)? **Capacity-confounded by
construction** - T has 3.6x M's trainable surface - and is therefore reported as suggestive
whatever it shows. Claim 1 is the clean version of this question.

### Mechanism readings - NEVER gates

`attn_entropy_norm` (1.0 = uniform attention, 0.0 = one state picked) and `attn_recency_mass`
(the share on the immediately preceding state).

> **Pre-registered joint reading, before any number exists.** A null on Claim 1 **with
> `attn_entropy_norm` near 1.0** is a **weaker** bound than a null with concentrated attention:
> uniform weights mean the readout never learned to select, so the null would be about
> optimisation rather than about information. It must be written up that way and must not be
> narrated as "the content is unusable".
>
> `attn_recency_mass` near 1.0 separates "uses episodic memory" from "uses the previous state",
> which is a much smaller claim and is not what this arm is for.

### Claim 4, VALIDITY GATE 1 - the read block is not negligible

`recall_concept_norm_ratio` (mean `||read|| / ||concept||`), **arm mean `>= 0.05`** for BOTH new
arms. If the read block is negligible beside the concept the head barely sees it and `T - U`
measures nothing. This is EXP-061's gate, reused unchanged because it was already shown able to
both pass and fail.

**Calibrated 2026-09-17**, depth 5, the real frozen E0 encoder, the real curriculum and the real
step caps, seeds 0-3 on both arms, **before this spec was committed and before any EXP-063
outcome existed**. Measured **on the representation directly** rather than through
`run_cube_baseline`, whose depth-5 held-out evaluation dominates its runtime and is not what the
gate reads - the same correction EXP-062's calibration needed.

| arm | measured range | margin at the worst seed |
|---|---|---|
| T `memory_attn` | **0.7836 - 0.7972** | **15.7x** |
| U `memory_attn_noise` | **0.7845 - 0.7937** | 15.7x |

**The two ranges overlapping is the magnitude match the control depends on, measured rather than
assumed.** A negligible read block would read near 0, so the gate can fail.

### Claim 5, VALIDITY GATE 2 - the attention had something to choose between

`attn_choice_steps / train_steps`, **arm mean `>= 0.15`** for BOTH new arms. Measured
**0.7601 - 0.7672** across the same 8 calibration cells.

**Maximum attainable, computed before the threshold was set** - the EXP-058 arithmetic that
would have caught an unsatisfiable gate. A step contributes only when at least two prior states
exist, so an episode of `L` steps contributes `max(0, L-2)` of `L`. The depth-1 curriculum stage
runs under a **2-step cap** (`max_steps_by_depth=((1, 2),)`) and contributes **exactly zero**
choice steps for a fifth of the budget. The ceiling over the whole curriculum is therefore well
under 1.0, which is why this gate is a **fraction** rather than a step count.

> **The calibration is an UPPER reading here, and the floor is deliberately NOT set from it.**
> The calibration policy is untrained, so its episodes run to the cap (mean **8.3** steps) and
> sit near the ceiling. A trained policy solves sooner, which drives this fraction **down**.
> Setting a floor just under a measurement taken in the wrong regime is precisely EXP-057's and
> EXP-058's mistake, and it is how a gate calibrated on a smoke run kills a real run.
>
> **So the floor is set against the other end.** Under near-optimal play - the regime that
> MINIMISES the fraction - the five stages contribute about 2, 2, 3, 4 and 5 steps against 0, 0,
> 1, 2 and 3 choice steps: `12,000 / 32,000 = 0.375`. **0.15 sits 2.5x below the worst credible
> trained-regime value and 5.1x below the measurement**, and still fails loudly if the cache
> never populates.

## 4. Cost

| | |
|---|---|
| Cells | **48** (2 new arms x 24 seeds) |
| Per cell | **3.05 h**, EXP-061's measurement at **6 workers**, same depth, same budget |
| Workers | **6**, which DIVIDES 48 into exactly 8 waves |
| Compute | **~24.4 h** |

Priced from a **same-worker-count** measurement. Cross-worker-count scalings have come in low
five times; same-count estimates have held. The attention adds two small matmuls per step
against a ~90 ms `brain.step`, so it is not expected to move the per-cell figure; if the first
wave disagrees, the measured figure replaces this one in `RESULTS.md`.

**Durability:** one record per cell, `--skip-existing`. Windows Update restarts this laptop at
about 03:31 local when an update is pending and has destroyed two runs already, so a wave
boundary is the unit of loss, not the run.

## 5. What this cannot answer

- **Whether the HIPPOCAMPUS can supply what attention needs.** T reads a perfect cache. A
  positive result localises the failure to the attractor read and says nothing about how to fix
  it; a null closes the question for both.
- **Whether a bigger readout would work.** `d_k=16` and no value projection is a deliberate
  floor on capacity. A null at this size does not rule out a larger one, and saying so
  afterwards would be the optional-stopping move this practice exists to prevent.
- **Anything about depth 7.** This is a depth-5 experiment, matched to EXP-059/061 so the arms
  can be reused. The memory line has never been measured at the frontier.
- **Anything neuromorphic.** T is an ordinary attention head. It is an instrument for bounding
  what the episodic signal is worth, not a proposal.
