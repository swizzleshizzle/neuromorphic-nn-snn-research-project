# EXP-063 results - the ceiling instrument came back NEGATIVE

> **COMPLETE.** 48 cells, two new arms, no tracebacks, no reboot. **All four validity gates
> PASSED**, so the claims below may be read.
>
> **HEADLINE: a learned attention over PERFECT episodic memory is 0.198 WORSE than no memory at
> all.** T measured **0.1154** against amnesic's **0.3138** (p 0.0000) and against the raw
> attractor read's **0.2183** (p 0.0059). Both secondaries are **REFUTED in the wrong
> direction**.
>
> **The primary is CONFIRMED by the pre-registered rule and must not be read as good news.**
> `T - U` = **+0.0612, p 0.0358** - real content beats matched noise. But the effect is
> concentrated on the seeds where the control's policy had **collapsed to zero**, and on the 10
> seeds where it had not, `T - U` is **+0.0320 at p 0.5078 with 5 positive and 5 negative
> differences**. Claim 1 is substantially a difference in **how often training collapsed**, not
> a clean measure of usable episodic content.
>
> **For the Phase 3 checkpoint: this closes the memory line.** The bound the ceiling instrument
> was built to produce is the strongest this project can state - with perfect episodic recall and
> a readout that learns what to attend to, memory does not help here.

**Pre-registration:** `docs/superpowers/specs/2026-09-17-exp063-learned-readout-design.md`.
**The aggregator was written before any EXP-063 number existed**, with 17 tests and 8 mutations,
all caught. **Completion was confirmed from counts alone** - 48 records, 48 head checkpoints,
zero python processes - **without reading the run log**, which prints one success rate per line.

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo`, worktree `C:\Users\mlgbr\wt-exp053` at `509406c`, 6 workers |
| Wall clock | 2026-09-17 02:07 to 20:40 UTC (22:07-16:40 laptop-local), **18.55 h** |
| Per cell | **2.32 h** across 8 clean waves, against a **3.05 h** estimate |
| Depth / budget | 5, 10,000 episodes, curriculum 1..5, `max_steps_by_depth=((1, 2),)` |
| Seeds | 0-23, frozen EXP-040 E0 encoders |
| Arms A, M | **REUSED from EXP-059**, 24 seeds, not re-run |
| Arm N | **REUSED from EXP-061**, 24 seeds, not re-run |
| Reboot | **none.** Uptime 65.7 h, last boot 2026-09-15 03:33, so the 03:31 window passed idle |

```bash
powershell -File C:\Users\mlgbr\launch063_wt.ps1 -Phase rl -Workers 6 -SkipExisting
.venv/bin/python -u experiments/063_learned_readout/aggregate.py
```

**The cost estimate came in 24% HIGH, and the reason is mechanical.** The memory readouts run an
extra full `hippo` forward - a T-step attractor unroll - on every policy step, plus a second one
every 8 steps for EXP-059's probe. The attention readouts run neither: they bypass the attractor
by design. So 2.32 h per cell rather than 3.05. **An estimate borrowed from an arm with a
different per-step cost is not a same-worker-count estimate**, which is the rule this was
supposed to satisfy.

> [!important] **REUSING EXP-059's AND EXP-061's ARMS REQUIRED PROVING THE CODE CHANGE WAS INERT,
> NOT ASSUMING IT.** All five pre-existing readouts were run at depth 1, seed 3 against the
> pre-change code in a clean worktree and against the post-change code, on nine recorded
> quantities: **identical to full float repr**. The attention module and the noise generator are
> constructed only for the two new modes, so no other mode's torch RNG stream shifts.
> `test_preexisting_modes_are_numerically_inert` locks it in, and mutation M8 - building the
> attention for every non-concept mode - fails it.

## The gates - all four PASSED

| gate | reading | arm T | arm U | floor | worst margin |
|---|---|---|---|---|---|
| 1 | `recall_concept_norm_ratio` | **0.8516** (min 0.8200) | **0.8570** (min 0.7864) | 0.05 | **15.7x** |
| 2 | `attn_choice_steps / train_steps` | **0.7261** (min 0.6834) | **0.7485** (min 0.6976) | 0.15 | 4.6x |

Gate 2's trained-regime reading (0.726-0.749) came in just under the untrained calibration
(0.760-0.767) and far above the reasoned near-optimal-play minimum of 0.375, so the arithmetic
in the spec was right and the floor was never close to binding.

## Claim 1, PRIMARY - `T` minus `U`. **+0.0612 at p 0.0358. CONFIRMED, and fragile.**

Approx 95% interval **[+0.0035, +0.1190]**. The point estimate clears the +0.05 bar and the
interval **barely** excludes zero.

> [!warning] **THE CONFIRMATION IS CONCENTRATED ON FLOOR-CENSORED SEEDS. Post-hoc split,
> diagnostic only - it does not and cannot change the verdict above.**
>
> | subset | n | `T - U` | p |
> |---|---|---|---|
> | all seeds | 24 | **+0.0612** | 0.0358 |
> | seeds where `U` = **0.000** | **14** | **+0.0821** | 0.0078 |
> | seeds where `U` > 0 | 10 | **+0.0320** | **0.5078**, 5 positive / 5 negative |
>
> **58% of the control's seeds scored exactly zero**, with a modal action fraction of 0.975 on
> those - dead policies, not merely weak ones. So the primary is largely measuring that
> matched-magnitude noise on the policy path **collapses training more often** than real state
> vectors do. That is a real effect of content, and it is **not** "the readout extracted useful
> episodic information and acted on it".
>
> This is the EXP-062 censoring lesson arriving from a new direction. There it was a bounded
> metric extrapolated; here it is a bounded metric **contrasted**, with one arm pinned at the
> floor on most seeds.

## Claim 2 - `T` minus `A`. **-0.1983 at p 0.0000. REFUTED in the wrong direction.**

Approx 95% interval **[-0.2531, -0.1436]**. Amnesic - the memory-FREE nonlinear expansion of the
current concept - beats the learned episodic readout by nearly 20 points.

## Claim 3 - `T` minus `M`. **-0.1029 at p 0.0059. REFUTED in the wrong direction.**

Approx 95% interval **[-0.1733, -0.0325]**. **The learned readout is worse than the raw
attractor read it was built to improve on**, despite 3.6x the trainable surface and a perfect
cache in place of a lossy attractor.

## The mechanism - both the attention AND the policy collapsed

| arm | success | zero seeds | modal action frac | train entropy | revisit | optimality |
|---|---|---|---|---|---|---|
| A, amnesic | **0.3138** | 2/24 | 0.529 | 0.356 | 0.322 | 0.671 |
| M, memory | 0.2183 | 3/24 | 0.607 | 0.343 | 0.337 | 0.648 |
| **T, attn** | **0.1154** | **8/24** | 0.709 | 0.300 | 0.375 | 0.470 |
| **U, attn-noise** | **0.0542** | **14/24** | 0.827 | 0.228 | 0.479 | 0.282 |

**It is a monotone ladder.** Success falls, policy entropy falls, the modal action fraction
rises, revisiting rises and optimality collapses, in the same order, across all four arms.

**The attention itself collapsed to a near-deterministic pick**: `attn_entropy_norm` **0.0446**
for T and **0.0420** for U against a uniform value of 1.0, range 0.014-0.142. `attn_recency_mass`
is **0.227**, so it is not picking the previous state - it saturates onto one arbitrary earlier
state. A hard-picked single prior state, injected as a block with **85% of the concept's own
norm**, is a large high-variance input to a linear head, and the entropy trace says training did
not survive it.

> [!warning] **A CONFOUND I DID NOT PRE-REGISTER, AND HAD THE NUMBERS TO CATCH.**
>
> | arm | `||read|| / ||concept||` |
> |---|---|
> | N, `memory_noise` | **0.3041** |
> | **T / U** | **0.8516 / 0.8570** |
>
> The spec argued the contrast carried no **width** confound and flagged **capacity**. It said
> nothing about read-block **MAGNITUDE**, and the new arms inject a block **2.8x larger relative
> to the concept** than the reused arms do. It is structural, not a bug: a convex combination of
> rate-coded concepts keeps its norm, where the attractor's recall comes through `fc_out` and a
> Leaky neuron and is much smaller.
>
> **T and U are matched to EACH OTHER on this (0.8516 vs 0.8570), so Claim 1 is clean on this
> axis. Claims 2 and 3 are not.** Their refutations are consistent with "the learned readout
> hurts" but cannot separate that from "a bigger block on the policy path hurts".
>
> **The calibration printed 0.78-0.80 for the new arms and EXP-061's spec printed 0.165-0.326 for
> M on the same page of notes. I had both numbers before dispatch and did not compare them.**
> Matching a control to its arm is not the same as matching either to the arms being contrasted
> against - and only the first was checked.

## Pre-registered joint reading, and what it means for the checkpoint

The spec's contract for "Claim 1 confirmed, Claim 2 not confirmed" reads: *the content is usable
but not enough to beat a memory-free nonlinear expansion of the current state; memory stays
CLOSED NEGATIVE, with a sharper sentence.* That fires, and the sharper sentence is available:

**The episodic signal is not worthless - it beats matched noise - but every readout tried so far
puts the policy further behind than having no memory at all, and a learned one is the worst of
the three.** The ceiling instrument did its job: it was built so that a negative result would be
decisive, and it is, because no readout over the lossy attractor read can beat one over a
perfect cache.

**What this still cannot answer**, unchanged from the spec plus one addition:

- **Whether a magnitude-matched learned readout would behave differently.** The new confound
  above makes this the obvious next question, and it is cheap: scale the read block to the
  attractor read's norm ratio and rerun arm T alone, 24 cells. **Not run here, and not folded
  into this write-up as a caveat that rescues the result.**
- **Whether the attention collapse is fixable** (entropy regularisation, logit temperature, a
  smaller `d_k`). A null at `d_k=16` does not rule out a different capacity, and saying so
  afterwards would be the optional-stopping move this practice exists to prevent.
- **Anything about depth 7.** This is depth 5, matched to EXP-059/061 for arm reuse.
- **Anything neuromorphic.** T is an ordinary attention head, used as an instrument.
