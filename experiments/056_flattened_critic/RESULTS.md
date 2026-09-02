# EXP-056 results - the critic's benefit IS its within-episode state-dependence

> **COMPLETE.** 12 seeds, no tracebacks. **The most surprising of the three pre-registered
> outcomes is the one that happened.**
>
> **Flattening `V(s_t)` to its own episode mean costs -0.0646 at p 0.0234** and collapses the
> critic to the level of the plain EMA baseline it was supposed to beat (-0.0112 against it,
> p 0.6709). **Within-episode state-dependence is not decoration; it is the whole effect.**
>
> **And the instrument that said otherwise was wrong.** EXP-053's final-stage explained variance
> of 0.0021 was read as "the mechanism is absent". Not only is the mechanism present, but the
> flattened arm's critic **fits returns slightly BETTER at every stage** while performing worse.
> Explained variance did not merely fail to predict the effect; it pointed the wrong way.

**Pre-registration:** `docs/superpowers/specs/2026-09-02-exp056-flattened-critic-design.md`,
committed at `48e1e36` before any number existed. Every threshold below is from that commit.

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo` (Intel Ultra 9 185H), worktree `C:\Users\mlgbr\wt-exp053` at `e7e0d73`, 6 workers |
| Wall clock | 2026-09-02, 00:25 to 06:08, **5 h 43 m**, 12 cells |
| Seeds | 0 to 11 |
| Arm | EXP-053 arm B copied field for field, `flatten_critic=True` the only change |
| Controls | neither re-run: EXP-053 arm B (0.2004), EXP-051 EMA baseline (0.1471) |

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch056_wt.ps1 -Phase rl -Workers 6
.venv/bin/python -u experiments/056_flattened_critic/aggregate.py
```

## Claim 3, the validity gate - checked BEFORE Claim 1, and it PASSES emphatically

If `V` barely varied within an episode there would be nothing for flattening to remove and
Claim 1 would be vacuous. It varies a great deal, and at depth 1 it varies **more than twice as
much as the returns it is subtracted from**:

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| `V` within-episode RMS | 0.6532 | 1.8596 | 1.9641 | 2.2001 | 2.1658 | 1.9724 | 1.8362 |
| returns' within-episode RMS | 0.3048 | 1.2293 | 1.8213 | 2.5516 | 3.2550 | 3.8398 | 4.4245 |
| **ratio** | **2.143** | **1.513** | 1.078 | 0.862 | 0.665 | 0.514 | 0.415 |

The pre-registered condition was that a ratio below 0.05 at every stage makes Claim 1's null
uninterpretable. The smallest ratio here is **0.415**, more than eight times the threshold.

## Claim 1, PRIMARY - `F` BELOW `B`. **-0.0646 at p 0.0234.**

| | value |
|---|---|
| arm `F`, flattened | **0.1358** |
| arm `B`, full critic | 0.2004 |
| paired delta | **-0.0646** |
| W-L-T | **3-9-0** |
| exact p | **0.0234** |
| Bonferroni (2 comparisons) | 0.025 |

Clears the 0.05 bar downward and clears Bonferroni. **It clears Bonferroni by 6.4% of its
margin**, which is the same knife-edge EXP-053's Claim 1 sat on (p 0.0498 against 0.05), and that
should be said plainly rather than buried.

**The pre-registered reading, verbatim from the spec:** *"Within-episode state-dependence is doing
real work despite explaining ~0 of final-stage return variance. That would be a genuine surprise
and the most interesting outcome."*

## Claim 2, SECONDARY - flattening collapses the critic to the EMA baseline

`F` minus EXP-051's EMA control: **-0.0112, p 0.6709, W-L-T 6-6-0**, approx 95% interval
**[-0.0656, +0.0431]**.

**NOT CONFIRMED**, and reported as a bound. The interval reaches the +0.05 bar in both
directions, so n=12 does not resolve whether the flattened critic is better or worse than having
no critic at all. **What it does establish is that the +0.0533 EXP-053 measured is gone.** Strip
the within-episode state-dependence and a learned critic buys nothing detectable over a lagging
`beta=0.1` exponential average.

## Per seed, and the dead-seed count

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `F` flattened | 0.235 | 0.185 | 0.195 | 0.155 | 0.170 | 0.100 | **0.000** | 0.170 | 0.195 | 0.105 | **0.000** | 0.120 |
| `B` full critic | 0.275 | 0.115 | 0.155 | 0.175 | 0.145 | 0.195 | 0.190 | 0.270 | 0.260 | 0.230 | 0.160 | 0.235 |

**Dead seeds: `F` 2, `B` 0, EMA control 1.** EXP-053's Claim 5 was that the critic rescued the one
seed that died outright, 0/12 against 1/12. **Flattening does not merely lose that rescue, it ends
up worse than no critic at all on this measure.** Two of the three seeds where `F` loses most
heavily (6 and 10) are the ones that died.

## The methodological result, which may outlast the experimental one

**Explained variance is a bad instrument for whether a baseline helps.** Two independent ways this
run says so:

1. **Final-stage `critic_ev` was 0.0021 in arm B, and the thing it was measuring turned out to be
   load-bearing.** That number was read as "the mechanism is measurably absent". It was corrected
   on 2026-09-02 for generalising from one stage, and this experiment shows the corrected reading
   was still not conservative enough: the mechanism is not just present, it is the entire effect.
2. **The worse-performing arm has the better-fitting critic, at every stage:**

   | depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
   |---|---|---|---|---|---|---|---|
   | `F` critic ev | +0.5091 | +0.2642 | +0.1235 | -0.0068 | -0.0497 | -0.0383 | +0.0055 |
   | `B` critic ev | +0.4702 | +0.2054 | +0.0983 | -0.0292 | -0.0668 | -0.0577 | +0.0021 |

   **Caveat, and it is a real one:** `F` and `B` have different policies, so their critics see
   different trajectories and these two rows are not a controlled comparison. The direction is
   still notable, and it is the opposite of what "fit the critic better, learn better" predicts.

A baseline in REINFORCE is unbiased for **any** function of state. Its job is variance reduction
across the timesteps of an episode, not accurate prediction of the pooled stage-level return
distribution, and `critic_ev` measures the latter. **This project should stop treating `critic_ev`
as a proxy for critic usefulness.** It joins the retired probe, pretraining move-accuracy, the
entropy trace and `S` on the list of instruments that move against the thing they were trusted to
track.

## What this changes

1. **EXP-053's Claim 1 now has a mechanism.** "A learned critic raises depth-7 success" is
   supported, and so, now, is *why*: its within-episode state-dependence. Both halves of that
   sentence were open for a week.
2. **The 2026-08-31 handoff's proposed follow-up would have answered the wrong question**, and
   badly. It was disqualified before running for a confound aligned with its own hypothesis.
3. **`critic_ev` must not gate future critic work.** A pilot that selects on it, or an arm
   cancelled because of it, would be selecting on an instrument this run shows is uninformative
   at best. EXP-053's own lr pilot predicted arm B would be null on exactly this basis, and
   running the pre-registered arm anyway is what caught it.

## What is NOT claimed

- **Not that the flattened critic is worse than no critic.** Claim 2 is a bound; the interval
  reaches the bar in both directions.
- **Not that calibration contributes nothing.** This arm keeps calibration and between-episode
  state-dependence, so it cannot separate them. The spec pre-registered the successor that can:
  **a CONSTANT CRITIC**, a single learned scalar with no state input, fitted by the same MSE loss.
  It differs from arm B in exactly one way and from the EMA baseline in exactly one way, and it
  has no one-step degeneracy.
- **Not localised to a curriculum stage.** The gate table shows the removal was largest at depth
  1, and the by-stage `critic_ev` says the critic predicts best there, but this design measures
  only final depth-7 success. Stage localisation needs per-stage held-out evaluation, which no
  driver does yet.
- **n=12, and p 0.0234 against a 0.025 threshold.** A twelve-seed result that clears its
  correction by that margin is worth repeating before it is built on.
