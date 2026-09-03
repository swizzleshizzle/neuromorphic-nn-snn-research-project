# EXP-057 design - is the critic's benefit CALIBRATION, or state-dependence?

> **PRE-REGISTERED. Committed before any number exists.** Thresholds fixed at commit time.
> **Date:** 2026-09-03 · **Phase:** 3 · **Grounds:** EXP-051, EXP-053, EXP-056.

## 0. The question, and why it needs its own arm

EXP-053 measured a learned critic raising depth-7 success by **+0.0533 at p 0.0498**. EXP-056 then
established that flattening `V(s_t)` to its own episode mean costs **-0.0646 at p 0.0234** and
collapses the arm to the EMA baseline's level. **So within-episode state-dependence is load-bearing.**

**What is still open is the original question**, and EXP-056 says so in its own section 5: it kept
calibration and between-episode state-dependence fixed, so it cannot separate them. Two live
candidates remain for what a critic supplies that a `beta=0.1` EMA does not:

1. **CALIBRATION.** The critic's level is fitted by MSE on current data. The EMA **lags**.
2. **BETWEEN-EPISODE STATE-DEPENDENCE.** `V`'s episode mean varies with the scramble.

**A constant critic separates them.** A single learned scalar, no state input, fitted by the same
MSE loss with the same optimizer and rate:

- it differs from **arm B** in exactly one way: it cannot see the state, at all
- it differs from the **EMA baseline** in exactly one way: it is fitted rather than exponentially averaged

**It has no one-step degeneracy**, which is what disqualified the arm the 2026-08-31 handoff
proposed. That arm formed `G_t - mean(G)`, exactly zero on a one-step episode, and depth 1 averages
1.22 steps per episode. A learned scalar is not the episode's own mean, so no advantage collapses.

## 1. The arm

**One arm, `C`, 12 seeds. EXP-053's arm B copied field for field, with ONE change.**

```
arm B (EXP-053):  critic = nn.Linear(brain.content, 1)   65 parameters
arm C (this):     critic = a single learned scalar        1 parameter
```

Everything else is arm B's config: `arm="regionalized"`, `readout="concept"`, depth 7, 10,000
episodes, `curriculum=(1..7)`, `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`,
`normalize_advantages=False`, `max_depth=7`, EXP-040's E1 encoder **FROZEN**, and
**`critic_lr = 0.01` FIXED** at the value EXP-053's pilot selected blind to success rate. There is
no new hyperparameter, so **there is no pilot**.

**Initialisation matches arm B's bias distribution.** `nn.Linear` draws its bias from
`U(-1/sqrt(fan_in), +1/sqrt(fan_in))` with `fan_in = brain.content`. The scalar is drawn from the
same distribution so that initialisation is not a second difference. The realised value will differ
from arm B's bias at the same seed because the RNG draw order differs; that is unavoidable and is
not a confound, because the distribution is what matters across 12 seeds.

**`normalize_advantages` must stay `False`.** It subtracts the advantage mean, which is
algebraically the returns' mean, and would impose the disqualified batch-mean baseline on top of
this arm.

## 2. Controls, none of which is re-run

| control | mean | what it is |
|---|---|---|
| **EXP-051**, `exp051_transfer_d7` | **0.1471** | the lagging EMA baseline, `beta=0.1` |
| **EXP-053 arm B**, `exp053_critic_d7` | **0.2004** | the full state-dependent critic |
| **EXP-056 arm F**, `exp056_flat_d7` | **0.1358** | the critic with WITHIN-episode dependence removed |

All three are on disk with the same 12 seeds and the same E1 encoders, so all three contrasts are
paired by seed.

## 3. Claims

Paired by seed, exact permutation over all `2**12` sign flips. Interval reporting uses EXP-055's
`describe_contrast`, whose gating was fixed at `439f6bc`.

### Claim 1, PRIMARY - `C` minus the EMA control. **Is a fitted constant better than a lagging one?**

**CONFIRMED at `delta >= +0.05` and `p <= 0.05.`** Deliberately the same bar and alpha as EXP-053's
Claim 1 and EXP-056's Claim 2, so all three numbers are directly comparable.

**This is the calibration hypothesis, asked at last with a control that can answer it.** A
confirmation would mean a meaningful part of the critic's value is simply being well-calibrated and
unlagged, with no state-dependence involved.

### Claim 2, SECONDARY - `C` minus arm `B`. **The total cost of removing all state-dependence.**

Not directional. Pre-registered readings:

| outcome | reading |
|---|---|
| `C` significantly below `B` by >= 0.05 | state-dependence is worth that much in total. Read beside EXP-056's -0.0646 to see how much of it was within-episode. |
| indistinguishable | a bound, never an equivalence, at ~28% power |
| `C` significantly above `B` | the state input was net harmful. Consistent with EXP-056's finding that `V`'s variation can be noise, and would be a genuine surprise. |

### Claim 3, TERTIARY - `C` minus arm `F`. **What did BETWEEN-episode dependence contribute?**

Arm `F` removed within-episode dependence and kept between-episode. Arm `C` removes both. So
`C - F` isolates the between-episode contribution.

**Pre-registered warning, because this contrast is the most tempting to over-read:** at ~28% power
an "indistinguishable" verdict here is the modal outcome whatever the truth is, and it must be
reported as a bound. **It must NOT be read as "between-episode state-dependence contributes
nothing".**

### Claim 4, THE VALIDITY GATE - is the critic actually constant?

**A CONDITION, not a report.** EXP-056 added `critic_within_rms`, the pooled within-episode RMS of
`V`. For a state-blind critic every timestep in an episode reads the same scalar, so it must be
**below 1e-6 at every stage**.

> [!note] AMENDED 2026-09-03, BEFORE ANY NUMBER EXISTS. This clause originally said "exactly 0.0",
> which is unachievable in floating point: `v.mean()` over identical floats reassociates, and the
> implementation measures **6.8e-10** in a smoke run. The same effect was measured in EXP-056,
> where a constant critic drifted 1.34e-07 in head weights while a real one moved 1.80e-01.
>
> **1e-6 sits about three orders above that noise floor and six orders below any genuine
> variation**: arm B's measured within-episode RMS is 0.65 to 2.20. The gate keeps all of its
> discriminating power. Amended in the same commit as the test that measured it, and no
> experimental number existed at the time.

**If any stage reports `critic_within_rms` at or above 1e-6, the arm is not what this spec
describes and every claim above is void.** The aggregator checks this first and refuses the rest
otherwise. This costs nothing, reuses instrumentation that already exists, and would catch a wiring
error that otherwise produces a perfectly plausible number.

Also reported, descriptively: `critic_ev` by stage. A constant predictor should explain
approximately none of the pooled stage-level return variance, and **that is expected rather than
informative** - EXP-056 established that `critic_ev` does not track whether a baseline helps, so it
appears here for the record and decides nothing.

### Multiplicity

**Three policy comparisons, Bonferroni 0.0167** (0.05 / 3). Claim 4 is a condition with no p-value
and belongs to no family.

### Power, stated before the numbers exist

The measured paired sd on this project's real depth-7 comparison arms is **0.0826** (se 0.0238), so
**n=12 gives roughly 28% power for a +0.05 effect**. **An "indistinguishable" verdict is the modal
outcome on Claims 2 and 3 whatever the truth is.** Every null below is a bound reported with its
interval, never an equivalence. The aggregator prints this paragraph above the numbers.

Claim 1 is better placed only if the effect is large: if a fitted constant is worth what arm B's
full critic was worth over the EMA, the delta is around +0.05 and power is still only ~28%. **This
experiment is well powered for nothing.** It is run anyway because the question is load-bearing and
the arm is cheap, and that trade is recorded here rather than discovered later.

## 4. Cost

| phase | what | cost |
|---|---|---|
| 1 | arm `C`, 12 cells, depth 7, 6 workers | **~5.7 h** |

EXP-056 measured 5 h 43 m for exactly this shape: 12 depth-7 cells, 6 workers, 2 clean waves. Worker
count is chosen from memory headroom, not wall clock: EXP-055 measured that more workers buy nothing
on RL cells, and RL workers run about 1.05 GB private each.

## 5. What this experiment CANNOT answer

- **It cannot localise anything to a curriculum stage.** `critic_ev` is +0.4702 at depth 1 and
  negative from depth 4, so the critic's usefulness is almost certainly concentrated early, but this
  design measures only final depth-7 success. Stage localisation needs per-stage held-out
  evaluation, which no driver does.
- **It cannot resolve a small effect.** See the power section. Three underpowered nulls are a
  plausible outcome and would leave the calibration question open rather than answered.
- **It says nothing about depth 6, or about whether any of this compounds.** The encoder line does
  not compound (EXP-049) and its advantage erodes with depth (EXP-051); no claim here extends past
  depth 7 on this recipe.
