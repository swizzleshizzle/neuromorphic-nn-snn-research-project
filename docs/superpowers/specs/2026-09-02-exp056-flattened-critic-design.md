# EXP-056 design - does the critic's WITHIN-EPISODE state-dependence do anything?

> **PRE-REGISTERED. Committed before any number exists.** Thresholds fixed at commit time.
> **Date:** 2026-09-02 · **Phase:** 3 · **Grounds:** EXP-051, EXP-053, EXP-046.

## 0. The gap this closes, and the correction that created it

EXP-053's Claim 1 is **the only claim that experiment confirmed**: a learned critic raised depth-7
success by **+0.0533 at p 0.0498**, clearing both pre-registered thresholds by a hair. Its
mechanism was written up as absent, on an explained variance of **0.0021**.

**That reading was corrected on 2026-09-02 and the correction is why this experiment exists.** The
0.0021 is the FINAL STAGE only. By stage, over the same 12 seeds:

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| critic explained variance | **+0.4702** | **+0.2054** | +0.0983 | -0.0292 | -0.0668 | -0.0577 | +0.0021 |
| seeds with ev > 0 | 12/12 | 12/12 | 12/12 | 3/12 | 0/12 | 0/12 | 8/12 |
| steps per episode | 1.22 | 2.50 | 4.15 | 6.76 | 9.77 | 12.55 | 15.57 |

Systematic, not noise. **`V(s)` is a good predictor early and a useless one late**, and Claim 1
measures depth-7 success after training through the whole curriculum, so the benefit is free to
originate anywhere in it.

### Why NOT the arm the 2026-08-31 handoff proposed

That handoff proposed replacing the EMA baseline with a **per-episode batch mean** and no critic.
**That arm is disqualified.** It forms `G_t - mean(G)`, which is **exactly zero for a one-step
episode**, and depth 1 averages **1.22 steps per episode** with 78% of episodes solved. It would
lose gradient precisely in the stages where the critic predicts best, so a loss against the critic
could not distinguish "state-dependence matters" from "this arm got no gradient at stage 1".
**The confound would be aligned with the hypothesis**, which is the one thing a control may never
be. Recorded in EXP-053's `RESULTS.md` so it is not re-proposed.

## 1. The arm

**One arm, `F`, 12 seeds. EXP-053's arm B copied field for field, with ONE change.**

The critic is built, read, and fitted **exactly as in arm B** - same `make_critic`, same
`critic_lr = 0.01` selected by EXP-053's blind pilot, same MSE loss, same detach at the read site.
The single change is in how the advantage is formed:

```
arm B (EXP-053):  advantages = returns - v.detach()
arm F (this):     advantages = returns - v.detach().mean()
```

Everything else is arm B's config: `arm="regionalized"`, `readout="concept"`, depth 7, 10,000
episodes, `curriculum=(1..7)`, `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`,
`normalize_advantages=False`, `max_depth=7`, EXP-040's E1 encoder **frozen**.

**`normalize_advantages` must stay `False` and this is load-bearing.** It computes
`(advantages - advantages.mean()) / std`, and subtracting the advantage mean is algebraically the
same as subtracting the return mean. Turning it on would silently impose the disqualified
batch-mean baseline on top of this arm.

### What this holds fixed, and what it therefore tests

Flattening `V` to its own episode mean keeps:

- **calibration** - the level is still fitted by MSE on the current episode, not an EMA that lags
- **between-episode state-dependence** - the episode mean of `V` still varies with the scramble
- **the critic's fitting dynamics** - same loss, same optimizer, same rate

and removes exactly one thing: **the within-episode variation of `V(s_t)` across timesteps.**

**So this experiment tests within-episode state-dependence and nothing else.** It cannot speak to
between-episode state-dependence. See section 5.

## 2. Controls, neither of which is re-run

| control | mean | what it is |
|---|---|---|
| **EXP-053 arm B**, `exp053_critic_d7` | **0.2004** | the full state-dependent critic |
| **EXP-051**, `exp051_transfer_d7` | **0.1471** | the EMA baseline, `beta=0.1` |

Both are on disk with the same 12 seeds and the same E1 encoders, so both contrasts are paired.

## 3. Claims

Paired by seed, exact permutation over all `2**12` sign flips (4096; no scipy in the venv).
Interval reporting uses EXP-055's `describe_contrast`, whose gating was fixed at `439f6bc`.

### Claim 1, PRIMARY - `F` minus arm `B`. Three readings, all informative.

The measured paired sd for arm B minus its control is **0.0826** (se 0.0238), and that is the best
available prior for this contrast's spread.

| outcome | pre-registered reading |
|---|---|
| **indistinguishable** | Within-episode state-dependence contributes nothing detectable. Report as an INTERVAL, never as equivalence. |
| **`F` significantly ABOVE `B`** by >= +0.05 | `V`'s within-episode variation was **harmful noise** in the baseline. Consistent with ev ~ 0 at depth 7: variation uncorrelated with returns adds variance to the gradient without cancelling any. |
| **`F` significantly BELOW `B`** by >= 0.05 | Within-episode state-dependence is doing real work despite explaining ~0 of final-stage return variance. That would be a genuine surprise and the most interesting outcome. |

### Claim 2, SECONDARY - `F` minus the EXP-051 EMA control

**CONFIRMED at `delta >= +0.05` and `p <= 0.05`.** Deliberately the same bar and alpha as EXP-053's
Claim 1, so the two numbers are directly comparable. This asks whether a critic stripped of
within-episode state-dependence still beats the lagging EMA at all.

### Claim 3, THE VALIDITY GATE - did the intervention remove anything?

**This is a CONDITION, not a report.** If `V` barely varies within an episode, flattening is a
no-op and Claim 1's null means nothing.

New instrumentation, accumulated per curriculum stage exactly as `critic_ev` already is:

- `critic_within_ss` - sum over episodes of `sum_t (v_t - mean(v))^2`
- `critic_within_n` - the timestep count

reported as an RMS within-episode spread of `V`, beside the RMS within-episode spread of the
returns it is subtracted from.

**Pre-registered condition: if the RMS within-episode spread of `V` is below 5% of the returns'
own within-episode RMS spread at every stage, Claim 1's null is UNINTERPRETABLE** and must be
reported as "the intervention removed nothing measurable", not as "within-episode state-dependence
does not matter". The aggregator prints this gate above Claim 1 and refuses the interpretation
otherwise, in the same style as EXP-055's shape gate.

### Multiplicity

Two policy comparisons, Claim 1 and Claim 2, **Bonferroni 0.025**. Claim 3 is a condition with no
p-value and belongs to neither count.

### Power, stated before the numbers exist

At a paired sd of 0.0826, n=12 gives roughly **28% power** for a +0.05 effect. **Two of the three
Claim 1 outcomes above are therefore unlikely to be resolved even if true**, and an
"indistinguishable" verdict is the modal outcome regardless of the truth. It must be reported as a
bound with its interval and never as evidence that the arms behave identically. This paragraph is
printed by the aggregator above the number, per EXP-055's precedent.

The contrast may in fact be tighter than 0.0826: `F` and `B` share seeds, encoders, and critic
initialisation, and differ only in advantage formation. That is a hope, not a basis for a power
claim, and no threshold below depends on it.

## 4. Cost

| phase | what | cost |
|---|---|---|
| 1 | arm `F`, 12 cells, depth 7, 6 workers | **~4.4 h** (EXP-053 arm B measured 7,915 s/cell; 12 cells on 6 workers is 2 clean waves) |

No pilot is needed: there is no new hyperparameter. `critic_lr` is fixed at the 0.01 EXP-053's
pilot selected blind to success rate, and **must not be re-selected**, or this stops being arm B
with one change.

Worker count is chosen from memory headroom per the playbook, not for wall clock: EXP-055 measured
that 8 workers bought nothing over 6.

## 5. What this experiment CANNOT answer, and the arm that would

**It does not test whether the critic's benefit is "just calibration".** Flattening keeps the
episode mean of `V`, which still varies with the state from episode to episode. A critic that
cannot see the state at all would test that, and the natural successor arm is:

> **A CONSTANT CRITIC**: a single learned scalar parameter, no state input, fitted by the same MSE
> loss with the same optimizer and rate. It differs from arm B in exactly one way (it cannot see
> the state) and from the EMA baseline in exactly one way (it is fitted rather than exponentially
> averaged). It has no one-step degeneracy, because it is a learned scalar rather than the
> episode's own mean.

That arm is named here so that the calibration question has a pre-registered home and does not get
answered by reinterpreting this one after the fact.

**It also does not localise the benefit to a curriculum stage.** The by-stage explained variance
above says the critic predicts early and not late, but this design measures only final depth-7
success. A stage-localisation arm would need per-stage held-out evaluation, which no driver
currently does.
