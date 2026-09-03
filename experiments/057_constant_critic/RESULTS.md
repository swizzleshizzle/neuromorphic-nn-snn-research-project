# EXP-057 results - calibration is NOT the mechanism

> **COMPLETE.** 12 seeds, no tracebacks, 5 h 06 m. **Validity gate PASSED**, so the claims below
> may be read.
>
> **HEADLINE: the calibration hypothesis gets no support at all.** A critic fitted by MSE but
> blind to the state is **indistinguishable from the lagging EMA baseline it replaced: +0.0088 at
> p 0.7822.** That is not merely non-significant, it is a **near-zero point estimate** on the one
> question this arm was built to answer.
>
> **Read beside EXP-056, the picture is consistent.** Every arm without state-dependence lands
> around 0.14 to 0.16; only the full state-dependent critic reaches 0.2004.

**Pre-registration:** `docs/superpowers/specs/2026-09-03-exp057-constant-critic-design.md`,
committed at `e93baf6`, Claim 4's gate amended at `3b270b4`, **both before any number existed**.

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo`, worktree `C:\Users\mlgbr\wt-exp053` at `3b270b4`, 6 workers |
| Wall clock | 2026-09-02 22:58 to 2026-09-03 04:03, **5 h 06 m**, 12 cells, 2 clean waves |
| Arm | EXP-053 arm B copied field for field, `constant_critic=True` the only change |
| Critic | 1 learned scalar against arm B's 65 parameters, `critic_lr` 0.01 fixed |

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch057_wt.ps1 -Phase rl -Workers 6
.venv/bin/python -u experiments/057_constant_critic/aggregate.py
```

## Claim 4, the validity gate - PASSED, but the threshold was badly chosen

The critic is genuinely state-blind. Worst-seed within-episode RMS of `V`, by stage:

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| worst-seed `V` within RMS | 0.0 | 2.70e-07 | 3.64e-07 | 3.57e-07 | 2.18e-07 | 4.57e-07 | **5.01e-07** |
| margin under the 1e-6 ceiling | inf | 3.7x | 2.7x | 2.8x | 4.6x | 2.2x | **2.0x** |

> [!warning] THE GATE PASSED WITH 2.0x MARGIN, NOT THE "THREE ORDERS" THE AMENDMENT CLAIMED.
> **This is a defect in the gate's design, recorded because it nearly produced a false VOID.**
> The 1e-6 ceiling was calibrated on a **depth-3 smoke run** measuring 6.8e-10. Depth 7 runs 15.6
> steps per episode against a return RMS of 4.42, so `v.mean()` reassociates over more and larger
> terms and the floor rises to 5.0e-07, roughly 700x higher. **An absolute threshold does not
> scale with episode length or return magnitude, and this one was set from the wrong regime.**
>
> **The fix, for any reuse: gate on the RATIO to the returns' own within-episode RMS.** Here that
> is 1.1e-07 at depth 7, against arm B's measured 0.415 in EXP-056. **Six orders of margin instead
> of a factor of two**, and scale-free. The absolute form should not be reused.
>
> The gate reports the **worst seed**, not the mean, because one varying seed is enough to break
> the state-blindness claim. That part was right.

`critic_ev` by stage is **descriptive and decides nothing** (EXP-056 retired it as a proxy). For
the record it runs +0.0267 at depth 1 and -0.14 to -0.36 from depth 2 on, which is what a constant
predictor should do against a pooled stage-level return distribution.

## Claim 1, PRIMARY - the calibration hypothesis is NOT supported

**`C` minus the EMA control: +0.0088, W-L-T 5-6-1, p 0.7822.** NOT CONFIRMED.

| | success |
|---|---|
| arm `C`, constant critic | **0.1558** |
| EXP-051 EMA baseline | 0.1471 |

Approx 95% interval **[-0.0538, +0.0713]**, which reaches the bar in both directions, so n=12 does
not exclude a real effect and this is reported as a bound.

**But the point estimate is what matters here, and it is +0.0088.** This is a different kind of
null from Claim 2 below. Being *fitted by MSE on current data* rather than *exponentially averaged
with a lag* is worth essentially nothing, and the data gives no hint of an effect to be
underpowered about. **The hypothesis the 2026-08-31 handoff proposed - that the critic is simply a
better-calibrated constant - is not supported.**

## Claim 2 - `C` minus arm `B`. Do NOT call this a null.

**-0.0446, W-L-T 4-8-0, p 0.0552**, approx 95% interval **[-0.0899, +0.0007]**.

Formally indistinguishable: it misses alpha 0.05 and is far from Bonferroni 0.0167. **But the point
estimate is substantial and its interval barely includes zero.** Calling this "no difference" would
repeat EXP-050's Claim 4 error, where a pre-registered condition was satisfied and the inference
drawn from it was still wrong.

**It agrees in sign and roughly in size with EXP-056's -0.0646 at p 0.0234.** Two independently
pre-registered arms, removing state-dependence in two different ways, both cost the arm about
0.05 to 0.065.

## Claim 3 - `C` minus arm `F`. Pre-registered as unreadable, and it is.

**+0.0200, W-L-T 7-5-0, p 0.3623**, interval **[-0.0255, +0.0655]**.

The spec pre-registered the warning that this contrast is the most tempting to over-read and that
an indistinguishable verdict is the modal outcome at ~28% power. **It must NOT be read as
"between-episode state-dependence contributes nothing."** Arm `F` kept between-episode dependence
and arm `C` removes it; this design cannot resolve the difference at n=12.

## The four arms together

| arm | success | what it has |
|---|---|---|
| arm `B`, full critic | **0.2004** | calibration + between-episode + within-episode |
| arm `C`, constant critic (this) | 0.1558 | calibration only |
| EXP-051 EMA baseline | 0.1471 | neither, and it lags |
| EXP-056 arm `F`, flattened | 0.1358 | calibration + between-episode |

**Everything without within-episode state-dependence sits between 0.1358 and 0.1558, a spread of
0.02. The one arm that has it sits at 0.2004.** That is the shape of the evidence, and it is
consistent across two pre-registered experiments.

**It is a SHAPE, not a set of resolved contrasts.** Only EXP-056's `F` vs `B` cleared its
threshold (p 0.0234). Every contrast in this experiment is a bound. The pattern is what is
persuasive, and n=12 is what stops any single line of it from being conclusive.

## What this changes

1. **The calibration hypothesis is closed, and it was the last live alternative.** EXP-053's
   +0.0533 is not explained by the critic being a better-calibrated constant. Combined with
   EXP-056, the benefit is within-episode state-dependence.
2. **The 2026-08-31 handoff's item 2 is now doubly answered**: its proposed arm was disqualified on
   design grounds before running, and the question it wanted to ask has been answered by a control
   that works, in the negative.
3. **Absolute validity thresholds must be calibrated in the regime they will run in.** This gate
   was set from a depth-3 smoke run and evaluated at depth 7, and it passed with 2x margin rather
   than the three orders claimed. Use a ratio.

## What is NOT claimed

- **Not that a constant critic equals the EMA.** Claim 1 is a bound; the interval reaches +0.0713.
- **Not that `C` and `B` are equivalent.** Claim 2's point estimate is -0.0446 and it is explicitly
  not called a null.
- **Not that between-episode state-dependence is worthless.** Claim 3 cannot resolve it, by design
  and by prior admission.
- **Not a resolved decomposition.** The four-arm table is a shape across two experiments at n=12,
  not an additive budget. Repeating EXP-056 and EXP-057 at higher n is the obvious next step if
  anything is to be built on this.
