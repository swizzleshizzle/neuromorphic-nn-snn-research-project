# EXP-066 results - the brain's own pathway does not learn a policy, at any learning rate

> **COMPLETE.** 24 cells, no tracebacks, no reboot. **All six validity gates PASSED**, so the
> claims below may be read.
>
> **HEADLINE: all three arms are at the floor, across three orders of magnitude of
> `region_lr`** - 1e-4 at **0.0017**, 1e-3 at **0.0000**, 1e-2 at **0.0000** - on a
> configuration where a **390-parameter linear head scores 0.3229**.
>
> ==**EXP-064's null was NOT an optimisation artifact.**== Lowering the learning rate by 100x
> does not rescue the pathway. That was the hypothesis EXP-064 recorded and could not test, and
> it is now refuted.
>
> **And the mechanism is worse than "it does not learn".** Across the sweep, the more the regions
> trained, the more completely the policy collapsed. ==Training the prefrontal-to-motor pathway
> actively destroys the policy, monotonically in how far the regions move.==

**Pre-registration:** `docs/superpowers/specs/2026-09-20-exp066-region-lr-sweep-design.md`.
**The aggregator and its 11 tests were committed before dispatch**, and completion was confirmed
from **counts alone** - 24 records, 24 checkpoints, zero python processes - **without reading the
run log**, which prints one success rate per line.

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo`, worktree `C:\Users\mlgbr\wt-exp053` at `3256292`, 6 workers |
| Wall clock | 2026-09-20 02:40 to 23:00 laptop-local, **20.3 h** against an **18 h** estimate (**13% over**) |
| Depth / budget | 5, 10,000 episodes, curriculum 1..5, `max_steps_by_depth=((1, 2),)` |
| Seeds | 0-11, frozen EXP-040 E0 encoders |
| Arm L2 | **REUSED from EXP-064**, 12 seeds, not re-run |
| Reboot | **none.** Uptime 139.6 h, last boot 2026-09-15 |

```bash
powershell -File C:\Users\mlgbr\launch066_wt.ps1 -Phase rl -Workers 6 -SkipExisting
.venv/bin/python -u experiments/066_region_lr_sweep/aggregate.py
```

> [!warning] **COST LESSON: a per-arm price decomposed from a MIXED run is not validated by the
> total matching.**
> EXP-064's 24 mixed cells took 14.0 h against a 13.7 h estimate, a 2% error that looked like
> confirmation of both per-arm figures (motor 4.40 h, control 2.43 h). It was not. ==A mixed run
> constrains only the WEIGHTED SUM; both components can be wrong in opposite directions and still
> reconcile.== EXP-066 is the first **pure** motor run, and it gives the real number: **~5.1 h per
> motor cell**, not 4.40. `train_steps` is ~82-83k here against EXP-064's 83,117, so it is not
> doing more work per cell - the split was simply under-determined.

## The gates - all six PASSED

| arm | `region_lr` | `region_drift` | `motor_rate_mean` |
|---|---|---|---|
| L4 | 1e-4 | **0.0790** (min 0.0695) | **0.0973** (min 0.0623) |
| L3 | 1e-3 | **0.4597** (min 0.3330) | **0.1360** (min 0.0928) |
| L2 | 1e-2 | **2.7051** (min 1.9724) | **0.1534** (min 0.1354) |

**The mechanism worked in every arm.** The regions genuinely trained and the spiking pathway
genuinely fired, at every learning rate. Nothing here failed for a boring reason.

> [!success] **Leaving the drift gate open at the top was the right call, and it was close.**
> The spec refused an upper bound on `region_drift` because the 10,000-episode drift at these
> rates had never been measured and a ceiling guessed from a 120-episode calibration is the
> regime error that killed EXP-057's gate. ==A plausible guess would have been ~0.3. Arm L3
> measured **0.4597**, and a 0.3 ceiling would have VOIDED a perfectly good arm.==

## Claim 1, PRIMARY - one-sample against a 0.02 bar, per arm. **ALL THREE REFUTED.**

| arm | `region_lr` | mean | 95% CI | seeds > 0 | verdict |
|---|---|---|---|---|---|
| L4 | 1e-4 | **0.0017** | [-0.0004, +0.0037] | 3/12 | **REFUTED, at the floor** |
| L3 | 1e-3 | **0.0000** | [+0.0000, +0.0000] | 0/12 | **REFUTED, at the floor** |
| L2 | 1e-2 | **0.0000** | [+0.0000, +0.0000] | 0/12 | **REFUTED, at the floor** |

L4's three nonzero seeds are **0.005, 0.005 and 0.010**. Its interval includes zero. The bar is
EXP-062's, reused unchanged so the two readings stay comparable: that experiment judged **0.0163**
with 7 of 12 seeds above zero to be *at the floor*, and this is an order of magnitude below it.

## The mechanism - training the pathway COLLAPSES the policy, monotonically

| arm | `region_drift` | success | `greedy_modal_action_frac` | `optimality` |
|---|---|---|---|---|
| L4 1e-4 | 0.0790 | **0.0017** | **0.850** | 0.1675 |
| L3 1e-3 | 0.4597 | 0.0000 | **0.969** | 0.0000 |
| L2 1e-2 | 2.7051 | 0.0000 | **1.000** | 0.0000 |

==**The ordering is monotone and inverse: the further the regions moved, the more completely the
evaluated policy collapsed to a single action.**== At 1e-2 it plays one move every step. The limit
of *not training the pathway at all* is where it is least bad.

**So the pathway is not merely failing to learn - the learning signal is destroying it.** This is
a stronger statement than EXP-064 could make, and it is what three points across three orders buy
that one point could not.

**The per-stage trace shows where it goes.** Arm L4, seed 0, `train_solved_frac` by curriculum
stage: **0.280 -> 0.225 -> 0.111 -> 0.028 -> 0.002**, with entropy falling **1.763 -> 0.219**.
Against measured chance floors of 20.8% at depth 1 and ~1.4% at depth 3, the depth-3 figure of
**0.111 is roughly 8x its floor** - so the pathway does learn something real on the shallow
curriculum and then collapses as depth rises. **It is not incapable; it is unstable.**

## Claim 2, SECONDARY and DESCRIPTIVE

The best arm reaches **0.5%** of EXP-043's linear-head **0.3229** on the same configuration.
**Reported with no p-value and no verdict**: that arm trains **390** parameters against this
arm's **15,540**, a 40x gap. It is context, never a control.

## What must NOT be concluded

- **NOT** that the five-region topology is worthless in principle. The recipe, head, curriculum
  and step caps were all tuned around a concept-reading linear head, and this pathway inherited
  every one of them unchanged. That is the fair comparison available, and it is **not** a neutral
  starting point.
- **NOT** that some `region_lr` between the sampled points would work. Three points across three
  orders is a coarse sweep chosen to **bound** the question, and the monotone collapse runs the
  wrong way for an interior optimum.
- **NOT** that prefrontal and motor cannot be trained. They trained at every rate; the gates say
  so. What fails is the policy that results.

## Where this leaves the topology question

EXP-064 asked whether the brain's own pathway beats a matched control and could not answer,
because the control collapsed. EXP-066 asked the prior question - **can it learn at all?** - and
the answer is **no, at any learning rate tried, and training it makes things monotonically
worse.**

==That closes the thread EXP-064 opened.== Combined with the Phase 3 assessment's finding that no
arm-versus-arm contrast can test the topology while it sits off the policy path, the honest
position for the write-up is: **the five-region topology was put on the policy path for the first
time in EXP-064, trained across three orders of magnitude in EXP-066, and did not carry a policy.**
