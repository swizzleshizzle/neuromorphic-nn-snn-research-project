# EXP-037 Results - curriculum stage weighting is not a lever, and back-loading actively hurts

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The
> interpretation contract was committed at `3fac3a6` **before the data existed**; no threshold
> was edited while filling this in.
>
> **Provenance:** 48 records (4 cells x 12 seeds) plus 48 head checkpoints. 2x2 cube, concept
> readout, frozen brain at random init, 10,000 episodes, evaluated on the held-out shell at
> `cfg.depth`. Run 2026-08-05 20:01:16 to 2026-08-06 16:31:25 on the laptop `SwizzlesDuo` over
> SSH from the VPS with `--workers 10`, **20.5 h wall**, exit 0, **zero tracebacks**. Dispatched
> at commit `022d8b8`. Records in `outputs/` (gitignored); the `*_head.pt` checkpoints beside
> them ARE tracked. The 25% arm is EXP-036's depth-4 cell and was not re-run. Regenerate at the
> bottom.

## Headline

**Refuted, and informatively so.** Stage weighting is not a lever, and the failure is not a null
- it is a monotone decline. Spending *more* of the budget at the evaluated depth makes things
**worse**, and the mechanism is visible: the arms that spend longer at the deep end collapse
harder.

**The equal split, which nobody chose and which was simply the obvious default in EXP-034, is at
or near optimal.**

## The curve

Depth 4, 10,000 episodes, 12 seeds, paired against the 25% arm with an exact permutation test
over all 4,096 sign flips.

| share at eval depth | mean | sd | modal frac | env steps | vs 25% | exact p |
|---|---|---|---|---|---|---|
| 12.5% | 0.1535 | 0.086 | 0.719 | 75,008 | −0.0056 | 0.8848 |
| **25% (EXP-036)** | **0.1591** | 0.074 | **0.685** | 80,000 | - | - |
| 50% | 0.1078 | 0.101 | 0.735 | 90,008 | −0.0514 | 0.1191 |
| 75% | 0.0921 | 0.089 | 0.757 | 100,004 | **−0.0670** | **0.0171** |
| 100% | - | - | - | - | refuted in EXP-034 | - |

**The 25% arm has both the best success rate and the lowest modal fraction.** Everything to its
right is worse on both.

## Claim 1 - is weighting a lever? REFUTED

Pre-registered: 50% must beat 25% by **>= 0.03 with p <= 0.05**.

Observed: **−0.0514 at p 0.1191**, W-L-T 3-8-1. Not merely short of the bar - the wrong sign.

**The equal default stands. The contract says do not go hunting for another share that works,
and there is no reason to: the trend is monotone downward across everything tested.**

## Claim 2 - is there an interior optimum? THE PRE-REGISTERED WORDING DOES NOT APPLY

The rule was "50% > 75% confirms an interior optimum". Observed: 50% beats 75% by +0.0157 at
p 0.4941, so the rule's condition technically fires.

> [!warning] Reporting that as "interior optimum confirmed" would have been wrong
> The rule was written **expecting 50% to beat 25%**. It does not. With both back-loaded arms
> scoring below the equal split, the peak is **at or below 25%**, not between 50% and 100%.
> The curve is **declining across the whole tested range from 25% upward**, and 50% vs 75% is
> a comparison between two points on that decline, neither of which is a peak.
>
> `aggregate.py` printed the misleading sentence on first run and was corrected to read the
> ordering rather than the single comparison the rule names. **A decision rule written for the
> outcome you expect can produce a true-but-misleading verdict when the result goes the other
> way.** That is the failure logged here; the thresholds themselves were not touched.

## Claim 3 - the control. NOT AS PREDICTED, AND THAT IS THE FINDING

Pre-registered: 12.5% must be **worse** than 25%, otherwise success is not tracking share at the
evaluated depth.

Observed: **−0.0056 at p 0.8848**, W-L-T 5-6-1. **Indistinguishable from 25%.**

Halving the share at the evaluated depth - from 2,500 episodes to 1,252 - **changed nothing
measurable**. Doubling it to 5,002 made things worse. Quadrupling it to 7,501 made things
significantly worse.

**So performance does not track share the way the starvation hypothesis predicted.** It is flat
below the equal split and declines above it. The bootstrap stages are doing more work than the
final stage, and the final stage saturates early.

This is why the control arm was worth its twelve runs. Without it, "50% is worse" would have
been compatible with a simple monotone story in which less is always better; the flat low end
refutes that too.

## Claim 4 - depth 6. STILL EXACTLY ZERO, AND SLIGHTLY MORE COLLAPSED

| arm | mean | seeds above zero | modal frac |
|---|---|---|---|
| EXP-036, equal (16.7% share) | 0.0000 | 0/12 | 0.975 |
| EXP-037, 50% share | **0.0000** | **0/12** | **0.982** |

Tripling the episodes at depth 6 - 1,666 to 5,000 - moved nothing off the floor, and the modal
fraction went **up**.

**Depth 6's failure is definitively not starvation.** EXP-036 diagnosed it as collapse from the
instruments; this is the intervention that would have fixed it if the diagnosis had been wrong,
and it did not. More time at a depth the policy cannot learn drives it further into a constant
action. **The lever at depth 6 is EXP-031/032 territory, not the curriculum.**

## Claim 5 - mechanism. The gain and the collapse move together

| share | modal frac | entropy | success |
|---|---|---|---|
| 12.5% | 0.719 | 0.178 | 0.1535 |
| **25%** | **0.685** | 0.217 | **0.1591** |
| 50% | 0.735 | 0.230 | 0.1078 |
| 75% | 0.757 | 0.249 | 0.0921 |

The uniform floor is **0.3375**. **The best arm is the least collapsed**, and modal fraction rises
monotonically with share from 25% upward, exactly tracking the decline in success.

> [!note] CORRECTED 2026-09-04 (audit `^bbd0`). This said **0.354**, which is the **depth-3**
> figure at a 9-step budget. **This table is depth 4**, which runs `2d+3 = 11` steps, and the floor
> there is **0.3375** (EXP-036's random arms; 0.3352 simulated). **No conclusion changes**: Claim 5
> is a within-table comparison of 0.685 against 0.757, and the floor is context for how collapsed
> those are. If anything the correct floor makes every arm here slightly *more* collapsed relative
> to chance, which strengthens the claim rather than weakening it.

This is a coherent mechanism rather than a bare score: **time spent at the deepest stage pushes
the policy toward a constant action.** Deep states give sparser reward, so more of the budget
spent there means more updates on mostly-failed episodes, and the policy degenerates. The
bootstrap stages are not filler - they are what keeps the policy varied enough to keep learning.

Note entropy rises with share while modal fraction also rises. EXP-035's discriminator
(collapse = low entropy with HIGH modal fraction) is about the collapse *signature*; here the
two move together, which is a third pattern and worth remembering: **entropy alone would have
suggested the back-loaded arms were MORE exploratory, when they were more degenerate.**

## Claim 6 - the confound, and why it makes the result stronger

Holding episodes fixed does not hold compute fixed. The back-loaded arms spent **more**
environment steps at the same episode budget: +12.5% at 50%, +25.0% at 75%.

**The asymmetry biased toward the arms that lost.** They had more compute and still scored
worse. So the negative result is *stronger* than a matched-steps design would have required -
had the back-loaded arms won by a small margin, this section would have been a serious caveat,
and instead it removes one.

The 12.5% arm spent **6.2% fewer** steps than the equal arm and matched it. That direction is
also clean.

## Limitations

- **One budget (10,000 episodes) and one depth for the main axis.** EXP-035 showed depth 3
  climbing 0.397 to 0.500 between 10k and 30k without saturating, so it remains possible that
  weighting matters at a budget where the final stage is not already saturated.
- **Nothing below 12.5% was tested.** The low end is flat between 12.5% and 25%, but the
  behaviour further down is unknown and this experiment says nothing about it.
- **Only the weights moved.** The curriculum is always `(1..d)`; which depths appear at all, and
  in what order, is untested.
- **Adaptive advancement is untested** and is the obvious remaining curriculum question. This
  result makes it less attractive, not more: if the final stage saturates early, advancing into
  it sooner is unlikely to help.
- Depth 4's held-out set is 133 states; depth 6's is 200.

## Lead for the next experiment

**The curriculum is now closed as a lever.** EXP-034 established it works, EXP-035 showed more
episodes still buy gains but expensively, EXP-036 found where it breaks, and EXP-037 shows its
one free tuning knob does nothing. What remains:

1. **Depth 4 is still the frontier** (0.1591, learning and capped, no gap, no collapse), and it
   now needs an intervention that is not the curriculum.
2. **Depth 6 needs a collapse fix, not more time.** EXP-032's stabilizers were refuted *at depth
   3*, where the failure was not collapse. At depth 6 it demonstrably is. That is a different
   question at a different depth and it is now well motivated.
3. **The encoder** (vault Stage 2). EXP-033 Finding 1 stands, EXP-033's probe gives a measured
   way to check whether pretraining raises the ceiling, and depth 4 is where it would show first.
4. **The EXP-030 memory re-ask** remains cheap, with 96 trained heads now serialised across
   EXP-036 and EXP-037.

**Refuted, do not revisit:** width (EXP-033), volume alone (EXP-034), curriculum stage weighting
(this experiment), and starvation as the explanation for depth 6.

## Regenerate

```bash
.venv/bin/python -u experiments/037_curriculum_weighting/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10

# re-read the records and re-apply the pre-registered rules at any time.
# EXP-036's records MUST be present: they are the 25% comparator and every claim is paired
# against them. They are gitignored, so re-fetch from the laptop if outputs/ is empty.
.venv/bin/python experiments/037_curriculum_weighting/aggregate.py
```
