# EXP-044 Results - depth 7 works at matched coverage. The deficit was STARVATION, not depth.

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract was
> committed at `b992854` **before the data existed**, and `aggregate.py` at `62b825e` **while arm A
> was still running** - there was no record on disk to peek at. No threshold was edited, and arm B
> was dispatched only because arm A refuted, which is the trigger the spec fixed in advance.
>
> **Provenance:** 36 records (12 arm A + 12 arm B + 12 floor) and 24 head checkpoints. Depth 7 on
> EXP-040's pretrained encoders, curriculum `(1..7)`, `max_steps_by_depth=((1,2),)`,
> `entropy_beta=0.0`, `normalize_advantages=False`, `heldout_cap=200`, on the laptop `SwizzlesDuo`
> at `--workers 12`. **Arm A** (10,000 episodes) 2026-08-14 23:52 UTC, <= 7.3 h.
> **Arm B** (44,000 episodes) 2026-08-16 00:50 UTC to 2026-08-17 02:17 UTC, **25.4 h**, zero
> tracebacks. Floor arm separately, after the `stage_trace` fix.

## Headline

**Depth 7 does not clear the bar at 10,000 episodes, and clears it comfortably at 44,000.**

| arm | episodes | coverage | mean | margin | seeds above bar | verdict |
|---|---|---|---|---|---|---|
| A | 10,000 | 0.044 | 0.0621 | **-2.38 SE** | 4 / 12 | **REFUTED** |
| **B** | **44,000** | **0.191** | **0.1971** | **+7.86 SE** | **12 / 12** | **CONFIRMED** |

By the reading fixed in the spec before either number existed: **arm A's failure was starvation,
not difficulty.** Depth 7 works once each training state is seen as often as depth 6's were.

> [!important] THE BREAK POINT IS NOT FOUND, and this is the second time it has moved out of reach
> What has been found is a **budget scaling law**. Depth is not the variable that was binding
> between 6 and 7; coverage is. Every earlier number in this series was taken at a fixed 10,000
> episodes, so each one carries an unmeasured coverage confound of its own.

## The comparison, and the confound in it

> [!danger] CORRECTED 2026-08-17, after EXP-037 was re-read
> This section first framed the finding as *coverage at the deepest shell*. **That framing is not
> supported, and EXP-037 is evidence against it.** Arm B did not raise deepest-shell coverage in
> isolation - it raised the total budget 4.4x, which gave **4.4x more episodes at every stage**,
> shallow ones included (1,432 -> 6,290 per stage). Deepest-shell coverage is one of several
> quantities that moved, and it was chosen because the design named it.
>
> **EXP-037 held the total fixed at depth 4 and shifted the budget toward the evaluated depth.
> Success fell monotonically: 0.1591 at an equal split, 0.1078 at 50%, 0.0921 at 75%.** Raising
> deepest-shell coverage *at fixed budget* made things worse. So the coverage story is in tension
> with data this project already had.
>
> **What stands:** arm A was starved of **total budget**, and quadrupling it fixed depth 7. That
> is a clean within-depth manipulation and it is what Claim 2's "starvation" verdict rests on.
> **What does not stand:** any claim that deepest-shell coverage is the operative variable, or
> that the depth series restates as a coverage series.

Depth 6 at 10,000 episodes and depth 7 at 44,000 sit at the same **deepest-shell coverage** - but
not at the same anything else. Depth 6 got 1,670 episodes per stage; depth 7 at 44,000 got 6,290:

| | depth 6 @ 10,000 | depth 7 @ 44,000 |
|---|---|---|
| episodes per training state | 0.190 | **0.191** |
| mean success | 0.1800 | **0.1971** |
| sd | 0.0985 | **0.0428** |

The gap is +0.0171 against a combined standard error of ~0.031, so the two are indistinguishable.
But **they are matched on one statistic and unmatched on the rest**, so this is a coincidence of
one summary number, not a controlled comparison. The honest reading is *"depth 7 reaches depth 6's
score when given 4.4x the budget"* - and **not** that the two are equivalent at equal coverage.

## Claim 1 (PRIMARY) - is depth 7 working? A REFUTED, B CONFIRMED

Pre-registered: mean **>= BAR**, margin **>= 1.0 SE**, **>= 8 of 12** seeds above BAR, where
`BAR = max(0.10, 2 x the measured floor)`. The floor measured **exactly 0.0000**, so 0.10 binds.

Arm B passes all three, and by the widest margins in the project: **+7.86 SE** and **12 of 12**
seeds above the bar. No prior arm here has had every seed clear it.

**No p-value, by design.** There is no prior depth-7 arm to pair against; the uncertainty rides on
the margin and seed-count conditions. A paired p would mean a baseline was invented afterwards.

Arm B per seed: `0.205 0.165 0.265 0.185 0.210 0.145 0.105 0.220 0.225 0.210 0.195 0.235`

## Claim 2 - the escalation. RESOLVED: starvation

| A | B | conclusion | |
|---|---|---|---|
| fails | **works** | **the failure was starvation** | **this is what happened** |
| fails | fails | the break point IS depth 7 | did not happen |

Written before arm B ran, and it decided the reading rather than leaving it to be argued after.

## Claim 3 - failure counts, DESCRIPTIVE, no p-value

Arm A **1 / 12** seeds at exactly 0.000 (seed 6). Arm B **0 / 12**. Reported as counts with **no
test**: n=12 cannot show a count went to zero.

## Claim 4 - variance, DESCRIPTIVE

| depth | mean | sd | cv |
|---|---|---|---|
| 4 (10k) | 0.5351 | 0.1012 | 0.19 |
| 5 (10k) | 0.3412 | 0.1277 | 0.37 |
| 6 (10k) | 0.1800 | 0.0985 | 0.55 |
| 7 (10k) | 0.0621 | 0.0551 | 0.89 |
| **7 (44k)** | **0.1971** | **0.0428** | **0.22** |

Feeding depth 7 properly did not merely raise the mean - it **cut the coefficient of variation
from 0.89 to 0.22**, the tightest in the whole series. The seed sensitivity that looked like a
property of depth was substantially a property of a starved budget.

## Claim 5 - the pre-committed null. NOT REACHED

It required arm B to fail. It did not.

## Coverage across the series, and what this does NOT license

Coverage at the fixed 10,000-episode budget every earlier depth used:

| depth | shell | train side | episodes at that stage | coverage | mean @ 10k |
|---|---|---|---|---|---|
| 3 | 120 | 90 | 3,334 | **37.04** | 0.3972 |
| 4 | 534 | 401 | 2,500 | **6.23** | 0.5351 |
| 5 | 2,256 | 2,056 | 2,000 | **0.973** | 0.3412 |
| 6 | 8,969 | 8,769 | 1,670 | **0.190** | 0.1800 |
| 7 | 33,058 | 32,858 | 1,432 | **0.044** | 0.0621 |

> [!warning] Coverage does NOT explain the whole series, and this table shows why
> Depth 3 has **37x** the coverage of depth 4 and scores **lower** (0.3972 against 0.5351). If
> coverage were the whole story that could not happen. What is established is the **depth 6 to
> depth 7 step**, where a matched-coverage comparison exists. Depths 4, 5 and 6 have **never** been
> re-run at higher coverage, so whether their numbers are also budget-limited is **untested**.
>
> The tempting sentence - *"every depth just needs more episodes"* - is exactly the overreach this
> project keeps catching. One matched pair is one matched pair.

## Limitations

- **One matched-coverage comparison**, at one pair of depths, with one budget ratio.
- **Depths 4-6 were not re-run** at raised coverage. The scaling law is a hypothesis about them.
- **44,000 episodes is 4.4x the compute** for a result that ties depth 6. Nothing here says the
  recipe got better; it says the measurement at depth 7 was under-fed.
- **One cap value, pretrained encoders only**, as in EXP-042/043. A frozen-encoder arm at depth 7
  is untested, as is fine-tuning the encoder during RL.
- Depths 1-7 are **0.9%** of the state space. A random scramble sits at **depth 11**. Nothing here
  implies the cube is solved.

## Lead for the next experiment

1. **Separate total budget from deepest-shell coverage, at fixed budget.** Depth 7, 10,000
   episodes - the same cost as arm A - with `curriculum_weights` set so stage 7 gets arm B's 6,290
   episodes out of that 10,000. If it scores like arm B (0.197), coverage is the operative
   variable. If it scores like arm A (0.062) or worse, the variable is total budget and EXP-037's
   monotone decline extends to depth 7. **EXP-037 predicts worse**, which makes this a real test
   rather than a confirmation. ~7 h, and it decides which story the write-up above may tell.
2. **Only then, re-run depth 5 or 6 at raised total budget** to see whether the depth series is
   a budget series. ~25 h, and it rests on the answer to (1).
3. **Depth 8 at matched coverage needs ~174,000 episodes** (shell 114,149), roughly **100 h** at
   the observed throughput. The matched budget grows about 4x per depth, so brute-forcing the
   frontier gets expensive faster than the frontier moves.
4. **Fine-tune the encoder during RL** - still untested.
5. **Re-run depth 5 with 24+ seeds** to settle EXP-043's Claim 1 (p 0.0815).

**Still refuted and CLOSED:** width (EXP-033), volume alone (EXP-034), curriculum stage weighting
(EXP-037), starvation at depth 6 (EXP-037), trainer stabilizers (EXP-038), deleting the depth-1
stage (EXP-042).

> Note that "starvation at depth 6" was refuted by EXP-037 and **stays refuted** - that experiment
> tested reallocating a fixed budget between curriculum stages, not raising the total. EXP-044
> raises the total. They are different interventions and do not contradict each other.

## Regenerate

```bash
.venv/bin/python -u experiments/044_depth7_frontier/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 12 --skip-existing            # arm A + floor
.venv/bin/python -u experiments/044_depth7_frontier/run.py --episodes 44000 \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 12 --skip-existing --no-floor  # arm B
.venv/bin/python experiments/044_depth7_frontier/aggregate.py
# Add --dry-run to inspect the pre-flight banner without starting anything.
```
