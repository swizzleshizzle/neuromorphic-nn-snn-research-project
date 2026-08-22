# EXP-043 Results - depth 6 works, and the break point is now past everything measured

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract
> was committed at `cbbe9d4` **before the data existed**, and `aggregate.py` was written **after
> the records were fetched but before any value in them was read**. No threshold was edited.
>
> **Provenance:** 24 records (2 depths x 12 seeds) plus 24 head checkpoints. Depths 5 and 6 on
> EXP-040's pretrained encoders, 10,000 episodes, curriculum `(1..d)`,
> `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`, `normalize_advantages=False`. Run 2026-08-13
> 13:13 to 2026-08-14 on the laptop `SwizzlesDuo`, `--workers 10`, zero tracebacks. The Tailscale
> path degraded to a relay mid-run and ssh was intermittent for about an hour; the run was
> unaffected (launcher pid unchanged throughout). Baseline is EXP-040 and was **not** re-run.

## Headline

**Depth 6 now works.** By EXP-036's own pre-registered rule, and this time with a real margin
rather than on noise.

| depth | frozen encoder (EXP-036) | pretrained (EXP-040) | **+ depth-1 cap** |
|---|---|---|---|
| 4 | 0.1591 | 0.3471 | **0.5351** (EXP-042) |
| 5 | 0.0396 **BROKEN** | 0.2304 | **0.3412** |
| 6 | 0.0000 **BROKEN** | 0.1037 (at the bar) | **0.1800 WORKING** |

**Every depth from 3 to 6 now clears the working bar.** The break point - depth 5 when EXP-036
measured it, and unmoved by EXP-037 or EXP-038 - is now **past depth 6 and unmeasured**.

## Claim 1 (PRIMARY) - does the cap transfer? SPLIT

Pre-registered: **>= +0.05** at **p <= 0.05**, paired against EXP-040.

| depth | delta | W-L-T | exact p | verdict |
|---|---|---|---|---|
| 5 | **+0.1108** | 8-4-0 | **0.0815** | **REFUTED** |
| 6 | +0.0762 | 9-3-0 | **0.0273** | **CONFIRMED** |

> [!warning] Depth 5 has the LARGER effect and still fails the rule. Report it as refuted.
> +0.1108 is more than double the 0.05 bar. It misses on **p**, not on effect size, because the
> per-seed differences are wide: `+0.430 -0.015 +0.265 +0.220 +0.480 +0.090 -0.105 +0.010 -0.115
> +0.030 +0.050 -0.010`. Four seeds got worse, two by more than 0.10.
>
> **The pre-registered verdict is REFUTED and that is what stands.** Calling a p of 0.0815 a
> pass because the point estimate is big is exactly the move EXP-037 logged and EXP-040 avoided.
> The honest sentence is: *"the depth-5 improvement is large but not statistically established
> at n=12."*
>
> It is also **not** a null. A follow-up with more seeds is well motivated, and is the one place
> in this experiment where more data would clearly change the answer.

### RESOLVED 2026-08-22 at n=24: the depth-5 effect is real

The follow-up above was run. Seeds 12-23 were pretrained and both arms executed for them
(EXP-040's uncapped depth-5 cell had only ever covered seeds 0-11, so the new seeds had no
baseline to pair against until now). **The bar was NOT re-negotiated**: still `>= +0.05` at
`p <= 0.05`. Only `n` changed, and it was fixed at 24 in advance.

| | delta | W-L-T | exact p | verdict |
|---|---|---|---|---|
| original 12 (above) | +0.1108 | 8-4-0 | 0.0815 | REFUTED |
| **replication, the new 12** | **+0.0975** | 8-3-1 | **0.0400** | **CONFIRMED** |
| all 24 | +0.1042 | 16-7-1 | 0.0055 | CONFIRMED |

**The replication passes on its own**, which is the number that matters. Those twelve seeds were
commissioned before any of their values existed and none of them influenced the decision to run,
so they are a clean test of whether the original effect was real. It was, and at almost exactly
the original effect size (+0.0975 against +0.1108).

> [!warning] The pooled 24-seed figure is the best estimate but is NOT a clean pre-registration.
> The first twelve seeds were seen before the second twelve were commissioned, and the extension
> was decided **because** p sat near the bar. That is optional stopping. What contains it: the
> bar is unchanged, n was fixed at 24 in advance rather than "add seeds until it passes", and the
> replication is reported separately so it can be judged alone. **Report all three rows.**

Means: capped **0.3229** vs baseline **0.2188** across 24 seeds.

Reproduce with `.venv/bin/python experiments/043_cap_at_depth_5_6/aggregate_d5_24seeds.py`
(exact permutation over all `2**24` sign flips, about 30 s).

Depth 6's smaller effect passes because it is more consistent: 9 of 12 seeds improved and only
one moved against by more than 0.05.

## Claim 2 - is depth 6 genuinely working? YES

Pre-registered: mean **>= 0.10**, margin **>= 1.0 SE**, and **>= 8 of 12** seeds above the bar.
The extra conditions exist because EXP-040 cleared the bare rule on noise.

| | EXP-040 | **EXP-043** |
|---|---|---|
| mean | 0.1037 | **0.1800** |
| margin over 0.10 | +0.11 SE | **+2.81 SE** |
| seeds above 0.10 | 5/12 | **10/12** |
| verdict | at the bar | **WORKING** |

**This is the claim the extra conditions were written for.** EXP-040's 0.1037 satisfied
`>= 0.10` and meant almost nothing; 0.1800 at +2.81 SE with 10 of 12 seeds means what the rule
was trying to say all along.

## Claim 3 - failure counts, descriptive, no p-value

| depth | EXP-040 | EXP-043 |
|---|---|---|
| 5 | 2/12 | **0/12** |
| 6 | 3/12 | **1/12** |

Fisher's exact cannot separate these at n=12 (~0.48). **Suggestive only** - the reason to
believe the fix is Claim 1's depth-6 result and EXP-042's mechanism, not these counts.

The remaining depth-6 failure is **seed 6**, which also scored worst at depth 5 (0.080). Worth a
look, but one seed is not a pattern.

## Claim 4 - the variance collapse is real but weaker at depth

| depth | sd before | sd after | ratio |
|---|---|---|---|
| 4 (EXP-042) | 0.2242 | 0.1012 | **0.45** |
| 5 | 0.1590 | 0.1277 | 0.80 |
| 6 | 0.1206 | 0.0985 | 0.82 |

The spread narrows at every depth, but the halving seen at depth 4 does **not** repeat. So
EXP-040's "powerful but unreliable" caveat was **largely** the trap at depth 4 and only
**partly** so at 5 and 6. Some genuine seed sensitivity remains deeper, and depth 5's four
regressions are where it shows.

## Claim 5 - the pre-committed null. NOT TRIGGERED

Confirmed at depth 6, so the trap is **not** a depth-4-only phenomenon.

## What this changes

1. **The frontier is unmeasured for the first time since EXP-036.** Depths 3-6 all work. Depth 7
   needs `ExactBFSDistance(max_depth=8)`; the depth-7 shell is roughly 58,000 states against
   depth 6's 8,969, so the held-out cap and the BFS build both need checking before dispatch.
2. **Two levers compound.** Pretraining took depth 6 from 0.0000 to 0.1037; the cap took it to
   0.1800. Neither alone gets there.
3. **Depth 5 deserves a bigger n.** +0.1108 at p 0.0815 is the clearest case in this project so
   far where 12 seeds is the binding constraint rather than the effect.

## Limitations

- **The baseline has no `stage_trace`**, so the mechanism was not paired here. It is established
  at depth 4 (EXP-042) and assumed to transfer; that assumption is untested at 5 and 6.
- **One cap value (2)** and **one budget (10,000 episodes)**.
- **Pretrained encoders only.** A frozen-encoder arm under the cap is untested.
- **Depth 5's verdict is a p-value miss, not a demonstrated absence.** Do not cite it as "the cap
  does not help at depth 5".
- Nothing past depth 6. A random 2x2 scramble is at **depth 11**, and depths 10-12 hold 97.5% of
  the state space.

## Lead for the next experiment

1. **Find the new break point.** Depth 7, and probably 8. This is the obvious next run and the
   first time in the project the answer is genuinely unknown.
2. **Re-run depth 5 with 24 or more seeds** to settle Claim 1 there.
3. **Fine-tune the encoder during RL.** Still untested, and the frozen version now carries four
   working depths.

**Still refuted and CLOSED:** width (EXP-033), volume alone (EXP-034), curriculum stage weighting
(EXP-037), starvation at depth 6 (EXP-037), trainer stabilizers (EXP-038), deleting the depth-1
stage (EXP-042).

## Regenerate

```bash
.venv/bin/python -u experiments/043_cap_at_depth_5_6/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10 --skip-existing
# EXP-040's records are the paired baseline and MUST be present; they are gitignored.
.venv/bin/python experiments/043_cap_at_depth_5_6/aggregate.py
```
