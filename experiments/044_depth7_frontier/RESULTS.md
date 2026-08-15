# EXP-044 Results - depth 7 does not work, and the break point is NOT yet located

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract was
> committed at `b992854` **before the data existed**, and `aggregate.py` at `62b825e` **while arm A
> was still running** - there was no record on disk to peek at. No threshold was edited.
>
> **Provenance:** 24 records (12 training + 12 floor) and 12 head checkpoints. Depth 7 on EXP-040's
> pretrained encoders, 10,000 episodes, curriculum `(1..7)`, `max_steps_by_depth=((1,2),)`,
> `entropy_beta=0.0`, `normalize_advantages=False`, `heldout_cap=200`. Training arm dispatched
> **2026-08-14 23:52 UTC** on the laptop `SwizzlesDuo`, `--workers 12`, all 12 records present by
> **07:08 UTC**, so **wall time was at most 7.3 h** - faster than the 9-12 h estimated. The floor
> arm ran separately at ~07:15 UTC (see "the floor arm crashed" below).

## Headline

**Depth 7 scores 0.0621 and is REFUTED against the pre-registered bar of 0.10.**

But **the break point is not located yet**, and this experiment deliberately cannot claim it is.
Claim 2's escalation is triggered and unrun: per-state coverage at depth 7 is **0.044** episodes
per training state against depth 6's **0.190**, so *"the task is harder"* and *"each state was seen
a quarter as often"* are not separable from this arm alone. That was written down before the
numbers existed precisely so it could not be argued either way afterwards.

| depth | mean | sd | source |
|---|---|---|---|
| 4 | 0.5351 | 0.1012 | EXP-042 |
| 5 | 0.3412 | 0.1277 | EXP-043 |
| 6 | 0.1800 | 0.0985 | EXP-043 |
| **7** | **0.0621** | **0.0551** | **EXP-044** |

> [!important] 0.0621 is not zero, and the difference matters
> The measured floor is **exactly 0.0000** on all twelve seeds, and **11 of 12** seeds scored above
> it, with **4 of 12** above the 0.10 bar. Depth 7 is **off the floor and below the bar** - the same
> position EXP-040's depth 6 occupied at 0.1037 before the cap moved it to 0.1800 and made it work.
> Do not write "depth 7 fails"; write "depth 7 does not clear the bar."

## Claim 1 (PRIMARY) - is depth 7 working? REFUTED

Pre-registered: mean **>= BAR**, margin **>= 1.0 SE**, and **>= 8 of 12** seeds above BAR, where
`BAR = max(0.10, 2 x the measured floor)`. All three had to hold.

| condition | value | verdict |
|---|---|---|
| mean >= 0.1000 | **0.0621** | FAIL |
| margin >= 1.0 SE | **-2.38 SE** | FAIL |
| seeds above bar >= 8 | **4 / 12** | FAIL |

**No p-value, by design.** Depth 7 has never been attempted, so there is no paired baseline; the
uncertainty rides on the margin and seed-count conditions. A paired p here would mean a baseline
was invented after the fact.

Per seed: `0.140 0.050 0.010 0.025 0.035 0.050 0.000 0.110 0.110 0.010 0.040 0.165`

## Claim 2 - the pre-registered escalation. TRIGGERED, NOT YET RUN

Arm B is depth 7 at **44,000 episodes**, which matches depth 6's 0.190 episodes per training
state. Its reading was fixed in advance:

| A | B | conclusion |
|---|---|---|
| fails | **works** | the failure was **starvation, not difficulty**. The break point is not found, and every depth's number becomes a function of coverage. |
| fails | fails | **the break point IS depth 7** for this recipe. |

**Cost, re-estimated from what arm A actually did:** 465,175 env steps per run against arm A's
105,740, so **4.40x**, and arm A took at most 7.3 h. That puts arm B near **32 h**, not the ~52 h
the design guessed from a more conservative throughput figure.

## Claim 3 - failure counts, DESCRIPTIVE, no p-value

**1 of 12** seeds at exactly 0.000 (seed 6, which was also EXP-043's only depth-6 failure and its
worst depth-5 seed). Reported as a count with **no test**: n=12 cannot show a count went to zero.

## Claim 4 - variance across depths, DESCRIPTIVE

The spread falls with the mean (sd 0.1012 -> 0.1277 -> 0.0985 -> 0.0551), so the coefficient of
variation is rising: **0.19, 0.37, 0.55, 0.89**. Deeper runs are proportionally *less* consistent
even as their absolute spread shrinks, which is what a distribution being squeezed against a floor
looks like.

The means fall by ratios of **0.64, 0.53, 0.34** per depth. That is not a cliff, and it is not a
constant factor either - the decay is accelerating.

## Claim 5 - the pre-committed null. NOT YET REACHED

Reaching it needs arm B. A refuted Claim 1 **with** a refuted arm B would be the first located
break point since EXP-036, and a result rather than a failure. This experiment has half of that.

## The floor arm crashed, and finding out cost minutes instead of hours

`arm="random"` short-circuits training and so never enters the loop that fills `stage_trace`,
which the record assembles unconditionally. It raised `UnboundLocalError` **after** both
evaluations had run. The combination had gone unexercised for two experiments: `stage_trace`
arrived with EXP-042, every floor measured before it predates the telemetry, and the existing
floor test runs with no curriculum. EXP-044 is the first to ask for a measured floor **with** one.

Caught by smoke-testing the floor path on the VPS while arm A was in flight, not by the run. In
the dispatched pool it fired about four hours in, when the first training run freed a worker; the
loop raised, `shutdown(wait=True)` let the other eleven finish, and **all 12 training records were
written anyway**. The floor was then re-run in minutes after the fix (`62b825e`, with a regression
test verified to fail against the pre-fix code).

**The standing rule earned its keep twice here.** Measuring the floor rather than assuming it cost
minutes, returned exactly 0.0000 - and flushed out a real defect on the way.

## Limitations

- **Arm B is unrun, so the headline is conditional.** "Depth 7 does not clear the bar at 10,000
  episodes" is what this shows. Whether depth 7 is *beyond* the recipe is unanswered.
- **One budget, one cap value, pretrained encoders only.** Same as EXP-043.
- **A frozen-encoder arm at depth 7** is untested, as is fine-tuning the encoder during RL.
- Nothing past depth 7. Depths 1-7 are **0.9%** of the state space; a random scramble is at depth 11.

## Lead for the next experiment

1. **Arm B, 44,000 episodes, ~32 h.** It is pre-registered, its reading is fixed, and it is the
   only thing that turns this into a located frontier.
2. **Fine-tune the encoder during RL** - still untested, and the frozen version now carries four
   working depths and a fifth that is off the floor.
3. **Re-run depth 5 with 24+ seeds** to settle EXP-043's Claim 1.

**Still refuted and CLOSED:** width (EXP-033), volume alone (EXP-034), curriculum stage weighting
(EXP-037), starvation at depth 6 (EXP-037), trainer stabilizers (EXP-038), deleting the depth-1
stage (EXP-042).

## Regenerate

```bash
.venv/bin/python -u experiments/044_depth7_frontier/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 12 --skip-existing
.venv/bin/python experiments/044_depth7_frontier/aggregate.py
# EXP-040's exp040_encoder_s*.pt must be present; they are gitignored and live on the laptop.
```
