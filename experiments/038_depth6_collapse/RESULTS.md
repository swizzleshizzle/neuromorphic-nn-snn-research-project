# EXP-038 Results - collapse is a symptom at depth 6 too, and the stabilizers are closed

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The
> interpretation contract was committed at `32b72a4` **before the data existed**, and the two
> corrections the calibration pilot forced (`6416379`) are dated and explained in the spec.
> No threshold was edited while filling this in.
>
> **Provenance:** 48 records (4 cells x 12 seeds) plus 48 head checkpoints. 2x2 cube, concept
> readout, frozen brain at random init, 10,000 episodes, curriculum `(1..d)` at equal stage
> weights, `normalize_advantages=True` throughout, evaluated on the held-out shell at
> `cfg.depth`. Dispatched at commit `6416379`, run 2026-08-07 20:15:11 to 2026-08-08 19:02:18
> on the laptop `SwizzlesDuo` over SSH from the VPS with `--workers 10`. **22 h 47 m wall**,
> exit 0, **zero tracebacks**. All four comparators are EXP-036 cells and were not re-run.
> Records in `outputs/` (gitignored); the `*_head.pt` checkpoints ARE tracked. Regenerate at
> the bottom.

## Headline

**Refuted, and the closing of a line.** EXP-032 showed that de-collapsing the depth-3 policy did
not make it solve cubes, but depth 3 was never collapse-limited. Depth 6 was: modal action
fraction **0.975**, rising to 0.982 under 3x the episodes, with starvation already ruled out.
It was the strongest remaining case for the trainer stabilizers.

**The stabilizers de-collapsed depth 6 exactly as intended, and it bought nothing.**

| arm | mean | seeds at 0 | modal | entropy |
|---|---|---|---|---|
| EXP-036 random (the floor) | 0.0008 | 10/12 | 0.309 | - |
| EXP-036 trained baseline | 0.0000 | 12/12 | 0.975 | 0.204 |
| beta 0.05 | 0.0008 | 11/12 | 0.860 | 0.283 |
| **beta 0.2** | **0.0021** | 7/12 | **0.631** | 1.078 |
| beta 0.8 | 0.0004 | 11/12 | 0.666 | 1.681 |

Modal fraction fell from **0.975 to 0.631** - a policy that played one action for 97.5% of a
rollout now plays it 63% of the time. The intervention worked on the thing it targets. Success
went from 0.0000 to **0.0021**, against a random floor of 0.0008.

> [!important] What 0.0021 actually is, stated plainly
> Five of twelve seeds solved **exactly one** of their 200 held-out states; the other seven
> solved none. The entire depth-6 "gain" is **five single solves out of 2,400 evaluations**.
> The random arm managed two. This is not a small effect - it is noise with a mean attached,
> which is precisely why Claim 1 was pre-registered against the random arm at a 0.02 bar
> rather than against the 0.0000 baseline.

## Claim 1 (PRIMARY) - is it a lever at depth 6? REFUTED

Pre-registered: mean **>= 0.02** AND **p <= 0.017** (Bonferroni over three cells) AND
modal **>= 0.45**, paired against EXP-036's **random** arm.

| beta | mean | vs random | W-L-T | exact p | modal | verdict |
|---|---|---|---|---|---|---|
| 0.05 | 0.0008 | +0.0000 | 1-2-9 | 1.0000 | 0.860 | fail |
| 0.2 | 0.0021 | +0.0013 | 5-2-5 | 0.4531 | 0.631 | fail |
| 0.8 | 0.0004 | -0.0004 | 1-2-9 | 1.0000 | 0.666 | fail |

**No cell came within an order of magnitude of the 0.02 bar.** The best is 0.0021, which is 2.6x
the random floor and a tenth of the bar, at p 0.4531.

**Depth 6's collapse is a symptom, not the binding constraint.** That is the same conclusion
EXP-032 reached at depth 3, now established at the depth where the collapse diagnosis was
strongest rather than weakest.

## Claim 2 (THE DISCRIMINATOR) - was the dose axis actually saturated? YES

This is what makes Claim 1 a real refutation rather than a repeat of EXP-032's central
limitation ("the sweep was bounded too low; every trend was still moving at the boundary").

| | value |
|---|---|
| beta 0.8 entropy | **1.681** |
| ceiling (log 6) | 1.792 |
| fraction of ceiling | **94%** (bar: 90%) |

**DOSE AXIS SATURATED.** The sweep reached the limit of what an entropy bonus can do, so
"a higher beta might have worked" is not available as an explanation.

No cell triggered the instrument-broken check: nothing landed at the uniform anchor while
scoring above the floor.

## Claim 3 - depth 5 coherence. REFUTED, AND WITH THE WRONG SIGN

This is the powered arm - depth 6 has twelve seeds at exactly 0.0000 and 1/200 resolution, so a
partial effect would be invisible there. Depth 5 sits at 0.0396 +- 0.027, where an effect is
measurable.

| | value |
|---|---|
| EXP-036 trained baseline | **0.0396** |
| EXP-038 beta 0.2 | **0.0046** |
| delta | **-0.0350** |
| W-L-T | **1-10-1** |
| exact p | **0.0020** |

Pre-registered: **>= +0.02** at p <= 0.05. Observed is not merely short of the bar - it is
**significantly negative**. The stabilizer made depth 5 roughly **nine times worse**, losing on
ten of twelve seeds.

**The powered arm is where this experiment earned its design.** Depth 6 alone would have given
a null confounded with "below resolution". Depth 5 says the intervention is actively harmful
where the measurement can see, which is a far stronger statement, and it replicates EXP-032
Finding 3 (`beta=0.1+normalize` scored 0.117 at depth 2 against the baseline's 0.380).

## Claim 4 - mechanism. Modal does NOT fall monotonically, and that was predicted

| depth | beta | modal | entropy | success |
|---|---|---|---|---|
| 6 | baseline | 0.975 | 0.204 | 0.0000 |
| 6 | 0.05 | 0.860 | 0.283 | 0.0008 |
| 6 | **0.2** | **0.631** | 1.078 | 0.0021 |
| 6 | 0.8 | **0.666** | 1.681 | 0.0004 |

Modal falls 0.975 -> 0.860 -> 0.631 and then **rises** to 0.666 while entropy keeps climbing.

> [!note] The spec predicted this shape before the run
> Section 5a argued that pushing beta higher would eventually drive modal fraction **back up**,
> because the entropy term flattens the logits toward zero and a flat-logit **argmax is a
> deterministic tie-break**, i.e. a constant action. Training entropy and greedy degeneracy are
> different axes: `mean_train_entropy` describes the stochastic **sampled** policy, while
> `greedy_modal_action_frac` describes the **argmax** policy used at evaluation.
>
> The turn is small (0.631 -> 0.666) and rests on one dose point, so it is **consistent with**
> the prediction rather than a confirmation of it. But it is the reason the instrument check
> was restated as entropy saturation, and had it been left as "modal must reach uniform" this
> experiment would have been scored by a rule that could never fire.

**Entropy alone would have told the wrong story again.** It rises monotonically across every
cell while success does not, and while modal turns. Third distinct pattern in this project
after EXP-035's and EXP-037's - read both, always.

## Claim 5 - the closing verdict. THE STABILIZERS ARE CLOSED

Pre-registered: if Claims 1 and 3 both refute, the trainer stabilizers are closed as a lever.

**Both refuted.** Depth 6 was the one regime where the failure the stabilizers target is the
failure the instruments diagnose, and the dose axis provably saturated. There is no remaining
version of this question worth asking.

> [!important] Refuted and CLOSED. Do not revisit without a new reason.
> - **Width** (EXP-033)
> - **Volume alone** (EXP-034)
> - **Curriculum stage weighting** (EXP-037)
> - **Starvation at depth 6** (EXP-037)
> - **Trainer stabilizers / entropy + advantage normalization** (this experiment)

## What this leaves

Nothing cheap remains inside the current architecture. Every knob that could be turned without
changing what the network *is* has now been turned and measured.

**EXP-039, run the same weekend, is the counterpart:** inverse-model pretraining of the sensory
encoder raised the linear probe at depth 4 from 0.447 to 0.786 (p 0.0005, 12-0) and beat the
raw-facelet linear ceiling. **The lever is representational, not credit-assignment.** These two
results are cleaner together than either is alone: EXP-038 closes the trainer, EXP-039 opens
the encoder.

## Limitations

- **`normalize_advantages` pinned True**, on EXP-032's evidence that beta alone moves nothing
  and normalization alone is harmful. This experiment says nothing about the `normalize=False`
  half of the plane at depth 6.
- **One budget** (10,000 episodes). EXP-036 Claim 5 stands: no statement of the form "the
  architecture cannot do depth N" is supported.
- **Three betas** spanning to 94% of the entropy ceiling. Betas beyond 0.8 are untested, though
  Claim 4's turn suggests they would make modal fraction worse rather than better.
- **Depth 5 got one beta**, not the full dose axis.
- The depth-6 measurement is resolution-bound at 1/200 per seed. A true effect below ~0.005
  could not be seen there, which is exactly why Claim 3 exists.

## A note on the cost model

Estimated 21 h from EXP-037's measured throughput; actual **22 h 47 m**, 8.6% over. That is the
first cube estimate in this project to land close.

**A mid-run revision to ~28.9 h was WRONG and worse than the original.** It extrapolated a rate
from a partial record count while the expensive depth-6 cells dominated the completed set; the
cheaper depth-5 tail then finished faster than the extrapolation implied. **Do not re-forecast
from a partial sample that is not representative of the remaining work** - the up-front model
built from matched-workload throughput was the better instrument.

## Regenerate

```bash
.venv/bin/python -u experiments/038_depth6_collapse/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10

# EXP-036's records MUST be present: they carry all four comparators (depth 6 random 0.0008
# and trained 0.0000, depth 5 trained 0.0396 and random 0.0000) and every claim is paired
# against them. They are gitignored, so re-fetch from the laptop if outputs/ is empty.
.venv/bin/python experiments/038_depth6_collapse/aggregate.py
```
