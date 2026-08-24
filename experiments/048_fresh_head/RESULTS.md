# EXP-048 Results - the encoder really did improve, and its own head was holding it back

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract
> was committed at `652b517` **before any number existed**, `aggregate.py` was written before
> dispatch, and the interpretation grid was fixed in advance so the reading could not be chosen
> afterwards. **No threshold was edited while filling this in.**
>
> **Provenance:** 12 records, one per seed. Depth 6, 10,000 episodes, curriculum `(1..6)`,
> `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`, `normalize_advantages=False`,
> `encoder_state_path` = EXP-047's confirmatory fine-tuned encoders, **`encoder_lr=None` (frozen,
> 390 trainable parameters)**. Laptop `SwizzlesDuo`, 6 workers, dispatched 2026-08-22 23:42,
> interrupted by the laptop sleeping and resumed to completion by 2026-08-23 20:58. Runs are
> seeded and byte-identical across scheduling, so the interruption changed nothing. Arms A
> (EXP-043) and C (EXP-047) were **not** re-run. Records in `outputs/` (gitignored); `*_head.pt`
> tracked. Regenerate at the bottom.

## Headline

**EXP-047's gain was not co-adaptation. The encoder genuinely improved - and pairing it with a
fresh head is, if anything, better than keeping the head it was trained with.**

| arm | encoder | during RL | head | mean | sd |
|---|---|---|---|---|---|
| A | EXP-040 pretrained | frozen | fresh | 0.1800 | 0.0985 |
| **B** | **EXP-047 fine-tuned** | **frozen** | **fresh** | **0.3112** | 0.1140 |
| C | EXP-047 fine-tuned | trained jointly | co-adapted | 0.2700 | 0.0810 |

**Claim 1 CONFIRMED: B - A = +0.1312, W-L-T 11-1-0, exact p 0.0059.**

This was the outcome I expected least. The pre-registration said a refuted Claim 1 would be the
most informative result, because it would align three independent measurements. It refuted the
*opposite* way.

## Claim 1 (PRIMARY) - did the encoder itself get better? CONFIRMED

Pre-registered: **>= +0.05** at **p <= 0.05**, paired, exact permutation over `2**12`.

| seed | A | B | C | B - A |
|---|---|---|---|---|
| 0 | 0.210 | 0.445 | 0.380 | +0.235 |
| 1 | 0.260 | 0.395 | 0.245 | +0.135 |
| 2 | 0.005 | 0.355 | 0.230 | +0.350 |
| 3 | 0.195 | 0.285 | 0.285 | +0.090 |
| 4 | 0.105 | 0.290 | 0.190 | +0.185 |
| **5** | 0.165 | **0.005** | 0.275 | **-0.160** |
| 6 | 0.000 | 0.230 | 0.115 | +0.230 |
| 7 | 0.275 | 0.300 | 0.315 | +0.025 |
| 8 | 0.225 | 0.365 | 0.345 | +0.140 |
| 9 | 0.210 | 0.295 | 0.210 | +0.085 |
| 10 | 0.195 | 0.370 | 0.255 | +0.175 |
| 11 | 0.315 | 0.400 | 0.395 | +0.085 |

**Seed 5 is the single regression and it is a collapse, not a decline**: 0.165 -> 0.005, with the
same encoder that gave 0.275 in arm C. A fresh head on a good encoder can still fail; the
failure mode from EXP-040 (2 of 12 seeds collapsing on normal-looking inputs) has not gone away.
Arm B has **0** seeds at exactly 0.0000 against arm A's 1, so this is a shift in which seeds
fail, not an increase.

## Claim 2 - is B below C? NO, AND THE POINT ESTIMATE GOES THE OTHER WAY

**B - C = +0.0412, W-L-T 9-2-1, exact p 0.2393.**

> [!warning] This is **not** evidence that B beats C, and the spec pre-committed not to call a
> null equivalence either. n=12 establishes neither. The honest sentences are: *"B is not
> detectably worse than C"* and *"the point estimate favours B, at p 0.24"*.

**Claim 3, retention** = `(B - A) / (C - A)` = `+0.1312 / +0.0900` = **1.46**. Above 1.0, which
would mean the co-adapted head was a mild liability. Given Claim 2's p of 0.24, treat 1.46 as
"at least 1.0, plausibly more" rather than as a point value.

### Why a fresh head could beat the head that was trained alongside

There is a clean mechanism, and it makes predictions. In EXP-047 the head chased a **moving
target**: every early gradient was computed against an encoder that no longer exists by the end.
The final head is partly fitted to intermediate encoder states. A fresh head trains 10,000
episodes against a **fixed, already-good** encoder, so every gradient is on-target.

A second observation supports this. **Arm B's generalisation gap is 0.0000** (per-seed -0.090 to
+0.075, averaging to exactly zero by coincidence), against **+0.0433** for arm C. The co-adapted
pair overfits slightly; the fresh head on a frozen encoder does not overfit at all.

This suggests a **two-stage recipe** - fine-tune the encoder, then throw the head away and retrain
it - rather than end-to-end joint training being the thing that works.

## THE BUDGET ACCOUNTING, AND A CORRECTION TO THIS EXPERIMENT'S OWN SPEC

> [!important] **The spec was wrong on this point and it matters.** Section 2 says *"Arm B costs
> 1.0x per step, not 1.33x - it is frozen. So unlike EXP-047 there is no compute confound to
> price: B and A are compute-identical."*
>
> That is true **per step** and **false for the pipeline.** Arm B's encoder is the product of
> EXP-047's 10,000 episodes at 1.33x. Arm B therefore embodies **~2.33x** arm A's total RL
> compute, and there is very much a confound to price.

Priced against EXP-046's curve (0.22 success per log10 of spend at depth 6):

| arm | total compute vs A | budget-equivalent | actual gain | **beats the curve by** |
|---|---|---|---|---|
| C | 1.33x | +0.0272 | +0.0900 | **+0.0628** |
| B | 2.33x | +0.0808 | +0.1312 | **+0.0504** |

**This substantially tempers the headline.** +0.1312 sounds like a large step beyond EXP-047's
+0.0900; measured against what the extra compute alone buys, the two arms beat the budget curve
by **about the same amount**, ~0.05-0.06. The consistent reading is: *the fine-tuning pipeline is
worth roughly +0.05 over spending the same compute on more episodes* - not that each stage
compounds.

For orientation, EXP-046 measured depth 6 at 4.4x compute = **0.3225**. Arm B reaches **0.3112**
at **2.33x**. Better than the curve, and cheaper than 4.4x, but not a different regime.

## Claim 4 - the pre-registered interpretation grid

Claim 1 CONFIRMED, Claim 2 not significant. The grid, fixed in the spec before the data:

> **THE ENCODER GENUINELY IMPROVED.** EXP-047's leak-free probe was the wrong instrument, or too
> insensitive at n=12. The probe's negative depth-6 delta still needs explaining.

That last sentence is now the open problem, and it is a real one.

## The unexplained tension - the probe says one thing, the policy says another

EXP-047's probe on these same encoders measured, at depth 6, the depth everything is scored at:

- standard split: **-0.0097** (the gain went negative)
- leak-free slice: **+0.0050**, p 0.5732

Yet a fresh linear head on those same frozen encoders scores **+0.1312** over the original
encoder, 11-1, p 0.0059. **Both readouts are linear on the same 64-dimensional concept.** They
should not disagree this sharply, and one of them is measuring the wrong thing.

Candidate explanations, none tested:

1. ~~The probe scores top-1 optimal-move accuracy while the policy only needs a distance-reducing
   move.~~ **WRONG, corrected 2026-08-24 before any follow-up was run.** `optimal_move_mask`
   already means *"which moves strictly reduce distance-to-solved"*, and `top1_accuracy` already
   asks whether the argmax move is one of them. The two metrics were never different in the way
   this claimed. Left visible rather than deleted, because it was published in this file, in the
   vault and in a commit message as "the explanation I find most likely".

2. **THE LIKELIEST EXPLANATION, and this project has already evidenced it: the probe measures
   CAPACITY, the policy measures EXPLOITABILITY.** The probe trains to convergence with full
   oracle supervision, so it reports the best a linear readout could do. REINFORCE trains on
   sparse reward, so the policy reports what it could actually find. **EXP-028 already
   established this family is optimization-limited rather than encoder-limited** - gaussian noise
   *doubled* held-out navigation there. Fine-tuning could leave decodability flat while making
   the code far easier for REINFORCE to exploit, which is exactly the pattern seen.

3. **Single-step versus chained.** The probe scores one move from a state; the policy chains up
   to `2d+3` of them, so small per-step advantages compound.
4. **Different feature pipelines.** The probe batches `SensoryCortex`; the policy loops
   `brain.step`. EXP-039's Control B established these agree in *distribution*, not per-sample.
5. **The probe fits jointly across depths 1-6** with a stratified split, so its depth-6 numbers
   come from a readout shaped by shallow shells the policy never evaluates on.

**This is the most valuable open question in the project right now**, because the probe has been
the mechanism instrument for EXP-033, EXP-039 and EXP-047. If it does not track policy-relevant
representation quality, several mechanism readings need revisiting.

## What this does and does not license

**Licensed:**
- "EXP-047's gain was not co-adaptation. The fine-tuned encoder is better, and it transfers to a
  fresh head."
- "The two-stage recipe reaches 0.3112 at depth 6 against a 0.1800 baseline."
- "Training a fresh head on the frozen fine-tuned encoder is at least as good as keeping the
  co-adapted head, and removes the generalisation gap."

**NOT licensed:**
- "Fine-tuning compounds with retraining." Both arms beat the budget curve by ~0.05; the second
  stage mostly buys back what its own compute costs.
- "The RL objective is what improved the encoder." Still uncontrolled - see spec section 4. Arm
  B's encoder had 10,000 episodes of extra shaping of *some* kind.
- "B beats C." p 0.2393.

## What to do next

1. **Resolve the probe tension.** ~~Score the probe on "any distance-reducing move".~~ That was
   based on a misreading and the metrics are already the same thing. The corrected test is
   **capacity versus learnability**: fit the same probe at increasing epoch budgets (5, 15, 50,
   300) on the encoders already serialised. If B only matches A at convergence but reaches it far
   sooner, capacity is flat and *learnability* improved - which would reconcile the probe with
   the policy and put EXP-028's "optimization-limited" finding at the centre of the cube line
   too. Offline, minutes. **Do this first.**
2. **Iterate the two-stage recipe.** Fine-tune again from arm B's encoder and retrain a head. If
   the budget accounting above is right, the third stage should yield about +0.05 again over its
   own compute cost, not more.
3. **Diagnose seed 5**, which collapsed to 0.005 from an encoder that scored 0.275 in arm C. That
   is a head-initialisation failure on a known-good encoder, and it is the cleanest instance of
   the EXP-040 collapse the project has.

## Regenerate

```bash
.venv/bin/python -u experiments/048_fresh_head/run.py --workers 6
.venv/bin/python experiments/048_fresh_head/aggregate.py
```

Seeded runs are byte-identical across worker scheduling, so a re-run reproduces every number
above exactly. This one was interrupted mid-flight by the host sleeping and resumed without
incident, which is that property being relied on rather than merely asserted.
