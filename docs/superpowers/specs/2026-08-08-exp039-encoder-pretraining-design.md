# EXP-039 design - does inverse-model pretraining raise the probe ceiling?

**Status: pre-registration. Written 2026-08-08, before any EXP-039 number exists.**
Every threshold below is fixed. If one is edited after data arrives, that edit is the finding
and must be reported as such.

This is **vault Stage 2, first increment**: the moment the SNN stops being a fixed random
projection. See `300 Efforts/Active/Coding/Neuromorphic Development/road-to-a-solved-cube.md`.

## 1. The question

Everything the cube line has achieved runs on a **frozen randomly-initialised brain** and a
`Linear(64 -> 6)` head: 390 trainable parameters. The vault's Wall 1 is the reason that has a
ceiling, and it is measured in our own data (EXP-033, top-1 accuracy of a linear probe on
"which moves reduce distance-to-solved"):

| depth | 3 | 4 | 5 | pooled |
|---|---|---|---|---|
| chance (measured) | 0.181 | 0.182 | 0.194 | 0.191 |
| **facelets (144d)** | 0.956 | **0.766** | 0.598 | 0.648 |
| concept @ 64 (shipped) | 0.631 | **0.459** | 0.377 | 0.407 |
| concept @ 512 | 0.825 | 0.638 | 0.479 | 0.528 |

Two facts set up this experiment:

1. **The frozen concept is far below what the raw observation supports** - 0.459 against 0.766
   at depth 4. The representation is throwing information away.
2. **Width is refuted as the route to closing it** (EXP-033 Finding 1). Eight times the width
   reaches only 0.638, still below the facelets probe, and the doublings are saturating.

So the remaining move is to **train** the encoder rather than widen it. A linear probe on a
*trained* encoder is a **nonlinear** function of the observation, so unlike width it is not
bounded by the raw-facelet linear ceiling at all.

**This experiment involves no reinforcement learning.** It is supervised pretraining plus an
offline probe, which makes it cheap, fast to iterate, and runnable on the VPS.

## 2. The intervention

**Inverse-model pretraining**, per the vault: predict the move from a state pair.

```
concept = SensoryCortex(n_obs=144, hidden=128, concept=64)     # the SHIPPED config
inverse_head = Linear(2 * 64 -> 6)
loss = CrossEntropy(inverse_head([rate(concept(s)), rate(concept(s'))]), a)
```

Trained by BPTT through snntorch's surrogate gradients. `SensoryCortex` is
`Linear(144->128) -> Leaky -> Linear(128->64) -> Leaky`, about **26,816 trainable parameters**
against the policy head's 390.

**Width stays at 64**, deliberately. It matches the shipped policy exactly, makes the
comparison against 0.459 direct, and EXP-033 already refuted width as a lever. This experiment
changes **one thing**: whether the encoder is trained.

**Fully self-supervised. No oracle labels.** The move `a` is known because we applied it.
Distance-to-solved appears nowhere in training - it builds probe labels offline only, exactly
the instrument/input line `CLAUDE.md` draws.

### The pre-registered risk, stated plainly

> An inverse model learns **what a move did**, not **which move is good**. The probe asks the
> second question. **Whether inverse-dynamics pretraining transfers to optimal-move
> decodability IS the hypothesis under test.**

A null is therefore a real finding about representation learning, not a failed build. It would
say: learning cube dynamics does not by itself make optimality linearly readable, and Stage 2
needs a different objective (a value/heuristic target, which trades away some of the
"learns through experience" claim - see section 7).

## 3. Arms

Four, at matched states and matched splits, 12 seeds.

| arm | what it is | role |
|---|---|---|
| **trained** | concept@64 after inverse-model pretraining | the intervention |
| **frozen** | concept@64 at random init | the paired control |
| **facelets** | raw 144-d one-hot | the linear ceiling on the observation |
| **chance** | measured from the label set | the floor |

## 4. Two controls that decide whether this is valid at all

### 4a. The probe's held-out states must be excluded from pretraining

If the encoder pretrained on the states the probe scores, "held-out" accuracy measures an
encoder that has already seen them. The inverse model never sees distance labels, so it cannot
memorise optimality directly - but it can memorise state-specific structure, and that is enough
to inflate the number.

**Rule: a pretraining pair is dropped if EITHER endpoint is a probe-held-out state.** Both
endpoints, not just `s`, because `s'` is fed through the same encoder and its facelets are seen
just as directly.

*Accepted cost:* dropping by both endpoints slightly biases the pair distribution toward the
interior of the train set. Recorded as a limitation; correctness beats coverage here.

### 4b. The frozen arm is re-measured through THIS pipeline, not taken from EXP-033

EXP-033 computed features per state via `brain.step` (B=1). This experiment batches
`SensoryCortex` directly, which is far faster but **consumes the Poisson generator
differently**, so the spike draws are not the same. Exact reproduction of 0.459 is therefore
not expected and its absence would not be a defect.

**Consequence: 0.459 is an EXTERNAL SANITY CHECK, never the internal comparator.** Bar 1 is
paired against the frozen arm measured here, on the same seeds, through the same code.

> If the frozen arm lands outside **0.459 +- 0.10** at depth 4, stop and explain it before
> reading anything else. That would mean the batched pipeline is not measuring what EXP-033
> measured, and every number in the experiment inherits the discrepancy.

### 4c. Instrument check on the batched features

The batched feature extractor must reproduce `concept_rate(brain.step(...))` **in
distribution** for a single state - same mean rate to within Monte-Carlo error over repeated
draws. Exact equality is impossible (different RNG consumption) and asserting it would be a
test that cannot pass; asserting nothing would be a test that cannot fail.

## 5. The pre-registered contract

All tests are **exact paired permutation over all 2^12 = 4096 sign flips**, two-sided, on
matched seeds. No scipy in the venv.

### Bar 1 (PRIMARY) - did the encoder learn anything usable?

Probe top-1 at **depth 4**, trained vs frozen, paired per seed.

> **CONFIRMED**: mean delta **>= +0.05** with **p <= 0.05**.
> Otherwise **REFUTED**: inverse-dynamics pretraining does not improve optimal-move
> decodability at the frontier depth.

+0.05 is chosen against EXP-033's own measured spread: it is roughly the gap one width doubling
buys (0.459 -> 0.517), so the bar is "at least as much as doubling the width", against an
intervention that costs no extra width at all.

### Bar 2 (THE THESIS BAR) - does the encoder supply nonlinearity?

> **CLEARED** if trained concept@64 at depth 4 exceeds the **facelets arm measured here**,
> paired per seed. EXP-033's published **0.766** is an external check, not the comparator.

> [!note] Revised 2026-08-08 after the calibration pilot, before any EXP-039 result
> The first draft tested against the published 0.766. The pilot showed why that is wrong for
> the same reason Control B exists: this experiment fits the probe jointly over depths **1-6**
> where EXP-033 used **1-5**, and depth 6 is 75% of the states, so the joint fit sits deeper
> and shallow depths shift. Measured: facelets reads **0.769** at depth 4 (against 0.766, fine)
> but **0.900** at depth 3 (against 0.956). Depth 4 happens to agree; depth 3 does not, and
> relying on that coincidence would be luck rather than method.
>
> The pilot also confirmed the pipeline reproduces all three reference arms at depth 4:
> facelets 0.769 vs 0.766, frozen 0.463 vs 0.459, chance 0.184 vs 0.182.

This is the claim that matters for the project. A linear probe on the trained concept beating
*any* linear map on the observation means the encoder is contributing genuine nonlinear
structure - it is the first moment the SNN earns its place, and width provably cannot get there
(concept@512 reaches 0.638).

**Bar 2 is deliberately hard and may well not clear on the first objective.** It is stated in
advance so that clearing Bar 1 alone is not later described as if it cleared Bar 2.

### Bar 3 - does it hold across depth?

Report all of depths **3, 4, 5, 6** for every arm. Pre-registered reading:

- Gains at depth 3 alone are the **least** interesting: the frozen concept already reaches
  0.631 there and facelets 0.956, so there is little room and the policy already works.
- **Depth 5 and 6 are where Wall 1 bites hardest** (facelets 0.598 at depth 5, and the RL
  policy is at 0.0396 and 0.0000). A gain that grows with depth is a materially stronger result
  than a uniform shift, and must be reported as such.

Depth 6 requires `ExactBFSDistance(max_depth=7)`: labelling a depth-6 state needs its
neighbours' distances, and `optimal_move_mask` correctly refuses rather than guessing when a
neighbour falls outside the table.

### Bar 4 - the null is pre-committed as a result

> If Bar 1 refutes, report it as: **inverse dynamics is not sufficient to make optimality
> linearly readable.** That is a finding about what "learning how the cube works" buys, and it
> redirects Stage 2 to a value/heuristic objective rather than leaving the stage open-ended.

## 6. Implementation constraints

- **Reuse EXP-033's `probe.py` unmodified**, imported by file path (`experiments` is not a
  package and its directories start with digits, so a normal import is impossible). Reimplementing
  the probe would make the comparison against EXP-033 meaningless.
- **The core belongs in `src/neuromorphic/training/encoder_pretrain.py`**, not in the experiment
  directory: the Stage 2 follow-on (RL with a pretrained encoder) needs the same machinery, and
  an experiment-local copy would fork immediately.
- **Do not modify `SensoryCortex`.** It is already differentiable through snntorch's surrogate
  gradients and `_record` is a no-op when recording is off. If it must change, every frozen-arm
  number in EXP-033 becomes non-comparable.
- **No assertion that cannot fail.** Every test added must fail against the pre-change code.
- Pretraining hyperparameters (epochs, batch size, lr) are **calibrated by a smoke run** and
  recorded, not guessed - the EXP-038 pilot's lesson. See section 6a.

## 6a. Calibration, run 2026-08-08 before the real run

Seed 0, depths 1-6, 48,233 pairs, batch 256, 40 epochs, ~1,015 s per run on the VPS.

| lr | move-acc @0 | @20 | @39 | depth-4 frozen -> trained |
|---|---|---|---|---|
| 1e-3 | 0.198 | 0.394 | 0.425 | 0.463 -> 0.791 |
| **3e-3** | 0.237 | 0.432 | **0.455** | 0.463 -> 0.784 |

**Selected: lr 3e-3, 40 epochs.**

> [!important] The selection rule, and why it is not "whichever scored higher"
> **Hyperparameters are chosen by the PRETRAINING OBJECTIVE (move-naming accuracy), never by
> the probe.** Selecting on the probe would tune the outcome metric on the very quantity the
> pre-registered bars are about, and every bar below would then be reporting a choice rather
> than a finding.
>
> On that rule lr 3e-3 wins: 0.455 against 0.425 move-accuracy in the same budget. Note it
> would have LOST on the probe (0.784 against 0.791), which is exactly the situation the rule
> exists to decide, and a reminder that the two metrics are not the same axis.

**Move-naming accuracy has not saturated at 40 epochs** (0.447 -> 0.456 -> 0.455 over the last
ten). 40 is a budget chosen for turnaround, not a converged optimum, and that is a stated
limitation: a longer budget is untested and might move every arm.

**n=1 signal, recorded so the real run cannot be described as confirming a prediction it never
made:** the calibration seed put trained depth-4 at 0.784-0.791 against a frozen 0.463 and a
locally-measured facelet ceiling of 0.769. If that holds at n=12 it clears Bar 1 by a wide
margin and clears Bar 2 by a narrow one. **One seed is not a result** - EXP-026's n=5 lied and
the de-noised result flipped.

## 7. What this experiment cannot say

- **Nothing about policy success.** It measures what the representation *supports*, not what
  REINFORCE extracts from it. EXP-033 Finding 2 is the caution: an oracle probe at depth 3
  supported 48% while the RL policy managed 22%. Whether a raised ceiling converts into a better
  policy is the **Stage 2 follow-on**, and it needs the laptop.
- **Nothing about objectives other than the inverse model.** A distance-regression arm would
  very likely score higher - it is nearly the probe's own label - but it trains on the oracle
  and weakens the "learns through experience" claim. Deliberately excluded, not overlooked.
- **Nothing about width.** Fixed at 64 by design.
- **Nothing about depths past 6**, and depth 11 is where a random scramble lives.
