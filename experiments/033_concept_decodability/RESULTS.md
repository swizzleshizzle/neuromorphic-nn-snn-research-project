# EXP-033 Results - Concept Decodability (is the representation the wall?)

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. **Provenance:** 48
> probe fits (4 concept widths x 12 seeds), 2,943 states covering distances 1-5, 25% held out stratified
> by depth. Run 2026-08-01 on the laptop `SwizzlesDuo` over SSH from the VPS with `--workers 16`, wall
> clock 38m (14:38:02 to 15:16:26), exit 0. The oracle-ceiling measurement in the second half was run on
> the VPS. Records in `experiments/033_concept_decodability/outputs/` (gitignored). **The contract was
> committed in `run.py` before any number existed** (commit `c81d916`). Regenerate at the bottom.

## The question

EXP-032 ruled out optimisation-by-stochasticity: de-collapsing the policy made it *worse* at solving
cubes. That left the representation as the suspect. The cube brain is **frozen at random
initialisation** and only a `Linear(64 -> 6)` head trains, which is **390 trainable parameters**, so
nothing in the system can learn what a move does to a cube.

**No reinforcement learning happens in Part 1.** A linear probe is fit from features to "which moves
reduce distance-to-solved", offline. That isolates representation from optimisation, which every prior
cube experiment confounded.

Distance-to-solved builds labels for an offline probe only. It is never a model input.

## Held-out top-1 optimal-move accuracy

The policy trains on depths 1-3; depths 4-5 are included to see whether decodability degrades.

| features | d1 | d2 | d3 | d4 | d5 | pooled |
|---|---|---|---|---|---|---|
| chance (measured) | 0.167 | 0.183 | 0.181 | 0.182 | 0.194 | 0.191 |
| **facelets (144d)** | **1.000** | **1.000** | **0.956** | 0.766 | 0.598 | 0.648 |
| concept @ 64 | 0.583 | 0.833 | 0.631 | 0.459 | 0.377 | 0.407 |
| concept @ 128 | 0.875 | 0.893 | 0.742 | 0.517 | 0.420 | 0.456 |
| concept @ 256 | 0.875 | 0.905 | 0.789 | 0.576 | 0.461 | 0.501 |
| concept @ 512 | 0.792 | 0.952 | 0.825 | 0.638 | 0.479 | 0.528 |

The measured chance floor is 0.191, not 1/6 = 0.167, because states differ in how many optimal moves
they have. Assuming 1/6 would have overstated every result by about 2.4 points.

## Finding 1: width helps and is refuted as the lever

Pooled accuracy by width, with the gain from each doubling:

| width | pooled | gain |
|---|---|---|
| 64 | 0.407 | |
| 128 | 0.456 | +0.049 |
| 256 | 0.501 | +0.044 |
| 512 | 0.528 | +0.027 |

Eight times the width closes about half the gap to the facelets probe and the doublings are clearly
saturating. The residual gap at width 512 is **+0.120** against a **pre-registered sufficiency bar of
0.050**. Extrapolating the halving increments, an unbounded random projection lands near 0.58, still
short of the 0.648 a linear map extracts from the raw observation.

**Contract outcome A is confirmed on this comparison:** the frozen random encoder destroys information
that is present in the observation, and adding neurons does not recover it. Most starkly at depth 1,
where a linear probe on raw facelets is **perfect (1.000)** and the shipped concept@64 manages 0.583.

## Finding 2: the representation is NOT the first bottleneck

This measurement was **not** in the pre-registration, and it is the one that matters.

Take the *same* frozen concept@64 the RL policy uses, fit the linear probe on it with supervised
labels, then run that probe as the greedy policy in the real environment:

| depth | oracle-probe policy | actual RL policy | ratio |
|---|---|---|---|
| 1 | 0.708 | 0.875 | 0.8x |
| 2 | 0.738 | 0.380 | 1.9x |
| **3** | **0.481** | **0.022** | **22x** |

Identical representation, identical head shape, identical environment. The only difference is how the
390 parameters were chosen. **At depth 3 the frozen random concept supports 48% success; REINFORCE
extracts 2.2%.**

So the information needed to solve depth-3 cubes is already present in the representation the policy
has been reading all along, and linearly extractable by exactly the head it already has. The failure
is that reinforcement learning never finds those weights.

**CORRECTION (2026-08-02, EXP-035): this is a reference point, NOT a ceiling.** Seven of twelve
RL seeds later exceeded 0.481, topping out at 0.633. The probe was optimised for per-step
move-optimality while the task rewards solving, and a 9-step budget on a 3-move scramble leaves slack
for non-optimal moves. The claim this experiment actually establishes is unaffected: the frozen
representation carries far more usable signal than v1 extracted.

**This is an oracle-supervised reference, not an agent.** Its labels come from the distance table, so it is not a
legitimate policy. It measures what a supervised readout extracts from this representation. That is
precisely what makes it the right diagnostic: it separates "the features cannot support the task" from
"the learning algorithm cannot find the solution."

Why the learning fails is consistent with everything else on record: at depth 3 the reward is
essentially never obtained, so REINFORCE has almost nothing to reinforce across its 600 episodes. That
also retro-explains EXP-032, where adding entropy did not help because exploration was never the
problem, and EXP-030, where memory could not help a policy that had not learned anything to begin with.

## Where the pre-registration fell short

The contract offered three outcomes: A (facelets high, concept low), B (both low), C (concept ~
facelets and both high). The observed pattern is **A on the probe comparison and the spirit of C on the
policy comparison**, which no branch covered: concept meaningfully below facelets, yet far above what
the RL policy achieves.

Recording this rather than retrofitting the result into branch A. The contract was useful precisely
because its gap is now visible instead of invisible.

## Limitations

- **Depth 1 held-out is 2 states per seed** (25% of a 6-state shell). The `d1` column is noisy, and RL
  beating the oracle probe there is most likely that plus the probe being fit jointly across depths
  1-3 rather than specialised to depth 1. Do not read the depth-1 inversion as a result.
- **The oracle probe uses privileged labels.** It bounds what the representation supports; it says
  nothing about whether an unsupervised learner could reach it.
- **One episode budget (600) and one learning rate.** The learning-signal hypothesis is consistent with
  the data but is not yet tested directly.
- Probe capacity is fixed at a linear map by design, to match the policy head. A nonlinear probe would
  answer a different question.

## Lead for the next experiment

**Priority inverted from where this experiment started.** Encoder pretraining was the plan; it should
now be second, because Finding 2 shows the current encoder already carries 22x more usable signal at
depth 3 than the policy exploits.

1. **Fix the learning signal first.** Cheapest decisive test: keep the frozen concept@64 and the linear
   head exactly as they are, and vary only how the head is trained. A curriculum (train depth 1, then
   2, then 3) and a larger episode budget are both cheap and neither requires privileged information at
   run time. The pre-registered target is straightforward: RL success at depth 3 should move toward
   0.481. That is a *measured* reference rather than an assumed one, which is what makes it a fair bar.
   (EXP-035 later exceeded it; see the correction above.)
2. **Then engage the encoder.** Finding 1 stands: the encoder does lose real information, up to 40
   points at depth 1. Once the learning signal is not the binding constraint, that gap becomes worth
   closing, and this probe gives a measured way to check whether pretraining actually raised it.

**Do not pursue width.** It is refuted against a bar set before the numbers existed.

## Regenerate

```bash
.venv/bin/python -u experiments/033_concept_decodability/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16
```
