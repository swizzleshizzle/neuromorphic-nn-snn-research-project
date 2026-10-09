# A five-region spiking brain learns a 2x2 cube, mostly for ordinary reasons

**Research report, DRAFT 0.** Started 2026-09-25 (week 25), ahead of Phase 4 (Oct 5 to Dec 27).

> **Status of this draft.** Every section is written, sections 1 to 9 and Appendix A.
>
> **Ground rules for the whole report.** Every number cites the experiment that produced it.
> Every claim that was pre-registered says whether it was confirmed, refuted, void or unresolved,
> in the spec's own terms. Negative results get the same space as positive ones.

---

## 1. Summary

This project spent a year building a five-region spiking neural network (sensory cortex,
hippocampus, prefrontal cortex, a thalamic router and a motor region, 510 neurons) and training it
with reinforcement learning to solve a 2x2 Rubik's cube. It worked, in the sense the plan defined:
the regionalized network solves 2x2 cubes from scrambles up to **depth 8** (0.0783 success against
a measured chance floor of exactly 0.0000, EXP-062), far past the 1-move checkpoint the plan set
and the 3-move stretch goal.
Since then, a search wrapped around the trained networks, with a judge the network learned from
its own look-ahead, has reached **0.1933** at depth 9 (EXP-074) and, in an exploratory run, about one
typical random scramble in ten (section 9).

**It mostly did not work for the reasons the architecture was built to test.** Across 74 numbered
experiments, 58 of them with a committed results file (Appendix A), almost every architectural
idea lost to a training idea:

| idea | kind | verdict |
|---|---|---|
| a curriculum over scramble depth | training | **the largest single win**: depth 3 went from 2.2% to 50.0% with no architectural change (EXP-034/035) |
| training the spiking encoder, first by self-supervised pretraining and then during RL | training | **works and replicates**: +0.19 at depths 4 and 5 (EXP-040), a further +0.090 at depth 6 (EXP-047, p 0.0020) |
| a learned critic as a baseline | training | **works and replicates**, and it is ordinary RL (EXP-056/060) |
| episodic memory in the hippocampal region | architecture | **hurts**, and the recall is indistinguishable from matched noise (EXP-059/061), even over a perfect cache (EXP-063) |
| a neuromodulatory bus carrying the learning signal | architecture | **bought nothing** once it was made load-bearing (EXP-053) |
| the five-region topology itself | architecture | **not testable in this configuration**: only the sensory region is on the policy path, so no arm-versus-arm contrast can measure topology (section 5.6) |

**The honest headline is narrow.** Training a spiking encoder with surrogate gradients is the one
change that is both neuromorphic in substrate and clearly pays. It is not neuromorphic in its
*learning rule*: the gradients are backpropagated, not local. Everything else that worked is
ordinary reinforcement learning running on top of a characterized spiking feature extractor.

**Two findings are more useful than the solver.** The first is a **budget law**: success at a
given depth is linear in the logarithm of training spend, about 0.22 per decade, with no knee
(EXP-044/045/046, held out of sample by EXP-062). It turned "can the network reach depth N?" from
an architecture question into arithmetic, and the same arithmetic priced the project's own
headline goal (random depth-11 scrambles) out at about 33 days of compute per seed. The second is
the **method** in section 4, which is the reason any of the negative results above can be
believed. It detected, repeatedly and before publication, when the project's own ideas were not
working.

**What a reader can reproduce, and how.** Re-evaluating a committed checkpoint reproduces a
previous re-evaluation exactly, measured across two machines for EXP-036. It reproduces a
*published* per-seed number only where that has been checked: evaluation encodes the cube as
random spikes, and the published runs drew them from a different point in the random stream. For
the depth 7 to 9 checkpoints, 27 of 36 re-evaluated cells differ from their published solved
counts, while each depth's pooled mean agrees within a 95% interval (EXP-070). Retraining from a
seed reproduces neither: on a second machine the per-seed results scatter by up to 0.30 while two
of three pre-registered verdicts survive (EXP-067). Section 7 states the guarantee precisely.

---

## 2. What was asked

This section quotes the targets as they were written, so that section 6 can grade the project
against the plan rather than against a memory of it. Two documents set targets: the original plan
of 2026-03-31, and a harder roadmap the project wrote for itself on 2026-08-02. Quotations keep the
source's words; long dashes in the originals are rendered as colons or commas.

### 2.1 The original plan

The plan (`neuromorphic-project-plan.md`, in the project vault) set a 12-month programme whose
deliverable was *"a regionalized spiking neural network that learns to solve a 2x2 Rubik's Cube
through experience, with observable specialization across brain-inspired functional regions,
running on commodity hardware."* It named three levels of success:

| level | as written |
|---|---|
| Minimum viable | *"A multi-region spiking network where regions demonstrably specialize and cooperate on a spatial reasoning task (grid navigation)"* |
| Target | *"The regionalized system learns to solve a 2x2 Rubik's Cube from any scramble, trained through experience (reinforcement) rather than supervised labels"* |
| Stretch | *"The system demonstrates continual learning: it can learn a second task without catastrophic forgetting of the first"* |

The work was divided into phases, four of them closing on a written checkpoint:

| phase | dates | checkpoint, as written |
|---|---|---|
| 0. Foundations | Apr 6 to May 3 | *"(1) Explain what a LIF neuron does and why it's different. (2) Explain what backprop accomplishes and why it's problematic for spiking networks. (3) Explain what an RL agent needs to learn a task. (4) Read a 20-line PyTorch code block and narrate what each section does."* |
| 1. Single-region SNNs | May 4 to May 31 | *"(1) Working spiking MNIST classifier (95%+). (2) Recurrent SNN handling sequential input. (3) STDP demo showing unsupervised pattern learning. (4) Written 'what I learned' document."* |
| 2. The regionalized brain | Jun 1 to Jul 19 | *"(1) Architecture spec document. (2) Working multi-region SNN on grid navigation. (3) Monitoring dashboard. (4) Evidence of regional specialization. (5) Honest assessment of what works and doesn't."* |
| 3. The Rubik's cube challenge | Jul 25 to Sep 27 | *"(1) Regionalized SNN solves 2x2 cubes from at least 1-move scrambles (3-move is stretch). (2) Comparison vs monolithic on at least one metric. (3) Interpretability analysis. (4) Clear documentation."* |
| 4. Documentation and capstone | Oct 5 to Dec 27 | no checkpoint; technical documentation, a research write-up, and extension planning |

Three details of the plan matter later. Its cube environment specified *"24 facelets, 12 possible
moves"* and a motor region expanded *"from 4 to 12 actions"*; the system uses 6, for the reason
given in section 3. Its Phase 3 training protocol asked for a *"monolithic baseline comparison
(same neuron count, single region)"*, which is the comparison that section 5.6 finds cannot answer
the question it was meant to. And its extension list for Phase 4 (scaling, neuromorphic hardware,
*"replacing surrogate gradients with fully local learning"*, continual learning) is the list
section 8 answers.

### 2.2 The roadmap the project set itself

On 2026-08-02, immediately after the curriculum reached 50% at depth 3 (EXP-035), the project wrote
a harder roadmap (`road-to-a-solved-cube.md`, also in the vault), on the grounds that the Phase 3
checkpoint had already been passed and the plan's *target*, a cube solved *"from any scramble"*,
had not. It set four stages:

| stage | goal, as written |
|---|---|
| 1 | *"prove generalisation, find the break point"* |
| 2 | *"unfreeze the brain (the architectural moment)"*: train the encoder, by cube-dynamics pretraining or end to end |
| 3 | *"dense signal"*: a *"value function carried through the `neuromod` pathway"* |
| 4 | *"full 2x2 (the actual deliverable)"*: depth-11 random scrambles, judged *"genuinely achievable... nothing about it requires new science"* |

Stage 4 is the plan's target restated as a number: 11 is the modal distance of a random 2x2 from
solved under this move set (a property of the exact BFS table, not an experimental result; stated
in `docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md` section 1). It is the harder of the two tests, and section 6 grades against both.

### 2.3 What changed on 2026-10-05

The plan's Phase 4 included tagging a public release. That came from a parallel media and
publishing track which was dropped on 2026-06-25, and on 2026-10-05 the release itself was dropped
as well. **The goal the project now works towards is the one the plan called its target: a spiking
network that solves a randomly scrambled 2x2 cube.** This report is documentation of the work so
far, not a release, and section 9 points at the road from here.

## 3. The system

This section gives enough of the architecture and the task to follow section 5, and states the
one fact that decides what section 5 can and cannot measure. The authoritative descriptions are
`docs/architecture-spec-v3.md` for the regions and wiring and `docs/adr/0001-multi-region-training-strategy.md`
for what trains; the figures below were checked against `src/neuromorphic/`.

### 3.1 The brain

Five regions of leaky integrate-and-fire (LIF) neurons, built with snnTorch on PyTorch, 510
neurons in all:

| region | neurons | structure | role in the design |
|---|---|---|---|
| sensory cortex | 192 | feedforward, 144 inputs to 128 hidden to a 64-neuron **concept** | turns the observation into a compact code |
| hippocampus | 150 | recurrent, a Hebbian attractor with store and recall | episodic memory of states already seen |
| prefrontal cortex | 150 | 100 state neurons and 50 transform neurons | combines concept and recall into a utility per action |
| thalamic router | 12 | two per action | gates which utilities reach the motor region |
| motor cortex | 6 | one per action, winner-take-all | selects the move |

A neuromodulatory bus runs alongside the regions and carries a global reward (dopamine) signal.
Every region shares the same neuron parameters (decay 0.9, threshold 1.0, reset by subtraction),
and each decision is a window of 32 simulation steps. Spikes are binary; what one region passes to
another is a population code, not a decoded number. One full step of the brain costs about 90 ms
on the project's machines, which dominates every runtime in the project.

Training is by **surrogate gradients**: the spike's step function is replaced by a smooth
stand-in on the backward pass, so ordinary backpropagation can run through the network. Nothing in
the system learns by a local rule. A three-factor eligibility trace was built in Phase 2 as a
demonstration (EXP-021) and never used to train anything.

### 3.2 What is on the policy path

**With `recall=False`, which is every arm in this report except the memory experiments, the
policy reads the sensory concept and nothing else.** The action comes from a trainable head on the
64 concept neurons: in the base recipe a single linear layer, 64 x 6 weights and 6 biases, **390
parameters**, trained by REINFORCE. The prefrontal, router and motor regions still run on every
step and their activity can be recorded and visualised, but their output does not reach the
action. The hippocampus is bypassed entirely.

So **318 of the 510 neurons (hippocampus 150, prefrontal 150, router 12, motor 6) are off the
policy path.** This is stated here, before any result, because it decides what an experiment can
measure. A comparison that varies an off-path region measures nothing; a comparison against a
control matched on total neurons measures width, not topology. Section 5.6 follows this through.

The recipe that section 5 arrives at adds three things to that head, all on the same path: a
curriculum over scramble depth (EXP-034/035), a sensory encoder pretrained on cube dynamics and
then fine-tuned during RL (EXP-040/047), and a learned critic as a baseline (EXP-056). The memory
experiments switch recall on and widen what the head reads: the concept plus the hippocampus's
recall code and a familiarity scalar (EXP-059/061), or a learned attention over a cache of
earlier states (EXP-063).

### 3.3 The task

**Observation.** The 24 facelets of a 2x2 cube, each one-hot over 6 colours (144 inputs),
encoded as Poisson spike trains. The network sees raw facelets and nothing else.

**Actions.** Six: clockwise and anticlockwise quarter turns of the U, R and F faces, with the
down-left-back corner held fixed. This deliberately departs from the plan's 12 moves (section
2.1). A 2x2 has no centre pieces, so turning the opposite face is the same as turning the whole
cube and then this face (`U` equals `D'` up to orientation), and holding one corner fixed removes
the redundancy without losing any state up to whole-cube orientation. **The simplification is 2x2-only**: a 3x3 has
fixed centres and needs all six faces, 12 or 18 moves.

**Reward.** Sparse: -1 per move and +10 on reaching solved. Episodes are cut off at a step budget
of `2d + 3` moves for a scramble of depth `d`.

**Difficulty is exact.** Every state's true distance to solved comes from a breadth-first search
over the whole state space, and states are grouped into **shells** by that distance. "Depth `d`"
in this report always means a state exactly `d` moves from solved, never "scrambled with `d`
random moves", which can land closer. The distance is an instrument for choosing and scoring
states. It is never an input to the network, and a reward-shaping option that would use it exists
only as an unused fallback.

**Train and held-out states.** Depths 1 and 2 have 6 and 27 states and are evaluated whole, so
they measure performance on the training distribution. From depth 3, each shell is split into
training and held-out states, with the held-out side capped at 200: 30 states at depth 3, 133 at
depth 4, and 200 from depth 5 on (EXP-029). Unless a section says otherwise, every success rate in
this report is measured on held-out states the policy never trained on.

**Chance is measured, not assumed.** A uniform random policy is run on the same states with the
same budget. Because a random walk can stumble into solved within `2d + 3` moves, the floor at
depth 1 is **20.8%**, not 1/6, falling to 4.3% at depth 2 and 1.4% at depth 3 (EXP-029) and to
exactly 0.0000 by depth 8 (EXP-062). Every "working" verdict in section 5 is made against the
measured floor at its depth.

---

## 4. Method, and why the negative results can be believed

A project that tests its own ideas has every incentive to find that they work. Most of this
report's results are negative, and the reason they can be trusted is that the method was built to
make a false positive expensive and a false negative visible. This section describes that method
through the specific failures that shaped it. Every rule below exists because something went wrong
first.

### 4.1 Pre-registration, and what it bought

From EXP-036 onward, every experiment committed its specification before any number existed:
the arms, the claims, the thresholds, the statistical test, the validity gate, and how each
possible outcome would be read. The aggregator that produces the verdicts was usually committed in
the same change. Afterwards each claim is marked confirmed, refuted, void or unresolved in the
spec's own terms.

The value is concrete. In Phase 2, **EXP-028's headline refuted its own pre-registration**:
the experiment was designed to show that a degraded concept vector would hurt navigation, and
Gaussian noise instead **doubled** held-out navigation from 43% to 83%. Without a written
prediction the result would have been easy to narrate as expected. With one, it had to be
reported as the refutation it was, and it redirected the phase.

Two refinements came later, each from a mistake:

- **The unit of a replication is the verdict, not the number.** EXP-067 fixed in advance that a
  replication which moves a number without moving a verdict has succeeded, so that a small shift
  could not later be spun as failure nor a large one waved away.
- **A threshold you can state is a threshold you can encode.** EXP-068's spec correctly declared
  a band of outcomes unresolvable, but left it in prose; the aggregator printed CONFIRMED for a
  result inside the band. The write-up reports it as unresolved, and the code was deliberately not
  edited afterwards, because "the edit only made the result weaker" is the same argument that would
  justify the reverse.

### 4.2 Controls that could not be fooled

**Three arms, not two.** Every memory experiment compared a memory arm against both an amnesic
arm and a shuffled-memory arm. A two-arm design would have published a false positive **three
times**. In EXP-030, memory beat the shuffle by 10.8 points and the amnesic control by 1.2
(p 0.91); the headline comparison was measuring the harm of *wrong* memory, not the benefit of
*right* memory.

**Measured floors, not assumed ones.** The chance floor at depth 1 is 21%, not 1/6, because a
random walk with a `2d+3` step budget can stumble into the solved state. Every depth has its own
measured floor, and every "working" verdict has to clear both a multiple of it and an absolute bar.

**A control must be a working reference, not just a matched one.** EXP-064 matched its control's
parameter count to 0.36%, and the matching killed the control: a head that scored 0.3229 in an
earlier experiment dropped to 0.0000. That was the third distinct matching failure in the project.
EXP-030 matched a control so closely it became bit-identical to its arm, EXP-063 matched a control
to its arm but neither to the baseline, and EXP-064 matched capacity but not competence.

### 4.3 Gates that decide whether a result may be read at all

Each experiment pre-registers a validity gate: a condition that, if it fails, voids every claim
regardless of how interesting the numbers look. Gates turned out to be the most error-prone part
of the method, because they read like bookkeeping and get less review than the claims they guard.

| failure | what happened | the rule it produced |
|---|---|---|
| EXP-058 | the gate required more stored memories than an episode has steps; **unsatisfiable by construction**, 20.2 h of compute voided | compute the gate's maximum attainable value before committing it |
| EXP-057 | calibrated at depth 3, evaluated at depth 7; passed with 2x margin instead of the claimed three orders | calibrate in the regime it will run in; prefer a ratio to an absolute |
| EXP-064 | both gates passed and both arms scored exactly 0.0000, so the primary contrast was 0 before the run started | gate the comparison's resolution, not just the arm's mechanism |
| EXP-067 | reused a tolerance calibrated where replication noise is zero; it could pass only about one time in three | reusing a threshold needs a power statement in the new regime |

The practice that came out of this is to write down, **before dispatch**, what the threshold can
and cannot detect. EXP-068 and EXP-069 both carry a simulated power table in their specs.

### 4.4 Tests that can fail, and mutation testing to prove it

**Never write an assertion that cannot fail.** Four real defects in this codebase hid behind
assertions that passed regardless, including a chance floor that was one random realisation
replayed across 12 seeds (reported as 33.3% against a true 20.3%), and a hippocampus whose store
operation assigned instead of accumulating, so it held exactly one pattern.

Since EXP-061, aggregators and instruments are **mutation-tested**: the implementation is broken
deliberately, one defect at a time, and each test must fail against the defect it names. This has
caught instruments that were blind to the exact defect they existed to detect. EXP-061's leak
detector compared two quantities taken before the substitution it was checking, so a 50% leak
passed every assertion. EXP-063's freeze test built its fixture under `no_grad`, which disarmed the
test it was part of. In EXP-068, three of eight mutations survived a first pass: a square grid
cannot see a swap of rows and columns, and a saturated p-value is identical whether or not its
random source is seeded.

### 4.5 Seeds, pairing, and a confound worth more than most effects

**At least 12 seeds**, after five seeds produced a result in Phase 2 that flipped once de-noised
(EXP-026). **Paired designs** wherever possible: every arm of a comparison runs on the same seeds,
and the test is an exact permutation over all `2^n` sign flips of the per-seed differences (4,096
for n = 12). The project deliberately has no SciPy; exact tests at this scale are cheap and assume
nothing.

Pairing turned out to matter more than expected. A cube seed fixes the task draw, the training
trajectory and, where used, the pretrained encoder, all at once, and **seed quality is a large,
persistent property**: across 8 arms from five experiments, per-seed success correlates at +0.419,
positive in 28 of 28 arm pairs, with a standard deviation of 0.0906. That puts about 0.039 of noise
on any comparison between two different sets of seeds, against published effects of 0.05 to 0.09,
and exactly none on a paired comparison. An unexplained "level shift" between two seed blocks in
EXP-060 was this and nothing else (exact p 0.5066) (`docs/seed-effect.md`).

EXP-068 then found where the effect lives, and the answer was neither obvious candidate. Crossing
the task-draw seed against the training-trajectory seed at depth 3, the **interaction** carries 65%
of the variance, the task draw 27% and the trajectory 8%. A seed's quality is the specific pairing,
not a property of either part, which is exactly why pairing within seed works. EXP-069 asked the same question at
depth 5 with a pretrained encoder and got the same interaction share, 0.652 against 0.650, while the
main effects changed hands: task draw and trajectory together carry exactly 0.000, and the encoder
carries 0.348, which its pre-registered band reports as unresolved rather than major. Two
decompositions in different regimes agree that a seed's quality is mostly the combination of its parts.

### 4.6 Instruments that were retired

Five instruments that looked like measurements of progress were retired after they were shown to
track a coarse intervention without tracking the outcome: a linear probe of the concept vector,
pretraining move-accuracy, the entropy trace, an encoder-distance statistic, and the critic's
explained variance. **Each worked as a threshold and failed as a gradient.** Several were unanimous
at p 0.0005 across seeds, which measures the instrument's consistency, not its connection to the
result. The report uses the policy's own behaviour instead: success, revisit rate and optimality
(`docs/retired-instruments.md`).

### 4.7 What the method cost

EXP-058 was voided on a gate that could not pass (20.2 h). Two runs were lost to operating-system
restarts on the training laptop (about 19 and 4 CPU-hours); the difference between them was only
whether completed cells had been written to disk, which is why every cell now banks its own record.
Several experiments were run purely as controls or replications, with no chance of a new headline.
The report treats that cost as the price of being able to publish the negative results in
section 5.

---

## 5. Results

Each line of inquiry is reported as a claim, its verdict, and its mechanism where one was found,
in roughly the order the project learned them. Unless stated otherwise, a success rate is the mean
held-out success over 12 seeds, and a p-value is an exact paired permutation test over all 4,096
sign flips of the per-seed differences.

### 5.1 Depth, and the budget law

**The first cube result was a failure, as the pre-registration expected.** The v1 recipe (a frozen,
randomly initialised encoder and a linear REINFORCE head) solved 87.5% at depth 1 and 38.0% at
depth 2, then **2.2% at depth 3 against a measured floor of 1.4%** (EXP-029). The predicted shape,
solid at depth 1, degrading at 2, near chance by 3, was confirmed.

**The policy had collapsed, not run out of capacity.** At depth 3 the trained policy spent 0.932 of
each episode on its single most common move, against 0.354 for a uniform policy, and 9 of 12 seeds
were at or above 0.95 (EXP-031). The stabilisers that had fixed collapse on the grid did not
transfer: across a sweep of entropy bonus and advantage normalisation, no cell passed the
pre-registered gate, normalisation alone made collapse worse, and the best cell still had 3 of 12
seeds collapsed with success at 0.006 (EXP-032). An oracle-supervised fit of the same 390 weights on
the same frozen features, with labels taken from the distance table, solved 0.481 of depth-3
states against REINFORCE's 0.022 (EXP-033). That is a reference rather than an agent, but it
showed the signal was in the representation and the learner was not finding it.

**The curriculum was the lever.** Splitting the same episode budget across depths 1, 2 and 3
instead of spending it all at depth 3 raised depth-3 success from 0.0222 to 0.2556 at 3,000
episodes (+0.2361, 11-0-1, p 0.0010), while five times the episodes without a curriculum moved it
by -0.0028 (p 1.0000): with no reward to learn from, more training only drove the policy further
into a constant action (EXP-034). Given more budget the curve kept climbing, to **0.5000 at 30,000
episodes**, 22.7 times the v1 baseline, and it had not saturated (+0.1028 from 10,000 to 30,000
against a pre-registered 0.02 bar; EXP-035). This is the largest single improvement in the
project, and it changed nothing in the architecture.

**A break point at depth 5, which turned out to be a budget.** At a fixed 10,000 episodes the
curriculum worked at depth 4 (0.1591) and broke at depth 5 (0.0396, under the 0.10 bar) and depth
6 (0.0000 on all 12 seeds), exactly where the pre-registration had predicted (EXP-036). The
train-to-held-out gap was real (+0.1093, p 0.0059) but below the 0.15 magnitude bar, so that
question was reported as inconclusive by its own rule. Two later changes moved the break point:
the pretrained encoder of section 5.2, and a fix for a trap at depth 1. A face move has order 4, so a
policy that plays one move over and over still solves exactly a third of depth-1 episodes within
their 5-move budget; the first curriculum stage therefore rewards collapse, and the two seeds that
fell into it never recovered (EXP-041). Capping the depth-1 training budget at 2 moves raised depth 4 from 0.3471 to
0.5351 and helped every seed (EXP-042), and depth 6 then worked at 0.1800 (EXP-043).

The decisive step was depth 7. At 10,000 episodes it missed the bar (0.0621, refuted); at 44,000 it
cleared it on all 12 seeds (0.1971, confirmed; EXP-044). The obvious explanation, that the bigger
budget simply gave the deepest stage more episodes, was tested directly and **refuted with the
wrong sign**: giving depth 7 the same deepest-stage episodes inside the 10,000 budget dropped
success to 0.0142 (-0.0479, 0-11-1, p 0.0010), with the policy collapsing onto one move (EXP-045).
The operative variable is total budget. Depth 6 then responded to the same 4.4x multiplier by
+0.1425 (12-0-0, p 0.0005), and a midpoint at 25,000 episodes landed on the straight line through
the endpoints in log spend (EXP-046).

**The budget law.** Success at a given depth rises by about **0.22 per tenfold increase in
training episodes** (0.2215 measured at depth 6), with **no knee**: the midpoint deviated from the
log-linear prediction by +0.0048 against a standard error of 0.0242 (EXP-046). A depth at 4.4 times
the budget scores like the depth above it at the original budget. **Every "depth N stopped working"
in this project had meant "depth N stopped working at 10,000 episodes."**

**The law held out of sample.** Fitted at depths 3 to 7, it predicted **0.0588** at depth 8. The
measurement was **0.0783**, 95% interval **[0.0523, 0.1044]**, which contains the prediction; every
one of the 12 seeds solved something, against a chance floor of exactly 0.0000 (EXP-062, confirmed).
Depth 9 measured **0.0163**, below the pre-registered 0.02 bar, so it is reported as at the floor,
although 7 of its 12 seeds score above zero. **The frontier is depth 8.**

**The censoring trap.** EXP-062 also priced the plan's target. The two measured steps are unequal:
depth 7 to 8 costs 0.1221 and depth 8 to 9 costs 0.0620. Averaging them would make depth 11 look
far cheaper, but the second step is compressed because success cannot fall below zero, so an arm
near the floor must show a smaller decline. Using only the step with both endpoints clear of the
floor, **depth 11 at depth-7 parity costs about 33 days of compute per seed**, about 18 days of
wall clock for a 12-seed arm (EXP-062). The censored average would have said 9.4 days. The general
rule is to extrapolate a bounded metric only from steps that are well clear of its bound.

What the budget law does and does not say matters for section 9. It describes the reactive policy
measured here, which makes one forward pass and one move per step and never looks ahead.

### 5.2 The spiking encoder, the one neuromorphic win

Until week 20 every cube result used a sensory encoder frozen at its random initialisation; only
the 390-parameter head learned. The roadmap's Stage 2 was to train it.

**Pretraining by inverse dynamics.** The encoder was trained, self-supervised and without any
distance labels, to name the move that connects two cube states, on 48,233 state pairs for 40
epochs (EXP-039). Frozen into the usual policy, it raised held-out success at every depth tested
(EXP-040, all paired against EXP-036 on the same seeds):

| depth | random encoder (EXP-036) | pretrained encoder (EXP-040) | delta | exact p |
|---|---|---|---|---|
| 4 | 0.1591 | **0.3471** | +0.1880 | 0.0337 |
| 5 | 0.0396 | **0.2304** | +0.1908 | 0.0020 |
| 6 | 0.0000 | **0.1037** | +0.1037 | 0.0039 |

Depth 4 more than doubled, depth 5 went from broken to working, and depth 6 left the floor. Only
the encoder's weights changed; the head and every other setting were held fixed.

**More pretraining is worse, and its own metric cannot tell.** Doubling pretraining to 80 epochs
halved depth-6 success (0.0887 against 0.1800, -0.0912, p 0.0078; EXP-050). Scanning the epoch
count showed the benefit saturating early: 10 epochs (0.2012) bought everything 40 did, and 20
against 40 was a null (+0.0050, p 0.8418), so the inherited 40 was not wrong (EXP-052). Meanwhile
the pretraining objective's move-accuracy kept rising all the way to 80 epochs. At the other edge,
one epoch was not enough (0.0854, 42.5% of the 10-epoch value), and epochs 2 to 10 bought a further
+0.1158 (p 0.0098; EXP-055).

**Fine-tuning during RL.** Letting the encoder's two layers (26,816 parameters) train alongside
the head raised depth-6 success from 0.1800 to **0.2700** at the same episode count (+0.0900,
10-1-1, p 0.0020; EXP-047). For scale, the budget law needed 4.4 times the episodes to take the
same cell to 0.3225; fine-tuning costs 1.33 times per step. **The gain lives in the encoder, not in
a head co-adapted to it**: the fine-tuned encoder, frozen and paired with a fresh head, scored
0.3112 (+0.1312 over the pretrained encoder, 11-1-0, p 0.0059; EXP-048). And it is RL's
*objective* that matters, not the extra gradient steps: spending the same steps on more
pretraining is the 80-epoch arm above, which lost to the fine-tuned encoder by 0.2225 (1-11-0,
p 0.0010; EXP-050).

**What did not hold up.**

- **The mechanism claim failed its own control.** The probe said the fine-tuned representation
  improved (+0.0398, p 0.0010); a leak-free slice of the same measurement said it did not
  (+0.0050, p 0.5732), and the spec had committed to reporting the weaker (EXP-047). The score
  improved; that the representation generalised could not be shown. The probe itself was later
  retired (section 4.6): re-analysed, it got the direction of pretraining's effect right and
  carried no information about which seeds gained or by how much (`experiments/probe_reanalysis/`).
- **It does not compound.** A second round of fine-tuning bought the same fixed increment over its
  own compute as the first: +0.0628, +0.0504 and +0.0540 above the budget-equivalent across three
  arms, flat (EXP-049). The best depth-6 cell, two rounds of fine-tuning, reached 0.3525.
- **Its advantage over budget fades at depth 7.** The fine-tuned encoder transferred to a depth it
  had never trained on (+0.0850, 10-1-1, p 0.0039), but only 0.65 of the depth-6 gain arrived, and
  its excess over simply buying budget fell from about +0.05 at depth 6 to **+0.0079** at depth 7
  (EXP-051). Its value is largest where budget is cheapest.

**What kind of win this is.** This is the one change in the project that is neuromorphic in
substrate and clearly pays: the trained component is the spiking encoder itself, and it carries
the gain. It is **not** neuromorphic in its learning rule. Both pretraining and fine-tuning
backpropagate surrogate gradients through the spiking layers; nothing here is local, Hebbian or
reward-modulated at the synapse.

### 5.3 The critic

The roadmap's Stage 3 asked for a denser learning signal. The ordinary RL answer is a learned
critic: a value estimate `V(s)` subtracted from the return, so the head learns from how much better
or worse an outcome was than expected from that state. Here the critic is a linear readout of the
same 64-neuron concept, 65 parameters, replacing the lagging moving-average baseline the head had
used until then.

**It works, narrowly at first.** At depth 7 the critic raised success from 0.1471 to **0.2004**
(+0.0533, 8-4-0, p 0.0498), against a pre-registered bar of +0.05 at alpha 0.05 (EXP-053, Claim 1,
confirmed). Both margins were thin, and the report at the time said so. Worse, the instrument meant
to show the mechanism said it was absent: the critic's explained variance in the final stage was
0.0021, no better than a constant.

**The mechanism is within-episode state-dependence.** EXP-056 kept the critic but flattened its
output to its own episode mean, so it could still track how well episodes go on average but could
no longer distinguish one state in an episode from another. That cost **-0.0646** (p 0.0234) and
put the arm level with the plain moving-average baseline (-0.0112 against it, p 0.6709). The
flattened critic fitted returns slightly *better* than the full one at every stage while performing
worse, so explained variance had not merely missed the mechanism; it had pointed the wrong way, and
it was retired (section 4.6).

**Calibration is not the mechanism.** The alternative reading was that the critic helps because it
is fitted by least squares and so better calibrated than a lagging average. A single learned
scalar, fitted the same way but blind to state, was indistinguishable from the moving average
(+0.0088, p 0.7822; EXP-057). Every arm without state-dependence landed between about 0.14 and
0.16; only the full critic reached 0.2004.

**It replicates.** Because EXP-056 had cleared its bar with little room, it was repeated on 10
seeds that had never been used for the question (seeds 14 to 23, with encoders manufactured for
them). The flattening cost was **-0.0925** (2-8-0, p 0.0156), larger than the original, clearing
the -0.05 bar and a Bonferroni correction (EXP-060, confirmed). Pooled over 22 seeds it is -0.0773,
p 0.0005. A level difference between the two seed blocks that this replication showed was later
traced entirely to the seed effect of section 4.5.

**What kind of win this is.** The critic is standard reinforcement learning. It reads the spiking
encoder's output, but nothing about it is neuromorphic, and it was the recipe component that made
depth 7 and beyond work at all (EXP-062's validity gate requires it). The neuromorphic version of
the same idea, routing the learning signal through the neuromodulatory bus, is section 5.5.

### 5.4 Episodic memory

The hippocampal region was built to remember states already visited in an episode, on the theory
that a policy which can recognise where it has been can stop going in circles. Every memory
experiment compared three arms, not two: **memory** (the head reads the concept plus the
hippocampus's recall of the current state), **shuffled** (the same, but recall is computed from a
different earlier state, so the memory is real but wrong), and **amnesic** (the same feed-forward
transform of the current state, queried against an emptied memory). Memory minus amnesic isolates
the stored content; memory minus shuffled alone, the obvious two-arm design, can be won by wrong
memory doing harm.

**First attempt: a null on a policy that had learned nothing.** At depths 1 to 3, memory beat
shuffled by +10.8 points at depth 2 (p 0.078) and beat amnesic by +1.2 (p 0.908) (EXP-030). The
primary comparison was measuring the harm of wrong memory, not the benefit of right memory, and
a two-arm design would have reported a near-significant win. The whole experiment was then found
to have run on a collapsed policy that played one move regardless of input (EXP-031), so no
change to its features could have shown up.

**Second attempt: void.** Re-asked on a working policy, EXP-058's validity gate required each
episode to store more than 10 memories on average. Episodes at that depth average fewer steps than
that because the policy solves them, so the gate could not pass; it measured 6.17, and by the
contract every claim was void (EXP-058). The data was valid; the specification was not. That cost
about 20 hours of compute and produced the gate-calibration rule of section 4.3.

**Third attempt: memory hurts.** At depth 5 on 24 seeds, with a gate that could pass and did,
memory scored **0.2183** against amnesic **0.3138**: **-0.0954, p 0.0056**, clearing the
pre-registered bar and a Bonferroni correction (EXP-059). The spec had committed in advance to
reporting this as a real finding rather than a failed confirmation. And correct memory was
indistinguishable from wrong memory: memory minus shuffled was +0.0204 (p 0.4268). What cost the
policy was reading the stored content at all.

**The recall is noise.** EXP-061 replaced the recall with noise matched to its magnitude. Real
memory and matched noise were indistinguishable (+0.0210, p 0.4989), and both cost about 0.1
against amnesic (noise minus amnesic -0.1165, p 0.0001). The recall block is useful for what its
transform says about the current state, and mixing in stored content destroys that as thoroughly
as random noise does. The primary was a null, which the spec had pre-registered as a bound rather
than evidence, so the hypothesis is reported as supported, not confirmed.

**The ceiling: perfect memory still loses.** The last possibility was that the attractor's recall
was too lossy and a better readout could use memory that the existing one could not. EXP-063
built a ceiling instrument to answer that: a learned attention over a **perfect** cache of the
episode's earlier states. It scored **0.1154**, 0.198 below amnesic (p 0.0000) and below the raw
attractor read as well (p 0.0059); both secondaries were refuted in the wrong direction. Its
primary, real content against matched noise, was confirmed by the pre-registered rule (+0.0612,
p 0.0358), but the effect sat on seeds where the control's training had collapsed to zero; on the
10 seeds where it had not, the difference was +0.0320 (p 0.5078, 5 up and 5 down). The confirmed
primary is substantially a difference in how often training collapsed, and it is not read as
evidence that memory helps. A second confound, that the learned readout also puts a larger block
of features on the policy path, was not pre-registered, and the two cross-arm secondaries cannot
separate it from the readout itself (EXP-063).

**Verdict: closed negative, at its ceiling.** With perfect episodic recall and a readout that
learns what to attend to, memory does not help this policy. The hippocampus earns no place on
this task. This is a statement about a reactive policy that chooses one move per step; whether
remembering visited states helps a policy that searches is a different question. EXP-071 asked a
narrow version of it with a plain lookup table rather than the hippocampus, and found that it does
(section 9).

### 5.5 The neuromodulatory bus

The design's neuromorphic answer to credit assignment was a neuromodulatory bus: a global dopamine
signal that, as in three-factor learning rules, decides *when* plasticity happens. Until EXP-053 the
bus existed in code but nothing read it; `Brain.learn()` wrote the reward to it and no synapse ever
changed as a result.

**EXP-053 made it load-bearing.** In arm G, the encoder's fine-tuning updates (section 5.2) were
applied only on steps where a signed dopamine signal, the reward against a moving average, cleared
a running-median threshold. The gate behaved exactly as designed, opening on 0.4987 of updates
against an intended half, with no tuned constant. Two comparisons were pre-registered at depth 6:

| claim | arm G | control | delta | W-L-T | exact p | bar | verdict |
|---|---|---|---|---|---|---|---|
| 2: gating against ungated fine-tuning | 0.3004 | 0.2700 | +0.0304 | 9-3-0 | 0.1323 | +0.05 | **not confirmed** |
| 3: dopamine gate against a random gate at the same rate | 0.3004 | 0.2654 | +0.0350 | 7-5-0 | 0.1167 | +0.03 | **not confirmed** |

The direction is positive in both, and in Claim 3 the delta clears its bar while the significance
does not. But the spec had fixed the reading of exactly this outcome before any number existed:
with neither claim confirmed, *"the neuromorphic claim is REFUTED, not deferred. 'We need a better
gate' is NOT an available conclusion from this experiment"* (EXP-053). That verdict stands.

**One mechanism number moved, and it is a lead, not a rescue.** Against the random gate at the
same rate, the dopamine-gated arm revisited states less often (revisit rate -0.0262, p 0.0083).
That comparison isolates the gate's signal from its rate, and it moved in the predicted direction.
It was one of six descriptive tests with no bar, so its p sits exactly at the Bonferroni threshold
and no lower, and the random control matches rate but not the timing structure of the dopamine
gate. The honest statement is that the dopamine signal changes trajectories relative to random
gating, and that change did not convert into a significant difference in success.

**Verdict: the bus was wired in and bought nothing measurable.** The neuromorphic claim for
Stage 3's credit assignment is refuted. The ordinary RL answer to the same problem, the critic of
section 5.3, was confirmed in the same experiment. Across the whole project, the only point at
which a neuromorphic component took part in learning is when the spiking encoder began to train
(section 5.2); this experiment is not a second one.

### 5.6 Topology, and why it cannot be measured here

The question the architecture was built to answer is whether dividing a spiking network into
specialised regions helps. The plan's test was a monolithic network with the same neuron count
(section 2.1). That test was run, and it turned out not to measure topology.

**The monolithic comparison measured width, in the control's favour.** In the v1 baseline the
regionalized brain and a 510-neuron monolithic stack tied at depth 1 (87.5% each), and the
regionalized brain led at depth 2 (38.0% against 30.6%) and depth 3 (2.2% against 0.8%) (EXP-029).
The paired differences were inside the seed noise (+7.4 points at depth 2 with an sd of 38.1, 7
wins to 5), so the pre-registered reading was a null. Section 3.2 explains why the comparison was
lopsided: the monolithic network puts all 510 of its neurons on the policy path, while the
regionalized brain puts 192 there. **The control had 2.66 times the usable capacity and still did
not win.** That makes the regionalized result less likely to be a width artifact, but it is not
evidence that the topology helps.

**Matching the path instead is vacuous.** The natural fix is a monolithic network of 192 neurons,
matched on the policy path. But that network is the same module with the same arguments and the
same seed as the brain's own sensory region. Checked through the real agent-construction path
across 6 seeds, every weight is bit-identical and the concept differs by exactly 0.000. Removing
the off-path regions from the brain instead changes nothing either, because they are already
disconnected from the action. This was caught before any compute was spent, and
`tests/training/test_topology_contrast_is_vacuous.py` locks it in; a failure of that test would be
good news, because it would mean topology had become measurable (`docs/phase3-honest-assessment.md`
section 2a).

**Putting the brain's own pathway on the policy path does not work.** The remaining route was to
let the action come from the prefrontal, router and motor regions themselves, trained end to end.
EXP-064 did this at depth 5 (arm P) against a control whose head was widened to match its
parameter count (arm C). Both validity gates passed: the regions genuinely trained (drift 2.71) and
the spiking pathway genuinely fired (no silent steps). **Both arms scored exactly 0.0000**, so the
primary contrast was zero by construction and nothing about topology may be read from it (EXP-064).
The control was broken: the same configuration with a linear head had scored 0.3229 over 24 seeds,
and the one added hidden layer collapsed it to a single action (entropy 0.0120). Arm P failed
differently: it kept exploring (entropy 1.11, revisit rate 0.259 against the control's 0.669) and
never learned to solve anything.

**And it fails at every learning rate tried.** EXP-064's arm P had used a region learning rate
whose drift looked high, so EXP-066 swept it over three orders of magnitude. All three arms were at
the floor (0.0017 at 1e-4, 0.0000 at 1e-3 and 1e-2), on a configuration where the 390-parameter
linear head scores 0.3229 (EXP-066). The relationship ran the wrong way: the further the regions
moved, the more completely the policy collapsed onto one move (modal fraction 0.850, 0.969, 1.000).
The per-stage trace shows the pathway learning something real on the shallow curriculum, about 8
times its floor at depth 3, and then collapsing as depth rises. It is not incapable; it is
unstable.

**Verdict: not testable in this configuration.** No arm-versus-arm contrast can measure topology
when the topology is not on the policy path, and the one attempt to put it there produced a
pathway that the training signal destroys. **Answering the topology question needs an architecture
change, not another experiment.** The plan's Phase 3 criterion 2 is graded with this in mind in
section 6.

### 5.7 Reproducibility and the seed effect

The last three experiments asked how far the results above can be trusted outside the machine and
the seeds that produced them. Section 4.5 gives the seed effect as a method; section 7 gives the
reproduction guarantee. This section reports the verdicts.

**Retraining on a second machine reproduces the conclusions, not the numbers.** EXP-067 retrained
EXP-036's depth-3 cell, 12 seeds, on a different x86 machine with the same torch version, and
imported EXP-036's own thresholds rather than setting new ones:

| claim | published | retrained | verdict |
|---|---|---|---|
| 1: mean within 0.02 | 0.3972 | 0.4250 (+0.0278) | **not replicated** |
| 2: the "working" verdict | working | working, 15x its floor | **replicated** |
| 3: the gap verdict | +0.1093, inconclusive | +0.0602, inconclusive | **replicated** |

Two of three pre-registered verdicts survived and the numeric bar did not. Each needs its caveat.
Claim 1's tolerance had been calibrated where replication noise is exactly zero, on one machine;
across machines the per-seed deltas have an sd of 0.1496, and under the null of equivalent machines
the 0.02 bar fails about two times in three. Claim 1 stands as not replicated, because rewriting a
gate after its number exists is the mistake section 4.3 describes, but it carries little weight
(section 4.3, last row). Claim 2 had a wide margin on both machines and was never at risk, and
Claim 3's inconclusive zone is 0.10 wide, the easiest verdict to land in twice (EXP-067).

**The divergence is all in training.** The floor arm, which does no training, was byte-identical on
12 of 12 seeds across the two machines, so states, held-out splits and scramble streams are fully
portable. Per seed, the trained results moved by between -0.3000 and +0.2333 while the mean moved
0.0278 (EXP-067).

**Seed quality is large, persistent and mostly an interaction.** Across 8 depth-5 arms from five
experiments, per-seed success correlated at +0.419, positive in 28 of 28 pairs of arms
(`docs/seed-effect.md`). Two crossed designs then decomposed it:

| | regime | interaction share | main effects | primary verdict |
|---|---|---|---|---|
| EXP-068 | depth 3, task draw x training trajectory, 10 x 10 | **0.650** | task draw 0.267, trajectory 0.084 | Claim 1 (the task draw is a minority of the effect, share under 0.35) **unresolved**: 0.267 is under the bar but inside the pre-registered 0.25 to 0.45 band; Claim 2 (trajectory resolves) **confirmed**, p 0.0235 |
| EXP-069 | depth 5, pretrained encoder x task draw and trajectory, 8 x 8 | **0.652** | encoder 0.348, task draw and trajectory together 0.000 | Claim 1 (the encoder is a major carrier, share at least 0.35) **unresolved**: 0.348, real (p 0.0001) but inside the band; Claim 2 (interaction replicates) **replicated** |

Two decompositions in different regimes agree that a seed's quality is mostly the specific
combination of its parts, while the main effects change hands with the regime (EXP-068, EXP-069).
The practical consequence is the one already in use: compare arms only on the same seeds, where
this effect cancels exactly, and never across two different seed sets, where it adds about 0.039
of noise to effects of 0.05 to 0.09. EXP-068's aggregator printed CONFIRMED for a result inside the
band its spec had declared unresolvable; the report follows the spec (section 4.1).

## 6. Graded against the plan

This section grades the project against the targets quoted in section 2, in their own words. The
Phase 2 and Phase 3 grades are the ones recorded at those checkpoints
(`docs/phase2-honest-assessment.md`, `docs/phase3-honest-assessment.md`); they are restated here,
not re-awarded.

### 6.1 The phase checkpoints

**Phase 0** was a checkpoint of understanding (explain a LIF neuron, backpropagation, an RL agent;
narrate PyTorch). It produced no artifact that can be graded after the fact, and none is claimed.

**Phase 1: met, late in one item.** Spiking MNIST classifiers reached 95.08% to 98.05% across three
variants (EXP-009, recorded in `docs/2026-05-25-phase-1-audit.md` because that experiment predates
the results-file rule). A recurrent spiking network on row-at-a-time MNIST reached 79.12% against a
feed-forward control's 41.84% (EXP-011). The STDP demonstration was missing at the Phase 1 audit and
was built the same day: three of four outputs became cleanly selective for one pattern each
(EXP-012). The "what I learned" document is the week-8 wrap-up note in the project vault.

**Phase 2: one met, three partial**, as graded at the time:

| criterion | grade | note |
|---|---|---|
| (1) architecture spec | partial | regions and wiring accurate; the training strategy lived only in an ADR |
| (2) multi-region SNN on grid navigation | partial | it navigates held-out goals, but 1 of 5 regions is on the policy path and the brain itself did not learn |
| (3) monitoring dashboard | met | renders real multi-region traces; a replay tool, not a live monitor |
| (4) regional specialisation | partial, then strong | the trained sensory concept decodes task structure (R2 0.86 to 0.90) and beats every other region on 12 of 12 seeds (EXP-027); "through learning" leans on another experiment's contrast |
| (5) honest assessment | the assessment itself | |

**Phase 3: three of four met or exceeded, one partial.**

| criterion | grade | note |
|---|---|---|
| (1) solves from 1-move scrambles, 3-move stretch | **exceeded** | depth 1 at 87.5% in the v1 baseline (EXP-029); the depth-3 stretch reached 50.0% (EXP-035); the series runs to depth 8 at 0.0783 against a floor of exactly 0.0000 (EXP-062) |
| (2) comparison against a monolithic network | **partial** | it exists and is committed (EXP-029), but it measured width rather than topology, and the path-matched version is vacuous (section 5.6) |
| (3) interpretability analysis | **met, then substantially retracted** | extensive (EXP-027, EXP-033, EXP-056/057, EXP-061, EXP-063), but five instruments were retired and many original conclusions were withdrawn (section 4.6) |
| (4) clear documentation | **met** | all 35 Phase 3 experiments carry a committed results file with provenance and a regeneration command; 28 have a pre-registered spec |

Criterion 2 stays partial deliberately. Re-running the old comparison on the working recipe would
reproduce the same width confound at a higher success rate, and the only design that could close
it needs an architecture in which a region other than the sensory one influences the action
(section 5.6). Closing a checkbox with a comparison whose flaw is understood is what the grading
exists to prevent.

### 6.2 The plan's three levels of success

| level | grade | why |
|---|---|---|
| minimum viable: regions specialise and cooperate on grid navigation | **partial** | the sensory region demonstrably specialises (EXP-027); the regions do not demonstrably cooperate, because only one of them reaches the action |
| target: solve a 2x2 *"from any scramble"* through reinforcement | **not met** | the learning is genuinely by reinforcement, but the frontier is depth 8 (EXP-062), depth 9 is at the floor, and a uniformly random 2x2 is most often 11 moves from solved |
| stretch: continual learning without forgetting | **not attempted** | no experiment trained a second task |

### 6.3 The roadmap the project set itself

| stage | goal | status |
|---|---|---|
| 1 | prove generalisation, find the break point | **done** (EXP-036), and the break point later turned out to be a budget, not a wall (section 5.1) |
| 2 | train the encoder | **delivered** (EXP-040/047/048), though its stated success metric, the probe ceiling, was retired along the way; the gain does not compound (EXP-049) |
| 3 | a dense signal through the neuromodulatory bus | **closed on both threads**: the critic works and replicates (EXP-053/056/057/060); the bus was made load-bearing and bought nothing (EXP-053) |
| 4 | depth-11 random scrambles, *"the actual deliverable"* | **priced out**: about 33 days of compute per seed with the current reactive policy (EXP-062) |

The roadmap had judged Stage 4 *"genuinely achievable... nothing about it requires new science."*
That was written before the budget law existed. It was right about the science and wrong about
the arithmetic: with the policy as built, it is the exchange rate between compute and depth that
stops it. Section 9 takes up what would change that exchange rate.

## 7. Reproducing the results

### 7.1 The guarantee

**Re-evaluating the tracked checkpoints reproduces re-evaluation exactly, and reproduces a
published per-seed number only where that has been checked. Retraining from a seed reproduces
neither.** These were measured on two x86 machines running the same torch
(`docs/reproducibility-audit.md`, EXP-067, EXP-070).

- **Re-evaluation is deterministic and portable.** EXP-036's published depth-3 checkpoints, loaded
  and re-evaluated on both machines, agree on success rate, mean steps, optimality and revisit
  rate to the full floating-point representation. One derived mean, the modal-action fraction,
  differs by 1 to 2 units in the last place on two of three seeds, because it is a mean of
  per-episode fractions and its summation order is the one thing that moves. EXP-070's harness,
  re-run on one machine into a separate directory, was identical in every field except wall-clock
  time (EXP-070, Gate 0(a)).
- **Re-evaluation does not always reproduce the published number.** Evaluation is stochastic: the
  observation is encoded as Poisson spikes drawn from the evaluation generator. A published record
  was evaluated on whatever state of that stream training had left behind, and a re-evaluation
  starts a fresh one. For EXP-036 the two agree. For the 36 depth 7 to 9 cells of EXP-053 arm B and
  EXP-062, **27 differ from their published solved counts**, the largest being depth 8 seed 3 at
  22 of 200 published against 8 of 200 re-evaluated. Pooled per depth, every re-evaluated mean lies
  inside the Wilson 95% interval of the published one: 0.2067 against 0.2004 at depth 7, 0.0688
  against 0.0783 at depth 8 (inside by 0.0006), and 0.0171 against 0.0163 at depth 9 (EXP-070,
  Gate 0(b); the per-depth figures are in the roadmap spec's pre-launch amendment).
- **Retraining is reproducible on one machine and not across two.** On the machine that made them,
  seeded runs are byte-identical across worker scheduling, which the project has used repeatedly as
  a correctness check on its seeding. On a second machine, a retrained EXP-036 head shares **0 of
  its 390 parameters** with the published one (cosine 0.524), and that cell uses no pretrained
  encoder, so the divergence is not specific to pretraining. Across 12 seeds the retrained
  success rates move by -0.3000 to +0.2333 per seed (EXP-067).
- **The task layer is fully portable.** States, held-out splits and scramble streams were
  byte-identical on 12 of 12 seeds across the two machines; every bit of the divergence is in
  training (EXP-067).
- **Findings mostly survive retraining.** Two of three pre-registered verdicts replicated on the
  second machine and the numeric bar did not (section 5.7). That is one experiment, not a general
  result.

So the claim this report makes is: **re-evaluating a tracked checkpoint from a fresh clone
reproduces any other re-evaluation of it exactly, and reproduces the published per-seed numbers
where that has been checked (EXP-036); elsewhere it reproduces the published per-depth means
within sampling error, not the per-seed counts (EXP-053 arm B, EXP-062). Retraining from a seed
reproduces the findings, as far as has been tested, but not the numbers.** Do not read "seeded
runs are byte-identical" without the qualifier "on the same machine".

### 7.2 What is tracked, and what is not

- **Tracked:** every trained policy head (`*_head.pt`), the pretrained and fine-tuned encoders that
  later experiments load (the E0 encoders were added after the audit found them laptop-only), each
  experiment's `run.py` and `aggregate.py`, its pre-registered spec under
  `docs/superpowers/specs/`, and its `RESULTS.md` with provenance and regeneration commands. About
  1,080 checkpoint files in all.
- **Not tracked:** the per-run JSON records under each `experiments/*/outputs/`, which are
  gitignored, with one exception: EXP-070 commits its 324 records. **A fresh clone therefore cannot
  re-run any other experiment's `aggregate.py` directly**; the numbers in each `RESULTS.md` are the
  committed record of what those aggregators printed. Re-deriving them from a clone means
  re-evaluating the checkpoints to regenerate the records first, which reproduces per-depth means
  rather than per-seed counts wherever the two have not been checked against each other.

### 7.3 Commands

Environment: Python 3.10 or later (3.10 on the Linux machine, 3.13 on the Windows laptop that ran
most experiments), torch 2.13.0 CPU build, snnTorch, Gymnasium. Install into a virtual environment
and run everything through it.

```bash
# Re-evaluate published checkpoints and print metrics plus an action-sequence digest
.venv/bin/python -u scripts/verify_eval_portability.py --depth 3 --seeds 0 1 2

# Retrain a published cell into a scratch directory and compare its head byte for byte
.venv/bin/python -u scripts/verify_e2e_reproduction.py --depth 3 --seeds 0

# Encoder pretraining: regenerate and compare against the tracked encoders
.venv/bin/python -u experiments/040_pretrained_encoder_policy/verify_regeneration.py --seeds 0

# The seed-effect numbers of section 4.5
.venv/bin/python -u scripts/seed_effect.py
```

Each experiment's own launch and aggregation commands are at the foot of its `RESULTS.md`. Most
experiments were run on a 22-core laptop over SSH with a PowerShell launcher; the cube `run.py` drivers
themselves are plain Python and most take `--workers`.

### 7.4 Limits of the guarantee

- **Published per-seed numbers have been checked for two experiments.** EXP-036 (three seeds,
  exact) and EXP-053 arm B with EXP-062 (36 cells, pooled means only). For every other experiment,
  a re-evaluated per-seed number should be expected to differ from the published one by sampling
  error. **A published per-seed value must never be used as the paired baseline for a re-evaluated
  arm**: re-evaluate the baseline too, on the same stream discipline, which is what EXP-070 did.
- **There is no general re-evaluation driver.** `verify_eval_portability.py` is written against
  EXP-036, and EXP-070's harness against the depth 7 to 9 checkpoints.
- **Two x86 CPU machines only.** No GPU, no ARM.
- **The `dashboard/` JavaScript app** has its own toolchain and lockfile and was not part of the
  audit.

## 8. Limitations and what would come next

### 8.1 Limitations

**What was solved.** A 2x2 cube, from held-out states up to 8 moves from solved, at 0.0783 at
depth 8 (EXP-062). Depth 9 is at the floor and depth 11, the modal distance of a random cube, was
priced out rather than attempted. The plan's target, a cube solved *"from any scramble"*, is not
met (section 6.2).

**What the solver is.** With `recall=False`, the policy is the 192-neuron sensory region and a
390-parameter linear head, plus a 65-parameter critic; 318 of the 510 neurons do not reach the
action (section 3.2). The results in section 5 that are positive are results about that path. They
are not evidence about the five-region brain as a whole, and the report does not present them as
such.

**The topology was never tested.** No arm-versus-arm contrast in this configuration can measure it,
and the attempt to put the brain's own pathway on the policy path collapsed at every learning rate
tried (section 5.6). Whether regionalisation helps is an open question, not a negative answer.

**Nothing learns by a local rule.** All training backpropagates surrogate gradients. A three-factor
eligibility trace was built as a demonstration in Phase 2 (EXP-021) and never trained anything; the
one attempt to let a neuromodulatory signal govern plasticity was refuted (EXP-053). The project's
neuromorphic content is the spiking substrate, not the learning.

**The policy is reactive.** It makes one forward pass and one move per step, with no look-ahead.
Every depth result, and the budget law that prices them, describes that kind of policy (section
5.1). The memory result is also about a reactive policy (section 5.4).

**2x2 only.** The 6-move action set is a 2x2 simplification that does not carry to a 3x3, which has
fixed centres and needs 12 or 18 moves (section 3.3). The exact distance-to-solved table that every
depth result is built on exists because the 2x2 state space is small enough to enumerate. A 3x3
has no such table, so the project's main instrument would have to be replaced, not just scaled.

**Scale.** 510 neurons, against the plan's *"thousands to tens of thousands"*. Nothing here says
how any result changes with size.

**Effect sizes are close to the noise.** Most published effects are 0.05 to 0.09 in success rate,
at n = 12 seeds. Seed quality alone moves success with an sd of about 0.09 and puts about 0.039 of
noise on any comparison between two different seed sets (section 4.5). Every comparison in this
report is paired on seeds, which removes that noise, but a reader comparing numbers from two
different experiments should not assume they share seeds.

**Mechanism claims are weaker than score claims.** Five instruments were retired after they failed
to track outcomes (section 4.6), and several original mechanism conclusions were withdrawn with
them. Where a section names a mechanism, it rests on an arm-versus-arm contrast (as for the critic
in section 5.3) or is marked as a lead.

**Reproduction is of checkpoints, not of training.** Retraining on a different machine reproduces
findings, as far as one experiment has tested, but not numbers; only two x86 CPU machines were
tested (section 7).

**Efficiency was never measured.** The plan's motivation was the efficiency of brain-like
computation. No experiment measured energy, synaptic operations or any other cost of the spiking
implementation against a conventional network, and this report makes no efficiency claim.

**One task family, and no continual learning.** The plan's stretch goal, a second task learned
without forgetting the first, was not attempted (section 6.2).

### 8.2 The plan's extensions, against what was found

The plan listed four extensions for Phase 4 (section 2.1). Each is stated here against what the
project actually showed, not as a promise.

**Scaling to more neurons.** Nothing here suggests that size is the binding constraint. A
monolithic control with 2.66 times the on-path capacity did not beat the regionalized brain
(EXP-029), adding a hidden layer to the head collapsed the policy (EXP-064), and the depth series
turned out to be governed by training budget, not by any measured limit of the network (section
5.1). Scaling is worth testing only alongside a change that puts more of the network on the policy
path, or it measures nothing.

**Neuromorphic hardware.** The trained parts of this network could in principle run on a spiking
chip, since they are LIF populations with binary spikes. But training would stay off-chip, because
it uses surrogate gradients, and the efficiency benefit that motivates hardware was never measured
in software. Porting first would answer a question this project has not yet posed.

**Replacing surrogate gradients with local learning.** This is the extension the evidence is least
encouraging about. The one component that clearly pays, the trained spiking encoder, was trained
with surrogate gradients (section 5.2), and the one test of reward-gated plasticity bought nothing
(section 5.5). A local rule would have to match a gradient-trained encoder before it could replace
one, and no experiment here has made that comparison.

**Continual learning.** Not attempted, and not a near-term step: it presupposes a first task that
is solved.

What the project does point at is in section 9.

## 9. The road from here

The goal is now the one the plan called its target: a spiking network that solves a randomly
scrambled 2x2 (section 2.3). The budget law says more of the same will not reach it, since depth
11 costs about 33 days of compute per seed (section 5.1). But that law describes a **reactive**
policy, which must choose correctly about 11 times in a row with one forward pass per move.
Nothing in it constrains a policy that looks ahead. The roadmap that follows from this
(`docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md`, sections 1 and 4) splits the
skill into a general procedure (imagine moves, compare where they lead, choose) and a learned
judgement (how close does this position look), the shape of the strongest published cube solvers.
Each stage has a pre-registered decision point, and the next starts only on its answer.

**The first stage has run, and its primary was not significant.** EXP-070 added a look-ahead of
up to three moves to the existing checkpoints, retraining nothing, and scored candidate sequences
with the policy head itself. At depth 8 with three-move look-ahead, scoring reached **0.0854**
against **0.0838** for the matched control, which runs the same look-ahead only to check whether
any sequence reaches solved (+0.0017, p 0.3818, **not significant**). The test could detect about
0.01, so any benefit of the policy head as a judge is below that, not shown to be zero. Of the
three secondary cells, two were **refuted** by the spec's mechanical rule for a non-positive point
estimate (depth 8 and depth 9 at two moves; differences far inside the noise, so "no benefit"
rather than harm) and one was **not significant** (depth 9 at three moves). Claim 2, the practical
target of 0.10 at depth 9, was **not met**: scored look-ahead reached 0.0250 there, against the reflex policy's 0.0171 (EXP-070).
Look-ahead scored by the policy also left the policy's tendency to revisit states unchanged, at
about 0.45 of moves at depth 8. That tendency is not a new finding: EXP-062's records already
showed revisit rates of 0.445 at depth 8 and 0.552 at depth 9. What EXP-070 adds is that a search
judged by the same policy inherits the same cycles.

**What it points at, as leads rather than findings.** Two observations were not pre-registered
and are recorded as leads. The goal check on its own, without any scoring, raised success at depth
7 (+0.0254, 11 of 12 seeds) and depth 8 (+0.0150); this is a property of the search procedure, not
of the network. And the reflex policy's high modal-action fraction, two thirds of moves at depth 9,
suggests its loops are one face turned repeatedly, an inference from aggregates rather than an
observation (EXP-070).

**The second stage has run too: the critic is not a judge, and refusing to revisit helps.**
EXP-071 tested two primaries at depth 7, again on the existing checkpoints with nothing trained:

| claim | arm | control | delta | verdict |
|---|---|---|---|---|
| 1: the critic of section 5.3 as the judge of a 3-move search | 0.0729 | 0.2321 (look-ahead used only as a goal check) | -0.1592, worse on 12 of 12 seeds | **refuted** |
| 2: the reflex policy forbidden to re-enter a visited state | 0.2346 | 0.2067 (the reflex alone) | +0.0279, 9 of 12, p 0.0115 | **confirmed** |

The critic ranks positions better than chance, and both pre-registered ranking gates passed, but it
picks an improving move 34% of the time against the policy head's 50%, and steering by it compounds
that error over every move. It was trained as a baseline for reducing variance in the policy
gradient, not as an estimate of distance to solved. By the roadmap's own rule for this stage, which
required a learned judge to beat the policy head's look-ahead, the critic as trained is retired as a
judge, and the route to a learned judge moves to the roadmap's third stage: training one from its own
look-ahead (EXP-071).

The no-revisit rule is a hand-coded visited-state table inside the search, not a spiking component
and not the memory mechanism that section 5.4 closed, which fed recall to the policy head as an
input. The largest effect in the experiment is a pre-registered **secondary**, read as a pattern
and not as a confirmation: policy-scored 3-move look-ahead with the rule solved **0.3217** against
0.2454 without it (+0.0762, 12 of 12 seeds, p 0.0002). Against the reflex alone the same
configuration is **+0.1150**, a figure that is **exploratory** and was not pre-registered (EXP-071).
**The gain is search scaffolding around the frozen network**: a simulator to imagine moves, a
three-move tree, and a lookup table of visited states. The spiking network learned nothing new in
either experiment, and the one learned judge tried there, the critic, failed.

**A third experiment carried that configuration to the frontier, and both primaries were confirmed.**
EXP-072 ran policy-scored 3-move look-ahead with the no-revisit rule at depths 8 and 9, on the same
checkpoints with nothing trained, against two pre-registered primaries at alpha 0.025 each. Every
validity gate passed, including a check that the harness reproduced EXP-070's records for the
paired references exactly:

| claim | arm | control | delta | verdict |
|---|---|---|---|---|
| 1, depth 8: look-ahead with the rule against look-ahead without it | 0.1496 | 0.0854 | +0.0642, 11 of 12 seeds better and one tied, p 0.0005 | **confirmed** |
| 2, depth 9: look-ahead with the rule against the reflex | 0.0629 | 0.0171 (the reflex alone) | +0.0458, 12 of 12, p 0.0002 | **confirmed** |

Claim 1 confirms at depth 8 what EXP-071 showed only as a secondary at depth 7, and Claim 2 takes
depth 9 off the floor. The six secondary contrasts are read as a pattern, not as confirmations, and
the pattern is the one seen at depth 7: the rule alone gave a small gain just short of significance
(+0.0150, p 0.0547 at depth 8; +0.0113, p 0.0508 at depth 9), and look-ahead added a large one only
once the rule was in place (+0.0658 and +0.0346 over the rule alone, p 0.0002 and 0.0005). Across
EXP-071 and EXP-072 the configuration's gain over the reflex is +0.115, +0.081 and +0.046 at depths
7, 8 and 9: the absolute gain shrinks as the reflex's base rate does, while the relative gain grows
(1.56x, 2.17x, 3.68x). That trend is **exploratory**, three depths from two experiments, and was not
pre-registered (EXP-072).

The framing of EXP-071 applies unchanged. This is search scaffolding around frozen networks, and
nothing was trained. It does not reach the goal either: a random 2x2 is usually 11 moves from
solved, and at depth 9 the configuration solves about 6% of held-out states. The roadmap's remaining
routes were a learned judge (its third stage), a learned or spiking form of the no-revisit rule, and
deeper search, and the project chose the learned judge.

**The third stage has run: a judge the network learns from its own look-ahead, and both primaries
were confirmed.** One disclosure comes first. While the project was diagnosing the experiment this
one replaced, early judges were run on two of its evaluation cells (depth 9, seeds 0 and 3) before
its spec was written, and those numbers shaped the design. The spec therefore prints every primary
twice, on all 12 seeds and on the 10 seeds that were never looked at, and the two lines agree
(EXP-074).

EXP-073 had planned a judge reading the 64-unit concept. Its pilots predicted that its training
gate would fail, diagnostics then showed that the gate measured the wrong quantity and that the
64-unit readout was itself a bottleneck, and EXP-073 was superseded before any evaluation seed was
trained, so **none of its claims was tested** (EXP-073). EXP-074 replaced it with a judge reading all
192 sensory neurons, the 64-unit concept and the 128 hidden units, trained by value iteration from
its own 3-move look-ahead and never from the true distances. The spiking encoder underneath was
trained with it, by surrogate gradients as in EXP-047. The judge then scores the leaves of the same
3-move search with the no-revisit rule, and was compared at depth 9 with the best configuration so
far and with the same judge reading only the 64-unit concept. Every validity gate passed:

| claim (depth 9) | arm | control | 12 seeds | 10 seeds (0 and 3 dropped) | verdict |
|---|---|---|---|---|---|
| 1: the 192-unit judge against policy-scored look-ahead | 0.1933 | 0.0629 | +0.1304, 12 of 12, p 0.0002 | +0.1340, p 0.0010 | **confirmed** |
| 2: the 192-unit judge against the 64-unit judge | 0.1933 | 0.1292 | +0.0642, 10 of 12 better and two tied, p 0.0010 | +0.0605, p 0.0039 | **confirmed** |

The test could detect a difference of about 0.025, and the observed ones are about 12 and 6
standard errors. The secondaries, read as a pattern and not as confirmations, are all positive: the
192-unit judge also leads at depth 7 (**0.7908** against 0.3217) and depth 8 (0.3792 against
0.1496), and at depth 9 the 64-unit judge beats policy-scored look-ahead (+0.066) and training its
encoder beats freezing it (+0.063), which answers EXP-073's two questions as patterns rather than
claims. One exploratory run, with no claim attached, went to depth 11, the typical distance of a
random scramble, using each seed's depth-9 agent: the 192-unit judge solved **0.1013** of held-out
states against 0.0308 for policy-scored look-ahead (EXP-074).

This is the first result in section 9 where the network learned something new. The judge's readout
and the spiking encoder beneath it were trained, without any answers, from the network's own
search. The search itself is still scaffolding: the simulator, the three-move tree and the visited
table are hand-coded. About one random scramble in ten is solved, so the goal is closer but not met.

None of this changes the learning rule. Training remains surrogate-gradient backpropagation, and
the roadmap does not claim that search makes the network neuromorphic in how it learns, or that
learned weights will transfer from a 2x2 to a 3x3. What it aims to transfer is the recipe.

## Appendix A. Experiment index

Every experiment folder, with the title its committed results file gives itself. **This table is
generated** by `scripts/experiment_index.py` from the folders on disk, and
`.venv/bin/python -u scripts/experiment_index.py --check` fails if it has drifted. Folders with no
committed record are Phase 0 and 1 learning exercises, early grid-world work, and one dashboard
recording (EXP-065), all of which predate or fall outside the 2026-07-13 rule that every experiment
commits its results. Verdicts are in each results file and, for the experiments this report draws
on, in section 5; a verdict parsed out of free-form prose would be a guess, so the table does not
carry one.

<!-- experiment-index:start (generated by scripts/experiment_index.py; do not edit) -->
74 numbered folders, 58 with a committed results file.

| id | folder | results file | title |
|---|---|---|---|
| EXP-001 | `001_smoke_test` | none | no committed record |
| EXP-002 | `002_week1_mnist_classifier` | none | no committed record |
| EXP-003 | `003_week2_lif_neuron_viz` | none | no committed record |
| EXP-004 | `004_week3_1_pytorch_handson` | none | no committed record |
| EXP-005 | `005_week3_2_pytorch_training_loops` | none | no committed record |
| EXP-006 | `006_week4_1_gymnasium_intro` | none | no committed record |
| EXP-007 | `007_week4_2_qlearning_grid_agent` | none | no committed record |
| EXP-008 | `008_week5_snn_mnist_baseline` | none | no committed record |
| EXP-009 | `009_week6_snn_mnist_optimization` | none | no committed record |
| EXP-010 | `010_week7_recurrent_sn` | none | no committed record |
| EXP-011 | `011_week7_sequential_mnist` | `experiments/011_week7_sequential_mnist/results.md` | Sequential MNIST results |
| EXP-012 | `012_week7_stdp_demo` | `experiments/012_week7_stdp_demo/results.md` | STDP-WTA demo results |
| EXP-013 | `013_week9_sensory_bringup` | `experiments/013_week9_sensory_bringup/results.md` | Sensory Cortex bring-up results |
| EXP-014 | `014_week9_motor_wta_bringup` | `experiments/014_week9_motor_wta_bringup/results.md` | Motor Cortex + WTA bring-up results |
| EXP-015 | `015_week9_open_loop_bringup` | `experiments/015_week9_open_loop_bringup/results.md` | Open-loop bring-up results (Sensory → Prefrontal → Motor) |
| EXP-016 | `016_week9_router_gating_bringup` | `experiments/016_week9_router_gating_bringup/results.md` | Thalamic Router gating bring-up results |
| EXP-017 | `017_week9_hippocampus_bringup` | `experiments/017_week9_hippocampus_bringup/results.md` | Hippocampus store/recall bring-up results |
| EXP-018 | `018_week9_neuromod_bringup` | `experiments/018_week9_neuromod_bringup/results.md` | Neuromodulatory bus bring-up results |
| EXP-019 | `019_week10_pattern_completion` | `experiments/019_week10_pattern_completion/results.md` | Hippocampus pattern completion results |
| EXP-020 | `020_week10_closed_loop` | `experiments/020_week10_closed_loop/results.md` | Week-10 closed loop results |
| EXP-021 | `021_week10_rstdp_taste` | `experiments/021_week10_rstdp_taste/results.md` | R-STDP first taste results |
| EXP-022 | `022_week11_dashboard_trace` | none | no committed record |
| EXP-023 | `023_week11_brain_training` | none | no committed record |
| EXP-024 | `024_grid_generalization` | none | no committed record |
| EXP-025 | `025_head_capacity` | none | no committed record |
| EXP-026 | `026_sensory_pretrain` | none | no committed record |
| EXP-027 | `027_encoder_characterization` | `experiments/027_encoder_characterization/RESULTS.md` | Encoder Characterization (committed record) |
| EXP-028 | `028_sensory_ablation` | `experiments/028_sensory_ablation/RESULTS.md` | Sensory-Code Ablation (dose-response) |
| EXP-029 | `029_cube_baseline` | `experiments/029_cube_baseline/RESULTS.md` | v1 Cube Baseline (fail-first) |
| EXP-030 | `030_memory_engagement` | `experiments/030_memory_engagement/RESULTS.md` | Memory Engagement (does episodic memory content help?) |
| EXP-031 | `031_policy_collapse` | `experiments/031_policy_collapse/RESULTS.md` | Policy Collapse (was the EXP-030 memory null measured on a degenerate policy?) |
| EXP-032 | `032_collapse_sweep` | `experiments/032_collapse_sweep/RESULTS.md` | Collapse Sweep (do the trainer stabilizers fix the cube policy?) |
| EXP-033 | `033_concept_decodability` | `experiments/033_concept_decodability/RESULTS.md` | Concept Decodability (is the representation the wall?) |
| EXP-034 | `034_learning_signal` | `experiments/034_learning_signal/RESULTS.md` | Learning Signal (the curriculum works, and volume alone does nothing) |
| EXP-035 | `035_budget_scaling` | `experiments/035_budget_scaling/RESULTS.md` | Budget Scaling (50% at depth 3, and the curve has not saturated) |
| EXP-036 | `036_generalisation_gap` | `experiments/036_generalisation_gap/RESULTS.md` | the generalisation gap and the depth break point |
| EXP-037 | `037_curriculum_weighting` | `experiments/037_curriculum_weighting/RESULTS.md` | curriculum stage weighting is not a lever, and back-loading actively hurts |
| EXP-038 | `038_depth6_collapse` | `experiments/038_depth6_collapse/RESULTS.md` | collapse is a symptom at depth 6 too, and the stabilizers are closed |
| EXP-039 | `039_encoder_pretraining` | `experiments/039_encoder_pretraining/RESULTS.md` | the encoder learns, and it overtakes the observation exactly where the observation fails |
| EXP-040 | `040_pretrained_encoder_policy` | `experiments/040_pretrained_encoder_policy/RESULTS.md` | the ceiling converts, and the break point moves |
| EXP-041 | `041_seed_collapse_diagnosis` | `experiments/041_seed_collapse_diagnosis/RESULTS.md` | why 2 of 12 seeds collapse: depth 1 pays 33% for the worst possible policy |
| EXP-042 | `042_depth1_trap` | `experiments/042_depth1_trap/RESULTS.md` | repricing depth 1 fixes the failures AND helps every seed |
| EXP-043 | `043_cap_at_depth_5_6` | `experiments/043_cap_at_depth_5_6/RESULTS.md` | depth 6 works, and the break point is now past everything measured |
| EXP-044 | `044_depth7_frontier` | `experiments/044_depth7_frontier/RESULTS.md` | depth 7 works at 4.4x the budget. The deficit was STARVATION, not depth. |
| EXP-045 | `045_budget_vs_coverage` | `experiments/045_budget_vs_coverage/RESULTS.md` | it was TOTAL BUDGET. Back-loading a fixed budget makes depth 7 much worse. |
| EXP-046 | `046_depth6_budget` | `experiments/046_depth6_budget/RESULTS.md` | the depth series is a BUDGET series. 4.4x buys about one depth. |
| EXP-047 | `047_encoder_finetuning` | `experiments/047_encoder_finetuning/RESULTS.md` | fine-tuning works, and the mechanism does not survive its own control |
| EXP-048 | `048_fresh_head` | `experiments/048_fresh_head/RESULTS.md` | the encoder really did improve, and its own head was holding it back |
| EXP-049 | `049_second_round` | `experiments/049_second_round/RESULTS.md` | constant returns, as predicted, and the probe now moves the wrong way twice |
| EXP-050 | `050_objective_vs_gradient` | `experiments/050_objective_vs_gradient/RESULTS.md` | the objective is what matters, and more pretraining is actively harmful |
| EXP-051 | `051_depth7_transfer` | `experiments/051_depth7_transfer/RESULTS.md` | the gain transfers, but its advantage over budget does not |
| EXP-052 | `052_pretraining_optimum` | `experiments/052_pretraining_optimum/RESULTS.md` | pretraining saturates early, and its own metric cannot tell you that |
| EXP-053 | `053_neuromod_stage3` | `experiments/053_neuromod_stage3/RESULTS.md` | the critic clears its bar by a hair, and the neuromorphic claim is refuted |
| EXP-054 | `054_sequence_blindness` | `experiments/054_sequence_blindness/RESULTS.md` | the sequence-blindness hypothesis is REFUTED |
| EXP-055 | `055_pretraining_left_edge` | `experiments/055_pretraining_left_edge/RESULTS.md` | the 0-to-10 pretraining window |
| EXP-056 | `056_flattened_critic` | `experiments/056_flattened_critic/RESULTS.md` | the critic's benefit IS its within-episode state-dependence |
| EXP-057 | `057_constant_critic` | `experiments/057_constant_critic/RESULTS.md` | calibration is NOT the mechanism |
| EXP-058 | `058_memory_reask` | `experiments/058_memory_reask/RESULTS.md` | VOID. The validity gate failed, and the fault is in the spec. |
| EXP-059 | `059_memory_depth5` | `experiments/059_memory_depth5/RESULTS.md` | episodic memory HURTS a policy that works |
| EXP-060 | `060_flattened_critic_replication` | `experiments/060_flattened_critic_replication/RESULTS.md` | EXP-056 REPLICATES on fresh seeds |
| EXP-061 | `061_noise_matched_recall` | `experiments/061_noise_matched_recall/RESULTS.md` | the hippocampal recall is NOISE, measured rather than inferred |
| EXP-062 | `062_depth_frontier` | `experiments/062_depth_frontier/RESULTS.md` | the budget law holds out of sample, and Stage 4 is priced out |
| EXP-063 | `063_learned_readout` | `experiments/063_learned_readout/RESULTS.md` | the ceiling instrument came back NEGATIVE |
| EXP-064 | `064_motor_policy_path` | `experiments/064_motor_policy_path/RESULTS.md` | the run completed, the experiment did not |
| EXP-065 | `065_cube_dashboard_trace` | none | no committed record |
| EXP-066 | `066_region_lr_sweep` | `experiments/066_region_lr_sweep/RESULTS.md` | the brain's own pathway does not learn a policy, at any learning rate |
| EXP-067 | `067_cross_machine_replication` | `experiments/067_cross_machine_replication/RESULTS.md` | the TASK is portable, the TRAINING is not, and 2 of 3 verdicts survive |
| EXP-068 | `068_seed_decomposition` | `experiments/068_seed_decomposition/RESULTS.md` | the seed effect is mostly INTERACTION, and neither factor owns it |
| EXP-069 | `069_encoder_decomposition` | `experiments/069_encoder_decomposition/RESULTS.md` | at depth 5 the task draw and trajectory vanish, and the encoder takes their place |
| EXP-070 | `070_lookahead_existing` | `experiments/070_lookahead_existing/RESULTS.md` | look-ahead on the networks we already have |
| EXP-071 | `071_critic_and_no_revisit` | `experiments/071_critic_and_no_revisit/RESULTS.md` | the critic as judge, and a no-revisit rule in the search |
| EXP-072 | `072_p3v_frontier` | `experiments/072_p3v_frontier/RESULTS.md` | look-ahead plus no-revisit moves the depth 8-9 frontier |
| EXP-073 | `073_learned_judge` | `experiments/073_learned_judge/RESULTS.md` | superseded by EXP-074, no evaluation |
| EXP-074 | `074_wide_judge` | `experiments/074_wide_judge/RESULTS.md` | a learned judge that reads the whole sensory region |
| unnumbered | `probe_reanalysis` | `experiments/probe_reanalysis/RESULTS.md` | Probe re-analysis - EXP-033, EXP-039 and EXP-047 with trajectory metrics beside them |
<!-- experiment-index:end -->
