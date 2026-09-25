# A five-region spiking brain learns a 2x2 cube, mostly for ordinary reasons

**Research report, DRAFT 0.** Started 2026-09-25 (week 25), ahead of Phase 4 (Oct 5 to Dec 27).

> **Status of this draft.** Sections 1 and 4 are written. Every other section is a **stub**: its
> heading, the claim it has to carry, the numbers it will cite, and where they come from. Stubs
> are marked `STUB` so a reader can tell drafted prose from scaffolding at a glance. Nothing in a
> stub is new; every number is quoted from a committed `RESULTS.md`.
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

**It mostly did not work for the reasons the architecture was built to test.** Across 68 completed
numbered experiments, almost every architectural idea lost to a training idea:

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

**What a reader can reproduce, and how.** Re-evaluating any of the committed checkpoints
reproduces the published numbers exactly, on any machine tested. Retraining from a seed does not:
on a second machine the per-seed results scatter by up to 0.30 while two of three pre-registered
verdicts survive (EXP-067). Section 7 states the guarantee precisely.

---

## 2. What was asked `STUB`

**Must carry:** the original 12-month plan and its four phase checkpoints, quoted, so section 6
can grade against the plan as written rather than as remembered.

- The 2026-03-31 plan: Phase 0 foundations, Phase 1 single-region SNNs, Phase 2 multi-region
  brain, Phase 3 the cube capstone, Phase 4 this report and a public release.
- The Phase 3 criteria, verbatim from `neuromorphic-project-plan.md`: (1) solve from 1-move
  scrambles, 3-move stretch; (2) comparison against a monolithic network on at least one metric;
  (3) interpretability analysis; (4) clear documentation.
- The later, harder roadmap the project set itself: `road-to-a-solved-cube.md`, Stages 1 to 4.
- **Sources:** vault `neuromorphic-project-plan.md`, `docs/phase2-honest-assessment.md`,
  `docs/phase3-honest-assessment.md` section 1.

## 3. The system `STUB`

**Must carry:** enough architecture for a reader to follow section 5, and the one invariant that
decides what can be measured.

- The five regions and their neuron counts (192 sensory, 150 hippocampus, 150 prefrontal,
  12 router, 6 motor). Surrogate-gradient LIF neurons; `brain.step` costs about 90 ms.
- **The invariant:** with `recall=False` the policy head reads the sensory concept only, so
  **318 of 510 neurons are off the policy path**. State this before any result, not after.
- The task: raw facelets in, 6 moves out (the 2x2 simplification holds DLB fixed). Difficulty is
  exact BFS distance to solved, used as an instrument and never as an input.
- Train/held-out split per depth shell, capped at 200 held-out states; chance floors measured.
- **Sources:** `docs/architecture-spec-v1.md` to `v3`, `docs/adr/`, `CLAUDE.md` architecture
  invariants, EXP-029.

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
not a property of either part, which is exactly why pairing within seed works. (EXP-069, running at
the time of writing, asks the same question of the pretrained encoder at depth 5.)

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

## 5. Results `STUB`

**Must carry:** each line of inquiry as a claim, its verdict, and its mechanism where one was
found. In the order the project learned them, following `docs/phase3-honest-assessment.md` section
5, extended through week 25.

### 5.1 Depth, and the budget law `STUB`
- v1 baseline 2.2% at depth 3 against a 1.4% floor (EXP-029); collapse diagnosed, not incapacity
  (EXP-031/032). Curriculum: 2.2% to 50.0% (EXP-034/035).
- Break point at depth 5 (EXP-036), later **reframed as a budget, not a wall** (EXP-044/045/046):
  about 0.22 success per log10 of spend, no knee. Held out of sample at depth 8: predicted 0.0588,
  measured 0.0783, 95% interval [0.0523, 0.1044] (EXP-062).
- Frontier: depth 8 at 0.0783; depth 9 at 0.0163, at the floor. **The censoring trap**: steps near
  a bounded floor look cheaper than they are; use only steps clear of it.

### 5.2 The spiking encoder, the one neuromorphic win `STUB`
- Pretraining by inverse dynamics: +0.1880 at depth 4, +0.1908 at depth 5, depth 6 off the floor
  at 0.1037 (EXP-040). Fine-tuning during RL: +0.0900 at depth 6, 10-1-1, p 0.0020 (EXP-047).
  The gain lives in the encoder, not the head: +0.1312, p 0.0059 (EXP-048). It does not compound
  (EXP-049).
- Say plainly: surrogate gradients, not local learning.

### 5.3 The critic `STUB`
- Works; its benefit is within-episode state-dependence, not calibration (EXP-056/057).
  Replicated on fresh seeds at -0.0925, p 0.0156 (EXP-060). Ordinary RL.

### 5.4 Episodic memory `STUB`
- EXP-030 null on a policy that had learned nothing; EXP-059 memory hurts (-0.0954, p 0.0056);
  EXP-061 the recall is noise (+0.0210, p 0.4989 against matched noise); EXP-063 a learned
  attention over a *perfect* cache is still 0.198 worse than no memory. Closed at its ceiling.

### 5.5 The neuromodulatory bus `STUB`
- Made load-bearing and bought nothing (EXP-053 arm G). The neuromorphic claim for credit
  assignment is refuted.

### 5.6 Topology, and why it cannot be measured here `STUB`
- The monolithic control had 2.66x the on-path capacity and still lost narrowly at depths 2 and 3.
- The on-path-matched contrast is vacuous: bit-identical weights and outputs across 6 seeds
  (`tests/training/test_topology_contrast_is_vacuous.py`). The brain's own motor pathway collapses
  at every region learning rate tried (EXP-064/066). **Answering the topology question needs an
  architecture change, not an experiment.**

### 5.7 Reproducibility and the seed effect `STUB`
- EXP-067, EXP-068 and `docs/seed-effect.md`; see sections 4.5 and 7.

## 6. Graded against the plan `STUB`
- Phase 3 checkpoint: 3 of 4 criteria met or exceeded, 1 partial (criterion 2, and why it stays
  partial). The self-set roadmap: Stages 1 to 3 closed, Stage 4 priced out at about 33 days per
  seed. **Sources:** `docs/phase3-honest-assessment.md` sections 1 and 2.

## 7. Reproducing the results `STUB`
- **The guarantee: re-evaluate the checkpoints, never retrain from the seed.** Re-evaluation matches
  every headline metric to full float representation across machines, with one derived mean off by
  1 ULP. Retraining is byte-identical on one machine and not across two: 0 of 390 parameters match,
  and per-seed success scatters by up to 0.30, while the task layer (states, splits, scrambles) is
  byte-identical on 12 of 12 seeds (EXP-067).
- Commands, tracked artifacts, environment (torch 2.13.0+cpu; Python 3.10 and 3.13 both tested).
- **Sources:** `docs/reproducibility-audit.md`, EXP-067, `scripts/verify_*`.

## 8. Limitations and what would come next `STUB`
- 2x2 only; the 6-move set does not transfer to a 3x3. Depth 11 not attempted. Topology untested
  for the reason in 5.6. Surrogate gradients throughout.
- Extensions from the plan (local learning rules, neuromorphic hardware, scaling), each stated
  against what this project actually showed rather than as a promise.

## Appendix A. Experiment index `STUB`
- One row per EXP-001 to EXP-069: question, verdict, `RESULTS.md` path. Generate from the
  experiment folders rather than by hand, so it cannot drift.
