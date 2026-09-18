# Phase 3 Honest Assessment (the Rubik's Cube capstone)

**Written 2026-09-17**, covering EXP-029 to EXP-063, weeks 16 to 24.
Companion to `docs/phase2-honest-assessment.md`, which established the format: grade every
pre-registered criterion, then disclose what does not work before anyone has to ask.

> **The one-sentence version.** The checkpoint this phase was set in March is **exceeded by a wide
> margin**, the roadmap the project then wrote for itself is **not finished and its headline goal
> is priced out on evidence**, and **two of the three scientific questions it opened closed
> negative**. The most defensible thing Phase 3 produced is not a cube solver. It is a method that
> reliably detected when its own ideas did not work.

## 1. Scorecard against the pre-registered Phase 3 checkpoint

The criteria, quoted verbatim from `neuromorphic-project-plan.md`: *(1) Regionalized SNN solves
2x2 cubes from at least 1-move scrambles (3-move is stretch). (2) Comparison vs monolithic on at
least one metric. (3) Interpretability analysis. (4) Clear documentation.*

| Criterion | Grade | Note |
|---|---|---|
| (1) Solves from 1-move scrambles, 3-move stretch | **Exceeded** | Depth 1 was solved in the v1 baseline itself (87.5%, EXP-029). The stretch goal, depth 3, reached **50.0%** (EXP-035), **22.7x** the v1 baseline's 2.2%. The series now runs to **depth 8 at 0.0783** (EXP-062) against a *measured* chance floor of **exactly 0.0000**. |
| (2) Comparison vs monolithic | **Partial, and confounded in the direction that makes it MORE interesting** | It exists and is committed (EXP-029): depth 1 ties at 87.5% each, depth 2 is 38.0% against 30.6%, depth 3 is 2.2% against 0.8%. The gaps are weak (the depth-2 sd across seeds is ~24 points, depth 3 is near the 1.4% floor for both) and it was never re-run once the recipe worked. **But the control is mismatched on the axis that matters, against the regionalized arm, and the obvious fix turns out to be vacuous.** See section 2a and its 2026-09-17 correction: this criterion is not merely unmeasured, it is **unanswerable without an architecture change**. |
| (3) Interpretability analysis | **Met, then substantially retracted** | Extensive: decodability (EXP-033), specialization (EXP-027), the critic's mechanism (EXP-056/057), the recall's information content (EXP-061), the readout ceiling (EXP-063). **But five instruments were retired** (`docs/retired-instruments.md`), including the EXP-033 probe that carried most mechanism claims through weeks 17 to 20. The analysis is real; a large fraction of its original *conclusions* were withdrawn by later work. |
| (4) Clear documentation | **Met** | **All 35 Phase 3 experiments (EXP-029 to EXP-063) carry a committed results file** with provenance and a regeneration command, a standing rule since the 2026-07-13 audit found EXP-027's numbers living only in a gitignored `outputs/`. **28 of them also have a pre-registered spec** (EXP-036 onward, unbroken except EXP-041). Plus ADRs, `docs/retired-instruments.md`, `docs/playbooks/remote-experiment-runs.md`, and 663 tests. Repo-wide the results-file figure is 49 of 64 folders; the gaps are all Phase 0/1 learning exercises that predate the rule. |

**Three of four met or exceeded, one partial.** Criterion 2 is the weak one and is weak for a
reason worth stating: **the architecture question stopped being the interesting question.** Once
the curriculum and the budget law explained the depth series, nobody went back to re-run a
topology contrast that had been measured on a policy which had learned nothing.

### 2a. The monolithic control is neuron-matched on the wrong quantity, and it favours the control

This was found on 2026-09-17 by reading the existing code and numbers, with nothing re-run. It is
the largest free correction available to criterion 2, and it is the project's own standing habit
applied to itself: **ask what a control holds fixed besides the thing you named.**

`MonolithicBrain` is documented as "neuron-matched to the five-region `Brain`" and spends the
entire budget on one flat stack: `hidden = 510 - 64 = 446` plus `concept = 64`. So **all 510 of
its neurons are on the policy path.**

The regionalized arm's policy path is its **sensory region alone** whenever `recall=False`, which
is every arm in the phase except the memory ones. That region is **192 neurons**. The other 318
(hippocampus 150, prefrontal 150, router 12, motor 6) are off-path.

| arm | total neurons | ON the policy path |
|---|---|---|
| regionalized | 510 | **192** |
| monolithic | 510 | **510** |

**The control therefore had 2.66x the on-path capacity, and still lost at depth 2 (30.6% against
38.0%) and depth 3 (0.8% against 2.2%).** "Matched on total neuron count" reads as scrupulously
fair and is in fact a 2.66x handicap on the only axis the policy can use.

Three consequences, and the third decides what to do about criterion 2:

1. The regionalized arm's weak win is **less likely to be a width artifact than the raw numbers
   suggest**, because width ran the other way. That is a modest upgrade to the result.
2. It is still **not** evidence that the five-region topology helps. Both gaps are inside the seed
   noise, and a 2.66x capacity deficit losing narrowly is not the same as topology winning.
3. ==**Re-running the old contrast on the working recipe would NOT fix criterion 2.**== It would
   reproduce this exact confound at a higher success rate and cost about 24 cells of compute to
   learn nothing new. What criterion 2 actually needs is a design that **matches ON-PATH capacity**
   (a monolithic arm at 192, or a regionalized arm whose off-path regions are removed rather than
   merely bypassed) and pre-registers which of width and topology it is testing. That is a real
   experiment, not a re-run, and it is **deliberately not being done to close a checkpoint box.**

> **CORRECTION, 2026-09-17, before any cell was run: point 3's proposed experiment is VACUOUS and
> both of its suggestions are wrong.** Left above as written, because this document was tagged
> `phase-3-checkpoint` with that recommendation in it and silently editing it would hide the
> mistake rather than record it.
>
> The on-path-matched contrast compares a network **against itself**. `Brain` builds its sensory
> region as `SensoryCortex(n_obs=144, hidden=128, concept=64, seed=s)`;
> `MonolithicBrain(total_neurons=192, content=64)` builds `SensoryCortex(n_obs=144, hidden=192-64,
> concept=64, seed=s)`. Same module, same arguments, same seed. **Measured across 6 seeds through
> the real `make_agent` path: every weight bit-identical, and the concept identical at
> `max|diff| = 0.000e+00`.** The second suggestion fails for the same reason: removing regions that
> are already architecturally disconnected from the action changes nothing.
>
> **The real conclusion is stronger than the experiment would have been.** The topology question is
> not answerable by ANY arm-versus-arm contrast in this configuration, because the topology is not
> on the policy path. Total-matching measures width (in the control's favour, as above);
> on-path-matching measures nothing. There is no third arm that does not first require **changing
> the architecture so that a region other than `sensory` influences the action**. That is v2 design
> work, not a checkpoint experiment.
>
> **This is EXP-030's lesson recurring**: *a path-matched control can turn out bit-identical to the
> arm it is controlling for.* It was caught this time because the identity was checked **before**
> dispatch rather than after, and it cost nothing instead of 24 to 48 cells.
> `tests/training/test_topology_contrast_is_vacuous.py` locks it in, and **a failure of that file
> is good news**: it would mean the topology had become measurable for the first time.

**Criterion 2 is therefore graded Partial and left Partial, with the confound documented and the
correct experiment specified.** Closing it by re-running a contrast whose flaw is now understood
would be the box-ticking this assessment exists to avoid.

## 2. Scorecard against the roadmap the project wrote for itself

`road-to-a-solved-cube.md` set four stages after the March plan, and they are the harder and more
honest test. This is where the phase does not finish.

| Stage | Goal | Status |
|---|---|---|
| 1 | Prove generalisation, find the break point | **DONE** (2026-08-04, EXP-036). And the break point turned out not to exist as an architectural fact: see below. |
| 2 | Engage the encoder | **DELIVERED**, and its **stated success metric was retired** mid-stage (the EXP-033 probe) and replaced. Fine-tuning the encoder during RL is confirmed (EXP-047) and the gain lives in the encoder rather than its head (EXP-048), but it does **not compound** (EXP-049). |
| 3 | Credit assignment, the value function on `neuromod` | **CLOSED on both threads, and the neuromorphic one closed NEGATIVE.** The critic works and replicates (EXP-056/057/060). ==The `neuromod` bus was made load-bearing and bought nothing== (EXP-053 arm G). |
| 4 | Depth-11 random scrambles, called *"the actual deliverable"* | **PRICED OUT at ~33 days of compute per seed** (EXP-062), about 18 days wall clock for a 12-seed arm. |

**Stage 4 is the stated deliverable of the whole capstone and it is not going to happen on this
hardware.** The road note called it *"genuinely achievable... nothing about it requires new
science"*, which was written **before** week 20's budget law existed. That judgement was correct
about the science and wrong about the arithmetic. **Nothing new is needed scientifically; it is
the exchange rate that stops it.**

## 3. What genuinely works

- **The recipe.** Depths 3 to 8 all work given budget. Depth 3 at 50.0%, depth 6 at 0.3525
  (EXP-049), depth 7 at 0.2004 (EXP-053 arm B, replicated by EXP-060), depth 8 at 0.0783.
- **The curriculum is the single largest win in the phase** (EXP-034/035): depth 3 went from 2.2%
  to 50.0% with **no architectural change at all**.
- **The budget law** (EXP-044/045/046, held out of sample by EXP-062): success is linear in the
  **logarithm** of spend, about **0.22 per log10**, with no knee. ==This is the single most useful
  thing the phase produced==, because it converts "can we reach depth N?" from an architecture
  question into an arithmetic one, and it is what priced Stage 4 out.
- **Training the spiking encoder** (week 20, EXP-039/040/047/048). **This is the one genuinely
  neuromorphic change that pays.** Everything else that worked is ordinary RL.
- **The critic**, confirmed, mechanism identified as within-episode state-dependence, and
  independently replicated on fresh seeds at -0.0925 (p 0.0156). It is also **ordinary RL**.
- **The instrumentation and the test suite.** 663 tests, per-cell durable records, byte-identical
  seeded reruns, `--skip-existing` resumability.

## 4. What does NOT work, and must be disclosed

### The neuromorphic claim

**Refuted for Stage 3.** `neuromod` was given a real job and produced no measurable benefit
(EXP-053 arm G). The five-region topology has never been shown to beat a width-matched
alternative on this task: **318 of the brain's 510 neurons are off the policy path** whenever
`recall=False`, which is every arm except the memory ones. The honest statement is that this is a
**characterized spiking feature-extractor with an ordinary RL policy on top**, not a
demonstrated-superior neuromorphic architecture.

### Episodic memory

**Closed negative, three times, with the third being a ceiling.**

| experiment | finding |
|---|---|
| EXP-030 | null, but measured on a policy that had learned nothing |
| EXP-059 | memory actively **HURTS** a working policy (-0.0954, p 0.0056) |
| EXP-061 | and the recall block is **noise**: matched-magnitude noise performs the same (+0.0210, p 0.4989) |
| EXP-063 | a learned attention over a **perfect** episodic cache is **0.198 worse than no memory at all** |

EXP-063 was built as a **ceiling instrument** precisely so a negative would be decisive, and it
is: no readout over the lossy attractor read can beat one over a perfect cache. **The hippocampus
earns no place on this task.**

### Statistics and method failures, stated plainly

- **EXP-058 is VOID** on a gate that could not pass (`mean_n_stored > 10` where episode length
  capped it at 7.76). 20.2 hours of compute, no result.
- **EXP-057's gate passed with 2.0x margin, not the three orders its spec claimed**, because it
  was calibrated at depth 3 and evaluated at depth 7. A near-miss false VOID.
- **8 experiments in this phase used a validity gate and 2 of them got it wrong**, both from
  the same defect: a threshold set in one regime and applied in another. Every gate since has
  been calibrated across the attainable range and expressed as a bounded or scale-free quantity.
- **Five instruments retired.** Each worked as a threshold and failed as a gradient.
- **EXP-063's read-block magnitude confound was not pre-registered** and both numbers needed to
  catch it were available before dispatch. Its two cross-arm secondaries cannot separate "the
  learned readout hurts" from "a bigger block on the policy path hurts".
- **EXP-054 refuted a hypothesis that four earlier experiments' framing depended on**, requiring
  those write-ups to be rewritten rather than amended.

### Scope never reached

- **No 3x3 cube.** The 6-move action set is a 2x2-only simplification.
- **Stage 4 not attempted**, per section 2.
- **Criterion 2's monolithic contrast never re-run** on the working recipe.
- **The magnitude-matched readout follow-up** named in EXP-063's RESULTS.md is deliberately
  not run, because pooling it with existing numbers would be optional stopping.

## 5. The through-line, for the write-up

Phase 3 asked whether a five-region spiking brain could learn to solve a cube. The sequence of
answers, in order:

1. **It cannot, at first** (EXP-029: 2.2% at depth 3, near a measured floor).
2. **The policy was collapsed, not incapable** (EXP-031/032). Fixing collapse was not the lever.
3. **The curriculum was the lever** (EXP-034/035): 2.2% to 50.0%, no architecture change.
4. **The depth series was a budget series** (EXP-044/045/046). There was never an architectural
   wall, only an exchange rate. ==This retroactively reframes every "break point" claim made in
   weeks 17 to 19.==
5. **The encoder was the second lever**, and training it is the one neuromorphic win (week 20).
6. **The critic was the third lever**, and it is ordinary RL (EXP-053/056/057/060).
7. **Memory is not a lever at all**, established three times and finally at its ceiling
   (EXP-030/059/061/063).
8. **`neuromod` is not a lever** (EXP-053 arm G). The neuromorphic claim for Stage 3 is refuted.
9. **The deliverable is priced out** (EXP-062), by the very law that made the phase legible.

**The shape of the story is that almost every architectural idea lost to a training idea, and the
project kept finding that out rather than not finding it out.** Two of the three lines it closed
closed negative. That is the result, and writing it up as anything else would be the failure the
whole practice exists to prevent.

## 6. What the discipline cost and what it bought

**Cost:** EXP-058 VOID (20.2 h). EXP-059 attempt 1 destroyed by a Windows Update reboot (19 CPU-h,
misdiagnosed as a sleep for 13 hours). EXP-062 attempt 1 partially destroyed (~4 CPU-h). Multiple
experiments run purely as controls or replications rather than as new findings.

**Bought:**

- **A two-arm memory design would have published a false win three times** (EXP-030, EXP-058,
  EXP-059). The amnesic control is the only thing that reveals it.
- **EXP-028's headline refuted its own pre-registration** (Phase 2), which is exactly why
  pre-registering is worth the friction.
- **Mutation testing found two instruments that were blind to the defect they existed for**
  (EXP-061's leak detector, EXP-059's aggregator gate) and one **test disarmed by its own fixture**
  (EXP-063).
- **The censoring trap**, found twice and in two different shapes: extrapolating a bounded metric
  (EXP-062, a 3.5x under-pricing) and contrasting one (EXP-063, where the confirmed primary is
  largely a collapse-frequency artifact).
- **n >= 12 seeds**, after n=5 lied in EXP-026 (Phase 2) and the de-noised result flipped.

## 7. What Phase 4 inherits

- A working, characterized depth-3-to-8 recipe with a committed record per cell.
- The budget law, which prices any future depth question in one line of arithmetic.
- Three closed scientific lines, two of them negative, all with mechanisms rather than just scores.
- A standing methods note (`docs/retired-instruments.md`), a gate-calibration rule, and a
  test-strength rule, each written from a specific failure that is named and dated.
- **One honest headline to write**: the spiking encoder trained during RL is the neuromorphic
  contribution. Everything else that worked was ordinary reinforcement learning, and the project
  is able to say so because it built the instruments to tell the difference.
