# EXP-034 Results - Learning Signal (the curriculum works, and volume alone does nothing)

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. **Provenance:** 48
> records (2 schedules x 2 episode budgets x 12 seeds), 2x2 cube, concept readout, frozen brain,
> evaluated at depth 3. Run 2026-08-01 on the laptop `SwizzlesDuo` over SSH from the VPS with
> `--workers 16`, wall clock 2h51m (16:48:23 to 19:39:18), exit 0, zero error markers. Records in
> `experiments/034_learning_signal/outputs/` (gitignored). **The contract was committed in `run.py`
> before any number existed** (commit `f1e695d`). Regenerate at the bottom.

## The question

EXP-033 measured an oracle ceiling: fitting the **same** frozen concept@64 and the **same**
`Linear(64 -> 6)` head with supervised labels solves **48.1%** of depth-3 cubes, while REINFORCE on
those identical 390 weights solves **2.2%**. The representation already carries the signal; the
learner does not find it.

This changes **only how the head is trained**. Frozen brain, 64-wide concept, linear head, depth-3
evaluation, all unchanged.

    schedule    direct       all episodes at depth 3 (shipped behaviour)
                curriculum   the SAME budget split across depths 1 -> 2 -> 3
    episodes    600          the budget every prior cube experiment used
                3000         5x, to separate "needs a better signal" from "needs more of it"

The budget is conserved inside a curriculum run (`curriculum_schedule` splits rather than multiplies),
so the curriculum arm never buys extra compute.

## Correctness audit

The `direct/600` cell is EXP-030/031/032's exact depth-3 concept configuration. Compared against the
EXP-031 records on all five measured fields across all 12 seeds: **identical**, success 0.0222 both.
The curriculum knob did not perturb the default path.

## Results

| cell | success | modal_frac | train_entropy | verdict vs pre-registered bars |
|---|---|---|---|---|
| direct/600 (baseline) | 0.0222 (sd 0.059) | 0.932 | 0.541 | no gain |
| curriculum/600 | 0.0972 (sd 0.134) | 0.888 | 0.515 | below the 0.10 bar |
| direct/3000 | 0.0194 (sd 0.058) | 0.968 | 0.220 | no gain |
| **curriculum/3000** | **0.2556** (sd 0.162) | 0.759 | 0.310 | **material gain** |

Paired per seed, n = 12, exact permutation over all 2^12 = 4096 sign flips:

| comparison | mean diff | W-L-T | exact p |
|---|---|---|---|
| curriculum @600 (matched budget) | +0.0750 | 6-1-5 | 0.0469 |
| **volume only, direct 600 -> 3000** | **-0.0028** | 2-1-9 | **1.0000** |
| curriculum @3000 | +0.2361 | **11-0-1** | **0.0010** |
| volume GIVEN curriculum, 600 -> 3000 | +0.1583 | 10-1-1 | 0.0107 |
| both vs baseline | +0.2333 | 10-1-1 | 0.0039 |

## Finding 1: more training is worth exactly nothing on its own

Five times the episodes, trained the shipped way, moves depth-3 success by **-0.003 at p = 1.000**.
Not a weak effect, not a slow effect: nothing, with a 2-1-9 split.

The instruments say why. From 600 to 3000 episodes of `direct` training, modal fraction **rises**
0.932 -> 0.968 and entropy **falls** 0.541 -> 0.220. With no reward to learn from, additional training
only drifts the policy further into determinism. **This is signal starvation observed directly rather
than inferred**, and it is the cleanest available refutation of "it just needs to train longer".

## Finding 2: the curriculum is what makes experience useful

At a matched 600-episode budget the curriculum is worth +0.075 (p = 0.047), a 4.4x improvement on
identical compute. That is real but it **misses the pre-registered 0.10 bar** at 0.0972, and the bar
does not move.

The result arrives when both are combined: **0.2556, an 11.5x improvement over baseline, 11-0-1 at
p = 0.001.**

**The structure is an interaction, not two main effects.** Extra episodes are worth -0.003 without the
curriculum and +0.158 with it. The curriculum does not merely add its own gain; it converts previously
worthless experience into useful experience, because a policy that has learned depths 1 and 2 actually
reaches the goal often enough at depth 3 to have something to reinforce.

## Finding 3: the gain is competence, not randomness

This is the attribution EXP-032 taught us to demand. There, success **fell** whenever entropy rose: the
entropy bonus bought exploration, not skill.

Here the opposite. Against baseline, `curriculum/3000` has **higher success (0.0222 -> 0.2556)** with
**lower entropy (0.541 -> 0.310)** and **lower modal fraction (0.932 -> 0.759)**. The policy became
simultaneously more successful, less random, and less degenerate. That combination is only available
to a policy that is genuinely reading its input.

## How close to the ceiling

`curriculum/3000` reaches **53.1% of the 0.481 oracle ceiling**, up from 4.6% at baseline. Per-seed:

```
0.000  0.067  0.067  0.100  0.233  0.267  0.300  0.333  0.400  0.400  0.433  0.467
```

The best seeds land at 0.433 and 0.467, essentially **at** the supervised ceiling measured in EXP-033.
So the ceiling is attainable by reinforcement learning given the right schedule. What separates the
arm's mean from the ceiling is now **seed variance**, not a representational limit.

## Pre-registered contract, claim by claim

1. **"Does any arm move depth-3 success materially above 0.022, fixed as reaching 0.10?" CONFIRMED.**
   `curriculum/3000` reaches 0.2556. `curriculum/600` does not, at 0.0972, and is reported as a miss.
2. **"If an arm reaches >= 0.35, the learning signal was the binding constraint." NOT MET.** 0.2556 is
   material but short of the ceiling neighbourhood. The learning signal is *a* binding constraint,
   demonstrably; the claim that it was *the* constraint is not established.
3. **"If NO arm clears 0.10 even at 3000 episodes, signal starvation is refuted." NOT TRIGGERED**, and
   the volume-only null makes starvation the better-supported reading rather than a weaker one.
4. **"Report the collapse instruments alongside." DONE**, and they carry Finding 3.

## Limitations

- **The trend has not saturated.** Curriculum success goes 0.097 -> 0.256 from 600 to 3000 episodes.
  Two budget points cannot say where it levels off, and the 0.35 bar may simply be a longer run away.
- **The curriculum is untuned**: stages fixed at (1, 2, 3) with an equal split. Stage proportions, more
  stages, and adaptive advancement are all unexplored.
- **One seed scored 0.000** and the sd is 0.162. The arm is high-variance, and the mean understates
  what the good seeds do while overstating what a single run should be expected to produce.
- **Depth 3 only.** Whether the curriculum extends to depth 4+ is untested, and the shells grow fast.
- The 0.481 ceiling is specific to concept@64. EXP-033 Finding 1 stands: a better representation would
  raise the ceiling itself.

## Lead for the next experiment

1. **Extend the budget under curriculum** (10,000 and 30,000 episodes). The cheapest question and the
   trend is still climbing. If it approaches 0.481, the representation becomes the binding constraint
   again and EXP-033 Finding 1 is the natural successor.
2. **Re-ask the EXP-030 memory question.** That null was measured on a policy that had learned nothing,
   which EXP-033 and this experiment together now make explicit. There is finally a policy that learns,
   so "does episodic memory help?" is a live question again rather than a foregone one. This is the
   most scientifically interesting thread the project has reopened.
3. **Then the encoder.** Once RL reliably approaches the ceiling, raising the ceiling is the next lever,
   and EXP-033 gives a measured way to check whether pretraining actually raised it.

## Regenerate

```bash
.venv/bin/python -u experiments/034_learning_signal/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16
```
