# EXP-035 Results - Budget Scaling (50% at depth 3, and the curve has not saturated)

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. **Provenance:** 24
> records (2 budgets x 12 seeds), 2x2 cube, concept readout, frozen brain at random init, curriculum
> (1, 2, 3), evaluated on the depth-3 held-out shell. Run 2026-08-02 on the laptop `SwizzlesDuo` over
> SSH from the VPS with `--workers 16`, wall clock 8h55m (21:53:43 to 06:48:24), exit 0, zero error
> markers. Records in `experiments/035_budget_scaling/outputs/` (gitignored). **The saturation rule
> was committed in `run.py` before any number existed** (commit `b6feee5`). Regenerate at the bottom.

## The question

EXP-034 established the curriculum as the active ingredient but had only two budget points and an
unsaturated trend. Two points cannot distinguish "still climbing" from "about to level off", and
those imply completely different next experiments. This adds 10,000 and 30,000.

## The curve

Curriculum (1, 2, 3), depth-3 held-out success, mean over the same 12 seeds throughout:

| episodes | success | modal_frac | train_entropy | best seed | seeds at 0.000 |
|---|---|---|---|---|---|
| 600 | 0.0972 (sd 0.134) | 0.888 | 0.515 | 0.367 | 5 |
| 3,000 | 0.2556 (sd 0.162) | 0.759 | 0.310 | 0.467 | 1 |
| 10,000 | 0.3972 (sd 0.121) | 0.630 | 0.184 | 0.633 | 0 |
| **30,000** | **0.5000** (sd 0.120) | 0.580 | 0.107 | 0.633 | 0 |

Paired per seed, n = 12, exact permutation over 4096 sign flips:

| step | mean diff | W-L-T | exact p |
|---|---|---|---|
| 600 -> 3,000 | +0.1583 | 10-1-1 | 0.0107 |
| 3,000 -> 10,000 | +0.1417 | 9-2-1 | 0.0137 |
| 10,000 -> 30,000 | +0.1028 | 10-2-0 | 0.0137 |

**Pre-registered saturation rule: 30,000 must beat 10,000 by at least 0.02. Observed +0.1028, five
times the bar. NOT SATURATED.**

For scale, against numbers measured elsewhere in this repo:

```
random policy (EXP-029, measured)     1.4%
v1 baseline (EXP-029 to EXP-032)      2.2%
EXP-034 curriculum @ 3,000           25.6%
EXP-035 curriculum @ 30,000          50.0%     <- 22.7x the v1 baseline
```

Nothing about the architecture changed to get here. Same frozen randomly-initialised brain, same
64-wide concept, same `Linear(64 -> 6)` head, same 390 trainable parameters. Only the order the
problems were presented in, and for how long.

## Correction: 0.481 was never a ceiling, and this experiment proves it

EXP-033 measured 0.481 by fitting a supervised linear probe on the frozen concept@64 and running it
as a policy. That number was called an "oracle ceiling" in EXP-033, EXP-034 and in the EXP-035
pre-registration, and **that framing was wrong**.

**Seven of twelve seeds now exceed it**, topping out at 0.633.

The probe was optimised for **per-step move-optimality**, not for solving. Those objectives diverge:
with a 9-step budget on a 3-move scramble there is slack, so a policy can take a non-optimal move and
still solve comfortably. Reinforcement learning optimises solving directly and can therefore beat a
readout trained on the stricter target.

What EXP-033 actually established survives intact and was always the real claim: the frozen
representation carries far more usable signal than v1 extracted. 0.481 remains a useful reference
point for that. It is not an upper bound, and this file is the record of the correction.

## Finding: entropy alone cannot distinguish collapse from convergence

At 30,000 episodes the training policy has entropy **0.107**, which is 6% of the log 6 = 1.792
ceiling. Read alone that is a more extreme number than anything in EXP-032, where low entropy meant a
degenerate policy.

The difference is the second instrument. Across the budget curve:

| | entropy | modal_frac | success |
|---|---|---|---|
| EXP-032 collapse (`beta=0, norm=True`) | 0.452 | **0.987** | 0.003 |
| EXP-035 convergence (30,000 episodes) | 0.107 | **0.580** | 0.500 |

**Collapse is low entropy with HIGH modal fraction: one action everywhere.** **Convergence is low
entropy with LOW modal fraction: a confident, state-dependent policy.** The uniform floor is 0.354, so
0.580 is much nearer varied play than constant play.

Either instrument alone would have been ambiguous here. Building both in EXP-031 paid off in an
experiment run four days later, for a reason not anticipated when they were written.

## The seed variance was transient

EXP-034 Finding 4 bounded the variance and could not attribute it, and flagged that EXP-035 would
partially answer it for free. It did.

| budget | sd | seeds at 0.000 |
|---|---|---|
| 3,000 | 0.162 | 1 |
| 10,000 | 0.121 | 0 |
| 30,000 | 0.120 | 0 |

Seed 6, which had the 3rd-best encoder of twelve and scored **0.000** at 3,000 episodes, reaches
**0.400 at 10,000**. The bottom five seeds gained the most while the top two drifted slightly down:
regression to a common level, which is what early-training luck looks like once you train past it.

**"Restart and select" would have been the wrong lesson. "Train longer" is the right one.** The
remaining sd of 0.120 is stable between 10,000 and 30,000, so what is left may be genuine and is what
the planned encoder/train seed decomposition (now possible, commit `12bbbf8`) should target.

## Limitations

- **Still not saturated**, so the ceiling of this approach is unknown. Per-step gains are shrinking
  (+0.158, +0.142, +0.103) against roughly 3-5x budget increases each time, which is consistent with
  a slow log-linear climb rather than an imminent plateau, but three points do not fix a functional
  form.
- **Cost is becoming the constraint**: 30,000 episodes is about 5.5 hours per run, so the next point
  on this curve costs a day of laptop time for one cell.
- **Depth 3 only.** Nothing here says the curriculum extends to depth 4 or beyond, and shells grow
  fast (534 states at depth 4, 2,256 at depth 5).
- **The curriculum is still untuned**: fixed (1, 2, 3), equal thirds, no adaptive advancement.
- **One architecture.** The frozen random encoder is unchanged, so EXP-033 Finding 1 (the encoder
  discards information a linear probe recovers from raw facelets) remains untested as a lever.

## Lead for the next experiment

The cheap lever is now expensive and the interesting questions are elsewhere:

1. **Re-ask the EXP-030 memory question.** That null was measured on a policy at 2.2% that had learned
   essentially nothing. There is now one at 50%. "Does episodic memory reduce cycling and improve
   solving?" is a live question for the first time, and the machinery all exists.
2. **Tune the curriculum** before buying more episodes. Stage proportions and adaptive advancement are
   free compared with another 3x budget step.
3. **Decompose the residual variance** with `encoder_seed` x `train_seed`, now that sd has stabilised
   at 0.120 and is no longer dominated by a transient.
4. **Then the encoder.** With RL no longer the binding constraint, EXP-033 Finding 1 becomes the next
   real lever, and EXP-033's probe gives a measured way to check whether pretraining raised anything.

## Regenerate

```bash
.venv/bin/python -u experiments/035_budget_scaling/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16
```
