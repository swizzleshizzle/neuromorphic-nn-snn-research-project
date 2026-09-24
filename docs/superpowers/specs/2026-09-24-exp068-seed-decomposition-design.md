# EXP-068 design - is the seed effect the TASK DRAW or the TRAINING TRAJECTORY?

> **PRE-REGISTERED. Committed before any EXP-068 number exists.**
> **Date:** 2026-09-24 · **Phase:** 3 (final compute slot) · **Grounds:** `docs/seed-effect.md`,
> EXP-067, EXP-036.

## 0. The question, and why it is now worth a run

`docs/seed-effect.md` established that **seed quality is a large, reproducible, inherited
confound**: sd **0.0906**, correlating **+0.419** across 8 arms from five experiments, positive in
28 of 28 arm pairs. It is banked in `CLAUDE.md` as a standing hazard worth more than most of this
project's published effects. ==What it does not say is WHERE the effect lives.==

A cube seed fixes two independent things at once:

| | what it fixes | portable across machines? |
|---|---|---|
| **`split_seed`** | the train/held-out state split, i.e. **the task draw** | **YES, proven.** EXP-067's floor arm is byte-equal on 12 of 12 seeds |
| **`train_seed`** | head init, action sampling, the env scramble stream, the readout rng, i.e. **the training trajectory** | **NO, proven.** 0 of 390 parameters match, cosine 0.524 |

**EXP-067 already put a number on the split of those two, by accident.** Because a machine change
preserves the task draw exactly and redraws the trajectory entirely, the cross-machine per-seed
correlation **estimates the task draw's share of variance**. It measured **r = +0.428, r-squared
0.183**.

==So there is an independent, quantitative, out-of-sample prediction to test: the task draw should
account for roughly **18%** of per-seed variance, and the training trajectory for the rest.== That
is what makes this a test rather than an exploration.

**`split_seed` and `train_seed` are already separate, crossable fields on `CubeConfig`**, with
`resolve_seed` falling back to `cfg.seed`, and `tests/training/test_seed_split.py` already covering
the split. No new machinery is needed, which is why this fits the last compute slot.

## 1. Design - a full 10x10 cross at depth 3

`split_seed` and `train_seed` each take the values **0 to 9**, fully crossed: **100 cells**. The
configuration is EXP-036's depth-3 cell, obtained by `dataclasses.replace` on the config that
EXP-036's own `sweep_configs` builds, **never retyped**, so every other field matches by
construction.

**Depth 3** because it is the cheapest cell that works (EXP-036: 0.3972 against a 0.0139 floor),
because EXP-067 measured its cross-machine correlation there, and because depths 5 and 6 sit at or
near the floor where there is no variance to decompose.

**The diagonal (`split == train`) is the ordinary configuration**, so the grid contains the normal
experiment as a special case and the decomposition is anchored to something known.

> [!danger] **THE FILENAME COLLISION TRAP, AND IT IS DOCUMENTED IN THE CODE ITSELF.**
> `record_filename` encodes tag, arm, depth, seed and sigma, and **NOT `split_seed` or
> `train_seed`**. Its docstring names *"a seed-decomposition sweep"* as exactly the case that
> silently lands 100 cells in one file. ==The tag therefore encodes both seeds== (`exp068_sp3tr7`),
> and the driver asserts all 100 record filenames are distinct **using the real
> `record_filename`** rather than a copy of it, as that docstring asks.

## 2. The statistic - a two-way decomposition, not a correlation

For cell value `x[i][j]` with `i` the split seed and `j` the train seed:

```
x[i][j] = mu + A[i] + B[j] + I[i][j]
```

Two-way ANOVA without replication (there is no replication to have: within one machine a cell is a
deterministic function of its two seeds). Variance components:

```
sigma2_I = MS_I                          df 81
sigma2_A = (MS_A - MS_I) / 10            df  9      the TASK DRAW
sigma2_B = (MS_B - MS_I) / 10            df  9      the TRAINING TRAJECTORY
```

**A variance decomposition rather than a correlation, deliberately.** The correlation form of this
question is hopeless at this scale: EXP-067's own r of +0.428 carries p = 0.1686 at n = 12, and
detecting an r near 0.18 against zero needs n of roughly 150. ==Row and column means over a 10x10
grid average out the interaction term and give each main effect 9 degrees of freedom against 81,
which is the whole reason for crossing rather than sampling.==

**Significance by randomization, since the repo has no scipy.** The null is "neither factor
matters": shuffle all 100 values across the grid and recompute. 20,000 shuffles at a **fixed**
`random.Random(20260924)`, so the aggregator is deterministic and reruns reproduce.

## 3. Claims

### Claim 1, PRIMARY - the task draw is a MINORITY of the seed effect

**`sigma2_A / (sigma2_A + sigma2_B + sigma2_I) < 0.35`.**

- **CONFIRMED** below 0.35. EXP-067 predicts about **0.18**.
- **REFUTED** at or above 0.35, which would mean the effect lives in the task draw and EXP-067's
  cross-machine correlation was misread.

> [!warning] **WHAT THIS BAR CAN ACTUALLY DETECT, stated before dispatch.**
> This is the EXP-067 lesson applied. That experiment imported a tolerance calibrated in a
> zero-noise regime and it could only pass about a third of the time; the failure was not writing
> down what the bar could see.
>
> Here the share is estimated with **df 9 against 81**. Taking EXP-067's point estimate as truth
> (sigma2_A about 0.18 of total) the expected `MS_A / MS_I` is about **5.3**, comfortably clear of
> its randomization null. The sampling spread on the *share* is roughly **+/- 0.10**, so a true
> 0.18 sits about **1.7 standard errors** below the 0.35 bar.
>
> ==That is adequate, not generous, and it is stated here rather than discovered afterwards.== A
> true share anywhere between about 0.25 and 0.45 will not be cleanly resolved by this design, and
> a result in that band must be reported as unresolved rather than rounded to the nearest verdict.

### Claim 2, SECONDARY - the trajectory's main effect resolves

**`MS_B / MS_I` exceeds its randomization null at p < 0.05.** This is the positive directional
prediction: if the seed effect is mostly the training trajectory, that main effect must be visible.
A null here with Claim 1 confirmed would mean the effect is mostly *interaction*, which is a third
answer and must be reported as such rather than folded into either.

### Claim 3, VALIDITY GATE - there is something to decompose

**Both conditions required:**

1. **Total sd across the 100 cells >= 0.05.** EXP-036 measured a depth-3 per-seed sd of **0.1158**
   on this exact configuration, so the expected value is about twice the bar and it **can pass**; a
   grid collapsed to one value reads ~0 and it **can fail**.
2. **At least one main effect resolves at p < 0.05.** If neither does, the grid is interaction and
   noise, there is no decomposition to report, and the experiment is **VOID**.

> **This gates the COMPARISON's resolution, not an arm's mechanism**, which is the week-24 lesson.
> Condition 1 alone would pass on a grid whose variance is entirely interaction, and Claim 1's
> share would then be a ratio of two noise estimates.

### On multiplicity

Claims 1 and 2 are one directional prediction each, from a single prior (EXP-067). Claim 2 is
tested at **0.05** and Claim 1 is a threshold rather than a test. No Bonferroni correction is
applied and none is needed; this is recorded so it is not read as an omission.

## 4. Cost

| | |
|---|---|
| Cells | **100** (10 x 10), all trained, no floor cells needed |
| Workers | **10**, which divides 100 into **10 clean waves** |
| Per cell | **TO BE MEASURED on the laptop before the full launch**, at 10 workers |
| Reference | 46 min/cell at depth 3 on the 2-core VPS at 2 workers (EXP-067) |

==The cost line is deliberately unfilled.== The only depth-3 figure in hand is a VPS measurement,
and the rule is to price from a **same-machine, same-worker-count** measurement. A 10-cell
calibration wave runs first and this section is amended with the measured number **before the full
launch**. Amending a cost before dispatch is legitimate and has precedent in EXP-060; **no claim,
threshold or test may be amended at any point.**

**Durability:** every cell writes its own record, and the launcher always passes `--skip-existing`,
so an interruption costs at most the one wave in flight.

## 5. What this cannot answer

- **Whether the split lives in the eval draw or the train draw.** `split_seed` fixes both sides of
  the partition at once, so this measures "the task draw" as a unit.
- **Anything about depths other than 3**, or about configurations with a pretrained encoder.
  EXP-036 takes none, which is why it isolates the question and equally why it generalises poorly.
- **The seed effect measured in `docs/seed-effect.md`**, which is at depth 5 across arms with an E0
  encoder. This tests the *mechanism* at depth 3 on a clean configuration; if the answers differ,
  the encoder is the obvious next suspect.
- **Interaction structure.** `sigma2_I` is estimated as a lump and not modelled.
