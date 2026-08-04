# EXP-036 Results - the generalisation gap and the depth break point

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The skeleton,
> with every claim written in as an unfilled row, was committed at `248f437` **before the data
> existed**; this is that skeleton filled in. No threshold was edited while filling it in.
>
> **Provenance:** 96 records (4 depths x 12 seeds x {trained, random}) plus 48 head checkpoints.
> 2x2 cube, concept readout, frozen brain at random init, curriculum `(1..d)`, 10,000 episodes
> split across stages, evaluated on the held-out shell at `cfg.depth`. Run 2026-08-03/04 on the
> laptop `SwizzlesDuo` over SSH from the VPS with `--workers 16`, dispatched at commit `87a965b`.
> Wall clock about 21 h against an 11.8 h estimate (see Limitations). Exit clean, **zero
> tracebacks**. Records in `outputs/` (gitignored); the `*_head.pt` checkpoints beside them ARE
> tracked. Regenerate at the bottom.

## Headline

**The curriculum breaks at depth 5, exactly as pre-registered.** Depth 4 still works at 15.9%,
which is 51x its measured floor. Depth 6 is a total failure: **0.0000 across all twelve seeds**,
and the instruments say it is policy collapse, not merely weak learning.

The gap question came back **inconclusive by its own pre-registered rule**, and that is the
honest answer rather than a disappointing one: the gap is real and significant, but its
magnitude landed in the dead zone the contract defined in advance.

## Replication gate

**Nothing below would be trustworthy without this.** The depth-3 cell is exactly EXP-035's
10,000-episode cell: same seeds, same curriculum, same budget, same machine.

| | value |
|---|---|
| EXP-035 depth 3 @ 10k | 0.397 |
| EXP-036 depth 3 | **0.3972** |
| delta (tolerance 0.02) | **+0.0002** |
| verdict | **PASS** |

Stronger than the tolerance implies. A field-by-field comparison of all twelve depth-3 cells is
**byte-identical on every measured quantity**:

```
compared:  12
identical: 12
differing: 0
PASS: every pre-existing field reproduced exactly. The change is neutral.
```

So the train-side evaluation and head serialisation added in `ad19c41` perturbed nothing, which
is what the ordering constraint in `run_cube_baseline` was designed to guarantee.

## The curve

| depth | n | held-out | train | gap | exact p | measured floor | bar | modal | entropy | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 12 | **0.3972** +-0.121 | 0.5065 | **+0.1093** | **0.0059** | 0.0139 | 0.100 | 0.630 | 0.184 | working |
| 4 | 12 | **0.1591** +-0.074 | 0.1642 | +0.0050 | 0.6289 | 0.0031 | 0.100 | 0.685 | 0.217 | working |
| 5 | 12 | **0.0396** +-0.027 | 0.0367 | -0.0029 | 0.7520 | 0.0000 | 0.100 | 0.779 | 0.236 | **BROKEN** |
| 6 | 12 | **0.0000** +-0.000 | 0.0000 | +0.0000 | 1.0000 | 0.0008 | 0.100 | 0.975 | 0.204 | **BROKEN** |

Per-depth detail:

| depth | best seed | seeds at 0.000 | n train-eval | n held-out |
|---|---|---|---|---|
| 3 | 0.6333 | 0/12 | 90 | 30 |
| 4 | 0.2782 | 0/12 | 200 | 133 |
| 5 | 0.0950 | 2/12 | 200 | 200 |
| 6 | 0.0000 | **12/12** | 200 | 200 |

**Null gap** from the `random` arm, which cannot overfit, so this is what zero looks like at this
sample size. A trained gap must clear **this**, not clear zero.

| depth | mean null gap | sd | range | p |
|---|---|---|---|---|
| 3 | -0.0065 | 0.0312 | [-0.100, +0.011] | 0.8203 |
| 4 | -0.0023 | 0.0053 | [-0.015, +0.005] | 0.1875 |
| 5 | +0.0004 | 0.0014 | [+0.000, +0.005] | 1.0000 |
| 6 | -0.0008 | 0.0019 | [-0.005, +0.000] | 0.5000 |

Every null gap is within 0.007 of zero, so the instrument is sound and the depth-3 gap of
**+0.1093 clears it by more than 16x**.

## Claim 1 - the gap. INCONCLUSIVE

| | value |
|---|---|
| mean gap at depth 3 | **+0.1093** |
| sd | 0.0963 |
| exact p (4096 sign flips) | **0.0059** |
| per seed | -0.011, 0.089, 0.144, -0.011, 0.211, 0.167, 0.044, -0.056, 0.167, 0.189, 0.233, 0.144 |
| **verdict** | **INCONCLUSIVE** |

Pre-registered: below 0.05 with p > 0.05 refutes coverage; at or above 0.15 with p <= 0.05
confirms overfitting. **0.1093 at p 0.0059 satisfies neither.**

> [!important] The pre-registration earned its keep here
> The gap is **statistically unambiguous** - p = 0.0059, 9 of 12 seeds positive, clearing a
> measured null of -0.0065. Read post hoc it is very tempting to call this "overfitting
> established" and go run the train-fraction sweep. **The magnitude bar of 0.15 was set in
> advance precisely so that temptation could not decide it**, and 0.1093 does not reach it.
>
> The rule was written that way for a reason: a gap that is real but modest does not tell you
> that more coverage will help. It says the policy does about 11 points better on states it
> trained on, while still generalising to 39.7% on states it never saw.

**Consequence for the roadmap: the Stage-1a train-fraction sweep is NOT cancelled, and NOT
scheduled.** It stays parked. If it is ever run it needs a reason beyond this number.

**Note the gap only exists where the policy learns.** Depths 4, 5 and 6 have gaps of +0.005,
-0.003 and 0.000, all with p >= 0.63. A policy that has learned nothing cannot overfit, which is
a coherence check on the whole measurement.

## Claim 2 - the break point. CONFIRMED

Pre-registered: "working" requires clearing **both** twice the measured floor **and** 0.10
absolute. Prediction, from EXP-033's raw-facelet probe falling 0.956 / 0.766 / 0.598 at depths
3/4/5: **it breaks at depth 5**.

| depth | held-out | 2x floor | absolute bar | EXP-033 probe | verdict |
|---|---|---|---|---|---|
| 3 | 0.3972 | 0.0278 | 0.100 | 0.956 | working |
| 4 | 0.1591 | 0.0063 | 0.100 | 0.766 | working |
| 5 | 0.0396 | **0.0000** | 0.100 | 0.598 | **BROKEN** |
| 6 | 0.0000 | 0.0017 | 0.100 | - | **BROKEN** |

**First broken depth: 5. Prediction CONFIRMED.**

> [!warning] The revised bar is what made this readable, and the shipped driver proves it
> **The measured floor at depth 5 is exactly 0.0000** - across 12 seeds a random policy solved a
> depth-5 cube zero times. Under the original rule ("below twice the measured floor") the bar at
> depth 5 was **0.0000**, so a policy at 3.96% would have been scored **"working"**.
>
> That is not hypothetical. The dispatched commit `87a965b` predates the fix in `248f437`, so
> **the driver that actually ran printed `depth 5 ... working`**. `aggregate.py`, carrying the
> corrected rule, prints `BROKEN`. The defect was caught on synthetic records before any real
> number existed; this run is the proof it would have mattered.

## Claim 3 - if depth 6 still worked

It does not: 0.0000 on all twelve seeds. **Claim 2's prediction is not refuted**, and Wall 1 is
not further out than EXP-033's probe trend implied. Nothing to log as a refutation here.

## Claim 4 - instruments

Modal fraction rises **monotonically with depth**, and that is the mechanism:

| depth | modal frac | entropy | held-out | diagnosis |
|---|---|---|---|---|
| 3 | 0.630 | 0.184 | 0.3972 | convergence: low entropy, low modal fraction |
| 4 | 0.685 | 0.217 | 0.1591 | converging, drifting toward collapse |
| 5 | 0.779 | 0.236 | 0.0396 | substantially collapsed |
| 6 | **0.975** | 0.204 | 0.0000 | **collapse: one action for nearly the whole rollout** |

The uniform floor for modal fraction is 0.354. **Depth 6 at 0.975 is the EXP-032 collapse
signature almost exactly** (0.987 there), and EXP-035's discriminator is what makes it readable:
entropy alone would not separate depth 3 (0.184, working) from depth 6 (0.204, dead). Entropy is
essentially flat across all four depths while modal fraction moves 0.630 -> 0.975.

**So depth 6 did not fail by learning something weak. It failed by collapsing to a constant
action**, which is a different problem with different fixes.

## Claim 5 - the budget confound

**This is the break point at 10,000 episodes, not a property of the architecture.** EXP-035
showed depth 3 climbing 0.397 -> 0.500 between 10k and 30k without saturating. Depth 5 at 3.96%
may be under-trained rather than out of reach, and depth 6's collapse may be what a sparse signal
looks like before enough episodes to escape it.

**No statement of the form "the architecture cannot do depth N" is supported by this experiment.**

## Limitations

- **Fixed at 10,000 episodes.** See Claim 5. This is the single biggest caveat.
- **The curriculum is untuned**: `(1..d)`, equal stages, no adaptive advancement. Depth 6 splits
  10,000 episodes across six depths, so it gets only about 1,667 episodes at each - including
  1,667 at depth 6 itself. That may be the whole story of its collapse.
- **One architecture.** The frozen random encoder is unchanged, so EXP-033 Finding 1 remains
  untested as a lever.
- Depth 3's held-out side is only 30 states, so its resolution is 1/30. Depths 4 to 6 are
  133/200/200.
- The train-side sample is capped at 200, so at depths 4 to 6 the gap is measured on a subset of
  the train side rather than all of it. Matched to the held-out cap by design.
- Nothing past depth 6. Depth 11 is where a random scramble lives.
- **Wall clock was about 21 h against an 11.8 h estimate.** Not a result caveat, but the cost
  model is unreliable: throughput ran at ~43% utilisation because 16 workers at 920 MB private
  each drove system commit to 96% and the workers spent most of their time paging. See the
  2026-08-03 handoff section 1.6.

## Lead for the next experiment

Ordered by what this experiment actually changed.

1. **Depth 4 is the frontier now, and it is a real one.** 15.9% held-out at 51x its floor, with
   no gap and no collapse - the policy is learning and simply capped. It is the natural target
   for the first intervention that is not "more episodes".
2. **Depth 6's failure is collapse, not weakness.** That is EXP-031/032 territory and those
   stabilizers were refuted as a lever *at depth 3*. Whether they help a genuinely harder problem
   is a different question and is now askable.
3. **Curriculum tuning moved up.** Equal stages give depth 6 only ~1,667 episodes at its own
   depth. Adaptive advancement, or weighting later stages, is free next to another budget step
   and is the most likely cheap win.
4. **The EXP-030 memory re-ask is now cheap**, because 48 trained heads are serialised and
   tracked. It no longer needs a retrain.
5. **Then the encoder** (vault Stage 2). EXP-033's probe gives a measured way to check whether
   pretraining raises the ceiling at depths 4 to 6, and depth 4 is where that would show first.

**Still refuted, do not revisit:** width (EXP-033), and volume alone (EXP-034).

## Regenerate

```bash
.venv/bin/python -u experiments/036_generalisation_gap/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16

# re-read the records and re-apply the pre-registered rules at any time
.venv/bin/python experiments/036_generalisation_gap/aggregate.py

# confirm the code changes were neutral against EXP-035's depth-3 cell.
# EXP-035's outputs hold BOTH its 10k and 30k cells, which share (arm, depth, seed, sigma),
# so the 10k records must be selected first or the collision guard correctly refuses to compare.
mkdir -p /tmp/e035_10k
cp experiments/035_budget_scaling/outputs/exp035_curriculum_e10000_*.json /tmp/e035_10k/
.venv/bin/python scripts/verify_instrument_neutrality.py \
    /tmp/e035_10k experiments/036_generalisation_gap/outputs \
    --key-by cell \
    --new-fields train_success_rate,n_train_eval,generalisation_gap \
    --exempt tag,encoder_seed,train_seed,split_seed
```

The three seed keys must be exempted because EXP-035's records were written before `12bbbf8`
added them to `CubeConfig`, so their config dicts lack the keys entirely. That is provenance,
not a perturbation.
