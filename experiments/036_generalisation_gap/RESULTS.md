# EXP-036 Results - the generalisation gap and the depth break point

> **STATUS: RUNNING. No numbers yet.** This skeleton was committed 2026-08-03 with the
> pre-registered claims already written in as unfilled rows, so the interpretation contract is
> on disk before the data is. Fill the tables in, then mark each claim CONFIRMED or REFUTED.
> **Do not edit a threshold while filling this in.** If a result lands awkwardly against a bar,
> that is the finding.
>
> **Provenance (to complete on fill-in):** 96 records (4 depths x 12 seeds x {trained, random}),
> 2x2 cube, concept readout, frozen brain at random init, curriculum (1..d), 10,000 episodes
> split across stages, evaluated on the held-out shell at `cfg.depth`. Run 2026-08-03/04 on the
> laptop `SwizzlesDuo` over SSH from the VPS with `--workers 16`. Dispatched at commit
> `87a965b`. Wall clock: ____. Exit code: ____. Records in
> `experiments/036_generalisation_gap/outputs/` (gitignored; the `*_head.pt` checkpoints beside
> them ARE tracked). Regenerate at the bottom.

## The question

Two, sharing one dispatch.

The vault's `road-to-a-solved-cube.md` set as Stage 1 "does the depth-3 policy generalise or
memorise its 90 training states?". **That question was already answered and the note was wrong
on the premise.** `split_shell` partitions depth 3 into 90 train / 30 eval, and EXP-035's
headline 50.0% was always **held-out** success. It generalises.

What had never been measured in any cube experiment is the **gap**: train-side success was
never reported, so it was unknown whether the policy overfits its 90 states or is
capacity-limited in a way coverage cannot fix. That decides whether a train-fraction sweep is
worth running at all.

Combined with the depth 4/5/6 break point because measuring the gap needs a retrain either way
(no trained weights were saved anywhere before this experiment), and a retrain that sweeps
depth answers both for one night of laptop time.

## Replication gate

**Nothing below is trustworthy until this passes.** The depth-3 cell is exactly EXP-035's
10,000-episode cell: same seeds, same curriculum `(1,2,3)`, same budget, same machine.

| | value |
|---|---|
| EXP-035 depth 3 @ 10k | 0.397 |
| EXP-036 depth 3 | ____ |
| delta (tolerance 0.02) | ____ |
| verdict | ____ |

A mismatch means either the EXP-036 code changes moved something, or this did not run on the
same machine (seeded runs are **not** reproducible across platforms; see the 2026-08-02
handoff section 4).

## The curve

| depth | n | held-out | train | gap | exact p | measured floor | bar | modal | entropy | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | | | | | | | | | | |
| 4 | | | | | | | | | | |
| 5 | | | | | | | | | | |
| 6 | | | | | | | | | | |

**Null gap** from the `random` arm, which cannot overfit, so this is what zero looks like at
this sample size. A trained gap must clear **this**, not clear zero.

| depth | mean null gap | sd | range | p |
|---|---|---|---|---|
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |

## Claim 1 - the gap. CONFIRMED / REFUTED / INCONCLUSIVE

Pre-registered: mean gap over 12 seeds, exact paired permutation over 4096 sign flips.

- mean gap **< 0.05** with **p > 0.05** -> coverage **refuted** as a lever, and the Stage-1a
  train-fraction sweep is **cancelled, not run**.
- mean gap **>= 0.15** with **p <= 0.05** -> overfitting **established**, sweep justified.
- anything else -> inconclusive, act on neither.

| | value |
|---|---|
| mean gap at depth 3 | ____ |
| sd | ____ |
| exact p | ____ |
| per seed | ____ |
| **verdict** | ____ |

**Consequence for the roadmap:** ____

## Claim 2 - the break point. CONFIRMED / REFUTED

Pre-registered: "working" requires held-out success to clear **both** twice the measured floor
**and** 0.10 absolute. **Prediction: it breaks at depth 5**, from EXP-033's raw-facelet probe
falling 0.956 / 0.766 / 0.598 at depths 3/4/5 against a chance of about 0.19.

| depth | held-out | bar | EXP-033 probe | verdict |
|---|---|---|---|---|
| 3 | | | 0.956 | |
| 4 | | | 0.766 | |
| 5 | | | 0.598 | |
| 6 | | | - | |

**First broken depth:** ____ · **Prediction:** ____

## Claim 3 - if depth 6 still works

If depth 6 clears both bars, Wall 1 sits further out than EXP-033's probe trend implied and the
linear head has more room than that experiment suggested. **That refutes Claim 2's prediction
and is logged as a refutation, not smoothed over.**

Status: ____

## Claim 4 - instruments

EXP-035 established that entropy alone cannot separate collapse from convergence: collapse is
low entropy with **high** modal fraction (0.987), convergence is low entropy with **low** modal
fraction (0.580). Any depth that failed must be diagnosed as one or the other, not just scored.

| depth | modal frac | entropy | diagnosis |
|---|---|---|---|
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |

## Claim 5 - the budget confound

**This is the break point at 10,000 episodes, not a property of the architecture.** EXP-035
showed depth 3 climbing 0.397 -> 0.500 between 10k and 30k without saturating, so a depth that
failed here may only be under-trained. Any statement of the form "the architecture cannot do
depth N" is unsupported by this experiment.

## Limitations

- Fixed at 10,000 episodes. A failing depth may be under-trained rather than out of reach.
- The curriculum is still untuned: `(1..d)`, equal stages, no adaptive advancement.
- One architecture. The frozen random encoder is unchanged, so EXP-033 Finding 1 (the encoder
  discards information a linear probe recovers from raw facelets) remains untested as a lever.
- Depth 3's held-out side is only 30 states, so its resolution is 1/30. Depths 4 to 6 are
  133/200/200.
- Nothing past depth 6. Shells grow fast and depth 11 is where a random scramble lives.

## Lead for the next experiment

To complete once the numbers exist. The candidates going in:

1. **The EXP-030 memory re-ask**, which head checkpointing has now made cheap.
2. **Tune the curriculum**, still free relative to buying budget.
3. **Decompose the residual variance** with `encoder_seed` x `train_seed`.
4. **The encoder** (vault Stage 2), which is where the SNN would first earn its place.

## Regenerate

```bash
.venv/bin/python -u experiments/036_generalisation_gap/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16

# re-read the records and re-apply the pre-registered rules at any time
.venv/bin/python experiments/036_generalisation_gap/aggregate.py

# confirm the code changes were neutral against EXP-035's depth-3 cell
.venv/bin/python scripts/verify_instrument_neutrality.py \
    experiments/035_budget_scaling/outputs experiments/036_generalisation_gap/outputs \
    --key-by cell --new-fields train_success_rate,n_train_eval,generalisation_gap --exempt tag
```
