# EXP-054 Results - the sequence-blindness hypothesis is REFUTED

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract was
> committed at `fb885ec` and amended at `cdd0ef8`, **both before any number existed**. No threshold
> was edited while filling this in.
>
> **Provenance:** 60 records, 5 arms x 12 seeds. Exact BFS shells `d = 1..6`, up to 60 states per
> shell, `num_steps=32`, `content=64`. Encoders read from EXP-052 (`e10`, `e20`), EXP-040 (`e40`)
> and EXP-050 (`e80`); the 0-epoch arm is reconstructed from `encoder_seed`. VPS `liquidweb-vps`,
> 2026-08-30, **8.8 s for all 60 cells** - the statistic trains nothing and `concept_rates` is
> batched. Zero tracebacks.
>
> **Regenerate:**
> ```bash
> .venv/bin/python -u experiments/054_sequence_blindness/run.py
> .venv/bin/python experiments/054_sequence_blindness/aggregate.py
> ```

## Headline

**Pretraining CREATES distance structure in its first 10 epochs and then nothing changes. Over-training does not destroy it. The sequence-blindness hypothesis, leaned on by four experiments, is refuted.**

| arm | epochs | S | sd | S_cross | level | move-acc | policy |
|---|---|---|---|---|---|---|---|
| E0 | 0 | **0.0100** | 0.0046 | 0.0115 | 0.0038 | - | 0.0000 |
| E10 | 10 | **0.0242** | 0.0057 | 0.0246 | 0.0228 | 0.383 | 0.2012 |
| E20 | 20 | 0.0241 | 0.0071 | 0.0252 | 0.0226 | 0.414 | 0.1850 |
| E40 | 40 | 0.0246 | 0.0079 | 0.0259 | 0.0235 | 0.437 | 0.1800 |
| E80 | 80 | 0.0244 | 0.0073 | 0.0261 | 0.0235 | 0.452 | 0.0887 |

**The four trained arms are indistinguishable.** Their spread is **0.00053**, which is **0.08x** the
typical within-arm sd of 0.0070. Meanwhile the policy across those same arms halves, 0.2012 to
0.0887.

## Claim 1 (PRIMARY) - does S fall with pretraining? **NOT CONFIRMED**

Pre-registered: confirmed if `S` decreases in at least 2 of 3 adjacent contrasts at `p <= 0.05`.

| contrast | delta | exact p | |
|---|---|---|---|
| E10 -> E20 | -0.0001 | 0.8984 | not significant |
| E20 -> E40 | +0.0005 | 0.6553 | not significant |
| E40 -> E80 | -0.0003 | 0.7441 | not significant |

**0 of 3.** Not a weak decrease - no decrease at all, and the deltas are two orders of magnitude
below the within-arm sd.

**This closes a four-experiment hand-wave.** EXP-050 and EXP-052 both invoked sequence-blindness to
explain why more pretraining hurts the policy. It does not happen. Whatever causes EXP-052's
collapse from 0.2012 to 0.0887, **it is not the loss of distance structure in the concept code.**

## Claim 3 - THE FLOOR, and it refutes the spec's own prediction

| | value |
|---|---|
| E0 minus E10, paired | **-0.0143** |
| exact p | **0.0020** |

**The spec predicted a random encoder might score HIGHEST**, on the Johnson-Lindenstrauss argument
that random projections preserve geometry, and pre-registered that reading in advance so it could
not be invented afterwards. **The opposite happened.** A random encoder has *less* shell structure
than a pretrained one, significantly so.

So pretraining is not spending sequence-structure to buy move-structure. It **builds** distance
structure, entirely within the first 10 epochs, and then saturates.

## Claim 2 - THE TRADEOFF. S explains neither curve.

`S` is flat across 10 to 80 epochs while move-accuracy climbs monotonically (0.383 to 0.452) and
the policy halves. It therefore explains **neither** the pretext metric's rise nor the policy's
collapse. Its only real movement is the 0-to-10 jump, which is where the policy also goes from
0.0000 to its maximum.

**`S` is a threshold, not a gradient**: it detects that pretraining happened at all, and says
nothing about how much.

## The result is NOT a clustering artifact

The spec amendment at `cdd0ef8` recorded that `S` is 91% within-shell clustering on a synthetic
fixture, and required `S_cross` (the same fit with `|d1 - d2| = 0` excluded) to be reported beside
it. **`S_cross` tracks `S` exactly** - 0.0115 at E0, then 0.0246 / 0.0252 / 0.0259 / 0.0261, equally
flat. The `level` control moves the same way (0.0038 then ~0.023 throughout), so nothing here is
representational collapse either.

The per-separation table shows a genuine **gradient**, not a step - E10 runs 0.0626, 0.0380, 0.0129,
-0.0012, -0.0298, -0.0751 as `|dd|` goes 0 to 5. The code really is graded by distance, and stays
that way at 80 epochs.

## Claim 4 - PASSED NOMINALLY, AND THE VERDICT IS UNINTERPRETABLE

The aggregator printed **CLAIM 4 PASSED**: within-arm correlations (-0.119, -0.245, -0.329, -0.147)
agree in sign with the between-arm correlation (-0.600), so `S` is not disqualified.

**Do not use that verdict.** The between-arm Spearman is computed over four arm means whose spread
is 0.08x the within-arm sd - Claim 1 found no significant difference between any adjacent pair. It
is a rank correlation over noise, and it would have produced some sign no matter what the data said.

> [!warning] THIS IS EXP-052'S PROCESS FAILURE, REPEATING IN THIS EXPERIMENT'S OWN AGGREGATOR
> EXP-052's aggregator "declared monotone decreasing from the ordering of four means, three of which
> were indistinguishable at p 0.49 to 0.84". The rule added afterwards was to require significance
> before naming a shape. **Claim 4 names a shape - an agreeing sign - from four indistinguishable
> means, and nothing gated it on Claim 1 having found variation first.**
>
> The disqualifier was built to fail safe, and it did: it never issued a false RETIRED. But it can
> issue a meaningless PASSED, and a later spec citing "S was checked and cleared" would be citing
> nothing. **Fix before reuse: gate Claim 4's between-arm comparison on the arms being
> distinguishable.**

> [!note] THE FIX LANDED 2026-08-31. Re-running the aggregator on these same records now prints
> **CLAIM 4 UNEVALUATED**, not PASSED.
> `axis_is_resolvable` tests the widest available contrast on each axis by exact paired
> permutation, and `claim4_verdict` now takes `between_resolvable` as a REQUIRED argument checked
> before any sign is compared. On these records:
>
> ```
> S:      widest contrast E40 vs E20, p 0.6553 -> NOT separable
> policy: widest contrast E10 vs E80, p 0.0063 -> separable
> ```
>
> The Spearman had one real axis and one noise axis, which is precisely the failure and is now
> visible in the output rather than only in this file. The gate fails safe in BOTH directions: an
> unresolvable axis can no more retire `S` than clear it, so it is checked ahead of the within-arm
> coherence branch. **The numbers in this document are unchanged; only the verdict wording is.**

**`S` is therefore neither cleared nor retired by this run.** It is unevaluated as a policy
predictor, because the epoch series does not vary it.

## What this changes

1. **Sequence-blindness is refuted and should stop being cited.** EXP-050 and EXP-052's
   interpretations need rewriting without it; their numbers stand.
2. **EXP-052's collapse is unexplained again.** The leading remaining candidates are the inverse
   model over-fitting single-step move identity at the expense of something else the policy needs,
   or the 80-epoch arm's optimizer reset (below). Neither is measured.
3. **Pretraining saturates by 10 epochs on this instrument too**, independently confirming EXP-052's
   efficiency finding from a completely different measurement.

## Caveats recorded in advance

- **E80 is WARM-STARTED, not trained 80 epochs from scratch.** `extend_pretrain.py:89` loads the
  E40 encoder and runs 40 more epochs with a **fresh optimizer**, while E10/E20/E40 come from
  scratch via EXP-052's sweep. The 40-to-80 contrast is therefore "40 more epochs with an optimizer
  reset", not a clean continuation. Recorded in the spec amendment before this run.
- **`S` is a shell-separability statistic**, not a sequence-sensitivity one, and is named that way
  here deliberately. See the `cdd0ef8` amendment.
- The shallow shells are small - depth 1 has 6 states, depth 2 has 27 - so their per-shell means are
  noisy and no claim rests on a single shell pair.
