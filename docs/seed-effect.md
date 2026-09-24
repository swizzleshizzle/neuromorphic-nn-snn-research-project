# The seed effect, and why EXP-060's "unexplained level shift" needs no explanation

**Standing note, not an experiment.** No new compute was run. Every number below comes from records
that already existed, plus 20 JSON records copied read-only from the laptop. Written 2026-09-23.

> **HEADLINE: the level shift EXP-060 flagged was already on record on 2026-08-22**, three weeks
> before the encoders it suspected were manufactured. The same seeds run low at every stage of the
> pipeline, in experiments that share nothing but the seed numbers.
>
> **And against the size of the seed effect, the split is not unusual: exact two-sided p = 0.5066**
> over all 646,646 partitions. It is the 50th percentile. There is nothing left to explain.

EXP-060's `RESULTS.md` said *"the level shift is unexplained and this experiment cannot explain
it"*, and named two suspects: the seeds themselves, and that its encoders were manufactured on
2026-09-12 rather than 2026-08-21. **The first suspect is right, the second is ruled out**, and a
third the write-up did not list (library drift between the two runs) is ruled out as well.

```bash
.venv/bin/python -u scripts/seed_effect.py                      # regenerates the numbers below
.venv/bin/python -m pytest tests/experiments/test_seed_effect.py -q
```

## 1. The deficit predates every suspect

Four stages ran the **same recipe over both seed blocks**, so each is a clean cross-block reading.
Exact two-sided permutation over 646,646 partitions, seeds 0-11 against 14-23:

| stage | depth | seeds 0-11 | seeds 14-23 | difference | p |
|---|---|---|---|---|---|
| `exp040_pre_d5` | 5 | 0.2304 | 0.1860 | -0.0444 | 0.5074 |
| `exp043_capped_d5` | 5 | 0.3412 | 0.2900 | -0.0513 | 0.4289 |
| `exp043_capped_d6` | 6 | 0.1800 | 0.1145 | **-0.0655** | 0.1110 |
| `exp047_ft_d6_lr0.0001` | 6 | 0.2700 | 0.2205 | -0.0495 | 0.2339 |

**The top two rows are the load-bearing ones.** EXP-043's own `RESULTS.md` dates them: *"RESOLVED
2026-08-22 at n=24: the depth-5 effect is real. Seeds 12-23 were pretrained and both arms executed
for them."* Those cells used August encoders and the August library, and they already show the
deficit at roughly the size EXP-060 later found at depth 7.

> [!warning] **These four rows are ONE observation, not four.** Every stage inherits the same
> per-seed E0 encoder, the same split and the same `train_seed`, and each stage feeds the next. They
> agree because they are the same seeds measured repeatedly, which is why none of them is quoted as
> independent confirmation and why the test that matters is section 4.

## 2. The September encoder batch is not different, measured rather than assumed

All 24 E1 encoders (`exp047_ft_d6_lr0.0001_*_encoder.pt`) are on this VPS, both the August batch
(seeds 0-13) and the September one (14-23). Compared frozen, against their own E0 parents:

| | seeds 0-11 | seeds 14-23 |
|---|---|---|
| `\|\|E1 - E0\|\| / \|\|E0\|\|` | 0.030077 (sd 0.001291) | 0.029371 (sd 0.000863) |
| identical key and shape structure | yes, all 24 | yes, all 24 |

**The two batches were fine-tuned by the same amount from the same kind of parent**, and the
difference between them is smaller than one within-batch standard deviation. A manufacturing defect
(wrong parent, wrong learning rate, truncated run) would not look like this.

## 3. Two further candidates, both eliminated

**Library drift.** EXP-056 ran at `e7e0d73`, EXP-060 at `2363871`. Three commits touch
`cube_baseline.py` between them. `e9a81c6` changes **only a docstring**. The other two add
`constant_critic` (opt-in, defaults False) and a recall probe that lives on `MemoryReadout`'s memory
branch. **EXP-060's arms use `readout="concept"` and neither switch is set**, so the library is
inert for them.

**A harder held-out draw.** `split_seed` fixes which 200 states of the shell are held out, so a seed
could in principle draw a hard evaluation set. It would then leave an *easier* train complement, so
the signature is eval down and train up. Across the 8 depth-5 arms spanning both blocks, the train
side moves **with** the eval side in 7 of 8 (mean -0.0162 against -0.0263). **The policy is worse,
not the exam harder.**

## 4. What is actually going on: seed quality is large, real, and inherited

A cube seed fixes the E0 encoder, the train/held-out split and the head init at once, so it is a
single persistent property every downstream experiment inherits.

| | |
|---|---|
| arms measured (depth 5, spanning both blocks) | **8**, across EXP-040, 043, 059, 061, 063 |
| mean per-seed correlation over the 28 arm pairs | **+0.419** (min +0.170, max +0.644) |
| arm pairs with a positive correlation | **28 of 28** |
| per-seed effect sd | **0.0906** |
| range, seed 21 to seed 11 | **-0.2048 to +0.1521** |

**Noise cannot correlate at +0.419 across experiments that share no records**, and these arms differ
in readout, in memory, in attention and in run date. Seed quality also transfers across depth: it
predicts depth-7 success with a positive correlation in **9 of 9** depth-7 arms.

**Against a per-seed sd of 0.0906, the 0-11 versus 14-23 split is exactly typical:**

| | |
|---|---|
| observed block difference in per-seed effect | **-0.0263** |
| exact two-sided p over 646,646 partitions | **0.5066** |
| block difference after dropping seeds 21 and 22 | **+0.0089** |

Seeds **21 and 22 are the 1st and 3rd worst of all 24**, and both fall in EXP-060's block. Removing
them flips the sign. EXP-060's own arms show it directly: seed 22 scored **0.0000** on the full
critic arm at depth 7, and seed 21 scored 0.0000 on the flattened arm.

**Per-arm block differences at depth 5 already range from -0.0728 to +0.0459 across the 8 arms.**
EXP-060's -0.0539 and -0.0818 sit at, and just past, the low end of a spread that was there all
along.

## 5. What this costs, and what to do about it

With a per-seed sd of 0.0906:

| comparison | se from the seed effect alone |
|---|---|
| one 12-seed arm mean | 0.0262 |
| **two DISJOINT seed sets, 12 against 10** | **0.0388** |
| **two arms on the SAME seeds, paired** | **0.0000, it cancels exactly** |

**A disjoint-seed comparison carries 0.0388 of noise before anything else is counted, and this
project's published effects run 0.05 to 0.09.** That is the same order as the findings.

1. **Never compare arms across different seed sets.** Pair within seed. EXP-060's primary did
   exactly this and survived a 0.06 level shift untouched, which is the design working as intended.
2. **A block difference of about 0.05 between two disjoint seed sets is the expected reading, not an
   anomaly.** Reach for section 4's test before reaching for an explanation.
3. **Residualise per seed before attributing a block difference to anything.** The raw per-seed mean
   is dominated by which arms covered which seeds.
4. **An arm that covers only one block cannot be used.** At depth 7 the block and the experiment are
   the same variable, which is why section 1 uses depths 5 and 6.

## What is NOT claimed

- **Not that nothing else contributed.** This shows no further cause is *needed*. It does not prove
  none exists, and a small contribution from the September run cannot be excluded at this n.
- **Not that the depth-7 shift was tested directly.** No depth-7 arm spans both blocks, so its
  magnitude is read against the depth-5 and depth-6 spread rather than permuted in place.
- **Not that seeds 21 and 22 are defective.** They are the low tail of an ordinary distribution.
  Dropping seeds because they score badly would be the outcome-dependent editing this repo exists to
  avoid, and section 4 quotes the figure both ways for that reason.
- **Not a statement about the critic.** EXP-060's Claim 1 is paired within seed and is untouched by
  everything here. Its pooled Claim 2 remains ill-advised, now for one clear reason (optional
  stopping) rather than two.
- **Not that `scripts/seed_effect.py` measures anything but success rate.** It is the outcome, not a
  proxy, and deliberately so. See `docs/retired-instruments.md`.
