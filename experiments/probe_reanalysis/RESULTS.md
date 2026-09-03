# Probe re-analysis - EXP-033, EXP-039 and EXP-047 with trajectory metrics beside them

> **NOT AN EXPERIMENT.** No data generated, no arm run, no claim pre-registered. This directory is
> deliberately unnumbered so it cannot be mistaken for one. Everything here is re-analysis of
> records already on disk, closing item 5 of the 2026-08-31 handoff.
>
> **HEADLINE: the probe got the DIRECTION right and carries no information about WHICH SEEDS or
> BY HOW MUCH.** Pretraining genuinely helped: success rose significantly at all three depths.
> But across **15 correlations between a probe movement and a behavioural one, 2 were nominally
> significant and 0 survived Bonferroni** - and both nominal hits run the **wrong way**.

**Provenance:** 2026-09-03, this VPS, no compute beyond permutation tests. Sources: EXP-039 (12
probe records), EXP-036 (frozen policy, depths 4-6), EXP-040 (pretrained policy, depths 4-6),
EXP-047 (`probe_confirm.json` plus its 12 lr-1e-4 policy records). Regenerate:

```bash
.venv/bin/python experiments/probe_reanalysis/reanalyse.py
```

Spearman p-values are permutation tests, 20,000 shuffles at a **fixed seed** so the numbers
reproduce. `spearman` is imported from EXP-054's aggregator rather than reimplemented.

## A. EXP-039's inference - the direction was right, the per-seed signal is absent

**The original inference:** inverse-model pretraining raises the linear probe at every depth,
every seed, at p 0.0005, the floor of the exact test at n=12. That is what motivated the entire
pretrained-encoder line from EXP-040 onward.

Per-seed probe gain, put beside the per-seed behavioural change (EXP-040 minus EXP-036):

| depth | probe top1 | success_rate | revisit_rate | optimality |
|---|---|---|---|---|
| 4 | **+0.3396**, 12-0-0, p 0.0005 | **+0.1880**, p 0.0337 | +0.0658, p 0.2778 | -0.0925, p 0.3604 |
| 5 | **+0.2537**, 12-0-0, p 0.0005 | **+0.1908**, p 0.0020 | +0.0177, p 0.7710 | +0.0612, p 0.6021 |
| 6 | **+0.2317**, 12-0-0, p 0.0005 | **+0.1037**, p 0.0039 | -0.0725, p 0.2036 | **+0.4959**, p 0.0039 |

**Pretraining worked.** Success rose significantly at all three depths, and at depth 6 optimality
moved too. Nothing here retracts the pretrained-encoder line.

**But the probe does not rank the seeds.** Spearman between each seed's probe gain and its own
behavioural change:

| depth | vs success_rate | vs revisit_rate | vs optimality |
|---|---|---|---|
| 4 | -0.021, p 0.9539 | +0.168, p 0.6053 | +0.245, p 0.4445 |
| 5 | -0.580, p 0.0507 | +0.545, p 0.0724 | **-0.671, p 0.0210** |
| 6 | +0.140, p 0.6677 | +0.420, p 0.1754 | +0.175, p 0.5888 |

Eight of nine unresolved, and the ninth is **negative**: at depth 5 the seeds whose probe improved
most became **less** optimal.

## B. EXP-047 did NOT over-claim, and this confirms it

This section reproduces EXP-047's own reported numbers exactly, which is the check that the
re-analysis is reading the data the way that experiment did:

| | this re-analysis | EXP-047's `RESULTS.md` |
|---|---|---|
| standard split, depth 4 | +0.0398, 11-0-1, p 0.0010 | +0.0398, 11-0, p 0.0010 |
| leak-free slice, depth 6 | +0.0050, 6-5-1, p 0.5732 | +0.0050, 6-5, p 0.5732 |

| correlation | rho | perm p |
|---|---|---|
| standard probe delta ~ success_rate | +0.042 | 0.9026 |
| standard probe delta ~ revisit_rate | **+0.678** | **0.0184** |
| standard probe delta ~ optimality | +0.273 | 0.3899 |
| leak-free probe delta ~ success_rate | -0.245 | 0.4460 |
| leak-free probe delta ~ revisit_rate | +0.224 | 0.4823 |
| leak-free probe delta ~ optimality | -0.308 | 0.3332 |

**EXP-047 pre-registered Claim 2 as SPLIT and reported the weaker slice**, concluding *"the policy
got better, and we cannot show the representation did."* **Nothing needs retracting.** It is
included because item 5 named it, and because confirming a correctly hedged inference is worth
doing rather than assuming.

These correlate a probe **change** against a behavioural **level**, because EXP-047's fine-tuned
arm has no seed-matched frozen twin whose trajectory metrics could be differenced. That is weaker
than section A and is labelled as such in the output.

## C. EXP-033 - one finding is unrepairable, the other never needed repairing

**Finding 1 (width helps the probe, saturating) has NO policy counterpart at any width but 64.**
Policies were never trained at 128, 256 or 512, so there are no trajectory metrics to pair against
the width sweep and none can be produced without new runs. **The width inference rests entirely on
a retired instrument.** It should be cited as *"wider random projections decode better"* and never
as *"width would not have helped the policy"* - the second was never measured.

**Finding 2 (the representation is not the first bottleneck) survives, and is different in kind.**
It did not infer policy quality from a probe number. It fitted the probe, then **ran it as a greedy
policy in the real environment** and compared success rates directly: 0.481 against REINFORCE's
0.022 at depth 3, a 22x gap on an identical representation and head shape. That is a behavioural
measurement, so the probe's retirement does not touch it. Its own 2026-08-02 correction (EXP-035
seeds exceeding the 0.481 "ceiling", so it is a reference point and not a ceiling) already stands.

## D. The tally, with multiplicity counted

**15 correlations tested. Bonferroni 0.0033. Nominally significant at 0.05: 2. Surviving: 0.**

| correlation | rho | p | |
|---|---|---|---|
| d5 probe ~ optimality | **-0.671** | 0.0210 | does not survive |
| EXP-047 standard probe ~ revisit_rate | **+0.678** | 0.0184 | does not survive |

**Both nominal hits run the wrong way.** A bigger depth-5 probe gain went with **lower**
optimality, and a bigger EXP-047 probe gain went with a **higher** revisit rate, which is more
cycling and therefore worse behaviour. Neither is a finding at n=12 under correction. But if the
probe tracked behaviour, chance hits would not be systematically inverted.

## What this changes

1. **The pretrained-encoder line is not weakened.** Pretraining raised success significantly at
   depths 4, 5 and 6. The intervention worked; the instrument that motivated it simply never
   measured why, and its unanimity invited a stronger reading than it could support.
2. **"Unanimous at p 0.0005" is not evidence of behavioural relevance.** The probe was unanimous
   at every depth in EXP-039 and still ranked the seeds no better than chance. Unanimity measures
   consistency of the *instrument*, not its connection to the outcome.
3. **EXP-033's width finding must be quoted narrowly** from now on, because it cannot be checked
   against behaviour without new runs.
4. **This is independent corroboration of the retirement.** EXP-050 retired the probe on
   *direction* grounds: two objectives moved it in opposite directions, both unanimous at p 0.0005.
   This adds a *per-seed* argument from entirely different records.

## What is NOT claimed

- **Not that the probe measures nothing.** It clearly measures linear decodability. The claim is
  narrower: decodability does not predict this policy's behaviour, per seed, at n=12.
- **Not that pretraining should be reconsidered.** Section A's success column is significant at all
  three depths.
- **Not a null with power behind it.** n=12 gives weak correlation power, so "no relationship
  resolved" is a bound. What makes the pattern persuasive is that it is 13 of 15 unresolved plus 2
  inverted, not any single number.
- **Nothing about `S`, the entropy trace, or move-accuracy.** Those are separate instruments with
  their own evidence, already recorded elsewhere.
