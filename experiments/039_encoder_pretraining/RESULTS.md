# EXP-039 Results - the encoder learns, and it overtakes the observation exactly where the observation fails

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The
> interpretation contract was committed at `b8b95f6` **before the data existed**, and the two
> pre-data corrections it carries (`0c77e5b`, `bb88580`) are dated and explained in the spec.
> No threshold was edited while filling this in.
>
> **Provenance:** 12 records (12 seeds x 1 arm set). 2x2 cube, `SensoryCortex(144 -> 128 -> 64)`
> at the shipped configuration, inverse-model pretraining on 48,233 state pairs, 40 epochs,
> batch 256, lr 3e-3. Probe over 11,912 states at depths 1-6, 2,979 held out, stratified by
> depth, EXP-033's `probe.py` imported unmodified. Run 2026-08-08 on the **VPS**
> (`liquidweb-vps`, 2 cores) at **1 worker**, 921 s/seed, ~3.1 h total, exit 0, zero
> tracebacks. **No reinforcement learning and no laptop.** Records in `outputs/` (gitignored).
> Regenerate at the bottom.

## Headline

**Vault Stage 2's premise holds.** Inverse-model pretraining - self-supervised, no oracle
labels - raises the linear probe at every depth, and the trained encoder **beats the
raw-facelet linear ceiling**, which no amount of width could do.

| depth | chance | facelets | frozen | **trained** | trained - frozen | exact p |
|---|---|---|---|---|---|---|
| 3 | 0.181 | 0.906 | 0.614 | **0.908** | +0.294 | 0.0005 |
| 4 | 0.182 | 0.742 | 0.447 | **0.786** | **+0.340** | **0.0005** |
| 5 | 0.194 | 0.618 | 0.406 | **0.660** | +0.254 | 0.0005 |
| 6 | 0.203 | 0.488 | 0.344 | **0.575** | +0.232 | 0.0005 |

p 0.0005 is the **floor of the exact test** at n=12 (2/4096): every seed moved the same way at
every depth.

## Control B first - is the pipeline measuring what EXP-033 measured?

**Nothing below is readable without this.** The frozen arm is re-measured here rather than
taken from EXP-033, because this batches `SensoryCortex` where EXP-033 looped `brain.step`,
consuming the Poisson generator differently.

| | measured here | EXP-033 | tolerance |
|---|---|---|---|
| frozen @ depth 4 | **0.447** +-0.038 | 0.459 | +-0.10 PASS |
| facelets @ depth 4 | 0.742 | 0.766 | +-0.10 PASS |
| chance @ depth 4 | 0.182 | 0.182 | - |

Separately verified: batched features agree with `brain.step` **in distribution**, mean
per-unit difference falling 0.0202 -> 0.0054 as draws go 12 -> 240, i.e. as 1/sqrt(N).

## Claim 1 (PRIMARY) - did the encoder learn anything usable? CONFIRMED

Pre-registered: depth-4 trained - frozen **>= +0.05** at **p <= 0.05**.

Observed: **+0.3396**, **W-L 12-0**, **exact p 0.0005**. Nearly **seven times** the bar.

Per-seed, frozen -> trained at depth 4:

```
0.463->0.784  0.463->0.828  0.522->0.731  0.410->0.746  0.448->0.806  0.388->0.806
0.425->0.873  0.418->0.731  0.410->0.784  0.485->0.821  0.455->0.746  0.470->0.776
```

The bar was set at what one width doubling buys (0.459 -> 0.517 in EXP-033). Pretraining
delivers that **six times over at the same width**, which is the sharpest possible statement of
why EXP-033 was right to refute width and right to point at the encoder.

## Claim 2 (THE THESIS BAR) - does the encoder supply nonlinearity? CLEARED

Pre-registered: depth-4 trained > the **facelets arm measured here**, paired per seed.

| | value |
|---|---|
| trained | **0.786** +-0.043 |
| facelets (measured here) | 0.742 |
| margin | **+0.0442** |
| W-L | 9-3 |
| exact p | **0.0107** |

**A linear probe on the trained concept beats any linear map on the raw observation.** The
encoder is contributing genuine nonlinear structure. This is the first result in the cube line
where the spiking network is doing work rather than serving as a fixed random projection.

> [!warning] The honest size of this margin
> +0.0442 is not large, and it is of the same order as the 0.024 by which this pipeline's
> facelets arm sits below EXP-033's published 0.766 (a consequence of fitting jointly over
> depths 1-6 rather than 1-5). Against the **published** ceiling the margin would be +0.020.
>
> The paired test is still the right one - both arms are measured on identical splits per seed,
> so the shift affects their common level and not their difference - but "cleared" here means
> *cleared narrowly at depth 4*, not *comfortably*. **Claim 3 is where this gets decisive.**

## Claim 3 - the depth profile. THE PRE-REGISTERED READING IS "NOT MONOTONE"

Pre-registered: report depths 3-6; a gain that **grows with depth** is materially stronger than
a uniform shift.

Observed, trained - frozen: **+0.294 / +0.340 / +0.254 / +0.232** at depths 3/4/5/6.

**It peaks at depth 4 and declines. By the rule as written, NOT MONOTONE.** That is the
pre-registered verdict and it stands.

> [!important] A different comparison DOES grow with depth, and it was not pre-registered
> The rule above compares trained against **frozen**. The thesis-relevant quantity is trained
> against **facelets** - whether the encoder overtakes the observation - and that margin grows
> monotonically:
>
> | depth | trained | facelets | margin | W-L | exact p |
> |---|---|---|---|---|---|
> | 3 | 0.908 | 0.906 | +0.003 | 5-7 | 1.0000 |
> | 4 | 0.786 | 0.742 | +0.044 | 9-3 | 0.0107 |
> | 5 | 0.660 | 0.618 | +0.042 | **12-0** | **0.0005** |
> | 6 | 0.575 | 0.488 | **+0.087** | **12-0** | **0.0005** |
>
> At depth 3 the trained encoder merely **matches** the facelets (p 1.000, W-L 5-7): the raw
> observation is already 0.906 linearly decodable there, so there is nothing to add. At depths
> 5 and 6 it beats the ceiling on **every single seed**.
>
> **This is reported as an observation, NOT as a pre-registered claim, and it must not be
> substituted for the Claim 3 verdict above.** EXP-037 logged the mirror-image failure: a
> decision rule written for an expected outcome producing a true-but-misleading verdict. The
> discipline is the same in both directions - report the rule's answer, then report what else
> the data shows, labelled as such.

**Why the two readings differ, mechanically:** trained-minus-frozen is bounded by how much
headroom the frozen encoder leaves, and the frozen encoder degrades with depth too. The
trained-minus-facelets margin instead tracks **where linear separability of the observation
collapses** - which is precisely Wall 1. The encoder helps most where the observation helps
least.

Note also the variance collapse with depth: trained sd is 0.041 / 0.043 / 0.018 / **0.007**.
At depth 6 all twelve seeds land within 0.02 of each other.

## Claim 4 - the null. NOT TRIGGERED

Pre-registered: if Claim 1 refutes, report inverse dynamics as insufficient and redirect
Stage 2 to a value/heuristic objective.

**Claim 1 confirmed at p 0.0005, so this does not apply.** The pre-registered risk - that an
inverse model learns *what a move did* rather than *which move is good* - **did not
materialise**. Learning cube dynamics does transfer to optimal-move decodability, and it
transfers strongly.

That is the substantive scientific content here: it was a real possibility that the transfer
would fail, it was written down in advance as the hypothesis under test, and it held.

## Instrument check - was the objective actually learned?

Move-naming accuracy **0.454** (range 0.449-0.457 across seeds) against a **1/6 = 0.167** floor
and a pre-registered `>= 0.30`. The encoder genuinely solved the inverse task, so the probe is
describing a trained encoder rather than a random one under another name.

## Limitations

- **This measures what the representation SUPPORTS, not policy success.** EXP-033 Finding 2 is
  the standing caution: an oracle probe supported 48% at depth 3 while the actual RL policy
  managed 22%. **Whether a raised ceiling converts into a better policy is untested here** and
  is the Stage 2 follow-on. It needs the laptop.
- **Not saturated.** Move-naming accuracy was still rising at 40 epochs (0.447 -> 0.456 -> 0.455
  over the last ten). 40 epochs is a turnaround budget, not a converged optimum.
- **One objective.** Only the inverse model. A distance-regression objective would likely score
  higher - it is nearly the probe's own label - but trains on the oracle and weakens the
  "learns through experience" claim. Deliberately excluded.
- **One width (64) and one architecture.** Fixed to match the shipped policy.
- **The joint probe fit spans depths 1-6** where EXP-033 used 1-5, which lowers the facelets arm
  by 0.024 at depth 4 and by 0.050 at depth 3. Internal comparisons are unaffected (shared
  splits); cross-experiment absolute levels are.
- **Nothing past depth 6.** A random 2x2 scramble lives at depth 11.
- Depth 3's held-out side is 30 states; depths 4-6 are 133/200/200.

## Lead for the next experiment

1. **The Stage 2 follow-on is now clearly the next experiment: train a policy on the pretrained
   encoder.** EXP-039 says the representation supports far more than the frozen one did. Whether
   REINFORCE can extract it is a separate question with a separate failure mode, and EXP-033
   Finding 2 says the gap between the two can be large.
2. **Depth 5 and 6 are the interesting targets now**, not depth 4. That inverts the standing
   position: the encoder's advantage over the observation is largest exactly where the policy
   currently scores 0.0396 and 0.0000.
3. **Longer pretraining is cheap and untested.** The objective had not saturated.
4. A **distance-regression arm** would bound how much of the remaining gap is reachable at all,
   as a measured upper bound rather than a shipped design.

**Refuted and CLOSED, unchanged:** width (EXP-033), volume alone (EXP-034), curriculum stage
weighting (EXP-037), starvation at depth 6 (EXP-037).

## Regenerate

```bash
# 1 WORKER on the VPS, deliberately. Two workers drove available memory to 311 MB, started
# swapping, and pushed load to 3.77 on a 2-core box - the thrash profile, which makes the
# machine unresponsive WITHOUT tripping the OOM killer. ~3.1 h at 1 worker.
.venv/bin/python -u experiments/039_encoder_pretraining/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 1 --skip-existing

.venv/bin/python experiments/039_encoder_pretraining/aggregate.py
```
