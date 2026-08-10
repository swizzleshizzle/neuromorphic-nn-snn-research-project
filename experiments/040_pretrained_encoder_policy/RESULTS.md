# EXP-040 Results - the ceiling converts, and the break point moves

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The
> interpretation contract was committed at `bb2b3e5` **before the data existed**. No threshold
> was edited while filling this in; the only change to `aggregate.py` after the run was to
> **print each margin in standard errors**, which adds precision to the report and moves no bar.
>
> **Provenance:** 36 records (3 depths x 12 seeds), 36 head checkpoints, 12 pretrained encoders.
> 2x2 cube, concept readout, `Linear(64 -> 6)` head, 10,000 episodes, curriculum `(1..d)` at
> equal stage weights, `entropy_beta=0.0`, `normalize_advantages=False`. Phase 1 (pretraining)
> 2026-08-09 14:40 to 16:20; phase 2 (policy) 17:54 to 2026-08-10 02:09. Laptop `SwizzlesDuo`,
> `--workers 10`, exit 0, **zero tracebacks**. All comparators are EXP-036 cells at the same
> seeds on the same machine and were not re-run. Records in `outputs/` (gitignored); the
> `*_head.pt` and `*_encoder_*.pt` files ARE tracked. Regenerate at the bottom.

## Headline

**The raised representational ceiling converts into policy success, at every depth tested.**

| depth | EXP-036 | **EXP-040** | delta | exact p | EXP-039 probe |
|---|---|---|---|---|---|
| 4 | 0.1591 | **0.3471** | **+0.1880** | 0.0337 | 0.447 -> 0.786 |
| 5 | 0.0396 | **0.2304** | **+0.1908** | 0.0020 | 0.406 -> 0.660 |
| 6 | 0.0000 | **0.1037** | **+0.1037** | 0.0039 | 0.344 -> 0.575 |

Exactly one thing changed against EXP-036: **which weights the frozen encoder holds.** The head
is still `Linear(64 -> 6)`, **390 trainable parameters**, and the encoder is frozen during RL in
both arms. Depth 4 more than doubled. **Depth 5, broken since EXP-036, is now working. Depth 6,
which was 0.0000 on all twelve seeds, is off the floor.**

## Claim 1 (PRIMARY) - does it convert at depth 4? CONFIRMED

Pre-registered: **>= +0.05** at **p <= 0.05**, paired against EXP-036's 0.1591.

| | value |
|---|---|
| delta | **+0.1880** |
| W-L-T | **10-2-0** |
| exact p | **0.0337** |
| pretrained | 0.3471 +-0.224 |

**The answer to EXP-033 Finding 2 is yes.** That finding - an oracle probe supporting 0.48 at
depth 3 while REINFORCE extracted 0.22 - was the reason this experiment was not a formality.
REINFORCE *can* use a better representation when it is given one.

## Claim 2 - does the break point move? YES AT DEPTH 5. AT DEPTH 6, READ THE MARGIN

EXP-036's "working" rule: **>= 2x the measured floor AND >= 0.10 absolute**. Floors are 0.0000
(d5) and 0.0008 (d6), so the binding bar is **0.10** at both.

| depth | mean | SE | vs bar | seeds above bar | verdict |
|---|---|---|---|---|---|
| 5 | 0.2304 | 0.0459 | **+2.84 SE** | 10/12 | **WORKING** |
| 6 | 0.1037 | 0.0348 | **+0.11 SE** | 5/12 | at the bar |

**Depth 5 is a real move.** It has been BROKEN since EXP-036 measured it, and EXP-037 and
EXP-038 both failed to shift it. It now clears the pre-registered bar by nearly three standard
errors with 10 of 12 seeds individually above it.

> [!warning] Depth 6 clears the rule by 0.0037 against a standard error of 0.0348
> The pre-registered rule fires - 0.1037 >= 0.10 is true - and reporting that as "depth 6 is
> working" would be **true-but-misleading**, which is the exact failure EXP-037 logged when its
> aggregator printed "INTERIOR OPTIMUM" on a result that did not support one.
>
> **The honest statement is: depth 6 moved decisively OFF THE FLOOR (0.0000 -> 0.1037 at
> p 0.0039, 9 of 12 seeds above zero where previously none were) and sits AT the working bar,
> not above it.** Only 5 of 12 seeds individually clear 0.10. Whether depth 6 works needs a
> larger n or a better arm, and this experiment does not settle it.
>
> `aggregate.py` now prints the margin in standard errors and refuses to let a sub-1-SE pass go
> unremarked. The rule itself was not changed.

## Claim 3 - mechanism. The gain arrives WITH collapse falling, which is the trustworthy direction

| depth | modal EXP-036 | modal EXP-040 | entropy EXP-040 | delta success |
|---|---|---|---|---|
| 4 | 0.685 | **0.623** | 0.329 | +0.1880 |
| 5 | 0.779 | **0.645** | 0.364 | +0.1908 |
| 6 | 0.975 | **0.675** | 0.370 | +0.1037 |

Modal action fraction falls at **every** depth while success rises at every depth, and the
largest fall is at depth 6 (0.975 -> 0.675), where the policy was most degenerate.

**This is the mechanism EXP-038 predicted would be the trustworthy one.** EXP-038 showed that
*forcing* de-collapse via an entropy bonus bought nothing - collapse is a symptom. Here collapse
falls as a **consequence** of the task becoming learnable, and the success gain comes with it.
Same instrument, opposite causal direction, opposite outcome.

Note entropy also rose (0.204-0.236 -> 0.329-0.370). Entropy and modal fraction moved in
opposite directions here, which is a **fourth** distinct pattern after EXP-035's, EXP-037's and
EXP-038's. **Read both, always. No single relationship between them is the general rule.**

## Claim 4 - the null. NOT TRIGGERED

Pre-registered: if Claim 1 refuted while EXP-039's probe result stood, the finding would be that
the representation was never the binding constraint and Stage 3 was next.

**Claim 1 confirmed, so this does not apply.** The representation *was* a binding constraint.

## The finding that is not in any pre-registered claim: variance tripled

| depth | EXP-036 sd | EXP-040 sd | EXP-040 seeds at exactly 0.000 |
|---|---|---|---|
| 4 | 0.074 | **0.224** | **2/12** (EXP-036 had none) |
| 5 | 0.027 | **0.159** | 2/12 |
| 6 | 0.000 | **0.121** | 3/12 |

Depth-4 per-seed: `0.188 0.526 0.000 0.256 0.000 0.286 0.188 0.519 0.669 0.489 0.556 0.489`.

**Seeds 2 and 4 fail completely at depths 4 and 5**, scoring 0.000 where EXP-036's worst seed
scored 0.015. Their pretraining looked entirely normal - move-naming accuracy 0.434 and 0.433,
squarely inside the 0.430-0.437 band every seed produced.

So the intervention is **powerful but unreliable**: it roughly triples the mean while also
producing catastrophic failures that the baseline never produced, and **the pretraining metric
does not predict which seeds will fail.** That is a concrete, actionable target for the next
experiment, and it is not something either the probe (EXP-039) or the mean could have revealed.

## Limitations

- **The encoder is frozen during RL.** Fine-tuning end-to-end is untested and is the obvious
  next increment now that the frozen version has worked.
- **Depth 6 is at the bar, not above it** (see Claim 2). Do not cite it as "working".
- **One pretraining objective** (inverse model) and one budget (40 epochs, not saturated -
  EXP-039 found move-naming accuracy still rising).
- **One width (64) and one policy budget (10,000 episodes).** EXP-035 had depth 3 still climbing
  between 10k and 30k, so these are not ceilings.
- **Nothing past depth 6.** A random 2x2 scramble lives at **depth 11**, and the states at
  depths 7-9 are 14% of the cube against depth 1-6's 0.32%.
- **12 seeds with sd up to 0.224** means the depth-4 mean carries an SE of 0.065. The
  direction is solid; the magnitude is not precise.

## Lead for the next experiment

1. **Diagnose the failing seeds.** 2 of 12 collapse to 0.000 with normal-looking pretraining.
   Understanding that is worth more than another depth, because it is the difference between a
   0.35 mean and a ~0.42 mean at depth 4, and because an unreliable lever is hard to build on.
2. **Fine-tune the encoder during RL.** The frozen version works; the vault's Stage 2 explicitly
   contemplates end-to-end training as the stronger form.
3. **Longer pretraining**, which is cheap and whose objective had not saturated.
4. **Push to depth 7+.** The break point has moved for the first time; find where it now sits.
5. **Stage 3 is NOT cancelled but is no longer the forced next move.** EXP-038 closed
   credit-assignment as a *stabilizer* question, and this result says representation was a real
   constraint - but sparse reward at depth 11 is still Wall 2, and `neuromod` is still idle.

**Still refuted and CLOSED:** width (EXP-033), volume alone (EXP-034), curriculum stage weighting
(EXP-037), starvation at depth 6 (EXP-037), trainer stabilizers (EXP-038).

## Regenerate

```bash
# Phase 1 pretrains 12 encoders (~1.6 h at 10 workers), phase 2 runs 36 policies (~8 h).
# --skip-existing resumes both phases.
.venv/bin/python -u experiments/040_pretrained_encoder_policy/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10 --skip-existing

# EXP-036's records MUST be present: they carry all three comparators and every claim is
# paired against them. They are gitignored, so re-fetch from the laptop if outputs/ is empty.
.venv/bin/python experiments/040_pretrained_encoder_policy/aggregate.py
```
