# EXP-047 Results - fine-tuning works, and the mechanism does not survive its own control

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The
> interpretation contract was committed at `69bf1dc` **before any code existed that could
> produce a number**, and `aggregate.py` was written before dispatch. **No threshold was edited
> while filling this in.** One post-run change was made to `aggregate.py` and it moves no bar:
> `load()` assumed every `*.json` in the outputs directory was a record, which crashed on this
> experiment's `probe_*.json` (lists) and `selected_lr.json`. Same class of change as EXP-040's
> "print each margin in standard errors".
>
> **Provenance:** 18 records (3 rates x 2 pilot seeds, plus 12 confirmatory), 18 fine-tuned
> encoders, 18 head checkpoints. 2x2 cube, depth 6, 10,000 episodes, curriculum `(1..6)`,
> `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`, `normalize_advantages=False`, encoders
> initialised from EXP-040's `exp040_encoder_s*.pt`, `encoder_lr=1e-4` **selected mechanically**
> (see below). Laptop `SwizzlesDuo`. Pilot 2026-08-20 20:43 to 08-21 00:58 at 6 workers;
> confirmatory 08-21 01:00 to 12:57 at 10 workers; probes at 4 workers. Baseline is EXP-043's
> depth-6 cell and was **not** re-run. Zero tracebacks. Records in `outputs/` (gitignored); the
> `*_head.pt` and `*_encoder.pt` files ARE tracked. Regenerate at the bottom.

## Headline

**Fine-tuning the encoder during RL beats the frozen encoder at matched episodes, and the gain
is too large to be the extra compute.** It is the first lever this project has found that
changes the exchange rate instead of paying it.

| | mean | sd |
|---|---|---|
| frozen encoder, 10,000 episodes (EXP-043) | 0.1800 | 0.0985 |
| **fine-tuned encoder, 10,000 episodes** | **0.2700** | 0.0810 |

**+0.0900 paired, W-L-T 10-1-1, exact p 0.0020.**

For scale, EXP-046 needed **4.4x the episodes** to take depth 6 from 0.1800 to 0.3225.
Fine-tuning reaches 0.2700 at **1.33x** the per-step cost and the same episode count.

> [!warning] THE MECHANISM CLAIM DOES NOT SURVIVE ITS OWN CONTROL, AND THAT IS THE MORE
> INTERESTING HALF. The probe says the representation improved (+0.0398 at depth 4, 11-0,
> p 0.0010). The **leak-free slice says it did not** (+0.0050, 6-5, p 0.5732). The spec
> pre-committed to reporting the weaker, so: **the score improved and we cannot show the
> representation generalised.** See Claim 2.

## This is a different architecture, and it is reported as one

**Claim 3, printed first by `aggregate.py` deliberately, because it says what is being compared.**

| | frozen (every prior cube result) | EXP-047 |
|---|---|---|
| head | `Linear(64 -> 6)` = 390 | `Linear(64 -> 6)` = 390 |
| `sensory.fc1` / `fc2` | frozen | **trainable** (18,560 + 8,256) |
| **trainable total** | **390** | **27,206** (70x) |
| wall clock per step | 1.0x | **1.33x** (56.17 -> 74.87 ms, measured) |

Every record carries `trainable_params`; all 12 confirmatory cells report **27,206**.

**"The same 390 trainable parameters" is not true of this arm and never will be.** It is not a
cell of the depth series and must never be tabulated as one.

## How `encoder_lr` was chosen, and why the choice is trustworthy

`encoder_lr` had no prior. It was selected by a pilot under a rule fixed in spec section 5.2
before any pilot number existed, executed by `select_lr.py`, which **reads only the probe output
and cannot see a success rate**.

```
      lr   mean d4   mean delta     worst  gate   per-seed
   1e-03    0.7687      -0.0299   -0.0896  FAIL   s12:0.791->0.821  s13:0.806->0.716
   1e-04    0.8470      +0.0485   +0.0299  PASS   s12:0.791->0.821  s13:0.806->0.873
   1e-05    0.8060      +0.0075   -0.0075  PASS   s12:0.791->0.784  s13:0.806->0.828
SELECTED encoder_lr = 0.0001
```

Two design choices earned their place here:

1. **The gate requires BOTH seeds, not the mean.** At 1e-3 seed 12 *improved* by +0.030 while
   seed 13 fell by -0.090. A per-seed requirement kills it; a mean-only gate would have been a
   much closer call. The rate that damages half the seeds is exactly the rate not to carry to
   twelve.
2. **The pilot ran on seeds 12-13, disjoint from the confirmatory 0-11.** EXP-039 section 6a
   refused to pick its own lr by the probe because the probe was its outcome metric, and recorded
   that the probe would have chosen the other rate. EXP-047 inherits that trap one level up, so
   the defences are stacked: probe-only selection, on seeds that carry no claim. **Nothing that
   decides a claim was used to make this choice.**

**The pilot's success rates corroborate the choice without having informed it** (1e-3: 0.110,
0.090; 1e-4: 0.360, 0.140; 1e-5: 0.255, 0.130). That the probe and the score agree on the winner
is itself a small piece of mechanism evidence - and it is only worth anything *because* the
selection could not see the scores.

## Claim 1 (PRIMARY, PAIRED) - does fine-tuning beat the frozen encoder? CONFIRMED

Pre-registered: **>= +0.05** at **p <= 0.05**, paired against EXP-043 `exp043_capped_d6`, exact
permutation over all `2**12 = 4096` sign flips.

Observed: **+0.0900**, **W-L-T 10-1-1**, **exact p 0.0020**.

Per-seed deltas:

```
+0.170  -0.015  +0.225  +0.090  +0.085  +0.110  +0.115  +0.040  +0.120  +0.000  +0.060  +0.080
```

One seed regressed, by 0.015. One was exactly flat.

### The confound, priced rather than run

Fine-tuning costs **1.33x** per step, and matching episodes hands it that extra compute. EXP-046
measured depth 6's budget curve at **0.22 success per log10 of spend**, log-linear with no knee,
so:

> 1.33x compute is worth **0.22 x log10(1.33) = +0.027**.

**The observed +0.0900 is 3.3x that.** The result is not the extra compute wearing a disguise.

**What this still cannot rule out**, as pre-registered: the pricing assumes a budget curve
measured on a *frozen* encoder transfers to an arm that spends its extra compute on backward
passes rather than extra episodes, and that log-linearity holds down to 1.33x - inside the
measured 4.4x range but not separately verified there.

## Claim 2 (MECHANISM) - did the representation improve, or did the head fit itself to it? SPLIT

Pre-registered with an **asymmetry**: RL fine-tunes on the RL split while the probe holds out a
different one, so most states the standard probe scores were seen during fine-tuning.
**Degradation would be clean; improvement is confounded.**

| | delta | W-L | exact p |
|---|---|---|---|
| standard split, depth 4 | **+0.0398** | 11-0 | 0.0010 |
| **leak-free slice, depth 6** | **+0.0050** | 6-5 | **0.5732** |

They disagree, and **the spec pre-committed to reporting the weaker**. So the honest sentence is:
*the policy got better, and we cannot show the representation did.*

### The depth profile is why, and it points at memorisation

| probe depth | before | after | delta |
|---|---|---|---|
| 3 | 0.8833 | 0.9500 | **+0.0667** |
| 4 | 0.7998 | 0.8396 | +0.0398 |
| 5 | 0.6538 | 0.6850 | +0.0312 |
| **6 (the evaluated depth)** | 0.6839 | 0.6742 | **-0.0097** |

**The gain shrinks monotonically with depth and goes negative exactly where the policy is
scored.** Depth 3 has 90 states and the curriculum spends five of its six stages at depths 1-5,
so the encoder sees those shells thousands of times. That is the signature of the encoder
memorising shallow states, not of it learning a better code for optimality.

The leak-free slice - depth 6's 200 RL held-out states, which EXP-040 also excluded from
pretraining, so **neither stage ever saw them** - agrees: +0.0050 at p 0.5732.

**Had only the standard probe been run, this would read as a clean 11-0 mechanism story.** It is
the control that stops it. Same lesson as EXP-030, where a shuffle-null would have published a
false positive that a third arm refuted.

### So where does the +0.0900 come from?

Not from a representation that is better in general. The remaining explanation is that the
encoder and head **co-adapt on the training distribution**: the encoder becomes a better feature
extractor *for this policy* without becoming a better encoder of optimality. That is a real and
useful effect - the held-out score genuinely rose - but it is a narrower claim than "RL improved
the representation", and it predicts the gain would **not** transfer to a fresh head or a
different depth. **Neither of those was tested and both are cheap.** See "What to do next".

Generalisation gap is small, so this is not gross policy overfitting: train 0.3133, held-out
0.2700, gap **+0.0433**.

## Claim 3 - architecture accounting. Reported above.

## Claim 4 - collapse. Descriptive, no p-value.

A 70x larger trainable surface is a 70x larger surface to collapse, and stage 1's 2-step depth-1
cap now shapes the encoder as well as the head. **It did not collapse.**

| | EXP-047 | EXP-045's collapse signature |
|---|---|---|
| deepest-stage entropy | 0.3586 -> 0.3292 | 0.5914 -> 0.0979 |
| entropy min | 8.15e-05 | 2.70e-06 |
| train solve rate | 0.2871 | 0.0218 |
| seeds at exactly 0.0000 | **0/12** | - |

The frozen baseline had **1** seed at exactly 0.0000; the fine-tuned arm has **none**. Descriptive
only - n=12 cannot show a failure count went to zero.

## Claim 5 - the null was pre-committed, and did not occur

Claim 1 confirmed, so the pre-registered redirection (a different pretraining objective) is
**not** triggered by this result. It remains the right move if the transfer tests below fail.

## What this does and does not license

**Licensed:**
- "Fine-tuning the encoder during RL raises depth-6 success from 0.1800 to 0.2700 at matched
  episodes, p 0.0020, and the gain is 3.3x what the extra compute alone buys."
- "It is the first intervention that improves the exchange rate rather than paying it."

**NOT licensed:**
- "RL improved the encoder's representation." The leak-free control says otherwise at the
  evaluated depth.
- Any comparison against the depth series that does not say **27,206 trainable parameters**.
- Extrapolating to depth 7+. Untested.

## What to do next

1. **Test the co-adaptation hypothesis, cheaply.** Freeze a fine-tuned encoder and train a
   *fresh* head on it. If the gain survives, the encoder really is better and the leak-free probe
   is simply the wrong instrument. If it vanishes, co-adaptation is confirmed. The encoders and
   heads are all serialised, so this costs one RL run per seed and no pretraining.
2. **Fine-tune only at the evaluated depth.** The depth profile suggests the shallow curriculum
   stages are where the memorisation happens. Restricting `encoder_lr` to the final stage is a
   one-line config change against a strong prior.
3. **Depth 7 at 10,000 episodes**, where the frozen arm scores 0.0621. If fine-tuning holds its
   +0.09 there, the break point moves again.

## Regenerate

```bash
# encoders for seeds 12-23 (prerequisite; ~1.7 h at 10 workers)
.venv/bin/python -u experiments/047_encoder_finetuning/pretrain_seeds.py \
    --seeds 12 13 14 15 16 17 18 19 20 21 22 23 --workers 10
# pilot, probe, mechanical selection
.venv/bin/python -u experiments/047_encoder_finetuning/run.py --mode pilot --workers 6
.venv/bin/python -u experiments/047_encoder_finetuning/probe_encoders.py --mode pilot
.venv/bin/python -u experiments/047_encoder_finetuning/select_lr.py
# confirmatory arm, probe, verdicts
.venv/bin/python -u experiments/047_encoder_finetuning/run.py --mode confirm --workers 10
.venv/bin/python -u experiments/047_encoder_finetuning/probe_encoders.py --mode confirm
.venv/bin/python experiments/047_encoder_finetuning/aggregate.py
```

Seeded runs are byte-identical across worker scheduling, so a re-run should reproduce every
number above exactly.
