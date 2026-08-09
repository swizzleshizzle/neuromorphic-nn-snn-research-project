# EXP-040 design - does a raised representational ceiling become a better policy?

**Status: pre-registration. Written 2026-08-09, before any EXP-040 number exists.**
Every threshold below is fixed. If one is edited after data arrives, that edit is the finding
and must be reported as such.

Vault **Stage 2, second increment**. EXP-039 built the encoder; this asks whether the policy can
use it.

## 1. The question

EXP-039 established that inverse-model pretraining transforms what the representation supports:

| depth | frozen concept | **pretrained concept** | facelets (linear ceiling) |
|---|---|---|---|
| 4 | 0.447 | **0.786** | 0.742 |
| 5 | 0.406 | **0.660** | 0.618 |
| 6 | 0.344 | **0.575** | 0.488 |

All at p 0.0005, 12-0. The trained encoder beats the raw-facelet linear ceiling at depths 4, 5
and 6 - something width provably cannot do (concept@512 reached only 0.638 at depth 4).

**But a probe is not a policy.** EXP-033 Finding 2 is the standing warning, measured in this
repo: at depth 3 an oracle probe on the frozen concept supported **0.48** success while the
actual REINFORCE policy managed **0.22**. Less than half of what the representation supported
was extracted.

**So: does the raised ceiling convert?** That is the entire question, and it is genuinely open
in both directions.

Meanwhile EXP-038 closed the other side of the ledger. Trainer stabilizers, curriculum
weighting, width, volume and starvation are all refuted. **The encoder is the only live lever
left**, which is why this experiment matters more than its size suggests.

## 2. The intervention - exactly one variable

For each seed: pretrain a `SensoryCortex` by inverse model (EXP-039's procedure), **freeze it**,
and run EXP-036's cube baseline unchanged on top.

> **The encoder is FROZEN during RL, not fine-tuned.** That keeps the trainable parameter count
> at the same **390** (`Linear(64 -> 6)`) as every cube experiment since EXP-029, so the only
> difference from EXP-036 is *which weights the frozen encoder holds*. Fine-tuning end-to-end
> would confound "a better representation" with "the encoder kept learning during RL", and
> those need separating before they are combined.

Everything else is EXP-036's configuration verbatim: `arm="regionalized"`,
`readout="concept"`, `sigma=0.0`, 10,000 episodes, curriculum `(1..d)` at equal stage weights
(EXP-037 found equal at or near optimal), `entropy_beta=0.0`, `normalize_advantages=False`
(EXP-038 closed those), evaluated on the held-out shell at `cfg.depth`.

**Depths 4, 5, 6 x 12 seeds = 36 runs.** All comparators are EXP-036 cells at the same seeds on
the same machine and are **not re-run**: depth 4 = 0.1591, depth 5 = 0.0396, depth 6 = 0.0000.

## 3. Three controls, and the first one is not optional

### 3a. Pretraining must exclude the RL held-out states

The policy is evaluated on `split_shell`'s held-out side. If the encoder was pretrained on those
states, the policy is being evaluated on states its representation was fitted to.

> **The exclusion set is the UNION of `split_shell` held-out states at depths 4, 5 and 6 for
> that seed** - because one encoder per seed serves all three depths - and a pair is dropped if
> **either endpoint** is in it, since `s'` passes through the same encoder as `s`.

**This is a different split from EXP-039's.** EXP-039 excluded the *probe's* `split_states`
partition; the RL split comes from `split_shell` keyed on `split_seed`. **EXP-039's encoders are
therefore NOT reusable here even if they had been serialised** - they were fitted while excluding
the wrong set. Re-pretraining is required by the design, not by an oversight.

### 3b. `cube_baseline.py` gains an encoder seam, and it must be provably neutral

`run_cube_baseline` builds its own agent through `make_agent`, so injecting a pretrained encoder
requires a config field. Added: `encoder_state_path: str | None = None`, loaded into
`brain.sensory` when set.

> **`None` must reproduce existing behaviour byte-for-byte.** A baseline is captured **before**
> the change (records at depth 3 plus the `make_agent` weight tensors at a fixed seed) and must
> reproduce **exactly** afterwards. EXP-036 set this precedent when it added head serialisation;
> the check is what lets every prior cube number stay comparable.

### 3c. The laptop's encoders must match the VPS procedure

EXP-039 pretrained on the VPS; this pretrains on the laptop, and **seeded runs are not
reproducible across platforms** (CLAUDE.md). So the encoders will not be byte-identical.

Sanity check, reported not gated: **move-naming accuracy >= 0.30** (EXP-039 measured 0.454,
range 0.449-0.457, against a 1/6 floor). If the laptop's pretraining lands far outside that
band, the procedure did not transfer and nothing below should be read.

## 4. The pre-registered contract

All comparisons **paired per seed** against EXP-036, exact permutation over all 2^12 = 4096 sign
flips, two-sided.

### Claim 1 (PRIMARY) - does it convert at depth 4?

Depth 4 is the **powered** arm: the only depth where the policy currently works (0.1591 at 51x
its floor, no gap, no collapse), so it is where an improvement is measurable at all. Depths 5
and 6 are resolution-bound at 1/200 per seed, which EXP-038 just demonstrated is enough to make
a null uninterpretable.

> **CONFIRMED**: mean delta **>= +0.05** with **p <= 0.05**.
> Otherwise **REFUTED**: a raised representational ceiling does not convert into policy success
> at the frontier depth.

+0.05 is **0.68 sd** of EXP-036's measured depth-4 spread (0.0739) and a **31% relative** gain
on 0.1591. It is deliberately harder than EXP-037's 0.03 bar, because EXP-039's probe effect was
enormous (+0.34, a 76% relative rise) and a marginal policy gain would not be the result the
probe predicts.

### Claim 2 - does the break point move?

EXP-036's rule for "working": **>= 2x the measured floor AND >= 0.10 absolute**. Floors are
0.0000 at depth 5 and 0.0008 at depth 6, so the binding bar is **0.10** at both.

> **BREAK POINT MOVES** if depth 5 reaches 0.10. That would be the single most consequential
> result of the cube line to date: the curriculum has broken at depth 5 since EXP-036 and
> nothing has moved it.

Depth 6 is reported but not expected to clear; it is where the encoder's *probe* advantage is
largest (+0.087) and the policy's starting point is worst (0.0000).

### Claim 3 - mechanism

Report `greedy_modal_action_frac` and `mean_train_entropy` per depth against EXP-036's
(0.685 / 0.779 / 0.975 modal at depths 4/5/6).

**A gain that arrives with modal fraction FALLING is a different and more trustworthy mechanism
than one that does not.** EXP-038 showed collapse is a symptom, so if a better representation
makes the task learnable, reduced collapse should follow the gain rather than cause it. Read
modal **with** entropy, never entropy alone - EXP-035, EXP-037 and EXP-038 each produced a
different relationship between the two.

### Claim 4 - the null is pre-committed, and it is informative

> If Claim 1 refutes while EXP-039's probe result stands, the finding is: **the representation
> was never the binding constraint on the policy; the readout or the learning signal is.** That
> is EXP-033 Finding 2 writ large, it would be the strongest evidence yet for **Stage 3** (a
> value function carried on the idle `neuromod` pathway), and it must be reported as a positive
> redirection rather than as a disappointing null.

Written down in advance so that a null cannot later be described as "inconclusive, needs a
bigger encoder".

## 5. Cost

| depth | steps/run | runs | steps |
|---|---|---|---|
| 4 | 80,000 | 12 | 0.96M |
| 5 | 90,000 | 12 | 1.08M |
| 6 | 100,020 | 12 | 1.20M |
| **total** | | **36** | **3.24M** |

At EXP-038's measured **46 steps/s** at 10 workers: **about 20 h**. Pretraining 12 encoders adds
roughly 15 minutes and is negligible.

Ten workers, not sixteen: measured 74.2% utilisation against 43.1%; the laptop is memory-bound
at ~920 MB private per worker.

## 6. Implementation constraints

- **Reuse `run_cube_baseline` unmodified apart from the encoder seam.** Any second change to the
  trainer invalidates the neutrality argument in 3b.
- **`cell_tag` must encode the arm**, since `record_filename` covers tag/arm/depth/seed/sigma and
  not `encoder_state_path`.
- **Serialise the pretrained encoder** next to the records. EXP-039 did not, and its 3.1 h of
  compute produced no reusable artifact. `*_encoder.pt` alongside `*_head.pt`.
- **No assertion that cannot fail.** Every test added must fail against the pre-change code.

## 7. What this experiment cannot say

- **Nothing about fine-tuning the encoder during RL.** Frozen by design; that is the next
  increment if this succeeds.
- **Nothing about objectives other than the inverse model**, and EXP-039 already notes a
  distance-regression arm would likely probe higher while training on the oracle.
- **Nothing about depths past 6**, and a random 2x2 scramble lives at depth 11.
- **Nothing at budgets other than 10,000 episodes.** EXP-035 had depth 3 still climbing between
  10k and 30k, so a refuted Claim 1 does not mean "never".
- **Nothing about width.** Fixed at 64 throughout.
