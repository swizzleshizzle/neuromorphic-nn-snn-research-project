# EXP-048 design - is the fine-tuned encoder better, or just co-adapted to its head?

> **PRE-REGISTERED. Committed before any number exists.** Every threshold below is fixed at
> commit time. A later change must be a separate commit, dated, with its reason, and made before
> the data it applies to exists.

## 1. The question, and why EXP-047 forces it

EXP-047 confirmed its primary claim: fine-tuning the sensory encoder during RL took depth 6 from
**0.1800 to 0.2700** at matched episodes (paired +0.0900, p 0.0020), a gain 3.3x larger than the
1.33x compute overhead can explain.

**But its mechanism claim split, and the honest reading was the weaker one.** The standard probe
said the representation improved (+0.0398 at depth 4, 11-0, p 0.0010); the leak-free slice said
it did not (+0.0050, p 0.5732). The probe's depth profile pointed at memorisation: the gain
shrank monotonically with depth and went **negative at depth 6**, the depth the policy is scored
at.

EXP-047's RESULTS.md named the remaining explanation and the test for it:

> The encoder and head **co-adapt on the training distribution**: the encoder becomes a better
> feature extractor *for this policy* without becoming a better encoder of optimality... it
> predicts the gain would **not** transfer to a fresh head.

**This experiment runs that test.** It is the cheapest decisive thing available and it needs no
new science.

## 2. What is being changed, precisely

**Nothing in the trainer.** Verified 2026-08-23: EXP-047's serialised encoders load through the
existing `encoder_state_path` seam (built for EXP-040) with `encoder_lr=None`, i.e. **frozen**.
State-dict keys match, weights load exactly, and the loaded encoder differs from its EXP-040
starting point by 6.49e-02 max on `fc1.weight`. This is a driver-only experiment.

Three arms, **two of which already exist and are not re-run**:

| arm | encoder | during RL | head | mean | source |
|---|---|---|---|---|---|
| **A** | EXP-040 pretrained | frozen | fresh | **0.1800** | EXP-043, exists |
| **B** | **EXP-047 fine-tuned** | **frozen** | **fresh** | **?** | **THIS EXPERIMENT** |
| **C** | EXP-047 fine-tuned | trained jointly | co-adapted | **0.2700** | EXP-047, exists |

**One variable between A and B: which frozen encoder the fresh head reads.** Same 12 seeds, same
curriculum `(1..6)`, same `max_steps_by_depth=((1,2),)`, same 10,000 episodes, same
`entropy_beta=0.0`, `normalize_advantages=False`, same split seeds.

**Arm B costs 1.0x per step, not 1.33x** - it is frozen. So unlike EXP-047 there is no compute
confound to price: B and A are compute-identical.

### The split is clean at both stages

Seed `s`'s fine-tuned encoder came from an RL run on seed `s`, which held out seed `s`'s
evaluation states. EXP-040's pretraining excluded the same states. So arm B is evaluated on
states **neither** stage ever trained on, exactly as A and C were.

## 3. Claims

All comparisons **paired per seed**, exact permutation over all `2**12 = 4096` sign flips.

### Claim 1 - PRIMARY. Does the fine-tuned encoder beat the original one with a fresh head?

**B minus A.** CONFIRMED at **>= +0.05** with **p <= 0.05** - the same bar EXP-043, EXP-045,
EXP-046 and EXP-047 all used.

This is the question "did the encoder itself get better", stripped of the head that was trained
alongside it.

### Claim 2 - Is B below C?

**B minus C.** Reported with its exact p. A significantly negative delta means part of EXP-047's
gain lived in the head-encoder pairing rather than in the encoder.

> [!warning] A NULL HERE IS NOT EVIDENCE OF EQUIVALENCE, and is pre-committed not to be
> described as one. n=12 cannot establish that B and C are the same. "B is not detectably worse
> than C" is the strongest sentence a null licenses.

### Claim 3 - Retention. Descriptive, no p-value.

```
retention = (B - A) / (C - A) = (B - 0.1800) / 0.0900
```

Reported as a single fraction with its per-seed spread. **1.0** means the encoder carries all of
EXP-047's gain; **0.0** means none of it and the whole effect was co-adaptation.

### Claim 4 - THE INTERPRETATION GRID, fixed here so it cannot be chosen afterwards

| Claim 1 | Claim 2 | reading |
|---|---|---|
| CONFIRMED | not significant | **The encoder genuinely improved.** EXP-047's leak-free probe was the wrong instrument, or too insensitive at n=12. The probe's depth profile still needs explaining. |
| CONFIRMED | significantly negative | **Both effects are real and partial.** The encoder improved AND the head co-adapted. Report retention as the split. |
| REFUTED | significantly negative | **Co-adaptation.** EXP-047's gain was in the pairing, not the encoder. This corroborates the leak-free probe and the memorisation reading. |
| REFUTED | not significant | **Incoherent - do not interpret.** B would be indistinguishable from both a 0.1800 arm and a 0.2700 arm, which means the instrument lacks the power to separate them. Report as underpowered and say so. |

### Claim 5 - the null is a real result

A refuted Claim 1 is the **most informative** outcome available here. It would mean the +0.0900
of EXP-047 is not a better representation at all, and it would align three independent
measurements: the leak-free probe (+0.0050, p 0.5732), the probe's negative depth-6 delta
(-0.0097), and this. EXP-047's headline would then need restating as *"joint training of encoder
and head beats training the head alone"* rather than *"the encoder improves"* - a narrower and
more accurate claim, and one that changes what to try next.

## 4. What this deliberately does NOT control for

**"Is it RL's objective, or just more gradient steps of any kind?"** Arm B's encoder has received
10,000 episodes of additional shaping that arm A's has not. A control would continue EXP-039's
inverse-model pretraining for an "equivalent" amount and then freeze it.

That arm is **not run here**, for a reason worth stating: there is no honest exchange rate
between RL episodes and pretraining epochs, so "equivalent" would be a number I chose. Inventing
it would make the control look rigorous while smuggling in a free parameter. EXP-040's RESULTS.md
already lists longer pretraining as an open item; it deserves its own pre-registration with its
own justification for the budget, not a hand-wave inside this one.

**Consequence:** a CONFIRMED Claim 1 licenses *"the fine-tuned encoder is better"*, **not**
*"the RL objective is what made it better"*.

## 5. Execution

12 cells, depth 6, 10,000 episodes, frozen. Arms A and C are not re-run.

**Six workers, not ten.** 12 cells on 10 workers is two waves with the second running 8 workers
idle; 12 on 6 is two clean waves, and 6 workers measured **0.115 s/step** against 10 workers'
~0.16 on this machine. Applying the estimator corrected on 2026-08-22:

```
ceil(12/6) * 100,962 steps * 0.115 s = 2 * 11,611 s = 6.4 h
```

against 8.8 h at 10 workers. Faster and a cleaner packing, which is the whole point of that
correction.
