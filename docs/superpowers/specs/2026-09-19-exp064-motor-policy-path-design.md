# EXP-064 design - put the brain's OWN pathway on the policy path, and train it

> **PRE-REGISTERED. Committed before any EXP-064 number exists.** Thresholds fixed at commit time.
> **Date:** 2026-09-19 · **Phase:** 3 (post-checkpoint) · **Grounds:** the Phase 3 assessment.

## 0. The question the whole project has never asked

`Brain.step` computes `utilities = pfc(concept)` -> `router` -> `motor` -> an action on **every
step of every experiment ever run**. That action is consumed **only** by `monitor/runner.py`, to
draw the dashboard. Training and evaluation both ignore it and use `head(concept_rate(out))`.

The consequence, measured:

| region | parameters | ever reached by an optimizer? |
|---|---|---|
| sensory | 26,816 | only when `encoder_lr` is set (EXP-047 onward) |
| hippocampus | 19,414 | **never** |
| prefrontal | 15,456 | **never** |
| motor | 42 | **never** |
| router | 0 | (holds no parameters) |

**34,912 of 61,728 brain parameters, 56.6%, are frozen at random initialisation forever**, and
**318 of 510 neurons are off the policy path** whenever `recall=False`. This is v1 by design
(ADR-0001), not an accident. But it means the capstone's central claim, *a regionalized spiking
brain that learns to solve a cube*, rests on a topology that **has never been on the policy path**.

The Phase 3 assessment established that no arm-versus-arm contrast can answer this without an
architecture change (`docs/phase3-honest-assessment.md` section 2a and its correction). **This is
that change.**

## 1. The arms

12 seeds each, depth 5, EXP-059/061/063's configuration field for field, frozen EXP-040 E0
encoder. **Nothing is reused from a prior experiment: both arms are new and paired within seed.**

| arm | readout | trains | trainable | tag |
|---|---|---|---|---|
| **P** | `motor` | prefrontal + motor + a 6x6 head | **15,540** | `exp064_motor_d5` |
| **C** | `concept` | one MLP head, `hidden=218`, brain frozen | **15,484** | `exp064_mlp_d5` |

**P reads the motor region's spike rates**, so REINFORCE's gradient flows back through
motor -> router -> prefrontal via snnTorch's surrogate gradients. The head is a **6x6 affine**: a
learned temperature and bias on the brain's own decision, not a second policy on top of it.

**C is capacity-matched to 0.36%** (15,484 against 15,540). That is the point. EXP-063 shipped a
confound by matching a control to its arm and neither to the baseline; here the control exists
purely to hold trainable capacity fixed while the **route** changes. The question is exactly:
**does a spiking prefrontal-to-motor pathway learn a policy as well as a conventional MLP of the
same size reading the same features?**

**The hippocampus is OFF in both arms.** Memory hurts on this task (EXP-059/061/063), so
`use_memory` is now keyed to the memory modes explicitly rather than to `readout != "concept"`,
which would have engaged it for P and tested two changes at once.
`test_the_hippocampus_is_OFF_for_the_motor_arm` asserts the outcome (`mean_n_stored == 0`) rather
than the switch. **Mutation V3 fails that test**, and without it the revert was a live blind spot.

## 2. Claims

**Test:** paired over the 12 shared seeds, **exact** sign-flip permutation over all
`2**12 = 4,096` flips. Approximate 95% intervals from the paired `t` at `df = 11` (2.201).

### Claim 1, PRIMARY - `P` minus `C` on held-out success. NOT directional.

Both directions are interesting and the spec refuses to privilege one:

- **CONFIRMED (topology helps)** at `>= +0.05` and `p <= 0.05`.
- **REFUTED (topology costs)** at `<= -0.05` and `p <= 0.05`.
- **NULL otherwise.** ==A null means the brain's own pathway is **neither better nor measurably
  worse** than a conventional MLP of the same size.== For this project that is a genuinely
  notable outcome and it is **still a BOUND at n=12, not proof of equivalence.** Written here so
  it cannot be upgraded afterwards into "the topology works".

### Claim 2, VALIDITY GATE 1 - the regions actually trained

`region_drift`, the relative parameter movement `||theta_end - theta_init|| / ||theta_init||`
over prefrontal and motor, **arm P mean `>= 0.01`**.

A ratio, so scale-free across differently sized tensors. **A frozen pathway reads exactly 0.0**,
so the gate can fail; calibration measured **0.598** at `region_lr=1e-2`, so it can pass with
**60x margin**. This is the EXP-047 instrument: that experiment reported a 70x trainable surface
while the parameter moved by exactly 0.0, so counting parameters in an optimizer is not evidence.

### Claim 3, VALIDITY GATE 2 - the spiking pathway actually fires

`motor_rate_mean`, mean firing rate over the action units, **arm P mean `>= 0.02`**.

**A silent motor region hands the head a constant zero vector**, and the arm degenerates into a
bias-only policy that still trains and still reports an ordinary success rate. That is this
design's specific vacuity risk and it is not hypothetical: at **depth 1** the pathway was silent
on **67% of steps**.

**Calibrated in the regime it will run in**, which is what made the difference: at depth 5 with
the real frozen E0 encoder, seeds 0-2, the rate measured **0.105 to 0.150** and `silent_frac`
fell to **0.000 to 0.011**. The floor sits at **5x** the worst calibration reading and well above
the 0.0 a dead pathway gives.

### Multiplicity

One substantive contrast, read at `p <= 0.05`. Claims 2 and 3 are conditions, not contrasts.

## 3. Cost, priced from measurement

**The two arms do NOT cost the same per step, and pricing them with one figure is the mistake
EXP-063 made** (its estimate came in 24% high because the attention arms skipped the attractor
unroll the memory arms ran).

Measured on this VPS at depth 5 with the real encoder, 520 steps each:

| arm | ms/step | ratio to the memory arm | laptop h/cell |
|---|---|---|---|
| P `motor` (grad through the spiking unroll) | **104.69** | **1.444x** | **~4.40** |
| C `concept` + MLP head (brain frozen) | **57.66** | 0.795x | **~2.43** |
| reference: `memory` arm | 72.51 | 1.000x | 3.05 (EXP-059/061, measured) |

Scaled by **ratio against an arm whose laptop cost is known** rather than by a cross-machine
guess. **24 cells = ~82 CPU-h = ~13.7 h wall at 6 workers.**

## 4. What this cannot answer

- **Whether a different `region_lr` would change the verdict.** It is set to **1e-2, equal to the
  head's `lr`**, so the policy is optimized as one object. `1e-3` was also calibrated (drift
  0.093) and **not swept**. A null is a bound **at this learning rate**, and saying otherwise
  afterwards would be the optional stopping this practice exists to prevent.
- **Whether the hippocampus or the router would help.** Both stay out: memory is closed negative,
  and the router holds no parameters.
- **Anything at depth 7 or 8.** This is depth 5, matched to the recent well-characterized regime.
- **Whether P would win with the v1 recipe's tuning.** The curriculum, encoder pretraining and
  step caps were all tuned around a concept-reading linear head. P inherits them unchanged, which
  is the fair comparison available and is **not** a neutral starting point for it.
