# EXP-050 design - was it RL's objective, or just more gradient?

> **PRE-REGISTERED. Committed before any number exists.** Every threshold below is fixed at
> commit time. A later change must be a separate commit, dated, with its reason, and made before
> the data it applies to exists.

## 1. The control this line has deferred twice

EXP-047 fine-tuned the encoder during RL; EXP-048 proved the improvement is in the **encoder**
(freeze it, train a fresh head, +0.1312 at p 0.0059); EXP-049 priced it at a **constant ~+0.05
per round** over that round's own compute.

Every one of those specs names the same uncontrolled alternative, and both times it was deferred
for the same honest reason - there is no obvious exchange rate between RL episodes and
pretraining epochs, so any "matched" control risked smuggling in a free parameter:

> **Arm B's encoder had 10,000 episodes of extra shaping that arm A's did not. This cannot say
> whether RL's OBJECTIVE caused the improvement or merely more gradient steps of any kind.**

**It is now settleable, because the natural match turns out not to need a free parameter.**

## 2. The match, and why it is not arbitrary

REINFORCE calls `optimizer.step()` **once per episode**, so EXP-047's 10,000 episodes are
**10,000 encoder updates**. EXP-040's inverse-model pretraining runs 66,192 pairs at batch 256 for
40 epochs, which is `ceil(66192/256) * 40` = **10,360 encoder updates**.

**The two objectives already apply almost exactly the same number of gradient steps to the
encoder - 1.036x apart.** Nothing was tuned to make that true; it falls out of configurations
fixed in EXP-040 and EXP-047 for unrelated reasons. So "another 40 epochs of pretraining" is a
step-matched control that nobody chose the size of.

The parallel is exact:

| | starting encoder | fresh task head | encoder updates | objective |
|---|---|---|---|---|
| EXP-047 -> E1 | E0 | policy head | 10,000 | **REINFORCE on sparse reward** |
| **EXP-050 -> E0+** | E0 | inverse-model head | 10,360 | **supervised move-naming** |

> [!important] **The match FAVOURS the control, deliberately.** Each pretraining update sees 256
> pairs; each RL update sees one episode of ~15 environment steps. The control therefore gets
> roughly **17x more data per step** and a clean supervised gradient instead of a noisy policy
> gradient. **A control that is handed the advantage and still loses makes the conclusion
> stronger, not weaker.** This asymmetry is stated now so it cannot be presented later as a
> caveat discovered after the fact.

### 2.1 A third asymmetry, found in the smoke test and recorded BEFORE the run

Warm-starting from E0 and training **one** epoch (259 updates) moves `fc1` by **7.855%** of its
norm. EXP-047's **10,000** RL updates moved it **2.91%**. Per update that is roughly **100x more
movement** from the pretraining objective.

So the control is favoured on three axes at once, not one: **~17x more data per update, a clean
supervised gradient instead of a noisy policy gradient, and ~100x more parameter movement per
update.** All three point the same way.

**Consequence for the grid, stated now:** a **refuted** Claim 1 becomes *stronger* - a control
this heavily favoured getting nothing is decisive. But if Claim 1 **confirms and F beats B**, the
win **cannot be attributed to the objective** rather than to the sheer magnitude of the updates,
and that cell must be reported as *"more pretraining wins, mechanism unattributed"*. It would
still be actionable - the practical advice would be "pretrain longer, it is offline and cheaper" -
but it would not answer the question this experiment is named for.

## 3. Design

Three arms, **two of which already exist and are not re-run**. All three are
`frozen encoder + fresh head, 10,000 episodes` - **identical RL compute, 1.0 unit each.**

| arm | encoder | how the encoder was made | mean |
|---|---|---|---|
| A (EXP-043) | E0 | 40 epochs inverse-model | **0.1800** |
| **F** | **E0+** | **80 epochs inverse-model (40 + 40 more)** | **?** |
| B (EXP-048) | E1 | 40 epochs inverse-model + 10,000 RL updates | **0.3112** |

**One variable across the three: how the encoder's extra ~10,000 gradient steps were spent** -
not at all (A), on the pretraining objective (F), or on the RL objective (B).

E0+ is produced by `train_inverse_model(..., sensory=<E0 loaded>)`, which already supports
warm-start; the inverse-model head is fresh, exactly as EXP-047's policy head was.

## 4. Claims

Paired per seed, exact permutation over `2**12`.

### Claim 1 - PRIMARY. Does more pretraining improve the encoder at all? **F minus A.**

CONFIRMED at **>= +0.05** with **p <= 0.05**, the standing bar.

EXP-040's RESULTS explicitly flagged that pretraining **had not saturated** and listed "longer
pretraining, which is cheap" as an open item, so this is a live possibility rather than a
formality.

### Claim 2 - DECISIVE. Does more pretraining MATCH RL fine-tuning? **F minus B.**

Reported with its exact p. This is the comparison the experiment exists for.

> A null here is **not** evidence of equivalence at n=12 and is pre-committed not to be called
> one. "F is not detectably different from B" is the strongest sentence a null licenses.

### Claim 3 - THE INTERPRETATION GRID, fixed here so it cannot be chosen afterwards

| F - A | F - B | reading |
|---|---|---|
| ~0, not significant | strongly negative | **RL'S OBJECTIVE IS WHAT MATTERS.** The strongest available result for the fine-tuning line: a step-matched, data-advantaged control gets nothing. |
| ~+0.13 (i.e. ~ B - A) | ~0 | **IT WAS JUST MORE GRADIENT.** Deflates EXP-047/048/049 substantially - and the practical consequence is *stop doing RL fine-tuning and just pretrain longer*, which is **offline and far cheaper**. |
| between | negative | **BOTH CONTRIBUTE.** Report the split; the objective is worth `(B - F)` and generic gradient `(F - A)`. |
| **significantly positive** | **positive** | **MORE PRETRAINING BEATS RL FINE-TUNING.** The whole EXP-047/048/049 line is superseded by something cheaper and offline. Unlikely, but EXP-040 said the objective had not saturated, so it is not dismissible. |

### Claim 4 - Does the probe's anti-correlation belong to the RL objective specifically?

This is the sharpest test available of EXP-049's biggest finding.

- **EXP-039**: inverse-model pretraining **raises** the depth-4 probe, +0.3396, 12-0.
- **EXP-049**: RL fine-tuning **lowers** the depth-6 probe, monotonically, 0-12 at p 0.0005, while
  policy success nearly doubles.

**Pre-registered prediction: E0+ probes HIGHER than E0** (more of the objective that raised it),
**while arm F's policy gain is smaller than arm B's**. If both hold, the anti-correlation is a
property of **the RL objective**, not of gradient descent on the encoder in general - and the
probe is specifically blind to whatever RL is adding.

If instead E0+ probes **lower** too, the anti-correlation is generic to encoder training and the
finding is broader but less pointed.

Measured with `experiments/048_fresh_head/diagnose_probe_tension.py`, unmodified.

### Claim 5 - the null is a real result and is pre-committed

A refuted Claim 1 - more pretraining does nothing - **closes the "more gradient" alternative
completely** and is the cleanest possible outcome. It would mean the encoder improvement in
EXP-047/048/049 is attributable to the RL objective, and the three specs that deferred this
control can have their scope caveats lifted rather than restated.

## 5. What this still does not control for

**The pretraining objective is inverse dynamics specifically.** A refuted Claim 1 says *this*
objective adds nothing further, not that *no* supervised objective could. EXP-047's Claim 5
already names value/heuristic pretraining as the redirection if the fine-tuning line stalls; that
remains a separate question.

## 6. Execution

| phase | what | workers | cost |
|---|---|---|---|
| 1 | 12 x 40 more epochs from E0, warm-started -> E0+ | 10 | ~1.7 h |
| 2 | 12 RL cells, E0+ frozen + fresh head | 6 | ~3.9 h |
| 3 | probe E0 vs E0+ (Claim 4), offline on the VPS | 2 | ~20 min |

**Phase 1 at ten workers, phase 2 at six.** EXP-040 measured 12 encoders at 10 workers in 100
min, and this session's phase 0 measured 12 at *twelve* workers in 132 min, so 10 is the better
setting for pretraining. For the RL arm, EXP-049's arm E measured 12 frozen cells at 6 workers in
**3.9 h**, which is two clean waves and beats 10 workers' ragged two.

Total ~6 h.
