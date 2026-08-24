# EXP-049 design - does the two-stage recipe compound, or buy back its own compute?

> **PRE-REGISTERED. Committed before any number exists.** Every threshold below is fixed at
> commit time. A later change must be a separate commit, dated, with its reason, and made before
> the data it applies to exists.

## 1. The question

EXP-047 fine-tuned the encoder during RL; EXP-048 froze the result and trained a fresh head. That
two-stage recipe took depth 6 from **0.1800 to 0.3112**. EXP-048's own budget accounting then
tempered it:

| arm | encoder | head | total RL compute | budget-equivalent | actual | **excess** |
|---|---|---|---|---|---|---|
| A (EXP-043) | E0 pretrained, frozen | fresh | 1.00 | +0.0000 | 0.1800 | - |
| C (EXP-047) | E1 fine-tuned, joint | co-adapted | 1.33 | +0.0272 | 0.2700 | **+0.0628** |
| B (EXP-048) | E1 fine-tuned, frozen | fresh | 2.33 | +0.0808 | 0.3112 | **+0.0504** |

*(One unit = 10,000 episodes at the frozen per-step cost. Fine-tuning is 1.33 units. Budget
equivalents from EXP-046's measured 0.22 success per log10 of spend at depth 6.)*

Both arms beat the budget curve by **about the same ~0.05**. That is a **constant-return** model:
each stage buys a fixed increment over its own cost rather than compounding. EXP-048's RESULTS
stated the prediction explicitly - *"the third stage should yield about +0.05 again over its own
compute cost, not more"* - and this experiment tests it.

**The distinction matters more than the increment does.** If the recipe compounds, iterating is
the route past the budget wall EXP-046 priced. If returns are constant at +0.05 per round, then
iterating is just a slightly better way to spend compute and the wall is still there.

## 2. Design

Two new arms, run in sequence because the second needs the first's output. **One variable versus
arm B: whether the encoder has been through one fine-tuning round or two.**

| arm | encoder | during RL | head | total RL compute |
|---|---|---|---|---|
| **D** | **E1** (EXP-047's) | **fine-tuned -> E2** | co-adapted | 2.66 |
| **E** | **E2** (arm D's) | **frozen** | fresh | 3.66 |

Everything else is held at EXP-047/048's values: depth 6, 10,000 episodes, curriculum `(1..6)`,
`max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`, `normalize_advantages=False`, 12 seeds.

**`encoder_lr` stays at 1e-4, deliberately not re-piloted.** To ask "does a second *identical*
round help", the round must be identical - re-selecting the rate would confound "second round"
with "different rate". Verified 2026-08-24 that this is safe: round 1 left the weight scale
essentially unchanged (mean `|fc1|` 72.048 -> 72.074, +0.04%), moving the weights by only **2.90%**
of their norm, so the operating point the rate was chosen at still holds.

> That 2.90% is worth keeping in view on its own: **round 1 moved the encoder by under 3% in norm
> and bought +0.1312 in policy success.** Whatever it changed, it was not a large change.

## 3. Claims

Paired per seed, exact permutation over `2**12`.

### Claim 1 - PRIMARY. Does a second round beat what its compute alone buys?

**E minus B.** CONFIRMED at **>= +0.05** with **p <= 0.05**, the bar every experiment since
EXP-043 has used.

> [!important] **I PREDICT THIS REFUTES, AND SAY SO NOW.** The constant-return model fitted to C
> and B predicts arm E lands near **0.354** (its budget-equivalent 0.3040, plus the ~0.05 each
> round has bought), giving **E - B of about +0.043** - *below* the bar. So the pre-registered
> prediction is refutation, and **CONFIRMATION would be the surprise**, meaning returns compound.
>
> This also means the bar sits uncomfortably close to the confound: the extra round's compute is
> itself worth **+0.0432** on EXP-046's curve, only 1.16x below the +0.05 bar. **A delta between
> +0.0432 and +0.05 is pre-committed as UNINTERPRETABLE**, not as a near-miss. Claim 2 is the
> quantity that actually carries the answer.

### Claim 2 - THE REAL QUANTITY. Excess over budget-equivalent, in series.

Arm E's **excess** = `E - (0.1800 + 0.22 * log10(3.66))` = `E - 0.3040`, set beside C's **+0.0628**
and B's **+0.0504**.

| observed excess | reading |
|---|---|
| ~+0.05, in line with C and B | **CONSTANT RETURNS.** Each round buys a fixed increment over its own cost. Iterating is a better way to spend compute, not an escape from the budget wall. |
| clearly above +0.08 | **COMPOUNDING.** The recipe is the route past EXP-046's wall and should be iterated further. |
| clearly below +0.03 | **DIMINISHING.** The first round was special - most likely because E0 was pretrained on a different objective and had the most to gain. |

Descriptive, with the per-seed spread. n=12 on a difference of differences is weak, so this is
read as a trend across three arms, not a test.

### Claim 3 - Does round 2 help the co-adapted arm too? **D minus C**, with its exact p.

Free: arm D must run anyway to produce E2. Its score answers whether the joint-training arm
improves as well, or whether the benefit only appears after the head is refreshed.

### Claim 4 - MECHANISM. Do the trajectory statistics keep moving?

EXP-048 localised the gain to **trajectories, not single-step accuracy**: revisits fell
0.4652 -> 0.3808 (p 0.0454) and optimality rose 0.6445 -> 0.7716 (p 0.0132) while probe accuracy
went slightly *down*.

**Falsifiable prediction:** if that mechanism is real and still operating, arm E should show
`eval_revisit_rate` **lower** and `optimality` **higher** than arm B. If success rises while both
of these stay flat, EXP-048's mechanism does not generalise to round 2 and the explanation is
incomplete.

### Claim 5 - Does the probe keep drifting down?

EXP-048 found the fine-tuned encoder slightly *worse* for single-step move prediction at every
epoch budget (-0.0097 at 300 epochs, 1-11 seeds). **Prediction: E2 probes lower than E1 again.**
A probe that instead *rises* in round 2 would break the clean story that policy gain and probe
accuracy are decoupled here.

Run with `diagnose_probe_tension.py`, which already exists.

### Claim 6 - the null is a real result and is pre-committed

A refuted Claim 1 with Claim 2 near +0.05 is the **expected** outcome and is **not** a
disappointment: it would establish the recipe's return as a **constant per round**, which makes
the cost of any target depth calculable in advance and closes "just iterate it" as a strategy.
The next move would then be a **different second-stage objective**, not a third round.

## 4. What this does not control for

Same limitation as EXP-048, inherited: arm E's encoder has had **two** rounds of extra shaping of
*some* kind, and this still cannot separate "RL's objective improved it" from "more gradient steps
improved it". Unchanged, and still deserving its own pre-registration.

## 5. Execution

24 cells in two sequential arms, ~15 h total.

**Six workers, not ten** - 12 cells on 6 is two clean waves at the measured 0.115 s/step, against
two waves on 10 with the second running 8 idle at ~0.16. Per the estimator corrected 2026-08-22:

```
arm D  ceil(12/6) * 132,299 steps * 0.115 s = 2 * 15,214 s = 8.5 h
arm E  ceil(12/6) * 100,962 steps * 0.115 s = 2 * 11,611 s = 6.5 h
```

Arm E **cannot start until arm D has written all 12 encoders**, and the launcher gates on that
artifact rather than on an exit code.
