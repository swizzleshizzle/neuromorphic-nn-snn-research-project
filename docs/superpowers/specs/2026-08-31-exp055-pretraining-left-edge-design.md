# EXP-055 design - the 0-to-10 pretraining window, where both instruments say the action is

> **PRE-REGISTERED. Committed before any number exists.** Thresholds fixed at commit time.
> **Date:** 2026-08-31 · **Phase:** 3 · **Grounds:** EXP-036, EXP-039, EXP-040, EXP-043,
> EXP-050, EXP-052, EXP-054.

## 0. The gap this closes

**Two independent instruments now agree that everything happens before 10 epochs and nothing
happens after, and nobody has measured a single point inside that window.**

| epochs | depth-6 policy | `S` (EXP-054) | source |
|---|---|---|---|
| **0** | **0.0000** | **0.0100** | EXP-036 policy, EXP-054 `S` |
| **1, 2, 3, 5** | **unmeasured** | **unmeasured** | **this experiment** |
| 10 | **0.2012** | 0.0242 | EXP-052, EXP-054 |
| 20 | 0.1850 | 0.0241 | EXP-052, EXP-054 |
| 40 | 0.1800 | 0.0246 | EXP-043, EXP-054 |
| 80 | 0.0887 | 0.0244 | EXP-050, EXP-054 |

EXP-052 established that 10 epochs buys everything 40 does. EXP-054 established that the encoder's
distance structure is built entirely before 10 epochs and is not destroyed by over-training. **Both
curves are flat from 10 onward and both have a single unexplained jump at the left edge.**

**What is at stake.** The EXP-039/040 line motivated inverse-model pretraining as the way to fix a
representation that "throws information away", and 40 epochs was inherited rather than measured.
EXP-052 cut that to 10. **If one epoch already reaches the plateau, pretraining's contribution is
almost entirely "stop being randomly initialised"**, and the objective itself is doing far less work
than four experiments of narrative imply. That would be substantive, not an efficiency note.

## 1. Design

**Four new arms - 1, 2, 3 and 5 epochs - at 12 seeds each. One variable: epochs.**

Everything else is copied from EXP-052's Phase 2, which itself copied EXP-043's depth-6 cell field
for field: `arm="regionalized"`, `readout="concept"`, depth 6, 10,000 episodes,
`curriculum=(1..6)`, `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`,
`normalize_advantages=False`, `max_depth=6`, and the encoder **FROZEN** (390 trainable).

Encoders come from EXP-052's `pretrain_sweep.py`, which already accepts `--epochs` as a list,
trains **from scratch** rather than warm-starting, and applies EXP-040's `rl_heldout_union`
exclusions. Those exclusions are load-bearing: without them an arm could win by leakage rather
than by epochs.

**Anchors are not re-run.** 0 epochs is EXP-036, 10 is EXP-052, 20 is EXP-052, 40 is EXP-043, 80 is
EXP-050. With the four new arms this is a nine-point curve concentrated where anything moves.

### Two measurements per arm, and one of them is free

- **Policy**: held-out success at depth 6. Expensive, about 4.3 h per arm.
- **`S`**: EXP-054's sequence-sensitivity. Needs only the encoders, so it lands after roughly
  0.5 h of pretraining and long before any RL finishes. 60 cells took 8.8 s.

`S` is reported as a result in its own right, not as a gate on which RL arms run. **All four RL arms
are pre-registered here and run regardless of what `S` shows**, because choosing arms after seeing
`S` would be outcome-dependent selection.

## 2. Claims

Paired by seed, exact permutation over all `2**12` sign flips (4096; no scipy in the venv).

### Claim 1 - PRIMARY. Is there a real ramp between 1 and 10 epochs?

**`e10 - e1`. CONFIRMED at `>= +0.05` with `p <= 0.05`**: epochs 2 through 10 buy something real.

> [!important] IF IT IS NOT SIGNIFICANT, THE OUTPUT IS AN INTERVAL, NOT AN EQUIVALENCE, AND NOT A
> BOUND
> A non-significant difference is **not** evidence that the two are equal. The paired-difference
> sd measured on this project's real arms is 0.10-0.14 (e10 vs e20 sd 0.137, se 0.040; e10 vs e40
> sd 0.102, se 0.029), which at n=12 makes a non-significant result consistent with true effects
> up to roughly **+0.09** - nearly twice the +0.05 bar. Reporting that case as "bounds the effect
> below +0.05" is false, not conservative: n=12 is under-powered for this contrast (see Power,
> below), and the honest statement admits that.
>
> If the contrast is not significant and `|delta| < 0.05`, the required output reports `delta` and
> an approximate 95% interval (`mean +/- 2.201 * se`, the two-sided t multiplier at df=11), and
> says plainly that n=12 does not resolve the question and that the interval includes effects
> larger than the bar.
>
> It may **not** be reported as "one epoch is as good as ten", and it may **not** be reported as a
> bound the data cannot support. This project has a standing habit of converting a null into a
> claim, and n=12 is exactly the sample size where that is tempting and wrong.
>
> The claim is also directional (`e10 - e1 >= +0.05`), not `|e10 - e1| >= 0.05`. A significant
> difference at or beyond the bar in the **opposite** direction is a real result but is not a
> confirmation of this claim, and must be reported as a significant result running the other way,
> not as "CONFIRMED".

### Claim 2 - THE FLOOR, and the potential headline. `e1` against 0 epochs.

EXP-036 measured every seed at exactly **0.0000**, so this is descriptive rather than a paired
test - there is no variance on one side.

**If `e1` already lands near 0.20, pretraining's contribution is almost entirely escaping random
initialisation**, and the EXP-039/040 framing needs rewriting. Report the absolute value and say so
plainly if it holds.

### Claim 3 - SHAPE, and it is mechanically gated

Adjacent contrasts, each paired with an exact p: `e1 -> e2`, `e2 -> e3`, `e3 -> e5`, `e5 -> e10`.

**A shape may be named ONLY where a contrast is significant.** Everywhere else the report says
"indistinguishable" and stops. No "monotone", no "saturating", no "rising" from the ordering of
means alone.

> [!danger] THIS RULE EXISTS BECAUSE I BROKE IT THREE DAYS AGO
> EXP-052's aggregator "declared monotone decreasing, peak below 10" from the ordering of four
> means, three of which were indistinguishable at p 0.49 to 0.84. The rule adopted afterwards was
> to require significance before naming a shape.
>
> **EXP-054's aggregator then violated that same rule on 2026-08-30**, printing a Claim 4 verdict
> derived from a rank correlation over four arm means whose spread was 0.08x their own within-arm
> sd. It failed safe, but it produced a meaningless "PASSED" that a later spec could have cited.
>
> So the gate is a **condition in the aggregator**, not an intention in the prose. The aggregator
> must refuse to emit a shape word for any contrast that is not significant.

### Claim 4 - DO THE TWO CURVES SATURATE TOGETHER?

`S` across 1, 2, 3, 5, under the same KIND of significance gate as Claim 3 - no shape word without
significance - but its own family and its own Bonferroni threshold (see Multiplicity), reported
beside the policy curve. `S` is also near-deterministic per seed, so a trivially small paired delta
can still be significant; the within-arm sd is reported alongside so a trivial-but-significant
delta reads as trivial rather than as a shape.

Both `S` and policy are known to be flat from 10 onward. This asks whether they **turn over at the
same point**. A dissociation - `S` saturating at 1 epoch while policy keeps rising to 5, or the
reverse - would separate **"the encoder has the structure"** from **"the policy can use it"**, a
distinction no experiment in this project has yet been able to draw.

`S` here means the pre-registered statistic exactly as EXP-054 computes it, and `S_cross` is
reported beside it per that experiment's amendment. Neither carries a bar.

### Multiplicity, stated before the numbers

**Two families, counted separately.** The `e1` floor (Claim 2) is descriptive - EXP-036's
zero-epoch arm has no variance, so there is no p-value to correct for - and belongs to neither
count.

**Policy family: five pre-registered comparisons** - Claim 1 (the primary, `e10 - e1`) plus the
four adjacent contrasts in Claim 3. Claim 1 keeps its own 0.05 bar as the single primary. The
four Claim 3 contrasts are read against a Bonferroni threshold of **p <= 0.01** (0.05 / 5)
whenever one of them is used to name a shape.

**S family: four pre-registered comparisons** - the Claim 4 adjacent contrasts on `S`. This is a
separate family from the policy tests, not folded into the same count, because it tests a
different statistic. Read against a Bonferroni threshold of **p <= 0.0125** (0.05 / 4) whenever
one of them is used to name a shape.

### Power, stated before the numbers

At a paired-difference sd of about 0.11 (measured on this project's real arms: e10 vs e20 sd
0.137, e10 vs e40 sd 0.102), n=12 gives roughly **28% power** for Claim 1's own +0.05 effect, and
roughly **10% power** at the Bonferroni threshold for an adjacent Claim 3 step of the same size.
Four "indistinguishable" verdicts in Claim 3 are the likely outcome whatever the true shape is,
and must **not** be read as evidence that the window is flat.

The experiment is well powered for the outcome it cares about most instead: if `e1` is near zero,
`e10 - e1` is about +0.20, where power is near 1.00. It is badly powered for fine shape inside the
window, and that asymmetry is worth stating before any number exists rather than discovering it
in four rows of "indistinguishable".

## 3. What would refute the premise

- **A significant ramp across 1 to 10 with significant adjacent steps.** Pretraining has a genuine
  dose-response inside the window, the EXP-039/040 story survives intact, and the only correction is
  that its optimum is 10 rather than 40.
- **`e1` near 0.0000.** One epoch is not enough to escape the random-init floor, and the window's
  interesting region is 2 to 10 rather than 0 to 1.
- **`S` and policy dissociating.** Would be the most informative outcome and is the reason `S` is
  measured here at all.

## 4. Compute

| phase | what | cost |
|---|---|---|
| 1 | pretrain 1, 2, 3, 5 epochs x 12 seeds, 10 workers | ~0.5 h |
| 2 | `S` over the 48 new encoders | seconds |
| 3 | four RL arms, 12 seeds each, 6 workers | ~17.2 h |
| | **total** | **~18 h** |

Phase 1's estimate scales from EXP-052's measured 1.3 h for 24 encoders at 10 and 20 epochs; these
arms are strictly cheaper because they train for fewer epochs.

**Every arm runs 10,000 episodes with a frozen encoder, so there is no episode-budget confound and
EXP-046's curve does not apply.** Pretraining cost differs across arms by construction - that is the
independent variable, not a confound.

Per `docs/playbooks/remote-experiment-runs.md`: dispatch over Tailscale via the `ssh laptop` alias,
run the launcher in the ssh FOREGROUND and background it on the controller side, gate on artifacts
rather than exit codes, and round to whole waves - 12 cells on 6 workers is two waves.

## 5. Deliberately not in this experiment

- **Any fine-tuning arm.** Every arm here is frozen, matching EXP-052 and EXP-043. Fine-tuning is a
  different architecture (27,206 trainable against 390) and must never be tabulated with these.
- **Re-running any anchor.** 0, 10, 20, 40 and 80 all exist and are paired by seed.
- **The EXP-033 probe and the entropy trace.** Both retired; neither appears in any claim.
- **Choosing RL arms based on `S`.** All four are pre-registered and run regardless.
