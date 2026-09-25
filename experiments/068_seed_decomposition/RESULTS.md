# EXP-068 results - the seed effect is mostly INTERACTION, and neither factor owns it

> **COMPLETE.** 100 cells, 10 x 10, no tracebacks. **Validity gate PASSED**, so the claims may be
> read. Completion confirmed from **counts** (100 records, 100 checkpoints, zero python processes)
> **without reading the run log**.
>
> **HEADLINE: the largest component is the INTERACTION at 0.650.** The seed effect is mostly not
> attributable to either factor on its own; it is the specific PAIRING of a task draw with a
> training trajectory. Of the two main effects the **task draw is 3.2x the trajectory** (0.267
> against 0.084), which is the opposite of the intuition that training randomness dominates.
>
> **Claim 1 is UNRESOLVED, not confirmed**, because 0.267 falls inside the band the spec
> pre-registered as unresolvable. That band was written before any number existed and it binds.

**Pre-registration:** `docs/superpowers/specs/2026-09-24-exp068-seed-decomposition-design.md`,
committed with `run.py` and `aggregate.py` at `605e410` **before any cell ran**. Only the cost
section was ever amended, at `e201619`, after the calibration wave and before the full launch.

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo`, worktree `C:\Users\mlgbr\wt-exp053` at `605e410`, **10 workers** |
| Calibration wave | 10 diagonal cells, 2026-09-23 23:19 to 2026-09-24 00:47 local |
| Full run | 90 cells, 2026-09-24 00:50 to 13:15 local, **12.42 h** against a **13.2 h** estimate |
| Cell | EXP-036 depth 3 via `dataclasses.replace`, never retyped |

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch068_wt.ps1 -Phase full -Workers 10 -SkipExisting
.venv/bin/python -u experiments/068_seed_decomposition/aggregate.py
```

**The calibration paid for itself.** Scaling EXP-067's 46 min/cell VPS figure would have
under-priced this by **44%**; measured, the first cell took 66.1 min and the wave 87.9. Workers
held only **71% CPU-to-wall**, because 10 workers on an Ultra 9 185H spill onto its E-cores.
==The wave time prices the run, not the fastest cell==: 66.1 predicts 9.9 h against a real 12.4.

## The internal validation, which was free and is the strongest check here

**The 10 diagonal cells (`split == train`, the ordinary configuration) are BYTE-IDENTICAL to
EXP-036 on all 10 seeds.** Delta exactly 0.0000 at every seed.

| | |
|---|---|
| same machine, same seeds, same config, **different commit** | **0.0000 on 10 of 10** |
| different machine (EXP-067) | **+0.0278**, which missed EXP-036's own 0.02 bar |

Three things follow, none of which cost a cell:

1. **The `dataclasses.replace` construction reproduces EXP-036's config exactly.** Setting
   `split_seed` and `train_seed` explicitly to the same value is identical to leaving them unset,
   as `resolve_seed` intends. The grid is anchored to a known experiment.
2. **Library drift since EXP-036 is numerically INERT on this path**, across many commits.
3. ==**That isolates EXP-067's cross-machine miss to the MACHINE**==, because same-machine drift
   alone reproduces to the last float.

> [!warning] **The aggregator's line "diagonal mean 0.4100, EXP-036 published 0.3972" invites a
> false reading of a discrepancy. There is none.** EXP-036's published mean is over **12** seeds;
> the diagonal covers **0-9**. EXP-036's own seeds 0-9 average **exactly 0.4100**. A subset mean
> compared against a full-set mean is not a difference, and the aggregator should have said which
> seeds it was quoting.

## Claim 3, the validity gate - PASSED

| condition | value | bar | |
|---|---|---|---|
| total sd across the 100 cells | **0.1513** | >= 0.05 | **PASS** |
| at least one main effect resolves | **p = 0.0000** | < 0.05 | **PASS** |

Both halves mattered. The second is the resolution check: a grid that was entirely interaction
would clear the sd bar while making Claim 1 a ratio of two noise estimates.

## The decomposition

Two-way ANOVA without replication. Within one machine a cell is a deterministic function of its
two seeds, so the interaction IS the error term. Randomization over 20,000 shuffles at a fixed
seed, so the p-values reproduce exactly.

| component | MS | F vs MS_I | variance | **share** | p |
|---|---|---|---|---|---|
| **A, task draw** (`split_seed`) | 0.07921 | **5.103** | 0.00637 | **0.267** | **0.0000** |
| **B, trajectory** (`train_seed`) | 0.03559 | 2.293 | 0.00201 | **0.084** | **0.0235** |
| **I, interaction** | 0.01552 | - | 0.01552 | **0.650** | - |

## Claim 1, PRIMARY - **UNRESOLVED.** Share 0.267 against a 0.35 bar.

The mechanical verdict is CONFIRMED: 0.267 is below 0.35. **The spec does not allow that reading.**
It pre-registered that *"a true share anywhere between about 0.25 and 0.45 will not be cleanly
resolved by this design, and a result in that band must be reported as unresolved rather than
rounded to the nearest verdict."* **0.267 is in the band. The verdict is UNRESOLVED.**

> [!important] **DISCLOSURE: the aggregator prints CONFIRMED and then the band as a warning. The
> SPEC binds, not the print.** The band was written into the spec but not into the verdict logic,
> so `aggregate.py` reports the bar and flags the band separately. **The code was NOT edited after
> the numbers existed**, even to make the output more conservative, because editing an aggregator
> post hoc is the practice this repo exists to avoid and "it made the result weaker" is exactly the
> argument that would justify the reverse next time.
>
> **The lesson for the next spec: put the unresolved band IN the verdict function, not only in the
> prose.** A reader running the aggregator sees CONFIRMED.

**EXP-067's out-of-sample prediction survives.** It predicted about **0.183** from a cross-machine
correlation with a wide interval at n=12; this measured **0.267**. Same region, same conclusion
that the task draw is a minority, and the prediction was made from an experiment that shares no
cells with this one.

## Claim 2, SECONDARY - **CONFIRMED.** The trajectory main effect resolves at p 0.0235.

F_B = 2.293. The positive directional prediction fired: training-trajectory identity does carry a
real, detectable share of the variance. **It is simply much smaller than expected, and smaller than
the task draw's.**

## What this changes

1. ==**The seed effect is mostly not decomposable.**== At 0.650 the interaction is larger than both
   main effects combined. A seed's quality is the specific pairing of its task draw with its
   trajectory, not a property either one carries. **You cannot neutralise the confound by
   controlling one factor**, which is what a decomposition is usually sought for.
2. **`docs/seed-effect.md`'s practical rule is unchanged and now better founded.** Pair within
   seed. Pairing works precisely because it holds the *combination* fixed, which is the only thing
   that reproduces.
3. **Of the two main effects, the TASK DRAW is the bigger one, 3.2x.** Which 200 states a seed
   holds out matters more than where the optimiser starts. That was not the expected ordering.
4. **A same-machine, same-worker-count cost estimate held again**, 12.42 h against 13.2.
5. **Library drift is inert on this path**, measured rather than assumed, which retroactively
   strengthens every same-machine comparison in the repo that spans commits.

## What is NOT claimed

- **Not that the task draw's share is 0.267 precisely.** The design resolves it to about +/- 0.10,
  which is why 0.25 to 0.45 was pre-registered as unresolvable, and 0.267 sits inside that.
- **Not that the interaction is structureless.** `sigma2_I` is estimated as a lump. Whether it has
  exploitable structure is untested and would need replication within a cell, which a deterministic
  system cannot provide without varying something else.
- **Not that this generalises past depth 3 or past an encoder-free configuration.**
  `docs/seed-effect.md` measured its effect at depth 5 with an E0 encoder. If the mechanism differs
  there, the encoder is the obvious next suspect.
- **Not that `split_seed` separates the eval draw from the train draw.** It fixes both sides of the
  partition at once, so "task draw" is measured as a unit.
- **Not a claim about EXP-036's verdicts.** This re-uses its cell, not its questions.
