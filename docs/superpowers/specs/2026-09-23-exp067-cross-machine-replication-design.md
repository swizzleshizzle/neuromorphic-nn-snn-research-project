# EXP-067 design - do the FINDINGS survive retraining on a different machine?

> **PRE-REGISTERED. Committed before any EXP-067 number exists.** Thresholds fixed at commit
> time, and **every one of them is EXP-036's own, reused unchanged.**
> **Date:** 2026-09-23 · **Phase:** 3 (post-checkpoint) · **Grounds:** EXP-035, EXP-036, the
> week-25 reproducibility audit.

## 0. The one thing the audit could not measure

`docs/reproducibility-audit.md` established that **retraining is not portable**: EXP-036 depth 3
seed 0, retrained on this VPS, shares **0 of 390** parameters with its published head (cosine
0.524), and that experiment uses no pretrained encoder. Re-evaluating a tracked checkpoint **is**
portable, so the release guarantee is *re-evaluate, never retrain*.

The audit's section 7 names what it could not check, and this is the first item:

> **Whether the findings survive retraining.** Section 1 argues they plausibly would, from one
> encoder's move-accuracy sitting inside the seed spread. **That is an argument, not a
> measurement.**

==This measures it.== The distinction matters for the write-up: *"the numbers are
machine-specific but the conclusions are not"* is a much stronger and more useful statement than
*"the numbers are machine-specific"*, and it is not currently supported by anything.

## 1. Why EXP-036 depth 3, and why no new threshold is invented

**EXP-036 already carries a replication gate, and it is same-machine.** Its depth-3 cell is
exactly EXP-035's 10,000-episode cell - same seeds, curriculum, budget and machine - and it
reports:

| | value |
|---|---|
| EXP-035 depth 3 @ 10k | 0.397 |
| EXP-036 depth 3 | **0.3972** |
| delta, tolerance **0.02** | **+0.0002** |
| verdict | **PASS**, and byte-identical on every measured quantity |

So the identical replication has already been done **on one machine** and it passed perfectly.
==EXP-067 is that same replication on a **different machine**, against the **same bar**.== Every
threshold below is lifted from `experiments/036_generalisation_gap/run.py` without modification,
which is the point: a replication that invents its own bar is not a replication.

**Depth 3 and not 4, 5 or 6** because depth 3 is the cell with the existing same-machine
replication to compare against, and because depths 5 and 6 sit at or near the floor where a
verdict cannot move.

## 2. The arms

`sweep_configs` is imported from EXP-036 and filtered to depth 3, rather than retyped.
**Retyping is how a replication silently tests a different cell**, and this design has exactly
one job, which is to be the same cell.

| arm | cells | note |
|---|---|---|
| trained, `regionalized` | 12 (seeds 0-11) | the cell under replication |
| floor, `arm="random"` | 12 (seeds 0-11) | needed for the "working" bar; no training, nearly free |

Run on **this VPS**, deliberately. The laptop produced the originals, so re-running there tests
determinism, which is already known to pass 12 of 12. **Only a second machine tests the claim.**

## 3. Claims - all three verdicts are EXP-036's, re-evaluated

### Claim 1, PRIMARY - the replication bar

**`|mean(depth-3 success) - 0.3972| <= 0.02`**, EXP-036's own tolerance against its own measured
value.

- **REPLICATED** if within tolerance.
- **NOT REPLICATED** otherwise, and then the size and sign of the delta are the finding.

### Claim 2 - the "working" verdict

EXP-036 calls a depth *working* when the trained mean clears **both** `BREAK_MULTIPLE = 2.0`
times the **measured** floor **and** `BREAK_ABSOLUTE = 0.10`. Depth 3 was **working**. Does that
verdict survive?

### Claim 3 - the gap verdict

EXP-036's gap contract: **refuted below 0.05**, **confirmed at or above 0.15**, and *inconclusive
in between*. Depth 3 measured **+0.1093** and landed in the dead zone, which EXP-036 reported as
"inconclusive by its own pre-registered rule". ==Does the verdict land in the **same zone**?== A
replication that moves a number without moving a verdict is a successful replication.

> **The unit of replication is the VERDICT, not the number.** The audit already proved the
> numbers differ. Pre-registered here so that a small numeric shift cannot later be presented as
> a failure, nor a large one waved away.

### Claim 4, VALIDITY GATE - the run must genuinely DIFFER from the original

==This gate is inverted, and that is deliberate.== If the retrained checkpoints came back
byte-identical to the published ones, this machine would not be meaningfully different and the
experiment would be testing nothing.

**Required: at least one of the 12 trained heads differs byte-wise from its published
counterpart.** The audit already measured seed 0 at 0 of 390 parameters matching, so the gate
can pass; a same-machine re-run would fail it, so it can fail.

**This is the week-24 lesson applied**: *gate the comparison's resolution, not just the arm's
mechanism.* Here the resolution is that the two sides are actually different runs.

## 4. Cost

| | |
|---|---|
| Cells | **24** (12 trained + 12 floor) |
| Per trained cell | **~46 min**, measured on this VPS during the audit (2,788 s for depth 3 seed 0) |
| Floor cells | no training, nearly free |
| Workers | **2**, this VPS has 2 cores |
| Estimate | **~4.6 h** |

**Priced from a measurement on the machine it will run on**, which is the rule. The only prior
figure for this cell is a laptop figure and would have been wrong here.

## 5. What this cannot answer

- **Whether every finding in the project survives**, only this one. Depth 3 of EXP-036 is a
  single cell of a single experiment, chosen because it is the one with an existing
  same-machine replication to sit beside.
- **Whether a third machine agrees.** Two x86 CPUs is not a portability survey.
- **Anything about the encoder-dependent experiments.** EXP-036 takes no pretrained encoder,
  which is exactly why it isolates the question - and equally why it says nothing about the
  fifteen experiments that do.
