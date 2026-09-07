# EXP-058 results - VOID. The validity gate failed, and the fault is in the spec.

> [!danger] **NO CLAIM IN THIS EXPERIMENT MAY BE REPORTED.**
> Claim 3 was pre-registered as a **condition**: *"arm M's `mean_n_stored` must exceed 10... If it
> does not... every claim above is void."* It measured **6.17**. The contract is honoured here
> rather than argued with.
>
> **The gate was UNSATISFIABLE BY CONSTRUCTION, so this is a specification failure, not an
> experimental one.** The data is valid. The contract that would have licensed reading it is not.

**Pre-registration:** `docs/superpowers/specs/2026-09-04-exp058-memory-reask-design.md`, committed
at `1ba9089` before any number existed. **No threshold has been edited**, before or after.

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo`, worktree at `4a96137`, 6 workers, launched 2026-09-04 20:02 |
| Arms | A (amnesic), M (memory), S (shuffled), **all 12/12. Run completed 2026-09-06 17:59.** |
| Cost, measured | **3.37 h per cell** (6 workers x 13.46 CPU-hours over 4 cells each) |
| Projected total | 6 waves x 3.37 h = **~20.2 h**, inside the spec's stated 17-30 h |
| Interruption | the laptop slept in transit for ~26 h of the 39.7 h wall clock. **No work was lost**: the process tree survived and resumed on wake, and no record was corrupted. |

## Why the gate failed

**1. It required more stores than an episode has steps.**

`mean_n_stored` counts stores within a single episode, and `cube_baseline` clears the hippocampus
per episode, so it is bounded above by that episode's step count. Depth-6 episodes here average
**7.76 steps**, because this policy *solves* most of them well before the 15-step cap.

| arm | mean_steps | mean_n_stored | stores per step |
|---|---|---|---|
| A, amnesic | 7.76 | 6.02 | 0.776 |
| M, memory | 7.75 | 6.17 | **0.796** |

**Requiring `> 10` required a mean episode longer than this arm can produce.** The threshold was set
from an assumption about episode length rather than a measurement of it. **This is the same defect
as EXP-057's absolute 1e-6 gate**, which was calibrated on a depth-3 smoke run and evaluated at
depth 7: *a threshold chosen in one regime and applied in another.* Twice in three experiments.

**2. It could not discriminate the arms it was meant to validate.**

`cube_baseline` sets `use_memory = (cfg.readout != "concept")`, so **all three arms store**. The
amnesic arm zeroes `W_rec` at the READ site only. Arm A's 6.02 against arm M's 6.17 is the same
quantity measured twice, not a difference between arms. **A gate on storing was never going to
validate a contrast about recalling.**

## What the gate was FOR is satisfied, and that is not the same thing

The gate existed to catch a specific historical defect: `Hippocampus.store()` once assigned instead
of accumulating and held **exactly one pattern**, behind an assertion that only checked
`count_nonzero(W_rec) > 0`. **0.796 stores per step is not that.** The hippocampus is accumulating
normally.

But *"the failure I was guarding against did not occur"* is not *"the condition I wrote passed"*.
Treating them as equivalent, after seeing the numbers, is precisely the reasoning a pre-registration
exists to prevent. **So the claims stay void.**

## Claim 3's SECOND gate passed

The spec gated two things. The `mean_n_stored > 10` condition failed as described above. The other
condition, that arm S's `unshuffled_frac` stay **below 0.20** so the shuffle-null is genuinely a
null, **PASSED**: mean 0.1652, worst seed 0.1727. **The shuffle machinery works.** Only the
storing gate failed, and only because it was mis-specified.

## The numbers, which this document does NOT license anyone to interpret

Recorded for provenance only. **They carry no CONFIRMED or REFUTED status and must not be cited as
a result.**

| arm | success | revisit_rate | mean_steps | mean_n_stored |
|---|---|---|---|---|
| A, amnesic | **0.3425** | 0.2541 | 7.76 | 6.02 |
| M, memory | 0.3154 | 0.2620 | 7.75 | 6.17 |
| S, shuffled | 0.2979 | 0.2613 | 7.82 | 6.06 |

| contrast | delta | W-L-T | p |
|---|---|---|---|
| M - A (would have been Claim 1) | -0.0271 | 4-7-1 | 0.2246 |
| M - A on `revisit_rate` (Claim 2) | +0.0079 | 9-3-0 | 0.1479 |
| M - S (would have been Claim 4) | +0.0175 | 7-5-0 | 0.5659 |
| S - A (not a claim) | **-0.0446** | 3-9-0 | 0.0859 |

Context, not a control: `exp049_fresh2_d6` (concept readout) = 0.3579.

> [!important] THE ORDERING REPRODUCES EXP-030'S TRAP EXACTLY, AND THAT IS WORTH RECORDING EVEN
> FROM A VOID RUN.
> **Memory beats the shuffle-null (+0.0175) and does NOT beat the amnesic control (-0.0271).**
> EXP-030 found the identical structure: +10.8 points over the shuffle-null, +1.2 over amnesic.
> Six years of cube experiments apart, on a policy 15x better, **the same two-arm design would
> have reported memory as a win both times.**
>
> The largest contrast here is **S - A at -0.0446, p 0.0859**, the closest thing to a signal in the
> table: *incorrect* memory hurts. That is what a memory-versus-shuffle-null comparison actually
> measures, and it is why the spec fixed M vs A as primary before any number existed. **That
> decision was vindicated even though the gate that would have licensed reading it was not.**
>
> None of this is a result. It is a reason to run the successor.

## What a correct gate would be

**Verify that RECALL differs between the arms, not that storing happens.** The arms differ at the
read site, so the check belongs there: a non-zero norm of the recall block in arm M against an
**exactly zero** one in arm A, per stage. That is measurable with what the readout already computes,
it discriminates by construction, and it has no threshold to mis-calibrate because arm A's value is
exactly zero by definition.

## The re-run problem, stated plainly

**A re-run on seeds 0-11 cannot un-see these numbers.** Seeded runs in this project are
byte-identical - a property relied on elsewhere as a correctness check - so re-running the same
seeds under a corrected gate reproduces exactly these records. That is laundering, not replication.

An uncontaminated re-test needs one of:

1. **New seeds.** Expensive for the same reason the EXP-056 repeat was: the base config uses
   EXP-049's E2 encoders, which exist for seeds 0-11 only. Extending them means re-running the
   EXP-047 fine-tune and the EXP-049 second round at the new seeds first.
2. **A different measurement.** The same question at a different depth, or on a different base
   config whose encoders already exist at more seeds. Cheaper, and it is a genuinely new
   measurement rather than a repeat.

**What is NOT contaminated:** the claim thresholds (+0.05 on success, -0.02 on revisit_rate,
alpha 0.05) were pre-registered and have not been touched. Only a validity precondition was
mis-stated. A successor may reuse those thresholds; it may not reuse these seeds and call the
result a test.

## What this experiment did establish

- **The memory machinery runs at depth 6 on a working policy**, at 0.796 stores per step, with the
  shuffle-null verified genuinely null at `unshuffled_frac` 0.1652. The three-arm design executes
  end to end. That was never true before: EXP-030 ran it on a 2.2% policy.
- **The successor has a concrete prior**: if the ordering above holds, the effect to detect is
  around -0.03 for M vs A and -0.045 for S vs A, both well under the +0.05 bar the spec set. A
  successor should either raise n or lower the bar deliberately, and say which before running.
- **The cost is now measured**: 3.37 h per cell, ~20.2 h for 36 cells at 6 workers.
- **A long run survives a laptop sleeping in transit.** 26 h of the 39.7 h wall clock was sleep; the
  process tree resumed and no record was lost or corrupted.
