# EXP-067 results - the TASK is portable, the TRAINING is not, and 2 of 3 verdicts survive

> **COMPLETE.** 24 cells, 12 trained and 12 floor, no tracebacks. **Validity gate PASSED**, so the
> claims below may be read. Completion was confirmed from **counts alone** (24 records, 12
> checkpoints, zero python processes) **without reading the run log**, which prints one success rate
> per line.
>
> **HEADLINE: retraining EXP-036 depth 3 on a second machine reproduces the CONCLUSIONS and not the
> NUMBERS.** Claim 1 (the numeric bar) is **NOT REPLICATED** at +0.0278 against a 0.02 tolerance.
> Claims 2 and 3, both of which are verdicts, **replicate exactly**.
>
> **The sharpest result is not in the claims.** The floor arm, which does no training at all, is
> **byte-equal on 12 of 12 seeds** across the two machines. So the states, the scramble streams and
> the RNG are perfectly portable, and **every bit of the divergence is in training**. Per seed that
> divergence is large: deltas run from **-0.3000 to +0.2333**, sd **0.1496**, on an arm whose mean
> only moved 0.0278.

**Pre-registration:** `docs/superpowers/specs/2026-09-23-exp067-cross-machine-replication-design.md`,
committed with `run.py` and `aggregate.py` at `67572ab` **before any EXP-067 number existed**. Every
threshold is EXP-036's own, imported from its `run.py` rather than retyped, and the configs come
from its `sweep_configs` filtered to depth 3.

## Provenance

| | |
|---|---|
| Run | **this VPS** (`liquidweb-vps`, 2 cores, 4 GB), 2 workers, branch `exp-067-cross-machine-replication` |
| Wall clock | 2026-09-23 05:50 to 10:30 UTC, **~4.7 h**, against a **4.6 h** estimate |
| Originals | `SwizzlesDuo` (Intel Ultra 9 185H, 22 cores), the machine that produced EXP-036 |
| torch | **2.13.0+cpu on both**, so the torch version is controlled |
| Python | **3.10.12 here against 3.13.14 there**, so this is a different ENVIRONMENT, not only a different CPU |

**Run here deliberately.** The laptop produced the originals, so re-running there would test
determinism, which already passes 12 of 12. Only a second machine tests the claim.

```bash
nohup .venv/bin/python -u experiments/067_cross_machine_replication/run.py --workers 2 --skip-existing &
.venv/bin/python -u experiments/067_cross_machine_replication/aggregate.py
```

**The cost estimate held**, priced from a 2,788 s measurement of this exact cell on this machine
during the week-25 audit. That is the second time a same-machine estimate has held after five scaled
ones came in low.

## Claim 4, the validity gate (INVERTED) - PASSED

**12 of 12 retrained heads differ byte-wise from their published counterparts.**

The gate is inverted on purpose: had the checkpoints come back identical, this machine would not be
meaningfully different and the experiment would be testing nothing. **A same-machine re-run fails
this gate**, which is what makes it a real gate rather than a formality. Week 24's lesson applied:
gate the comparison's resolution, not just the arm's mechanism.

## The per-seed table, which is the substantive result

| seed | published (laptop) | retrained (VPS) | delta | floor, published | floor, retrained |
|---|---|---|---|---|---|
| 0 | 0.2667 | 0.2000 | -0.0667 | 0.0333 | 0.0333 |
| 1 | 0.6333 | 0.5667 | -0.0667 | 0.1000 | 0.1000 |
| 2 | 0.4000 | 0.5333 | +0.1333 | 0.0333 | 0.0333 |
| 3 | 0.2000 | 0.4333 | **+0.2333** | 0.0000 | 0.0000 |
| 4 | 0.4667 | 0.6000 | +0.1333 | 0.0000 | 0.0000 |
| 5 | 0.4000 | 0.5000 | +0.1000 | 0.0000 | 0.0000 |
| 6 | 0.4000 | 0.4333 | +0.0333 | 0.0000 | 0.0000 |
| 7 | 0.3333 | 0.4333 | +0.1000 | 0.0000 | 0.0000 |
| 8 | 0.4667 | 0.4667 | +0.0000 | 0.0000 | 0.0000 |
| 9 | 0.5333 | 0.2333 | **-0.3000** | 0.0000 | 0.0000 |
| 10 | 0.2667 | 0.1000 | -0.1667 | 0.0000 | 0.0000 |
| 11 | 0.4000 | 0.6000 | +0.2000 | 0.0000 | 0.0000 |
| **mean** | **0.3972** | **0.4250** | **+0.0278** | **0.0139** | **0.0139** |

> [!important] **THE FLOOR COLUMN IS THE CONTROL, AND IT IS PERFECT.** The random arm trains
> nothing, so it reads the task and the RNG alone. It is **identical on all 12 seeds**, including
> the two non-zero ones. Python 3.10 and 3.13 therefore build the same shells, the same held-out
> splits and the same scramble streams.
>
> **That isolates the divergence entirely to the training path**, and it is what makes the trained
> column interpretable. Without it, "the machines differ" could have meant the seeds drew different
> cubes.

**Within a machine the same comparison is exact.** EXP-036 against EXP-035 measured +0.0002 and was
byte-identical on every measured quantity. Across machines the per-seed sd is **0.1496**. So seed
identity pins the result completely on one machine and **not at all** across two.

## Claim 1, PRIMARY - the replication bar. **+0.0278 against 0.02. NOT REPLICATED.**

The spec pre-registered that on a miss *"the size and sign of the delta are the finding"*:
**the retrained mean is HIGHER, 0.4250 against 0.3972.** This is not a weaker machine.

> [!warning] **DISCLOSURE: Claim 1 WAS UNDERPOWERED BY CONSTRUCTION, AND THAT IS THIS EXPERIMENT'S
> OWN DESIGN ERROR.** The cross-machine per-seed sd is 0.1496, so the se of the mean delta at n=12
> is **0.0432**. EXP-036's tolerance of 0.02 is **0.46 se**. Under the null that the two machines
> are equivalent, the exact sign-flip distribution puts **P(|delta| <= 0.02) at 0.326**: ==even with
> perfectly equivalent machines this bar fails about two times in three.==
>
> **This is the gate-calibration rule, and I broke it while quoting it.** The spec argued that *"a
> replication that invents its own bar is not a replication"* and imported EXP-036's tolerance
> unchanged. That tolerance was calibrated in a **same-machine** regime where the replication noise
> is **exactly zero**. Applying it to a regime with an se of 0.0432 is precisely *"a threshold
> chosen in one regime and applied in another"*.
>
> **It was calibratable before dispatch, from a number already printed on screen.** The week-25
> audit retrained this exact cell at seed 0 one day earlier. `scripts/verify_e2e_reproduction.py`
> prints the retrained success rate in a column (line 104) under a header that says it is
> *"Comparing the TRAINED POLICY HEAD byte for byte, not a success rate"*. That value was 0.2000
> against a published 0.2667: **one subtraction, 3.3x the tolerance, and the power problem is
> visible.** Same shape as EXP-064, where the floor was calibratable from a number already in the
> repo.
>
> **The verdict is NOT being overturned.** The bar was pre-registered, the delta exceeds it, and
> Claim 1 reads NOT REPLICATED. Rewriting it now is the EXP-058 mistake. What this disclosure
> changes is the **weight** Claim 1 carries: it confirms the numbers are machine-specific, which the
> audit already established, and it could not have done much more than that.

**Descriptive only, not pre-registered and carrying no verdict:** paired delta +0.0278, sd 0.1496,
W-L-T 7-4-1, exact paired permutation over 4,096 sign flips **p = 0.5869**. The mean delta is not
distinguishable from zero. That is reported because the direction and size were pre-registered as
the finding, and pretending to more precision than n=12 supports would misrepresent them.

## Claim 2 - the "working" verdict. **WORKING. REPLICATED.**

| | |
|---|---|
| measured floor, this machine | **0.0139** |
| needs | >= 2.0x floor (0.0278) **and** >= 0.10 |
| trained mean | **0.4250** |
| EXP-036's verdict | WORKING |

**Clears the absolute bar by 4.2x and the floor multiple by 15x.** Not a close call on either
machine, which is the honest caveat: this verdict had a wide margin and was never at risk.

## Claim 3 - the gap verdict. **SAME ZONE. REPLICATED.**

| | published | retrained |
|---|---|---|
| gap | **+0.1093** | **+0.0602** |
| zone (refuted < 0.05, confirmed >= 0.15) | **INCONCLUSIVE** | **INCONCLUSIVE** |

The number moved by 0.049, which is large, and the verdict did not move, which is the point the
spec fixed in advance: *a replication that moves a number without moving a verdict is a successful
replication.* **The honest caveat is that the inconclusive zone is 0.10 wide**, so it is the easiest
of the three verdicts to land in twice.

## What this changes

1. **The audit's open question is answered for this cell.** Its section 7 said the findings would
   *plausibly* survive retraining and called that **an argument, not a measurement**. It is now
   measured: **2 of 3 verdicts survived, the numeric bar did not.**
2. **Phase 4 can make the stronger statement, with a stated limit.** *"The published numbers are
   machine-specific; the conclusions tested so far are not"* is supportable for EXP-036 depth 3.
   **It is one cell of one experiment** and must be written that way.
3. **The reproduction guarantee is unchanged and now better evidenced: re-evaluate the tracked
   checkpoints, never retrain from the seed.** Retraining gives a valid run, not the published one.
4. **The task layer is provably portable.** That is new, it was free, and it matters: it means a
   reader on another machine gets the *same problem*, so a re-run is a fair test rather than a
   different experiment.
5. **A same-machine tolerance must not be reused across machines without a power statement.**
   Banked in `CLAUDE.md` alongside the gate-calibration rule.

## What is NOT claimed

- **Not that the machines are equivalent.** The primary is a miss at the pre-registered bar. The
  descriptive p of 0.5869 says n=12 cannot resolve it, which is a statement about resolution, not
  evidence of equivalence.
- **Not that every finding in the project survives retraining.** One cell of one experiment, chosen
  because it has an existing same-machine replication to sit beside.
- **Not a CPU result.** Python differs (3.10.12 against 3.13.14) as well as the processor, so the
  cause is not isolated. torch is controlled at 2.13.0+cpu on both.
- **Not anything about the encoder-dependent experiments.** EXP-036 takes no pretrained encoder,
  which is why it isolates the question and equally why it says nothing about the fifteen that do.
- **Not that a third machine would agree.** Two x86 environments is not a portability survey.
- **Not that Claim 2 is strong evidence.** It cleared its bar by 4.2x on both machines and could not
  realistically have moved.
