# EXP-046 design - is the depth series a BUDGET series? Depth 6 at 4.4x.

**Status: pre-registration. Written 2026-08-18, before any EXP-046 number exists.**
Thresholds are fixed. If one is edited after data arrives, that edit is the finding.

## 1. Why

EXP-044 took depth 7 from **0.0621 to 0.1971** by raising the budget 10,000 -> 44,000, and EXP-045
established that the operative variable is **total budget** rather than exposure at the deepest
shell: giving stage 7 the same episode count *inside* the smaller budget scored **0.0142**, a
paired **-0.0479** at **p 0.0010**, with the policy collapsing (modal 0.561 -> 0.847, stage-7
entropy falling to 2.7e-06 on a 2.2% training solve rate).

That leaves one question, and it is the one that decides how every earlier number reads:

> **Was depth 7 specifically starved, or is every depth in this series budget-limited?**

Every published depth here was measured at a fixed 10,000 episodes. If depth 6 responds to 4.4x
the way depth 7 did, then **"depth N stopped working" has meant "depth N stopped working at 10,000
episodes"** all along, and the series needs restating. If depth 6 barely moves, depth 7 was the
special case - plausibly because its shell is **3.7x** larger (33,058 against 8,969).

## 2. Why 44,000 and not 51,000

An earlier note proposed 51,000 to give depth 6 "depth-5-like exposure". **That target was a
coverage figure, and EXP-045 refuted coverage as the operative variable**, so designing around it
would build on the thing just disproved.

**44,000 is the same 4.4x multiplier that worked at depth 7**, which makes this a direct analogue
of the EXP-044 A->B step rather than a new quantity. It is also 3.7 h cheaper.

## 3. The arm - 12 runs, one variable

| | EXP-043 depth 6 (baseline, NOT re-run) | **EXP-046** |
|---|---|---|
| depth | 6 | 6 |
| **total episodes** | **10,000** | **44,000** |
| curriculum | `(1..6)` uniform | `(1..6)` uniform |
| encoders, cap, entropy, seeds | EXP-040, `((1,2),)`, 0.0, 0-11 | identical |
| measured | **0.1800** | - |

Paired per seed against EXP-043's `exp043_capped_d6` cells: same seeds, same encoders, same cap,
same curriculum. **One variable: the episode budget.** A paired p-value is therefore legitimate
here, as it was in EXP-045 and was not in EXP-044.

Cost: 418,011 env steps per run, **4.40x** the baseline, about **23 h** at 12 workers.

**No floor arm.** EXP-036 measured depth 6's chance floor at **0.0008**, so the 0.10 working bar
binds, exactly as in EXP-043 and EXP-044.

## 4. The contract

Paired per seed, exact permutation over 2^12 = 4,096 sign flips, two-sided.

### Claim 1 (PRIMARY) - does 4.4x help depth 6 as it helped depth 7?

> **CONFIRMED** if the paired delta is **>= +0.05** at **p <= 0.05**.

Depth 7's gain from the same multiplier was **+0.1350**. The +0.05 bar is the same one EXP-043 and
EXP-045 used, so the three are directly comparable.

**A delta between 0 and +0.05, or positive at p > 0.05, refutes the strong reading without showing
the effect is zero.** Report it that way rather than rounding to either story.

### Claim 2 - the escalation, conditional and pre-registered

> **If Claim 1 CONFIRMS**, run a **25,000-episode midpoint** (~13 h) to see whether returns are
> still climbing at 44,000 or already flattening. A confirmed Claim 1 makes the *shape* of the
> budget response the next question; without it the midpoint measures nothing.
>
> **If Claim 1 REFUTES**, the midpoint is NOT run.

### Claim 3 - MECHANISM, descriptive

From `stage_trace` at the deepest stage, against EXP-045's collapse signature:

| | EXP-045 (collapsed) |
|---|---|
| entropy, first 10% -> last 10% | 0.5914 -> 0.0979 |
| entropy minimum | 2.7e-06 |
| training solve rate | 0.0218 |

If more budget helps, the deep stage should **not** show that collapse; if it does not help,
whether it collapsed anyway localises the failure. Also report `greedy_modal_action_frac` against
the baseline's. Descriptive: no threshold pre-registered.

### Claim 4 - failure counts, DESCRIPTIVE, no p-value

Seeds at exactly 0.000, against EXP-043's 1/12 at depth 6. **n=12 cannot show a count went to
zero**, so no test.

### Claim 5 - the pre-committed null, and it is a real result

> A refuted Claim 1 means **depth 7 was specifically starved and the series is NOT simply a budget
> series.** The most likely reason is shell size - depth 7's is 3.7x depth 6's - which would make
> the budget requirement a function of how much there is to learn rather than of depth as such.
> It would also mean the earlier numbers stand roughly as published, with "at 10,000 episodes"
> attached as a caveat rather than as a correction.

## 5. What would make this wrong

- **Any difference from EXP-043's depth-6 cell other than `episodes`.** Encoders, cap, entropy,
  seeds, curriculum and `heldout_cap` are copied.
- **Tag collision.** `record_filename` does not cover `episodes`, so the tag carries the budget:
  `exp046_d6_e44000`.
- **Computing a p-value against depth 7's numbers.** Different depth, unpaired. Claim 1 is paired
  against depth 6 at 10,000 and nothing else.

## 6. Regenerate

```bash
.venv/bin/python -u experiments/046_depth6_budget/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 12 --skip-existing
.venv/bin/python experiments/046_depth6_budget/aggregate.py
# EXP-043's exp043_capped_d6 records are the paired baseline and MUST be present.
```
