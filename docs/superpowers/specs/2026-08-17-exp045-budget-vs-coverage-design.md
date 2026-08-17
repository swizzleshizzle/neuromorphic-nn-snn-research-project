# EXP-045 design - was it the total budget, or the episodes at the deepest shell?

**Status: pre-registration. Written 2026-08-17, before any EXP-045 number exists.**
Thresholds are fixed. If one is edited after data arrives, that edit is the finding.

## 1. Why

EXP-044 arm B took depth 7 from **0.0621 to 0.1971** by raising the episode budget from 10,000 to
44,000. The write-up first called that a **coverage** result - episodes per training state at the
deepest shell, 0.044 to 0.191 - and **that framing is not supported**, because arm B raised the
total budget and therefore raised *everything*: stage 7 went 1,432 to 6,290 episodes, and so did
every shallow stage.

Two explanations survive arm B, and they are not separable from it:

| | prediction for this experiment |
|---|---|
| **H-coverage** - what matters is episodes at the deepest shell | back-loading a fixed budget reproduces arm B's gain |
| **H-budget** - what matters is total training | back-loading a fixed budget does nothing, or hurts |

**EXP-037 already points at H-budget.** At depth 4 with a fixed 10,000 episodes it shifted share
toward the evaluated depth and success fell **monotonically**: 0.1591 at the equal split, 0.1078
at 50%, 0.0921 at 75%. That is H-coverage's prediction failing in the opposite direction.

> [!important] This is not re-litigating a closed question
> "Curriculum stage weighting" is on the refuted-and-closed list, and it stays there as an answer
> to *"is weighting a lever for performance?"* - it is not. **This asks a different question:**
> which quantity explains EXP-044 arm B. It also runs in a different regime - depth 7 rather than
> 4, a **pretrained** encoder rather than a frozen random one, and the depth-1 cap in place.
>
> **EXP-037's result is this experiment's prediction.** That makes it a risky test of whether a
> prior finding generalises, not a second bite at it.

## 2. The arm - 12 runs, one variable

| | EXP-044 arm A (baseline, NOT re-run) | **EXP-045** |
|---|---|---|
| depth | 7 | 7 |
| **total episodes** | **10,000** | **10,000** |
| `curriculum_weights` | `()` uniform | **`(1,1,1,1,1,1,10)`** |
| episodes at stage 7 | 1,432 | **6,250** |
| coverage at stage 7 | 0.044 | **0.190** |
| encoders, cap, entropy, seeds | EXP-040, `((1,2),)`, 0.0, 0-11 | identical |

**6,250 matches arm B's 6,290 to within 0.6%**, and 0.190 matches depth 6's 0.190 exactly. So this
arm has arm B's deepest-shell exposure on arm A's budget. Everything else is copied from arm A,
which is the paired baseline: same seeds, same encoders, same total budget.

**Unlike EXP-044, there IS a paired baseline here**, so Claim 1 is paired and carries a p-value.

Cost: 141,875 env steps per run, **1.34x** arm A (back-loading buys longer episodes), so about
**10 h** at 12 workers on the laptop.

## 3. The contract

Paired per seed against EXP-044 arm A, exact permutation over 2^12 = 4,096 sign flips, two-sided.

### Claim 1 (PRIMARY) - does deepest-shell coverage reproduce arm B's gain?

> **H-coverage CONFIRMED** if the paired delta is **>= +0.05** at **p <= 0.05**.
> **H-coverage REFUTED** otherwise.

Arm B's gain over arm A was **+0.1350**. If coverage is the operative variable, this arm should
recover a large part of that, and +0.05 is a generous floor - a third of it.

**A delta between 0 and +0.05, or positive with p > 0.05, refutes H-coverage as *the* explanation
without showing the effect is zero.** Say that, rather than rounding it to either story.

### Claim 2 - does it clear the working bar? Same rule as EXP-044, for comparability

mean **>= 0.10**, margin **>= 1.0 SE**, **>= 8 of 12** seeds above. The floor is already measured
at **exactly 0.0000** (EXP-044, 12 seeds), so 0.10 binds and no floor arm is re-run.

### Claim 3 - MECHANISM, descriptive

`greedy_modal_action_frac`, against arm A's and against EXP-037's back-loading trend (0.685 at the
equal split rising to 0.757 at 75%). **If back-loading hurts, this says whether it hurts by driving
collapse** - which is the mechanism EXP-037 saw. Prefer a mechanism measurement to a score.

Descriptive only: one number per seed, and no threshold was pre-registered for it.

### Claim 4 - failure counts, DESCRIPTIVE, no p-value

Seeds at exactly 0.000, against arm A's 1/12. **n=12 cannot show a count went to zero**, so no test.

### Claim 5 - the pre-committed null, and it is the interesting outcome

> If Claim 1 refutes, **the operative variable is TOTAL BUDGET**, and EXP-037's monotone decline
> survives a pretrained encoder, the depth-1 cap, and three extra depths. That is a real result:
> it means the whole depth series is a **budget** series, every "depth N stopped working" result
> in this project is confounded with episodes, and the next experiment is **depth 6 at raised
> total budget** rather than anything about coverage.
>
> It would also retire the phrase "coverage" from the EXP-044 write-up entirely.

## 4. What would make this wrong

- **Any difference from arm A other than the weights.** Encoders, cap, entropy, seeds, total
  episodes and `heldout_cap` are all copied. `record_filename` does not cover
  `curriculum_weights`, so the tag must differ or these would overwrite arm A's records.
- **Reading `stage_trace` as if it were paired.** Arm A has one; it is descriptive here.
- **Treating a null as "weighting does nothing".** It would mean weighting does not *rescue* a
  starved run, which is narrower.

## 5. Regenerate

```bash
.venv/bin/python -u experiments/045_budget_vs_coverage/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 12 --skip-existing
.venv/bin/python experiments/045_budget_vs_coverage/aggregate.py
# EXP-044 arm A's records are the paired baseline and MUST be present.
```
