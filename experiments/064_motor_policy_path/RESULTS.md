# EXP-064 results - the run completed, the experiment did not

> **COMPLETE and UNINFORMATIVE.** 24 cells, no tracebacks, no reboot, both validity gates
> PASSED. ==**And the primary cannot be read, because both arms scored exactly 0.0000.**==
>
> **HEADLINE: the capacity-matched control is BROKEN, and that is the only finding here.**
> EXP-043 ran this same configuration with a **linear** head at depth 5 and scored **0.3229**
> over 24 seeds. Arm C differs from it by exactly one field, `head_hidden=218`, and scored
> **0.0000** with a policy entropy of **0.0120** and a modal action fraction of **1.000**.
> Matching capacity with an MLP head cost 0.32 and bought a dead policy.
>
> **NOTHING ABOUT THE TOPOLOGY MAY BE READ FROM THIS RUN.** A difference between two arms that
> both sit on the floor is 0 by construction.

**Pre-registration:** `docs/superpowers/specs/2026-09-19-exp064-motor-policy-path-design.md`.
**The aggregator was written before any EXP-064 number was seen**: the run finished during other
work, completion was confirmed from **counts alone** (24 records, zero python processes), and
neither the log nor any record was opened until `aggregate.py` and its 11 tests existed.

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo`, worktree `C:\Users\mlgbr\wt-exp053` at `a47387f`, 6 workers |
| Wall clock | 2026-09-19 00:00 to 14:00 laptop-local, **14.0 h** against a **13.7 h** estimate |
| Estimate error | **2%**, from pricing each arm separately by ratio against a known laptop cost |
| Depth / budget | 5, 10,000 episodes, curriculum 1..5, `max_steps_by_depth=((1, 2),)` |
| Seeds | 0-11, frozen EXP-040 E0 encoders |
| Reboot | **none.** Uptime 112 h, last boot 2026-09-15 |

```bash
powershell -File C:\Users\mlgbr\launch064_wt.ps1 -Phase rl -Workers 6 -SkipExisting
.venv/bin/python -u experiments/064_motor_policy_path/aggregate.py
```

## What the gates said, and why they were not enough

| gate | reading | arm P | floor | verdict |
|---|---|---|---|---|
| 1 | `region_drift` | **2.7051** (min 1.9724) | 0.01 | PASS, 197x |
| 2 | `motor_rate_mean` | **0.1534** (min 0.1354) | 0.02 | PASS, 6.8x |

**Both gates did exactly what they were designed to do.** The prefrontal and motor regions
really trained (relative drift 2.71, so they moved further than their own initial magnitude),
and the spiking pathway really fired (`motor_silent_frac` **0.0000**). The mechanism was live.

> [!danger] **THE GATE I DID NOT WRITE, and the number to write it with was already in the repo.**
> Both gates guard **arm P's mechanism**. Neither can detect that the **comparison has no
> resolution**. Both arms scored 0.0000, so the primary was decided before the run started.
>
> ==A contrast needs a working reference, not just a matched one.== The floor was calibratable
> **before dispatch** from `EXP-043`, which ran this exact config with a linear head at depth 5
> and scored **0.3229**. A gate of "arm C mean success >= 0.10" would have caught this, and
> every input it needed was sitting in `experiments/043_*/outputs` the whole time.

## The claim, as pre-registered

**Claim 1, PRIMARY - `P` minus `C`: +0.0000, p 1.0000, W-L-T 0-0-12. NULL.**

The verdict stands as written. **It is vacuous.** `aggregate.py` gained a **post-hoc**
resolution check, added after the numbers existed and labelled as such in the source: it is a
correction to interpretation text that had no branch for "both arms scored zero" and would
otherwise have printed *"competitive"*. It does not alter the verdict or any threshold.

## The mechanism, which is the part worth keeping

| reading | arm P (motor) | arm C (MLP head) |
|---|---|---|
| held-out success | **0.0000** | **0.0000** |
| train-side success | **0.0000** | **0.0000** |
| `mean_train_entropy` | **1.1131** | **0.0120** |
| `greedy_modal_action_frac` | 1.000 | 1.000 |
| `revisit_rate` | **0.2591** | 0.6686 |
| `region_drift` | 2.7051 | n/a (frozen, as designed) |

**The two arms failed differently, and only one of them failed in a familiar way.**

- **Arm C collapsed.** Entropy **0.0120** is a dead policy, and it cycles (revisit 0.669). This
  is EXP-031's policy collapse, reached by adding a hidden layer to a head that works fine
  without one.
- **Arm P did not collapse during training.** Its entropy stayed healthy across the curriculum
  (1.758 -> 0.915 over stages 1 to 5) and it revisits **less than half** as often as the
  control. Its per-stage `train_solved_frac` decayed 0.173 -> 0.001 as depth rose. It trained,
  it fired, it explored, and it never learned to solve anything.

**`region_drift` of 2.71 is itself a warning.** The regions moved nearly three times their own
initial magnitude. `region_lr` was set to 1e-2 to match the head's lr, calibrated over 120
episodes at drift 0.598; over 10,000 episodes it reached 2.71. That is a plausible reason arm P
learned nothing, and **it is a hypothesis, not a finding** - the lr was deliberately not swept
and the spec said a null would be a bound at this lr.

## What must NOT be concluded

- **Not** that the five-region topology is competitive. The control was broken.
- **Not** that the topology fails. Arm P was never measured against a working reference.
- **Not** that `region_lr=1e-2` is wrong. That is an untested hypothesis about drift 2.71.

## What a follow-up would need

1. **A control that works.** Either the linear-head concept arm (EXP-043's 0.3229, reusable at
   these seeds) accepting the capacity mismatch and *saying so*, or a capacity-matched control
   demonstrated to clear the floor before it is used as a reference.
2. **A resolution gate**: the control arm must clear a floor calibrated from EXP-043.
3. **A `region_lr` that keeps drift bounded**, chosen by calibrating over the real 10,000-episode
   budget rather than 120 episodes.

**None of this is being run now.** The honest position is that EXP-064 cost 14 hours, produced a
working mechanism and no usable contrast, and the reason it failed was a control that was never
checked for competence.
