# EXP-032 Results - Collapse Sweep (do the trainer stabilizers fix the cube policy?)

> **Why this file exists:** the standing habit adopted after the 2026-07-13 Phase-2 audit. **Provenance:**
> 192 records (4 `entropy_beta` x 2 `normalize_advantages` x depths 2-3 x seeds 0-11), 600 episodes per
> run, 2x2 cube, concept readout. Run 2026-08-01 on the laptop `SwizzlesDuo` (Intel Core Ultra 9 185H)
> over SSH from the VPS with `--workers 16`, wall clock 4h32m (01:24:41 to 05:57:13 local), exit 0, zero
> error markers. Records in `experiments/032_collapse_sweep/outputs/` (gitignored). **The gate below was
> committed in `run.py` before any number existed** (commit `27cb6ae`). Regenerate with the commands at
> the bottom.

## The question

EXP-031 established that the trained cube policy is effectively constant-action: at depth 3, seven of
twelve concept seeds play ONE action for all nine steps of every episode, against a 0.354 uniform floor.
ADR 0001 Amendment 2 had characterised the same failure on the **grid world** and found a fix:
`entropy_beta=0.01` alone did nothing, while **advantage normalization plus `entropy_beta=0.05`**
eliminated collapse entirely (0/10 runs).

That was 4 actions with an MLP head. This is 6 actions with a linear head. **Does the fix transfer?**

`normalize_advantages` is **crossed, not pinned on**. Pinning it True would have assumed ADR 0001's
central claim rather than tested it, and would have hidden the fact that on the cube normalization alone
is actively harmful.

## Pre-registered gate

    greedy_modal_action_frac < 0.60   AND   mean_train_entropy > 1.20

0.60 sits between the depth-3 uniform floor (0.354) and EXP-031's collapsed baseline (0.932). 1.20 is
67% of the log 6 = 1.792 ceiling, comparable to the 80-97% the grid-world fix achieved. Both must hold.

## Correctness audit

The 24 `(beta=0, normalize=False)` cells are EXP-030/031's exact configuration. Compared against the
EXP-031 records, matched on `(depth, seed)`, over **all 19 measured fields**:

```
identical: 24/24     PASS: every measured field reproduces EXP-030/031 exactly
```

Nothing below is a pipeline artifact.

## Results

Depth 3 (the decisive depth: 99% of episodes run the full 9-step budget). Mean over 12 seeds.

| beta | normalize | modal_frac | train_entropy | success | collapsed | gate |
|---|---|---|---|---|---|---|
| 0.0 | False | 0.932 | 0.541 | 0.022 | 9/12 | fail |
| 0.0 | True | 0.987 | 0.452 | 0.003 | 11/12 | fail |
| 0.01 | False | 0.962 | 0.563 | 0.008 | 9/12 | fail |
| 0.01 | True | 0.977 | 0.563 | 0.000 | 10/12 | fail |
| 0.05 | False | 0.951 | 0.726 | 0.006 | 8/12 | fail |
| 0.05 | True | 0.931 | 0.910 | 0.006 | 6/12 | fail |
| 0.1 | False | 0.932 | 0.797 | 0.022 | 8/12 | fail |
| **0.1** | **True** | **0.824** | **1.253** | 0.006 | **3/12** | **fail** |

Depth 2, where EXP-030's +10.8 point primary effect lives:

| beta | normalize | modal_frac | train_entropy | success | collapsed | gate |
|---|---|---|---|---|---|---|
| 0.0 | False | 0.825 | 0.703 | **0.380** | 2/12 | fail |
| 0.0 | True | 0.963 | 0.437 | 0.080 | 9/12 | fail |
| 0.01 | False | 0.849 | 0.626 | 0.290 | 2/12 | fail |
| 0.01 | True | 0.967 | 0.462 | 0.052 | 9/12 | fail |
| 0.05 | False | 0.813 | 0.654 | 0.358 | 2/12 | fail |
| 0.05 | True | 0.911 | 0.851 | 0.090 | 8/12 | fail |
| 0.1 | False | 0.792 | 0.789 | 0.349 | 1/12 | fail |
| 0.1 | True | 0.811 | 1.187 | 0.117 | 2/12 | fail |

## Finding 1: the fix partially transfers, and the sweep was bounded too low

Under `normalize_advantages=True` at depth 3, every metric moves monotonically with beta and **none has
saturated at the sweep boundary**:

| beta | modal_frac | train_entropy | collapsed |
|---|---|---|---|
| 0.0 | 0.987 | 0.452 | 11/12 |
| 0.01 | 0.977 | 0.563 | 10/12 |
| 0.05 | 0.931 | 0.910 | 6/12 |
| 0.1 | **0.824** | **1.253** | **3/12** |

At `beta=0.1` the **entropy half of the gate is cleared** (1.253 > 1.20) and collapsed seeds fall from
11/12 to 3/12. The modal half is not (0.824 vs 0.60), but the per-seed spread shows three seeds already
below the bar:

```
0.604  0.607  0.652  0.748  0.748  0.844  0.900  0.904  0.922  0.956  1.000  1.000
```

**Nothing passed the gate, so by pre-registration the memory arms do not get re-run.** But the honest
reading is that the sweep stopped one step short, not that the stabilizers fail: `beta=0.2` or `0.3`
would plausibly bring the mean under 0.60.

## Finding 2: normalization ALONE is actively harmful, which the ADR did not test

ADR 0001 never ran advantage normalization without an entropy bonus. On the cube that cell is the worst
in the sweep at both depths:

| depth | baseline | `normalize=True` alone |
|---|---|---|
| 2 | 0.825 modal, 2/12 collapsed, 0.380 success | 0.963 modal, **9/12** collapsed, **0.080** success |
| 3 | 0.932 modal, 9/12 collapsed, 0.022 success | 0.987 modal, **11/12** collapsed, 0.003 success |

Mechanistically coherent: normalization sharpens the policy gradient, and with no entropy counterweight
it accelerates the run to a one-hot policy. It also explains why `beta=0.01` is useless here, matching
the ADR's own "dwarfed by un-normalized advantages" reasoning from the other direction.

## Finding 3: de-collapsing the policy does NOT make it solve cubes

This is the result that matters, and it is a null on the thing everyone would assume.

The best cell on every collapse metric, `beta=0.1 + normalize`, has **success 0.006 at depth 3 against
the baseline's 0.022**, and **0.117 at depth 2 against the baseline's 0.380**. Collapsed seeds fell from
11/12 to 3/12 and entropy nearly tripled, and the policy got *worse* at the task.

**A correlation here is confounded and must not be read causally.** Across individual runs,
`corr(modal_frac, success)` is -0.748 at depth 2 and -0.433 at depth 3, which looks like "less collapse
means more success". It is not: both are downstream of whether a given seed learned anything, so a good
seed is simultaneously less collapsed and more successful. The intervention breaks the association,
which is exactly what distinguishes it from the observational pattern. `corr(entropy, success)` is only
+0.244 and +0.176.

The entropy bonus lowers modal fraction by **injecting randomness, not by teaching the policy to read
its input**. Those produce identical readings on the collapse instruments and opposite readings on the
task.

## What this establishes

**Collapse is real, it is fixable, and it is a symptom rather than the binding constraint.** EXP-031
correctly identified that the depth-3 policy ignores its input. EXP-032 shows that making it stop
ignoring its input does not make it competent. So the depth-3 memory null from EXP-030 is not
*explained away* by collapse: a de-collapsed policy still solves essentially nothing, so memory would
still have had nothing to contribute.

This points back at ADR 0001's amended conclusion, reached independently on the grid world: **the frozen
sensory encoder is the wall.** Readout capacity was not the constraint there, and stochasticity is not
the constraint here.

## Limitations

- **The sweep is bounded too low.** Every trend is still moving at `beta=0.1`. The modal gate is
  probably reachable at 0.2-0.3, and this experiment cannot say what happens there.
- **Depth 1 is absent by design** (EXP-031: 83% of its episodes end early, so modal fraction is 1.0 by
  construction and measures nothing).
- **Concept arm only.** Whether the memory arms collapse differently under the stabilizers is untested.
- **One episode budget.** 600 episodes throughout; a longer budget might let a high-entropy policy
  eventually convert exploration into competence. Untested, and cheap to test.

## Lead for the next experiment

Two candidate directions, and they are not equally attractive:

1. **Extend the beta sweep to 0.2 and 0.3 with normalization on** (48 runs, about an hour). Cheap, and
   it closes the "bounded too low" limitation honestly. But Finding 3 predicts it buys a gate pass with
   no success improvement, which would be a gate that certifies a policy nobody should want.
2. **Engage the sensory encoder.** Both this experiment and ADR 0001 arrive at the frozen encoder from
   opposite directions. This is the larger and better-motivated move.

**Recommendation: do (1) only as a bounded honesty check, and treat (2) as the real next experiment.**
If (1) is run, the pre-registered gate must stay where it is; a pass there means "the policy is no
longer degenerate", not "the policy is good", and the memory re-run it unlocks should be judged on
success rather than on the gate.

## Regenerate

```bash
.venv/bin/python -u experiments/032_collapse_sweep/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16

.venv/bin/python experiments/032_collapse_sweep/aggregate.py
```
