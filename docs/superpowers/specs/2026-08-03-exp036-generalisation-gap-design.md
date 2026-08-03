# EXP-036 Design: the generalisation gap and the depth break point

Written 2026-08-03, week 18 session 1, before any number exists.

## Why this experiment exists

The vault note `road-to-a-solved-cube.md` names as Stage 1: "does the depth-3 policy generalise
or memorise its 90 training states? Vary the train fraction and watch held-out success."

**That question is already answered and the note is wrong on the premise.** `split_shell`
(`src/neuromorphic/training/cube_baseline.py:90`) partitions the depth-3 shell 90/30, and
`run_cube_baseline` calls `evaluate_states` on the eval side. Measured:

```
depth 1: shell=    6 train=    6 eval=   6 heldout=False
depth 2: shell=   27 train=   27 eval=  27 heldout=False
depth 3: shell=  120 train=  90 eval=  30 heldout=True
depth 4: shell=  534 train= 401 eval= 133 heldout=True
depth 5: shell= 2256 train=2056 eval= 200 heldout=True
depth 6: shell= 8969 train=8769 eval= 200 heldout=True
```

The EXP-035 headline of **50.0% at depth 3 is already held-out success on 30 states the policy
never trained on**. It generalises. What has never been measured is the *gap*: train-side
success has not been reported in any cube experiment, so we do not know whether the policy is
overfitting 90 states or is capacity-limited in a way coverage cannot fix.

The second question is where the approach breaks. Nothing past depth 3 has been tested at all
(open debt item 10 in the 2026-08-02 handoff).

## Two questions, one dispatch

Combining them is a cost decision, not a conceptual one. Measuring the gap requires a retrain
(see "The checkpoint problem" below), and a retrain that sweeps depth answers both for the
price of one.

## Pre-registered interpretation contract

Per the standing habit from the 2026-07-13 audit: the decision rules are fixed here, before
the numbers exist, and each will be marked confirmed or refuted in `RESULTS.md`.

**Claim 1 - the gap.** Define `gap_d = train_success(d) - heldout_success(d)`, both measured on
samples capped at 200 states so the two sides carry matched noise.

- At depth 3, `gap < 0.05` **refutes coverage as a lever** and cancels the vault's Stage-1a
  train-fraction sweep. It will not be run.
- `gap >= 0.15` establishes overfitting and justifies the sweep.
- Between 0.05 and 0.15 is inconclusive: report it, act on neither.

**Claim 2 - the break point.** "Broken at depth d" means held-out success below **twice the
measured random floor at depth d**. The floor is measured per depth via the `arm="random"` path,
never assumed. The repo's own history is the reason: the floor at depth 1 is 21%, not 1/6,
because a random walk with a `2d+3` budget can stumble into solved.

**Prediction, recorded so it can be wrong:** the curriculum breaks at **depth 5**. This comes
from EXP-033's raw-facelet linear probe, which falls 0.956 / 0.766 / 0.598 at depths 3/4/5
against a chance of about 0.19.

**Claim 3.** If depth 6 is still above twice its floor, Wall 1 sits further out than the probe
trend implied and the linear head has more room than EXP-033 suggested. That refutes the
prediction in Claim 2 and gets logged as a refutation, not smoothed over.

**Claim 4 - instruments.** Report `greedy_modal_action_frac` and `mean_train_entropy` per cell.
EXP-035 established that entropy alone cannot separate collapse from convergence: collapse is
low entropy with HIGH modal fraction (0.987), convergence is low entropy with LOW modal
fraction (0.580). A depth that fails should be diagnosed as one or the other, not just scored.

**Claim 5 - the budget confound, stated up front.** This measures where the curriculum breaks
*at 10,000 episodes*. EXP-035 showed depth 3 climbs 0.397 -> 0.500 between 10k and 30k and had
not saturated, so a depth that fails here may only be under-trained. The claim is bounded to a
matched budget, which is the comparison EXP-034 established as the fair one. **Do not report
the break point as a property of the architecture.**

## Design

| | |
|---|---|
| depths | 3, 4, 5, 6 |
| curriculum | `(1..d)` at each depth, equal split of the budget |
| seeds | 0-11 (n = 12, the standing minimum) |
| episodes | 10,000, split across curriculum stages, never multiplied |
| arms | `regionalized` plus `random` at each depth for the floors |
| tag | `exp036` |

`record_filename` encodes tag, arm, depth, seed and sigma, so this sweep collides with nothing.
Budget is constant across cells, which matters because the filename does **not** encode
`episodes` or `curriculum`.

**Built-in validation.** Depth 3 at 10k with seeds 0-11 and curriculum `(1,2,3)` is exactly
EXP-035's 10,000-episode cell. Run on the same machine it should reproduce **0.397
byte-identically**. This validates the code changes below against real experiment data rather
than only against the depth-1 smoke values.

## Why 10,000 and not 30,000

30,000 is the budget the 50% headline was measured at, so it is the more faithful setting for
the gap. It was rejected on cost: roughly 20 hours of wall clock, spanning more than one night,
and it would forfeit the free EXP-035 replication check. At 10,000 the depth-3 policy scores
0.397, which is 28x the measured random floor - unambiguously a working policy, and enough for
a gap to be visible if one exists.

## The checkpoint problem

`run_cube_baseline` writes a JSON record and nothing else. **No trained weights are ever
saved.** Every question of the form "evaluate that trained policy differently" therefore costs
a full retrain: this experiment, the EXP-030 memory re-ask, and any depth-transfer probe.

The head is a `Linear(64 -> 6)`, 390 parameters. Serialising it is close to free and removes
that cost permanently. Doing it now rather than later is the difference between one overnight
run and three.

## Code changes

All three land in `src/neuromorphic/training/cube_baseline.py` and all three run **after**
training completes, so no RNG stream is perturbed.

1. **Capped train-side eval.** New record fields `train_success_rate` and `n_train_eval`. The
   train-side sample is capped at `heldout_cap` and drawn deterministically from `split_seed`.

   **The cap is not an optimisation, it is load-bearing.** Measured cost of an *uncapped*
   train-side eval, at 90 ms per `brain.step`:

   | depth | train states | steps | cost per seed |
   |---|---|---|---|
   | 3 | 90 | 9 | 1.2 min |
   | 4 | 401 | 11 | 6.6 min |
   | 5 | 2056 | 13 | 40.1 min |
   | 6 | 8769 | 15 | **197.3 min** |

   Uncapped, depth 6 costs 3.3 hours per seed - 40 core-hours across 12 seeds, against roughly
   2.6 hours of actual training per run. It would dominate the experiment it is instrumenting.
   The cap also makes the gap a fair comparison: 200 against 200 carries matched noise, where
   200 against 8,769 does not.

2. **Head serialisation.** Write the head's parameters next to the JSON record.

3. **Tag-agnostic record comparison.** `scripts/verify_instrument_neutrality.py` fails on zero
   shared filenames, and `exp035` / `exp036` records never share a filename. The replication
   check needs comparison keyed on (arm, depth, seed, sigma) with tag excluded.

## Testing

Governed by the repo's test-strength rule: **an assertion that cannot fail is not a test.**

- **Byte-identity.** Depth 1, seed 0, 600 episodes, `tag=exp030_concept` must still produce
  `success_rate 0.6666666666666666`, `revisit_rate 0.16633266533066132`,
  `eval_revisit_rate 0.25`, `greedy_modal_action_frac 1.0`,
  `mean_train_entropy 0.5422023858095053`. These are exact equalities on values measured on
  this machine, so they fail if the RNG streams move at all.
- **The cap binds.** At a depth where the train side exceeds `heldout_cap`, assert
  `n_train_eval == heldout_cap` and strictly less than the full train-side size. A test that
  only asserts `n_train_eval > 0` would pass against an uncapped implementation and is
  therefore not a test of this change.
- **The cap is deterministic.** Two runs at the same `split_seed` select the same train-side
  sample; a different `split_seed` selects a different one. The second half is what makes the
  first half meaningful.
- **Round-trip the head.** Load the serialised parameters into a fresh `Linear` and assert it
  reproduces the recorded `success_rate` exactly. Asserting only that the file exists would
  pass against a checkpoint saved before training.
- **Tag-agnostic comparison.** Must report a real mismatch when one field differs, and must
  still fail loudly on zero shared keys rather than reporting a vacuous pass.

## Cost and dispatch

Roughly 107 core-hours, 16 workers, about **6.7 hours wall clock**. One overnight run. The
laptop has 13.4 GB free against roughly 3.1 GB for 16 workers at the measured 195 MB each.

Dispatch follows `docs/playbooks/remote-experiment-runs.md`, with one correction the playbook
needs: **`ssh laptop`, not `ssh mlgbr@swizzlesduo.tailda519d.ts.net`.** The raw hostname does
not match the `Host swizzlesduo laptop` block in `~/.ssh/config`, so ssh never offers
`id_ed25519_backup` and falls back to the encrypted `id_ed25519`, which cannot sign under
BatchMode. Verified failing on 2026-08-03.

The laptop's clone was at `b6feee5` at design time, two commits behind. It must be synced
before dispatch.

## Out of scope

- The train-fraction sweep itself. Claim 1 decides whether it happens at all.
- Anything at depth 7 or beyond.
- The EXP-030 memory re-ask. It becomes cheap once head serialisation lands, which is an
  argument for landing it here, not for widening this experiment.
- Tuning the curriculum. Still fixed `(1..d)`, equal stages, no adaptive advancement.

## Related

- `experiments/035_budget_scaling/RESULTS.md` - the 50% result and the 10k cell this replicates
- `experiments/033_concept_decodability/RESULTS.md` - the probe trend behind the depth-5 prediction
- vault `Neuromorphic Development/road-to-a-solved-cube.md` - Stage 1, whose premise this corrects
- `docs/handoffs/SESSION-HANDOFF-2026-08-02.md` - section 2 reference values, open debt item 10
