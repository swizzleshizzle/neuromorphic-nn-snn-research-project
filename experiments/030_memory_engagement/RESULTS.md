# EXP-030 Results - Memory Engagement (does episodic memory content help?)

> **Why this file exists:** the standing habit adopted after the 2026-07-13 Phase-2 audit. Every experiment
> commits a curated, in-repo results record so the authoritative numbers never live only in a gitignored
> `outputs/` folder. **Provenance:** 144 records (4 arms x depths 1-3 x 12 seeds 0-11), 600 episodes per run,
> 2x2 cube, exact-distance shells, run 2026-07-29 on the laptop `SwizzlesDuo` (Intel Core Ultra 9 185H,
> 22 cores) over SSH with `--workers 8`. Phase 1 (concept arm, 36 runs) took about 31 minutes; phase 2
> (108 memory runs) ran 21:04:09 to 23:56:10, so 2h52m. Peak resident memory was about 195 MB per worker,
> 1.58 GB total, so `--workers 8` was conservative here. **The interpretation contract in
> `docs/superpowers/specs/2026-07-27-memory-engagement-design.md` section 6 was committed BEFORE any numbers
> existed** (git `5c32925`). Regenerate with the command at the bottom.

## What this tests

EXP-029 established that with `recall=False` only the sensory region reaches the policy head, so v1 cannot
answer any question about the other regions. EXP-030 puts the hippocampus genuinely on the policy path and
asks whether episodic memory **content** improves cube solving. Four arms, all reading through the same
REINFORCE head:

| arm | head input | width | isolates |
|---|---|---|---|
| `concept` | sensory concept only | 64 | the EXP-029 baseline recipe |
| `memory` | concept + recall + familiarity | 129 | the full memory readout |
| `memory_shuffled` | concept + recall/familiarity queried with a DIFFERENT visited state | 129 | memory **correspondence** |
| `memory_amnesic` | same feed-forward expansion of the CURRENT state, `W_rec` zeroed at read time | 129 | memory **content** |

The amnesic arm exists because a pre-run measurement found that **65% of the recall block's energy is a
memory-free nonlinear transform of the current concept** (cosine 0.802 against a `W_rec`-zeroed version).
Without it, a `memory > memory_shuffled` result is equally explained by "the head got 64 more features of
what it is already looking at."

## The gate: are there cycles for memory to break?

Pre-registered as a stop condition. Measured on the concept arm before the memory arms were launched.

| depth | greedy revisit rate | (training-policy) | seeds with greedy > 0.05 |
|---|---|---|---|
| 1 | 0.089 | 0.161 | 6/12 |
| 2 | 0.327 | 0.252 | 12/12 |
| 3 | 0.604 | 0.423 | 12/12 |

**The gate passed decisively.** At depth 3 the greedy revisit rate is 0.604, and 7 of 12 seeds sit at exactly
0.667, which with a 9-step budget is 3 unique states in 9 steps: a deterministic policy falling into an
absorbing 3-cycle and riding it to the budget. That is precisely the failure mode memory was supposed to
break. Note the greedy rate EXCEEDS the training-policy rate at depths 2 and 3, the opposite of the
"stochastic policy revisits by construction" artifact the gate was designed to screen out.

## Results

Success rate, mean over 12 seeds, standard deviation in brackets.

| depth | concept | memory | shuffled | amnesic |
|---|---|---|---|---|
| 1 | **87.5%** (sd 14.4) | 80.6% (sd 23.4) | 79.2% (sd 24.7) | 83.3% (sd 17.4) |
| 2 | **38.0%** (sd 24.5) | 31.2% (sd 21.8) | 20.4% (sd 15.5) | 29.9% (sd 23.4) |
| 3 | 2.2% (sd 5.9) | 1.1% (sd 3.0) | 0.6% (sd 1.9) | 0.0% (sd 0.0) |

Paired per matched (depth, seed), n = 12. `p` is an exact two-sided paired permutation test over all
2^12 = 4096 sign flips, so no normal approximation is involved.

| comparison | depth | mean diff | sd | wins | losses | ties | exact p |
|---|---|---|---|---|---|---|---|
| **memory - shuffled** (PRIMARY) | 1 | +1.4 pts | 27.0 | 6 | 2 | 4 | 1.000 |
| | 2 | **+10.8 pts** | 18.1 | 8 | 3 | 1 | **0.078** |
| | 3 | +0.6 pts | 1.3 | 2 | 0 | 10 | 0.500 |
| **memory - amnesic** (CONTENT) | 1 | -2.8 pts | 25.5 | 3 | 5 | 4 | 0.844 |
| | 2 | **+1.2 pts** | 28.0 | 5 | 6 | 1 | **0.908** |
| | 3 | +1.1 pts | 3.0 | 2 | 0 | 10 | 0.500 |
| memory - concept (SECONDARY) | 1 | -6.9 pts | 21.9 | 2 | 5 | 5 | 0.406 |
| | 2 | -6.8 pts | 36.1 | 4 | 6 | 2 | 0.535 |
| | 3 | -1.1 pts | 6.1 | 2 | 1 | 9 | 1.000 |

## The finding: the fourth arm inverted the conclusion

Read the depth-2 arm ordering, which is where all the headroom is:

**shuffled 20.4% < amnesic 29.9% ~ memory 31.2% < concept 38.0%**

- **Removing stored content entirely costs 1.2 points** (memory vs amnesic, p = 0.91). The amnesic arm has
  the identical head width and the identical feed-forward expansion of the current state; the only thing it
  lacks is anything stored in the attractor. It performs the same.
- **Feeding the head a wrong state costs 10.8 points** (memory vs shuffled, p = 0.078). The shuffled arm
  performs *worse than the arm with no memory at all*.

So the pre-registered primary comparison is positive, but it is not evidence that memory content helps. It
is evidence that **incorrect** memory actively misleads the policy. The spec anticipated exactly this
ambiguity and pre-registered the resolution: "Any claim about memory content rests on the `memory` vs
`memory_amnesic` comparison." That comparison is a clean null.

**Had the experiment shipped with three arms instead of four, the honest reading of `memory` beating
`memory_shuffled` by 10.8 points at p = 0.078 would have been "memory content helps."** It does not.

## The mechanistic null: memory did not reduce cycling

Revisit rate is reported for every arm, as pre-registered. If memory were doing the job the architecture
spec claims for it (recognize visited states, avoid cycles), the memory arm should revisit LESS than the
concept arm. It does not.

| depth | concept | memory | shuffled | amnesic |
|---|---|---|---|---|
| 1 | 0.089 | 0.102 | 0.116 | 0.083 |
| 2 | 0.327 | 0.368 | 0.447 | 0.376 |
| 3 | 0.604 | 0.609 | 0.663 | 0.666 |

The memory arm revisits slightly MORE than the concept arm at every depth. The cycles were there (the gate
established that), memory was on the policy path (mean stored patterns: 2.9, 6.3, 9.0 per episode), and the
cycling did not decrease. This is a direct null on the mechanism, independent of the success-rate null.

## Pre-registered contract, claim by claim

**1. "Primary: `memory` beats `memory_shuffled`, paired per seed at n = 12." CONFIRMED IN SIGN, but it does
not mean what it was written to mean.** +10.8 points at depth 2 (p = 0.078, 8-3-1), null at depths 1 and 3.
The contract named this "the claim the experiment stands or falls on," but the amnesic arm shows the gap is
attributable to the shuffled arm being harmed rather than to memory content helping. The primary comparison
alone would have licensed a false claim.

**2. "Secondary: `memory` beats `concept`." REFUTED.** `memory` is worse than `concept` at every depth
(-6.9, -6.8, -1.1 points), none significant. The 65 extra features bought nothing and cost a little. Since
the contract already ruled that no content claim may rest on this comparison, the refutation costs nothing,
but it does mean the engagement step did not pay for itself.

**3. "Revisit rate is reported for every arm." DONE, and it carries its own null.** See the section above.
The pre-registration said a null alongside a near-zero revisit rate would be a statement about the task
rather than about memory. Revisit rates were NOT near zero (0.089 / 0.327 / 0.604), so this null is about
memory.

**4. "Capacity is measured, and the two mechanisms degrade differently." CONSISTENT.** Mean stored patterns
per episode were 2.9, 6.3 and 9.0 at depths 1 to 3. The pre-registration predicted the 64-wide completion
code would contribute at depth 3 and that a memory win concentrated at shallow depths would be the expected
pattern. There is no memory win at any depth once the amnesic control is applied, so the capacity model was
not put to a real test.

**5. "A null is a result." APPLIED.** This document reports one and stops rather than exploring until
something separates.

**6. "Familiarity carries elapsed-time signal, matched across the three memory arms." HOLDS as designed.**
All three memory arms read familiarity from a hippocampus at the same point in the same episode, so the
drift is matched and both primary comparisons survive it. It remains a reason no claim may rest on
`memory` vs `concept`.

## Limitation found in the data, not pre-registered

**The shuffle control is diluted at depth 1.** `unshuffled_frac` (the fraction of steps where fewer than two
states had been visited, so the shuffle fell back to the current state) was **0.321 at depth 1**, 0.152 at
depth 2 and 0.112 at depth 3. At depth 1 roughly a third of the shuffled arm's steps were not actually
shuffled, so the depth-1 primary comparison is weaker than n = 12 suggests. Depth 2, where the effect lives,
is diluted 15%. This does not change any conclusion (the depth-2 primary is positive and the content
comparison is null regardless), but it should be designed out of any follow-up.

## Cross-check: the concept arm reproduces EXP-029 exactly

| depth | EXP-030 `concept` | EXP-029 `regionalized` |
|---|---|---|
| 1 | 87.5% (sd 14.4) | 87.5% (sd 14.4) |
| 2 | 38.0% (sd 24.5) | 38.0% (sd 24.5) |
| 3 | 2.2% (sd 5.9) | 2.2% (sd 5.9) |

This confirms spec success criterion 3 ("`mode='concept'` leaves every existing experiment bit-identical")
empirically at the experiment level, not only by unit test. Separately, the 36 concept records were produced
twice by independent invocations with different worker scheduling and were **byte-identical 36/36**, so the
numbers above are reproducible rather than one scheduling accident.

## Verdict

> [!warning] Episodic memory content does not improve cube solving in v1, and memory did not reduce cycling.
> The `memory` arm beats the shuffle-null by 10.8 points at depth 2, but beats the amnesic control by 1.2
> points (p = 0.91). Since the amnesic arm holds head width and the feed-forward expansion of the current
> state fixed and removes only the stored content, **the stored content is doing nothing**; the shuffle-null
> gap is the cost of feeding the policy a wrong state, not the benefit of memory. Revisit rates did not fall
> in the memory arm at any depth despite abundant cycles to break. `memory` is also slightly worse than plain
> `concept` everywhere, so the engagement step did not pay for itself.

**What this does not say.** It does not say episodic memory is useless for this task. It says THIS memory
implementation, on a 2x2 cube at depths 1 to 3 with a 2d+3 step budget and a linear head, contributes
nothing. The recall code is load-limited by design (pairwise cosine 0.666 at 2 stored patterns, 0.990 by 15),
and episodes here store 3 to 9 patterns, squarely inside the degrading range.

**Lead for EXP-031.** The mechanism null is the more informative half. Memory was on the policy path and
cycles were abundant, yet cycling did not decrease, which points at the readout rather than at storage: a
linear head over a 129-wide concatenation may simply be unable to use a familiarity scalar to veto an action.
Attribution (completion code vs familiarity scalar) was deliberately deferred pending a positive result;
there is no effect to attribute, so that follow-up is now moot as specified.

## Regenerate

```powershell
.venv\Scripts\python.exe experiments\030_memory_engagement\run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 8
.venv\Scripts\python.exe experiments\030_memory_engagement\aggregate.py
```

Phase 1 stops at the revisit gate by design. Pass `--skip-gate` only after reading the greedy revisit rate.
