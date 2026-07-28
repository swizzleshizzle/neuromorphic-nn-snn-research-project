# EXP-030 Memory Engagement (design)

**Date:** 2026-07-27 · **Phase:** 3 (first engagement step) · **Grounds:** `docs/architecture-spec-v3.md` section 2.2, `docs/phase3-kickoff-brief.md`, `docs/superpowers/specs/2026-07-25-cube-baseline-design.md`, ADR-0001 and its amendments.

## Goal

Put the hippocampus genuinely on the policy path and test whether episodic memory content improves cube solving. This is the first engagement step: EXP-029 established that v1 with `recall=False` cannot answer any question about regionalization, because only the sensory region reaches the policy head.

## The blocker this spec removes (measured 2026-07-27)

Three findings, each verified rather than argued. They are the reason this spec exists.

1. **Flipping `recall=True` is a no-op for the policy.** The head reads `concept`, which is computed upstream of the hippocampus. Turning recall on changes PFC utilities, but the head's input is bit-identical. The kickoff brief's "recall-in-loop is the first engagement lever" cannot work as stated: the lever is not connected to anything.
2. **The recall code carries no state information.** Mean pairwise cosine across distinct inputs is **0.998 on the cube and 0.9995 on the grid**, with recall standard deviation across inputs of 0.002. The concepts going in are clearly distinguishable (cosine 0.71 and 0.80). The output is near-constant.
3. **The root cause: `store()` overwrites.** It executes `self.W_rec = self.recurrent_gain * w`, an assignment, so storing a second pattern discards the first. The hippocampus holds exactly **one** pattern, and the resulting single attractor dominates `fc_out` regardless of input. Verified: after storing A then B, `W_rec` equals B alone, not A+B.

**This is a gap, not a bug in its original context.** EXP-017/019 validated a single-pattern delay task, and `test_attractor_holds_pattern_through_delay` still passes. The gap is against arch-spec-v3's Phase-3 claim: *track move history, recognize visited states, avoid cycles*. That needs multi-pattern storage, which does not exist.

Two tests should have caught this and are too weak to. `test_store_imprints_recurrent_weights` asserts only `count_nonzero(W_rec) > 0`, which cannot distinguish accumulate from overwrite. `test_recall_is_content_specific` asserts only `not torch.equal(...)`, which passes on a 0.2% difference. This is the third instance in two days of a too-weak assertion hiding a real defect (the EXP-029 chance floor was the first).

## 1. `Hippocampus` becomes a real memory (library)

`src/neuromorphic/regions/hippocampus.py`:

- **`store()` accumulates:** `W_rec += recurrent_gain * outer(s, s) / n_neurons`, diagonal zeroed. Keep a list of stored patterns for instrumentation.
- **`clear()`** zeroes `W_rec` and the pattern list. Required for per-episode memory; impossible today.
- **`familiarity(content) -> Tensor [B]`**: one scalar per batch element, the Hopfield field alignment `s @ W_rec @ s / n_neurons`, where `s` is the bipolar top-k pattern of that element's content drive. High when the state sits near a stored attractor. It reuses `W_rec` rather than adding a lookup table, so familiarity remains a property of the attractor rather than bookkeeping bolted alongside it. Returns zeros when nothing is stored.
- `forward()` keeps its signature and shape. With multiple patterns stored it now completes toward a mixture rather than a single fixed point.

## 2. Readout seam in `reinforce.py` (library)

`concept_rate(out)` becomes `policy_features(out, mode)`:

| mode | returns | width |
|---|---|---|
| `concept` | sensory concept only | 64 |
| `memory` | concept, recall, familiarity | 129 |
| `memory_shuffled` | concept, plus recall and familiarity queried with a DIFFERENT visited state | 129 |

**How the shuffle-null works.** Do not permute the recall vector's elements: a scalar familiarity cannot be permuted meaningfully, and element permutation destroys the code's structure as well as its correspondence. Instead, query the hippocampus with a uniformly random *other* state from the current episode's visited set. Both readouts stay real, in-distribution, and correctly scaled; only the correspondence between the agent's current state and the memory it receives is destroyed. That is exactly the confound the control must isolate. When fewer than two states have been visited, fall back to the current state and record that the step was un-shuffled.

`action_distribution` gains `recall` and `store` parameters instead of hardcoding both `False`.

**`reinforce.py` is modified by this spec.** The "do not touch `reinforce.py`" constraint was specific to EXP-029, where the point was that the file was already environment-agnostic. Engagement necessarily changes what the policy reads, so that constraint does not carry over.

`mode='concept'` must be bit-identical to today's behavior so every 024-029 driver is unaffected.

## 3. `experiments/030_memory_engagement/` (driver)

`run.py`, `aggregate.py`, `outputs/` (gitignored), `RESULTS.md` (committed). Same shape as 029.

## 4. Protocol

| Axis | Value |
|---|---|
| Arms | `concept` (64), `memory` (129), `memory_shuffled` (129) |
| Depths | pinned to EXP-029's knee; default 3 to 6 |
| Seeds | 0 to 11 |
| Storage | every visited state, `clear()` at episode start |
| `max_steps` | `2d + 3`, unchanged from EXP-029 |
| Runs | 3 arms x 4 depths x 12 seeds = 144 |

**The primary comparison is `memory` vs `memory_shuffled`.** The shuffle-null holds head width and activity statistics fixed while destroying the correspondence between state and memory, so a difference is attributable to memory **content** rather than head capacity. This is the repo's established methodology from EXP-027, and it is the control EXP-029 lacked when it compared a 128-wide stack against a 446-wide one.

**Attribution is deliberately deferred.** Separating the 64-wide completion code from the familiarity scalar would need five arms and 240 runs. Establish that memory content does anything first; spend a follow-up attributing it only if there is an effect to attribute. Comparing before establishing the mechanism works is the specific error EXP-029 made.

## 5. Pre-flight gate: does the policy revisit states at all?

At `max_steps = 2d + 3`, an agent at depth 4 has 11 steps. **If it rarely revisits a state, cycle-avoidance memory has nothing to do**, and a null would mean "there were no cycles to avoid" rather than "memory does not help."

**EXP-029 cannot supply this number.** `run_cube_baseline` never saves a policy head, so there is no trained policy to load and no checkpoint to reuse. The gate is therefore measured inside EXP-030 itself: the `concept` arm is the EXP-029 baseline recipe, so instrument it to record, per episode, how many steps revisit an already-visited state.

Run the `concept` arm first, read the revisit rate, and only then launch the two memory arms. If revisits are rare, the design needs rethinking (most likely longer step budgets), not a 144-run null. This costs nothing extra, since the `concept` arm is needed regardless, and it makes the gate a reported number rather than a silent assumption.

## 6. Pre-registered interpretation contract

Written before the numbers exist.

- **Primary:** `memory` beats `memory_shuffled`, paired per seed at n = 12. This is the claim the experiment stands or falls on.
- **Secondary:** `memory` beats `concept`. If primary holds but secondary does not, the gain came from head width, not memory.
- **Revisit rate is reported for every arm**, not only as the gate. A null alongside a near-zero revisit rate is a statement about the task, not about memory, and the writeup must say which it is.
- **Capacity is measured, not assumed.** Classic Hopfield capacity is about `0.14 * N`, so roughly 21 patterns at N = 150. A depth-6 episode stores up to 15, which is inside capacity but close. Report familiarity across the episode; if everything reads familiar by step 10 the signal is exhausted, and that is a finding.
- **A null is a result** and is written up as one rather than explored until something separates.

## 7. Testing

The gates that would have caught the defect this spec fixes:

- `store()` accumulates: after storing A then B, `W_rec` equals A + B, not B alone
- `clear()` zeroes `W_rec` and the stored-pattern list
- recall discriminates: mean pairwise cosine across distinct inputs below a stated threshold (it is 0.998 today, so the threshold must be well below that)
- `familiarity()` separates visited from novel states by a stated margin
- `policy_features` returns the documented width for each mode
- `mode='concept'` is bit-identical to current behavior under a fixed generator
- The full suite stays at 340 or more

## Non-goals

- R-STDP (deferred by the kickoff brief)
- Prefrontal engagement, including the degenerate utility readout and its hidden transform
- The monolithic and regionalization comparison; that becomes meaningful only once memory demonstrably contributes
- Attribution of completion versus familiarity (a follow-up, conditional on a positive primary result)
- 3x3 anything

## Success criteria

1. `store()` accumulates and `clear()` works, both proven by tests that would fail against today's code.
2. The recall code discriminates states, replacing the measured 0.998 cosine with a number well below the stated threshold.
3. `mode='concept'` leaves every existing experiment bit-identical.
4. The revisit-rate gate is measured and reported before the main run.
5. A collapse curve for three arms across the chosen depths, with the paired `memory` vs `memory_shuffled` test at n = 12.
6. Every claim in the section-6 contract confirmed or explicitly refuted in `RESULTS.md`.
7. Reusable machinery under `src/neuromorphic/`; only the driver under `experiments/`.
