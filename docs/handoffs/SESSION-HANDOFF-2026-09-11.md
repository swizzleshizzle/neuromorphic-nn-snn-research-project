# Session Handoff - 2026-09-11 (Week 23) - EXP-059 COMPLETE AND MERGED

> **Nothing is running. The laptop is FREE. `main` is at `054db85`, clean, no branches, no PRs.**
>
> **Suite: 591 passed** under `-m "not slow"` (356 outside `tests/training`, 31 in
> `test_cube_baseline.py`, 204 in the rest), plus 22 slow-marked deselected.
>
> **EXP-059 answered the memory question, and the answer is that memory HURTS.**

## 1. EXP-059 - the headline, and why it is stronger than a null

**`M` minus `A` on held-out success: -0.0954 at p 0.0056**, clearing the pre-registered 0.05 bar
downward and clearing Bonferroni 0.0167. W-L-T 6-17-1 across 24 seeds.

| arm | readout | success |
|---|---|---|
| **A**, amnesic | `memory_amnesic` | **0.3138** |
| **M**, memory | `memory` | 0.2183 |
| **S**, shuffled | `memory_shuffled` | 0.1979 |

**That reading was fixed in the spec before any number existed** - a significant negative is a real
finding required in the headline, not a failed confirmation - specifically so it could not be
softened afterwards.

### The sharpest result is Claim 4, and it is easy to misread

**`M` minus `S` is +0.0204 at p 0.4268: correct memory is INDISTINGUISHABLE from wrong memory.**
Reading the attractor at all costs 0.095; the content being correct buys nothing measurable.

**Arm A is not "no recall".** `MemoryReadout`'s docstring records that 65% of the recall block is a
memory-free transform of the current concept, so A is *recall features without stored content*.
**Adding stored content is precisely what costs.**

### A two-arm design would have published a win for the THIRD time

Memory versus shuffle-null measures **+0.0204** here and reads as a modest win. Only the amnesic arm
shows both memory arms losing to it by about 0.10. EXP-030 and EXP-058 had the identical shape.
**Three arms is not thoroughness; it is the minimum that answers the question.**

### The mechanism claim did NOT confirm, and its sign runs the wrong way

`revisit_rate` moved **+0.0157 at p 0.4450** where confirmation needed `<= -0.02`. Pre-registered
joint reading: performance moved while the cycling mechanism did not. **Whatever memory did here, it
was not the anti-cycling story, and must not be narrated as one.**

### Supporting checks that make the result hard to explain away

- **Not a held-out artefact**: `train_success_rate` shows the same ordering (A 0.2823, M 0.1925,
  S 0.1696) and the generalisation gap is within 0.006 across arms.
- **Arm A reproduces the known config**: 0.3138 against `exp043_capped_d5`'s 0.3229.
- **`optimality` tracks success** (A 0.6707, M 0.6477, S 0.5745) while `revisit_rate` is flat.

## 2. The validity gate PASSED, and the gate-calibration rule paid for itself

| condition | measured | threshold | |
|---|---|---|---|
| arm M `recall_content_cos` | **0.8514** (range 0.7527-0.9042) | < 0.95 | PASS |
| arm S `unshuffled_frac` | **0.1480** | < 0.20 | PASS |

**Fourth gate this project has used, and the first right by construction.** Calibrated across its
full attainable range before the spec existed, it can fail (empty attractor = 1.000000) and can pass
(real run = 0.8128), and it is a **cosine, so bounded and scale-free** - which is what stops it
repeating EXP-057's regime-dependence.

**It was also checked EARLY**, on arm M alone, while the run was still going: a failure would have
meant 13 h of remaining arm-S compute spent on a void experiment. That is not peeking - Claim 3 is a
condition, not a contrast, arm A was never loaded, and the records went to a scratchpad so the
aggregator could not compute anything on partial data.

> [!note] One cross-check in the aggregator is structurally INERT, and it is better to know.
> It prints arm A's `recall_content_cos` expecting ~1.0. It is `None` for all 24 seeds: the
> `memory_amnesic` branch returns at `cube_baseline.py:402`, before `_probe_recall` at line 418.
> Correct behaviour, but the check can never fire.

## 3. What else this week produced

- **`docs/retired-instruments.md`** - the standing note, with a `CLAUDE.md` pointer. **Every one of
  the five works as a THRESHOLD and fails as a GRADIENT.**
- **`experiments/059_memory_depth5/aggregate.py` + 22 tests**, written while zero records existed,
  each test verified to fail against the bug it names by mutating the aggregator eleven ways.
- **EXP-060 spec** - `docs/superpowers/specs/2026-09-10-exp060-flattened-critic-replication-design.md`.
- **Playbook**: an unreachable tailscale peer is an **unknown**, not a paused job; the laptop is
  **UTC-4**; do not scale per-cell cost by hand, use `steps()`; check reboot history before a long
  dispatch.

## 4. Open items

1. **EXP-060 is ready to dispatch and the laptop is free.** ~14 h. **DO NOT DISPATCH UNTIL WINDOWS
   UPDATE IS DEFERRED** - that is what destroyed EXP-059's first attempt, and the registry write is
   blocked by the permission classifier here, so it needs Michael: Settings > Windows Update > Pause
   updates.
2. **The obvious next question EXP-059 cannot answer: WHY does the read hurt?** Claim 3 rules out an
   *empty* attractor, which is far narrower than ruling out a badly scaled or uninformative recall
   code. **That is the live scientific thread**, and it is more interesting than EXP-055's leads.
3. **EXP-055's two leads**, both needing compute: what one epoch builds that helps policy while
   making `S` worse than random, and whether `e2` is a cheaper recipe at 74% of `e10`.
4. **Vault `4f13` ("re-ask the EXP-030 memory question") IS EXP-059 and can now be ticked** - but
   that is Michael's call, not something to tick because the work looks finished.
5. **Vault, needs Michael**: `0576` dashboard render, `0817` Phase 0/1 checkpoints.

## 5. Standing facts

- **READ `docs/retired-instruments.md` BEFORE PUTTING AN INSTRUMENT IN A SPEC.**
- **READ "THE GATE-CALIBRATION RULE" IN `CLAUDE.md` BEFORE WRITING A GATE.** Calibrate across the
  attainable range; prefer a ratio or a bounded quantity to an absolute.
- **The critic question is CLOSED**: the benefit is within-episode state-dependence, not calibration.
- **The memory question is now ANSWERED at depth 5**: the recall read is harmful, and its content's
  correctness is worth nothing measurable.
- **An unreachable peer is an UNKNOWN.** Nothing is durable until a cell completes.
- **n >= 12 seeds. Measure the chance floor. No scipy.**

## 6. Pointers

- `experiments/059_memory_depth5/RESULTS.md` - the full write-up
- `docs/retired-instruments.md`, `CLAUDE.md`
- `docs/playbooks/remote-experiment-runs.md` - the corrected last-seen table, cost estimation
- Vault: `experiment-log.md` (through EXP-059, plus the gate-calibration addendum)
