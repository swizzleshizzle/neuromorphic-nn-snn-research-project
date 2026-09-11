# EXP-059 results - episodic memory HURTS a policy that works

> **COMPLETE.** 72 cells, 24 seeds, three arms, no tracebacks. **Validity gate PASSED**, so the
> claims below may be read.
>
> **HEADLINE: memory does not merely fail to help - it HURTS, by -0.0954 at p 0.0056.** That
> clears the pre-registered 0.05 bar downward and clears Bonferroni. **This reading was
> pre-registered as a real finding rather than a failed confirmation**, precisely so it could not
> be softened after the fact.
>
> **AND CORRECT MEMORY IS INDISTINGUISHABLE FROM WRONG MEMORY.** `M` minus `S` is **+0.0204 at
> p 0.4268**. Both sit far below the amnesic control. **What costs the policy is READING the
> attractor's stored content at all, not the content being wrong.**
>
> **THE THREE-ARM DESIGN WAS ESSENTIAL, FOR THE THIRD TIME.** A memory-versus-shuffle-null design
> would have measured +0.0204, called it a win, and published it. The amnesic arm is the only
> thing that reveals both arms losing to it by roughly 0.10.

**Pre-registration:** `docs/superpowers/specs/2026-09-09-exp059-memory-depth5-design.md`. The
aggregator was written on 2026-09-09 **while zero records existed**, verified by probe, and was not
edited after the numbers arrived.

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo`, worktree `C:\Users\mlgbr\wt-exp053` at `a709511`, 6 workers |
| Wall clock | 2026-09-09 23:14 to 2026-09-11 14:32 UTC, **39.3 h**, 72 cells |
| Arms | `exp043_capped_d5` field for field, varying ONLY the readout. Encoder FROZEN (exp040 E0) |
| Seeds | 0 to 23, all three arms, no missing cells |
| Per-cell cost | arm A 3.054 h, arm M 3.42 h, arm S ~3.5 h |

**This is the SECOND attempt.** The first was destroyed by a Windows Update reboot 3.75 h in with
zero records written, costing ~19 CPU-hours. Nothing from it survives or contaminates this run.

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch059_wt.ps1 -Phase rl -Workers 6 -SkipExisting
.venv/bin/python -u experiments/059_memory_depth5/aggregate.py
```

## Claim 3, the validity gate - PASSED, and this time the calibration held

| condition | measured | threshold | |
|---|---|---|---|
| arm M `recall_content_cos` | **0.8514** | < 0.95 | PASS |
| arm S `unshuffled_frac` | **0.1480** | < 0.20 | PASS |

Across seeds the cosine runs **0.7527 to 0.9042**, so **no single seed** reaches the threshold, and
every one sits below the 0.9437 that a randomly-loaded attractor measured during calibration. The
probe fired 13,231 to 15,808 times per seed.

**Two of the three validity gates this project has used were wrong.** EXP-057's absolute threshold
was calibrated at depth 3 and evaluated at depth 7, passing with 2.0x margin instead of the three
orders its spec claimed. EXP-058's was unsatisfiable by construction and voided the experiment.
**This one was calibrated across its full attainable range before the spec was written** - empty
attractor 1.000000, random loaded 0.9437, real depth-5 run 0.8128 - **and passed with the margin
that calibration predicted.** The gate-calibration rule paid for itself.

> [!note] **A cross-check in the aggregator is structurally inert, and it is better to say so.**
> The aggregator prints arm A's `recall_content_cos` as a sanity reading, expecting ~1.0 because
> arm A zeroes `W_rec` at the read site. **It is `None` for all 24 seeds.** The `memory_amnesic`
> branch of `MemoryReadout.__call__` returns at `cube_baseline.py:402`, before `_probe_recall` at
> line 418, so the probe never runs for that arm. That is correct behaviour - the cosine would be
> trivially 1.0 - but **the check can never fire and buys nothing.** It degrades gracefully rather
> than crashing or warning falsely, and the gate stands on arm M's own measured range.

## Claim 1, PRIMARY - `M` minus `A`. **-0.0954 at p 0.0056. MEMORY HURTS.**

| arm | readout | success | sd | seeds at exactly 0.000 |
|---|---|---|---|---|
| **A**, amnesic | `memory_amnesic` | **0.3138** | 0.1318 | 2 of 24 |
| **M**, memory | `memory` | **0.2183** | 0.1346 | 3 of 24 |
| **S**, shuffled | `memory_shuffled` | **0.1979** | 0.1510 | 5 of 24 |

| | value |
|---|---|
| paired delta | **-0.0954** |
| W-L-T | **6-17-1** |
| p (sampled, 200,000 draws at a fixed seed) | **0.0056** |
| Bonferroni across 3 contrasts | 0.0167 - **cleared** |
| paired sd / se | 0.1497 / 0.0306 |

**The spec pre-registered this exact reading**: a significant delta at or beyond -0.05 is a real
finding that memory HURTS, required in the headline, and **not** a failed confirmation. That
wording was fixed before any number existed for the obvious reason.

**It is not a held-out artefact.** `train_success_rate` shows the same ordering - A 0.2823,
M 0.1925, S 0.1696 - and the generalisation gap is within 0.006 across all three arms.

**Arm A reproduces the known configuration**, which is the cheap sanity check on the whole run:
`exp043_capped_d5` measured **0.3229** and arm A lands at **0.3138**. Context, not a control, and
no claim is paired against it.

## Claim 2, THE MECHANISM - NOT CONFIRMED, and the sign is wrong

**`M` minus `A` on `revisit_rate`: +0.0157, p 0.4450**, approx 95% interval [-0.0217, +0.0531].

Confirmation required **<= -0.02**: memory was supposed to REDUCE cycling. It did not reduce it,
and the point estimate runs the other way.

| arm | `revisit_rate` | `optimality` | `mean_steps` |
|---|---|---|---|
| A, amnesic | **0.3217** | **0.6707** | 6.32 |
| M, memory | 0.3374 | 0.6477 | 5.95 |
| S, shuffled | 0.3417 | 0.5745 | 5.52 |

**Pre-registered joint reading:** performance moved while the cycling mechanism did not. **Whatever
memory did here, it was NOT the anti-cycling story this design was built to test, and it must not
be narrated as one.**

## Claim 4, SECONDARY - `M` minus `S`. **+0.0204 at p 0.4268.**

Approx 95% interval **[-0.0316, +0.0725]**, which still reaches the 0.05 bar, so n=24 does not
resolve this in either direction.

**THIS MEASURES THE HARM OF *INCORRECT* MEMORY, AND IS NOT EVIDENCE ABOUT THE BENEFIT OF CORRECT
MEMORY.** That sentence is required by the spec because EXP-030's headline came from this contrast
and was misread for months.

**And here it is the whole point.** Correct content buys **+0.0204, indistinguishable from zero**,
while reading the attractor at all costs **-0.0954, highly significant**. The harm is in the read,
not in the content being wrong.

## What this changes

1. **The EXP-030 trap is now confirmed on a working policy, at n=24, with a gate that passed.**
   EXP-030 found memory beating the shuffle-null and not the amnesic control on a policy 15x worse.
   EXP-058 reproduced the ordering but was VOID. **This run reproduces it significantly**, and in
   the same direction: A > M > S.
2. **A two-arm memory-versus-shuffle-null design would have published a false positive for the
   third time.** It measures +0.0204 here. The amnesic arm is what shows both memory arms losing
   to it by about 0.10. **Three arms is not thoroughness, it is the minimum that answers the
   question.**
3. **The hippocampal recall is noise on the policy path at this scale.** Arm A still reads a
   recall block - `MemoryReadout`'s docstring records that 65% of it is a memory-free transform of
   the current concept - so A is not "no recall features", it is "recall features without stored
   content". **Adding stored content is what costs, and the content's correctness is worth
   nothing measurable.**
4. **`optimality` tracks success and `revisit_rate` does not.** Optimality orders A 0.6707,
   M 0.6477, S 0.5745, matching the success ordering; `revisit_rate` is flat to slightly inverted.
   Worth noting since both are on the approved-instrument list in `docs/retired-instruments.md`.

## What is NOT claimed

- **Not that memory is useless in general.** This is depth 5, a frozen EXP-040 encoder, this
  readout, this hippocampus. A different readout could plausibly use the same stored content.
- **Not that `M` and `S` are equivalent.** Claim 4 is a **bound**, and its interval reaches the
  0.05 bar in both directions.
- **Not that memory increases cycling.** Claim 2's +0.0157 is indistinguishable from zero and is
  reported as a bound, not a reversal.
- **Not a mechanism for WHY the read hurts.** Claim 3 rules out an empty attractor, which is far
  narrower than establishing that the recall code is badly scaled, badly conditioned, or simply
  uninformative. That is the obvious next question and this design cannot answer it.
- **Not anything about depths 6 or 7.** EXP-058's depth-6 cells exist but are void.
